import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class SparseRefereeRouter(nn.Module):
    """
    Context-Aware Sparse Referee Router with Straight-Through Estimator (STE).
    Generates dynamic on/off binary gating decisions [m_audio, m_video] in {0, 1}^2
    conditioned on unimodal embeddings, cross-modal discrepancy, and cross-modal product:
    x_route = [f_video || f_audio || |f_video - f_audio| || f_video * f_audio] (dim = embed_dim * 4 = 896).
    """
    def __init__(self, embed_dim: int = 224, hidden_dim: int = 32) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        in_dim = embed_dim * 4  # 224 * 4 = 896
        self.router_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        # FC1: Kaiming uniform (standard for GELU activation)
        nn.init.kaiming_uniform_(self.router_mlp[0].weight, a=math.sqrt(5))
        if self.router_mlp[0].bias is not None:
            nn.init.zeros_(self.router_mlp[0].bias)

        # FC2: Small Normal (mean=0.0, std=0.01) + Positive bias (+0.5) so initial exploration starts receptive (p ~= 0.62)
        nn.init.normal_(self.router_mlp[2].weight, mean=0.0, std=0.01)
        if self.router_mlp[2].bias is not None:
            nn.init.constant_(self.router_mlp[2].bias, 0.5)

    def forward(
        self,
        f_video: torch.Tensor,
        f_audio: torch.Tensor,
        u_tie: Optional[torch.Tensor] = None,
        f_joint: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        delta_f = torch.abs(f_video - f_audio)
        prod_f = f_video * f_audio
        x_route = torch.cat([f_video, f_audio, delta_f, prod_f], dim=-1)  # [B, in_dim=896]

        logits = self.router_mlp(x_route)               # [B, 2]
        probs = torch.sigmoid(logits)                    # [B, 2] in (0, 1)

        p_audio = probs[:, 0]                            # [B]
        p_video = probs[:, 1]                            # [B]

        # Straight-Through Estimator (STE): hard binary forward, continuous gradient backward
        m_audio_hard = (p_audio >= 0.5).float()
        m_video_hard = (p_video >= 0.5).float()

        m_audio = p_audio + (m_audio_hard - p_audio).detach()
        m_video = p_video + (m_video_hard - p_video).detach()

        return {
            "m_audio": m_audio,
            "m_video": m_video,
            "prob_audio": p_audio,
            "prob_video": p_video,
            "m_audio_hard": m_audio_hard,
            "m_video_hard": m_video_hard,
            "router_logits": logits
        }


def _make_subspace_head(dim: int, hidden_dim: int = 112) -> nn.Sequential:
    """Helper to build a 2-layer MLP head with LayerNorm using 100% native PyTorch initialization."""
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, 1)
    )


class PairwiseBoundaryTournamentHead(nn.Module):
    """
    Hierarchical Pairwise Cross-Boundary Tournament Head with Sparse Mixture-of-Referees (SMoR, ~331K params).
    
    Level 1: Feeding Activity Gating
      - Distinguishes None (No feeding, quiet water) from Active Feeding (Weak, Medium, Strong).
      - p_feeding = sigmoid(w_act^T * f) in (0, 1)
      - p_none = 1 - p_feeding

    Level 2: 3-Way Pairwise Cross-Boundary Tournament with Sparse Mixture-of-Referees (SMoR)
      - B12: Weak <-> Medium    -> Base Head on f_joint + Audio STFT Referee + Video Kinematics Referee + Router B12
      - B23: Medium <-> Strong  -> Base Head on f_joint + Audio STFT Referee + Video Kinematics Referee + Router B23
      - B13: Weak <-> Strong    -> Base Head on f_joint + Audio STFT Referee + Video Kinematics Referee + Router B13

    Sparse Referee Dynamic Intervention:
      logit = logit_base + u_tie * (m_A * gamma_A * logit_A + m_V * gamma_V * logit_V)
      where u_tie = exp(-|logit_base| / 2.0) represents referee indecisiveness,
      and [m_A, m_V] in {0, 1}^2 are discrete binary decisions via Straight-Through Estimator (STE).

    Tournament Scoring (Borda count):
      - V_Weak   = P(W > M) + P(W > S)
      - V_Medium = (1 - P(W > M)) + P(M > S)
      - V_Strong = (1 - P(W > S)) + (1 - P(M > S))
      Sum(V_Weak + V_Medium + V_Strong) = 3.0 (Strict invariant)
    """
    def __init__(
        self,
        dim: int = 224,
        temperature: float = 2.0,
        enable_b12_audio: bool = True,
        enable_b12_video: bool = True,
        enable_b23_audio: bool = True,
        enable_b23_video: bool = True,
        enable_b13_audio: bool = True,
        enable_b13_video: bool = True,
        tie_breakers: Optional[Any] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.dim = dim
        self.temperature = temperature

        # Parse tie_breakers if provided as dict or config object
        if tie_breakers is not None:
            if hasattr(tie_breakers, "b12"):
                enable_b12_audio = getattr(tie_breakers.b12, "enable_audio", True)
                enable_b12_video = getattr(tie_breakers.b12, "enable_video", True)
                enable_b23_audio = getattr(tie_breakers.b23, "enable_audio", True)
                enable_b23_video = getattr(tie_breakers.b23, "enable_video", True)
                enable_b13_audio = getattr(tie_breakers.b13, "enable_audio", True)
                enable_b13_video = getattr(tie_breakers.b13, "enable_video", True)
            elif isinstance(tie_breakers, dict):
                b12_cfg = tie_breakers.get("b12", {})
                b23_cfg = tie_breakers.get("b23", {})
                b13_cfg = tie_breakers.get("b13", {})
                if isinstance(b12_cfg, dict):
                    enable_b12_audio = b12_cfg.get("enable_audio", True)
                    enable_b12_video = b12_cfg.get("enable_video", True)
                    enable_b23_audio = b23_cfg.get("enable_audio", True)
                    enable_b23_video = b23_cfg.get("enable_video", True)
                    enable_b13_audio = b13_cfg.get("enable_audio", True)
                    enable_b13_video = b13_cfg.get("enable_video", True)

        self.enable_b12_a = bool(enable_b12_audio)
        self.enable_b12_v = bool(enable_b12_video)
        self.enable_b23_a = bool(enable_b23_audio)
        self.enable_b23_v = bool(enable_b23_video)
        self.enable_b13_a = bool(enable_b13_audio)
        self.enable_b13_v = bool(enable_b13_video)

        # Level 1: Feeding Activity Gate (None vs Feeding)
        self.activity_head = nn.Sequential(
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Level 2: 3 Specialized Pairwise Subspace Heads + Referees
        # B12: Weak vs Medium
        self.head_b12 = _make_subspace_head(dim, 112)
        if self.enable_b12_a:
            self.head_b12_a = _make_subspace_head(dim, 112)
            self.gamma_12_a = nn.Parameter(torch.tensor(1.0))
        else:
            self.head_b12_a = None
            self.gamma_12_a = None

        if self.enable_b12_v:
            self.head_b12_v = _make_subspace_head(dim, 112)
            self.gamma_12_v = nn.Parameter(torch.tensor(1.0))
        else:
            self.head_b12_v = None
            self.gamma_12_v = None

        # B23: Medium vs Strong
        self.head_b23 = _make_subspace_head(dim, 112)
        if self.enable_b23_a:
            self.head_b23_a = _make_subspace_head(dim, 112)
            self.gamma_23_a = nn.Parameter(torch.tensor(1.0))
        else:
            self.head_b23_a = None
            self.gamma_23_a = None

        if self.enable_b23_v:
            self.head_b23_v = _make_subspace_head(dim, 112)
            self.gamma_23_v = nn.Parameter(torch.tensor(1.0))
        else:
            self.head_b23_v = None
            self.gamma_23_v = None

        # B13: Weak vs Strong
        self.head_b13 = _make_subspace_head(dim, 112)
        if self.enable_b13_a:
            self.head_b13_a = _make_subspace_head(dim, 112)
            self.gamma_13_a = nn.Parameter(torch.tensor(1.0))
        else:
            self.head_b13_a = None
            self.gamma_13_a = None

        if self.enable_b13_v:
            self.head_b13_v = _make_subspace_head(dim, 112)
            self.gamma_13_v = nn.Parameter(torch.tensor(1.0))
        else:
            self.head_b13_v = None
            self.gamma_13_v = None

        # Level 2 SMoR Routers: 3 independent routers for B12, B23, B13
        self.use_sparse_moe_routing = bool(kwargs.get("use_sparse_moe_routing", True))
        self.router_hidden_dim = int(kwargs.get("router_hidden_dim", 32))

        if self.use_sparse_moe_routing:
            self.router_b12 = SparseRefereeRouter(embed_dim=dim, hidden_dim=self.router_hidden_dim)
            self.router_b23 = SparseRefereeRouter(embed_dim=dim, hidden_dim=self.router_hidden_dim)
            self.router_b13 = SparseRefereeRouter(embed_dim=dim, hidden_dim=self.router_hidden_dim)
        else:
            self.router_b12 = None
            self.router_b23 = None
            self.router_b13 = None

    def forward(
        self,
        f: torch.Tensor,
        f_audio: Optional[torch.Tensor] = None,
        f_video: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # 1. Level 1: Feeding Activity Gate
        logit_act = self.activity_head(f).squeeze(-1)       # [B]
        p_feeding = torch.sigmoid(logit_act)                # [B] in (0, 1)
        p_none = torch.clamp(1.0 - p_feeding, min=1e-6)     # [B]

        # 2. Level 2: 3 Pairwise Cross-Boundary Logits with Referees & SMoR Routing
        # B12 (Weak vs Medium)
        logit_12_base = self.head_b12(f).squeeze(-1)        # [B] (Positive -> Weak, Negative -> Medium)
        u_tie_12 = torch.exp(-torch.abs(logit_12_base) / 2.0)
        ref_effect_12 = torch.zeros_like(logit_12_base)

        if self.use_sparse_moe_routing and self.router_b12 is not None and f_audio is not None and f_video is not None:
            r12 = self.router_b12(f_video=f_video, f_audio=f_audio, u_tie=u_tie_12, f_joint=f)
            m_12_a, m_12_v = r12["m_audio"], r12["m_video"]
            prob_12_a, prob_12_v = r12["prob_audio"], r12["prob_video"]
            m_12_a_h, m_12_v_h = r12["m_audio_hard"], r12["m_video_hard"]
        else:
            m_12_a = torch.ones_like(logit_12_base)
            m_12_v = torch.ones_like(logit_12_base)
            prob_12_a = torch.full_like(logit_12_base, 0.5)
            prob_12_v = torch.full_like(logit_12_base, 0.5)
            m_12_a_h = m_12_a
            m_12_v_h = m_12_v

        if self.enable_b12_a and self.head_b12_a is not None and f_audio is not None:
            logit_12_a = self.head_b12_a(f_audio).squeeze(-1)
            ref_effect_12 = ref_effect_12 + m_12_a * self.gamma_12_a * logit_12_a
        else:
            logit_12_a = logit_12_base

        if self.enable_b12_v and self.head_b12_v is not None and f_video is not None:
            logit_12_v = self.head_b12_v(f_video).squeeze(-1)
            ref_effect_12 = ref_effect_12 + m_12_v * self.gamma_12_v * logit_12_v
        else:
            logit_12_v = logit_12_base

        logit_12 = logit_12_base + u_tie_12 * ref_effect_12

        # B23 (Medium vs Strong)
        logit_23_base = self.head_b23(f).squeeze(-1)        # [B] (Positive -> Medium, Negative -> Strong)
        u_tie_23 = torch.exp(-torch.abs(logit_23_base) / 2.0)
        ref_effect_23 = torch.zeros_like(logit_23_base)

        if self.use_sparse_moe_routing and self.router_b23 is not None and f_audio is not None and f_video is not None:
            r23 = self.router_b23(f_video=f_video, f_audio=f_audio, u_tie=u_tie_23, f_joint=f)
            m_23_a, m_23_v = r23["m_audio"], r23["m_video"]
            prob_23_a, prob_23_v = r23["prob_audio"], r23["prob_video"]
            m_23_a_h, m_23_v_h = r23["m_audio_hard"], r23["m_video_hard"]
        else:
            m_23_a = torch.ones_like(logit_23_base)
            m_23_v = torch.ones_like(logit_23_base)
            prob_23_a = torch.full_like(logit_23_base, 0.5)
            prob_23_v = torch.full_like(logit_23_base, 0.5)
            m_23_a_h = m_23_a
            m_23_v_h = m_23_v

        if self.enable_b23_a and self.head_b23_a is not None and f_audio is not None:
            logit_23_a = self.head_b23_a(f_audio).squeeze(-1)
            ref_effect_23 = ref_effect_23 + m_23_a * self.gamma_23_a * logit_23_a
        else:
            logit_23_a = logit_23_base

        if self.enable_b23_v and self.head_b23_v is not None and f_video is not None:
            logit_23_v = self.head_b23_v(f_video).squeeze(-1)
            ref_effect_23 = ref_effect_23 + m_23_v * self.gamma_23_v * logit_23_v
        else:
            logit_23_v = logit_23_base

        logit_23 = logit_23_base + u_tie_23 * ref_effect_23

        # B13 (Weak vs Strong)
        logit_13_base = self.head_b13(f).squeeze(-1)        # [B] (Positive -> Weak, Negative -> Strong)
        u_tie_13 = torch.exp(-torch.abs(logit_13_base) / 2.0)
        ref_effect_13 = torch.zeros_like(logit_13_base)

        if self.use_sparse_moe_routing and self.router_b13 is not None and f_audio is not None and f_video is not None:
            r13 = self.router_b13(f_video=f_video, f_audio=f_audio, u_tie=u_tie_13, f_joint=f)
            m_13_a, m_13_v = r13["m_audio"], r13["m_video"]
            prob_13_a, prob_13_v = r13["prob_audio"], r13["prob_video"]
            m_13_a_h, m_13_v_h = r13["m_audio_hard"], r13["m_video_hard"]
        else:
            m_13_a = torch.ones_like(logit_13_base)
            m_13_v = torch.ones_like(logit_13_base)
            prob_13_a = torch.full_like(logit_13_base, 0.5)
            prob_13_v = torch.full_like(logit_13_base, 0.5)
            m_13_a_h = m_13_a
            m_13_v_h = m_13_v

        if self.enable_b13_a and self.head_b13_a is not None and f_audio is not None:
            logit_13_a = self.head_b13_a(f_audio).squeeze(-1)
            ref_effect_13 = ref_effect_13 + m_13_a * self.gamma_13_a * logit_13_a
        else:
            logit_13_a = logit_13_base

        if self.enable_b13_v and self.head_b13_v is not None and f_video is not None:
            logit_13_v = self.head_b13_v(f_video).squeeze(-1)
            ref_effect_13 = ref_effect_13 + m_13_v * self.gamma_13_v * logit_13_v
        else:
            logit_13_v = logit_13_base

        logit_13 = logit_13_base + u_tie_13 * ref_effect_13

        # Head-to-head pairwise winning probabilities
        p_w_over_m = torch.sigmoid(logit_12)               # P(W > M)
        p_m_over_w = 1.0 - p_w_over_m                      # P(M > W)

        p_m_over_s = torch.sigmoid(logit_23)               # P(M > S)
        p_s_over_m = 1.0 - p_m_over_s                      # P(S > M)

        p_w_over_s = torch.sigmoid(logit_13)               # P(W > S)
        p_s_over_w = 1.0 - p_w_over_s                      # P(S > W)

        # 3. Tournament Borda Count Voting
        # Weak wins if it beats Medium AND beats Strong
        v_weak = p_w_over_m + p_w_over_s                   # [B] in [0, 2]
        # Medium wins if it beats Weak AND beats Strong
        v_medium = p_m_over_w + p_m_over_s                 # [B] in [0, 2]
        # Strong wins if it beats Weak AND beats Medium
        v_strong = p_s_over_w + p_s_over_m                 # [B] in [0, 2]

        v_voting = torch.stack([v_weak, v_medium, v_strong], dim=-1)  # [B, 3]

        # Softmax over normalized tournament votes (Rank order: Weak=1, Med=2, Strong=3)
        p_feeding_ranks = F.softmax(v_voting * self.temperature, dim=-1)  # [B, 3]
        p_weak_given_feed = p_feeding_ranks[:, 0]
        p_med_given_feed = p_feeding_ranks[:, 1]
        p_strong_given_feed = p_feeding_ranks[:, 2]

        # 4. Final Hierarchical Combination
        p_final_none = p_none
        p_final_weak = p_feeding * p_weak_given_feed
        p_final_medium = p_feeding * p_med_given_feed
        p_final_strong = p_feeding * p_strong_given_feed

        # Map to raw dataset class indexing: [0: None, 1: Strong, 2: Medium, 3: Weak]
        p_raw = torch.stack([p_final_none, p_final_strong, p_final_medium, p_final_weak], dim=-1)
        p_raw = p_raw / torch.sum(p_raw, dim=-1, keepdim=True)
        logits_raw = torch.log(torch.clamp(p_raw, min=1e-7))

        # Expected physical intensity on 0..3 ordinal scale
        # None=0, Weak=1, Medium=2, Strong=3
        expected_intensity = (
            p_final_none * 0.0 +
            p_final_weak * 1.0 +
            p_final_medium * 2.0 +
            p_final_strong * 3.0
        ).unsqueeze(-1)

        return {
            "logits": logits_raw,
            "probabilities": p_raw,
            "expected_intensity": expected_intensity,
            "intensity_score": expected_intensity,
            # Pairwise logits & probabilities for specialized loss calculation
            "logit_act": logit_act,
            "p_feeding": p_feeding,
            "logit_12": logit_12,
            "logit_12_base": logit_12_base,
            "logit_12_a": logit_12_a,
            "logit_12_v": logit_12_v,
            "u_tie_12": u_tie_12,
            "gamma_12_a": self.gamma_12_a,
            "gamma_12_v": self.gamma_12_v,
            "logit_23": logit_23,
            "logit_23_base": logit_23_base,
            "logit_23_a": logit_23_a,
            "logit_23_v": logit_23_v,
            "u_tie_23": u_tie_23,
            "gamma_23_a": self.gamma_23_a,
            "gamma_23_v": self.gamma_23_v,
            "logit_13": logit_13,
            "logit_13_base": logit_13_base,
            "logit_13_a": logit_13_a,
            "logit_13_v": logit_13_v,
            "u_tie_13": u_tie_13,
            "gamma_13_a": self.gamma_13_a,
            "gamma_13_v": self.gamma_13_v,
            # Pairwise matchup winning probabilities & Borda voting scores
            "p_w_over_m": p_w_over_m,
            "p_m_over_s": p_m_over_s,
            "p_w_over_s": p_w_over_s,
            "v_voting": v_voting,
            # SMoR Routing gates, probabilities & discrete states
            "prob_12_a": prob_12_a,
            "prob_12_v": prob_12_v,
            "m_12_a": m_12_a,
            "m_12_v": m_12_v,
            "m_12_a_hard": m_12_a_h,
            "m_12_v_hard": m_12_v_h,
            "prob_23_a": prob_23_a,
            "prob_23_v": prob_23_v,
            "m_23_a": m_23_a,
            "m_23_v": m_23_v,
            "m_23_a_hard": m_23_a_h,
            "m_23_v_hard": m_23_v_h,
            "prob_13_a": prob_13_a,
            "prob_13_v": prob_13_v,
            "m_13_a": m_13_a,
            "m_13_v": m_13_v,
            "m_13_a_hard": m_13_a_h,
            "m_13_v_hard": m_13_v_h,
        }


class MultimodalTournamentFusion(nn.Module):
    """
    Multimodal Fusion with Hierarchical Pairwise Cross-Boundary Tournament Engine with SMoR (~382K params).
    1. Gated Cross-Modal Fusion: g = sigmoid(W[f_V || f_A]).
    2. Pairwise Boundary Tournament Head: Level 1 Activity Gate + Level 2 3-Way Cross Tournament
       with Sparse Mixture-of-Referees (Audio STFT + Video Kinematics + Dynamic STE Routers) on B12, B23, B13.
    """
    def __init__(
        self,
        dim: int = 224,
        dropout: float = 0.1,
        enable_b12_audio: bool = True,
        enable_b12_video: bool = True,
        enable_b23_audio: bool = True,
        enable_b23_video: bool = True,
        enable_b13_audio: bool = True,
        enable_b13_video: bool = True,
        tie_breakers: Optional[Any] = None,
        use_sparse_moe_routing: bool = True,
        router_hidden_dim: int = 32,
        **kwargs
    ) -> None:
        super().__init__()
        self.dim = dim

        # 1. Gated Reliability Fusion
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 1),
            nn.Sigmoid()
        )
        self.norm_fused = nn.LayerNorm(dim)

        # 2. Residual refinement
        self.proj_joint = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dim)
        )

        # 3. Pairwise Boundary Tournament Decision Head
        self.tournament_head = PairwiseBoundaryTournamentHead(
            dim=dim,
            temperature=2.0,
            enable_b12_audio=enable_b12_audio,
            enable_b12_video=enable_b12_video,
            enable_b23_audio=enable_b23_audio,
            enable_b23_video=enable_b23_video,
            enable_b13_audio=enable_b13_audio,
            enable_b13_video=enable_b13_video,
            tie_breakers=tie_breakers,
            use_sparse_moe_routing=use_sparse_moe_routing,
            router_hidden_dim=router_hidden_dim,
            **kwargs
        )

    def forward(
        self,
        f_video: torch.Tensor,
        f_audio: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        # Step 1: Cross-modal adaptive reliability gating
        combined = torch.cat([f_video, f_audio], dim=-1)  # [B, dim * 2]
        g = self.gate(combined)                            # [B, 1]
        f_fused = self.norm_fused(g * f_video + (1.0 - g) * f_audio)
        f_joint = self.proj_joint(f_fused)

        # Step 2: Pairwise Boundary Tournament with Sparse Mixture-of-Referees (SMoR)
        out = self.tournament_head(f_joint, f_audio=f_audio, f_video=f_video)

        # Step 3: Package metrics
        out["f_fused"] = f_fused
        out["gate"] = g
        out["modality_weights"] = torch.cat([g, 1.0 - g], dim=-1)

        # Shannon entropy uncertainty
        entropy = -torch.sum(out["probabilities"] * torch.log(torch.clamp(out["probabilities"], min=1e-7)), dim=-1, keepdim=True)
        out["uncertainty"] = entropy / 1.386294

        return out
