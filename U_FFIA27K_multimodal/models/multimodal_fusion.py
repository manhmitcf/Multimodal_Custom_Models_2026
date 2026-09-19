import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


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
    Hierarchical Pairwise Cross-Boundary Tournament Head with Dual Cross-Modal Referees (~208K params).
    
    Level 1: Feeding Activity Gating
      - Distinguishes None (No feeding, quiet water) from Active Feeding (Weak, Medium, Strong).
      - p_feeding = sigmoid(w_act^T * f) in (0, 1)
      - p_none = 1 - p_feeding

    Level 2: 3-Way Pairwise Cross-Boundary Tournament with Dual Referees (Audio STFT + Video Kinematics)
      - B12: Weak <-> Medium    -> Base Head on f_joint + Optional Audio STFT + Optional Video Kinematics
      - B23: Medium <-> Strong  -> Base Head on f_joint + Optional Audio STFT + Optional Video Kinematics
      - B13: Weak <-> Strong    -> Base Head on f_joint + Optional Audio STFT + Optional Video Kinematics

    Dual Referee Intervention:
      logit = logit_base + u_tie * (gamma_A * logit_A + gamma_V * logit_V)
      where u_tie = exp(-|logit_base|) represents referee indecisiveness.

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

        # Level 2: 3 Specialized Pairwise Subspace Heads + Dual Referees
        # B12: Weak vs Medium
        self.head_b12 = _make_subspace_head(dim, 112)
        if self.enable_b12_a:
            self.head_b12_a = _make_subspace_head(dim, 112)
            self.gamma_12_a = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b12_a = None
            self.gamma_12_a = None

        if self.enable_b12_v:
            self.head_b12_v = _make_subspace_head(dim, 112)
            self.gamma_12_v = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b12_v = None
            self.gamma_12_v = None

        # B23: Medium vs Strong
        self.head_b23 = _make_subspace_head(dim, 112)
        if self.enable_b23_a:
            self.head_b23_a = _make_subspace_head(dim, 112)
            self.gamma_23_a = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b23_a = None
            self.gamma_23_a = None

        if self.enable_b23_v:
            self.head_b23_v = _make_subspace_head(dim, 112)
            self.gamma_23_v = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b23_v = None
            self.gamma_23_v = None

        # B13: Weak vs Strong
        self.head_b13 = _make_subspace_head(dim, 112)
        if self.enable_b13_a:
            self.head_b13_a = _make_subspace_head(dim, 112)
            self.gamma_13_a = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b13_a = None
            self.gamma_13_a = None

        if self.enable_b13_v:
            self.head_b13_v = _make_subspace_head(dim, 112)
            self.gamma_13_v = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b13_v = None
            self.gamma_13_v = None

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

        # 2. Level 2: 3 Pairwise Cross-Boundary Logits with Dual Referees
        # B12 (Weak vs Medium)
        logit_12_base = self.head_b12(f).squeeze(-1)        # [B] (Positive -> Weak, Negative -> Medium)
        u_tie_12 = torch.exp(-torch.abs(logit_12_base))
        ref_effect_12 = torch.zeros_like(logit_12_base)

        if self.enable_b12_a and self.head_b12_a is not None and f_audio is not None:
            logit_12_a = self.head_b12_a(f_audio).squeeze(-1)
            ref_effect_12 = ref_effect_12 + self.gamma_12_a * logit_12_a
        else:
            logit_12_a = logit_12_base

        if self.enable_b12_v and self.head_b12_v is not None and f_video is not None:
            logit_12_v = self.head_b12_v(f_video).squeeze(-1)
            ref_effect_12 = ref_effect_12 + self.gamma_12_v * logit_12_v
        else:
            logit_12_v = logit_12_base

        logit_12 = logit_12_base + u_tie_12 * ref_effect_12

        # B23 (Medium vs Strong)
        logit_23_base = self.head_b23(f).squeeze(-1)        # [B] (Positive -> Medium, Negative -> Strong)
        u_tie_23 = torch.exp(-torch.abs(logit_23_base))
        ref_effect_23 = torch.zeros_like(logit_23_base)

        if self.enable_b23_a and self.head_b23_a is not None and f_audio is not None:
            logit_23_a = self.head_b23_a(f_audio).squeeze(-1)
            ref_effect_23 = ref_effect_23 + self.gamma_23_a * logit_23_a
        else:
            logit_23_a = logit_23_base

        if self.enable_b23_v and self.head_b23_v is not None and f_video is not None:
            logit_23_v = self.head_b23_v(f_video).squeeze(-1)
            ref_effect_23 = ref_effect_23 + self.gamma_23_v * logit_23_v
        else:
            logit_23_v = logit_23_base

        logit_23 = logit_23_base + u_tie_23 * ref_effect_23

        # B13 (Weak vs Strong)
        logit_13_base = self.head_b13(f).squeeze(-1)        # [B] (Positive -> Weak, Negative -> Strong)
        u_tie_13 = torch.exp(-torch.abs(logit_13_base))
        ref_effect_13 = torch.zeros_like(logit_13_base)

        if self.enable_b13_a and self.head_b13_a is not None and f_audio is not None:
            logit_13_a = self.head_b13_a(f_audio).squeeze(-1)
            ref_effect_13 = ref_effect_13 + self.gamma_13_a * logit_13_a
        else:
            logit_13_a = logit_13_base

        if self.enable_b13_v and self.head_b13_v is not None and f_video is not None:
            logit_13_v = self.head_b13_v(f_video).squeeze(-1)
            ref_effect_13 = ref_effect_13 + self.gamma_13_v * logit_13_v
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
            "v_voting": v_voting
        }


class MultimodalTournamentFusion(nn.Module):
    """
    Multimodal Fusion with Hierarchical Pairwise Cross-Boundary Tournament Engine (~296K params).
    1. Gated Cross-Modal Fusion: g = sigmoid(W[f_V || f_A]).
    2. Pairwise Boundary Tournament Head: Level 1 Activity Gate + Level 2 3-Way Cross Tournament
       with Dual Cross-Modal Referees (Audio STFT + Video Kinematics) on B12, B23, B13.
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

        # Step 2: Pairwise Boundary Tournament with Dual Cross-Modal Referees
        out = self.tournament_head(f_joint, f_audio=f_audio, f_video=f_video)

        # Step 3: Package metrics
        out["f_fused"] = f_fused
        out["gate"] = g
        out["modality_weights"] = torch.cat([g, 1.0 - g], dim=-1)

        # Shannon entropy uncertainty
        entropy = -torch.sum(out["probabilities"] * torch.log(torch.clamp(out["probabilities"], min=1e-7)), dim=-1, keepdim=True)
        out["uncertainty"] = entropy / 1.386294

        return out
