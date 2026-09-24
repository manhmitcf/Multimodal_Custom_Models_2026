import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple


def _make_subspace_head(dim: int, hidden_dim: int = 112) -> nn.Sequential:
    """Helper to build a 2-layer MLP head with LayerNorm using 100% native PyTorch initialization."""
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, 1)
    )


def _parse_tie_breakers(
    tie_breakers: Any,
    enable_b12: bool = True,
    enable_b23: bool = True,
    enable_b13: bool = True
) -> Tuple[bool, bool, bool]:
    """
    Robustly parses tie_breakers configuration from Pydantic models (v1/v2),
    dictionaries, or objects with attributes for video tie-breakers ablation.
    """
    if tie_breakers is not None:
        if hasattr(tie_breakers, "model_dump"):
            d = tie_breakers.model_dump()
        elif hasattr(tie_breakers, "dict"):
            d = tie_breakers.dict()
        elif isinstance(tie_breakers, dict):
            d = dict(tie_breakers)
        else:
            d = {}

        def _extract(v, default):
            if isinstance(v, dict):
                if "enable_video" in v:
                    return bool(v["enable_video"])
                if "enable" in v:
                    return bool(v["enable"])
                if "video" in v:
                    return bool(v["video"])
                return default
            if hasattr(v, "enable_video"):
                return bool(getattr(v, "enable_video"))
            if hasattr(v, "enable"):
                return bool(getattr(v, "enable"))
            return bool(v)

        if d:
            def _get_val(k_flat, k_short, def_v):
                if k_flat in d:
                    return _extract(d[k_flat], def_v)
                if k_short in d:
                    return _extract(d[k_short], def_v)
                v_vid = d.get(f"{k_flat}_video", d.get(f"{k_short}_video", None))
                if v_vid is not None:
                    return _extract(v_vid, def_v)
                return def_v

            enable_b12 = _get_val("enable_b12", "b12", enable_b12)
            enable_b23 = _get_val("enable_b23", "b23", enable_b23)
            enable_b13 = _get_val("enable_b13", "b13", enable_b13)
        else:
            def _get_attr(k_flat, k_short, def_v):
                if hasattr(tie_breakers, k_flat):
                    return _extract(getattr(tie_breakers, k_flat), def_v)
                if hasattr(tie_breakers, k_short):
                    return _extract(getattr(tie_breakers, k_short), def_v)
                v_vid = getattr(tie_breakers, f"{k_flat}_video", getattr(tie_breakers, f"{k_short}_video", None))
                if v_vid is not None:
                    return _extract(v_vid, def_v)
                return def_v

            enable_b12 = _get_attr("enable_b12", "b12", enable_b12)
            enable_b23 = _get_attr("enable_b23", "b23", enable_b23)
            enable_b13 = _get_attr("enable_b13", "b13", enable_b13)

    return bool(enable_b12), bool(enable_b23), bool(enable_b13)


class PairwiseBoundaryTournamentHead(nn.Module):
    """
    Hierarchical Pairwise Cross-Boundary Tournament Head with 3 Configurable Video Kinematics Tie-Breakers (~168K params).

    Level 1: Feeding Activity Gating
      - Distinguishes None (No feeding, quiet water) from Active Feeding (Weak, Medium, Strong).
      - p_feeding = sigmoid(w_act^T * f) in (0, 1)
      - p_none = 1 - p_feeding

    Level 2: 3-Way Pairwise Cross-Boundary Tournament with 3 Video Kinematics Tie-Breakers
      - B12: Weak <-> Medium    -> Base Head on f_joint + Video Kinematics Referee on f_video
      - B23: Medium <-> Strong  -> Base Head on f_joint + Video Kinematics Referee on f_video
      - B13: Weak <-> Strong    -> Base Head on f_joint + Video Kinematics Referee on f_video

    Dynamic Tie-Breaker Intervention:
      logit = logit_base + gamma * u_tie * logit_video
      where u_tie = exp(-|logit_base| / 2.0) represents referee indecisiveness,
      and gamma is a learnable scalar initialized to 1.0.

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
        enable_b12: bool = True,
        enable_b23: bool = True,
        enable_b13: bool = True,
        tie_breakers: Optional[Any] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.dim = dim
        self.temperature = temperature

        # Parse tie_breakers configuration robustly
        self.enable_b12, self.enable_b23, self.enable_b13 = _parse_tie_breakers(
            tie_breakers, enable_b12, enable_b23, enable_b13
        )

        # Level 1: Feeding Activity Gate (None vs Feeding)
        self.activity_head = nn.Sequential(
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Level 2: 3 Specialized Pairwise Subspace Heads + Video Tie-Breakers
        # B12: Weak vs Medium
        self.head_b12 = _make_subspace_head(dim, 112)
        if self.enable_b12:
            self.head_b12_v = _make_subspace_head(dim, 112)
            self.gamma_12 = nn.Parameter(torch.tensor(1.0))
        else:
            self.head_b12_v = None
            self.gamma_12 = None

        # B23: Medium vs Strong
        self.head_b23 = _make_subspace_head(dim, 112)
        if self.enable_b23:
            self.head_b23_v = _make_subspace_head(dim, 112)
            self.gamma_23 = nn.Parameter(torch.tensor(1.0))
        else:
            self.head_b23_v = None
            self.gamma_23 = None

        # B13: Weak vs Strong
        self.head_b13 = _make_subspace_head(dim, 112)
        if self.enable_b13:
            self.head_b13_v = _make_subspace_head(dim, 112)
            self.gamma_13 = nn.Parameter(torch.tensor(1.0))
        else:
            self.head_b13_v = None
            self.gamma_13 = None

    def forward(
        self,
        f: torch.Tensor,
        f_video: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        # 1. Level 1: Feeding Activity Gate
        logit_act = self.activity_head(f).squeeze(-1)       # [B]
        p_feeding = torch.sigmoid(logit_act)                # [B] in (0, 1)
        p_none = torch.clamp(1.0 - p_feeding, min=1e-6)     # [B]

        # 2. Level 2: 3 Pairwise Cross-Boundary Logits & Probabilities
        # B12 Base proposal on f_joint + Video Kinematics Tie-Breaker
        logit_12_base = self.head_b12(f).squeeze(-1)        # [B] (Positive -> Weak, Negative -> Medium)
        if self.enable_b12 and self.head_b12_v is not None and f_video is not None:
            logit_12_v = self.head_b12_v(f_video).squeeze(-1)
            u_tie_12 = torch.exp(-torch.abs(logit_12_base) / 2.0)
            logit_12 = logit_12_base + self.gamma_12 * u_tie_12 * logit_12_v
        else:
            logit_12 = logit_12_base
            logit_12_v = logit_12_base
            u_tie_12 = torch.zeros_like(logit_12_base)

        # B23 Base proposal on f_joint + Video Kinematics Tie-Breaker
        logit_23_base = self.head_b23(f).squeeze(-1)        # [B] (Positive -> Medium, Negative -> Strong)
        if self.enable_b23 and self.head_b23_v is not None and f_video is not None:
            logit_23_v = self.head_b23_v(f_video).squeeze(-1)
            u_tie_23 = torch.exp(-torch.abs(logit_23_base) / 2.0)
            logit_23 = logit_23_base + self.gamma_23 * u_tie_23 * logit_23_v
        else:
            logit_23 = logit_23_base
            logit_23_v = logit_23_base
            u_tie_23 = torch.zeros_like(logit_23_base)

        # B13 Base proposal on f_joint + Video Kinematics Tie-Breaker
        logit_13_base = self.head_b13(f).squeeze(-1)        # [B] (Positive -> Weak, Negative -> Strong)
        if self.enable_b13 and self.head_b13_v is not None and f_video is not None:
            logit_13_v = self.head_b13_v(f_video).squeeze(-1)
            u_tie_13 = torch.exp(-torch.abs(logit_13_base) / 2.0)
            logit_13 = logit_13_base + self.gamma_13 * u_tie_13 * logit_13_v
        else:
            logit_13 = logit_13_base
            logit_13_v = logit_13_base
            u_tie_13 = torch.zeros_like(logit_13_base)

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
            "clipwise_output": logits_raw,
            "logits": logits_raw,
            "probabilities": p_raw,
            "expected_intensity": expected_intensity,
            "intensity_score": expected_intensity,
            # Pairwise logits & probabilities for specialized loss calculation
            "logit_act": logit_act,
            "p_feeding": p_feeding,
            "logit_12": logit_12,
            "logit_12_base": logit_12_base,
            "logit_12_v": logit_12_v,
            "u_tie_12": u_tie_12,
            "gamma_12": self.gamma_12,
            "gamma_12_v": self.gamma_12,
            "logit_23": logit_23,
            "logit_23_base": logit_23_base,
            "logit_23_v": logit_23_v,
            "u_tie_23": u_tie_23,
            "gamma_23": self.gamma_23,
            "gamma_23_v": self.gamma_23,
            "logit_13": logit_13,
            "logit_13_base": logit_13_base,
            "logit_13_v": logit_13_v,
            "u_tie_13": u_tie_13,
            "gamma_13": self.gamma_13,
            "gamma_13_v": self.gamma_13,
            # Pairwise matchup winning probabilities & Borda voting scores
            "p_w_over_m": p_w_over_m,
            "p_m_over_s": p_m_over_s,
            "p_w_over_s": p_w_over_s,
            "v_voting": v_voting
        }


class MultimodalTournamentFusion(nn.Module):
    """
    Multimodal Fusion with Hierarchical Pairwise Cross-Boundary Tournament Engine (~219K params).
    1. Gated Cross-Modal Fusion: g = sigmoid(W[f_V || f_A]).
    2. Pairwise Boundary Tournament Head: Level 1 Activity Gate + Level 2 3-Way Cross Tournament
       with 3 Configurable Video Kinematics Tie-Breakers on B12, B23, B13.
    """
    def __init__(
        self,
        dim: int = 224,
        dropout: float = 0.1,
        enable_b12: bool = True,
        enable_b23: bool = True,
        enable_b13: bool = True,
        tie_breakers: Optional[Any] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.dim = dim
        self.enable_b12, self.enable_b23, self.enable_b13 = _parse_tie_breakers(
            tie_breakers, enable_b12, enable_b23, enable_b13
        )

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

        # 3. Pairwise Boundary Tournament Decision Head with 3 Video Tie-Breakers
        self.tournament_head = PairwiseBoundaryTournamentHead(
            dim=dim,
            temperature=2.0,
            enable_b12=self.enable_b12,
            enable_b23=self.enable_b23,
            enable_b13=self.enable_b13,
            tie_breakers=tie_breakers,
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

        # Step 2: Pairwise Boundary Tournament with 3 Configurable Video Kinematics Tie-Breakers
        out = self.tournament_head(f_joint, f_video=f_video)

        # Step 3: Package metrics
        out["f_fused"] = f_fused
        out["gate"] = g
        out["modality_weights"] = torch.cat([g, 1.0 - g], dim=-1)

        # Shannon entropy uncertainty
        entropy = -torch.sum(out["probabilities"] * torch.log(torch.clamp(out["probabilities"], min=1e-7)), dim=-1, keepdim=True)
        out["uncertainty"] = entropy / 1.386294

        return out
