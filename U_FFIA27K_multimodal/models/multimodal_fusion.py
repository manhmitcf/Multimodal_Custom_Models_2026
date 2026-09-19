import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PairwiseBoundaryTournamentHead(nn.Module):
    """
    Flat 4-Class Round-Robin Tournament Head with 6 Pairwise Matchups
    and 6 Configurable Video Kinematics Tie-Breakers.
    
    Matchup Pairs (Binomial(4, 2) = 6):
      - B01: None (0) vs Strong (1)
      - B02: None (0) vs Medium (2)
      - B03: None (0) vs Weak (3)
      - B12: Strong (1) vs Medium (2)
      - B23: Medium (2) vs Weak (3)
      - B13: Strong (1) vs Weak (3)
    
    Sign Convention for (i, j) with i < j:
      - logit_ij > 0: Favors class i -> P(i > j) = sigmoid(logit_ij)
      - logit_ij < 0: Favors class j -> P(j > i) = 1 - sigmoid(logit_ij)
    
    Borda Count Voting:
      - V_0 (None)   = P(0 > 1) + P(0 > 2) + P(0 > 3)
      - V_1 (Strong) = P(1 > 0) + P(1 > 2) + P(1 > 3)
      - V_2 (Medium) = P(2 > 0) + P(2 > 1) + P(2 > 3)
      - V_3 (Weak)   = P(3 > 0) + P(3 > 1) + P(3 > 2)
      Invariant: Sum(V_c) = 6.0
    """
    def __init__(
        self,
        dim: int = 224,
        temperature: float = 2.0,
        enable_b01: bool = False,
        enable_b02: bool = False,
        enable_b03: bool = False,
        enable_b12: bool = False,
        enable_b23: bool = False,
        enable_b13: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.enable_b01 = bool(enable_b01)
        self.enable_b02 = bool(enable_b02)
        self.enable_b03 = bool(enable_b03)
        self.enable_b12 = bool(enable_b12)
        self.enable_b23 = bool(enable_b23)
        self.enable_b13 = bool(enable_b13)

        hidden_dim = dim // 2  # 112

        # ----------------------------------------------------------------------
        # 6 Base Pairwise Heads on Joint Multimodal Embedding f_joint [B, 224]
        # ----------------------------------------------------------------------
        def _make_base_head():
            return nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1)
            )

        def _make_video_head():
            return nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1)
            )

        # Pair 01: None (0) vs Strong (1)
        self.head_b01 = _make_base_head()
        if self.enable_b01:
            self.head_b01_v = _make_video_head()
            self.gamma_01 = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b01_v = None
            self.gamma_01 = None

        # Pair 02: None (0) vs Medium (2)
        self.head_b02 = _make_base_head()
        if self.enable_b02:
            self.head_b02_v = _make_video_head()
            self.gamma_02 = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b02_v = None
            self.gamma_02 = None

        # Pair 03: None (0) vs Weak (3)
        self.head_b03 = _make_base_head()
        if self.enable_b03:
            self.head_b03_v = _make_video_head()
            self.gamma_03 = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b03_v = None
            self.gamma_03 = None

        # Pair 12: Strong (1) vs Medium (2)
        self.head_b12 = _make_base_head()
        if self.enable_b12:
            self.head_b12_v = _make_video_head()
            self.gamma_12 = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b12_v = None
            self.gamma_12 = None

        # Pair 23: Medium (2) vs Weak (3)
        self.head_b23 = _make_base_head()
        if self.enable_b23:
            self.head_b23_v = _make_video_head()
            self.gamma_23 = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b23_v = None
            self.gamma_23 = None

        # Pair 13: Strong (1) vs Weak (3)
        self.head_b13 = _make_base_head()
        if self.enable_b13:
            self.head_b13_v = _make_video_head()
            self.gamma_13 = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b13_v = None
            self.gamma_13 = None

    def forward(
        self,
        f: torch.Tensor,
        f_video: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            f: Joint fused representation [B, dim]
            f_video: Optional video kinematics representation [B, dim] for tie-breaking
        """
        # --- B01: None (0) vs Strong (1) ---
        logit_01_base = self.head_b01(f).squeeze(-1)
        if self.enable_b01 and self.head_b01_v is not None and f_video is not None:
            logit_01_v = self.head_b01_v(f_video).squeeze(-1)
            u_tie_01 = torch.exp(-torch.abs(logit_01_base))
            logit_01 = logit_01_base + self.gamma_01 * u_tie_01 * logit_01_v
        else:
            logit_01 = logit_01_base
            logit_01_v = logit_01_base
            u_tie_01 = torch.zeros_like(logit_01_base)

        # --- B02: None (0) vs Medium (2) ---
        logit_02_base = self.head_b02(f).squeeze(-1)
        if self.enable_b02 and self.head_b02_v is not None and f_video is not None:
            logit_02_v = self.head_b02_v(f_video).squeeze(-1)
            u_tie_02 = torch.exp(-torch.abs(logit_02_base))
            logit_02 = logit_02_base + self.gamma_02 * u_tie_02 * logit_02_v
        else:
            logit_02 = logit_02_base
            logit_02_v = logit_02_base
            u_tie_02 = torch.zeros_like(logit_02_base)

        # --- B03: None (0) vs Weak (3) ---
        logit_03_base = self.head_b03(f).squeeze(-1)
        if self.enable_b03 and self.head_b03_v is not None and f_video is not None:
            logit_03_v = self.head_b03_v(f_video).squeeze(-1)
            u_tie_03 = torch.exp(-torch.abs(logit_03_base))
            logit_03 = logit_03_base + self.gamma_03 * u_tie_03 * logit_03_v
        else:
            logit_03 = logit_03_base
            logit_03_v = logit_03_base
            u_tie_03 = torch.zeros_like(logit_03_base)

        # --- B12: Strong (1) vs Medium (2) ---
        logit_12_base = self.head_b12(f).squeeze(-1)
        if self.enable_b12 and self.head_b12_v is not None and f_video is not None:
            logit_12_v = self.head_b12_v(f_video).squeeze(-1)
            u_tie_12 = torch.exp(-torch.abs(logit_12_base))
            logit_12 = logit_12_base + self.gamma_12 * u_tie_12 * logit_12_v
        else:
            logit_12 = logit_12_base
            logit_12_v = logit_12_base
            u_tie_12 = torch.zeros_like(logit_12_base)

        # --- B23: Medium (2) vs Weak (3) ---
        logit_23_base = self.head_b23(f).squeeze(-1)
        if self.enable_b23 and self.head_b23_v is not None and f_video is not None:
            logit_23_v = self.head_b23_v(f_video).squeeze(-1)
            u_tie_23 = torch.exp(-torch.abs(logit_23_base))
            logit_23 = logit_23_base + self.gamma_23 * u_tie_23 * logit_23_v
        else:
            logit_23 = logit_23_base
            logit_23_v = logit_23_base
            u_tie_23 = torch.zeros_like(logit_23_base)

        # --- B13: Strong (1) vs Weak (3) ---
        logit_13_base = self.head_b13(f).squeeze(-1)
        if self.enable_b13 and self.head_b13_v is not None and f_video is not None:
            logit_13_v = self.head_b13_v(f_video).squeeze(-1)
            u_tie_13 = torch.exp(-torch.abs(logit_13_base))
            logit_13 = logit_13_base + self.gamma_13 * u_tie_13 * logit_13_v
        else:
            logit_13 = logit_13_base
            logit_13_v = logit_13_base
            u_tie_13 = torch.zeros_like(logit_13_base)

        # ----------------------------------------------------------------------
        # Pairwise Matchup Probabilities
        # ----------------------------------------------------------------------
        p_0_over_1 = torch.sigmoid(logit_01)
        p_1_over_0 = 1.0 - p_0_over_1

        p_0_over_2 = torch.sigmoid(logit_02)
        p_2_over_0 = 1.0 - p_0_over_2

        p_0_over_3 = torch.sigmoid(logit_03)
        p_3_over_0 = 1.0 - p_0_over_3

        p_1_over_2 = torch.sigmoid(logit_12)
        p_2_over_1 = 1.0 - p_1_over_2

        p_2_over_3 = torch.sigmoid(logit_23)
        p_3_over_2 = 1.0 - p_2_over_3

        p_1_over_3 = torch.sigmoid(logit_13)
        p_3_over_1 = 1.0 - p_1_over_3

        # ----------------------------------------------------------------------
        # Borda Count Voting across 4 Classes (Each plays 3 matchups)
        # ----------------------------------------------------------------------
        v_0 = p_0_over_1 + p_0_over_2 + p_0_over_3
        v_1 = p_1_over_0 + p_1_over_2 + p_1_over_3
        v_2 = p_2_over_0 + p_2_over_1 + p_2_over_3
        v_3 = p_3_over_0 + p_3_over_1 + p_3_over_2

        # Raw class indexing: [0: None, 1: Strong, 2: Medium, 3: Weak]
        v_voting = torch.stack([v_0, v_1, v_2, v_3], dim=-1)  # [B, 4], sum = 6.0

        # Calibrated 4-class probabilities via temperature-scaled Softmax
        p_raw = F.softmax(v_voting * self.temperature, dim=-1)  # [B, 4]
        p_raw = p_raw / torch.sum(p_raw, dim=-1, keepdim=True)
        logits_raw = torch.log(torch.clamp(p_raw, min=1e-7))

        # Expected physical intensity on 0..3 ordinal scale: None=0, Weak=1, Medium=2, Strong=3
        expected_intensity = (
            p_raw[:, 0] * 0.0 +
            p_raw[:, 3] * 1.0 +
            p_raw[:, 2] * 2.0 +
            p_raw[:, 1] * 3.0
        ).unsqueeze(-1)

        return {
            "logits": logits_raw,
            "probabilities": p_raw,
            "expected_intensity": expected_intensity,
            "intensity_score": expected_intensity,
            # Pairwise logits & probabilities for specialized loss calculation
            "logit_01": logit_01,
            "logit_01_base": logit_01_base,
            "logit_01_v": logit_01_v,
            "u_tie_01": u_tie_01,
            "gamma_01": self.gamma_01,
            "logit_02": logit_02,
            "logit_02_base": logit_02_base,
            "logit_02_v": logit_02_v,
            "u_tie_02": u_tie_02,
            "gamma_02": self.gamma_02,
            "logit_03": logit_03,
            "logit_03_base": logit_03_base,
            "logit_03_v": logit_03_v,
            "u_tie_03": u_tie_03,
            "gamma_03": self.gamma_03,
            "logit_12": logit_12,
            "logit_12_base": logit_12_base,
            "logit_12_v": logit_12_v,
            "u_tie_12": u_tie_12,
            "gamma_12": self.gamma_12,
            "logit_23": logit_23,
            "logit_23_base": logit_23_base,
            "logit_23_v": logit_23_v,
            "u_tie_23": u_tie_23,
            "gamma_23": self.gamma_23,
            "logit_13": logit_13,
            "logit_13_base": logit_13_base,
            "logit_13_v": logit_13_v,
            "u_tie_13": u_tie_13,
            "gamma_13": self.gamma_13,
            # Pairwise matchup winning probabilities & Borda voting scores
            "p_0_over_1": p_0_over_1,
            "p_0_over_2": p_0_over_2,
            "p_0_over_3": p_0_over_3,
            "p_1_over_2": p_1_over_2,
            "p_2_over_3": p_2_over_3,
            "p_1_over_3": p_1_over_3,
            "v_voting": v_voting
        }


class MultimodalTournamentFusion(nn.Module):
    """
    Multimodal Fusion with Flat 4-Class Round-Robin Tournament Engine.
    1. Gated Cross-Modal Fusion: g = sigmoid(W[f_V || f_A]).
    2. Flat 4-Class Round-Robin Tournament Head with 6 Configurable Video Kinematics Tie-Breakers.
    """
    def __init__(
        self,
        dim: int = 224,
        dropout: float = 0.1,
        enable_b01: bool = False,
        enable_b02: bool = False,
        enable_b03: bool = False,
        enable_b12: bool = False,
        enable_b23: bool = False,
        enable_b13: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.dim = dim
        self.enable_b01 = bool(enable_b01)
        self.enable_b02 = bool(enable_b02)
        self.enable_b03 = bool(enable_b03)
        self.enable_b12 = bool(enable_b12)
        self.enable_b23 = bool(enable_b23)
        self.enable_b13 = bool(enable_b13)

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

        # 3. Flat 4-Class Pairwise Round-Robin Tournament Head
        self.tournament_head = PairwiseBoundaryTournamentHead(
            dim=dim,
            temperature=2.0,
            enable_b01=self.enable_b01,
            enable_b02=self.enable_b02,
            enable_b03=self.enable_b03,
            enable_b12=self.enable_b12,
            enable_b23=self.enable_b23,
            enable_b13=self.enable_b13,
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

        # Step 2: Flat 4-Class Round-Robin Tournament with Configurable Video Tie-Breakers
        out = self.tournament_head(f_joint, f_video=f_video)

        # Step 3: Package metrics
        out["f_fused"] = f_fused
        out["gate"] = g
        out["modality_weights"] = torch.cat([g, 1.0 - g], dim=-1)

        # Shannon entropy uncertainty
        entropy = -torch.sum(out["probabilities"] * torch.log(torch.clamp(out["probabilities"], min=1e-7)), dim=-1, keepdim=True)
        out["uncertainty"] = entropy / 1.386294

        return out
