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
    Flat 4-Class Round-Robin Tournament Head with 6 Pairwise Matchups
    and Configurable Dual Referees (Audio STFT & Video Kinematics).
    
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
    
    Dual Referee Intervention:
      u_tie = exp(-|logit_base|)
      logit = logit_base + u_tie * (enable_A * gamma_A * logit_A + enable_V * gamma_V * logit_V)
      where gamma_A, gamma_V are learnable scalars initialized to 0.5.
    
    Borda Count Voting:
      - V_0 (None)   = P(0 > 1) + P(0 > 2) + P(0 > 3)
      - V_1 (Strong) = P(1 > 0) + P(1 > 2) + P(1 > 3)
      - V_2 (Medium) = P(2 > 0) + P(2 > 1) + P(2 > 3)
      - V_3 (Weak)   = P(3 > 0) + P(3 > 1) + P(3 > 2)
      Strict Invariant: Sum(V_c) = 6.0
    """
    def __init__(
        self,
        dim: int = 224,
        temperature: float = 2.0,
        enable_b01_audio: bool = False,
        enable_b01_video: bool = False,
        enable_b02_audio: bool = False,
        enable_b02_video: bool = False,
        enable_b03_audio: bool = False,
        enable_b03_video: bool = False,
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

        # Parse tie_breakers configuration (object or dict)
        if tie_breakers is not None:
            if hasattr(tie_breakers, "b12"):
                enable_b01_audio = getattr(getattr(tie_breakers, "b01", None), "enable_audio", enable_b01_audio)
                enable_b01_video = getattr(getattr(tie_breakers, "b01", None), "enable_video", enable_b01_video)
                enable_b02_audio = getattr(getattr(tie_breakers, "b02", None), "enable_audio", enable_b02_audio)
                enable_b02_video = getattr(getattr(tie_breakers, "b02", None), "enable_video", enable_b02_video)
                enable_b03_audio = getattr(getattr(tie_breakers, "b03", None), "enable_audio", enable_b03_audio)
                enable_b03_video = getattr(getattr(tie_breakers, "b03", None), "enable_video", enable_b03_video)

                enable_b12_audio = getattr(getattr(tie_breakers, "b12", None), "enable_audio", enable_b12_audio)
                enable_b12_video = getattr(getattr(tie_breakers, "b12", None), "enable_video", enable_b12_video)
                enable_b23_audio = getattr(getattr(tie_breakers, "b23", None), "enable_audio", enable_b23_audio)
                enable_b23_video = getattr(getattr(tie_breakers, "b23", None), "enable_video", enable_b23_video)
                enable_b13_audio = getattr(getattr(tie_breakers, "b13", None), "enable_audio", enable_b13_audio)
                enable_b13_video = getattr(getattr(tie_breakers, "b13", None), "enable_video", enable_b13_video)
            elif isinstance(tie_breakers, dict):
                def _parse_pair(pair_name: str, def_a: bool, def_v: bool):
                    cfg = tie_breakers.get(pair_name, {})
                    if isinstance(cfg, dict):
                        return cfg.get("enable_audio", def_a), cfg.get("enable_video", def_v)
                    return tie_breakers.get(f"enable_{pair_name}_audio", def_a), tie_breakers.get(f"enable_{pair_name}_video", def_v)

                enable_b01_audio, enable_b01_video = _parse_pair("b01", enable_b01_audio, enable_b01_video)
                enable_b02_audio, enable_b02_video = _parse_pair("b02", enable_b02_audio, enable_b02_video)
                enable_b03_audio, enable_b03_video = _parse_pair("b03", enable_b03_audio, enable_b03_video)
                enable_b12_audio, enable_b12_video = _parse_pair("b12", enable_b12_audio, enable_b12_video)
                enable_b23_audio, enable_b23_video = _parse_pair("b23", enable_b23_audio, enable_b23_video)
                enable_b13_audio, enable_b13_video = _parse_pair("b13", enable_b13_audio, enable_b13_video)

        self.enable_b01_a = bool(enable_b01_audio)
        self.enable_b01_v = bool(enable_b01_video)
        self.enable_b02_a = bool(enable_b02_audio)
        self.enable_b02_v = bool(enable_b02_video)
        self.enable_b03_a = bool(enable_b03_audio)
        self.enable_b03_v = bool(enable_b03_video)
        self.enable_b12_a = bool(enable_b12_audio)
        self.enable_b12_v = bool(enable_b12_video)
        self.enable_b23_a = bool(enable_b23_audio)
        self.enable_b23_v = bool(enable_b23_video)
        self.enable_b13_a = bool(enable_b13_audio)
        self.enable_b13_v = bool(enable_b13_video)

        hidden_dim = dim // 2  # 112

        # ----------------------------------------------------------------------
        # Pair 01: None (0) vs Strong (1)
        # ----------------------------------------------------------------------
        self.head_b01 = _make_subspace_head(dim, hidden_dim)
        if self.enable_b01_a:
            self.head_b01_a = _make_subspace_head(dim, hidden_dim)
            self.gamma_01_a = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b01_a = None
            self.gamma_01_a = None

        if self.enable_b01_v:
            self.head_b01_v = _make_subspace_head(dim, hidden_dim)
            self.gamma_01_v = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b01_v = None
            self.gamma_01_v = None

        # ----------------------------------------------------------------------
        # Pair 02: None (0) vs Medium (2)
        # ----------------------------------------------------------------------
        self.head_b02 = _make_subspace_head(dim, hidden_dim)
        if self.enable_b02_a:
            self.head_b02_a = _make_subspace_head(dim, hidden_dim)
            self.gamma_02_a = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b02_a = None
            self.gamma_02_a = None

        if self.enable_b02_v:
            self.head_b02_v = _make_subspace_head(dim, hidden_dim)
            self.gamma_02_v = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b02_v = None
            self.gamma_02_v = None

        # ----------------------------------------------------------------------
        # Pair 03: None (0) vs Weak (3)
        # ----------------------------------------------------------------------
        self.head_b03 = _make_subspace_head(dim, hidden_dim)
        if self.enable_b03_a:
            self.head_b03_a = _make_subspace_head(dim, hidden_dim)
            self.gamma_03_a = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b03_a = None
            self.gamma_03_a = None

        if self.enable_b03_v:
            self.head_b03_v = _make_subspace_head(dim, hidden_dim)
            self.gamma_03_v = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b03_v = None
            self.gamma_03_v = None

        # ----------------------------------------------------------------------
        # Pair 12: Strong (1) vs Medium (2)
        # ----------------------------------------------------------------------
        self.head_b12 = _make_subspace_head(dim, hidden_dim)
        if self.enable_b12_a:
            self.head_b12_a = _make_subspace_head(dim, hidden_dim)
            self.gamma_12_a = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b12_a = None
            self.gamma_12_a = None

        if self.enable_b12_v:
            self.head_b12_v = _make_subspace_head(dim, hidden_dim)
            self.gamma_12_v = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b12_v = None
            self.gamma_12_v = None

        # ----------------------------------------------------------------------
        # Pair 23: Medium (2) vs Weak (3)
        # ----------------------------------------------------------------------
        self.head_b23 = _make_subspace_head(dim, hidden_dim)
        if self.enable_b23_a:
            self.head_b23_a = _make_subspace_head(dim, hidden_dim)
            self.gamma_23_a = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b23_a = None
            self.gamma_23_a = None

        if self.enable_b23_v:
            self.head_b23_v = _make_subspace_head(dim, hidden_dim)
            self.gamma_23_v = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b23_v = None
            self.gamma_23_v = None

        # ----------------------------------------------------------------------
        # Pair 13: Strong (1) vs Weak (3)
        # ----------------------------------------------------------------------
        self.head_b13 = _make_subspace_head(dim, hidden_dim)
        if self.enable_b13_a:
            self.head_b13_a = _make_subspace_head(dim, hidden_dim)
            self.gamma_13_a = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b13_a = None
            self.gamma_13_a = None

        if self.enable_b13_v:
            self.head_b13_v = _make_subspace_head(dim, hidden_dim)
            self.gamma_13_v = nn.Parameter(torch.tensor(0.5))
        else:
            self.head_b13_v = None
            self.gamma_13_v = None

    def _apply_referees(
        self,
        logit_base: torch.Tensor,
        f_audio: Optional[torch.Tensor],
        f_video: Optional[torch.Tensor],
        enable_a: bool,
        head_a: Optional[nn.Module],
        gamma_a: Optional[nn.Parameter],
        enable_v: bool,
        head_v: Optional[nn.Module],
        gamma_v: Optional[nn.Parameter]
    ):
        u_tie = torch.exp(-torch.abs(logit_base))
        ref_effect = torch.zeros_like(logit_base)

        if enable_a and head_a is not None and f_audio is not None:
            logit_a = head_a(f_audio).squeeze(-1)
            ref_effect = ref_effect + gamma_a * logit_a
        else:
            logit_a = logit_base

        if enable_v and head_v is not None and f_video is not None:
            logit_v = head_v(f_video).squeeze(-1)
            ref_effect = ref_effect + gamma_v * logit_v
        else:
            logit_v = logit_base

        logit_final = logit_base + u_tie * ref_effect
        return logit_final, logit_a, logit_v, u_tie

    def forward(
        self,
        f: torch.Tensor,
        f_audio: Optional[torch.Tensor] = None,
        f_video: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            f: Joint fused representation [B, dim]
            f_audio: Audio STFT representation [B, dim] for audio referee
            f_video: Video kinematics representation [B, dim] for video referee
        """
        # --- B01: None (0) vs Strong (1) ---
        logit_01_base = self.head_b01(f).squeeze(-1)
        logit_01, logit_01_a, logit_01_v, u_tie_01 = self._apply_referees(
            logit_01_base, f_audio, f_video,
            self.enable_b01_a, self.head_b01_a, self.gamma_01_a,
            self.enable_b01_v, self.head_b01_v, self.gamma_01_v
        )

        # --- B02: None (0) vs Medium (2) ---
        logit_02_base = self.head_b02(f).squeeze(-1)
        logit_02, logit_02_a, logit_02_v, u_tie_02 = self._apply_referees(
            logit_02_base, f_audio, f_video,
            self.enable_b02_a, self.head_b02_a, self.gamma_02_a,
            self.enable_b02_v, self.head_b02_v, self.gamma_02_v
        )

        # --- B03: None (0) vs Weak (3) ---
        logit_03_base = self.head_b03(f).squeeze(-1)
        logit_03, logit_03_a, logit_03_v, u_tie_03 = self._apply_referees(
            logit_03_base, f_audio, f_video,
            self.enable_b03_a, self.head_b03_a, self.gamma_03_a,
            self.enable_b03_v, self.head_b03_v, self.gamma_03_v
        )

        # --- B12: Strong (1) vs Medium (2) ---
        logit_12_base = self.head_b12(f).squeeze(-1)
        logit_12, logit_12_a, logit_12_v, u_tie_12 = self._apply_referees(
            logit_12_base, f_audio, f_video,
            self.enable_b12_a, self.head_b12_a, self.gamma_12_a,
            self.enable_b12_v, self.head_b12_v, self.gamma_12_v
        )

        # --- B23: Medium (2) vs Weak (3) ---
        logit_23_base = self.head_b23(f).squeeze(-1)
        logit_23, logit_23_a, logit_23_v, u_tie_23 = self._apply_referees(
            logit_23_base, f_audio, f_video,
            self.enable_b23_a, self.head_b23_a, self.gamma_23_a,
            self.enable_b23_v, self.head_b23_v, self.gamma_23_v
        )

        # --- B13: Strong (1) vs Weak (3) ---
        logit_13_base = self.head_b13(f).squeeze(-1)
        logit_13, logit_13_a, logit_13_v, u_tie_13 = self._apply_referees(
            logit_13_base, f_audio, f_video,
            self.enable_b13_a, self.head_b13_a, self.gamma_13_a,
            self.enable_b13_v, self.head_b13_v, self.gamma_13_v
        )

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
            # Pairwise logits & probabilities for specialized loss calculation & logging
            "logit_01": logit_01,
            "logit_01_base": logit_01_base,
            "logit_01_a": logit_01_a,
            "logit_01_v": logit_01_v,
            "u_tie_01": u_tie_01,
            "gamma_01_a": self.gamma_01_a,
            "gamma_01_v": self.gamma_01_v,

            "logit_02": logit_02,
            "logit_02_base": logit_02_base,
            "logit_02_a": logit_02_a,
            "logit_02_v": logit_02_v,
            "u_tie_02": u_tie_02,
            "gamma_02_a": self.gamma_02_a,
            "gamma_02_v": self.gamma_02_v,

            "logit_03": logit_03,
            "logit_03_base": logit_03_base,
            "logit_03_a": logit_03_a,
            "logit_03_v": logit_03_v,
            "u_tie_03": u_tie_03,
            "gamma_03_a": self.gamma_03_a,
            "gamma_03_v": self.gamma_03_v,

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
    2. Flat 4-Class Round-Robin Tournament Head with Configurable Dual Referees (Audio + Video).
    """
    def __init__(
        self,
        dim: int = 224,
        dropout: float = 0.1,
        enable_b01_audio: bool = False,
        enable_b01_video: bool = False,
        enable_b02_audio: bool = False,
        enable_b02_video: bool = False,
        enable_b03_audio: bool = False,
        enable_b03_video: bool = False,
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

        # 3. Flat 4-Class Pairwise Round-Robin Tournament Head
        self.tournament_head = PairwiseBoundaryTournamentHead(
            dim=dim,
            temperature=2.0,
            enable_b01_audio=enable_b01_audio,
            enable_b01_video=enable_b01_video,
            enable_b02_audio=enable_b02_audio,
            enable_b02_video=enable_b02_video,
            enable_b03_audio=enable_b03_audio,
            enable_b03_video=enable_b03_video,
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

        # Step 2: Flat 4-Class Round-Robin Tournament with Dual Referees
        out = self.tournament_head(f_joint, f_audio=f_audio, f_video=f_video)

        # Step 3: Package metrics
        out["f_fused"] = f_fused
        out["gate"] = g
        out["modality_weights"] = torch.cat([g, 1.0 - g], dim=-1)

        # Shannon entropy uncertainty
        entropy = -torch.sum(out["probabilities"] * torch.log(torch.clamp(out["probabilities"], min=1e-7)), dim=-1, keepdim=True)
        out["uncertainty"] = entropy / 1.386294

        return out
