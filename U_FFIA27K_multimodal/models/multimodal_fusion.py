import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PairwiseBoundaryTournamentHead(nn.Module):
    """
    Hierarchical Pairwise Cross-Boundary Tournament Head (~80K params).
    
    Level 1: Feeding Activity Gating
      - Distinguishes None (No feeding, quiet water) from Active Feeding (Weak, Medium, Strong).
      - p_feeding = sigmoid(w_act^T * f) in (0, 1)
      - p_none = 1 - p_feeding

    Level 2: 3-Way Pairwise Cross-Boundary Tournament
      - B12: Weak <-> Medium    -> P(W > M) = sigmoid(s_12)
      - B23: Medium <-> Strong  -> P(M > S) = sigmoid(s_23)
      - B13: Weak <-> Strong    -> P(W > S) = sigmoid(s_13)  [Cross-skipping protection boundary]

    Tournament Scoring (Borda count):
      - V_Weak   = P(W > M) + P(W > S)
      - V_Medium = (1 - P(W > M)) + P(M > S)
      - V_Strong = (1 - P(W > S)) + (1 - P(M > S))
    """
    def __init__(self, dim: int = 224, temperature: float = 2.0) -> None:
        super().__init__()
        self.dim = dim
        self.temperature = temperature

        # Level 1: Feeding Activity Gate (None vs Feeding)
        self.activity_head = nn.Sequential(
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Level 2: 3 Specialized Pairwise Subspace Expert Heads
        # B12: Weak vs Medium
        self.head_b12 = nn.Sequential(
            nn.Linear(dim, 112),
            nn.GELU(),
            nn.LayerNorm(112),
            nn.Linear(112, 1)
        )

        # B23: Medium vs Strong (Base Joint Representation Head)
        self.head_b23 = nn.Sequential(
            nn.Linear(dim, 112),
            nn.GELU(),
            nn.LayerNorm(112),
            nn.Linear(112, 1)
        )

        # B23 Audio STFT Tie-Breaker Head (Ultrasonic Bubble Bursts Specialist)
        self.head_b23_a = nn.Sequential(
            nn.Linear(dim, 112),
            nn.GELU(),
            nn.LayerNorm(112),
            nn.Linear(112, 1)
        )
        self.gamma_23 = nn.Parameter(torch.tensor(0.5))

        # B13: Weak vs Strong (Direct cross-boundary anchor protection)
        self.head_b13 = nn.Sequential(
            nn.Linear(dim, 112),
            nn.GELU(),
            nn.LayerNorm(112),
            nn.Linear(112, 1)
        )


    def forward(self, f: torch.Tensor, f_audio: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B = f.size(0)

        # 1. Level 1: Feeding Activity Gate
        logit_act = self.activity_head(f).squeeze(-1)       # [B]
        p_feeding = torch.sigmoid(logit_act)                # [B] in (0, 1)
        p_none = torch.clamp(1.0 - p_feeding, min=1e-6)     # [B]

        # 2. Level 2: 3 Pairwise Cross-Boundary Logits & Probabilities
        logit_12 = self.head_b12(f).squeeze(-1)            # [B] (Positive -> Weak, Negative -> Medium)
        logit_13 = self.head_b13(f).squeeze(-1)            # [B] (Positive -> Weak, Negative -> Strong)

        # B23 Base proposal on f_joint + Audio STFT Tie-Breaker
        logit_23_base = self.head_b23(f).squeeze(-1)       # [B] (Positive -> Medium, Negative -> Strong)
        if f_audio is not None:
            logit_23_a = self.head_b23_a(f_audio).squeeze(-1)
            u_tie_23 = torch.exp(-torch.abs(logit_23_base))  # peaks when base proposal is indecisive
            logit_23 = logit_23_base + self.gamma_23 * u_tie_23 * logit_23_a
        else:
            logit_23 = logit_23_base
            logit_23_a = logit_23_base
            u_tie_23 = torch.zeros_like(logit_23_base)


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
            "logit_23": logit_23,
            "logit_23_base": logit_23_base,
            "logit_23_a": logit_23_a,
            "u_tie_23": u_tie_23,
            "u_tie": u_tie_23,
            "gamma_23": self.gamma_23,
            "gamma": self.gamma_23,
            "logit_13": logit_13,
            "p_w_over_m": p_w_over_m,
            "p_m_over_s": p_m_over_s,
            "p_w_over_s": p_w_over_s,
            "v_voting": v_voting
        }


class ChannelGatedBilinearFusion(nn.Module):
    """
    Channel-wise Gated Bilinear Cross-Modal Fusion (CGB-Fusion).
    Combines:
      1. 224-Dimensional Channel-wise Gated Blending:
         g = sigmoid(W_g [f_V || f_A] + b_g) in (0, 1)^224 with b_g=0 -> g=0.5 at Step 0.
         z_V = GELU(W_V f_V), z_A = GELU(W_A f_A)
         f_gated = LayerNorm(g * z_V + (1 - g) * z_A)
      2. Second-Order Bilinear Cross-Modal Interaction:
         h_V = W_b1 f_V, h_A = W_b2 f_A
         f_bilinear = LayerNorm(h_V * h_A) (Variance stabilized via LayerNorm)
      3. Residual Joint Projection:
         f_joint = LayerNorm(f_gated + Dropout(GELU(W_joint [f_gated || f_bilinear])))
    """
    def __init__(self, dim: int = 224, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim

        # 1. Channel-wise Gating (dim-dimensional independent channel gates)
        self.gate_proj = nn.Linear(dim * 2, dim)
        self.proj_v = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU()
        )
        self.proj_a = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU()
        )
        self.norm_gated = nn.LayerNorm(dim)

        # 2. Second-Order Bilinear Interaction
        self.bilinear_v = nn.Linear(dim, dim)
        self.bilinear_a = nn.Linear(dim, dim)
        self.norm_bilinear = nn.LayerNorm(dim)

        # 3. Residual Joint Projection
        self.joint_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.norm_joint = nn.LayerNorm(dim)

        # Initialize weights with balanced step 0 gating
        self._init_weights()

    def _init_weights(self) -> None:
        # Balanced 50/50 Channel Gate at Step 0: bias=0, Xavier uniform weights with scale
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.1)
        nn.init.zeros_(self.gate_proj.bias)

        # Bilinear projections
        nn.init.xavier_uniform_(self.bilinear_v.weight)
        nn.init.zeros_(self.bilinear_v.bias)
        nn.init.xavier_uniform_(self.bilinear_a.weight)
        nn.init.zeros_(self.bilinear_a.bias)

        # Branch projections
        for m in self.proj_v.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.proj_a.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Joint projection
        for m in self.joint_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, f_video: torch.Tensor, f_audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Step 1: Channel-wise Gating
        combined = torch.cat([f_video, f_audio], dim=-1)     # [B, dim * 2]
        g = torch.sigmoid(self.gate_proj(combined))            # [B, dim] in (0, 1)
        z_v = self.proj_v(f_video)                             # [B, dim]
        z_a = self.proj_a(f_audio)                             # [B, dim]
        f_gated = self.norm_gated(g * z_v + (1.0 - g) * z_a)  # [B, dim]

        # Step 2: Second-Order Bilinear Cross-Modal Interaction
        h_v = self.bilinear_v(f_video)                         # [B, dim]
        h_a = self.bilinear_a(f_audio)                         # [B, dim]
        f_bilinear = self.norm_bilinear(h_v * h_a)             # [B, dim]

        # Step 3: Residual Joint Projection
        f_cat = torch.cat([f_gated, f_bilinear], dim=-1)       # [B, dim * 2]
        f_joint = self.norm_joint(f_gated + self.joint_proj(f_cat))  # [B, dim]

        return f_joint, f_gated, g


class MultimodalTournamentFusion(nn.Module):
    """
    Channel-wise Gated Bilinear Multimodal Tournament Fusion Engine.
    Combines:
      1. CGB-Fusion (ChannelGatedBilinearFusion): 224D Channel-wise Gating + Second-Order Bilinear Interaction + Residual Joint Refinement.
      2. Hierarchical Pairwise Boundary Tournament Head: Level 1 Activity Gate + Level 2 3-Way Cross Tournament
         with Audio STFT Tie-Breaker on B23 (Medium vs Strong).
    """
    def __init__(self, dim: int = 224, dropout: float = 0.1, **kwargs) -> None:
        super().__init__()
        self.dim = dim

        # 1. CGB-Fusion Core
        self.cgb_fusion = ChannelGatedBilinearFusion(dim=dim, dropout=dropout)

        # 2. Pairwise Boundary Tournament Decision Head
        self.tournament_head = PairwiseBoundaryTournamentHead(dim=dim, temperature=2.0)

    def forward(
        self,
        f_video: torch.Tensor,
        f_audio: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        # Step 1: Channel-wise Gated Bilinear Fusion
        f_joint, f_gated, g = self.cgb_fusion(f_video=f_video, f_audio=f_audio)

        # Step 2: Pairwise Boundary Tournament with Audio STFT Tie-Breaker on B23
        out = self.tournament_head(f_joint, f_audio=f_audio)

        # Step 3: Package metrics
        out["f_fused"] = f_joint
        out["f_gated"] = f_gated
        out["gate"] = g
        # Channel-mean modality weights for logging [B, 2]
        g_mean = g.mean(dim=-1, keepdim=True)
        out["modality_weights"] = torch.cat([g_mean, 1.0 - g_mean], dim=-1)

        # Shannon entropy uncertainty
        entropy = -torch.sum(out["probabilities"] * torch.log(torch.clamp(out["probabilities"], min=1e-7)), dim=-1, keepdim=True)
        out["uncertainty"] = entropy / 1.386294

        return out


# Canonical alias reflecting exact algorithm in paper
ChannelGatedBilinearTournamentFusion = MultimodalTournamentFusion
