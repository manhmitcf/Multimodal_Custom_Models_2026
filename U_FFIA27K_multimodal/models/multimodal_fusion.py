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


class TemporalCrossModalAttentionFusion(nn.Module):
    """
    Temporal Cross-Modal Attention Fusion (TCA-Fusion) (~0.47M params).
    1. Bidirectional Temporal Cross-Modal Attention:
       - Video-to-Audio (V -> A): Query=tokens_video, Key/Value=tokens_audio
       - Audio-to-Video (A -> V): Query=tokens_audio, Key/Value=tokens_video
       - Multi-Head Attention (num_heads=4, dim=224, head_dim=56).
    2. Temporal Mean Pooling: Collapses T=2 attended sequences to modality vectors.
    3. Channel-wise Adaptive Gating: Independent 224-dim gate vector g in (0, 1)^224.
    4. Hierarchical Pairwise Boundary Tournament Decision Engine.
    """
    def __init__(
        self,
        dim: int = 224,
        num_heads: int = 4,
        dropout: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # 1. Bidirectional Multi-Head Cross-Attention (4 heads @ dim 224)
        self.mha_v2a = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_v = nn.LayerNorm(dim)
        self.drop_v = nn.Dropout(dropout)

        self.mha_a2v = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_a = nn.LayerNorm(dim)
        self.drop_a = nn.Dropout(dropout)

        # Output normalizers for combined backbone + attended features
        self.norm_v_out = nn.LayerNorm(dim)
        self.norm_a_out = nn.LayerNorm(dim)

        # 2. Channel-wise Adaptive Gating: g in (0, 1)^dim
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.norm_fused = nn.LayerNorm(dim)

        # 3. Residual Joint Projection
        self.proj_joint = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dim)
        )

        # 4. Pairwise Boundary Tournament Decision Head (Kept 100% intact)
        self.tournament_head = PairwiseBoundaryTournamentHead(dim=dim, temperature=2.0)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        SOTA parameter initialization for newly introduced TCA components:
        - Multi-Head Attention: Truncated Normal (std=0.02)
        - Channel-wise Gate: Truncated Normal (std=0.02), bias=0 (50/50 starting balance)
        - Joint Projection: Truncated Normal (std=0.02)
        """
        for mha in (self.mha_v2a, self.mha_a2v):
            if hasattr(mha, "in_proj_weight") and mha.in_proj_weight is not None:
                nn.init.trunc_normal_(mha.in_proj_weight, std=0.02)
            if hasattr(mha, "in_proj_bias") and mha.in_proj_bias is not None:
                nn.init.constant_(mha.in_proj_bias, 0.0)
            if hasattr(mha, "out_proj") and hasattr(mha.out_proj, "weight"):
                nn.init.trunc_normal_(mha.out_proj.weight, std=0.02)
                if mha.out_proj.bias is not None:
                    nn.init.constant_(mha.out_proj.bias, 0.0)

        if hasattr(self.gate[0], "weight") and self.gate[0].weight is not None:
            nn.init.trunc_normal_(self.gate[0].weight, std=0.02)
        if hasattr(self.gate[0], "bias") and self.gate[0].bias is not None:
            nn.init.constant_(self.gate[0].bias, 0.0)

        for m in self.proj_joint:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        f_video: torch.Tensor,
        f_audio: torch.Tensor,
        tokens_video: Optional[torch.Tensor] = None,
        tokens_audio: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        B = f_video.size(0)

        # Ensure temporal tokens exist: [B, T=2, dim]
        if tokens_video is None or tokens_video.ndim < 3:
            tokens_video = f_video.unsqueeze(1).repeat(1, 2, 1)
        if tokens_audio is None or tokens_audio.ndim < 3:
            tokens_audio = f_audio.unsqueeze(1).repeat(1, 2, 1)

        # Step 1: Bidirectional Temporal Cross-Modal Attention
        # 1a. Video -> Audio: Visual frames query corresponding acoustic bursts
        attn_v2a, weights_v2a = self.mha_v2a(
            query=tokens_video,
            key=tokens_audio,
            value=tokens_audio
        )
        tokens_cross_v = self.norm_v(tokens_video + self.drop_v(attn_v2a))

        # 1b. Audio -> Video: Acoustic cues query corresponding visual kinematic frames
        attn_a2v, weights_a2v = self.mha_a2v(
            query=tokens_audio,
            key=tokens_video,
            value=tokens_video
        )
        tokens_cross_a = self.norm_a(tokens_audio + self.drop_a(attn_a2v))

        # Step 2: Temporal Pooling & Cross-Attended Motion Dynamics
        # 2a. Inter-frame motion dynamics after cross-attention
        if tokens_cross_v.size(1) >= 2:
            f_motion_cross = torch.abs(tokens_cross_v[:, 1] - tokens_cross_v[:, 0])
        else:
            f_motion_cross = tokens_cross_v[:, 0]

        # Combine backbone joint representation (spatial + motion + burst) with attended mean & motion
        f_v_cross = self.norm_v_out(f_video + tokens_cross_v.mean(dim=1) + f_motion_cross)

        # 2b. Acoustic rhythm dynamics after cross-attention
        if tokens_cross_a.size(1) >= 2:
            f_motion_a = torch.abs(tokens_cross_a[:, 1] - tokens_cross_a[:, 0])
        else:
            f_motion_a = tokens_cross_a[:, 0]

        f_a_cross = self.norm_a_out(f_audio + tokens_cross_a.mean(dim=1) + f_motion_a)

        # Step 3: Channel-wise Adaptive Reliability Gating
        combined = torch.cat([f_v_cross, f_a_cross], dim=-1)  # [B, dim * 2]
        g = self.gate(combined)                                # [B, dim]
        f_fused = self.norm_fused(g * f_v_cross + (1.0 - g) * f_a_cross)
        f_joint = self.proj_joint(f_fused)                     # [B, dim]

        # Step 4: Pairwise Boundary Tournament with Audio STFT Tie-Breaker on B23
        out = self.tournament_head(f_joint, f_audio=f_audio)

        # Step 5: Package metrics & attention representations
        out["f_fused"] = f_fused
        out["f_joint"] = f_joint
        out["gate"] = g
        # Modality weight vector [B, 2] computed as mean channel reliability
        g_mean = g.mean(dim=-1, keepdim=True)
        out["modality_weights"] = torch.cat([g_mean, 1.0 - g_mean], dim=-1)
        out["attn_weights_v2a"] = weights_v2a
        out["attn_weights_a2v"] = weights_a2v

        # Shannon entropy uncertainty
        entropy = -torch.sum(
            out["probabilities"] * torch.log(torch.clamp(out["probabilities"], min=1e-7)),
            dim=-1,
            keepdim=True
        )
        out["uncertainty"] = entropy / 1.386294

        return out


# Canonical aliases
MultimodalTournamentFusion = TemporalCrossModalAttentionFusion
MultimodalFusion = TemporalCrossModalAttentionFusion
