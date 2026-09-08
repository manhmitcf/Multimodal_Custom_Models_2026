import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CumulativeOrdinalBoundaryHead(nn.Module):
    """
    Cumulative Ordinal Decision Head (CORAL / Monotonic Boundary Head).
    Fish feeding intensity ordinal continuum:
      Rank 0: None
      Rank 1: Weak
      Rank 2: Medium
      Rank 3: Strong

    Projects feature representation into a 1D scalar score:
      s = w^T * f in R
    and learns 3 strictly monotonic cutoffs:
      b_1 < b_2 < b_3
    enforced via:
      b_1 = theta_1
      b_2 = b_1 + softplus(Delta_theta_2) + 0.5
      b_3 = b_2 + softplus(Delta_theta_3) + 0.5

    Cumulative exceedance probabilities:
      P(Rank >= k) = sigma(s - b_k)
    """
    def __init__(self, dim: int = 224, init_theta1: float = -1.0) -> None:
        super().__init__()
        self.score_proj = nn.Linear(dim, 1)

        # Monotonic cutoff parameters
        self.theta_1 = nn.Parameter(torch.tensor(init_theta1, dtype=torch.float32))
        self.delta_theta_2 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.delta_theta_3 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # Dataset raw indexing map:
        # Raw 0: None   (Rank 0)
        # Raw 1: Strong (Rank 3)
        # Raw 2: Medium (Rank 2)
        # Raw 3: Weak   (Rank 1)
        self.register_buffer(
            "rank_to_raw",
            torch.tensor([0, 3, 2, 1], dtype=torch.long)
        )

    def get_cutoffs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b_1 = self.theta_1
        b_2 = b_1 + F.softplus(self.delta_theta_2) + 0.5
        b_3 = b_2 + F.softplus(self.delta_theta_3) + 0.5
        return b_1, b_2, b_3

    def set_cutoffs(self, cutoffs: Tuple[float, float, float]) -> None:
        """Utility for Nelder-Mead post-calibration."""
        with torch.no_grad():
            b1, b2, b3 = cutoffs
            self.theta_1.copy_(torch.tensor(b1, dtype=torch.float32))
            # Inverse softplus for deltas
            d2 = max(b2 - b1 - 0.5, 1e-4)
            d3 = max(b3 - b2 - 0.5, 1e-4)
            inv_sp2 = math_inv_softplus(d2)
            inv_sp3 = math_inv_softplus(d3)
            self.delta_theta_2.copy_(torch.tensor(inv_sp2, dtype=torch.float32))
            self.delta_theta_3.copy_(torch.tensor(inv_sp3, dtype=torch.float32))

    def forward(self, f: torch.Tensor) -> Dict[str, torch.Tensor]:
        s = self.score_proj(f).squeeze(-1)  # [B]
        b_1, b_2, b_3 = self.get_cutoffs()

        # Cumulative exceedance probabilities
        p_ge_1 = torch.sigmoid(s - b_1)
        p_ge_2 = torch.sigmoid(s - b_2)
        p_ge_3 = torch.sigmoid(s - b_3)

        # Telescoping individual rank probabilities
        p_rank_0 = torch.clamp(1.0 - p_ge_1, min=1e-6)
        p_rank_1 = torch.clamp(p_ge_1 - p_ge_2, min=1e-6)
        p_rank_2 = torch.clamp(p_ge_2 - p_ge_3, min=1e-6)
        p_rank_3 = torch.clamp(p_ge_3, min=1e-6)

        p_rank = torch.stack([p_rank_0, p_rank_1, p_rank_2, p_rank_3], dim=-1)
        p_rank = p_rank / torch.sum(p_rank, dim=-1, keepdim=True)

        # Map to raw dataset class indexing: [None, Strong, Medium, Weak]
        p_raw = torch.stack([p_rank[:, 0], p_rank[:, 3], p_rank[:, 2], p_rank[:, 1]], dim=-1)
        logits_raw = torch.log(torch.clamp(p_raw, min=1e-7))
        expected_intensity = p_ge_1 + p_ge_2 + p_ge_3

        cutoffs = torch.stack([b_1, b_2, b_3])

        return {
            "score": s,
            "cutoffs": cutoffs,
            "p_raw": p_raw,
            "logits": logits_raw,
            "expected_intensity": expected_intensity,
            "cum_probs": torch.stack([p_ge_1, p_ge_2, p_ge_3], dim=-1)
        }


def math_inv_softplus(x: float) -> float:
    import math
    if x > 20.0:
        return x
    return math.log(max(math.exp(x) - 1.0, 1e-6))


class GatedBilateralBoundaryFusion(nn.Module):
    """
    Streamlined Gated Bilateral Boundary Fusion Engine (~50K params).
    1. Gated Fusion: Learns cross-modal reliability weight g = sigma(W[f_V || f_A]) (~0.5K params).
    2. Bilateral Boundary Heads:
       - Video CORAL Head: Evaluates video scalar score s_V with cutoffs b_V1 < b_V2 < b_V3.
       - Audio CORAL Head: Evaluates audio scalar score s_A with cutoffs b_A1 < b_A2 < b_A3.
    3. Blended Decision: Combines s_final = g * s_V + (1-g) * s_A and b_final = g * b_V + (1-g) * b_A.
    """
    def __init__(self, dim: int = 224, dropout: float = 0.1, **kwargs) -> None:
        super().__init__()
        self.dim = dim

        # 1. Gated Cross-Modal Reliability Gating: [B, dim * 2] -> [B, 1]
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 1),
            nn.Sigmoid()
        )
        self.norm_fused = nn.LayerNorm(dim)

        # Branch refinement projections
        self.proj_v = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dim)
        )
        self.proj_a = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dim)
        )

        # 2. Bilateral CORAL Heads
        self.head_v = CumulativeOrdinalBoundaryHead(dim=dim, init_theta1=-1.0)
        self.head_a = CumulativeOrdinalBoundaryHead(dim=dim, init_theta1=-1.0)

    def forward(
        self,
        f_video: torch.Tensor,
        f_audio: torch.Tensor,
        tokens_video: Optional[torch.Tensor] = None,
        tokens_audio: Optional[torch.Tensor] = None,
        f_burst_v: Optional[torch.Tensor] = None,
        f_burst_a: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            f_video: Spatiotemporal video embedding [B, dim]
            f_audio: Acoustic embedding [B, dim]

        Returns:
            Dictionary with final blended logits, probabilities, cutoffs,
            unimodal scores, and gate weights.
        """
        # Step 1: Gated Cross-Modal Fusion
        combined = torch.cat([f_video, f_audio], dim=-1)  # [B, dim * 2]
        g = self.gate(combined)                            # [B, 1] (Weight for Video)
        f_fused = self.norm_fused(g * f_video + (1.0 - g) * f_audio)  # [B, dim]

        # Step 2: Branch features with residual connection
        f_branch_v = self.proj_v(f_fused + f_video)  # [B, dim]
        f_branch_a = self.proj_a(f_fused + f_audio)  # [B, dim]

        # Step 3: Bilateral Head Evaluations
        out_v = self.head_v(f_branch_v)
        out_a = self.head_a(f_branch_a)

        s_v = out_v["score"]       # [B]
        s_a = out_a["score"]       # [B]
        b_v = out_v["cutoffs"]     # [3]
        b_a = out_a["cutoffs"]     # [3]

        # Step 4: Blended Final Decision
        g_scalar = g.squeeze(-1)   # [B]
        s_final = g_scalar * s_v + (1.0 - g_scalar) * s_a  # [B]

        # Weighted blended cutoffs per sample or mean
        g_mean = g.mean()
        b_final = g_mean * b_v + (1.0 - g_mean) * b_a  # [3]

        # Final blended cumulative probabilities
        p_ge_1 = torch.sigmoid(s_final - b_final[0])
        p_ge_2 = torch.sigmoid(s_final - b_final[1])
        p_ge_3 = torch.sigmoid(s_final - b_final[2])

        p_0 = torch.clamp(1.0 - p_ge_1, min=1e-6)
        p_1 = torch.clamp(p_ge_1 - p_ge_2, min=1e-6)
        p_2 = torch.clamp(p_ge_2 - p_ge_3, min=1e-6)
        p_3 = torch.clamp(p_ge_3, min=1e-6)

        p_rank = torch.stack([p_0, p_1, p_2, p_3], dim=-1)
        p_rank = p_rank / torch.sum(p_rank, dim=-1, keepdim=True)

        # Map to raw dataset class indexing: [None, Strong, Medium, Weak]
        p_raw = torch.stack([p_rank[:, 0], p_rank[:, 3], p_rank[:, 2], p_rank[:, 1]], dim=-1)
        logits_raw = torch.log(torch.clamp(p_raw, min=1e-7))
        expected_intensity = p_ge_1 + p_ge_2 + p_ge_3

        # Normalized Shannon entropy uncertainty
        entropy = -torch.sum(p_raw * torch.log(torch.clamp(p_raw, min=1e-7)), dim=-1, keepdim=True)
        norm_entropy = entropy / 1.386294  # log(4)

        modality_weights = torch.cat([g, 1.0 - g], dim=-1)  # [B, 2]

        return {
            "logits": logits_raw,
            "probabilities": p_raw,
            "intensity_score": s_final.unsqueeze(-1),
            "expected_intensity": expected_intensity.unsqueeze(-1),
            "uncertainty": norm_entropy,
            "modality_weights": modality_weights,
            "f_fused": f_fused,
            "gate": g,
            # Bilateral outputs for loss calculation
            "score_v": s_v,
            "score_a": s_a,
            "cutoffs_v": b_v,
            "cutoffs_a": b_a,
            "b_final": b_final,
            "cum_probs": torch.stack([p_ge_1, p_ge_2, p_ge_3], dim=-1),
            "cum_probs_v": out_v["cum_probs"],
            "cum_probs_a": out_a["cum_probs"],
            "probabilities_v": out_v["p_raw"],
            "probabilities_a": out_a["p_raw"],
        }


# Backward compatibility aliases
MultimodalBoundaryAwareFusion = GatedBilateralBoundaryFusion
SOTAMultimodalFusion = GatedBilateralBoundaryFusion
