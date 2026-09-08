import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional


class BoundaryDiscrepancyGate(nn.Module):
    """
    Boundary Discrepancy Gate (BDG).
    Detects cross-modal asynchrony between visual motion kinematics and acoustic splashing:
      Delta_{trans} = |f'_V - f'_A|
    At temporal class boundaries (Strong <-> Medium <-> Weak <-> None), audio splashing
    stops before fish swimming inertia decays. BDG calculates discrepancy and assigns
    activation mass to either:
      - Stable regime (within-class canonical state)
      - Boundary transition 0<->1 (None <-> Weak)
      - Boundary transition 1<->2 (Weak <-> Medium)
      - Boundary transition 2<->3 (Medium <-> Strong)
    """
    def __init__(self, dim: int = 224, hidden_dim: int = 64) -> None:
        super().__init__()
        # Input: Joint representation f_joint (dim) + Discrepancy delta_trans (dim) = dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4),  # [stable, bnd_01, bnd_12, bnd_23]
            nn.Softmax(dim=-1)
        )

    def forward(self, f_joint: torch.Tensor, delta_trans: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([f_joint, delta_trans], dim=-1)  # [B, dim * 2]
        regime_weights = self.mlp(combined)  # [B, 4]
        return regime_weights


class BoundaryAwareChannelRouting(nn.Module):
    """
    Boundary-Aware Channel Routing & Attention (BACA).
    Partitions the 224-dimensional latent channel space into:
      - 128 Core Channels: Representing stable canonical feeding regimes.
      - 96 Boundary Channels: Divided into 3 dedicated 32-channel subspaces:
          * Channels 128-159: None <-> Weak transition features
          * Channels 160-191: Weak <-> Medium transition features
          * Channels 192-223: Medium <-> Strong transition features

    The boundary subspaces are dynamically modulated by the Boundary Discrepancy Gate
    weights, enabling the network to specialize channel filters specifically for
    disambiguating ambiguous adjacent boundary samples without corrupting core state features.
    """
    def __init__(self, in_dim: int = 224, core_dim: int = 128, bnd_sub_dim: int = 32) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.core_dim = core_dim          # 128
        self.bnd_sub_dim = bnd_sub_dim    # 32 (32 * 3 = 96)
        self.total_bnd_dim = bnd_sub_dim * 3

        self.core_proj = nn.Sequential(
            nn.Linear(in_dim, core_dim),
            nn.LayerNorm(core_dim),
            nn.GELU()
        )

        self.bnd_01_proj = nn.Sequential(
            nn.Linear(in_dim, bnd_sub_dim),
            nn.LayerNorm(bnd_sub_dim),
            nn.GELU()
        )
        self.bnd_12_proj = nn.Sequential(
            nn.Linear(in_dim, bnd_sub_dim),
            nn.LayerNorm(bnd_sub_dim),
            nn.GELU()
        )
        self.bnd_23_proj = nn.Sequential(
            nn.Linear(in_dim, bnd_sub_dim),
            nn.LayerNorm(bnd_sub_dim),
            nn.GELU()
        )

        self.out_norm = nn.LayerNorm(core_dim + self.total_bnd_dim)  # 224

    def forward(
        self,
        f_joint: torch.Tensor,
        delta_trans: torch.Tensor,
        regime_weights: torch.Tensor,
        delta_burst: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            f_joint: Joint multimodal representation [B, 224]
            delta_trans: Cross-modal discrepancy [B, 224]
            regime_weights: Softmax regime weights [B, 4]
            delta_burst: Optional Peak-to-Average Dynamic Contrast [B, 224]
        Returns:
            f_routed: Channel-routed representation [B, 224]
        """
        w_stable = regime_weights[:, 0:1]  # [B, 1]
        w_bnd_01 = regime_weights[:, 1:2]  # [B, 1]
        w_bnd_12 = regime_weights[:, 2:3]  # [B, 1]
        w_bnd_23 = regime_weights[:, 3:4]  # [B, 1]

        # 1. Core Channels (128 ch)
        f_core = self.core_proj(f_joint) * (1.0 + w_stable)

        # 2. Boundary Channels (32 ch each)
        # Conditioned on joint representation perturbed by discrepancy signal
        f_bnd_input = f_joint + delta_trans
        f_bnd_01 = self.bnd_01_proj(f_bnd_input) * (1.0 + w_bnd_01)
        f_bnd_12 = self.bnd_12_proj(f_bnd_input) * (1.0 + w_bnd_12)

        # Subspace 2 (Medium <-> Strong): Modulated by Dynamic Burst Contrast (f_peak - f_mean)
        if delta_burst is not None:
            f_bnd_23_input = f_bnd_input + delta_burst
            burst_scale = 1.0 + torch.tanh(delta_burst.mean(dim=-1, keepdim=True))
            f_bnd_23 = self.bnd_23_proj(f_bnd_23_input) * (1.0 + w_bnd_23) * burst_scale
        else:
            f_bnd_23 = self.bnd_23_proj(f_bnd_input) * (1.0 + w_bnd_23)

        # 3. Channel Concatenation: 128 + 32 + 32 + 32 = 224 channels
        f_routed = torch.cat([f_core, f_bnd_01, f_bnd_12, f_bnd_23], dim=-1)
        f_routed = self.out_norm(f_routed)
        return f_routed



class CumulativeOrdinalBoundaryHead(nn.Module):
    """
    Cumulative Ordinal Decision Head (CORAL / Monotonic Boundary Head).
    Fish feeding is an ordinal continuum:
      Rank 0: None
      Rank 1: Weak
      Rank 2: Medium
      Rank 3: Strong

    Instead of predicting 4 unconstrained logits that clash 50/50 at boundaries,
    the model projects the fused representation into a single 1D scalar score:
      s = w^T * f_fused + c in R
    and learns 3 strictly monotonic cutoffs:
      b_1 < b_2 < b_3
    enforced via:
      b_1 = theta_1
      b_2 = b_1 + softplus(Delta_theta_2) + 0.05
      b_3 = b_2 + softplus(Delta_theta_3) + 0.05

    Cumulative probabilities:
      P(Rank >= 1) = sigma(s - b_1)
      P(Rank >= 2) = sigma(s - b_2)
      P(Rank >= 3) = sigma(s - b_3)

    This mathematically eliminates 50/50 boundary flips and class rank inversions.
    Outputs are mapped to raw dataset index order (0: None, 1: Strong, 2: Medium, 3: Weak).
    """
    def __init__(self, dim: int = 224) -> None:
        super().__init__()
        self.score_proj = nn.Linear(dim, 1)

        # Monotonic cutoff parameters
        # Initialized to spread cutoffs symmetrically around 0
        self.theta_1 = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.delta_theta_2 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.delta_theta_3 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # Mapping: Physical Ordinal Rank -> Raw Dataset Class Index
        # Rank 0 (None)   -> Raw 0
        # Rank 1 (Weak)   -> Raw 3
        # Rank 2 (Medium) -> Raw 2
        # Rank 3 (Strong) -> Raw 1
        # Inverse: Raw 0 -> Rank 0, Raw 1 -> Rank 3, Raw 2 -> Rank 2, Raw 3 -> Rank 1
        self.register_buffer(
            "rank_to_raw",
            torch.tensor([0, 3, 2, 1], dtype=torch.long)
        )

    def get_cutoffs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b_1 = self.theta_1
        b_2 = b_1 + F.softplus(self.delta_theta_2) + 0.05
        b_3 = b_2 + F.softplus(self.delta_theta_3) + 0.05
        return b_1, b_2, b_3

    def forward(self, f_fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. Scalar continuous feeding intensity score
        s = self.score_proj(f_fused).squeeze(-1)  # [B]

        # 2. Strictly monotonic cutoffs: b_1 < b_2 < b_3
        b_1, b_2, b_3 = self.get_cutoffs()

        # 3. Cumulative exceedance probabilities P(Rank >= k)
        p_ge_1 = torch.sigmoid(s - b_1)  # P(Rank >= 1)
        p_ge_2 = torch.sigmoid(s - b_2)  # P(Rank >= 2)
        p_ge_3 = torch.sigmoid(s - b_3)  # P(Rank >= 3)

        # 4. Telescoping individual rank probabilities
        p_rank_0 = torch.clamp(1.0 - p_ge_1, min=1e-6)
        p_rank_1 = torch.clamp(p_ge_1 - p_ge_2, min=1e-6)
        p_rank_2 = torch.clamp(p_ge_2 - p_ge_3, min=1e-6)
        p_rank_3 = torch.clamp(p_ge_3, min=1e-6)

        p_rank = torch.stack([p_rank_0, p_rank_1, p_rank_2, p_rank_3], dim=-1)  # [B, 4]
        # Normalize to ensure exact sum to 1
        p_rank = p_rank / torch.sum(p_rank, dim=-1, keepdim=True)

        # 5. Map to raw dataset class indexing:
        # Raw 0: None   (Rank 0)
        # Raw 1: Strong (Rank 3)
        # Raw 2: Medium (Rank 2)
        # Raw 3: Weak   (Rank 1)
        p_raw_0 = p_rank[:, 0]  # None
        p_raw_1 = p_rank[:, 3]  # Strong
        p_raw_2 = p_rank[:, 2]  # Medium
        p_raw_3 = p_rank[:, 1]  # Weak
        p_raw = torch.stack([p_raw_0, p_raw_1, p_raw_2, p_raw_3], dim=-1)  # [B, 4]

        # Logits compatible with CrossEntropy / downstream metrics
        logits_raw = torch.log(torch.clamp(p_raw, min=1e-7))

        # Continuous expected physical feeding intensity [0.0, 3.0]
        # (0.0 = completely satiated / None, 3.0 = maximum feeding frenzy / Strong)
        expected_intensity = p_ge_1 + p_ge_2 + p_ge_3  # [B]

        # Epistemic boundary uncertainty: Normalized Shannon entropy
        # Peaks near cutoffs when the model encounters transition ambiguity
        entropy = -torch.sum(p_raw * torch.log(torch.clamp(p_raw, min=1e-7)), dim=-1, keepdim=True)
        norm_entropy = entropy / 1.386294  # log(4) = 1.386294

        cutoffs = torch.stack([b_1, b_2, b_3])

        return {
            "logits": logits_raw,
            "probabilities": p_raw,
            "intensity_score": s.unsqueeze(-1),
            "expected_intensity": expected_intensity.unsqueeze(-1),
            "uncertainty": norm_entropy,
            "cutoffs": cutoffs,
            "cum_probs": torch.stack([p_ge_1, p_ge_2, p_ge_3], dim=-1)
        }


class MultimodalBoundaryAwareFusion(nn.Module):
    """
    Streamlined Multimodal Boundary-Aware Fusion Engine (~0.54M params):
    Replaces bloated multi-bottlenecks and Dempster's rule with a focused, high-precision
    temporal transition architecture:

      1. Bidirectional Cross-Attention (Bi-CA):
         Full exchange between Video Tokens [B, 4, 224] and Audio Tokens [B, 4, 224].
      2. Boundary Discrepancy Gate (BDG):
         Monitors cross-modal asynchrony Delta_{trans} = |f'_V - f'_A| to detect when
         feeding sounds terminate while water turbulence persists.
      3. Boundary-Aware Channel Routing & Attention (BACA):
         Partitions 224 channels into 128 Core + 96 Boundary (32 None/Weak, 32 Weak/Medium, 32 Medium/Strong).
      4. Cumulative Ordinal Boundary Head:
         Projects to 1D scalar intensity with strictly monotonic cutoffs b_1 < b_2 < b_3.
    """
    def __init__(
        self,
        dim: int = 224,
        num_heads: int = 4,
        classes_num: int = 4,
        core_dim: int = 128,
        bnd_sub_dim: int = 32,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        **kwargs  # Accept legacy kwargs like num_bottlenecks for backwards compatibility
    ) -> None:
        super().__init__()
        self.dim = dim
        self.classes_num = classes_num

        # 1. Bidirectional Cross-Attention
        self.cross_attn_v2a = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_a2v = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm_v = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)

        # Cross-modal feed-forward refinement (streamlined bottleneck ffn_dim=128)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim)
        )
        self.norm_fused = nn.LayerNorm(dim)

        # 2. Boundary Discrepancy Gate (BDG) with hidden_dim=64
        self.bdg = BoundaryDiscrepancyGate(dim=dim, hidden_dim=64)

        # 3. Boundary-Aware Channel Routing (BACA)
        self.baca = BoundaryAwareChannelRouting(in_dim=dim, core_dim=core_dim, bnd_sub_dim=bnd_sub_dim)

        # 4. Cumulative Ordinal Decision Head
        self.ordinal_head = CumulativeOrdinalBoundaryHead(dim=dim)

        # Modality reliability weights generator for logging (streamlined hidden_dim=32)
        self.modality_gate = nn.Sequential(
            nn.Linear(dim * 2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )

    def forward(
        self,
        f_video: torch.Tensor,
        f_audio: torch.Tensor,
        tokens_video: torch.Tensor,
        tokens_audio: torch.Tensor,
        f_burst_v: Optional[torch.Tensor] = None,
        f_burst_a: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            f_video: Spatiotemporal video embedding [B, dim]
            f_audio: Acoustic embedding [B, dim]
            tokens_video: Frame tokens sequence [B, T, dim]
            tokens_audio: Temporal audio tokens sequence [B, Ta, dim]
            f_burst_v: Optional Peak-to-Average video dynamic contrast [B, dim]
            f_burst_a: Optional Peak-to-Average audio dynamic contrast [B, dim]

        Returns:
            Dictionary containing logits, probabilities, uncertainty, modality_weights,
            intensity_score, expected_intensity, and routed features.
        """
        # 1. Bidirectional Cross-Attention
        # Video attends to Audio: What sounds correspond to this visual frame?
        tokens_v2a, _ = self.cross_attn_v2a(query=tokens_video, key=tokens_audio, value=tokens_audio)
        tokens_v_refined = self.norm_v(tokens_video + tokens_v2a)

        # Audio attends to Video: What visual movements correspond to this audio burst?
        tokens_a2v, _ = self.cross_attn_a2v(query=tokens_audio, key=tokens_video, value=tokens_video)
        tokens_a_refined = self.norm_a(tokens_audio + tokens_a2v)

        # Temporal pooling to vector + unimodal feature integration
        f_v_prime = tokens_v_refined.mean(dim=1) + f_video  # [B, dim]
        f_a_prime = tokens_a_refined.mean(dim=1) + f_audio  # [B, dim]

        # Dynamic Modality Reliability Gating
        modality_weights = self.modality_gate(torch.cat([f_v_prime, f_a_prime], dim=-1))  # [B, 2]
        alpha_v = modality_weights[:, 0:1]
        alpha_a = modality_weights[:, 1:2]

        # Feed-forward joint refinement weighted by modality reliability
        f_joint_raw = alpha_v * f_v_prime + alpha_a * f_a_prime
        f_joint = self.norm_fused(f_joint_raw + self.ffn(f_joint_raw))  # [B, dim]

        # Synthesize Peak-to-Average Dynamic Contrast (Burst feeding strike signal)
        if f_burst_v is not None and f_burst_a is not None:
            delta_burst = 0.5 * (f_burst_v + f_burst_a)
        elif f_burst_v is not None:
            delta_burst = f_burst_v
        elif f_burst_a is not None:
            delta_burst = f_burst_a
        else:
            delta_burst = None

        # 2. Boundary Discrepancy Gate (BDG)
        delta_trans = torch.abs(f_v_prime - f_a_prime)
        regime_weights = self.bdg(f_joint, delta_trans)

        # 3. Boundary-Aware Channel Routing (BACA)
        # Partitions into 128 Core + 96 Boundary with Dynamic Burst Contrast modulation
        f_fused = self.baca(f_joint, delta_trans, regime_weights, delta_burst=delta_burst)  # [B, 224]

        # 4. Cumulative Ordinal Boundary Head
        head_outputs = self.ordinal_head(f_fused)

        # Unimodal uncertainty approximations for logging
        u_v = 0.5 * regime_weights[:, 0:1] + 0.5 * (1.0 - alpha_v)
        u_a = 0.5 * regime_weights[:, 0:1] + 0.5 * (1.0 - alpha_a)

        return {
            "logits": head_outputs["logits"],
            "probabilities": head_outputs["probabilities"],
            "uncertainty": head_outputs["uncertainty"],
            "uncertainty_video": u_v,
            "uncertainty_audio": u_a,
            "modality_weights": modality_weights,
            "boundary_weights": regime_weights,
            "intensity_score": head_outputs["intensity_score"],
            "expected_intensity": head_outputs["expected_intensity"],
            "cutoffs": head_outputs["cutoffs"],
            "cum_probs": head_outputs["cum_probs"],
            "f_fused": f_fused,
            "delta_burst": delta_burst if delta_burst is not None else torch.zeros_like(f_fused),
        }




# Backwards compatibility alias
SOTAMultimodalFusion = MultimodalBoundaryAwareFusion

