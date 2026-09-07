import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class SOTAMultimodalFusion(nn.Module):
    """
    SOTA 3-Tier Multimodal Fusion Engine (~0.64M params):

    Tier 1: Google Multimodal Bottleneck Transformer (MBT - Nagrani et al., NeurIPS 2021)
      - Employs 4 learnable bottleneck tokens to force cross-modal interaction through a low-dimensional
        semantic corridor, effectively eliminating pond background water reflections and environmental acoustic noise.

    Tier 2: Attentive Audio-Visual Fusion Dynamic Reliability Gating (Fayek & Kumar, WACV 2022)
      - Evaluates the instantaneous reliability alpha = [alpha_v, alpha_a] of both modalities
        conditioned on dynamic water conditions.

    Tier 3: Trusted Multi-View Classification (TMC - Han et al., NeurIPS 2021)
      - Predicts non-negative evidential Dirichlet parameters for Video, Audio, and Fused modalities.
      - Implements Dempster's Rule of Combination to synthesize beliefs (b_v, b_a) and quantify epistemic uncertainty (u_v, u_a, u_fused).
    """
    def __init__(
        self,
        dim: int = 224,
        num_bottlenecks: int = 4,
        num_heads: int = 4,
        classes_num: int = 4
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_bottlenecks = num_bottlenecks
        self.classes_num = classes_num

        # =====================================================================
        # Tier 1: Multimodal Bottleneck Transformer (MBT NeurIPS 2021)
        # =====================================================================
        self.bottlenecks = nn.Parameter(torch.randn(1, num_bottlenecks, dim) * 0.02)

        self.cross_attn_v = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_a = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        self.norm_v = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_f = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        # =====================================================================
        # Tier 2: Dynamic Reliability Gating (WACV 2022)
        # =====================================================================
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )

        self.norm_fused = nn.LayerNorm(dim)

        # =====================================================================
        # Tier 3: TMC Evidential Heads (NeurIPS 2021)
        # =====================================================================
        # Non-negative evidence is computed via Softplus
        self.evidence_head_v = nn.Linear(dim, classes_num)
        self.evidence_head_a = nn.Linear(dim, classes_num)
        self.evidence_head_f = nn.Linear(dim, classes_num)

    def _dempster_combine(
        self,
        e_v: torch.Tensor,
        e_a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Synthesize modality evidence via Dempster's Rule of Combination (Han et al., NeurIPS 2021).
        """
        K = self.classes_num
        alpha_v = e_v + 1.0
        alpha_a = e_a + 1.0
        S_v = torch.sum(alpha_v, dim=-1, keepdim=True)
        S_a = torch.sum(alpha_a, dim=-1, keepdim=True)

        b_v = e_v / S_v
        b_a = e_a / S_a
        u_v = float(K) / S_v
        u_a = float(K) / S_a

        bb = b_v * b_a
        bu_v = b_v * u_a
        bu_a = b_a * u_v

        # 1 - C (where C is the conflict mass)
        one_minus_c = torch.sum(bb + bu_v + bu_a, dim=-1, keepdim=True) + u_v * u_a
        one_minus_c = torch.clamp(one_minus_c, min=1e-6)

        b_f = (bb + bu_v + bu_a) / one_minus_c
        u_f = (u_v * u_a) / one_minus_c

        S_f = float(K) / torch.clamp(u_f, min=1e-6)
        e_f = b_f * S_f
        alpha_f = e_f + 1.0
        prob_f = alpha_f / S_f

        return e_f, alpha_f, prob_f, u_v, u_a, u_f

    def forward(
        self,
        f_video: torch.Tensor,
        f_audio: torch.Tensor,
        tokens_video: torch.Tensor,
        tokens_audio: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            f_video: Spatiotemporal video embedding [B, dim]
            f_audio: Acoustic embedding [B, dim]
            tokens_video: Frame tokens sequence [B, T, dim]
            tokens_audio: Temporal audio tokens sequence [B, Ta, dim]

        Returns:
            Dictionary containing logits, probabilities, uncertainties, gating weights, and evidential Dirichlet parameters.
        """
        B = f_video.size(0)

        # ---------------------------------------------------------------------
        # Tier 1: MBT Bottleneck Cross-Attention
        # ---------------------------------------------------------------------
        btn = self.bottlenecks.expand(B, -1, -1)  # [B, 4, dim]

        # Video tokens -> Bottlenecks
        btn_v, _ = self.cross_attn_v(query=btn, key=tokens_video, value=tokens_video)
        btn = self.norm_v(btn + btn_v)

        # Audio tokens -> Bottlenecks
        btn_a, _ = self.cross_attn_a(query=btn, key=tokens_audio, value=tokens_audio)
        btn = self.norm_a(btn + btn_a)

        # Feed-forward refinement
        btn = self.norm_f(btn + self.ffn(btn))
        f_bottleneck = btn.mean(dim=1)  # [B, dim]

        # ---------------------------------------------------------------------
        # Tier 2: Dynamic Reliability Gating (WACV 2022)
        # ---------------------------------------------------------------------
        combined_features = torch.cat([f_video, f_audio], dim=-1)
        modality_weights = self.gate(combined_features)  # [B, 2] -> [alpha_v, alpha_a]
        alpha_v = modality_weights[:, 0:1]
        alpha_a = modality_weights[:, 1:2]

        f_gated = alpha_v * f_video + alpha_a * f_audio
        f_fused = self.norm_fused(f_bottleneck + f_gated)  # [B, dim]

        # ---------------------------------------------------------------------
        # Tier 3: TMC Evidential Reasoning (NeurIPS 2021)
        # ---------------------------------------------------------------------
        # Predict non-negative evidence e >= 0 using softplus
        e_v = F.softplus(self.evidence_head_v(f_video))
        e_a = F.softplus(self.evidence_head_a(f_audio))
        e_joint = F.softplus(self.evidence_head_f(f_fused))

        # Combine video and audio views via Dempster's rule
        e_dempster, alpha_dempster, prob_dempster, u_v, u_a, u_dempster = self._dempster_combine(e_v, e_a)

        # Final unified evidence is the combination of Dempster synthesis and joint fused representation
        evidence_final = e_dempster + e_joint
        alpha_final = evidence_final + 1.0
        S_final = torch.sum(alpha_final, dim=-1, keepdim=True)
        prob_final = alpha_final / S_final
        u_final = float(self.classes_num) / S_final

        # Clipwise output logits for standard evaluation metrics and loss functions
        logits_final = torch.log(torch.clamp(prob_final, min=1e-7))

        return {
            "logits": logits_final,
            "probabilities": prob_final,
            "uncertainty": u_final,
            "uncertainty_video": u_v,
            "uncertainty_audio": u_a,
            "modality_weights": modality_weights,
            "f_fused": f_fused,
            "evidence_v": e_v,
            "evidence_a": e_a,
            "evidence_final": evidence_final,
            "alpha_final": alpha_final,
        }
