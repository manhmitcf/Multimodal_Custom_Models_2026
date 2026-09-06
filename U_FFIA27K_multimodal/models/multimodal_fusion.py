import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class EnhancedFishMultimodalFusion(nn.Module):
    """
    Enhanced Physics-Informed Bi-Directional Multimodal Fusion for Fish Feeding Intensity Assessment:
    
    1. Bi-directional Cross-Attention:
       - Video-to-Audio: Visual motion queries acoustic impulse rhythm.
       - Audio-to-Video: Acoustic splash queries video visual motion to eliminate aeration pump false positives.
    2. Spectral-Spatial FiLM Modulation:
       - Acoustic frequency profile (2-8kHz) modulates static visual foam & pond texture.
    3. Phase-Lag Synchronization Score:
       - Computes peak correlation between visual water upheaval and underwater acoustic shockwave.
    4. Physics-Informed Reliability Gating:
       - Uses actual physical metrics (white foam ratio, foam expansion rate dA/dt, fish aggregation density)
         to dynamically weigh visual vs. acoustic confidence.
    """
    def __init__(self, feat_dim: int = 128, num_classes: int = 4) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        # 1. Bi-directional Cross-Attention
        self.cross_attn_a2v = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=4, batch_first=True)
        self.cross_attn_v2a = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=4, batch_first=True)
        self.norm_v = nn.LayerNorm(feat_dim)
        self.norm_a = nn.LayerNorm(feat_dim)

        # 2. FiLM Modulation (Frequency controls Foam Sensitivity)
        self.film_generator = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.SiLU()
        )

        # 3. Physics-Informed Reliability Gate
        # Input: [f_v, f_a, external_physics_feats (3 dimensions)]
        self.physics_gate = nn.Sequential(
            nn.Linear(feat_dim * 2 + 3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )

        # 4. Final Multimodal Classifier Head
        fused_dim = feat_dim * 2 + feat_dim + 1 + 3 # Dynamic(256) + Static(128) + Sync(1) + External(3)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(
        self,
        f_v_seq: torch.Tensor,
        f_a_seq: torch.Tensor,
        f_v_spa: torch.Tensor,
        f_v_mot: Optional[torch.Tensor] = None,
        f_a_frq: Optional[torch.Tensor] = None,
        f_a_rhy: Optional[torch.Tensor] = None,
        f_audio: Optional[torch.Tensor] = None,
        external_physics_feats: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            f_v_seq: Video frame sequence representation [B, T_v, feat_dim]
            f_a_seq: Audio temporal rhythm sequence representation [B, T_a, feat_dim]
            f_v_spa: Video spatial representation [B, feat_dim]
            f_v_mot: Video motion representation [B, feat_dim]
            f_a_frq: Audio frequency profile representation [B, feat_dim]
            f_a_rhy: Audio rhythm representation [B, feat_dim]
            f_audio: Overall acoustic representation [B, feat_dim]
            external_physics_feats: [B, 3] (white_foam_area, foam_rate, fish_density)
        """
        B = f_v_seq.size(0)
        device = f_v_seq.device

        if external_physics_feats is None or external_physics_feats.numel() == 0:
            external_physics_feats = torch.zeros(B, 3, device=device, dtype=f_v_seq.dtype)

        # 1. Bi-directional Cross-Attention
        # 1.1 Audio queries Video
        attn_v, _ = self.cross_attn_a2v(query=f_a_seq, key=f_v_seq, value=f_v_seq)
        f_a_enhanced = self.norm_a(f_a_seq + attn_v).mean(dim=1)
        if f_a_rhy is not None:
            f_a_enhanced = f_a_enhanced + f_a_rhy
        if f_audio is not None:
            f_a_enhanced = f_a_enhanced + f_audio

        # 1.2 Video queries Audio
        attn_a, attn_weights_v2a = self.cross_attn_v2a(query=f_v_seq, key=f_a_seq, value=f_a_seq)
        f_v_enhanced = self.norm_v(f_v_seq + attn_a).mean(dim=1)
        if f_v_mot is not None:
            f_v_enhanced = f_v_enhanced + f_v_mot

        # 1.3 Synchronization Score (Max alignment score across attention map)
        sync_score = attn_weights_v2a.max(dim=-1)[0].mean(dim=-1, keepdim=True) # [B, 1]

        # 2. FiLM Modulation (Spectral -> Spatial)
        film_params = self.film_generator(f_a_frq)
        gamma, beta = film_params.chunk(2, dim=-1)
        f_static = gamma * f_v_spa + beta # [B, feat_dim]

        # 3. Physics-Informed Gating
        gate_input = torch.cat([f_v_enhanced, f_a_enhanced, external_physics_feats], dim=-1)
        weights = self.physics_gate(gate_input) # [B, 2] -> [w_video, w_audio]
        w_v = weights[:, :1]
        w_a = weights[:, 1:]

        # Apply confidence-weighted dynamic features
        f_dynamic_gated = torch.cat([w_v * f_v_enhanced, w_a * f_a_enhanced], dim=-1) # [B, 2 * feat_dim]

        # 4. Joint Fusion Representation & Classification
        f_final = torch.cat([f_dynamic_gated, f_static, sync_score, external_physics_feats], dim=-1)
        logits = self.classifier(f_final)

        return {
            "logits": logits,
            "f_fused": f_final,
            "modality_weights": weights,
            "sync_score": sync_score,
            "w_video": w_v,
            "w_audio": w_a
        }


class MultimodalBottleneckFusion(nn.Module):
    """
    Preserved for backward compatibility with LiteFFIANet.
    """
    def __init__(self, dim: int = 192, num_bottlenecks: int = 4, num_heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.num_bottlenecks = num_bottlenecks
        self.bottlenecks = nn.Parameter(torch.randn(1, num_bottlenecks, dim) * 0.02)
        self.cross_attn_v = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_a = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm_v = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, v_tokens: torch.Tensor, a_tokens: torch.Tensor) -> torch.Tensor:
        B = v_tokens.size(0)
        btn = self.bottlenecks.expand(B, -1, -1)
        btn_v, _ = self.cross_attn_v(query=btn, key=v_tokens, value=v_tokens)
        btn = self.norm_v(btn + btn_v)
        btn_a, _ = self.cross_attn_a(query=btn, key=a_tokens, value=a_tokens)
        btn = self.norm_a(btn + btn_a)
        btn = self.norm_out(btn + self.mlp(btn))
        return torch.mean(btn, dim=1)


class AdaptiveModalityGate(nn.Module):
    """
    Preserved for backward compatibility with LiteFFIANet.
    """
    def __init__(self, dim: int = 192) -> None:
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, f_video: torch.Tensor, f_audio: torch.Tensor) -> torch.Tensor:
        return self.gate_mlp(torch.cat([f_video, f_audio], dim=-1))
