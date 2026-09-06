from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalCrossAttentionFusion(nn.Module):
    """
    Bidirectional Multi-Head Cross-Attention (BMCA) Fusion with Adaptive Reliability Gating.
    
    References:
    - Audio-Visual Fish Feeding Intensity Assessment: A Benchmark Dataset and Modality Noise-Robust Method
      (Meng Cui et al., IEEE TASE 2024 / arXiv:2309.05058)
    - ACAF-Net: Adaptive Cross-Attention Fusion for Fish Feeding Intensity Assessment (2024)
    - PMIN: Progressive Multimodal Interaction Network (arXiv:2506.14170)
    
    Resolves the bottleneck compression issue of MBT:
    1. Video queries Audio (V -> A) to locate splashing acoustic energy.
    2. Audio queries Video (A -> V) to focus on visual fish density regions.
    3. Adaptive Modality Reliability Gate (AMRG) dynamically balances visual and acoustic confidence.
    """
    def __init__(self, dim: int = 224, num_heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.v_to_a_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.a_to_v_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        self.norm_v = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)
        
        self.fuse_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.LayerNorm(dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, f_video: torch.Tensor, f_audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f_video: Joint video embedding [B, dim]
            f_audio: Joint audio embedding [B, dim]
            
        Returns:
            out: Fused cross-modal representation [B, dim]
            alpha: Adaptive visual confidence factor [B, 1] in [0, 1]
        """
        v_token = f_video.unsqueeze(1)
        a_token = f_audio.unsqueeze(1)

        # 1. Bidirectional Cross-Modal Attention
        v_enhanced, _ = self.v_to_a_attn(query=v_token, key=a_token, value=a_token)
        a_enhanced, _ = self.a_to_v_attn(query=a_token, key=v_token, value=v_token)

        v_out = self.norm_v(v_token + v_enhanced).squeeze(1)
        a_out = self.norm_a(a_token + a_enhanced).squeeze(1)

        # 2. Adaptive Reliability Gating
        alpha = self.gate(torch.cat([v_out, a_out], dim=-1))
        gated = alpha * v_out + (1.0 - alpha) * a_out

        # 3. Dense Cross-Feature Refinement
        f_fused = self.fuse_mlp(torch.cat([v_out, a_out], dim=-1))

        # 4. Final Normalized Output
        out = F.layer_norm(gated + f_fused, (self.dim,))
        return out, alpha


class MultimodalBottleneckFusion(nn.Module):
    """
    Multimodal Bottleneck Transformer (MBT) Fusion Layer.
    
    Reference:
    - Multimodal Bottleneck Transformer for Audio-Visual Recognition
      (Nagrani et al., NeurIPS 2021)
    """
    def __init__(self, dim: int = 224, num_bottlenecks: int = 4, num_heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.num_bottlenecks = num_bottlenecks
        
        # Learnable bottleneck tokens
        self.bottlenecks = nn.Parameter(torch.randn(1, num_bottlenecks, dim) * 0.02)
        
        # Bidirectional cross-attention modules
        self.cross_attn_v = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_a = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        self.norm_v = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, v_tokens: torch.Tensor, a_tokens: torch.Tensor) -> torch.Tensor:
        B = v_tokens.size(0)
        btn = self.bottlenecks.expand(B, -1, -1)

        # 1. Interact with Video
        btn_v, _ = self.cross_attn_v(query=btn, key=v_tokens, value=v_tokens)
        btn = self.norm_v(btn + btn_v)

        # 2. Interact with Audio
        btn_a, _ = self.cross_attn_a(query=btn, key=a_tokens, value=a_tokens)
        btn = self.norm_a(btn + btn_a)

        # 3. Feed-forward refinement
        btn = self.norm_out(btn + self.mlp(btn))

        # 4. Global pooling across bottleneck tokens
        f_fused = torch.mean(btn, dim=1)
        return f_fused


class AdaptiveModalityGate(nn.Module):
    """
    Adaptive Reliability Gating Mechanism.
    """
    def __init__(self, dim: int = 224) -> None:
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, f_video: torch.Tensor, f_audio: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([f_video, f_audio], dim=-1)
        alpha = self.gate_mlp(combined)
        return alpha
