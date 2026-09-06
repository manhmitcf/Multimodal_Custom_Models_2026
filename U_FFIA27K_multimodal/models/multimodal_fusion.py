import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalBottleneckFusion(nn.Module):
    """
    Multimodal Bottleneck Transformer (MBT) Fusion Layer.
    
    Reference:
    - Multimodal Bottleneck Transformer for Audio-Visual Recognition
      (Nagrani et al., NeurIPS 2021)
      
    Mechanism:
    Instead of full dense cross-attention between all video and audio tokens,
    a compact set of K learnable bottleneck tokens act as an information bridge:
        Video Tokens <--> Bottleneck Tokens <--> Audio Tokens
    Forces the network to compress and exchange only salient cross-modal cues,
    effectively filtering unimodal noise with linear computational complexity O(N*K).
    """
    def __init__(self, dim: int = 192, num_bottlenecks: int = 4, num_heads: int = 4) -> None:
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
        """
        Args:
            v_tokens: Video tokens [B, N_v, dim]
            a_tokens: Audio tokens [B, N_a, dim]
            
        Returns:
            f_fused: Fused multimodal bottleneck representation [B, dim]
        """
        B = v_tokens.size(0)
        btn = self.bottlenecks.expand(B, -1, -1)  # [B, K, dim]

        # 1. Interact with Video: Bottlenecks absorb salient visual patterns
        btn_v, _ = self.cross_attn_v(query=btn, key=v_tokens, value=v_tokens)
        btn = self.norm_v(btn + btn_v)

        # 2. Interact with Audio: Bottlenecks absorb acoustic rhythm & frequency
        btn_a, _ = self.cross_attn_a(query=btn, key=a_tokens, value=a_tokens)
        btn = self.norm_a(btn + btn_a)

        # 3. Feed-forward refinement
        btn = self.norm_out(btn + self.mlp(btn))

        # 4. Global pooling across bottleneck tokens
        f_fused = torch.mean(btn, dim=1)  # [B, dim]
        return f_fused


class AdaptiveModalityGate(nn.Module):
    """
    Adaptive Reliability Gating Mechanism.
    
    Dynamically estimates a gating factor alpha in [0, 1] conditioned on the joint state
    of visual and acoustic signals.
    - If water is turbid / glare occurs: alpha -> 0 (Audio trusted more).
    - If acoustic background noise / aeration pump spikes: alpha -> 1 (Video trusted more).
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
        """
        Args:
            f_video: Video embedding [B, dim]
            f_audio: Audio embedding [B, dim]
            
        Returns:
            alpha: Gating factor in [0, 1] of shape [B, 1]
        """
        combined = torch.cat([f_video, f_audio], dim=-1)
        alpha = self.gate_mlp(combined)
        return alpha
