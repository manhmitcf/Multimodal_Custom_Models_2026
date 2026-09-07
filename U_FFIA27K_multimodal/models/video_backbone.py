import torch
import torch.nn as nn
from typing import Tuple

try:
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
except ImportError:
    from torchvision.models import mobilenet_v2
    MobileNet_V2_Weights = None

from .motion_excitation import MotionExcitation


class VideoSpatiotemporalBackbone(nn.Module):
    """
    Enhanced Spatiotemporal Video Backbone extracting:
      - Group 1: Pure Spatial Features (f_spatial: fish aggregation, static foam, pond context)
                 extracted through all 19 layers of MobileNetV2 up to 1280 channels.
      - Group 2: Motion Dynamics (f_motion: splash velocity, water turbulence)
                 extracted via Motion Excitation on early difference: ΔF = |F_end - F_start|.
      - Combined: Joint Spatiotemporal Video Embedding (f_video)
      
    Efficiency Design:
    - Pretrained on ImageNet-1K with full 1280-d representation preserved.
    - Motion excitation computed at low-resolution stem (32 channels) to minimize FLOPs.
    - Total parameters: ~2.62M params (< 2.7M).
    """
    def __init__(self, embed_dim: int = 224, pretrained: bool = True) -> None:
        super().__init__()
        if MobileNet_V2_Weights is not None:
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            base_mobilenet = mobilenet_v2(weights=weights)
        else:
            base_mobilenet = mobilenet_v2(pretrained=pretrained)
        
        # Spatial Stem (Stages 0 - 2, 32 channels, H/4, W/4)
        self.stem = base_mobilenet.features[:7]
        
        # Lightweight Motion Excitation Module at early stem (32 channels)
        self.motion_excitation = MotionExcitation(in_channels=32, squeeze_factor=4)
        
        # Deep Stages (Stages 3 - 18, up to 1280 channels with rich ImageNet semantics)
        self.deep_stages = base_mobilenet.features[7:]
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projections to common embedding space
        self.spatial_proj = nn.Sequential(
            nn.Linear(1280, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.motion_proj = nn.Sequential(
            nn.Linear(32, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.joint_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Backward compatibility alias
        self.visual_proj = self.spatial_proj

    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            frames: Video tensor of shape [B, T, 3, H, W] where T >= 2.
                    frames[:, 0]  : First frame of the clip (Frame đầu)
                    frames[:, -1] : Last frame of the clip (Frame cuối)
                    
        Returns:
            f_video: Joint spatiotemporal video embedding [B, embed_dim]
            f_spatial: Pure spatial visual feature from LAST frame [B, embed_dim]
            f_motion: Motion dynamics feature between FIRST and LAST frame [B, embed_dim]
        """
        B, T, C, H, W = frames.shape
        if T == 1:
            f_end_early = self.stem(frames[:, 0])
            motion_mask = torch.zeros_like(f_end_early)
            f_motion = self.motion_proj(self.global_pool(motion_mask).flatten(1))
        elif T == 2:
            f_start_early = self.stem(frames[:, 0])
            f_end_early = self.stem(frames[:, -1])
            _, motion_mask = self.motion_excitation(f_end_early, f_start_early)
            f_motion = self.motion_proj(self.global_pool(motion_mask).flatten(1))
        else:
            # Multi-frame (T >= 3, e.g. T=4):
            # Run all T frames through early stem efficiently: [B*T, 3, H, W] -> [B*T, 32, H', W']
            frames_flat = frames.reshape(B * T, C, H, W)
            stem_flat = self.stem(frames_flat)
            _, c_s, h_s, w_s = stem_flat.shape
            stem_all = stem_flat.view(B, T, c_s, h_s, w_s)
            f_end_early = stem_all[:, -1]
            
            # Consecutive temporal difference across all steps: e.g. (|F1-F0| + |F2-F1| + |F3-F2|) / 3
            diff_consecutive = torch.mean(torch.abs(stem_all[:, 1:] - stem_all[:, :-1]), dim=1)
            motion_mask = self.motion_excitation.forward_diff(diff_consecutive)
            f_motion = self.motion_proj(self.global_pool(motion_mask).flatten(1))

        # 3. Pure Deep Spatial feature from LAST frame (Group 1 Spatial - 1280 channels)
        f_spatial_deep = self.deep_stages(f_end_early)  # [B, 1280, H/32, W/32]
        f_spatial = self.spatial_proj(self.global_pool(f_spatial_deep).flatten(1))

        # 4. Joint Spatiotemporal representation
        f_video = self.joint_proj(torch.cat([f_spatial, f_motion], dim=-1))

        return f_video, f_spatial, f_motion

