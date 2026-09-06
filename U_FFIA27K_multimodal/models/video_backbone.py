import torch
import torch.nn as nn
from typing import Tuple, Dict

try:
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
except ImportError:
    from torchvision.models import mobilenet_v2
    MobileNet_V2_Weights = None

from .motion_excitation import MotionExcitation


class VideoSpatiotemporalBackbone(nn.Module):
    """
    Unified Spatiotemporal Video Backbone extracting:
      - Group 1: Spatial Features (f_spatial: fish aggregation, static foam, pond context)
      - Group 2: Motion Features (f_motion: splash velocity, water turbulence)
      - Combined: Spatiotemporal Embedding (f_video)
      
    Efficiency Design:
    - Based on MobileNetV2 with ImageNet pretrained initialization.
    - Features up to Stage 6 (320 channels) are used, intentionally bypassing the 
      heavy 1280-channel expansion layer to save ~410,000 parameters.
    - Total parameters: ~1.88M params.
    """
    def __init__(self, embed_dim: int = 192, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if (pretrained and MobileNet_V2_Weights) else None
        base_mobilenet = mobilenet_v2(weights=weights)
        
        # Spatial Stem (Stages 0 - 2, 32 channels, H/4, W/4)
        self.stem = base_mobilenet.features[:7]
        
        # Lightweight Motion Excitation Module at early-mid stage
        self.motion_excitation = MotionExcitation(in_channels=32, squeeze_factor=4)
        
        # Spatiotemporal Deep Stages (Stages 3 - 6, up to 320 channels)
        self.deep_stages = base_mobilenet.features[7:18]
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Linear projections with LayerNorm
        self.visual_proj = nn.Sequential(
            nn.Linear(320, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.motion_proj = nn.Sequential(
            nn.Linear(32, embed_dim),
            nn.LayerNorm(embed_dim)
        )

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
        if T >= 2:
            frame_start = frames[:, 0]   # Frame đầu clip
            frame_end = frames[:, -1]    # Frame cuối clip (dùng cho Spatial)
        else:
            frame_start = frames[:, 0]
            frame_end = frames[:, 0]

        # 1. Early spatial stem representations
        f_start_early = self.stem(frame_start)  # [B, 32, H/4, W/4]
        f_end_early = self.stem(frame_end)      # [B, 32, H/4, W/4]

        # 2. Extract Motion Mask between Last Frame and First Frame (Group 2)
        # Calculates cumulative changes ΔF = |F_end - F_start|
        f_st_early, motion_mask = self.motion_excitation(f_end_early, f_start_early)

        # 3. Deep feature extraction
        f_st_deep = self.deep_stages(f_st_early)        # [B, 320, H/32, W/32]
        f_spatial_deep = self.deep_stages(f_end_early)  # [B, 320, H/32, W/32] (Spatial thuần từ frame cuối)

        # 4. Pooling and projections
        f_video = self.visual_proj(self.global_pool(f_st_deep).flatten(1))
        f_spatial = self.visual_proj(self.global_pool(f_spatial_deep).flatten(1))
        f_motion = self.motion_proj(self.global_pool(motion_mask).flatten(1))

        return f_video, f_spatial, f_motion

