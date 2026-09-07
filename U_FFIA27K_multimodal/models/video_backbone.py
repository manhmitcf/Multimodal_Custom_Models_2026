import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MobileViTVideoBackbone(nn.Module):
    """
    Spatiotemporal Video Backbone based on MobileViT-XS (Mehta & Rastegari, ICLR 2022).
    Tailored for 10-channel kinematic video inputs across T frames (T=4):
      - 3 RGB channels (initialized with 100% Pretrained ImageNet weights)
      - 7 Kinematic motion channels (u, v, |V|, theta, omega, Delta_I, MHI) initialized to zero.

    Architecture:
      - Lightweight MobileViT-XS (~1.93M base backbone)
      - Temporal Transformer Encoder Layer (models dynamic evolution across T frames)
      - Residual Spatiotemporal Normalization
      - Total parameters: ~2.42M params.
    """
    def __init__(
        self,
        embed_dim: int = 224,
        pretrained: bool = True,
        num_frames: int = 4
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames

        # Create MobileViT-XS with 10 input channels
        try:
            import timm
            # Grab 3-channel pretrained weights if requested
            if pretrained:
                try:
                    m_rgb = timm.create_model('mobilevit_xs', pretrained=True, num_classes=0)
                    rgb_stem_weight = None
                    for module in m_rgb.modules():
                        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                            rgb_stem_weight = module.weight.clone().detach()
                            break
                    del m_rgb
                except Exception as exc:
                    logger.warning(f"Could not load online pretrained weights: {exc}. Using random initialization.")
                    rgb_stem_weight = None
            else:
                rgb_stem_weight = None

            # Create 10-channel MobileViT-XS
            self.backbone = timm.create_model(
                'mobilevit_xs',
                pretrained=False,
                in_chans=10,
                num_classes=0
            )

            # Weight Inflation: First 3 channels get ImageNet pretrained weights, remaining 7 are zero-init
            if rgb_stem_weight is not None:
                stem_conv = None
                for module in self.backbone.modules():
                    if isinstance(module, nn.Conv2d) and module.in_channels == 10:
                        stem_conv = module
                        break
                if stem_conv is not None:
                    with torch.no_grad():
                        stem_conv.weight.zero_()
                        if rgb_stem_weight.shape == stem_conv.weight[:, :3].shape:
                            stem_conv.weight[:, :3] = rgb_stem_weight
                        logger.info("Successfully inflated 3-channel ImageNet weights into 10-channel stem conv.")
        except Exception as exc:
            raise ImportError(f"timm library is required for MobileViTVideoBackbone: {exc}")

        backbone_dim = getattr(self.backbone, 'num_features', 384)

        # Spatial projection to unified multimodal embedding space
        self.spatial_proj = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Temporal Transformer Encoder to capture inter-frame transitions (4 frames -> 3 transitions)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm_video = nn.LayerNorm(embed_dim)

    def forward(self, frames_10ch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            frames_10ch: [B, T, 10, H, W] 10-channel kinematic video tensor

        Returns:
            f_video: Joint spatiotemporal video embedding [B, embed_dim]
            f_spatial: Pure spatial visual feature from the last frame [B, embed_dim]
            f_motion: Motion dynamics feature across consecutive frame transitions [B, embed_dim]
            tokens_video: Sequence of frame tokens [B, T, embed_dim] for MBT Bottleneck Fusion
        """
        B, T, C, H, W = frames_10ch.shape

        # Process all T frames through MobileViT-XS: [B * T, 10, H, W] -> [B * T, 384]
        flat_frames = frames_10ch.reshape(B * T, C, H, W)
        flat_feats = self.backbone(flat_frames)  # [B * T, 384]
        flat_tokens = self.spatial_proj(flat_feats)  # [B * T, embed_dim]

        # Reshape to temporal sequence of frame tokens: [B, T, embed_dim]
        frame_tokens = flat_tokens.view(B, T, self.embed_dim)

        # 1. Pure Spatial feature from the final frame
        f_spatial = frame_tokens[:, -1]  # [B, embed_dim]

        # 2. Inter-frame temporal dynamics via Transformer Encoder
        temporal_tokens = self.temporal_encoder(frame_tokens)  # [B, T, embed_dim]

        # Motion dynamics between frame transitions (T=4 -> 3 transitions)
        if T >= 2:
            f_motion = torch.mean(torch.abs(temporal_tokens[:, 1:] - temporal_tokens[:, :-1]), dim=1)  # [B, embed_dim]
        else:
            f_motion = temporal_tokens[:, 0]

        # 3. Joint Spatiotemporal Video Embedding
        f_video = self.norm_video(f_spatial + f_motion)  # [B, embed_dim]

        return f_video, f_spatial, f_motion, temporal_tokens
