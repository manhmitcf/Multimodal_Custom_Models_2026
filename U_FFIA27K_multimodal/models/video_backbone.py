import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MobileViTVideoBackbone(nn.Module):
    """
    Spatiotemporal Video Backbone based on MobileViT-XS (Mehta & Rastegari, ICLR 2022).
    Tailored for 7-channel kinematic video inputs across T frames (T=4):
      - Channels 0-2 : Spatial RGB appearance (initialized with 100% Pretrained ImageNet weights)
      - Channels 3-4 : Optical Flow (u, v) swimming velocity
      - Channel 5    : Fluid Vorticity omega (swirling turbulence from feeding strike)
      - Channel 6    : Deceleration Delta|V| = |V_t| - |V_{t-1}| (temporal boundary transition signal)

    Architecture:
      - Lightweight MobileViT-XS (~1.93M base backbone)
      - Dual Token Extraction:
          * f_spatial: Static appearance token (shoal clustering, white water foam, pellets)
          * f_motion: Dynamic transition token (inter-frame flow and vorticity delta)
      - Temporal Transformer Encoder Layer (models dynamic evolution across T frames)
      - Residual Spatiotemporal Normalization
      - Total parameters: ~2.42M params.
    """
    def __init__(
        self,
        embed_dim: int = 224,
        pretrained: bool = True,
        num_frames: int = 4,
        in_chans: int = 7
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.in_chans = in_chans

        # Create MobileViT-XS with in_chans input channels (default 7)
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

            # Create in_chans-channel MobileViT-XS
            self.backbone = timm.create_model(
                'mobilevit_xs',
                pretrained=False,
                in_chans=self.in_chans,
                num_classes=0
            )

            # Weight Inflation: First 3 channels get ImageNet pretrained weights, remaining channels are zero-init
            if rgb_stem_weight is not None:
                stem_conv = None
                for module in self.backbone.modules():
                    if isinstance(module, nn.Conv2d) and module.in_channels == self.in_chans:
                        stem_conv = module
                        break
                if stem_conv is not None:
                    with torch.no_grad():
                        stem_conv.weight.zero_()
                        if rgb_stem_weight.shape == stem_conv.weight[:, :3].shape:
                            stem_conv.weight[:, :3] = rgb_stem_weight
                        logger.info(f"Successfully inflated 3-channel ImageNet weights into {self.in_chans}-channel stem conv.")
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

    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            frames: [B, T, C, H, W] 7-channel kinematic video tensor

        Returns:
            f_video: Joint spatiotemporal video embedding [B, embed_dim]
            f_spatial: Pure spatial visual appearance feature [B, embed_dim]
            f_motion: Motion dynamics feature across consecutive frame transitions [B, embed_dim]
            tokens_video: Sequence of frame tokens [B, T, embed_dim] for Bi-CA Multimodal Fusion
        """
        B, T, C, H, W = frames.shape

        # Process all T frames through MobileViT-XS: [B * T, C, H, W] -> [B * T, 384]
        flat_frames = frames.reshape(B * T, C, H, W)
        flat_feats = self.backbone(flat_frames)  # [B * T, 384]
        flat_tokens = self.spatial_proj(flat_feats)  # [B * T, embed_dim]

        # Reshape to temporal sequence of frame tokens: [B, T, embed_dim]
        frame_tokens = flat_tokens.view(B, T, self.embed_dim)

        # 1. Pure Spatial feature from the final frame (appearance of fish & water surface)
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
