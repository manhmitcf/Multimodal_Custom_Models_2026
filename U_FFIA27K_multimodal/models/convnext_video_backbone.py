import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from .motion_excitation import MotionExcitation


class TemporalShiftModule(nn.Module):
    """
    Temporal Shift Module (TSM, Lin et al., ICCV 2019).
    Shifts 1/8 channels to the past and 1/8 channels to the future.
    Zero parameters, zero FLOPs, facilitates temporal modeling in 2D networks.
    """
    def __init__(self, n_segment: int = 2, fold_div: int = 8) -> None:
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B * T, C, H, W]
        bt, c, h, w = x.size()
        t = max(1, self.n_segment)
        if bt % t != 0:
            return x
        b = bt // t
        x = x.view(b, t, c, h, w)

        fold = max(1, c // self.fold_div)
        out = torch.zeros_like(x)
        if t > 1:
            out[:, :-1, :fold] = x[:, 1:, :fold]                    # Shift left (future to present)
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold] # Shift right (past to present)
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]               # Identity for the rest
        else:
            out = x
        return out.view(bt, c, h, w)


class LayerNorm2d(nn.Module):
    """
    2D LayerNorm for ConvNeXt (applied along channel dimension for (B, C, H, W) tensors).
    """
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block with integrated Temporal Shift Module (TSM):
    - 7x7 Depthwise Convolution (large receptive field for surface water ripples & fish clusters)
    - TSM channel shifting for zero-parameter spatiotemporal dynamics
    - Channels-first LayerNorm
    - Inverted Bottleneck (1x1 Conv with 4x expansion) + GELU activation
    - Pointwise 1x1 Conv projection
    - Layer Scale with learnable scaling parameter gamma (init 1e-6)
    - Residual skip connection
    """
    def __init__(
        self,
        dim: int,
        use_tsm: bool = True,
        n_segment: int = 2,
        layer_scale_init_value: float = 1e-6
    ) -> None:
        super().__init__()
        self.use_tsm = use_tsm
        if use_tsm:
            self.tsm = TemporalShiftModule(n_segment=n_segment)
            
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1, bias=True)
        
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)
            if layer_scale_init_value > 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.use_tsm:
            x = self.tsm(x)
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return identity + x


class FishConvNeXtBackbone(nn.Module):
    """
    Custom ConvNeXt Video Backbone for Fish Feeding Intensity Assessment:
    - Inputs: 6-channel tensor [B, T, 6, H, W] (3 RGB + 1 Velocity Magnitude + 2 Directional vx, vy)
    - Macro Architecture:
        * Stem: Conv2d 6 -> 48, kernel 4x4, stride 4, LayerNorm
        * Stage 1: 48 channels, 2 ConvNeXt blocks, H/4, W/4 (56x56)
        * Downsample 1: Conv2d 48 -> 96, 2x2, stride 2, LayerNorm
        * Stage 2: 96 channels, 2 ConvNeXt blocks, H/8, W/8 (28x28) + Motion Excitation (ME2)
        * Downsample 2: Conv2d 96 -> 192, 2x2, stride 2, LayerNorm
        * Stage 3: 192 channels, 4 ConvNeXt blocks, H/16, W/16 (14x14) + Motion Excitation (ME3)
        * Downsample 3: Conv2d 192 -> 256, 2x2, stride 2, LayerNorm
        * Stage 4: 256 channels, 2 ConvNeXt blocks, H/32, W/32 (7x7)
    - Head & Projections:
        * Global Average Pooling -> embed_dim (256)
        * f_spatial: Visual spatial embedding [B, embed_dim]
        * f_motion: Motion dynamics embedding [B, embed_dim]
        * f_seq: Temporal frame sequence representation [B, T, embed_dim]
    - Parameter Budget: ~2.93M params.
    """
    def __init__(
        self,
        in_channels: int = 6,
        embed_dim: int = 256,
        depths: Tuple[int, ...] = (2, 2, 4, 2),
        dims: Tuple[int, ...] = (48, 96, 192, 256),
        n_segment: int = 2
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_segment = n_segment

        # Stem Downsample: 6 -> dims[0] (stride 4: 224x224 -> 56x56)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4, bias=True),
            LayerNorm2d(dims[0], eps=1e-6)
        )

        # Stage 1 (48 ch, 2 blocks)
        self.stage1 = nn.Sequential(*[
            ConvNeXtBlock(dims[0], use_tsm=True, n_segment=n_segment)
            for _ in range(depths[0])
        ])

        # Downsample 1: 48 -> 96 (stride 2: 56x56 -> 28x28)
        self.downsample1 = nn.Sequential(
            LayerNorm2d(dims[0], eps=1e-6),
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2, bias=True)
        )

        # Stage 2 (96 ch, 2 blocks) + Motion Excitation
        self.stage2 = nn.Sequential(*[
            ConvNeXtBlock(dims[1], use_tsm=True, n_segment=n_segment)
            for _ in range(depths[1])
        ])
        self.me2 = MotionExcitation(in_channels=dims[1], squeeze_factor=4)

        # Downsample 2: 96 -> 192 (stride 2: 28x28 -> 14x14)
        self.downsample2 = nn.Sequential(
            LayerNorm2d(dims[1], eps=1e-6),
            nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2, bias=True)
        )

        # Stage 3 (192 ch, 4 blocks) + Motion Excitation
        self.stage3 = nn.Sequential(*[
            ConvNeXtBlock(dims[2], use_tsm=True, n_segment=n_segment)
            for _ in range(depths[2])
        ])
        self.me3 = MotionExcitation(in_channels=dims[2], squeeze_factor=4)

        # Downsample 3: 192 -> 256 (stride 2: 14x14 -> 7x7)
        self.downsample3 = nn.Sequential(
            LayerNorm2d(dims[2], eps=1e-6),
            nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2, bias=True)
        )

        # Stage 4 (256 ch, 2 blocks)
        self.stage4 = nn.Sequential(*[
            ConvNeXtBlock(dims[3], use_tsm=False)
            for _ in range(depths[3])
        ])

        self.norm = LayerNorm2d(dims[3], eps=1e-6)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Projections to embed_dim (256)
        self.spatial_proj = nn.Sequential(
            nn.Linear(dims[3], embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.motion_proj = nn.Sequential(
            nn.Linear(dims[3], embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x_6ch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_6ch: [B, T, 6, H, W] or [B, 6, H, W]
        Returns:
            f_video: [B, embed_dim]
            f_spatial: [B, embed_dim]
            f_motion: [B, embed_dim]
            f_seq: [B, T, embed_dim]
        """
        if x_6ch.dim() == 4:
            x_6ch = x_6ch.unsqueeze(1)
        b, t, c, h, w = x_6ch.shape
        self.n_segment = t

        x = x_6ch.view(b * t, c, h, w)
        x = self.stem(x)
        x = self.stage1(x)

        x = self.downsample1(x)
        x = self.stage2(x)
        # Apply ME2 across temporal segments
        x_reshaped = x.view(b, t, x.size(1), x.size(2), x.size(3))
        if t >= 2:
            x_curr = x_reshaped[:, -1]
            x_prev = x_reshaped[:, 0]
            x_curr_mod, _ = self.me2(x_curr, x_prev)
            x_reshaped = torch.cat([x_reshaped[:, :-1], x_curr_mod.unsqueeze(1)], dim=1)
            x = x_reshaped.view(b * t, x.size(1), x.size(2), x.size(3))

        x = self.downsample2(x)
        x = self.stage3(x)
        # Apply ME3 across temporal segments
        x_reshaped = x.view(b, t, x.size(1), x.size(2), x.size(3))
        if t >= 2:
            x_curr = x_reshaped[:, -1]
            x_prev = x_reshaped[:, 0]
            x_curr_mod, _ = self.me3(x_curr, x_prev)
            x_reshaped = torch.cat([x_reshaped[:, :-1], x_curr_mod.unsqueeze(1)], dim=1)
            x = x_reshaped.view(b * t, x.size(1), x.size(2), x.size(3))

        x = self.downsample3(x)
        x = self.stage4(x)
        x = self.norm(x)

        # Global Pooling -> [B, T, dims[-1]]
        pooled = self.global_pool(x).view(b, t, -1)
        f_seq = pooled  # [B, T, 256]

        f_video = pooled.mean(dim=1)
        f_spatial = self.spatial_proj(pooled[:, -1])
        if t >= 2:
            f_motion = self.motion_proj(torch.abs(pooled[:, -1] - pooled[:, 0]))
        else:
            f_motion = self.motion_proj(pooled[:, 0])

        return f_video, f_spatial, f_motion, f_seq
