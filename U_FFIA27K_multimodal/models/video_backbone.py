import torch
import torch.nn as nn
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Guarantees active gradient propagation on small batch sizes by ensuring at least one sample is kept.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize to 0 or 1
    if random_tensor.sum() == 0:
        random_tensor[0] = 1.0
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Zero trainable parameters; identity pass-through during evaluation mode.
    """
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block (Liu et al., CVPR 2022).
    - Depthwise Conv 7x7
    - LayerNorm (across channels)
    - Pointwise Conv / Linear (dim -> 4*dim)
    - GELU activation
    - Pointwise Conv / Linear (4*dim -> dim)
    - LayerScale (gamma=1e-6)
    - DropPath (Stochastic Depth)
    - Residual Connection
    """
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0.0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        # Permute for LayerNorm & Linear: [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        return shortcut + self.drop_path(x)


class ConvNeXtNanoVideoBackbone(nn.Module):
    """
    Streamlined ConvNeXt-Nano Video Backbone tailored for 7-channel kinematic inputs across T=2 frames.
    Channels:
      - 0, 1, 2: Spatial RGB appearance
      - 3, 4:    Optical Flow (u, v) swimming velocity
      - 5:       Velocity Magnitude |V| = sqrt(u^2 + v^2)
      - 6:       Fluid Vorticity omega = dv/dx - du/dy

    Architecture:
      - Stem: Conv 4x4 (7 -> 48) + LayerNorm
      - Stage 1: 48ch  x 1 block  (LayerScale + DropPath linear schedule)
      - Stage 2: 96ch  x 1 block  (LayerScale + DropPath linear schedule)
      - Stage 3: 192ch x 3 blocks (LayerScale + DropPath linear schedule)
      - Stage 4: 384ch x 1 block  (LayerScale + DropPath linear schedule)
      - Total parameters: ~2.702M params.
    """
    def __init__(
        self,
        embed_dim: int = 224,
        in_chans: int = 7,
        dims: Tuple[int, ...] = (48, 96, 192, 384),
        depths: Tuple[int, ...] = (1, 1, 3, 1),
        num_frames: int = 2,
        drop_path_rate: float = 0.1,
        layer_scale_init_value: float = 1e-6,
        **kwargs
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.num_frames = num_frames
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value

        # 1. Stem: Patchify 4x4
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0], eps=1e-6)  # Equivalent to LayerNorm over [C, H, W]
        )

        # 2. Downsample layers between stages
        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            downsample = nn.Sequential(
                nn.GroupNorm(1, dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample)

        # 3. Stages with LayerScale and Stochastic Depth (Linear Schedule — Chuẩn Meta AI)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[
                ConvNeXtBlock(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                )
                for j in range(depths[i])
            ])
            cur += depths[i]
            self.stages.append(stage)

        # 4. Final normalization & projection
        self.norm_final = nn.LayerNorm(dims[-1], eps=1e-6)
        self.proj = nn.Sequential(
            nn.Linear(dims[-1], embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Residual normalization
        self.norm_video = nn.LayerNorm(embed_dim)

        # 5. Initialize weights with Meta AI Truncated Normal recipe
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """
        Meta AI ConvNeXt weight initialization:
        Truncated normal with std=0.02 for Linear and Conv2d, zero bias.
        """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature vector from [B * T, C, H, W].
        """
        x = self.stem(x)
        for i in range(4):
            if i > 0:
                x = self.downsample_layers[i - 1](x)
            x = self.stages[i](x)

        # Global Average Pooling: [B * T, 384, H', W'] -> [B * T, 384]
        x = x.mean(dim=[-2, -1])
        x = self.norm_final(x)
        x = self.proj(x)  # [B * T, embed_dim]
        return x

    def forward(
        self, frames_7ch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            frames_7ch: [B, T, 7, H, W] tensor (T=2)

        Returns:
            f_video: Joint spatiotemporal video embedding [B, embed_dim]
            f_spatial: Pure spatial visual appearance feature [B, embed_dim]
            f_motion: Motion dynamics feature between frames [B, embed_dim]
            f_burst_v: Peak-to-Average dynamic contrast [B, embed_dim]
            tokens_video: Sequence of frame tokens [B, T, embed_dim]
        """
        B, T, C, H, W = frames_7ch.shape

        # Process all T frames: [B * T, 7, H, W] -> [B * T, embed_dim]
        flat_frames = frames_7ch.reshape(B * T, C, H, W)
        flat_tokens = self.forward_features(flat_frames)  # [B * T, embed_dim]

        # Reshape to temporal sequence of frame tokens: [B, T, embed_dim]
        tokens_video = flat_tokens.view(B, T, self.embed_dim)

        # 1. Pure Spatial feature from the final frame (appearance of fish & water surface)
        f_spatial = tokens_video[:, -1]  # [B, embed_dim]

        # 2. Inter-frame motion dynamics
        if T >= 2:
            f_motion = torch.abs(tokens_video[:, 1] - tokens_video[:, 0])  # [B, embed_dim]
        else:
            f_motion = tokens_video[:, 0]

        # 3. Peak-to-Average Dynamic Contrast (Burst feeding strike intensity)
        f_mean_v = tokens_video.mean(dim=1)
        f_peak_v, _ = torch.max(tokens_video, dim=1)
        f_burst_v = f_peak_v - f_mean_v  # [B, embed_dim]

        # 4. Joint Spatiotemporal Video Embedding
        f_video = self.norm_video(f_spatial + f_motion + f_burst_v)  # [B, embed_dim]

        return f_video, f_spatial, f_motion, f_burst_v, tokens_video


# Canonical VideoBackbone alias
VideoBackbone = ConvNeXtNanoVideoBackbone
