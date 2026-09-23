import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def drop_path_1d(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample for 1D tensors [B, C, L].
    Protects active gradient propagation on small batch sizes by ensuring at least one sample is kept.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    if random_tensor.sum() == 0:
        random_tensor[0] = 1.0
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath1D(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample for 1D convolutional residual blocks.
    Zero trainable parameters; identity pass-through during evaluation mode.
    """
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path_1d(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.4f}"


class ConvNeXtBlock1D(nn.Module):
    """
    1D ConvNeXt Block operating along the Spectral Frequency Dimension (~0.83M total backbone).
    Receptive field captures local resonant bandwidths and harmonic structures:
      - 1D Depthwise Conv (kernel_size=7)
      - GroupNorm(1, dim) (Channel-wise LayerNorm equivalent for 1D)
      - Inverted Bottleneck: Pointwise 1x1 Conv (dim -> 4*dim) + GELU + Pointwise 1x1 Conv (4*dim -> dim)
      - LayerScale parameter (gamma=1e-6) for stable initialization and anti-overfitting
      - DropPath (Stochastic Depth) on residual connection
    """
    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        mlp_ratio: int = 4,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=dim
        )
        self.norm = nn.GroupNorm(1, dim)
        self.pwconv1 = nn.Conv1d(dim, mlp_ratio * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(mlp_ratio * dim, dim, kernel_size=1)

        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim, 1)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath1D(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return shortcut + self.drop_path(x)


class FrequencyConvNeXtAudioBackbone(nn.Module):
    """
    Frequency-Domain 1D ConvNeXt Audio Backbone (~832K params, ~0.04 GFLOPs).
    Processes high-resolution 2049-bin STFT frequency representations:
      - Stem: Conv1d(in_channels=2 -> 32, k=7, s=4, p=3) compresses 2049 -> 513 bins (~250 Hz/bin)
      - 4 ConvNeXt Stages: [32, 64, 128, 224] with block depths (1, 1, 2, 1)
      - Hierarchical downsamplers: 513 -> 257 -> 129 -> 65 bins
      - Stochastic Depth (DropPath): Linear schedule [0.0 -> 0.1] across 5 residual blocks
      - Global Frequency Average Pooling: AdaptiveAvgPool1d(1) -> [B, 224]
      - Regularized output: Dropout(p=0.1) before returning joint embedding f_audio
    """
    def __init__(
        self,
        in_channels: int = 2,
        embed_dim: int = 224,
        dims: Tuple[int, ...] = (32, 64, 128, 224),
        depths: Tuple[int, ...] = (1, 1, 2, 1),
        drop_path_rate: float = 0.1,
        dropout: float = 0.1,
        num_tokens: int = 2,
        **kwargs
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.drop_path_rate = drop_path_rate

        # 1. Stem: Patchify along frequency axis (2049 -> 513 bins)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, dims[0], kernel_size=7, stride=4, padding=3),
            nn.GroupNorm(1, dims[0])
        )

        # 2. Downsamplers between stages (513 -> 257 -> 129 -> 65)
        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            down = nn.Sequential(
                nn.GroupNorm(1, dims[i]),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1)
            )
            self.downsample_layers.append(down)

        # 3. Stages with Stochastic Depth (Linear Schedule across 5 blocks)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[
                ConvNeXtBlock1D(
                    dim=dims[i],
                    kernel_size=7,
                    mlp_ratio=4,
                    drop_path=dp_rates[cur + j]
                )
                for j in range(depths[i])
            ])
            cur += depths[i]
            self.stages.append(stage)

        # 4. Final normalization, pooling, and output regularizer
        self.norm_final = nn.GroupNorm(1, dims[-1])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # 5. Parameter initialization protocol
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """
        Meta AI ConvNeXt initialization protocol:
          - Conv1d & Linear: trunc_normal_(std=0.02), bias=0.0
          - GroupNorm & LayerNorm: weight=1.0, bias=0.0
        """
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Dual-channel STFT spectrum [B, 2, 2049] or single-channel [B, 2049] / [B, 1, 2049].

        Returns:
            f_audio: Joint acoustic embedding [B, embed_dim]
            f_frequency: Spectral frequency feature [B, embed_dim]
            f_rhythm: Spectral contrast / rhythm feature [B, embed_dim]
            f_burst_a: Acoustic burst feature [B, embed_dim]
            tokens_audio: Sequence of audio tokens [B, num_tokens, embed_dim]
        """
        # Ensure 3D tensor [B, C, L]
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.size(1) == 1 and self.in_channels == 2:
            # Expand single-channel input by concatenating zeros for transient contrast
            x = torch.cat([x, torch.zeros_like(x)], dim=1)

        # Stem patchify
        x = self.stem(x)

        # 4 ConvNeXt stages with interleaved downsampling
        for i in range(3):
            x = self.stages[i](x)
            x = self.downsample_layers[i](x)
        x = self.stages[3](x)

        # Final normalization & global frequency average pooling
        x = self.norm_final(x)
        f_audio = self.dropout(self.pool(x).squeeze(-1))

        # Replicate tokens for tournament fusion compatibility
        tokens_audio = f_audio.unsqueeze(1).repeat(1, self.num_tokens, 1)

        return f_audio, f_audio, f_audio, f_audio, tokens_audio


# Canonical Aliases
AudioBackbone = FrequencyConvNeXtAudioBackbone
AudioMLPBackbone = FrequencyConvNeXtAudioBackbone
