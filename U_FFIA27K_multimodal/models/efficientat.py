import os
import math
import logging
from typing import Optional, Tuple, List, Dict, Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops.misc import ConvNormActivation
from torch.hub import load_state_dict_from_url

logger = logging.getLogger(__name__)

OFFICIAL_MN01_AS_URL = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn01_as_mAP_298.pt"


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """Ensures all layers have channel numbers divisible by divisor (usually 8)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def cnn_out_size(in_size: int, padding: int, dilation: int, kernel: int, stride: int) -> int:
    s = in_size + 2 * padding - dilation * (kernel - 1) - 1
    return math.floor(s / stride + 1)


class SqueezeExcitation(nn.Module):
    """
    Channel / Dimension Squeeze-and-Excitation block.
    """
    def __init__(
        self,
        input_dim: int,
        squeeze_dim: int,
        se_dim: int,
        activation: Callable[..., nn.Module] = nn.ReLU,
        scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, squeeze_dim)
        self.fc2 = nn.Linear(squeeze_dim, input_dim)
        self.se_dim = [1, 2, 3]
        self.se_dim.remove(se_dim)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input_tensor: Tensor) -> Tensor:
        scale = torch.mean(input_tensor, self.se_dim, keepdim=True)
        shape = scale.size()
        scale = self.fc1(scale.squeeze(2).squeeze(2))
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale).view(shape)

    def forward(self, input_tensor: Tensor) -> Tensor:
        scale = self._scale(input_tensor)
        return scale * input_tensor


class ConcurrentSEBlock(nn.Module):
    """
    Concurrent Squeeze-and-Excitation Block from EfficientAT.
    """
    def __init__(
        self,
        c_dim: int,
        f_dim: int,
        t_dim: int,
        se_cnf: Dict
    ) -> None:
        super().__init__()
        dims = [c_dim, f_dim, t_dim]
        self.conc_se_layers = nn.ModuleList()
        for d in se_cnf['se_dims']:
            input_dim = dims[d - 1]
            squeeze_dim = make_divisible(input_dim // se_cnf['se_r'], 8)
            self.conc_se_layers.append(SqueezeExcitation(input_dim, squeeze_dim, d))

        if se_cnf['se_agg'] == "max":
            self.agg_op = lambda x: torch.max(x, dim=0)[0]
        elif se_cnf['se_agg'] == "avg":
            self.agg_op = lambda x: torch.mean(x, dim=0)
        elif se_cnf['se_agg'] == "add":
            self.agg_op = lambda x: torch.sum(x, dim=0)
        elif se_cnf['se_agg'] == "min":
            self.agg_op = lambda x: torch.min(x, dim=0)[0]
        else:
            raise NotImplementedError(f"SE aggregation operation '{se_cnf['se_agg']}' not implemented")

    def forward(self, input_tensor: Tensor) -> Tensor:
        se_outs = [se_layer(input_tensor) for se_layer in self.conc_se_layers]
        return self.agg_op(torch.stack(se_outs, dim=0))


class InvertedResidualConfig:
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = (activation == "HS")
        self.stride = stride
        self.dilation = dilation
        self.f_dim = None
        self.t_dim = None

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return make_divisible(channels * width_mult, 8)

    def out_size(self, in_size: int) -> int:
        padding = (self.kernel - 1) // 2 * self.dilation
        return cnn_out_size(in_size, padding, self.dilation, self.kernel, self.stride)


class InvertedResidual(nn.Module):
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        se_cnf: Dict,
        norm_layer: Callable[..., nn.Module],
        depthwise_norm_layer: Callable[..., nn.Module]
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_channels == cnf.out_channels)
        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # 1. Expand Pointwise Conv
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # 2. Depthwise Conv
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=depthwise_norm_layer,
                activation_layer=activation_layer,
            )
        )

        # 3. Squeeze-and-Excitation
        if cnf.use_se and se_cnf and se_cnf.get('se_dims') is not None:
            layers.append(ConcurrentSEBlock(cnf.expanded_channels, cnf.f_dim, cnf.t_dim, se_cnf))

        # 4. Project Pointwise Conv
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, inp: Tensor) -> Tensor:
        result = self.block(inp)
        if self.use_res_connect:
            result = result + inp
        return result


def _build_mobilenet_v3_conf(width_mult: float = 0.1, strides: Tuple[int, ...] = (2, 2, 2, 2)):
    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
        bneck_conf(16, 3, 64, 24, False, "RE", strides[0], 1),
        bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
        bneck_conf(24, 5, 72, 40, True, "RE", strides[1], 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        bneck_conf(40, 3, 240, 80, False, "HS", strides[2], 1),
        bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
        bneck_conf(112, 5, 672, 160, True, "HS", strides[3], 1),
        bneck_conf(160, 5, 960, 160, True, "HS", 1, 1),
        bneck_conf(160, 5, 960, 160, True, "HS", 1, 1),
    ]
    last_channel = adjust_channels(1280)
    return inverted_residual_setting, last_channel


class EfficientAT_MN01(nn.Module):
    """
    MobileNetV3 backbone configured as EfficientAT MN01 (width_mult=0.1, ~124K params).
    Fully convolutional frontend accepting high-resolution 2D STFT spectrograms [B, 1, T, F].
    """
    def __init__(
        self,
        in_channels: int = 1,
        width_mult: float = 0.1,
        strides: Tuple[int, ...] = (2, 2, 2, 2),
        input_dims: Tuple[int, int] = (2049, 250),
        se_conf: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        if se_conf is None:
            se_conf = {'se_dims': [1], 'se_agg': 'max', 'se_r': 4}

        inverted_residual_setting, last_channel = _build_mobilenet_v3_conf(width_mult=width_mult, strides=strides)

        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        depthwise_norm_layer = norm_layer

        layers: List[nn.Module] = []
        firstconv_output_channels = inverted_residual_setting[0].input_channels

        # Stem Conv: [B, 1, T, F] -> [B, 8, T/2, F/2]
        layers.append(
            ConvNormActivation(
                in_channels,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        f_dim, t_dim = input_dims
        f_dim = cnn_out_size(f_dim, 1, 1, 3, 2)
        t_dim = cnn_out_size(t_dim, 1, 1, 3, 2)

        for cnf in inverted_residual_setting:
            f_dim = cnf.out_size(f_dim)
            t_dim = cnf.out_size(t_dim)
            cnf.f_dim, cnf.t_dim = f_dim, t_dim
            layers.append(InvertedResidual(cnf, se_conf, norm_layer, depthwise_norm_layer))

        # Final 1x1 expansion conv: 16 -> 96 channels
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)
        self.out_channels = lastconv_output_channels  # 96 for width_mult=0.1

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)


class EfficientATAudioBackbone(nn.Module):
    """
    SOTA Acoustic Feature Extractor based on EfficientAT (MN01, ~0.2M parameters).
    Processes high-resolution 2D TKEO-STFT Spectrograms [B, 1, Time, 2049]:
      - Deep Inverted Residual Feature Extraction (MobileNetV3 width 0.1 with SE-Attention)
      - PANNS Dual Pooling (Avg + Max over Frequency)
      - Adaptive Temporal Token Sequence (num_tokens=2)
      - Dynamic Acoustic Burst Contrast (Peak - Mean)
    """
    def __init__(
        self,
        embed_dim: int = 224,
        num_tokens: int = 2,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        dropout: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        # 1. Base MN01 Backbone (~124K params)
        self.mn01 = EfficientAT_MN01(width_mult=0.1)
        mn_channels = self.mn01.out_channels  # 96

        # 2. Multimodal Projections to unified dimension 224
        self.proj = nn.Sequential(
            nn.Linear(mn_channels, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.token_proj = nn.Sequential(
            nn.Linear(mn_channels, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.norm_audio = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # 3. Load Pretrained Checkpoint
        if pretrained:
            self._load_pretrained_weights(pretrained_path)

    def _load_pretrained_weights(self, checkpoint_path: Optional[str] = None) -> None:
        state_dict = None
        # Check custom/local path
        if checkpoint_path and os.path.isfile(checkpoint_path):
            logger.info(f"Loading EfficientAT MN01 weights from local file: {checkpoint_path}")
            try:
                state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except Exception as e:
                logger.warning(f"Failed to load local checkpoint {checkpoint_path}: {e}")

        # Fallback to download or cache
        if state_dict is None:
            cache_dir = os.path.join(os.getcwd(), "pretrained")
            os.makedirs(cache_dir, exist_ok=True)
            local_cached = os.path.join(cache_dir, "mn01_as_mAP_298.pt")
            if os.path.isfile(local_cached):
                logger.info(f"Loading EfficientAT MN01 from cache: {local_cached}")
                state_dict = torch.load(local_cached, map_location="cpu", weights_only=False)
            else:
                logger.info(f"Downloading official EfficientAT MN01 checkpoint from: {OFFICIAL_MN01_AS_URL}")
                try:
                    state_dict = load_state_dict_from_url(
                        OFFICIAL_MN01_AS_URL,
                        model_dir=cache_dir,
                        map_location="cpu",
                        progress=True
                    )
                except Exception as e:
                    logger.warning(f"Could not download EfficientAT pretrained weights: {e}. Initializing from scratch.")

        if state_dict is not None:
            # Filter only features weights (ignore old 527-class classifier)
            feature_dict = {
                k.replace("features.", ""): v
                for k, v in state_dict.items()
                if k.startswith("features.")
            }
            missing, unexpected = self.mn01.features.load_state_dict(feature_dict, strict=False)
            logger.info(f"[*] EfficientAT MN01 pre-trained weights successfully loaded! (Missing: {len(missing)}, Unexpected: {len(unexpected)})")

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 2D TKEO-STFT Spectrogram tensor:
               Shape [B, 1, Time, Frequency] or [B, Time, Frequency] or [B, Frequency]
        Returns:
            f_audio: Joint acoustic embedding [B, embed_dim]
            f_frequency: Spectral frequency feature [B, embed_dim]
            f_rhythm: Temporal rhythm feature [B, embed_dim]
            f_burst_a: Peak-to-Average acoustic burst contrast [B, embed_dim]
            tokens_audio: Sequence of temporal audio tokens [B, num_tokens, embed_dim]
        """
        # Ensure 4D shape: [B, 1, T, F]
        if x.ndim == 2:
            # [B, F] -> [B, 1, 1, F]
            x = x.unsqueeze(1).unsqueeze(1)
        elif x.ndim == 3:
            # [B, T, F] -> [B, 1, T, F]
            x = x.unsqueeze(1)

        # 1. Backbone forward pass -> [B, 96, T', F']
        feat_map = self.mn01(x)

        # 2. Extract Temporal Sequence Tokens (pool over frequency axis F')
        temporal_seq = feat_map.mean(dim=3)  # [B, 96, T']
        temporal_pooled = F.adaptive_avg_pool1d(temporal_seq, self.num_tokens)  # [B, 96, num_tokens]
        tokens_audio = self.token_proj(temporal_pooled.transpose(1, 2))  # [B, num_tokens, embed_dim]

        # 3. Dual Pooling (Avg + Max over frequency and time)
        x_freq = feat_map.mean(dim=3)           # [B, 96, T']
        x_max = torch.max(x_freq, dim=2)[0]     # [B, 96]
        x_avg = torch.mean(x_freq, dim=2)       # [B, 96]
        x_dual = x_max + x_avg                  # [B, 96]

        # 4. Joint Acoustic Embedding
        f_audio_raw = self.proj(x_dual)         # [B, embed_dim]

        # 5. Acoustic Burst Dynamic Contrast
        f_mean_a = tokens_audio.mean(dim=1)
        f_peak_a = torch.max(tokens_audio, dim=1)[0]
        f_burst_a = f_peak_a - f_mean_a         # [B, embed_dim]

        f_frequency = self.proj(x_avg)
        f_rhythm = tokens_audio[:, -1]

        f_audio = self.norm_audio(f_audio_raw + f_burst_a)

        return f_audio, f_frequency, f_rhythm, f_burst_a, tokens_audio
