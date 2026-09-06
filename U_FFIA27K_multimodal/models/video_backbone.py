import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

try:
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
except ImportError:
    from torchvision.models import mobilenet_v2
    MobileNet_V2_Weights = None

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


class InvertedResidual(nn.Module):
    """
    Custom Inverted Residual Block with optional TSM.
    """
    def __init__(
        self,
        in_c: int,
        out_c: int,
        stride: int,
        expand_ratio: int,
        use_tsm: bool = False,
        n_segment: int = 2
    ) -> None:
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_c == out_c
        self.use_tsm = use_tsm
        if use_tsm:
            self.tsm = TemporalShiftModule(n_segment=n_segment)

        hidden_dim = int(round(in_c * expand_ratio))
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_c, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.use_tsm:
            x = self.tsm(x)
        if self.use_res_connect:
            return identity + self.conv(x)
        return self.conv(x)


class FishVideoBackbone(nn.Module):
    """
    Custom Lightweight Video Backbone for Fish Feeding Intensity Assessment:
    - Input: 4-channel tensor [B, T, 4, H, W] (3 RGB + 1 Frame Differencing channel)
    - TSM (Temporal Shift Module) integrated across intermediate stages (0 params)
    - Motion Excitation (ME) blocks highlighting splash and feeding ripples
    - Outputs:
        * f_video: Global spatiotemporal embedding [B, embed_dim]
        * f_spatial: Pure spatial visual feature [B, embed_dim]
        * f_motion: Motion dynamics feature [B, embed_dim]
        * f_seq: Temporal frame sequence representation [B, T, embed_dim]
    - Parameter Budget: ~1.45M params (< 1.6M).
    """
    def __init__(self, in_channels: int = 4, embed_dim: int = 128, n_segment: int = 2) -> None:
        super().__init__()
        self.n_segment = n_segment
        self.embed_dim = embed_dim

        # Stem Conv: accepts 4 channels [B*T, 4, H, W] -> stride 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True)
        )

        # Stage 1: 24 -> 32 channels
        self.stage1 = InvertedResidual(24, 32, stride=1, expand_ratio=2, use_tsm=True, n_segment=n_segment)

        # Stage 2: 32 -> 48 channels + Motion Excitation
        self.stage2 = nn.Sequential(
            InvertedResidual(32, 48, stride=2, expand_ratio=3, use_tsm=True, n_segment=n_segment),
            InvertedResidual(48, 48, stride=1, expand_ratio=3, use_tsm=True, n_segment=n_segment),
        )
        self.me2 = MotionExcitation(in_channels=48, squeeze_factor=4)

        # Stage 3: 48 -> 80 channels + Motion Excitation
        self.stage3 = nn.Sequential(
            InvertedResidual(48, 80, stride=2, expand_ratio=3, use_tsm=True, n_segment=n_segment),
            InvertedResidual(80, 80, stride=1, expand_ratio=3, use_tsm=True, n_segment=n_segment),
            InvertedResidual(80, 80, stride=1, expand_ratio=3, use_tsm=True, n_segment=n_segment),
        )
        self.me3 = MotionExcitation(in_channels=80, squeeze_factor=4)

        # Stage 4: 80 -> 128 channels
        self.stage4 = nn.Sequential(
            InvertedResidual(80, 128, stride=2, expand_ratio=4, use_tsm=True, n_segment=n_segment),
            InvertedResidual(128, 128, stride=1, expand_ratio=4, use_tsm=True, n_segment=n_segment),
        )

        # Stage 5: 128 -> 192 channels
        self.stage5 = nn.Sequential(
            InvertedResidual(128, 192, stride=2, expand_ratio=4, use_tsm=False),
            InvertedResidual(192, 192, stride=1, expand_ratio=4, use_tsm=False),
        )

        self.head_conv = nn.Sequential(
            nn.Conv2d(192, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.spatial_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.motion_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x_4ch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_4ch: [B, T, 4, H, W] or [B, 4, H, W]
        Returns:
            f_video: [B, embed_dim]
            f_spatial: [B, embed_dim]
            f_motion: [B, embed_dim]
            f_seq: [B, T, embed_dim]
        """
        if x_4ch.dim() == 4:
            x_4ch = x_4ch.unsqueeze(1)
        b, t, c, h, w = x_4ch.shape

        self.n_segment = t

        x = x_4ch.view(b * t, c, h, w)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)

        x_reshaped = x.view(b, t, x.size(1), x.size(2), x.size(3))
        if t >= 2:
            x_curr = x_reshaped[:, -1]
            x_prev = x_reshaped[:, 0]
            x_curr_mod, m_mask = self.me2(x_curr, x_prev)
            x_reshaped = torch.cat([x_reshaped[:, :-1], x_curr_mod.unsqueeze(1)], dim=1)
            x = x_reshaped.view(b * t, x.size(1), x.size(2), x.size(3))

        x = self.stage3(x)
        x_reshaped = x.view(b, t, x.size(1), x.size(2), x.size(3))
        if t >= 2:
            x_curr = x_reshaped[:, -1]
            x_prev = x_reshaped[:, 0]
            x_curr_mod, _ = self.me3(x_curr, x_prev)
            x_reshaped = torch.cat([x_reshaped[:, :-1], x_curr_mod.unsqueeze(1)], dim=1)
            x = x_reshaped.view(b * t, x.size(1), x.size(2), x.size(3))

        x = self.stage4(x)
        x = self.stage5(x)

        feat = self.head_conv(x).view(b, t, -1) # [B, T, embed_dim]
        f_seq = feat

        f_video = feat.mean(dim=1)
        f_spatial = self.spatial_proj(feat[:, -1])
        if t >= 2:
            f_motion = self.motion_proj(torch.abs(feat[:, -1] - feat[:, 0]))
        else:
            f_motion = self.motion_proj(feat[:, 0])

        return f_video, f_spatial, f_motion, f_seq


class VideoSpatiotemporalBackbone(nn.Module):
    """
    Preserved for backward compatibility with LiteFFIANet.
    """
    def __init__(self, embed_dim: int = 192, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if (pretrained and MobileNet_V2_Weights) else None
        base_mobilenet = mobilenet_v2(weights=weights)
        self.stem = base_mobilenet.features[:7]
        self.motion_excitation = MotionExcitation(in_channels=32, squeeze_factor=4)
        self.deep_stages = base_mobilenet.features[7:18]
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.visual_proj = nn.Sequential(nn.Linear(320, embed_dim), nn.LayerNorm(embed_dim))
        self.motion_proj = nn.Sequential(nn.Linear(32, embed_dim), nn.LayerNorm(embed_dim))

    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C, H, W = frames.shape
        frame_start = frames[:, 0]
        frame_end = frames[:, -1] if T >= 2 else frames[:, 0]
        f_start_early = self.stem(frame_start)
        f_end_early = self.stem(frame_end)
        f_st_early, motion_mask = self.motion_excitation(f_end_early, f_start_early)
        f_st_deep = self.deep_stages(f_st_early)
        f_spatial_deep = self.deep_stages(f_end_early)
        f_video = self.visual_proj(self.global_pool(f_st_deep).flatten(1))
        f_spatial = self.visual_proj(self.global_pool(f_spatial_deep).flatten(1))
        f_motion = self.motion_proj(self.global_pool(motion_mask).flatten(1))
        return f_video, f_spatial, f_motion
