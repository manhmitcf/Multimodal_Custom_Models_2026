import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def init_layer(layer: nn.Module) -> None:
    if hasattr(layer, "weight") and layer.weight is not None:
        if layer.weight.dim() >= 2:
            nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn: nn.BatchNorm2d | nn.BatchNorm1d) -> None:
    if bn.bias is not None:
        bn.bias.data.fill_(0.0)
    if bn.weight is not None:
        bn.weight.data.fill_(1.0)


class DepthwiseAudioBlock(nn.Module):
    """
    Inverted residual depthwise separable block tailored for audio spectrograms.
    """
    def __init__(self, in_channels: int, out_channels: int, expansion: int = 1) -> None:
        super().__init__()
        mid_channels = out_channels * expansion
        
        self.conv1a = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1a = nn.BatchNorm2d(in_channels)
        self.conv1b = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False)
        self.bn1b = nn.BatchNorm2d(mid_channels)
        self.conv1c = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn1c = nn.BatchNorm2d(out_channels)
        
        self.gelu = nn.GELU()
        self.is_shortcut = (in_channels != out_channels)
        if self.is_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn_shortcut = nn.BatchNorm2d(out_channels)
            init_layer(self.shortcut)
            init_bn(self.bn_shortcut)

        init_layer(self.conv1a)
        init_layer(self.conv1b)
        init_layer(self.conv1c)
        init_bn(self.bn1a)
        init_bn(self.bn1b)
        init_bn(self.bn1c)

    def forward(self, x: torch.Tensor, pool_size: Tuple[int, int] = (2, 2)) -> torch.Tensor:
        origin = x
        out = self.conv1a(self.gelu(self.bn1a(origin)))
        out = self.conv1b(self.gelu(self.bn1b(out)))
        out = self.conv1c(self.gelu(self.bn1c(out)))

        if self.is_shortcut:
            origin = self.bn_shortcut(self.shortcut(origin))
            
        out = origin + out
        return F.avg_pool2d(out, kernel_size=pool_size, stride=pool_size)


class AudioAcousticBackbone(nn.Module):
    """
    Acoustic Spectrogram Backbone extracting:
      - Group 3a: Frequency Features (f_frequency: 2-8 kHz feeding splash frequency profile)
      - Group 3b: Rhythm Features (f_rhythm: temporal cadence and splashing frequency impulses)
      - Combined: Acoustic Embedding (f_audio)
      
    Total parameters: ~1.63M params.
    """
    def __init__(self, embed_dim: int = 224) -> None:
        super().__init__()
        
        # 6-Stage Inverted Residual Feature Extractor
        self.block1 = DepthwiseAudioBlock(1, 16)
        self.block2 = DepthwiseAudioBlock(16, 32)
        self.block3 = DepthwiseAudioBlock(32, 64)
        self.block4 = DepthwiseAudioBlock(64, 128)
        self.block5 = DepthwiseAudioBlock(128, 256)
        self.block6 = DepthwiseAudioBlock(256, 512)

        # Frequency Squeeze-and-Excitation (Group 3a: Focus on 2-8kHz feeding frequencies)
        self.freq_attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 512),
            nn.Sigmoid()
        )

        # 1D Temporal Difference Convolution (Group 3b: Splash & Chew Rhythm)
        self.rhythm_conv = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim)
        )

        # Projection to common embedding space
        self.proj = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.joint_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, mel_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            mel_spec: Log-Mel Spectrogram tensor [B, 1, Ta, 128]
            
        Returns:
            f_audio: Joint acoustic embedding [B, embed_dim]
            f_frequency: Spectral frequency distribution feature [B, embed_dim]
            f_rhythm: Temporal rhythm cadence feature [B, embed_dim]
        """
        x = self.block1(mel_spec, (2, 2))
        x = self.block2(x, (2, 2))
        x = self.block3(x, (2, 2))
        x = self.block4(x, (2, 2))
        x = self.block5(x, (2, 2))
        x = self.block6(x, (1, 1))  # [B, 512, T', F']

        # 1. Temporal Rhythm Extraction (Pool over Frequency F')
        temporal_sequence = torch.mean(x, dim=3)              # [B, 512, T']
        rhythm_sequence = self.rhythm_conv(temporal_sequence) # [B, embed_dim, T']
        f_rhythm = torch.mean(rhythm_sequence, dim=2)         # [B, embed_dim]

        # 2. Frequency Profile Extraction (Pool over Time T')
        freq_sequence = torch.mean(x, dim=2)                  # [B, 512, F']
        freq_global = torch.mean(freq_sequence, dim=2)        # [B, 512]
        w_freq = self.freq_attention(freq_global)
        f_frequency = self.proj(freq_global * w_freq)

        # 3. Joint Acoustic Vector fusing frequency profile and temporal rhythm
        f_audio = self.joint_proj(torch.cat([f_frequency, f_rhythm], dim=-1))

        return f_audio, f_frequency, f_rhythm
