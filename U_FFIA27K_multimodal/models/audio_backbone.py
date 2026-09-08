import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def init_layer(layer: nn.Module) -> None:
    """Initialize a Linear or Convolutional layer."""
    if hasattr(layer, 'weight') and layer.weight is not None:
        nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn: nn.BatchNorm2d) -> None:
    """Initialize a BatchNorm layer."""
    if hasattr(bn, 'bias') and bn.bias is not None:
        bn.bias.data.fill_(0.)
    if hasattr(bn, 'weight') and bn.weight is not None:
        bn.weight.data.fill_(1.)


class ConvBlock5x5(nn.Module):
    """
    PANNS-style 5x5 Convolutional block with BatchNorm and ReLU.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.init_weights()

    def init_weights(self) -> None:
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, x: torch.Tensor, pool_size: Tuple[int, int] = (2, 2)) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.avg_pool2d(x, kernel_size=pool_size)
        return x


class LearnableFrequencyAttention(nn.Module):
    """
    Data-driven Frequency / Mel-bin Attention.
    Learns dynamic importance weights across the 128 Mel-frequency bins:
      - Pools across time: [B, 1, T, 128] -> [B, 128]
      - Multi-layer perceptron: 128 -> 32 -> 128 -> Sigmoid
      - Automatically emphasizes feeding splash frequencies and suppresses pump hum.
      - 0 manual hardcoding; ~8.3K parameters.
    """
    def __init__(self, mel_bins: int = 128, reduction: int = 4) -> None:
        super().__init__()
        hidden_dim = mel_bins // reduction  # 32
        self.mlp = nn.Sequential(
            nn.Linear(mel_bins, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, mel_bins),
            nn.Sigmoid()
        )

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spec: [B, 1, T, F] (F=128)
        Returns:
            mel_focused: [B, 1, T, F]
        """
        # Pool time dimension: [B, 1, T, F] -> [B, F]
        time_mean = mel_spec.mean(dim=2).squeeze(1)  # [B, F]
        w_freq = self.mlp(time_mean)  # [B, F] in (0, 1)
        w_freq = w_freq.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, F]
        return mel_spec * w_freq


class PANNSCNN6AudioBackbone(nn.Module):
    """
    PANNS-CNN6-Pro Acoustic Backbone (~1.75M params).
    Processes 128 Mel-frequency filterbank spectrograms [B, 1, Ta, 128]:
      - Stage 1: Learnable Frequency Attention across 128 Mel-bins (~8.3K params)
      - Stage 2: 4-stage 5x5 Conv blocks: [40, 80, 160, 320] channels (~1.68M params)
      - Stage 3: PANNS Dual Pooling (max(time) + avg(time)) capturing acoustic bursts
      - Stage 4: Projection to unified multimodal embedding space (embed_dim=224)
    """
    def __init__(
        self,
        embed_dim: int = 224,
        num_tokens: int = 2,
        channels: Tuple[int, ...] = (40, 80, 160, 320),
        dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.dropout_rate = dropout

        # 1. Learnable Frequency Attention (Freq-SE)
        self.freq_attention = LearnableFrequencyAttention(mel_bins=128, reduction=4)

        # 2. 4 PANNS 5x5 Conv Blocks
        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=channels[0])
        self.conv_block2 = ConvBlock5x5(in_channels=channels[0], out_channels=channels[1])
        self.conv_block3 = ConvBlock5x5(in_channels=channels[1], out_channels=channels[2])
        self.conv_block4 = ConvBlock5x5(in_channels=channels[2], out_channels=channels[3])

        # 3. Projection layers to unified multimodal embedding dimension (224)
        self.proj = nn.Sequential(
            nn.Linear(channels[3], embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Token projection for cross-modal alignment (num_tokens=2)
        self.token_proj = nn.Sequential(
            nn.Linear(channels[3], embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.norm_audio = nn.LayerNorm(embed_dim)

    def forward(
        self, mel_spec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mel_spec: Log-Mel Spectrogram [B, 1, Ta, 128]

        Returns:
            f_audio: Joint acoustic embedding [B, embed_dim]
            f_frequency: Spectral frequency feature [B, embed_dim]
            f_rhythm: Temporal rhythm feature [B, embed_dim]
            f_burst_a: Peak-to-Average acoustic burst contrast [B, embed_dim]
            tokens_audio: Sequence of temporal audio tokens [B, num_tokens, embed_dim]
        """
        # Step 1: Learnable Frequency Attention
        x = self.freq_attention(mel_spec)

        # Step 2: 4 PANNS Conv blocks with dropout
        x = self.conv_block1(x, pool_size=(2, 2))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv_block2(x, pool_size=(2, 2))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv_block3(x, pool_size=(2, 2))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv_block4(x, pool_size=(2, 2))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # Shape: [B, 320, T', F']

        # Step 3: Extract temporal sequence tokens (pool over frequency F')
        # [B, 320, T', F'] -> [B, 320, T']
        temporal_seq = x.mean(dim=3)
        # Adaptively pool time to num_tokens (default 2)
        temporal_pooled = F.adaptive_avg_pool1d(temporal_seq, self.num_tokens)  # [B, 320, num_tokens]
        tokens_audio = self.token_proj(temporal_pooled.transpose(1, 2))  # [B, num_tokens, embed_dim]

        # Step 4: PANNS Dual Pooling (max(time) + avg(time) over frequency-pooled features)
        # Code reference from original PANNS CNN6:
        #   x = torch.mean(x, dim=3)
        #   (x1, _) = torch.max(x, dim=2)
        #   x2 = torch.mean(x, dim=2)
        #   x = x1 + x2
        x_freq = torch.mean(x, dim=3)         # [B, 320, T']
        (x_max, _) = torch.max(x_freq, dim=2) # [B, 320]
        x_avg = torch.mean(x_freq, dim=2)     # [B, 320]
        x_dual = x_max + x_avg                # [B, 320]

        # Step 5: Linear projection to embed_dim (224)
        f_audio_raw = self.proj(x_dual)       # [B, embed_dim]

        # Step 6: Burst contrast & features
        f_mean_a = tokens_audio.mean(dim=1)
        f_peak_a, _ = torch.max(tokens_audio, dim=1)
        f_burst_a = f_peak_a - f_mean_a       # [B, embed_dim]

        f_frequency = self.proj(x_avg)
        f_rhythm = tokens_audio[:, -1]

        f_audio = self.norm_audio(f_audio_raw + f_burst_a)

        return f_audio, f_frequency, f_rhythm, f_burst_a, tokens_audio


# Backward compatibility alias
EfficientATAudioBackbone = PANNSCNN6AudioBackbone
AudioBackbone = PANNSCNN6AudioBackbone
