import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)


def init_layer(layer: nn.Module) -> None:
    """Initialize a Linear or Convolutional layer."""
    if hasattr(layer, 'weight') and layer.weight is not None:
        nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn: nn.Module) -> None:
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


class PANNSCNN6AudioBackbone(nn.Module):
    """
    PANNS-CNN6-Pro Acoustic Backbone (~1.83M params).
    Processes 128 Mel-frequency filterbank spectrograms [B, 1, Ta, 128]:
      - Stage 1: 4-stage 5x5 Conv blocks: [40, 80, 160, 320] channels (~1.83M params)
      - Stage 2: PANNS Dual Pooling (max(time) + avg(time)) capturing acoustic bursts
      - Stage 3: Projection to unified multimodal embedding space (embed_dim=224)
    """
    def __init__(
        self,
        embed_dim: int = 224,
        num_tokens: int = 2,
        channels: Tuple[int, ...] = (40, 80, 160, 320),
        dropout: float = 0.2,
        **kwargs
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.dropout_rate = dropout

        # 1. 4 PANNS 5x5 Conv Blocks
        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=channels[0])
        self.conv_block2 = ConvBlock5x5(in_channels=channels[0], out_channels=channels[1])
        self.conv_block3 = ConvBlock5x5(in_channels=channels[1], out_channels=channels[2])
        self.conv_block4 = ConvBlock5x5(in_channels=channels[2], out_channels=channels[3])

        # 2. Projection layers to unified multimodal embedding dimension (224)
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
        x = self.conv_block1(mel_spec, pool_size=(2, 2))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv_block2(x, pool_size=(2, 2))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv_block3(x, pool_size=(2, 2))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv_block4(x, pool_size=(2, 2))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        temporal_seq = x.mean(dim=3)
        temporal_pooled = F.adaptive_avg_pool1d(temporal_seq, self.num_tokens)
        tokens_audio = self.token_proj(temporal_pooled.transpose(1, 2))

        x_freq = torch.mean(x, dim=3)
        (x_max, _) = torch.max(x_freq, dim=2)
        x_avg = torch.mean(x_freq, dim=2)
        x_dual = x_max + x_avg

        f_audio_raw = self.proj(x_dual)

        f_mean_a = tokens_audio.mean(dim=1)
        f_peak_a, _ = torch.max(tokens_audio, dim=1)
        f_burst_a = f_peak_a - f_mean_a

        f_frequency = self.proj(x_avg)
        f_rhythm = tokens_audio[:, -1]

        f_audio = self.norm_audio(f_audio_raw + f_burst_a)

        return f_audio, f_frequency, f_rhythm, f_burst_a, tokens_audio


class PentaBandSpectralMLP(nn.Module):
    """
    Branch 1: Penta-Band Spectral MLP (~0.54M params).
    Decomposes the 2049-bin STFT into 5 specialized acoustic subbands:
      - Band 1 (0-5 kHz, 80 bins: 0..80): Aerator & water pump low-frequency rumble -> Linear(80, 64)
      - Band 2 (5-20 kHz, 240 bins: 80..320): Water surface splashing & pellet drop -> Linear(240, 96)
      - Band 3 (20-45 kHz, 400 bins: 320..720): Body turbulence & tail wagging -> Linear(400, 128)
      - Band 4 (45-85 kHz, 640 bins: 720..1360): Pharyngeal bone teeth feed crunching -> Linear(640, 160)
      - Band 5 (85-128 kHz, 689 bins: 1360..2049): Ultrasonic cavitation bubble collapse -> Linear(689, 192)
    Followed by Subband SE-Attention (inter-band channel modulation) and projection to embed_dim (224).
    """
    def __init__(self, embed_dim: int = 224, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Subband projections: 64 + 96 + 128 + 160 + 192 = 640 dims
        self.proj_b1 = nn.Sequential(
            nn.Linear(80, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )
        self.proj_b2 = nn.Sequential(
            nn.Linear(240, 96),
            nn.LayerNorm(96),
            nn.GELU()
        )
        self.proj_b3 = nn.Sequential(
            nn.Linear(400, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        self.proj_b4 = nn.Sequential(
            nn.Linear(640, 160),
            nn.LayerNorm(160),
            nn.GELU()
        )
        self.proj_b5 = nn.Sequential(
            nn.Linear(689, 192),
            nn.LayerNorm(192),
            nn.GELU()
        )

        # Subband Squeeze-and-Excitation (SE) Attention
        self.se_fc1 = nn.Linear(640, 128)
        self.se_act = nn.GELU()
        self.se_fc2 = nn.Linear(128, 5)

        # Output projection
        self.proj_out = nn.Sequential(
            nn.Linear(640, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.proj_b1, self.proj_b2, self.proj_b3, self.proj_b4, self.proj_b5, self.proj_out]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    init_layer(layer)
        init_layer(self.se_fc1)
        init_layer(self.se_fc2)

    def forward(self, spec_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec_vector: STFT spectral vector [B, 2049]
        Returns:
            f_frequency: Spectral frequency feature [B, embed_dim]
        """
        if spec_vector.ndim > 2:
            spec_vector = spec_vector.flatten(start_dim=1)
        if spec_vector.size(-1) > 2049:
            spec_vector = spec_vector[:, :2049]
        elif spec_vector.size(-1) < 2049:
            spec_vector = F.pad(spec_vector, (0, 2049 - spec_vector.size(-1)))

        h1 = self.proj_b1(spec_vector[:, 0:80])
        h2 = self.proj_b2(spec_vector[:, 80:320])
        h3 = self.proj_b3(spec_vector[:, 320:720])
        h4 = self.proj_b4(spec_vector[:, 720:1360])
        h5 = self.proj_b5(spec_vector[:, 1360:2049])

        h_all = torch.cat([h1, h2, h3, h4, h5], dim=-1)  # [B, 640]

        # Subband SE-Attention weights
        weights = F.softmax(self.se_fc2(self.se_act(self.se_fc1(h_all))), dim=-1)  # [B, 5]

        # Modulate subband features with mean-preserving scale 5.0
        h1_m = h1 * (weights[:, 0:1] * 5.0)
        h2_m = h2 * (weights[:, 1:2] * 5.0)
        h3_m = h3 * (weights[:, 2:3] * 5.0)
        h4_m = h4 * (weights[:, 3:4] * 5.0)
        h5_m = h5 * (weights[:, 4:5] * 5.0)

        h_weighted = torch.cat([h1_m, h2_m, h3_m, h4_m, h5_m], dim=-1)  # [B, 640]
        f_frequency = self.proj_out(h_weighted)  # [B, embed_dim]
        return f_frequency


class MultiTrackTemporalCadenceEngine(nn.Module):
    """
    Branch 2: 5-Track 1D Dilated Temporal Cadence Engine (~0.065M params).
    Processes 5-band temporal energy sequence [B, 5, Time_Steps]:
      - Conv1d (5 -> 32, k=5, s=2, p=2): Micro-burst detection per band
      - Dilated Conv1d (32 -> 64, k=3, s=2, p=2, dilation=2): Multi-frequency temporal cadence & bite interval
      - Conv1d (64 -> 64, k=3, s=1, p=1): Temporal rhythm aggregation
      - Global Pooling -> f_rhythm [B, embed_dim]
      - Dynamic Peak-to-Average Contrast -> f_burst_a [B, embed_dim]
      - Adaptive Pooling -> tokens_audio [B, num_tokens, embed_dim]
    """
    def __init__(self, embed_dim: int = 224, num_tokens: int = 2) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        self.conv1 = nn.Conv1d(5, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.act2 = nn.GELU()

        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(64)
        self.act3 = nn.GELU()

        self.proj_rhythm = nn.Sequential(
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.proj_tokens = nn.Sequential(
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for conv in [self.conv1, self.conv2, self.conv3]:
            init_layer(conv)
        for bn in [self.bn1, self.bn2, self.bn3]:
            init_bn(bn)
        for proj in [self.proj_rhythm, self.proj_tokens]:
            for layer in proj:
                if isinstance(layer, nn.Linear):
                    init_layer(layer)

    def forward(
        self,
        temporal_energy: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if temporal_energy is None:
            temporal_energy = torch.zeros(batch_size, 5, 251, device=device, dtype=dtype or torch.float32)
        elif temporal_energy.ndim == 2:
            temporal_energy = temporal_energy.unsqueeze(1).repeat(1, 5, 1)

        x = self.act1(self.bn1(self.conv1(temporal_energy)))
        x = self.act2(self.bn2(self.conv2(x)))
        x_cadence = self.act3(self.bn3(self.conv3(x)))  # [B, 64, T']

        # 1. Rhythm feature via global average pooling
        x_avg = x_cadence.mean(dim=-1)                   # [B, 64]
        f_rhythm = self.proj_rhythm(x_avg)               # [B, embed_dim]

        # 2. Audio tokens sequence for temporal cadence alignment
        x_tokens = F.adaptive_avg_pool1d(x_cadence, self.num_tokens)  # [B, 64, num_tokens]
        tokens_audio = self.proj_tokens(x_tokens.transpose(1, 2))      # [B, num_tokens, embed_dim]

        # 3. Dynamic acoustic burst contrast: peak token minus mean token
        f_peak_a, _ = torch.max(tokens_audio, dim=1)                  # [B, embed_dim]
        f_mean_a = tokens_audio.mean(dim=1)                           # [B, embed_dim]
        f_burst_a = f_peak_a - f_mean_a                               # [B, embed_dim]

        return f_rhythm, f_burst_a, tokens_audio


class DualBranchCadenceAudioBackbone(nn.Module):
    """
    Dual-Branch Cadence Audio Backbone (~0.61M params).
    Combines:
      - Branch 1: Penta-Band Spectral MLP (~0.54M params) on 2049-bin STFT
      - Branch 2: 5-Track 1D Dilated Temporal Cadence Engine (~0.065M params)
      - Joint Acoustic Embedding: LayerNorm(f_frequency + f_rhythm + f_burst_a)
    """
    def __init__(
        self,
        embed_dim: int = 224,
        num_tokens: int = 2,
        dropout: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        self.spectral_branch = PentaBandSpectralMLP(embed_dim=embed_dim, dropout=dropout)
        self.temporal_branch = MultiTrackTemporalCadenceEngine(embed_dim=embed_dim, num_tokens=num_tokens)
        self.norm_audio = nn.LayerNorm(embed_dim)

    def forward(
        self, x: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: AudioFrontendOutput, dict with keys 'spec_vector' and 'temporal_energy',
               tuple (spec_vector, temporal_energy), or raw STFT tensor [B, 2049].

        Returns:
            f_audio: Joint acoustic embedding [B, embed_dim]
            f_frequency: Spectral frequency feature [B, embed_dim]
            f_rhythm: Temporal rhythm feature [B, embed_dim]
            f_burst_a: Acoustic burst feature [B, embed_dim]
            tokens_audio: Sequence of audio tokens [B, num_tokens, embed_dim]
        """
        if isinstance(x, dict):
            spec_vector = x["spec_vector"]
            temporal_energy = x.get("temporal_energy", None)
        elif isinstance(x, (tuple, list)):
            spec_vector = x[0]
            temporal_energy = x[1] if len(x) > 1 else None
        elif isinstance(x, torch.Tensor):
            spec_vector = x
            temporal_energy = None
        else:
            raise TypeError(f"Unsupported audio input type: {type(x)}")

        f_frequency = self.spectral_branch(spec_vector)
        f_rhythm, f_burst_a, tokens_audio = self.temporal_branch(
            temporal_energy=temporal_energy,
            batch_size=spec_vector.size(0),
            device=spec_vector.device,
            dtype=spec_vector.dtype
        )

        f_audio = self.norm_audio(f_frequency + f_rhythm + f_burst_a)

        return f_audio, f_frequency, f_rhythm, f_burst_a, tokens_audio


# Default AudioBackbone uses DualBranchCadenceAudioBackbone
AudioBackbone = DualBranchCadenceAudioBackbone
AudioMLPBackbone = DualBranchCadenceAudioBackbone
EfficientATAudioBackbone = DualBranchCadenceAudioBackbone
