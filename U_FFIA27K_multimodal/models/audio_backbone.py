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


class FrequencyAttentionBlock(nn.Module):
    """
    Data-Driven Adaptive Frequency Attention Block (Group 3a Feature).
    
    Dynamically models inter-frequency dependencies across 128 Log-Mel frequency bins:
    - Eliminates rigid hardcoded acoustic priors, allowing the network to self-adapt to 
      ANY fish species (e.g., low-frequency swim bladder sounds vs. high-frequency surface splashes)
      and ANY aquaculture pond acoustic environment (various aeration machinery, pump types, water depths).
    - Dual Spectral Pooling:
        * Global Average Pooling captures baseline ambient noise profile of the specific pond.
        * Global Max Pooling captures transient impulsive bursts from feeding snaps and cavitation.
    - Shared Bottleneck MLP learns continuous channel-wise frequency modulation weights in [0, 1].
    - Initialized neutrally so no frequency band is artificially suppressed or boosted prior to training.
    """
    def __init__(
        self,
        n_mels: int = 128,
        reduction: int = 4
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        hidden_dim = max(16, n_mels // reduction)
        
        # Shared MLP for frequency channel attention
        self.mlp = nn.Sequential(
            nn.Linear(n_mels, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_mels, bias=True)
        )
        
        # Initialize final layer weights and bias near zero for a neutral identity-like starting point
        nn.init.normal_(self.mlp[2].weight, std=0.01)
        nn.init.constant_(self.mlp[2].bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Spectrogram tensor [B, 1, Time, Freq(128)] or [B, Time, Freq(128)]
        Returns:
            modulated_x: Spectrogram with frequency bands adaptively weighted [B, 1, Time, 128]
            weights: Dynamic frequency attention weights [B, 128]
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # x shape: [B, 1, T, F]
        # 1. Dual Spectral Pooling along time axis:
        avg_energy = x.mean(dim=(1, 2))  # [B, 128] - Ambient spectral profile
        max_energy = x.amax(dim=(1, 2))  # [B, 128] - Feeding impulsive peaks
        
        # 2. Shared MLP projection & combination
        logits = self.mlp(avg_energy) + self.mlp(max_energy) # [B, 128]
        
        # 3. Dynamic Attention Weights in range (0, 1)
        weights = torch.sigmoid(logits)  # [B, 128]
        
        # 4. Modulate spectrogram
        weights_expanded = weights.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 128]
        return x * weights_expanded, weights


class ConvBlock5x5(nn.Module):
    """
    PANNs CNN6 Convolutional Block:
    - 5x5 2D Convolution with stride 1, padding 2 (large time-frequency receptive field).
    - BatchNorm2d + ReLU activation.
    - Average Pooling (2, 2).
    - Xavier Uniform + BatchNorm weight initialization from PANNs.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        init_layer(self.conv)
        init_bn(self.bn)

    def forward(self, x: torch.Tensor, pool_size: Tuple[int, int] = (2, 2)) -> torch.Tensor:
        x = F.relu_(self.bn(self.conv(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)


class FishAudioBackbone(nn.Module):
    """
    PANNs CNN6-Enhanced Audio Backbone for Multimodal Fish Feeding Assessment:
    - Incorporates proven PANNs CNN6 architecture (Kong et al., IEEE/ACM TASLP 2020):
        * 4 Stages of 5x5 Convolutions with BatchNorm and ReLU (broader time-frequency footprint).
        * Dropout(0.2) after each stage for regularization against pond ambient noise.
        * PANNs Dual Temporal Pooling: Max + Mean over time captures impulsive cavitation snaps
          as well as sustained background feeding murmurs.
    - Data-Driven Adaptive Frequency Attention (Group 3a):
        * Modulates 128 Mel bands adaptively without hardcoded acoustic priors.
    - 1D Temporal Rhythm Stream (Group 3b):
        * Measures cadence and pulse repetition rate to resolve Weak vs Medium feeding intensity.
    - Preserves sequence features f_seq for 8-head Bi-directional Cross-Attention with video.
    - Parameter budget: ~0.99M params, maintaining total model strictly under 5.0M (~4.89M params).
    """
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 256,
        channels: Tuple[int, ...] = (32, 64, 96, 192)
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # 1. Data-Driven Frequency Attention Block (Group 3a)
        self.freq_attention = FrequencyAttentionBlock(n_mels=128, reduction=4)
        self.freq_proj = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # 2. 4-Stage PANNs CNN6 Backbone (5x5 Convolutions)
        self.conv1 = ConvBlock5x5(in_channels, channels[0])
        self.conv2 = ConvBlock5x5(channels[0], channels[1])
        self.conv3 = ConvBlock5x5(channels[1], channels[2])
        self.conv4 = ConvBlock5x5(channels[2], channels[3])
        self.drop = nn.Dropout2d(p=0.2)

        # 3. Sequence token projection for Multimodal Cross-Attention
        self.seq_proj = nn.Sequential(
            nn.Linear(channels[3], embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # 4. PANNs Dual Pooling Projection (Max + Mean over time)
        self.panns_pool_proj = nn.Sequential(
            nn.Linear(channels[3], embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # 5. 1D Temporal Rhythm Stream (Group 3b: pulse cadence & repetition)
        self.rhythm_stream = nn.Sequential(
            nn.Conv1d(channels[3], channels[3], kernel_size=5, padding=2, groups=channels[3], bias=False),
            nn.BatchNorm1d(channels[3]),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels[3], embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_dim)
        )
        self.rhythm_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # 6. Global Audio Feature Fusion
        self.audio_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, mel_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mel_spec: [B, 1, Time, 128] Log-Mel Spectrogram
        Returns:
            f_audio: [B, embed_dim]
            f_frequency: [B, embed_dim]
            f_rhythm: [B, embed_dim]
            f_seq: [B, T', embed_dim]
        """
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)

        # 1. Data-driven Frequency Attention
        weighted_mel, freq_weights = self.freq_attention(mel_spec)
        f_frequency = self.freq_proj(freq_weights) # [B, embed_dim]

        # 2. 4-Stage PANNs CNN6 (5x5 Convolutions)
        x = self.drop(self.conv1(weighted_mel, pool_size=(2, 2)))
        x = self.drop(self.conv2(x, pool_size=(2, 2)))
        x = self.drop(self.conv3(x, pool_size=(2, 2)))
        x = self.drop(self.conv4(x, pool_size=(1, 2))) # [B, 192, T', F']

        # 3. Collapse Frequency Axis (PANNs formulation) -> [B, 192, T']
        x_time = torch.mean(x, dim=3)

        # 4. Multimodal Sequence Tokens for Cross-Attention: [B, T', embed_dim]
        f_seq = self.seq_proj(x_time.permute(0, 2, 1))

        # 5. PANNs Dual Temporal Pooling (Max across time + Mean across time)
        (x_max, _) = torch.max(x_time, dim=2) # Peak cavitation burst clicks
        x_mean = torch.mean(x_time, dim=2)    # Sustained feeding ambient murmur
        f_panns = self.panns_pool_proj(x_max + x_mean) # [B, embed_dim]

        # 6. 1D Temporal Rhythm Stream
        rhythm_seq = self.rhythm_stream(x_time)
        f_rhythm = self.rhythm_proj(rhythm_seq.mean(dim=-1)) # [B, embed_dim]

        # 7. Joint Acoustic Feature: PANNs (Max+Mean) + Rhythm Cadence + Frequency Spectral Profile
        f_audio = self.audio_proj(f_panns + f_rhythm + f_frequency) # [B, embed_dim]

        return f_audio, f_frequency, f_rhythm, f_seq


FishPannsCNN6Backbone = FishAudioBackbone


class AudioAcousticBackbone(nn.Module):
    """
    Preserved for backward compatibility with LiteFFIANet.
    """
    def __init__(self, embed_dim: int = 192) -> None:
        super().__init__()
        self.block1 = DepthwiseAudioBlock(1, 16)
        self.block2 = DepthwiseAudioBlock(16, 32)
        self.block3 = DepthwiseAudioBlock(32, 64)
        self.block4 = DepthwiseAudioBlock(64, 128)
        self.block5 = DepthwiseAudioBlock(128, 256)
        self.block6 = DepthwiseAudioBlock(256, 512)

        self.freq_attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 512),
            nn.Sigmoid()
        )

        self.rhythm_conv = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim)
        )

        self.acoustic_proj = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.freq_proj = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.rhythm_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        x = self.block1(mel, pool_size=(2, 2))
        x = self.block2(x, pool_size=(2, 2))
        x = self.block3(x, pool_size=(2, 2))
        x = self.block4(x, pool_size=(2, 2))
        x = self.block5(x, pool_size=(2, 2))
        x = self.block6(x, pool_size=(1, 2))
        spatial_pool = x.mean(dim=2)
        freq_weights = self.freq_attention(spatial_pool.mean(dim=-1))
        f_frequency = self.freq_proj(freq_weights)
        time_pool = x.mean(dim=-1)
        f_rhythm_seq = self.rhythm_conv(time_pool)
        f_rhythm = self.rhythm_proj(f_rhythm_seq.mean(dim=-1))
        f_audio = self.acoustic_proj(x.mean(dim=(2, 3)))
        return f_audio, f_frequency, f_rhythm
