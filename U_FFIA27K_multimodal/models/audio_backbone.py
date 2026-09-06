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


class DepthwiseAudioBlock(nn.Module):
    """
    Inverted residual depthwise separable block tailored for audio spectrograms.
    """
    def __init__(self, in_channels: int, out_channels: int, expansion: int = 2) -> None:
        super().__init__()
        mid_channels = in_channels * expansion
        
        self.conv1a = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1a = nn.BatchNorm2d(mid_channels)
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
        identity = x
        out = self.gelu(self.bn1a(self.conv1a(x)))
        out = self.gelu(self.bn1b(self.conv1b(out)))
        out = self.bn1c(self.conv1c(out))

        if self.is_shortcut:
            identity = self.bn_shortcut(self.shortcut(identity))
            
        out = identity + out
        return F.avg_pool2d(out, kernel_size=pool_size, stride=pool_size)


class FishAudioBackbone(nn.Module):
    """
    Custom Lightweight Audio Backbone for Fish Feeding Intensity Assessment:
    - Input: Log-Mel Spectrogram [B, 1, Time, 128]
    - Frequency Attention (2-8kHz splash amplification)
    - 4-Stage Depthwise Separable 2D Convolutions
    - 1D Temporal Rhythm Stream (Pulse Repetition Rate)
    - Outputs:
        * f_audio: Global acoustic embedding [B, embed_dim]
        * f_frequency: Pure frequency spectral feature [B, embed_dim]
        * f_rhythm: Temporal rhythm feature [B, embed_dim]
        * f_seq: Temporal sequence feature [B, T', embed_dim]
    - Parameter Budget: ~0.77M params.
    """
    def __init__(self, in_channels: int = 1, embed_dim: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Frequency attention block directly at input (physics-grounded prior)
        self.freq_attention = FrequencyAttentionBlock(n_mels=128, reduction=4)

        # 5 Stages of Depthwise Inverted Residuals (~0.75M params)
        self.block1 = DepthwiseAudioBlock(in_channels, 32, expansion=2)
        self.block2 = DepthwiseAudioBlock(32, 64, expansion=2)
        self.block3 = DepthwiseAudioBlock(64, 128, expansion=3)
        self.block4 = DepthwiseAudioBlock(128, 192, expansion=3)
        self.block5 = DepthwiseAudioBlock(192, 256, expansion=3)

        # Frequency pooling & projection
        self.freq_proj = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # 1D Temporal Rhythm Stream (Group 3b: cadence & pulse repetition rate)
        self.rhythm_stream = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=5, padding=2, groups=256, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Conv1d(256, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_dim)
        )

        self.rhythm_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

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
        # Ensure 4D
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)

        # 1. Frequency Attention (Group 3a) with Physics Prior
        weighted_mel, freq_weights = self.freq_attention(mel_spec)
        f_frequency = self.freq_proj(freq_weights) # [B, embed_dim]

        # 2. Time-Frequency 2D Conv stages (5 stages)
        x = self.block1(weighted_mel, pool_size=(2, 2))
        x = self.block2(x, pool_size=(2, 2))
        x = self.block3(x, pool_size=(2, 2))
        x = self.block4(x, pool_size=(1, 2))
        x = self.block5(x, pool_size=(1, 2)) # [B, 256, T', F']

        # 3. Temporal Rhythm 1D Stream (Group 3b)
        # Average pool across frequency axis -> [B, 256, T']
        time_profile = x.mean(dim=-1)
        rhythm_seq = self.rhythm_stream(time_profile) # [B, embed_dim, T']
        
        # Transpose to [B, T', embed_dim] for sequence cross-attention
        f_seq = rhythm_seq.permute(0, 2, 1)

        # Global temporal pooling
        f_rhythm = self.rhythm_proj(rhythm_seq.mean(dim=-1)) # [B, embed_dim]

        # Combined acoustic embedding
        f_audio = self.audio_proj(f_frequency + f_rhythm)    # [B, embed_dim]

        return f_audio, f_frequency, f_rhythm, f_seq


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
