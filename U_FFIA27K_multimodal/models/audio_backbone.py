import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class EfficientATAudioBackbone(nn.Module):
    """
    Efficient Acoustic Backbone inspired by EfficientAT (Schmid et al., Interspeech 2023).
    Processes 128 Mel-frequency filterbank spectrograms [B, 1, Ta, 128]:
      - Stage 1: MobileNetV3-based multi-scale feature extractor with 1024-d conv head (~1.52M params)
      - Stage 2: Spectral Squeeze-and-Excitation (prioritizing 2-8 kHz feeding splash band)
      - Stage 3: Depthwise-separable 1D Temporal Rhythm Convolutions (splashing temporal impulses)
      - Total parameters: ~1.82M params.
    """
    def __init__(self, embed_dim: int = 224, pretrained: bool = False) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        try:
            import timm
            self.backbone = timm.create_model(
                'mobilenetv3_small_100',
                pretrained=pretrained,
                in_chans=1,
                num_classes=0
            )
        except Exception as exc:
            raise ImportError(f"timm is required for EfficientATAudioBackbone: {exc}")

        # Projections to unified multimodal embedding space from 1024-d conv head
        self.spatial_proj = nn.Sequential(
            nn.Linear(1024, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Frequency Squeeze-and-Excitation (focus on splash frequency band 2-8 kHz)
        self.freq_se = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 4, embed_dim),
            nn.Sigmoid()
        )

        # Depthwise-separable rhythm extractor (temporal cadence of feeding splashes)
        self.rhythm_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm1d(embed_dim)
        )

        # Residual normalization
        self.norm_audio = nn.LayerNorm(embed_dim)

    def forward(self, mel_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mel_spec: Log-Mel Spectrogram [B, 1, Ta, 128]

        Returns:
            f_audio: Joint acoustic embedding [B, embed_dim]
            f_frequency: Spectral frequency distribution feature [B, embed_dim]
            f_rhythm: Temporal rhythm cadence feature [B, embed_dim]
            tokens_audio: Sequence of temporal audio tokens [B, T', embed_dim] for MBT Bottleneck Fusion
        """
        # Feature extraction: [B, 1, Ta, 128] -> [B, 576, T', F'] -> conv_head -> [B, 1024, T', F']
        feat = self.backbone.forward_features(mel_spec)
        if hasattr(self.backbone, 'conv_head'):
            feat = self.backbone.conv_head(feat)
        if hasattr(self.backbone, 'act2'):
            feat = self.backbone.act2(feat)

        B, C, T_prime, F_prime = feat.shape

        # 1. Frequency Profile (Pool over Time T'): [B, C, F'] -> [B, C]
        freq_pool = feat.mean(dim=2).mean(dim=-1)  # [B, 1024]
        f_freq_raw = self.spatial_proj(freq_pool)  # [B, embed_dim]
        w_freq = self.freq_se(f_freq_raw)
        f_frequency = f_freq_raw * w_freq

        # 2. Temporal Rhythm Sequence (Pool over Frequency F'): [B, C, T', F'] -> [B, C, T']
        temporal_seq = feat.mean(dim=-1)  # [B, 1024, T']
        # Project channel dim: [B, T', 1024] -> [B, T', embed_dim]
        temporal_seq_proj = self.spatial_proj(temporal_seq.transpose(1, 2)).transpose(1, 2)  # [B, embed_dim, T']
        rhythm_seq = self.rhythm_conv(temporal_seq_proj)  # [B, embed_dim, T']

        tokens_audio = rhythm_seq.transpose(1, 2)  # [B, T', embed_dim]
        f_rhythm = tokens_audio.mean(dim=1)        # [B, embed_dim]

        # 3. Joint Acoustic Vector fusing frequency profile and temporal rhythm
        f_audio = self.norm_audio(f_frequency + f_rhythm)

        return f_audio, f_frequency, f_rhythm, tokens_audio
