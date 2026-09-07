import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torchaudio.transforms as AT

from config.train_config import AudioFeaturesConfig

logger = logging.getLogger(__name__)


class SpecAugment(nn.Module):
    """
    Differentiable SpecAugment for Log-Mel Spectrograms.
    Applies frequency masking and time masking.
    """
    def __init__(
        self,
        freq_mask_param: int = 16,
        time_mask_param: int = 64,
        freq_masks: int = 2,
        time_masks: int = 2,
    ) -> None:
        super().__init__()
        self.freq_mask = AT.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = AT.TimeMasking(time_mask_param=time_mask_param)
        self.freq_masks = freq_masks
        self.time_masks = time_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, 1, T, F] or [B, T, F]
        for _ in range(self.freq_masks):
            x = self.freq_mask(x)
        for _ in range(self.time_masks):
            x = self.time_mask(x)
        return x


class AudioFrontend(nn.Module):
    """
    GPU-based Audio Frontend extracting 128 Mel-frequency Filterbanks.
    Converts raw 1D waveforms [B, num_samples] into Log-Mel Spectrograms [B, 1, T, 128].
    """
    def __init__(self, config: Optional[AudioFeaturesConfig] = None) -> None:
        super().__init__()
        if config is None:
            self.config = AudioFeaturesConfig()
        else:
            self.config = config

        self.sample_rate = self.config.sample_rate
        self.n_fft = self.config.window_size
        self.hop_length = self.config.hop_size
        self.n_mels = self.config.mel_bins
        self.f_min = self.config.fmin
        self.f_max = min(self.config.fmax, self.sample_rate // 2)

        self.mel_spectrogram = AT.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
        )

        self.spec_augment = SpecAugment(
            freq_mask_param=self.config.freq_drop_width,
            time_mask_param=self.config.time_drop_width,
            freq_masks=self.config.freq_stripes_num,
            time_masks=self.config.time_stripes_num,
        )

        # Normalization over Mel bins
        self.bn = nn.BatchNorm2d(self.n_mels)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: [B, num_samples] float32 tensor
        Returns:
            mel_spec: [B, 1, time_steps, n_mels] normalized log-mel spectrogram
        """
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)

        # 1. Mel Spectrogram: [B, n_mels, time_steps]
        mel = self.mel_spectrogram(waveforms)

        # 2. Log compression (dB scale)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))

        # 3. Transpose to [B, 1, time_steps, n_mels]
        log_mel = log_mel.unsqueeze(1).transpose(2, 3)  # [B, 1, time_steps, n_mels]

        # 4. BatchNorm normalization along n_mels axis
        # Permute for BatchNorm2d (expects [B, C, H, W] where C is n_mels)
        # [B, 1, T, F] -> [B, F, T, 1]
        x_bn = log_mel.permute(0, 3, 2, 1)
        x_bn = self.bn(x_bn)
        log_mel = x_bn.permute(0, 3, 2, 1)  # back to [B, 1, T, F]

        # 5. SpecAugment during training
        if self.training:
            # SpecAugment expects [..., freq, time]
            x_aug = log_mel.transpose(-2, -1)  # [B, 1, F, T]
            x_aug = self.spec_augment(x_aug)
            log_mel = x_aug.transpose(-2, -1)  # [B, 1, T, F]

        return log_mel
