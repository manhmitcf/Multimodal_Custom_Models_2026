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
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from features.stft_ape import Spectrogram_APE

from config.train_config import AudioFeaturesConfig

logger = logging.getLogger(__name__)


def init_bn(bn: nn.BatchNorm2d) -> None:
    """
    Initialize BatchNorm2d weights with default values (bias = 0, weight = 1).
    """
    if bn.bias is not None:
        bn.bias.data.fill_(0.)
    if bn.weight is not None:
        bn.weight.data.fill_(1.)


class AudioFrontend(nn.Module):
    """
    GPU-based Audio Frontend extracting 128 Mel-frequency Filterbanks.
    Enhanced with Teager-Kaiser Energy Operator (TKEO) Adaptive Pre-Emphasis:
      Psi[x(n)] = x^2(n) - x(n-1) * x(n+1)
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

        use_tkeo = getattr(self.config, 'use_tkeo', True)
        alpha_max = getattr(self.config, 'alpha_max', 0.99)
        beta = getattr(self.config, 'beta', 0.8)

        # 1. Amplitude Spectrogram Extractor with TKEO Adaptive Pre-Emphasis
        if use_tkeo:
            logger.info("Enabling Teager-Kaiser Energy Operator (TKEO) Adaptive Pre-Emphasis Spectrogram Extractor.")
            self.spectrogram_extractor = Spectrogram_APE(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window='hann',
                center=True,
                pad_mode='reflect',
                freeze_parameters=True,
                alpha_max=alpha_max,
                beta=beta
            )
        else:
            self.spectrogram_extractor = Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window='hann',
                center=True,
                pad_mode='reflect',
                freeze_parameters=True
            )

        # 2. Logmel Filterbank Extractor on GPU using torchlibrosa
        self.logmel_extractor = LogmelFilterBank(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True
        )

        # 3. SpecAugment Spec Augmentation Extractor on GPU using torchlibrosa
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=getattr(self.config, 'time_drop_width', 64),
            time_stripes_num=getattr(self.config, 'time_stripes_num', 2),
            freq_drop_width=getattr(self.config, 'freq_drop_width', 8),
            freq_stripes_num=getattr(self.config, 'freq_stripes_num', 2)
        )

        # 4. BatchNorm normalization layer over Mel bins
        self.bn0 = nn.BatchNorm2d(self.n_mels)
        init_bn(self.bn0)

        # 5. Convert non-trainable DFT kernels and Mel filterbanks from Parameters to Buffers
        # so they are properly treated as constant Fourier basis functions (0 trainable parameters)
        for _, m in self.named_modules():
            for p_name, p in list(m.named_parameters(recurse=False)):
                if not p.requires_grad:
                    delattr(m, p_name)
                    m.register_buffer(p_name, p.data)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass converting raw 1D waveforms into 2D Mel-spectrograms.

        Args:
            input_tensor (torch.Tensor): Raw waveform tensor [Batch, Num_Samples].

        Returns:
            torch.Tensor: Augmented Log-Mel Spectrogram [Batch, 1, Time_Steps + 2, 128].
        """
        if input_tensor.ndim == 1:
            input_tensor = input_tensor.unsqueeze(0)

        # Step A: Raw 1D Waveform -> STFT 2D Spectrogram [Batch, 1, Time_Steps, Freq_Bins] via TKEO APE
        x = self.spectrogram_extractor(input_tensor)

        # Step B: Logmel filtering -> [Batch, 1, Time_Steps, Mel_Bins]
        x = self.logmel_extractor(x)

        # Step C: Pad time-steps dimension by 2 rows of zeros for shape alignment
        m = nn.ZeroPad2d((0, 0, 2, 0))
        x = m(x)

        # Step D: Transpose for BatchNorm2d along mel bins axis
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)  # Result shape: [Batch, 1, Time_Steps + 2, 128]

        # Step E: Apply SpecAugment masking during training
        if self.training:
            x = self.spec_augmenter(x)

        return x
