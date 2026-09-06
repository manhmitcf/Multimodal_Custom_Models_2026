import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from features.stft_ape import Spectrogram_APE


# Import centralized configuration from config package
from config import AudioFeaturesConfig as AudioFrontendConfig

# Ensure stdout/stderr UTF-8 encoding on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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
    GPU-based raw audio waveform preprocessing frontend utilizing torchlibrosa.
    """
    def __init__(self, config: Optional[AudioFrontendConfig] = None) -> None:
        """
        Initialize AudioFrontend module.
        """
        super(AudioFrontend, self).__init__()

        # If config is None, load dynamically from centralized TrainConfig
        if config is None:
            from config import TrainConfig
            self.config = TrainConfig.from_json().audio_features
        else:
            self.config = config

        self.mel_bins = self.config.mel_bins
        self.frontend_type = getattr(self.config, 'frontend_type', 'stft_gem').lower()
        self.gem_p = float(getattr(self.config, 'gem_p', 3.0))

        use_tkeo = getattr(self.config, 'use_tkeo', True)
        alpha_max = getattr(self.config, 'alpha_max', 0.99)
        beta = getattr(self.config, 'beta', 0.8)

        # 1. Amplitude Spectrogram Extractor with TKEO Adaptive Pre-Emphasis
        if use_tkeo:
            logger.info("Enabling Teager-Kaiser Energy Operator (TKEO) Adaptive Pre-Emphasis Spectrogram Extractor.")
            self.spectrogram_extractor = Spectrogram_APE(
                n_fft=self.config.window_size,
                hop_length=self.config.hop_size,
                win_length=self.config.window_size,
                window='hann',
                center=True,
                pad_mode='reflect',
                freeze_parameters=True,
                alpha_max=alpha_max,
                beta=beta
            )
        else:
            self.spectrogram_extractor = Spectrogram(
                n_fft=self.config.window_size,
                hop_length=self.config.hop_size,
                win_length=self.config.window_size,
                window='hann',
                center=True,
                pad_mode='reflect',
                freeze_parameters=True
            )

        # 2. Spectral Compression (STFT with GeM Pooling vs Mel Filterbank)
        fmax = min(self.config.fmax, self.config.sample_rate // 2)
        if self.frontend_type == "mel":
            self.logmel_extractor = LogmelFilterBank(
                sr=self.config.sample_rate,
                n_fft=self.config.window_size,
                n_mels=self.config.mel_bins,
                fmin=self.config.fmin,
                fmax=fmax,
                ref=1.0,
                amin=1e-10,
                top_db=None,
                freeze_parameters=True
            )
        else:
            self.logmel_extractor = None

        # 3. SpecAugment Spec Augmentation Extractor on GPU using torchlibrosa
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=getattr(self.config, 'time_drop_width', 64),
            time_stripes_num=getattr(self.config, 'time_stripes_num', 2),
            freq_drop_width=getattr(self.config, 'freq_drop_width', 8),
            freq_stripes_num=getattr(self.config, 'freq_stripes_num', 2)
        )

        # 4. BatchNorm normalization layer
        self.bn0 = nn.BatchNorm2d(self.mel_bins)
        init_bn(self.bn0)

        logger.info("==================================================")
        logger.info("Initialized AudioFrontend module on GPU:")
        logger.info(f"  - Frontend Representation:  {self.frontend_type.upper()} (GeM p={self.gem_p} Peak-Preserving)" if self.frontend_type != "mel" else "  - Frontend Representation:  MEL FILTERBANK")
        logger.info(f"  - Sample Rate:              {self.config.sample_rate} Hz (128 kHz)")
        logger.info(f"  - Window Size:              {self.config.window_size} (16 ms)")
        logger.info(f"  - Hop Size:                 {self.config.hop_size} (8 ms)")
        logger.info(f"  - Frequency Bins:           {self.config.mel_bins} (500 Hz/bin linear resolution)")
        logger.info(f"  - Fmin/Fmax:                {self.config.fmin} / {fmax} Hz")
        logger.info(f"  - TKEO Adaptive Pre-Emph:   {'ENABLED (alpha_max=' + str(alpha_max) + ', beta=' + str(beta) + ')' if use_tkeo else 'DISABLED'}")
        logger.info(f"  - SpecAugment Time Masking: Width={self.config.time_drop_width}, Stripes={self.config.time_stripes_num}")
        logger.info(f"  - SpecAugment Freq Masking: Width={self.config.freq_drop_width}, Stripes={self.config.freq_stripes_num}")
        logger.info("==================================================")

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass converting raw 1D waveforms into 2D linear-frequency STFT GeM spectrograms.

        Args:
            input_tensor (torch.Tensor): Raw waveform tensor [Batch, Num_Samples].

        Returns:
            torch.Tensor: Augmented Spectrogram tensor [Batch, 1, Time_Steps + 2, 128].
        """
        # Step A: Raw 1D Waveform -> STFT 2D Spectrogram [Batch, 1, Time_Steps, 1025] (via TKEO APE)
        x = self.spectrogram_extractor(input_tensor)
        
        # Step B: Time-frequency compression to 128 bins
        if self.frontend_type == "mel" and self.logmel_extractor is not None:
            x = self.logmel_extractor(x)
        else:
            # Linear STFT with GeM (p=3.0) Peak-Preserving Compression:
            # Slice 1024 bins (bins 1 to 1025, covering 62.5 Hz to 64,000 Hz)
            # Exactly 8 STFT bins per output band -> 500 Hz per band uniform linear spacing!
            s = x[:, :, :, 1:1025] # [Batch, 1, Time_Steps, 1024]
            s_p = torch.clamp(s, min=1e-8) ** self.gem_p
            gem_down = F.avg_pool2d(s_p, kernel_size=(1, 8), stride=(1, 8))
            x_gem = torch.clamp(gem_down, min=1e-8) ** (1.0 / self.gem_p)
            x = torch.log(torch.clamp(x_gem, min=1e-10))

        # Step C: Pad time-steps dimension by 2 rows of zeros for shape alignment
        m = nn.ZeroPad2d((0, 0, 2, 0))
        x = m(x)

        # Step D: Transpose for BatchNorm2d along frequency bins axis
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)  # Result shape: [Batch, 1, Time_Steps + 2, 128]

        # Step E: Apply SpecAugment masking during training
        if self.training:
            x = self.spec_augmenter(x)

        return x
