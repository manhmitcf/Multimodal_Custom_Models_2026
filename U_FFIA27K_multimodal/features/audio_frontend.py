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
import torch.nn.functional as F

from config.train_config import AudioFeaturesConfig

logger = logging.getLogger(__name__)


class AudioFrontendOutput(dict):
    """
    Structured dictionary and attribute container for Dual-Axis Audio Frontend outputs:
      - spectrogram: 2D Time-Frequency Log-Magnitude Spectrogram [Batch, 1, Time_Steps, 2049]
      - spec_vector: Normalized spectral vector across 2049 STFT bins [Batch, 2049]
      - temporal_energy: 5-band temporal energy sequence [Batch, 5, Time_Steps]
    """
    def __init__(
        self,
        spectrogram: torch.Tensor,
        spec_vector: torch.Tensor,
        temporal_energy: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__(spectrogram=spectrogram, spec_vector=spec_vector, temporal_energy=temporal_energy)
        self.spectrogram = spectrogram
        self.spec_vector = spec_vector
        self.temporal_energy = temporal_energy


class AudioFrontend(nn.Module):
    """
    GPU-based High-Resolution Dual-Axis TKEO-STFT Audio Frontend (256 kHz, 2049 frequency bins).
    Applies Teager-Kaiser Energy Operator (TKEO) Adaptive Pre-Emphasis, cuFFT RFFT,
    Log Magnitude, and extracts:
      1. spectrogram [Batch, 1, Time_Steps, 2049]: 2D Time-Frequency representation for Factorized Dual-Axis ConvNet.
      2. spec_vector [Batch, 2049]: Mean spectral density across time.
      3. temporal_energy [Batch, 5, Time_Steps]: Multi-track temporal energy profile across 5 physical bands.
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
        self.stft_bins = self.n_fft // 2 + 1  # 2049 for n_fft=4096
        self.alpha_max = float(getattr(self.config, 'alpha_max', 0.99))
        self.use_tkeo = bool(getattr(self.config, 'use_tkeo', True))

        # Register Hann window buffer
        window = torch.hann_window(self.n_fft)
        self.register_buffer('window', window)

        # Normalization layer over 2049 frequency bins
        self.norm = nn.LayerNorm(self.stft_bins)

        logger.info("==================================================")
        logger.info("Initialized TKEO-STFT Dual-Axis Audio Frontend (256 kHz):")
        logger.info(f"  - Sample Rate:        {self.sample_rate} Hz (256 kHz)")
        logger.info(f"  - FFT Size (n_fft):   {self.n_fft}")
        logger.info(f"  - Hop Length:         {self.hop_length}")
        logger.info(f"  - STFT Output Bins:   {self.stft_bins} linear bins")
        logger.info(f"  - TKEO Pre-Emphasis:  {self.use_tkeo} (alpha_max={self.alpha_max})")
        logger.info(f"  - Output 1:           spectrogram [B, 1, Time_Steps, 2049]")
        logger.info(f"  - Output 2:           spec_vector [B, 2049]")
        logger.info("==================================================")

    def forward(self, input_tensor: torch.Tensor) -> AudioFrontendOutput:
        """
        Args:
            input_tensor: Raw 1D audio waveform [Batch, Num_Samples].

        Returns:
            AudioFrontendOutput containing:
              - spectrogram: Full 2D spectrogram [Batch, 1, Time_Steps, 2049].
              - spec_vector: Normalized STFT spectral vector [Batch, 2049].
              - temporal_energy: 5-band temporal energy sequence [Batch, 5, Time_Steps].
        """
        if input_tensor.ndim == 1:
            input_tensor = input_tensor.unsqueeze(0)

        # 1. Padding for center alignment
        pad_amt = self.n_fft // 2
        x_padded = F.pad(input_tensor, (pad_amt, pad_amt), mode='reflect')

        # 2. Framing (sliding window) -> [Batch, Time_Steps, n_fft]
        frames = x_padded.unfold(dimension=-1, size=self.n_fft, step=self.hop_length)

        # 3. Vectorized TKEO Adaptive Pre-Emphasis
        if self.use_tkeo and self.alpha_max > 0:
            x_mid = frames[:, :, 1:-1]
            x_left = frames[:, :, :-2]
            x_right = frames[:, :, 2:]
            psi = x_mid**2 - x_left * x_right
            psi_full = torch.cat([psi[:, :, :1], psi, psi[:, :, -1:]], dim=-1)

            mean_psi = torch.mean(torch.abs(psi_full), dim=-1, keepdim=True)
            mean_energy = torch.mean(frames**2, dim=-1, keepdim=True)
            ctrl = mean_psi / (mean_energy + 1e-10)

            alpha = self.alpha_max * (1.0 - torch.exp(-ctrl))
            alpha = torch.clamp(alpha, min=0.1, max=self.alpha_max)

            frames_prev = torch.cat([frames[:, :, :1], frames[:, :, :-1]], dim=-1)
            frames = frames - alpha * frames_prev

        # 4. Windowing & cuFFT Real FFT -> [Batch, Time_Steps, 2049]
        if self.window.device != frames.device:
            self.window = self.window.to(frames.device)
        frames_win = frames * self.window
        complex_spec = torch.fft.rfft(frames_win, n=self.n_fft, dim=-1)

        # 5. Log Magnitude: log(|X| + 1e-8) -> [Batch, Time_Steps, 2049]
        log_mag = torch.log(torch.abs(complex_spec) + 1e-8)
        log_mag_norm = self.norm(log_mag)

        # Full 2D spectrogram [Batch, 1, Time_Steps, 2049]
        spectrogram = log_mag_norm.unsqueeze(1)

        # 6. Mean over time axis -> [Batch, 2049]
        spec_vector = log_mag_norm.mean(dim=1)

        # 7. Extract 5-Band Temporal Energy Sequence -> [Batch, 5, Time_Steps]
        e1 = log_mag[:, :, 0:80].mean(dim=-1)
        e2 = log_mag[:, :, 80:320].mean(dim=-1)
        e3 = log_mag[:, :, 320:720].mean(dim=-1)
        e4 = log_mag[:, :, 720:1360].mean(dim=-1)
        e5 = log_mag[:, :, 1360:2049].mean(dim=-1)
        temporal_energy = torch.stack([e1, e2, e3, e4, e5], dim=1)

        return AudioFrontendOutput(
            spectrogram=spectrogram,
            spec_vector=spec_vector,
            temporal_energy=temporal_energy
        )
