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


class SpecAugment2D(nn.Module):
    """
    2D SpecAugment Module for High-Resolution TKEO-STFT Spectrograms [B, 2049, T].
    Encapsulates dedicated 2D augmentation techniques for spectral distributions:
      1. Frequency Masking: Masks random contiguous frequency bands with the sample's minimum energy.
      2. Time Masking: Masks random contiguous time frames with the sample's minimum energy.
      3. Gaussian Spectral Jitter: Simulates hydrophone sensor thermal and quantization noise.
    100% Vectorized on GPU (zero Python loop, zero device sync), active only during training.
    """
    def __init__(
        self,
        freq_mask_max: int = 32,
        time_mask_max: int = 16,
        freq_mask_prob: float = 0.5,
        time_mask_prob: float = 0.5,
        noise_std: float = 0.02
    ) -> None:
        super().__init__()
        self.freq_mask_max = int(freq_mask_max)
        self.time_mask_max = int(time_mask_max)
        self.freq_mask_prob = float(freq_mask_prob)
        self.time_mask_prob = float(time_mask_prob)
        self.noise_std = float(noise_std)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: STFT spectrogram [B, F=2049, T].
        Returns:
            Augmented spectrogram with the same shape [B, F, T].
        """
        if not self.training:
            return spec

        out = spec.clone()
        B, F, T = out.shape
        device = out.device
        min_vals = out.amin(dim=(1, 2), keepdim=True)  # [B, 1, 1]

        # 1. Frequency Masking (Vectorized across batch)
        if self.freq_mask_max > 0 and self.freq_mask_prob > 0.0 and F > self.freq_mask_max:
            mask_decisions = (torch.rand(B, 1, 1, device=device) < self.freq_mask_prob)
            if mask_decisions.any():
                widths = torch.randint(1, self.freq_mask_max + 1, (B, 1, 1), device=device)
                starts = torch.randint(0, F - self.freq_mask_max, (B, 1, 1), device=device)
                freq_indices = torch.arange(F, device=device).view(1, F, 1)
                f_mask = (freq_indices >= starts) & (freq_indices < starts + widths) & mask_decisions
                out = torch.where(f_mask, min_vals, out)

        # 2. Time Masking (Vectorized across batch)
        if self.time_mask_max > 0 and self.time_mask_prob > 0.0 and T > self.time_mask_max:
            mask_decisions = (torch.rand(B, 1, 1, device=device) < self.time_mask_prob)
            if mask_decisions.any():
                widths = torch.randint(1, self.time_mask_max + 1, (B, 1, 1), device=device)
                starts = torch.randint(0, T - self.time_mask_max, (B, 1, 1), device=device)
                time_indices = torch.arange(T, device=device).view(1, 1, T)
                t_mask = (time_indices >= starts) & (time_indices < starts + widths) & mask_decisions
                out = torch.where(t_mask, min_vals, out)

        # 3. Gaussian Spectral Jitter
        if self.noise_std > 0.0:
            noise = torch.randn_like(out) * self.noise_std
            out = out + noise

        return out


class AudioFrontend(nn.Module):
    """
    GPU-based High-Resolution TKEO-STFT Audio Frontend (256 kHz, 2049 frequency bins).
    Applies Teager-Kaiser Energy Operator (TKEO) Adaptive Pre-Emphasis, cuFFT RFFT,
    Log Magnitude, Temporal Mean Pooling, 1D Spectral Augmentation (Cutout & Jitter),
    and Layer Normalization to extract a 2049-dimensional spectral vector.
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

        self.use_spectral_aug = bool(getattr(self.config, 'use_spectral_aug', True))
        self.freq_mask_max = int(getattr(self.config, 'freq_mask_max', getattr(self.config, 'cutout_width', 32)))
        self.time_mask_max = int(getattr(self.config, 'time_mask_max', 16))
        self.freq_mask_prob = float(getattr(self.config, 'freq_mask_prob', getattr(self.config, 'cutout_prob', 0.5)))
        self.time_mask_prob = float(getattr(self.config, 'time_mask_prob', 0.5))
        self.noise_std = float(getattr(self.config, 'noise_std', 0.02))

        # Register Hann window buffer
        window = torch.hann_window(self.n_fft)
        self.register_buffer('window', window)

        # Dedicated 2D SpecAugment Module
        if self.use_spectral_aug:
            self.spectral_augmenter = SpecAugment2D(
                freq_mask_max=self.freq_mask_max,
                time_mask_max=self.time_mask_max,
                freq_mask_prob=self.freq_mask_prob,
                time_mask_prob=self.time_mask_prob,
                noise_std=self.noise_std
            )
        else:
            self.spectral_augmenter = None

        # Normalization layer over 2049 frequency bins per time frame
        self.norm = nn.LayerNorm(self.stft_bins)

        logger.info("==================================================")
        logger.info("Initialized TKEO-STFT Audio Frontend (256 kHz, 2D SpecAugment):")
        logger.info(f"  - Sample Rate:        {self.sample_rate} Hz (256 kHz)")
        logger.info(f"  - FFT Size (n_fft):   {self.n_fft}")
        logger.info(f"  - Hop Length:         {self.hop_length}")
        logger.info(f"  - STFT Output Bins:   {self.stft_bins} linear bins")
        logger.info(f"  - TKEO Pre-Emphasis:  {self.use_tkeo} (alpha_max={self.alpha_max})")
        logger.info(f"  - 2D SpecAugment:     {'ENABLED' if self.use_spectral_aug else 'DISABLED'}")
        if self.use_spectral_aug:
            logger.info(f"    * Frequency Mask:   Max={self.freq_mask_max} bins, Prob={self.freq_mask_prob}")
            logger.info(f"    * Time Mask:        Max={self.time_mask_max} frames, Prob={self.time_mask_prob}")
            logger.info(f"    * Gaussian Jitter:  Noise Std={self.noise_std}")
        logger.info("==================================================")

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor: Raw 1D audio waveform [Batch, Num_Samples].

        Returns:
            torch.Tensor: Normalized STFT 2D spectrogram [Batch, 2049, Time_Steps].
        """
        if input_tensor.ndim == 1:
            input_tensor = input_tensor.unsqueeze(0)
        elif input_tensor.ndim == 3 and input_tensor.size(1) == 1:
            input_tensor = input_tensor.squeeze(1)

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

        # 6. Channel-first format: [Batch, 2049, Time_Steps] for 1D-CNN
        spec_2d = log_mag.transpose(1, 2)

        # 7. 2D SpecAugment (Frequency & Time Masking + Jitter)
        if self.spectral_augmenter is not None:
            spec_2d = self.spectral_augmenter(spec_2d)

        # 8. Layer Normalization across 2049 frequency bins per time frame
        out = self.norm(spec_2d.transpose(1, 2)).transpose(1, 2)

        return out

