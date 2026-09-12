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


class Spectral1DAugmentation(nn.Module):
    """
    1D Spectral Augmentation Module for High-Resolution TKEO-STFT vectors [B, 2049].
    Encapsulates dedicated 1D augmentation techniques for spectral distributions:
      1. 1D Frequency Cutout: Masks a narrow contiguous frequency band (cutout_width bins)
         with the sample's minimum energy (noise floor) instead of 0.0 to prevent energy explosion.
      2. Gaussian Spectral Jitter: Simulates hydrophone sensor thermal and quantization noise.
    """
    def __init__(
        self,
        cutout_width: int = 24,
        cutout_prob: float = 0.5,
        noise_std: float = 0.02
    ) -> None:
        super().__init__()
        self.cutout_width = int(cutout_width)
        self.cutout_prob = float(cutout_prob)
        self.noise_std = float(noise_std)

    def forward(self, spec_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec_vector: STFT spectral energy vector [B, 2049] or [2049].
        Returns:
            Augmented spectral vector with the same shape.
        """
        if not self.training:
            return spec_vector

        is_1d = (spec_vector.ndim == 1)
        out = spec_vector.unsqueeze(0).clone() if is_1d else spec_vector.clone()
        B, F = out.shape

        # 1. 1D Frequency Cutout
        if self.cutout_width > 0 and self.cutout_prob > 0.0 and F > self.cutout_width:
            mask_decisions = torch.rand(B, device=out.device) < self.cutout_prob
            if mask_decisions.any():
                start_indices = torch.randint(
                    0, F - self.cutout_width, (B,), device=out.device
                )
                min_vals = out.min(dim=-1, keepdim=True)[0]
                for b in range(B):
                    if mask_decisions[b]:
                        s = start_indices[b]
                        out[b, s : s + self.cutout_width] = min_vals[b]

        # 2. Gaussian Spectral Jitter
        if self.noise_std > 0.0:
            noise = torch.randn_like(out) * self.noise_std
            out = out + noise

        if is_1d:
            out = out.squeeze(0)
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
        self.cutout_width = int(getattr(self.config, 'cutout_width', 24))
        self.cutout_prob = float(getattr(self.config, 'cutout_prob', 0.5))
        self.noise_std = float(getattr(self.config, 'noise_std', 0.02))

        # Register Hann window buffer
        window = torch.hann_window(self.n_fft)
        self.register_buffer('window', window)

        # Dedicated 1D Spectral Augmentation Module
        if self.use_spectral_aug:
            self.spectral_augmenter = Spectral1DAugmentation(
                cutout_width=self.cutout_width,
                cutout_prob=self.cutout_prob,
                noise_std=self.noise_std
            )
        else:
            self.spectral_augmenter = None

        # Normalization layer over 2049 frequency bins
        self.norm = nn.LayerNorm(self.stft_bins)

        logger.info("==================================================")
        logger.info("Initialized TKEO-STFT Audio Frontend (256 kHz, 1D Spectral Augmentation):")
        logger.info(f"  - Sample Rate:        {self.sample_rate} Hz (256 kHz)")
        logger.info(f"  - FFT Size (n_fft):   {self.n_fft}")
        logger.info(f"  - Hop Length:         {self.hop_length}")
        logger.info(f"  - STFT Output Bins:   {self.stft_bins} linear bins")
        logger.info(f"  - TKEO Pre-Emphasis:  {self.use_tkeo} (alpha_max={self.alpha_max})")
        logger.info(f"  - Spectral 1D Aug:    {'ENABLED' if self.use_spectral_aug else 'DISABLED'}")
        if self.use_spectral_aug:
            logger.info(f"    * 1D Cutout Band:   Width={self.cutout_width} bins, Prob={self.cutout_prob}")
            logger.info(f"    * Gaussian Jitter:  Noise Std={self.noise_std}")
        logger.info("==================================================")

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor: Raw 1D audio waveform [Batch, Num_Samples].

        Returns:
            torch.Tensor: Normalized STFT spectral vector [Batch, 2049].
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

        # 5. Log Magnitude: log(|X| + 1e-8)
        log_mag = torch.log(torch.abs(complex_spec) + 1e-8)

        # 6. Mean over time axis -> [Batch, 2049]
        spec_vector = log_mag.mean(dim=1)

        # 7. 1D Spectral Augmentation for MLP (Cutout & Jitter)
        if self.spectral_augmenter is not None:
            spec_vector = self.spectral_augmenter(spec_vector)

        # 8. Layer Normalization
        out = self.norm(spec_vector)

        return out

