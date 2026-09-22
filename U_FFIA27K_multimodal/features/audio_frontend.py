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


class UnderwaterHydroAcousticAugmenter(nn.Module):
    """
    2D Time-Frequency Spectral Augmentation Module for Underwater Hydrophone Audio [B, 1, T=251, F=2049].
    Encapsulates physically grounded augmentation techniques:
      1. Subband Frequency Masking: Masks a narrow contiguous frequency band (cutout_width bins)
         with the sample's minimum energy (noise floor) instead of 0.0 to prevent energy explosion.
      2. Circular Time Shift: Simulates arbitrary clip start time (up to max_time_shift frames).
      3. Gaussian Sensor Jitter: Simulates hydrophone sensor thermal and quantization noise.
    """
    def __init__(
        self,
        cutout_width: int = 16,
        cutout_prob: float = 0.5,
        max_time_shift: int = 15,
        noise_std: float = 0.015
    ) -> None:
        super().__init__()
        self.cutout_width = int(cutout_width)
        self.cutout_prob = float(cutout_prob)
        self.max_time_shift = int(max_time_shift)
        self.noise_std = float(noise_std)

    def forward(self, spec_2d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec_2d: 2D Spectrogram tensor [B, 1, T, F] or [B, T, F].
        Returns:
            torch.Tensor: Augmented spectrogram with identical shape.
        """
        if not self.training:
            return spec_2d

        out = spec_2d.clone()
        B = out.size(0)
        T = out.size(-2)
        F_dim = out.size(-1)

        # 1. Subband Frequency Masking (Narrow band cutout with min energy)
        if self.cutout_width > 0 and self.cutout_prob > 0.0 and F_dim > self.cutout_width:
            mask_decisions = (torch.rand(B, 1, 1, 1, device=out.device) < self.cutout_prob)
            if mask_decisions.any():
                start_indices = torch.randint(
                    0, F_dim - self.cutout_width, (B, 1, 1, 1), device=out.device
                )
                freq_indices = torch.arange(F_dim, device=out.device).view(1, 1, 1, F_dim)
                cutout_mask = (freq_indices >= start_indices) & (freq_indices < start_indices + self.cutout_width) & mask_decisions
                min_vals = out.amin(dim=(-2, -1), keepdim=True)
                out = torch.where(cutout_mask, min_vals, out)

        # 2. Circular Time Shift (Vectorized per-sample circular roll)
        if self.max_time_shift > 0 and T > self.max_time_shift * 2:
            shifts = torch.randint(-self.max_time_shift, self.max_time_shift + 1, (B,), device=out.device)
            for i in range(B):
                shift = int(shifts[i].item())
                if shift != 0:
                    out[i] = torch.roll(out[i], shifts=shift, dims=-2)

        # 3. Gaussian Sensor Noise Jitter
        if self.noise_std > 0.0:
            noise = torch.randn_like(out) * self.noise_std
            out = out + noise

        return out


# Backward compatibility alias
Spectral1DAugmentation = UnderwaterHydroAcousticAugmenter


class AudioFrontend(nn.Module):
    """
    GPU-based High-Resolution TKEO-STFT Audio Frontend (256 kHz, 2049 frequency bins).
    Applies Teager-Kaiser Energy Operator (TKEO) Adaptive Pre-Emphasis, cuFFT RFFT,
    Log Magnitude, and 2D Layer Normalization to extract full 2D Spectrograms [Batch, 1, Time_Steps, 2049].
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
        self.cutout_width = int(getattr(self.config, 'cutout_width', 16))
        self.cutout_prob = float(getattr(self.config, 'cutout_prob', 0.5))
        self.max_time_shift = int(getattr(self.config, 'max_time_shift', 15))
        self.noise_std = float(getattr(self.config, 'noise_std', 0.015))

        # Register Hann window buffer
        window = torch.hann_window(self.n_fft)
        self.register_buffer('window', window)

        # Dedicated 2D Hydrophone Spectral Augmentation Module
        if self.use_spectral_aug:
            self.spectral_augmenter = UnderwaterHydroAcousticAugmenter(
                cutout_width=self.cutout_width,
                cutout_prob=self.cutout_prob,
                max_time_shift=self.max_time_shift,
                noise_std=self.noise_std
            )
        else:
            self.spectral_augmenter = None

        # Normalization layer over 2049 frequency bins
        self.norm = nn.LayerNorm(self.stft_bins)

        logger.info("==================================================")
        logger.info("Initialized TKEO-STFT Audio Frontend (256 kHz, 2D Spectrogram):")
        logger.info(f"  - Sample Rate:        {self.sample_rate} Hz (256 kHz)")
        logger.info(f"  - FFT Size (n_fft):   {self.n_fft}")
        logger.info(f"  - Hop Length:         {self.hop_length}")
        logger.info(f"  - STFT Output Bins:   {self.stft_bins} linear bins")
        logger.info(f"  - TKEO Pre-Emphasis:  {self.use_tkeo} (alpha_max={self.alpha_max})")
        logger.info(f"  - 2D Spectral Aug:    {'ENABLED' if self.use_spectral_aug else 'DISABLED'}")
        if self.use_spectral_aug:
            logger.info(f"    * Cutout Band Width:{self.cutout_width} bins (Prob={self.cutout_prob})")
            logger.info(f"    * Circular Shift:   +/-{self.max_time_shift} frames")
            logger.info(f"    * Gaussian Noise:   Std={self.noise_std}")
        logger.info("==================================================")

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor: Raw 1D audio waveform [Batch, Num_Samples].

        Returns:
            torch.Tensor: Normalized 2D STFT spectrogram [Batch, 1, Time_Steps, 2049].
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
        if self.window.device != frames.device or self.window.dtype != frames.dtype:
            self.window = self.window.to(device=frames.device, dtype=frames.dtype)
        frames_win = frames * self.window
        complex_spec = torch.fft.rfft(frames_win, n=self.n_fft, dim=-1)

        # 5. Log Magnitude: log(|X| + 1e-8) -> [Batch, Time_Steps, 2049]
        log_mag = torch.log(torch.abs(complex_spec) + 1e-8)

        # 6. Layer Normalization across 2049 frequency bins
        log_mag_norm = self.norm(log_mag)

        # 7. Channel dimension expansion -> [Batch, 1, Time_Steps, 2049]
        spec_2d = log_mag_norm.unsqueeze(1)

        # 8. 2D Hydrophone Spectral Augmentation (Cutout, Circular Shift, Noise Jitter)
        if self.spectral_augmenter is not None:
            spec_2d = self.spectral_augmenter(spec_2d)

        return spec_2d
