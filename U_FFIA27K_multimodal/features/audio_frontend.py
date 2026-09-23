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


class AdvancedSpectral1DAugmentation(nn.Module):
    """
    GPU-accelerated 1D Spectral Augmentation for High-Resolution STFT representations [B, C, F] or [B, F].
    Implements 4 physics-grounded bioacoustic transformations:
      1. Dual-Band Frequency Masking (SpecAugment 1D, Park et al., 2019):
         Masks 2 non-overlapping narrow frequency bands with the sample's noise floor.
      2. Spectral Tilt / Transmission Loss (Salamon & Bello, 2017; Thorpe's Equation):
         Modulates the frequency slope to simulate hydrophone-to-fish distance variations.
      3. Frequency Micro-Shift (Salamon & Bello, 2017):
         Rolls frequency bins slightly (+/- max_shift bins) to simulate water temperature & fish size variance.
      4. Additive Gaussian Spectral Jitter (Nanni et al., 2020):
         Simulates hydrophone sensor thermal and ADC quantization noise.
    """
    def __init__(
        self,
        cutout_width: int = 20,
        cutout_prob: float = 0.5,
        tilt_max: float = 0.05,
        max_shift: int = 12,
        noise_std: float = 0.015
    ) -> None:
        super().__init__()
        self.cutout_width = int(cutout_width)
        self.cutout_prob = float(cutout_prob)
        self.tilt_max = float(tilt_max)
        self.max_shift = int(max_shift)
        self.noise_std = float(noise_std)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: Spectral tensor [B, C, F] or [B, F].
        Returns:
            Augmented spectral tensor with identical shape and dtype.
        """
        if not self.training:
            return spec

        is_2d = (spec.ndim == 2)
        out = spec.unsqueeze(1).clone() if is_2d else spec.clone()
        B, C, F = out.shape

        # 1. Spectral Tilt: w(f) = 1.0 + beta * (2f / F - 1.0)
        if self.tilt_max > 0.0:
            beta = (torch.rand(B, 1, 1, device=out.device) * 2.0 - 1.0) * self.tilt_max
            freq_axis = torch.linspace(-1.0, 1.0, F, device=out.device).view(1, 1, F)
            tilt_weights = 1.0 + beta * freq_axis
            out = out * tilt_weights

        # 2. Frequency Micro-Shift: roll slightly along frequency axis
        if self.max_shift > 0:
            shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
            if shift != 0:
                out = torch.roll(out, shifts=shift, dims=-1)
                if shift > 0:
                    out[:, :, :shift] = out[:, :, shift:shift + 1]
                else:
                    out[:, :, shift:] = out[:, :, shift - 1:shift]

        # 3. Dual-Band Frequency Masking (2 independent narrow bands)
        if self.cutout_width > 0 and self.cutout_prob > 0.0 and F > self.cutout_width:
            min_floor = out.amin(dim=-1, keepdim=True)
            for _ in range(2):
                mask_decision = (torch.rand(B, 1, 1, device=out.device) < self.cutout_prob)
                if mask_decision.any():
                    starts = torch.randint(0, F - self.cutout_width, (B, 1, 1), device=out.device)
                    f_idx = torch.arange(F, device=out.device).view(1, 1, F)
                    band_mask = (f_idx >= starts) & (f_idx < starts + self.cutout_width) & mask_decision
                    out = torch.where(band_mask, min_floor, out)

        # 4. Additive Gaussian Spectral Jitter
        if self.noise_std > 0.0:
            out = out + torch.randn_like(out) * self.noise_std

        if is_2d:
            out = out.squeeze(1)
        return out


# Canonical Alias for backward compatibility
Spectral1DAugmentation = AdvancedSpectral1DAugmentation


class AudioFrontend(nn.Module):
    """
    GPU-based High-Resolution TKEO-STFT Audio Frontend (256 kHz, Dual-Channel Spectral Profile).
    Extracts a 2-channel 2049-bin spectral representation [B, 2, 2049]:
      - Channel 0: Stationary Power Spectral Density (Temporal Mean PSD)
      - Channel 1: Transient Cavitation / Feeding Burst Contrast (Temporal Max - Mean PSD)
    Applies Teager-Kaiser Energy Operator (TKEO) Adaptive Pre-Emphasis, cuFFT RFFT,
    Log Magnitude, Dual-Channel Pooling, and Advanced 1D Spectral Augmentation.
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
        self.cutout_width = int(getattr(self.config, 'cutout_width', 20))
        self.cutout_prob = float(getattr(self.config, 'cutout_prob', 0.5))
        self.tilt_max = float(getattr(self.config, 'tilt_max', 0.05))
        self.max_shift = int(getattr(self.config, 'max_shift', 12))
        self.noise_std = float(getattr(self.config, 'noise_std', 0.015))

        # Register Hann window buffer
        window = torch.hann_window(self.n_fft)
        self.register_buffer('window', window)

        # Advanced 1D Spectral Augmentation Module
        if self.use_spectral_aug:
            self.spectral_augmenter = AdvancedSpectral1DAugmentation(
                cutout_width=self.cutout_width,
                cutout_prob=self.cutout_prob,
                tilt_max=self.tilt_max,
                max_shift=self.max_shift,
                noise_std=self.noise_std
            )
        else:
            self.spectral_augmenter = None

        # Per-channel Layer Normalization over 2049 frequency bins
        self.norm_mean = nn.LayerNorm(self.stft_bins)
        self.norm_peak = nn.LayerNorm(self.stft_bins)

        logger.info("==================================================")
        logger.info("Initialized TKEO-STFT Audio Frontend (256 kHz, Dual-Channel 1D ConvNeXt Profile):")
        logger.info(f"  - Sample Rate:        {self.sample_rate} Hz (256 kHz)")
        logger.info(f"  - FFT Size (n_fft):   {self.n_fft}")
        logger.info(f"  - Hop Length:         {self.hop_length}")
        logger.info(f"  - STFT Output Bins:   {self.stft_bins} linear bins (Dual-Channel: Mean + Peak Contrast)")
        logger.info(f"  - TKEO Pre-Emphasis:  {self.use_tkeo} (alpha_max={self.alpha_max})")
        logger.info(f"  - Spectral 1D Aug:    {'ENABLED' if self.use_spectral_aug else 'DISABLED'}")
        if self.use_spectral_aug:
            logger.info(f"    * Dual-Band Mask:   Width={self.cutout_width} bins, Prob={self.cutout_prob}")
            logger.info(f"    * Spectral Tilt:    Max Slope={self.tilt_max}")
            logger.info(f"    * Micro-Shift:      Max Shift={self.max_shift} bins")
            logger.info(f"    * Gaussian Jitter:  Noise Std={self.noise_std}")
        logger.info("==================================================")

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor: Raw 1D audio waveform [Batch, Num_Samples],
                          precomputed [Batch, 2049], or [Batch, 2, 2049].

        Returns:
            torch.Tensor: Normalized dual-channel STFT spectral profile [Batch, 2, 2049].
        """
        # Handle precomputed spectral features
        if input_tensor.ndim == 2 and input_tensor.size(-1) == self.stft_bins:
            mean_norm = self.norm_mean(input_tensor)
            spec_dual = torch.stack([mean_norm, torch.zeros_like(mean_norm)], dim=1)
            if self.spectral_augmenter is not None:
                spec_dual = self.spectral_augmenter(spec_dual)
            return spec_dual
        elif input_tensor.ndim == 3 and input_tensor.size(-1) == self.stft_bins:
            if self.spectral_augmenter is not None:
                input_tensor = self.spectral_augmenter(input_tensor)
            return input_tensor

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

        # 5. Log Magnitude: log(|X| + 1e-8)
        log_mag = torch.log(torch.abs(complex_spec) + 1e-8)

        # 6. Dual-Channel Energy Profile over time axis
        mean_psd = log_mag.mean(dim=1)
        peak_psd = log_mag.max(dim=1).values - mean_psd

        # 7. Layer Normalization per channel
        mean_norm = self.norm_mean(mean_psd)
        peak_norm = self.norm_peak(peak_psd)
        spec_dual = torch.stack([mean_norm, peak_norm], dim=1)  # [Batch, 2, 2049]

        # 8. Advanced 1D Spectral Augmentation if enabled
        if self.spectral_augmenter is not None:
            spec_dual = self.spectral_augmenter(spec_dual)

        return spec_dual
