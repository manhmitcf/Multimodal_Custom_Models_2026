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

from config import AudioFeaturesConfig

logger = logging.getLogger(__name__)


class Spectral1DAugmentation(nn.Module):
    """
    1D Spectral Augmentation Module for High-Resolution TKEO-STFT representations.
    Encapsulates dedicated augmentation techniques for spectral distributions:
      1. Frequency Cutout: Masks a narrow contiguous frequency band (cutout_width bins)
         with the sample's minimum energy (noise floor) instead of 0.0 to prevent energy explosion.
         Operates identically on 1D vectors [B, 2049] or 2D spectrograms [B, T, 2049]
         by broadcasting the cutout mask across the temporal dimension.
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

    def forward(self, spec_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec_tensor: STFT spectral energy tensor [B, 2049], [2049], or [B, T, 2049].
        Returns:
            Augmented spectral tensor with the exact same shape.
        """
        if not self.training:
            return spec_tensor

        is_1d = (spec_tensor.ndim == 1)
        out = spec_tensor.unsqueeze(0).clone() if is_1d else spec_tensor.clone()

        # 1. Frequency Cutout (Vectorized mask on GPU, zero Python loop, zero device sync)
        if self.cutout_width > 0 and self.cutout_prob > 0.0:
            if out.ndim == 2:
                B, F = out.shape
                if F > self.cutout_width:
                    mask_decisions = (torch.rand(B, 1, device=out.device) < self.cutout_prob)
                    if mask_decisions.any():
                        start_indices = torch.randint(
                            0, F - self.cutout_width, (B, 1), device=out.device
                        )
                        freq_indices = torch.arange(F, device=out.device).unsqueeze(0)  # [1, F]
                        cutout_mask = (freq_indices >= start_indices) & (freq_indices < start_indices + self.cutout_width) & mask_decisions
                        min_vals = out.min(dim=-1, keepdim=True)[0]
                        out = torch.where(cutout_mask, min_vals, out)

            elif out.ndim == 3:
                B, _, F = out.shape
                if F > self.cutout_width:
                    mask_decisions = (torch.rand(B, 1, 1, device=out.device) < self.cutout_prob)
                    if mask_decisions.any():
                        start_indices = torch.randint(
                            0, F - self.cutout_width, (B, 1, 1), device=out.device
                        )
                        freq_indices = torch.arange(F, device=out.device).view(1, 1, F)  # [1, 1, F]
                        cutout_mask = (freq_indices >= start_indices) & (freq_indices < start_indices + self.cutout_width) & mask_decisions
                        min_vals = out.amin(dim=(-2, -1), keepdim=True)
                        out = torch.where(cutout_mask, min_vals, out)

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
    Log Magnitude, and Temporal Mean Pooling to extract a 2049-dimensional spectral vector.
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
        self.beta = float(getattr(self.config, 'beta', 0.8))
        self.use_tkeo = bool(getattr(self.config, 'use_tkeo', True))

        self.use_spectral_aug = bool(getattr(self.config, 'use_spectral_aug', False))
        self.cutout_width = int(getattr(self.config, 'cutout_width', 24))
        self.cutout_prob = float(getattr(self.config, 'cutout_prob', 0.5))
        self.noise_std = float(getattr(self.config, 'noise_std', 0.02))

        # Register Hann window buffer
        window = torch.hann_window(self.n_fft)
        self.register_buffer('window', window)

        # Dedicated 1D Spectral Augmentation Module (defaults to disabled)
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
        logger.info("Initialized TKEO-STFT Audio Frontend (256 kHz, Pure Spectral):")
        logger.info(f"  - Sample Rate:        {self.sample_rate} Hz (256 kHz)")
        logger.info(f"  - FFT Size (n_fft):   {self.n_fft}")
        logger.info(f"  - Hop Length:         {self.hop_length}")
        logger.info(f"  - STFT Output Bins:   {self.stft_bins} linear bins")
        logger.info(f"  - TKEO Pre-Emphasis:  {self.use_tkeo} (alpha_max={self.alpha_max}, beta={self.beta})")
        logger.info(f"  - Spectral 1D Aug:    {'ENABLED' if self.use_spectral_aug else 'DISABLED'}")
        if self.use_spectral_aug:
            logger.info(f"    * 1D Cutout Band:   Width={self.cutout_width} bins, Prob={self.cutout_prob}")
            logger.info(f"    * Gaussian Jitter:  Noise Std={self.noise_std}")
        logger.info("==================================================")

    def forward(self, input_tensor: torch.Tensor, return_2d: bool = False) -> torch.Tensor:
        """
        Args:
            input_tensor: Raw 1D audio waveform [Batch, Num_Samples].
            return_2d: If True, returns normalized 2D STFT spectrogram [Batch, Time_Steps, 2049]
                       for sequence and convolutional backbones (BC-ResNet, BiGRU, Conformer).
                       If False (default), returns temporally pooled vector [Batch, 2049] for MLP.

        Returns:
            torch.Tensor: Normalized STFT spectral tensor [Batch, 2049] or [Batch, Time_Steps, 2049].
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

            alpha_raw = self.alpha_max * (1.0 - torch.exp(-ctrl))
            alpha_raw = torch.clamp(alpha_raw, min=0.1, max=self.alpha_max)

            if self.beta > 0.0 and frames.size(1) > 1:
                B, T, _ = alpha_raw.shape
                alpha = torch.empty_like(alpha_raw)
                a_prev = torch.zeros(B, 1, 1, device=frames.device, dtype=frames.dtype)
                for t in range(T):
                    a_t = self.beta * a_prev + (1.0 - self.beta) * alpha_raw[:, t:t+1]
                    a_t = torch.clamp(a_t, min=0.1, max=self.alpha_max)
                    alpha[:, t:t+1] = a_t
                    a_prev = a_t
            else:
                alpha = alpha_raw

            frames_prev = torch.cat([torch.zeros_like(frames[:, :, :1]), frames[:, :, :-1]], dim=-1)
            frames = frames - alpha * frames_prev

        # 4. Windowing & cuFFT Real FFT -> [Batch, Time_Steps, 2049]
        if self.window.device != frames.device or self.window.dtype != frames.dtype:
            self.window = self.window.to(device=frames.device, dtype=frames.dtype)
        frames_win = frames * self.window
        complex_spec = torch.fft.rfft(frames_win, n=self.n_fft, dim=-1)

        # 5. Log Magnitude: log(|X| + 1e-8) -> [Batch, Time_Steps, 2049]
        log_mag = torch.log(torch.abs(complex_spec) + 1e-8)

        if return_2d:
            # 2D Spectrogram Path for Sequence/CNN Backbones
            if self.spectral_augmenter is not None:
                log_mag = self.spectral_augmenter(log_mag)
            out = self.norm(log_mag)
            return out
        else:
            # 1D Vector Path for MLP Backbone (Temporal Mean Pooling -> [Batch, 2049])
            spec_vector = log_mag.mean(dim=1)
            if self.spectral_augmenter is not None:
                spec_vector = self.spectral_augmenter(spec_vector)
            out = self.norm(spec_vector)
            return out
