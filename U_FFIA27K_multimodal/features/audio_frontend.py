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


def build_linear_triangular_filterbank(
    in_bins: int = 2049,
    out_bins: int = 512,
    sample_rate: int = 256000
) -> torch.Tensor:
    """
    Constructs a fixed Linear Triangular Filterbank matrix [in_bins, out_bins] (0 trainable parameters).
    Distributes out_bins triangular filters uniformly across the Nyquist frequency range [0, sample_rate / 2].
    Preserves linear spacing essential for ultrasound underwater bioacoustic signals (0 - 128 kHz).
    """
    nyquist = sample_rate / 2.0
    fft_freqs = torch.linspace(0, nyquist, in_bins)
    filter_freqs = torch.linspace(0, nyquist, out_bins + 2)
    fb = torch.zeros(in_bins, out_bins)
    for m in range(out_bins):
        f_left = filter_freqs[m]
        f_center = filter_freqs[m + 1]
        f_right = filter_freqs[m + 2]

        up_slope = (fft_freqs - f_left) / (f_center - f_left + 1e-10)
        down_slope = (f_right - fft_freqs) / (f_right - f_center + 1e-10)
        tri = torch.minimum(up_slope, down_slope)
        fb[:, m] = torch.clamp(tri, min=0.0)

    area = fb.sum(dim=0, keepdim=True)
    fb = fb / torch.where(area > 0, area, torch.ones_like(area))
    return fb


class DualSpecAugment2D(nn.Module):
    """
    Dual 2D SpecAugment Module on Filterbank Spectrogram [Batch, 512, Time_Steps].
    Designed specifically to eliminate memorization and combat overfitting:
      1. Dual Frequency Masking (num_freq_masks=2): Che 2 vệt tần số song song,
         mỗi vệt rộng tối đa 32 bins (tổng che ~10-12.5% phổ).
      2. Dual Time Masking (num_time_masks=2): Che 2 vệt thời gian song song,
         mỗi vệt rộng tối đa 16 khung hình.
      3. Gaussian Jitter: Nhiễu chuẩn Gaussian std=0.02 mô phỏng nhiễu cảm biến hydrophone.
      4. Che bằng năng lượng nền tối thiểu (min_vals) tránh xung đột năng lượng giả lập.
    """
    def __init__(
        self,
        freq_mask_max: int = 32,
        time_mask_max: int = 16,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        freq_mask_prob: float = 0.5,
        time_mask_prob: float = 0.5,
        noise_std: float = 0.02
    ) -> None:
        super().__init__()
        self.freq_mask_max = int(freq_mask_max)
        self.time_mask_max = int(time_mask_max)
        self.num_freq_masks = int(num_freq_masks)
        self.num_time_masks = int(num_time_masks)
        self.freq_mask_prob = float(freq_mask_prob)
        self.time_mask_prob = float(time_mask_prob)
        self.noise_std = float(noise_std)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: Tensor [Batch, Num_Bins, Time_Steps] (e.g. [B, 512, T]).
        """
        if not self.training:
            return spec

        out = spec.clone()
        B, F_bins, T_steps = out.shape

        # Sàn năng lượng thực tế của từng mẫu trong batch
        min_vals = out.amin(dim=(1, 2), keepdim=True)

        # 1. Dual Frequency Masking
        if self.freq_mask_max > 0 and F_bins > self.freq_mask_max:
            for _ in range(self.num_freq_masks):
                if torch.rand(1).item() < self.freq_mask_prob:
                    mask_decisions = (torch.rand(B, 1, 1, device=out.device) < self.freq_mask_prob)
                    if mask_decisions.any():
                        w = torch.randint(1, self.freq_mask_max + 1, (B, 1, 1), device=out.device)
                        f0 = torch.randint(0, F_bins - self.freq_mask_max, (B, 1, 1), device=out.device)
                        f_idx = torch.arange(F_bins, device=out.device).view(1, F_bins, 1)
                        mask = (f_idx >= f0) & (f_idx < (f0 + w)) & mask_decisions
                        out = torch.where(mask, min_vals, out)

        # 2. Dual Time Masking
        if self.time_mask_max > 0 and T_steps > self.time_mask_max:
            for _ in range(self.num_time_masks):
                if torch.rand(1).item() < self.time_mask_prob:
                    mask_decisions = (torch.rand(B, 1, 1, device=out.device) < self.time_mask_prob)
                    if mask_decisions.any():
                        w = torch.randint(1, self.time_mask_max + 1, (B, 1, 1), device=out.device)
                        t0 = torch.randint(0, T_steps - self.time_mask_max, (B, 1, 1), device=out.device)
                        t_idx = torch.arange(T_steps, device=out.device).view(1, 1, T_steps)
                        mask = (t_idx >= t0) & (t_idx < (t0 + w)) & mask_decisions
                        out = torch.where(mask, min_vals, out)

        # 3. Gaussian Spectral Jitter
        if self.noise_std > 0.0:
            noise = torch.randn_like(out) * self.noise_std
            out = out + noise

        return out


class AudioFrontend(nn.Module):
    """
    GPU-based High-Resolution TKEO-STFT + Linear Filterbank Frontend (256 kHz, 512 linear bins).
    Pipeline:
      1. Sóng âm thô 256 kHz [B, Num_Samples] -> Framing (Hann window)
      2. TKEO Adaptive Pre-Emphasis -> cuFFT RFFT -> [B, Time_Steps, 2049]
      3. Fixed Linear Triangular Filterbank: nén tuyến tính từ 2049 -> 512 bins (0 tham số học)
      4. Log Magnitude -> Transpose -> [B, 512, Time_Steps]
      5. Dual SpecAugment 2D (2 Freq Masks max 32 + 2 Time Masks max 16 + Jitter 0.02)
      6. Layer Normalization per time frame over 512 bins -> [B, 512, Time_Steps]
    """
    def __init__(self, config: Optional[AudioFeaturesConfig] = None) -> None:
        super().__init__()
        if config is None:
            self.config = AudioFeaturesConfig()
        else:
            self.config = config

        self.sample_rate = getattr(self.config, 'sample_rate', 256000)
        self.n_fft = getattr(self.config, 'window_size', 4096)
        self.hop_length = getattr(self.config, 'hop_size', 2048)
        self.stft_bins = self.n_fft // 2 + 1  # 2049
        self.filterbank_bins = getattr(self.config, 'mel_bins', 512)
        self.alpha_max = float(getattr(self.config, 'alpha_max', 0.99))
        self.use_tkeo = bool(getattr(self.config, 'use_tkeo', True))

        # Augmentation parameters
        self.use_spectral_aug = bool(getattr(self.config, 'use_spectral_aug', True))
        self.freq_mask_max = int(getattr(self.config, 'freq_mask_max', 32))
        self.time_mask_max = int(getattr(self.config, 'time_mask_max', 16))
        self.num_freq_masks = int(getattr(self.config, 'num_freq_masks', 2))
        self.num_time_masks = int(getattr(self.config, 'num_time_masks', 2))
        self.noise_std = float(getattr(self.config, 'noise_std', 0.02))

        # Register Hann window buffer
        window = torch.hann_window(self.n_fft)
        self.register_buffer('window', window)

        # Register Fixed Linear Triangular Filterbank Buffer [2049, 512] (0 tham số học)
        fb_matrix = build_linear_triangular_filterbank(
            in_bins=self.stft_bins,
            out_bins=self.filterbank_bins,
            sample_rate=self.sample_rate
        )
        self.register_buffer('filterbank', fb_matrix)

        # Dual SpecAugment 2D module
        if self.use_spectral_aug:
            self.spec_augmenter = DualSpecAugment2D(
                freq_mask_max=self.freq_mask_max,
                time_mask_max=self.time_mask_max,
                num_freq_masks=self.num_freq_masks,
                num_time_masks=self.num_time_masks,
                noise_std=self.noise_std
            )
        else:
            self.spec_augmenter = None

        # Layer Normalization over 512 filterbank bins
        self.norm = nn.LayerNorm(self.filterbank_bins)

        logger.info("==================================================")
        logger.info("Initialized TKEO-STFT Linear Filterbank Frontend (256 kHz, 512 bins):")
        logger.info(f"  - Sample Rate:        {self.sample_rate} Hz (256 kHz)")
        logger.info(f"  - FFT Size (n_fft):   {self.n_fft}")
        logger.info(f"  - Hop Length:         {self.hop_length}")
        logger.info(f"  - Raw STFT Bins:      {self.stft_bins} linear bins")
        logger.info(f"  - Linear Filterbank:  {self.filterbank_bins} bins (Fixed Triangular, 0 params)")
        logger.info(f"  - TKEO Pre-Emphasis:  {self.use_tkeo} (alpha_max={self.alpha_max})")
        logger.info(f"  - Dual SpecAugment:   {'ENABLED' if self.use_spectral_aug else 'DISABLED'}")
        if self.use_spectral_aug:
            logger.info(f"    * Dual Freq Masks:  Count={self.num_freq_masks}, Max={self.freq_mask_max} bins (~12.5% spectrum)")
            logger.info(f"    * Dual Time Masks:  Count={self.num_time_masks}, Max={self.time_mask_max} frames")
            logger.info(f"    * Gaussian Jitter:  Noise Std={self.noise_std}")
        logger.info("==================================================")

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor: Raw audio waveform [Batch, Num_Samples] or [Batch, 1, Num_Samples].

        Returns:
            torch.Tensor: Normalized Filterbank 2D Spectrogram [Batch, 512, Time_Steps].
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

        # 5. Linear Magnitude Spectrum
        mag_spec = torch.abs(complex_spec)  # [Batch, Time_Steps, 2049]

        # 6. Apply Fixed Linear Triangular Filterbank [2049, 512]
        if self.filterbank.device != mag_spec.device:
            self.filterbank = self.filterbank.to(mag_spec.device)
        fb_spec = torch.matmul(mag_spec, self.filterbank)  # [Batch, Time_Steps, 512]

        # 7. Log Magnitude: log(fb_spec + 1e-8)
        log_fb = torch.log(fb_spec + 1e-8)

        # 8. Channel-first layout -> [Batch, 512, Time_Steps]
        spec_2d = log_fb.transpose(1, 2)

        # 9. Dual SpecAugment 2D (Training only)
        if self.spec_augmenter is not None:
            spec_2d = self.spec_augmenter(spec_2d)

        # 10. Layer Normalization per time slice over 512 bins
        out = self.norm(spec_2d.transpose(1, 2)).transpose(1, 2)

        return out

