import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import DFTBase, STFT, Spectrogram


class STFT_APE(STFT):
    r"""
    PyTorch implementation of Short-Time Fourier Transform with 
    Teager-Kaiser Energy Operator (TKEO) Adaptive Pre-Emphasis (STFT_APE).
    
    References:
    - Teager-Kaiser Energy Operator for acoustic signal analysis:
      Psi[x(n)] = x^2(n) - x(n-1) * x(n+1)
    - STFT with TKEO Adaptive Pre-Emphasis for underwater and feeding sound enhancement.
    """
    def __init__(
        self,
        n_fft=2048,
        hop_length=None,
        win_length=None,
        window='hann',
        center=True,
        pad_mode='reflect',
        freeze_parameters=True,
        alpha_max=0.99,
        beta=0.8
    ):
        super(STFT_APE, self).__init__(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=freeze_parameters
        )
        self.alpha_max = float(alpha_max)
        self.beta = float(beta)

    def forward(self, input):
        x = input[:, None, :]   # (batch_size, 1, data_length)

        if self.center:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)

        # 1. Framing (sliding window)
        frames = x.unfold(dimension=-1, size=self.n_fft, step=self.hop_length)
        # Shape: (batch_size, 1, time_steps, n_fft)

        # =========================================================================
        # TKEO ADAPTIVE PRE-EMPHASIS (TKEO APE)
        # =========================================================================
        if self.alpha_max > 0:
            # A. Teager-Kaiser Energy Operator: Psi[n] = x[n]^2 - x[n-1]*x[n+1]
            x_mid = frames[:, :, :, 1:-1]
            x_left = frames[:, :, :, :-2]
            x_right = frames[:, :, :, 2:]
            psi = x_mid**2 - x_left * x_right
            psi_full = torch.cat([psi[:, :, :, :1], psi, psi[:, :, :, -1:]], dim=-1)

            # B. Normalized frame-wise TKEO energy ratio
            mean_psi = torch.mean(torch.abs(psi_full), dim=-1)   # (batch_size, 1, time_steps)
            mean_energy = torch.mean(frames**2, dim=-1)          # (batch_size, 1, time_steps)
            ctrl = mean_psi / (mean_energy + 1e-10)

            # C. Non-linear mapping & recursive temporal smoothing
            alpha_raw = self.alpha_max * (1.0 - torch.exp(-ctrl))
            alpha = torch.zeros_like(alpha_raw)
            alpha_prev = torch.zeros(input.size(0), 1, device=input.device, dtype=input.dtype)

            for t in range(frames.size(2)):
                alpha_t = self.beta * alpha_prev + (1.0 - self.beta) * alpha_raw[:, :, t]
                alpha_t = torch.clamp(alpha_t, min=0.1, max=self.alpha_max)
                alpha[:, :, t] = alpha_t
                alpha_prev = alpha_t

            # D. Internal adaptive high-pass filtering on each time frame
            frames_prev = torch.cat([frames[:, :, :, :1], frames[:, :, :, :-1]], dim=-1)
            frames = frames - alpha.unsqueeze(-1) * frames_prev
        # =========================================================================

        # 2. Linear projection using torchlibrosa DFT kernels
        w_real = self.conv_real.weight.squeeze(1)  # (n_fft // 2 + 1, n_fft)
        w_imag = self.conv_imag.weight.squeeze(1)  # (n_fft // 2 + 1, n_fft)

        real = F.linear(frames, w_real)  # (batch_size, 1, time_steps, n_fft // 2 + 1)
        imag = F.linear(frames, w_imag)  # (batch_size, 1, time_steps, n_fft // 2 + 1)

        return real, imag


class Spectrogram_APE(Spectrogram):
    r"""
    Calculate spectrogram using PyTorch with TKEO Adaptive Pre-Emphasis.
    100% compatible drop-in replacement for torchlibrosa.stft.Spectrogram.
    """
    def __init__(
        self,
        n_fft=2048,
        hop_length=None,
        win_length=None,
        window='hann',
        center=True,
        pad_mode='reflect',
        power=2.0,
        freeze_parameters=True,
        alpha_max=0.99,
        beta=0.8
    ):
        super(Spectrogram_APE, self).__init__(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            power=power,
            freeze_parameters=freeze_parameters
        )
        self.stft = STFT_APE(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=freeze_parameters,
            alpha_max=alpha_max,
            beta=beta
        )

    def forward(self, input):
        (real, imag) = self.stft.forward(input)
        spectrogram = real ** 2 + imag ** 2

        if self.power == 2.0:
            pass
        else:
            spectrogram = spectrogram ** (self.power / 2.0)

        return spectrogram
