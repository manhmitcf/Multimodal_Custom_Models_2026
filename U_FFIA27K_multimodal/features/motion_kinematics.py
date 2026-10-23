import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FishMotionKinematics5Ch(nn.Module):
    """
    Differentiable 5-Channel Motion Kinematics Extractor for Fish Feeding Intensity Assessment:
      - Channels 0-2 : Spatial RGB appearance (fish clustering, water surface perturbation)
      - Channels 3-4 : Dense Optical Flow (u, v) representing horizontal and vertical swimming velocities
      (Omits magnitude |V| and vorticity omega to eliminate high-frequency gradient noise)

    Input:
        frames_rgb: [B, T, 3, H, W] (T >= 1, nominally T=2)
    Output:
        frames_5ch: [B, T, 5, H, W]
        kinematics_summary: [B, 4] lightweight summary tensor
    """
    def __init__(self, image_size: int = 224) -> None:
        super().__init__()
        self.image_size = image_size

        # Fixed Sobel spatial gradient filters (channel=1)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, frames_rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _, H, W = frames_rgb.shape
        device = frames_rgb.device
        dtype = frames_rgb.dtype

        # Unnormalize ImageNet normalization to [0, 1] for physics calculations
        rgb_flat = frames_rgb.view(B * T, 3, H, W)
        if rgb_flat.min() < 0.0:
            mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
            rgb_01 = torch.clamp(rgb_flat * std + mean, 0.0, 1.0)
        else:
            rgb_01 = torch.clamp(rgb_flat, 0.0, 1.0)

        sobel_x = self.sobel_x.to(device=device, dtype=dtype)
        sobel_y = self.sobel_y.to(device=device, dtype=dtype)

        # 1. Grayscale luminance [B, T, 1, H, W]
        gray_flat = 0.299 * rgb_01[:, 0:1] + 0.587 * rgb_01[:, 1:2] + 0.114 * rgb_01[:, 2:3]
        gray_seq = gray_flat.view(B, T, 1, H, W)

        # 2. Spatial gradients Ix, Iy
        ix_flat = F.conv2d(gray_flat, sobel_x, padding=1)
        iy_flat = F.conv2d(gray_flat, sobel_y, padding=1)
        ix_seq = ix_flat.view(B, T, 1, H, W)
        iy_seq = iy_flat.view(B, T, 1, H, W)

        # 3. Temporal differences It
        it_seq = torch.zeros(B, T, 1, H, W, dtype=dtype, device=device)
        if T >= 2:
            it_seq[:, 0] = gray_seq[:, 1] - gray_seq[:, 0]
            it_seq[:, 1:] = gray_seq[:, 1:] - gray_seq[:, :-1]

        # 4. Optical Flow (u, v) via differential gradient formulation:
        grad_sq = ix_seq ** 2 + iy_seq ** 2 + 1e-4
        raw_u = - (ix_seq * it_seq) / grad_sq
        raw_v = - (iy_seq * it_seq) / grad_sq
        u_seq = torch.tanh(raw_u * 2.0)  # [B, T, 1, H, W] in [-1, 1]
        v_seq = torch.tanh(raw_v * 2.0)  # [B, T, 1, H, W] in [-1, 1]

        # Assemble finalized 5 channels: [R, G, B, u, v]
        frames_5ch = torch.cat([frames_rgb, u_seq, v_seq], dim=2)  # [B, T, 5, H, W]

        # Lightweight summary tensor [B, 4]
        kinematics_summary = torch.zeros(B, 4, device=device, dtype=dtype)

        return frames_5ch, kinematics_summary
