import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FishMotionKinematics8Ch(nn.Module):
    """
    Differentiable 8-Channel Kinematics Extractor for Fish Feeding Intensity Assessment (T=2 frames):
      - Channels 0-2 : Spatial RGB visual appearance (fish density, white water foam, surface pellets)
      - Channels 3-4 : Dense Optical Flow (u, v) representing horizontal and vertical swimming velocities
      - Channel 5    : Velocity Magnitude |V| = sqrt(u^2 + v^2) (absolute swimming kinetic intensity)
      - Channel 6    : Fluid Vorticity omega = dv/dx - du/dy (swirling vortex turbulence from feeding strike)
      - Channel 7    : Convective Acceleration |a_conv| = sqrt(ax^2 + ay^2) (centripetal/burst acceleration field)

    Input:
        frames_rgb: [B, T, 3, H, W] (T=2)
    Output:
        frames_8ch: [B, T, 8, H, W]
        kinematics_summary: [B, 4] summary statistics [v_mean, omega_max, v_max, convergence_flux]
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

        # Precompute radial centripetal vector field pointing to feeding center (H/2, W/2)
        y_grid = torch.linspace(-1.0, 1.0, image_size).view(image_size, 1).repeat(1, image_size)
        x_grid = torch.linspace(-1.0, 1.0, image_size).view(1, image_size).repeat(image_size, 1)
        dx = -x_grid
        dy = -y_grid
        dist = torch.sqrt(dx ** 2 + dy ** 2) + 1e-6
        center_field = torch.stack([dx / dist, dy / dist], dim=0).unsqueeze(0)  # [1, 2, H, W]
        self.register_buffer("center_field", center_field)

    def forward(self, frames_rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C, H, W = frames_rgb.shape
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

        # Ensure buffers match input device and dtype dynamically
        sobel_x = self.sobel_x.to(device=device, dtype=dtype)
        sobel_y = self.sobel_y.to(device=device, dtype=dtype)
        center_field = self.center_field.to(device=device, dtype=dtype)

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

        # 5. Velocity Magnitude |V| in [0, ~1.414]
        v_mag_seq = torch.sqrt(u_seq ** 2 + v_seq ** 2 + 1e-6)  # [B, T, 1, H, W]

        # Spatial derivatives of optical flow u, v
        u_flat = u_seq.view(B * T, 1, H, W)
        v_flat = v_seq.view(B * T, 1, H, W)
        du_dx = F.conv2d(u_flat, sobel_x, padding=1)
        du_dy = F.conv2d(u_flat, sobel_y, padding=1)
        dv_dx = F.conv2d(v_flat, sobel_x, padding=1)
        dv_dy = F.conv2d(v_flat, sobel_y, padding=1)

        # 6. Fluid Vorticity omega = dv/dx - du/dy in [-1, 1]
        omega_flat = torch.tanh((dv_dx - du_dy) * 4.0)
        omega_seq = omega_flat.view(B, T, 1, H, W)

        # 7. Convective Acceleration |a_conv| = sqrt(ax^2 + ay^2) in [0, 1]
        # ax = u*(du/dx) + v*(du/dy), ay = u*(dv/dx) + v*(dv/dy)
        ax_flat = u_flat * du_dx + v_flat * du_dy
        ay_flat = u_flat * dv_dx + v_flat * dv_dy
        a_mag_flat = torch.sqrt(ax_flat ** 2 + ay_flat ** 2 + 1e-6)
        acc_flat = torch.tanh(a_mag_flat * 4.0)
        acc_seq = acc_flat.view(B, T, 1, H, W)

        # Assemble finalized 8 channels:
        # [R, G, B, u, v, |V|, omega, acc]
        frames_8ch = torch.cat([
            frames_rgb,   # 3 ch (Spatial appearance, fish clustering, white water foam)
            u_seq,        # 1 ch (Flow horizontal velocity)
            v_seq,        # 1 ch (Flow vertical velocity)
            v_mag_seq,    # 1 ch (Velocity magnitude |V|)
            omega_seq,    # 1 ch (Fluid vorticity / swirling turbulence)
            acc_seq       # 1 ch (Convective acceleration / feeding burst)
        ], dim=2)  # [B, T, 8, H, W]

        # Kinematics Summary Statistics:
        v_mean = v_mag_seq.mean(dim=(1, 2, 3, 4), keepdim=True).view(B, 1)
        omega_max = torch.amax(torch.abs(omega_seq), dim=(1, 2, 3, 4), keepdim=True).view(B, 1)
        v_max = torch.amax(v_mag_seq, dim=(1, 2, 3, 4), keepdim=True).view(B, 1)

        flow_2d = torch.cat([u_seq, v_seq], dim=2)  # [B, T, 2, H, W]
        flux_pixel = (flow_2d * center_field.unsqueeze(1)).sum(dim=2, keepdim=True)
        active_flux = flux_pixel * (v_mag_seq > 0.05).float()
        convergence_flux = active_flux.mean(dim=(1, 2, 3, 4), keepdim=True).view(B, 1)

        kinematics_summary = torch.cat([v_mean, omega_max, v_max, convergence_flux], dim=-1)  # [B, 4]
        return frames_8ch, kinematics_summary


# Backward compatibility aliases
FishMotionKinematics9Ch = FishMotionKinematics8Ch
FishMotionKinematics10Ch = FishMotionKinematics8Ch
FishMotionKinematics7Ch = FishMotionKinematics8Ch
FishMotionKinematics = FishMotionKinematics8Ch
