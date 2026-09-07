import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FishMotionKinematics10Ch(nn.Module):
    """
    Differentiable 10-Channel Kinematics Extractor for Fish Feeding Intensity Assessment.
    Constructs a rich spatiotemporal representation across T frames (T=4):
      - Channels 0-2 : RGB visual appearance
      - Channels 3-4 : Dense Optical Flow (u, v) from spatiotemporal gradients
      - Channel 5    : Velocity Magnitude |V| = sqrt(u^2 + v^2)
      - Channel 6    : Motion Direction Angle theta = atan2(v, u) / pi
      - Channel 7    : Fluid Vorticity omega = dv/dx - du/dy (swirling turbulence from feeding frenzy)
      - Channel 8    : Frame Difference Delta I = |I_t - I_{t-1}|
      - Channel 9    : Motion History Image (MHI) showing temporal motion persistence

    Input:
        frames_rgb: [B, T, 3, H, W]
    Output:
        frames_10ch: [B, T, 10, H, W]
        kinematics_summary: [B, 4] summary statistics [v_mean, omega_max, splash_ratio, convergence_flux]
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

        # 3. Temporal differences It & Consecutive Frame Differences Delta I
        it_seq = torch.zeros(B, T, 1, H, W, dtype=dtype, device=device)
        diff_seq = torch.zeros(B, T, 1, H, W, dtype=dtype, device=device)

        if T >= 2:
            it_seq[:, 0] = gray_seq[:, 1] - gray_seq[:, 0]
            it_seq[:, 1:] = gray_seq[:, 1:] - gray_seq[:, :-1]
            diff_seq[:, 0] = torch.abs(gray_seq[:, 1] - gray_seq[:, 0])
            diff_seq[:, 1:] = torch.abs(gray_seq[:, 1:] - gray_seq[:, :-1])
        else:
            it_seq[:, 0] = 0.0
            diff_seq[:, 0] = 0.0

        # 4. Optical Flow (u, v) via differential formulation:
        # u = - (Ix * It) / (Ix^2 + Iy^2 + eps), v = - (Iy * It) / (Ix^2 + Iy^2 + eps)
        grad_sq = ix_seq ** 2 + iy_seq ** 2 + 1e-4
        raw_u = - (ix_seq * it_seq) / grad_sq
        raw_v = - (iy_seq * it_seq) / grad_sq
        u_seq = torch.tanh(raw_u * 2.0)  # [B, T, 1, H, W] in [-1, 1]
        v_seq = torch.tanh(raw_v * 2.0)  # [B, T, 1, H, W] in [-1, 1]

        # 5. Velocity Magnitude |V|
        v_mag_seq = torch.sqrt(u_seq ** 2 + v_seq ** 2 + 1e-6)  # [B, T, 1, H, W]

        # 6. Direction Angle theta = atan2(v, u) / pi in [-1, 1]
        theta_seq = torch.atan2(v_seq, u_seq + 1e-7) / 3.1415926535

        # 7. Fluid Vorticity omega = dv/dx - du/dy
        v_flat = v_seq.view(B * T, 1, H, W)
        u_flat = u_seq.view(B * T, 1, H, W)
        dv_dx = F.conv2d(v_flat, sobel_x, padding=1)
        du_dy = F.conv2d(u_flat, sobel_y, padding=1)
        omega_flat = torch.tanh((dv_dx - du_dy) * 4.0)  # normalized vorticity
        omega_seq = omega_flat.view(B, T, 1, H, W)

        # 8. Motion History Image (MHI)
        # Recurrent accumulation across frames
        mhi_list = []
        decay_step = 1.0 / float(max(T, 1))
        mhi_prev = torch.zeros(B, 1, H, W, dtype=dtype, device=device)
        for t in range(T):
            motion_mask = (diff_seq[:, t] > 0.05).float()
            mhi_curr = torch.clamp(mhi_prev - decay_step, min=0.0) + motion_mask
            mhi_curr = torch.clamp(mhi_curr, 0.0, 1.0)
            mhi_list.append(mhi_curr)
            mhi_prev = mhi_curr
        mhi_seq = torch.stack(mhi_list, dim=1)  # [B, T, 1, H, W]

        # Assemble all 10 channels:
        # [R, G, B, u, v, |V|, theta, omega, Delta_I, MHI]
        frames_10ch = torch.cat([
            frames_rgb,   # 3 ch
            u_seq,        # 1 ch
            v_seq,        # 1 ch
            v_mag_seq,    # 1 ch
            theta_seq,    # 1 ch
            omega_seq,    # 1 ch
            diff_seq,     # 1 ch
            mhi_seq,      # 1 ch
        ], dim=2)  # [B, T, 10, H, W]

        # 9. Kinematics Summary Statistics for Evidential Guidance:
        # [v_mean, omega_max, splash_ratio, convergence_flux]
        v_mean = v_mag_seq.mean(dim=(1, 2, 3, 4), keepdim=True).view(B, 1)
        omega_max = torch.amax(torch.abs(omega_seq), dim=(1, 2, 3, 4), keepdim=True).view(B, 1)
        splash_ratio = (diff_seq > 0.1).float().mean(dim=(1, 2, 3, 4), keepdim=True).view(B, 1)

        # Feeding convergence flux: dot product between (u, v) and center_field
        flow_2d = torch.cat([u_seq, v_seq], dim=2)  # [B, T, 2, H, W]
        flux_pixel = (flow_2d * center_field.unsqueeze(1)).sum(dim=2, keepdim=True)
        active_flux = flux_pixel * (v_mag_seq > 0.05).float()
        convergence_flux = active_flux.mean(dim=(1, 2, 3, 4), keepdim=True).view(B, 1)

        kinematics_summary = torch.cat([v_mean, omega_max, splash_ratio, convergence_flux], dim=-1)  # [B, 4]

        return frames_10ch, kinematics_summary
