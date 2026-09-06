import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class FishMotionKinematics(nn.Module):
    """
    Differentiable Motion Kinematics Extractor for Fish Feeding Intensity Assessment:
    
    1. Foam / Bubble Dynamics (Sủi bọt):
       - White foam mask based on high-intensity thresholding and high-pass spatial Laplacian.
       - Foam area ratio A_foam and foam expansion rate dA/dt across time.
       
    2. Fish Movement Velocity (Vận tốc di chuyển của cá):
       - Spatiotemporal image gradients (Ix, Iy, It) via differentiable Sobel operators.
       - Dense velocity vector field (vx, vy) with magnitude ||v|| = sqrt(vx^2 + vy^2).
       - Mean fish swimming speed v_mean.
       
    3. Fish Movement Direction & Feeding Convergence (Hướng di chuyển & Độ hội tụ máng ăn):
       - Unit orientation vectors (cos theta, sin theta).
       - Centripetal Feeding Flux Phi_feed: dot product between local fish velocity vectors
         and unit vectors pointing to the central feeding zone.
         * Weak feeding: Random, low-speed, non-convergent motion (Phi_feed ~ 0).
         * Medium/Strong feeding: High-velocity concerted rush towards feeding zone (Phi_feed >> 0).
         
    Outputs:
       - video_6ch: [B, T, 6, H, W] tensor (RGB + Velocity Magnitude + vx + vy)
       - kinematics_vector: [B, 4] tensor [A_foam, dA/dt, v_mean, Phi_feed]
    """
    def __init__(self, image_size: int = 224) -> None:
        super().__init__()
        self.image_size = image_size

        # Fixed Sobel filters for spatial gradients (channels=1)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        laplacian = torch.tensor([[0.0,  1.0, 0.0],
                                  [1.0, -4.0, 1.0],
                                  [0.0,  1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("laplacian", laplacian)

        # Precompute unit direction vectors towards central feeding zone [1, 2, H, W]
        # Central feeding zone is assumed at (H/2, W/2)
        y_coords = torch.linspace(-1.0, 1.0, image_size).view(image_size, 1).repeat(1, image_size)
        x_coords = torch.linspace(-1.0, 1.0, image_size).view(1, image_size).repeat(image_size, 1)
        
        # Vectors point FROM (x, y) TOWARDS center (0, 0): so delta is (-x, -y)
        dx = -x_coords
        dy = -y_coords
        dist = torch.sqrt(dx ** 2 + dy ** 2) + 1e-6
        unit_to_center_x = dx / dist
        unit_to_center_y = dy / dist
        center_field = torch.stack([unit_to_center_x, unit_to_center_y], dim=0).unsqueeze(0) # [1, 2, H, W]
        self.register_buffer("center_field", center_field)

    def forward(self, frames_rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frames_rgb: [B, T, 3, H, W] normalized video frames in [0, 1]
        Returns:
            frames_6ch: [B, T, 6, H, W] augmented tensor with velocity and direction
            kinematics: [B, 4] tensor [A_foam, dA/dt, v_mean, Phi_feed]
        """
        B, T, C, H, W = frames_rgb.shape
        device = frames_rgb.device
        dtype = frames_rgb.dtype

        # 1. Grayscale luminance conversion [B * T, 1, H, W]
        rgb_flat = frames_rgb.view(B * T, 3, H, W)
        gray_flat = 0.299 * rgb_flat[:, 0:1] + 0.587 * rgb_flat[:, 1:2] + 0.114 * rgb_flat[:, 2:3]
        gray_seq = gray_flat.view(B, T, 1, H, W)

        # 2. Sủi bọt (White Water Foam & Bubble Texture)
        # Foam has high brightness and high spatial variance
        brightness = rgb_flat.mean(dim=1, keepdim=True) # [B*T, 1, H, W]
        is_bright = F.relu(brightness - 0.75) * 4.0     # Soft thresholding
        lap_edges = torch.abs(F.conv2d(gray_flat, self.laplacian, padding=1))
        foam_map = (is_bright * (1.0 + lap_edges)).view(B, T, H, W) # [B, T, H, W]
        
        foam_per_frame = (foam_map > 0.3).float().mean(dim=(-2, -1)) # [B, T]
        a_foam = foam_per_frame.mean(dim=-1, keepdim=True)           # [B, 1]
        
        if T >= 2:
            da_dt = torch.abs(foam_per_frame[:, 1:] - foam_per_frame[:, :-1]).mean(dim=-1, keepdim=True) # [B, 1]
        else:
            da_dt = torch.zeros(B, 1, dtype=dtype, device=device)

        # 3. Vận tốc và Hướng di chuyển của cá (Fish Velocity & Direction)
        # Compute spatial gradients Ix, Iy on current frame
        ix_flat = F.conv2d(gray_flat, self.sobel_x, padding=1) # [B*T, 1, H, W]
        iy_flat = F.conv2d(gray_flat, self.sobel_y, padding=1) # [B*T, 1, H, W]

        # Temporal difference It = I_t - I_{t-1}
        it_seq = torch.zeros(B, T, 1, H, W, dtype=dtype, device=device)
        if T >= 2:
            it_seq[:, 1:] = gray_seq[:, 1:] - gray_seq[:, :-1]
        it_flat = it_seq.view(B * T, 1, H, W)

        # Differentiable optical flow approximation:
        # v = - (Grad_s * It) / (||Grad_s||^2 + eps)
        grad_mag_sq = ix_flat ** 2 + iy_flat ** 2 + 1e-4
        raw_vx = - (ix_flat * it_flat) / grad_mag_sq
        raw_vy = - (iy_flat * it_flat) / grad_mag_sq

        # Apply soft clipping to prevent noise explosions
        vx_flat = torch.tanh(raw_vx * 2.0) # [B*T, 1, H, W]
        vy_flat = torch.tanh(raw_vy * 2.0) # [B*T, 1, H, W]

        # Velocity magnitude ||v||
        v_mag_flat = torch.sqrt(vx_flat ** 2 + vy_flat ** 2 + 1e-6) # [B*T, 1, H, W]
        v_seq = v_mag_flat.view(B, T, 1, H, W)
        vx_seq = vx_flat.view(B, T, 1, H, W)
        vy_seq = vy_flat.view(B, T, 1, H, W)

        # Mean fish velocity across time and space
        v_mean = v_seq.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(-1) # [B, 1]

        # 4. Hướng hội tụ về máng ăn trung tâm (Centripetal Feeding Flux Phi_feed)
        # Dot product between (vx, vy) and center_field
        # center_field is [1, 2, H, W]
        flow_2d = torch.cat([vx_flat, vy_flat], dim=1) # [B*T, 2, H, W]
        flux_per_pixel = (flow_2d * self.center_field).sum(dim=1, keepdim=True) # [B*T, 1, H, W]
        
        # We only consider flux where movement actually happens
        active_flux = flux_per_pixel * (v_mag_flat > 0.05).float()
        phi_feed = active_flux.view(B, T, -1).mean(dim=-1).mean(dim=-1, keepdim=True) # [B, 1]

        # 5. Assemble 6-channel spatiotemporal tensor: [B, T, 6, H, W]
        # Channels: [R, G, B, Velocity_Magnitude, vx, vy]
        frames_6ch = torch.cat([frames_rgb, v_seq, vx_seq, vy_seq], dim=2)

        # 6. Assemble 4-dimensional kinematics vector: [B, 4]
        kinematics_vector = torch.cat([a_foam, da_dt, v_mean, phi_feed], dim=-1)

        return frames_6ch, kinematics_vector
