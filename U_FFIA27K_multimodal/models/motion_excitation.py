import torch
import torch.nn as nn


class MotionExcitation(nn.Module):
    """
    Motion Excitation (ME) Module for lightweight temporal dynamics modeling.
    
    References:
    - TEA: Temporal Excitation and Aggregation for Video Action Recognition (Li et al., CVPR 2020)
    - TDN: Temporal Difference Networks for Efficient Video Recognition (Wang et al., CVPR 2021)
    
    Mechanism:
    Computes feature-level temporal difference between consecutive frames:
        ΔF = |F_t - F_{t-1}|
    Passes ΔF through a channel-squeezed depthwise separable convolution to generate
    a spatial-temporal attention mask M in [0, 1].
    Modulates current spatial feature map:
        F_st = F_t * (1 + M)
    """
    def __init__(self, in_channels: int, squeeze_factor: int = 4) -> None:
        super().__init__()
        reduced_channels = max(8, in_channels // squeeze_factor)
        
        self.channel_squeeze = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.bn_squeeze = nn.BatchNorm2d(reduced_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.depthwise_conv = nn.Conv2d(
            reduced_channels,
            reduced_channels,
            kernel_size=3,
            padding=1,
            groups=reduced_channels,
            bias=False
        )
        self.bn_dw = nn.BatchNorm2d(reduced_channels)
        
        self.channel_expand = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_curr: torch.Tensor, f_prev: torch.Tensor):
        """
        Args:
            f_curr: Current frame feature map [B, C, H, W]
            f_prev: Previous frame feature map [B, C, H, W]
            
        Returns:
            f_st: Modulated Spatiotemporal feature map [B, C, H, W]
            motion_mask: Attention map of detected movements [B, C, H, W]
        """
        diff = torch.abs(f_curr - f_prev)
        
        x = self.channel_squeeze(diff)
        x = self.bn_squeeze(x)
        x = self.relu(x)
        
        x = self.depthwise_conv(x)
        x = self.bn_dw(x)
        x = self.relu(x)
        
        x = self.channel_expand(x)
        x = self.bn_expand(x)
        motion_mask = self.sigmoid(x)
        
        # Residual excitation
        f_st = f_curr * (1.0 + motion_mask)
        return f_st, motion_mask
