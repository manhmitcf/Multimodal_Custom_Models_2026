import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple


class CenterNorm(nn.Module):
    """
    CenterNorm as introduced in FAST (Naman & Zhang, ICASSP 2025).
    Replaces LayerNorm by subtracting mean without dividing by variance,
    preserving strict Lipschitz continuity and bounding gradients.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        mean = x.mean(dim=-1, keepdim=True)
        scale = self.dim / max(self.dim - 1, 1)
        x_centered = (x - mean) * scale
        return x_centered * self.gamma + self.beta


class ScaledCosineSimilarityAttention(nn.Module):
    """
    Scaled Cosine Similarity Attention (SCSA) from FAST (ICASSP 2025).
    L2-normalizes Query, Key, Value to ensure bounded Lipschitz constant
    and stabilize training on high-dynamic-range audio spectrograms.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable temperature and scaling factors
        self.tau = nn.Parameter(torch.tensor(1.0 / math.sqrt(self.head_dim)))
        self.nu = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Linear projections & L2 normalization per head for Lipschitz bound
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = F.normalize(q, p=2, dim=-1, eps=1e-8)
        k = F.normalize(k, p=2, dim=-1, eps=1e-8)
        v = F.normalize(v, p=2, dim=-1, eps=1e-8)

        # Scaled Cosine similarity
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.tau
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v) * self.nu
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class InvertedResidual2D(nn.Module):
    """MobileNetV2 Inverted Residual Block for 2D Spectrograms."""
    def __init__(self, in_ch: int, out_ch: int, stride: Tuple[int, int] = (1, 1), expand_ratio: int = 2) -> None:
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == (1, 1)) and (in_ch == out_ch)
        hidden_dim = int(round(in_ch * expand_ratio))

        layers = []
        if expand_ratio != 1:
            # 1x1 point-wise expansion
            layers.extend([
                nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU()
            ])

        # 3x3 depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            # 1x1 point-wise linear projection
            nn.Conv2d(hidden_dim, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class FASTBlock2D(nn.Module):
    """
    Core FAST Block (Naman & Zhang, ICASSP 2025):
    Combines local CNN feature extraction with global Lipschitz-aware transformer
    via 2D Spectrogram Unfolding and Folding.
    """
    def __init__(self, in_ch: int, out_ch: int, d_model: int = 128, num_heads: int = 4, patch_size: Tuple[int, int] = (2, 2)) -> None:
        super().__init__()
        self.ph, self.pw = patch_size
        self.patch_area = self.ph * self.pw

        self.local_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model)
        )

        # Lipschitz-aware Transformer
        self.norm1 = CenterNorm(d_model)
        self.attn = ScaledCosineSimilarityAttention(d_model, num_heads=num_heads)
        self.alpha1 = nn.Parameter(torch.tensor(0.1))  # Weighted Residual Shortcut

        self.norm2 = CenterNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.alpha2 = nn.Parameter(torch.tensor(0.1))

        # 2D fusion projection
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(d_model, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_ch, H, W]
        x_local = self.local_conv(x)  # [B, d_model, H, W]
        B, C, H, W = x_local.shape

        # Ensure H and W divisible by patch size
        pad_h = (self.ph - H % self.ph) % self.ph
        pad_w = (self.pw - W % self.pw) % self.pw
        if pad_h > 0 or pad_w > 0:
            x_local = F.pad(x_local, (0, pad_w, 0, pad_h))
            _, _, H_pad, W_pad = x_local.shape
        else:
            H_pad, W_pad = H, W

        num_patches_h = H_pad // self.ph
        num_patches_w = W_pad // self.pw
        num_patches = num_patches_h * num_patches_w

        # MobileViT unfold: [B, C, H_pad, W_pad] -> [B, d_model, num_patches_h, ph, num_patches_w, pw]
        # -> [B * patch_area, num_patches, d_model]
        x_unfolded = x_local.view(B, C, num_patches_h, self.ph, num_patches_w, self.pw)
        x_unfolded = x_unfolded.permute(0, 3, 5, 2, 4, 1).contiguous()
        x_seq = x_unfolded.view(B * self.patch_area, num_patches, C)

        # Lipschitz-aware Transformer with Weighted Residual Shortcuts
        x_seq = x_seq + self.alpha1 * self.attn(self.norm1(x_seq))
        x_seq = x_seq + self.alpha2 * self.ffn(self.norm2(x_seq))

        # Fold back to 2D: [B, C, H_pad, W_pad]
        x_folded = x_seq.view(B, self.ph, self.pw, num_patches_h, num_patches_w, C)
        x_folded = x_folded.permute(0, 5, 3, 1, 4, 2).contiguous()
        x_folded = x_folded.view(B, C, H_pad, W_pad)

        if pad_h > 0 or pad_w > 0:
            x_folded = x_folded[:, :, :H, :W]

        # 1x1 Conv fusion
        out = self.fusion_conv(x_folded)
        return out


class NanoFAST(nn.Module):
    """
    Standalone Nano-FAST Audio Classifier (~0.9M params).
    Inspired by Fast Audio Spectrogram Transformer (FAST - ICASSP 2025).
    - Preserves 2D Spectrogram representation throughout without premature 1D collapsing.
    - MobileNetV2 inverted residual blocks for local time-frequency patterns.
    - Lipschitz-aware attention blocks for training stability from scratch.
    - Output: 4 classes classification.
    """
    def __init__(self, num_classes: int = 4, in_channels: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.model_name = "NanoFAST"

        # 1. Asymmetric Stem: nén nhanh tần số 2049 và thời gian
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 9), stride=(2, 8), padding=(1, 4), bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 48, kernel_size=(3, 5), stride=(2, 4), padding=(1, 2), bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(),
        )  # Output: [B, 48, T//4, 65]

        # 2. Stage 1: Local Inverted Residual downsample to [B, 64, T//8, 33]
        self.stage1 = nn.Sequential(
            InvertedResidual2D(48, 64, stride=(2, 2), expand_ratio=2),
            InvertedResidual2D(64, 64, stride=(1, 1), expand_ratio=2),
        )  # Output: [B, 64, 32, 33]

        # FAST Block 1: operates at 32x33 resolution (num_patches = 16x17 = 272)
        self.fast_block1 = FASTBlock2D(in_ch=64, out_ch=96, d_model=128, num_heads=4, patch_size=(2, 2))
        # Output: [B, 96, 32, 33]

        # 3. Stage 2: Local Inverted Residual downsample to [B, 112, 16, 17]
        self.stage2 = nn.Sequential(
            InvertedResidual2D(96, 112, stride=(2, 2), expand_ratio=2),
            InvertedResidual2D(112, 112, stride=(1, 1), expand_ratio=2),
        )  # Output: [B, 112, 16, 17]

        # FAST Block 2: operates at 16x17 resolution (num_patches = 8x9 = 72)
        self.fast_block2 = FASTBlock2D(in_ch=112, out_ch=144, d_model=192, num_heads=4, patch_size=(2, 2))
        # Output: [B, 144, 16, 17]

        # 4. Head: Global Pooling + Classifier
        self.conv_head = nn.Sequential(
            nn.Conv2d(144, 224, kernel_size=1, bias=False),
            nn.BatchNorm2d(224),
            nn.SiLU()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(224, 224),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(224, num_classes)
        )

    def forward(self, spec_2d: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            spec_2d: [B, 1, T, 2049] or [B, T, 2049].
        """
        if spec_2d.ndim == 3:
            spec_2d = spec_2d.unsqueeze(1)

        x = self.stem(spec_2d)
        x = self.stage1(x)
        x = self.fast_block1(x)
        x = self.stage2(x)
        x = self.fast_block2(x)
        x = self.conv_head(x)  # [B, 160, H, W]

        # Global Average & Max Pooling across 2D time-frequency map
        feat_mean = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        feat_max = F.adaptive_max_pool2d(x, (1, 1)).flatten(1)
        embedding = feat_mean + feat_max  # [B, 160]

        logits = self.classifier(embedding)

        return {
            'logits': logits,
            'clipwise_output': logits,
            'embedding': embedding
        }
