import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class DepthwiseSeparableConv2d(nn.Module):
    """Efficient 2D Depthwise Separable Convolution for Spectrograms."""
    def __init__(self, in_ch: int, out_ch: int, stride: tuple = (1, 2)) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=(3, 5), stride=stride, padding=(1, 2), groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class ConformerConvModule(nn.Module):
    """
    Depthwise 1D Convolution Module from Conformer:
    LayerNorm -> Pointwise 1x1 -> GLU -> Depthwise 1D -> BatchNorm1d -> SiLU -> Pointwise 1x1 -> Dropout
    """
    def __init__(self, dim: int, kernel_size: int = 15, dropout: float = 0.1) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            dim, dim, kernel_size=kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, groups=dim, bias=False
        )
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        self.pointwise2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.pointwise1(x)
        x = self.glu(x)
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pointwise2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # [B, T, D]
        return residual + x


class ConformerBlock(nn.Module):
    """
    Macaron-style Conformer Block:
    x = x + 0.5 * FFN(x)
    x = x + MHSA(x)
    x = x + ConvModule(x)
    x = x + 0.5 * FFN(x)
    x = LayerNorm(x)
    """
    def __init__(self, dim: int = 192, num_heads: int = 4, mlp_ratio: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        # Feed-forward 1 (half-step)
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mlp_ratio),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )
        # Multi-Head Self-Attention
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        # Conformer 1D Conv Module
        self.conv_module = ConformerConvModule(dim, kernel_size=15, dropout=dropout)
        # Feed-forward 2 (half-step)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mlp_ratio),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x + 0.5 * self.ffn1(x)
        norm_x = self.norm_attn(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = self.conv_module(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class NanoConformer(nn.Module):
    """
    Standalone Nano-Conformer Audio Classifier (~1.5M params).
    Operates on 2D TKEO-STFT Spectrogram [B, 1, Time_Steps, 2049] without temporal mean.
    - Asymmetric Conv2d Subsampling: nén trục tần số 2049 -> 33 (giữ nguyên độ phân giải thời gian T).
    - 2 Conformer Blocks: tự chú ý nhịp điệu và bắt hình thái xung kích phát (clicks).
    - Dual-Pooling (Mean + Max) và Linear Classifier (4 classes).
    """
    def __init__(
        self,
        num_classes: int = 4,
        embed_dim: int = 160,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.model_name = "NanoConformer"
        self.embed_dim = embed_dim

        # 1. 2D Asymmetric Conv Subsampling
        self.cnn_subsampling = nn.Sequential(
            # Stage 1: [B, 1, T, 2049] -> [B, 32, T, 257] (stride freq = 8)
            nn.Conv2d(1, 32, kernel_size=(3, 9), stride=(1, 8), padding=(1, 4), bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            # Stage 2: [B, 32, T, 257] -> [B, 64, T, 65] (stride freq = 4)
            DepthwiseSeparableConv2d(32, 64, stride=(1, 4)),
            # Stage 3: [B, 64, T, 65] -> [B, 64, T, 33] (stride freq = 2)
            DepthwiseSeparableConv2d(64, 64, stride=(1, 2))
        )

        # Dynamic calculation of flattened feature dim along frequency
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 4, 2049)
            feat = self.cnn_subsampling(dummy)
            flattened_dim = feat.shape[1] * feat.shape[3]  # 64 * 33 = 2112

        # 2. Linear projection to conformer embedding space + Positional Encoding
        self.proj = nn.Linear(flattened_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 500, embed_dim) * 0.02)
        self.pos_dropout = nn.Dropout(dropout)

        # 3. Conformer Layers
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 4. Classifier Head
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, spec_2d: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            spec_2d: 2D Spectrogram tensor [B, 1, T, 2049] or [B, T, 2049].

        Returns:
            Dict containing 'logits', 'embedding', 'tokens'.
        """
        if spec_2d.ndim == 3:
            spec_2d = spec_2d.unsqueeze(1)  # [B, 1, T, 2049]

        # 1. CNN Feature Subsampling (T preserved, F compressed)
        feat = self.cnn_subsampling(spec_2d)  # [B, 64, T, 33]
        B, C, T, F_prime = feat.shape

        # 2. Reshape to temporal token sequence: [B, T, C * F_prime]
        tokens = feat.permute(0, 2, 1, 3).reshape(B, T, -1)
        tokens = self.proj(tokens)  # [B, T, embed_dim]
        tokens = tokens + self.pos_emb[:, :T, :]
        tokens = self.pos_dropout(tokens)

        # 3. Conformer Context Modeling
        for layer in self.conformer_layers:
            tokens = layer(tokens)

        tokens = self.norm(tokens)

        # 4. Dual-Pooling (Mean captures ambient energy, Max captures peak transient burst)
        feat_mean = tokens.mean(dim=1)
        feat_max = tokens.max(dim=1).values
        embedding = feat_mean + feat_max  # [B, embed_dim]

        # 5. Classification Logits
        logits = self.classifier(embedding)

        return {
            'logits': logits,
            'clipwise_output': logits,
            'embedding': embedding,
            'tokens': tokens
        }
