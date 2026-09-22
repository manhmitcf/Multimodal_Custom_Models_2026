import logging
from typing import Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def init_weights(m: nn.Module) -> None:
    """
    Standardized weight initialization protocol:
      - Depthwise Convolutions: Kaiming Normal (fan_out)
      - Regular / Pointwise Convolutions: Xavier Uniform
      - Linear layers: Xavier Uniform
      - Batch / Layer Normalizations: Weight=1.0, Bias=0.0
      - Biases: 0.0
    """
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        if m.groups > 1 and m.groups == m.in_channels:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        else:
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.ones_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)


# ------------------------------------------------------------------------------
# TẦNG 1: Spectral Stem Block (Nén tần số 2049 -> 33, T=63)
# ------------------------------------------------------------------------------
class SpectralStemBlock(nn.Module):
    """
    Stage 1: Asymmetric Frequency Downsampling Stem (~8.2k params, ~0.15 GFLOPs).
    Compresses 2049 spectral frequency bins to 33 bins across 3 hierarchical cells
    while optimizing temporal cadential frames (T=251 -> 126 -> 63 ~= 31.5 fps):
      - Cell 1: Conv2d(1 -> 32, k=(3, 7), s=(1, 8), p=(1, 3)) -> [B, 32, 251, 257]
      - Cell 2: Depthwise-Separable(32 -> 64, k=(3, 5), s=(2, 4), p=(1, 2)) -> [B, 64, 126, 65]
      - Cell 3: Depthwise-Separable(64 -> 64, k=(3, 3), s=(2, 2), p=(1, 1)) -> [B, 64, 63, 33]
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 64) -> None:
        super().__init__()
        # Cell 1: 8x frequency downsampling (2049 -> 257)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 7), stride=(1, 8), padding=(1, 3), bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        # Cell 2: 4x frequency, 2x temporal downsampling (257 -> 65, 251 -> 126) via Depthwise-Separable Conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 5), stride=(2, 4), padding=(1, 2), groups=32, bias=False),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        # Cell 3: 2x frequency, 2x temporal downsampling (65 -> 33, 126 -> 63) via Depthwise-Separable Conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False),
            nn.Conv2d(64, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3(self.conv2(self.conv1(x)))


# ------------------------------------------------------------------------------
# TẦNG 2: Acoustic Dual-Stream Block (Phân tách Tonal // Transient)
# ------------------------------------------------------------------------------
class AcousticDualStreamBlock(nn.Module):
    """
    Stage 2: Physics-Informed Acoustic Dual-Stream Block (~11.6k params, ~0.04 GFLOPs).
    Decouples underwater acoustic signals into two orthogonal physical components:
      - Tonal Stream: Depthwise temporal smoothing kernel (5, 1) isolates stationary aerator/flow hum.
      - Transient Stream: Dilated spectral kernel (1, 5, d=2) captures broadband feeding click bursts.
      - Pointwise Fusion: 1x1 Conv with residual connection.
    """
    def __init__(self, in_channels: int = 64, out_channels: int = 48) -> None:
        super().__init__()
        # Stream 1: Tonal Path (depthwise temporal smoothing across frames)
        self.tonal_path = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=in_channels, bias=False),
            nn.Conv2d(in_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        # Stream 2: Transient Path (depthwise broadband ultrasonic click detector, dilation=2)
        self.transient_path = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 5), stride=1, padding=(0, 4), dilation=(1, 2), groups=in_channels, bias=False),
            nn.Conv2d(in_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        # Pointwise Fusion & Residual Projection
        self.fusion = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tonal = self.tonal_path(x)
        trans = self.transient_path(x)
        fused = torch.cat([tonal, trans], dim=1)
        return self.act(self.fusion(fused) + self.shortcut(x))


# ------------------------------------------------------------------------------
# TẦNG 3: Cadence Conformer Block (Đo nhịp điệu cá ăn trên chuỗi thời gian)
# ------------------------------------------------------------------------------
def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Guarantees active gradient propagation on small batch sizes by ensuring at least one sample is kept.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize to 0 or 1
    if random_tensor.sum() == 0:
        random_tensor[0] = 1.0
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Zero trainable parameters; identity pass-through during evaluation mode.
    """
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


class ConformerConvModule1D(nn.Module):
    """
    1D Depthwise Convolutional Module for Conformer (~80.3k params):
    LayerNorm -> Pointwise 1x1 -> GLU -> Depthwise 1D (k=15) -> BatchNorm1d -> GELU -> Pointwise 1x1 -> Dropout
    Residual path protected by DropPath (Stochastic Depth).
    """
    def __init__(
        self,
        dim: int = 160,
        kernel_size: int = 15,
        dropout: float = 0.1,
        drop_path: float = 0.0
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            dim, dim, kernel_size=kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, groups=dim, bias=False
        )
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.pointwise2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        res = x
        x = self.layer_norm(x).transpose(1, 2)  # [B, D, T]
        x = self.pointwise1(x)
        x = self.glu(x)
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pointwise2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # [B, T, D]
        return res + self.drop_path(x)


class CadenceConformerBlock(nn.Module):
    """
    Macaron-style Conformer Block (~595.8k params, ~0.06 GFLOPs):
      x = x + DropPath(0.5 * FFN1(x))
      x = x + DropPath(MHSA(x))
      x = x + ConvModule1D(x) (with DropPath on conv branch)
      x = x + DropPath(0.5 * FFN2(x))
      x = LayerNorm(x)
    """
    def __init__(
        self,
        d_model: int = 160,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        drop_path: float = 0.1
    ) -> None:
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Feed-forward 1 (half-step)
        self.norm_ffn1 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout)
        )
        # Multi-Head Self-Attention over temporal cadence sequence (T=63)
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=num_heads, dropout=dropout, batch_first=True)

        # 1D Depthwise Convolution Module (k=15)
        self.conv_module = ConformerConvModule1D(
            dim=d_model, kernel_size=15, dropout=dropout, drop_path=drop_path
        )

        # Feed-forward 2 (half-step)
        self.norm_ffn2 = nn.LayerNorm(d_model)
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout)
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Half-step FFN 1
        x = x + self.drop_path(0.5 * self.ffn1(self.norm_ffn1(x)))
        # 2. Multi-Head Self-Attention
        norm_x = self.norm_attn(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + self.drop_path(attn_out)
        # 3. Depthwise 1D Convolution Module
        x = self.conv_module(x)
        # 4. Half-step FFN 2
        x = x + self.drop_path(0.5 * self.ffn2(self.norm_ffn2(x)))
        return self.final_norm(x)


# ------------------------------------------------------------------------------
# ĐẦU XUẤT: Decoupled Multi-Feature Head (Đồng bộ chuẩn Tournament Fusion)
# ------------------------------------------------------------------------------
class DecoupledFeatureHead(nn.Module):
    """
    Decoupled Multi-Feature Audio Projection Head (~108.2k params).
    Extracts 5 decoupled acoustic representations from the Conformer token sequence:
      1. f_frequency: Global temporal mean energy profile [B, embed_dim]
      2. f_rhythm: Final temporal cadence state token [B, embed_dim]
      3. f_burst_a: Peak acoustic burst contrast (Max - Mean) [B, embed_dim]
      4. f_audio: Joint normalized acoustic embedding [B, embed_dim]
      5. tokens_audio: Temporal token sequence for tournament interaction [B, num_tokens, embed_dim]
    """
    def __init__(self, d_model: int = 160, embed_dim: int = 224, num_tokens: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.proj_freq = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.Dropout(dropout)
        )
        self.proj_rhythm = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.Dropout(dropout)
        )
        self.proj_burst = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm_audio = nn.LayerNorm(embed_dim)

    def forward(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Global Frequency Energy Profile (Temporal Mean Pooling)
        mean_token = tokens.mean(dim=1)  # [B, d_model]
        f_frequency = self.proj_freq(mean_token)

        # 2. Feeding Rhythm / Cadence (Final state token)
        f_rhythm = self.proj_rhythm(tokens[:, -1, :])

        # 3. Acoustic Burst Contrast: Peak token minus Mean token
        max_token = tokens.max(dim=1).values
        f_burst_a = self.proj_burst(max_token - mean_token)

        # 4. Joint Acoustic Embedding (Normalized Scale Alignment)
        f_audio = self.norm_audio(f_frequency + f_rhythm + f_burst_a)

        # 5. Tournament Token Sequence Alignment
        tokens_audio = f_audio.unsqueeze(1).repeat(1, self.num_tokens, 1)

        return f_audio, f_frequency, f_rhythm, f_burst_a, tokens_audio


# ------------------------------------------------------------------------------
# MẠNG TOÀN DIỆN: PhyConformerBackbone (~0.99M params, ~0.29 GFLOPs)
# ------------------------------------------------------------------------------
class PhyConformerBackbone(nn.Module):
    """
    Physics-Informed Tri-Tier Audio Conformer Backbone (~0.994M params, ~0.286 GFLOPs).
    Replaces legacy AudioMLPBackbone (1.166M) with end-to-end 2D STFT spatiotemporal modeling:
      - Tier 1: SpectralStemBlock: Hierarchical 2049 -> 33 bins frequency downsampling (T=63).
      - Tier 2: AcousticDualStreamBlock: Orthogonal Tonal (5x1) // Transient (1x5, d=2) stream separation.
      - Tokenizer: Linear(1584 -> 160) + LayerNorm + Learnable Positional Encoding.
      - Tier 3: CadenceConformerBlock: Macaron MHSA + 1D ConvModule (k=15) for temporal feeding rhythm.
      - Head: DecoupledFeatureHead: Extracts 5 orthogonal outputs for tournament decision fusion.
    """
    def __init__(
        self,
        in_features: int = 2049,
        embed_dim: int = 224,
        num_tokens: int = 2,
        d_model: int = 160,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.drop_path = drop_path

        # 1. Tier 1: Spectral Stem Block (Nén tần số 2049 -> 33)
        self.tier1_stem = SpectralStemBlock(in_channels=1, out_channels=64)

        # 2. Tier 2: Acoustic Dual-Stream Block (Tách Tonal // Transient)
        self.tier2_acoustic = AcousticDualStreamBlock(in_channels=64, out_channels=48)

        # Spatial-to-Token Projector (48 channels * 33 bins = 1584 -> d_model=160)
        self.flatten_dim = 48 * 33
        self.tokenizer = nn.Sequential(
            nn.Linear(self.flatten_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        # Learnable Positional Embedding for up to 100 temporal frames (nominal T=63)
        self.pos_emb = nn.Parameter(torch.zeros(1, 100, d_model))

        # 3. Tier 3: Cadence Conformer Block (Đo nhịp điệu cá ăn trên chuỗi thời gian)
        self.tier3_conformer = CadenceConformerBlock(
            d_model=d_model,
            num_heads=4,
            mlp_ratio=4,
            dropout=dropout,
            drop_path=drop_path
        )

        # 4. Decoupled Multi-Feature Head
        self.head = DecoupledFeatureHead(
            d_model=d_model,
            embed_dim=embed_dim,
            num_tokens=num_tokens,
            dropout=dropout
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply deterministic weight initialization across all sub-modules."""
        self.apply(init_weights)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(
        self, x: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input audio tensor. Supports:
               - 4D Spectrogram [B, 1, T, 2049]
               - 3D Spectrogram [B, T, 2049]
               - 2D Spectral vector [B, 2049] (automatic temporal expansion)
               - Dict containing 'spectrogram' or 'spec_vector'

        Returns:
            f_audio: Joint acoustic embedding [B, embed_dim]
            f_frequency: Spectral frequency profile [B, embed_dim]
            f_rhythm: Temporal cadence feature [B, embed_dim]
            f_burst_a: Acoustic burst contrast [B, embed_dim]
            tokens_audio: Sequence of audio tokens [B, num_tokens, embed_dim]
        """
        # Unpack input representations
        if isinstance(x, dict):
            if "spectrogram" in x and x["spectrogram"] is not None:
                x_spec = x["spectrogram"]
            elif "spec_vector" in x:
                x_spec = x["spec_vector"]
            else:
                x_spec = next(iter(x.values()))
        elif isinstance(x, (tuple, list)):
            x_spec = x[0]
        elif isinstance(x, torch.Tensor):
            x_spec = x
        else:
            raise TypeError(f"Unsupported audio input type: {type(x)}")

        # Normalize tensor shape to [B, 1, T, 2049]
        if x_spec.ndim == 4:
            pass
        elif x_spec.ndim == 3:
            x_spec = x_spec.unsqueeze(1)
        elif x_spec.ndim == 2:
            x_spec = x_spec.unsqueeze(1).unsqueeze(1).repeat(1, 1, 251, 1)
        else:
            x_spec = x_spec.view(x_spec.size(0), 1, 1, -1).repeat(1, 1, 251, 1)

        # Guard frequency dimension boundary
        if x_spec.size(-1) > self.in_features:
            x_spec = x_spec[..., :self.in_features]
        elif x_spec.size(-1) < self.in_features:
            x_spec = F.pad(x_spec, (0, self.in_features - x_spec.size(-1)))

        # 1. Tier 1: Asymmetric Frequency Downsampling (2049 -> 33, T: 251 -> 63)
        feat_stem = self.tier1_stem(x_spec)  # [B, 64, T_stem, 33]

        # 2. Tier 2: Physics-Informed Tonal // Transient Separation
        feat_acoustic = self.tier2_acoustic(feat_stem)  # [B, 48, T_stem, 33]

        # Spatial-to-Token Serialization: [B, 48, T, 33] -> [B, T, 48 * 33] -> [B, T, d_model]
        B, C, T, F_prime = feat_acoustic.shape
        tokens = feat_acoustic.permute(0, 2, 1, 3).reshape(B, T, C * F_prime)
        tokens = self.tokenizer(tokens)

        # Add temporal positional encoding
        tokens = tokens + self.pos_emb[:, :T, :]

        # 3. Tier 3: Cadence Conformer Sequence Modeling
        tokens = self.tier3_conformer(tokens)  # [B, T, d_model]

        # 4. Decoupled Multi-Feature Head
        return self.head(tokens)


# Canonical AudioBackbone aliases for seamless codebase integration
AudioBackbone = PhyConformerBackbone
AudioMLPBackbone = PhyConformerBackbone
