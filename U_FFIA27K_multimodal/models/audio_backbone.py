import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List


def init_layer(layer: nn.Module) -> None:
    """Initialize a Linear or Convolutional layer."""
    if hasattr(layer, 'weight') and layer.weight is not None:
        nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)


# =========================================================================
# 1. Qualcomm AI Research BC-ResNet-8 (Interspeech 2021)
# Official Implementation: Broadcasted Residual Learning for Audio
# =========================================================================

class SubSpectralNorm(nn.Module):
    """
    Sub-Spectral Normalization (SSN) from Qualcomm AI Research.
    Splits the frequency dimension into sub-spectral groups and normalizes each group independently.
    """
    def __init__(self, num_features: int, spec_groups: int = 16, affine: str = "Sub", batch: bool = True, dim: int = 2):
        super().__init__()
        self.spec_groups = spec_groups
        self.affine_all = False
        affine_norm = False
        if affine == "Sub":
            affine_norm = True
        elif affine == "All":
            self.affine_all = True
            self.weight = nn.Parameter(torch.ones((1, num_features, 1, 1)))
            self.bias = nn.Parameter(torch.zeros((1, num_features, 1, 1)))
        if batch:
            self.ssnorm = nn.BatchNorm2d(num_features * spec_groups, affine=affine_norm)
        else:
            self.ssnorm = nn.InstanceNorm2d(num_features * spec_groups, affine=affine_norm)
        self.sub_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sub_dim in (3, -1):
            x = x.transpose(2, 3).contiguous()
        b, c, h, w = x.size()
        assert h % self.spec_groups == 0, f"Frequency height {h} must be divisible by spec_groups {self.spec_groups}"
        x = x.view(b, c * self.spec_groups, h // self.spec_groups, w)
        x = self.ssnorm(x)
        x = x.view(b, c, h, w)
        if self.affine_all:
            x = x * self.weight + self.bias
        if self.sub_dim in (3, -1):
            x = x.transpose(2, 3).contiguous()
        return x


class ConvBNReLU(nn.Module):
    """Convolution-Normalization-Activation block supporting SubSpectralNorm and Dilated Convolutions."""
    def __init__(
        self,
        in_plane: int,
        out_plane: int,
        idx: int,
        kernel_size=3,
        stride=1,
        groups: int = 1,
        use_dilation: bool = False,
        activation: bool = True,
        swish: bool = False,
        BN: bool = True,
        ssn: bool = False,
    ):
        super().__init__()

        def get_padding(k_size, dilated):
            rate = 1
            pad_len = (k_size - 1) // 2
            if dilated and k_size > 1:
                rate = int(2**idx)
                pad_len = rate * pad_len
            return pad_len, rate

        self.idx = idx

        if isinstance(kernel_size, (list, tuple)):
            padding = []
            rate = []
            for k in kernel_size:
                p_len, r = get_padding(k, use_dilation)
                rate.append(r)
                padding.append(p_len)
        else:
            padding, rate = get_padding(kernel_size, use_dilation)

        layers = [
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding, rate, groups, bias=False)
        ]
        if ssn:
            layers.append(SubSpectralNorm(out_plane, 5))
        elif BN:
            layers.append(nn.BatchNorm2d(out_plane))
        if swish:
            layers.append(nn.SiLU(True))
        elif activation:
            layers.append(nn.ReLU(True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BCResBlock(nn.Module):
    """Broadcasted Residual Block decomposing 2D and 1D temporal convolutions."""
    def __init__(self, in_plane: int, out_plane: int, idx: int, stride):
        super().__init__()
        self.transition_block = in_plane != out_plane
        kernel_size = (3, 3)

        # 2D part (f2)
        layers = []
        if self.transition_block:
            layers.append(ConvBNReLU(in_plane, out_plane, idx, 1, 1))
            in_plane = out_plane
        layers.append(
            ConvBNReLU(
                in_plane,
                out_plane,
                idx,
                (kernel_size[0], 1),
                (stride[0], 1),
                groups=in_plane,
                ssn=True,
                activation=False,
            )
        )
        self.f2 = nn.Sequential(*layers)
        self.avg_gpool = nn.AdaptiveAvgPool2d((1, None))

        # 1D part (f1)
        self.f1 = nn.Sequential(
            ConvBNReLU(
                out_plane,
                out_plane,
                idx,
                (1, kernel_size[1]),
                (1, stride[1]),
                groups=out_plane,
                swish=True,
                use_dilation=True,
            ),
            nn.Conv2d(out_plane, out_plane, 1, bias=False),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.f2(x)
        aux_2d_res = x
        x = self.avg_gpool(x)

        x = self.f1(x)
        x = x + aux_2d_res
        if not self.transition_block:
            x = x + shortcut
        x = F.relu(x, True)
        return x


def BCBlockStage(num_layers: int, last_channel: int, cur_channel: int, idx: int, use_stride: bool) -> nn.ModuleList:
    stage = nn.ModuleList()
    channels = [last_channel] + [cur_channel] * num_layers
    for i in range(num_layers):
        stride = (2, 1) if use_stride and i == 0 else (1, 1)
        stage.append(BCResBlock(channels[i], channels[i + 1], idx, stride))
    return stage


class BCResNets(nn.Module):
    """Qualcomm official BC-ResNet model."""
    def __init__(self, base_c: int = 64, num_classes: int = 224):
        super().__init__()
        self.num_classes = num_classes
        self.n = [2, 2, 4, 4]
        self.c = [
            base_c * 2,
            base_c,
            int(base_c * 1.5),
            base_c * 2,
            int(base_c * 2.5),
            base_c * 4,
        ]
        self.s = [1, 2]
        self._build_network()

    def _build_network(self) -> None:
        self.cnn_head = nn.Sequential(
            nn.Conv2d(1, self.c[0], 5, (2, 1), 2, bias=False),
            nn.BatchNorm2d(self.c[0]),
            nn.ReLU(True),
        )
        self.BCBlocks = nn.ModuleList([])
        for idx, n in enumerate(self.n):
            use_stride = idx in self.s
            self.BCBlocks.append(BCBlockStage(n, self.c[idx], self.c[idx + 1], idx, use_stride))

        self.classifier = nn.Sequential(
            nn.Conv2d(
                self.c[-2], self.c[-2], (5, 5), bias=False, groups=self.c[-2], padding=(0, 2)
            ),
            nn.Conv2d(self.c[-2], self.c[-1], 1, bias=False),
            nn.BatchNorm2d(self.c[-1]),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.c[-1], self.num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_head(x)
        for i, num_modules in enumerate(self.n):
            for j in range(num_modules):
                x = self.BCBlocks[i][j](x)
        x = self.classifier(x)
        x = x.view(-1, x.shape[1])
        return x


class BCResNet8AudioBackbone(nn.Module):
    """
    Qualcomm AI Research BC-ResNet-8 Audio Backbone (~0.38M params, Interspeech 2021).
    Broadcasted Residual Network with Sub-Spectral Normalization.
    Accepts 2D STFT spectrogram [B, T, 2049], adapts frequency bins to 40 via
    zero-parameter adaptive pooling, and produces joint acoustic embedding f_audio [B, 224].
    """
    requires_2d: bool = True

    def __init__(
        self,
        in_features: int = 2049,
        embed_dim: int = 224,
        base_c: int = 64,
        **kwargs
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.embed_dim = embed_dim
        self.base_c = base_c

        # Zero-parameter frequency adapter: 2049 STFT bins -> 40 native BC-ResNet frequency bins
        self.freq_adapter = nn.AdaptiveAvgPool2d((40, None))

        # Qualcomm official BCResNets architecture initialized from scratch
        self.net = BCResNets(base_c=base_c, num_classes=embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: STFT spectrogram [B, T, 2049] or [B, 2049].
        Returns:
            f_audio: Joint acoustic embedding [B, embed_dim=224].
        """
        if x.ndim == 3:
            # [B, T, F] -> [B, 1, F, T]
            x = x.transpose(1, 2).unsqueeze(1)
        elif x.ndim == 2:
            # Fallback for [B, F]
            x = x.unsqueeze(1).unsqueeze(-1)

        x = self.freq_adapter(x)  # [B, 1, 40, T]
        f_audio = self.net(x)      # [B, embed_dim=224]
        return f_audio


# =========================================================================
# 2. CRNN-BiGRU Audio Backbone (DCASE Task 4 Sequence Baseline)
# =========================================================================

class CRNNBiGRUAudioBackbone(nn.Module):
    """
    CRNN-BiGRU Audio Backbone (~0.65M params).
    SOTA Recurrent Audio Sequence model based on DCASE Task 4 CRNN baseline.
    Combines linear frequency projection (2049 -> 128) with a 2-layer Bidirectional GRU
    (hidden_size=112 -> 224 output) and temporal mean pooling to extract f_audio [B, 224].
    """
    requires_2d: bool = True

    def __init__(
        self,
        in_features: int = 2049,
        proj_dim: int = 128,
        hidden_dim: int = 112,
        num_layers: int = 2,
        embed_dim: int = 224,
        dropout: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # Frequency projection: 2049 -> proj_dim (128)
        self.proj = nn.Sequential(
            nn.Linear(in_features, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Bidirectional GRU: hidden_dim * 2 = 112 * 2 = 224
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: STFT spectrogram [B, T, 2049] or [B, 2049].
        Returns:
            f_audio: Joint acoustic embedding [B, embed_dim=224].
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)  # [B, 1, 2049]

        # Project frequency dimension
        h = self.proj(x)  # [B, T, proj_dim]

        # Bidirectional recurrent encoding
        gru_out, _ = self.gru(h)  # [B, T, hidden_dim * 2 = 224]

        # Temporal mean pooling
        pooled = gru_out.mean(dim=1)  # [B, 224]
        f_audio = self.out_norm(pooled)
        return f_audio


# =========================================================================
# 3. Conformer Audio Backbone (Google Interspeech 2020)
# =========================================================================

class ConformerAudioBackbone(nn.Module):
    """
    Conformer Audio Backbone (~0.79M params, Google Interspeech 2020).
    Convolution-augmented Transformer architecture for acoustic representation learning.
    Combines multi-head self-attention with depthwise convolutions to model both
    global context and local acoustic correlations.
    """
    requires_2d: bool = True

    def __init__(
        self,
        in_features: int = 2049,
        input_dim: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 256,
        num_layers: int = 2,
        depthwise_conv_kernel_size: int = 15,
        dropout: float = 0.1,
        embed_dim: int = 224,
        **kwargs
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.embed_dim = embed_dim

        # Input projection: 2049 -> 128
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        from torchaudio.models import Conformer
        self.conformer = Conformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout
        )

        # Output projection to match embed_dim (224)
        self.out_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: STFT spectrogram [B, T, 2049] or [B, 2049].
        Returns:
            f_audio: Joint acoustic embedding [B, embed_dim=224].
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)  # [B, 1, 2049]

        # Project frequency dimension
        h = self.in_proj(x)  # [B, T, 128]

        lengths = torch.full((h.size(0),), h.size(1), dtype=torch.long, device=h.device)
        conf_out, _ = self.conformer(h, lengths)  # [B, T, 128]

        # Temporal mean pooling
        pooled = conf_out.mean(dim=1)  # [B, 128]
        f_audio = self.out_proj(pooled)  # [B, 224]
        return f_audio


# =========================================================================
# 4. Existing High-Resolution STFT MLP Backbone (~1.17M params)
# =========================================================================

class AudioMLPBackbone(nn.Module):
    """
    High-Resolution STFT Audio MLP Backbone (~1.17M params).
    Processes 2049-dimensional TKEO-STFT spectral vectors [B, 2049]:
      - Layer 1: Linear(2049 -> 512) + LayerNorm(512) + GELU + Dropout(0.1)
      - Layer 2: Linear(512 -> 224) + LayerNorm(224) -> f_audio [B, 224]
    """
    requires_2d: bool = False

    def __init__(
        self,
        in_features: int = 2049,
        hidden_dim: int = 512,
        embed_dim: int = 224,
        dropout: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # 1. 2-layer MLP for 2049 STFT vector
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: STFT spectral feature vector [B, 2049]

        Returns:
            f_audio: Joint acoustic embedding [B, embed_dim=224]
        """
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
            if x.size(-1) != self.in_features:
                x = x[:, :self.in_features]

        h = self.dropout(self.act(self.ln1(self.fc1(x))))
        f_audio = self.ln2(self.fc2(h))
        return f_audio


# =========================================================================
# 5. Pure PyTorch Selective Scan Core (S6) & Bidirectional Mamba (AuM / ViM)
# Authentic Implementation based on Gu & Dao (2023) and KAIST AuM (2024)
# =========================================================================

def selective_scan_pure_pytorch(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pure PyTorch vectorized implementation of Selective State Space Model (S6).
    Discretization:
        deltaA_t = exp(delta_t * A)         [B, D, N]
        deltaB_t = delta_t * B_t           [B, D, N]
    Recurrent State Scan:
        h_t = deltaA_t * h_{t-1} + deltaB_t * u_t
        y_t = sum_n (h_t * C_t) + D * u_t
    Computations are explicitly executed in float32 for numerical stability.
    """
    orig_dtype = u.dtype
    u = u.float()
    delta = delta.float()
    A = A.float()
    B = B.float()
    C = C.float()

    batch_size, seq_len, d_in = u.shape
    d_state = A.shape[-1]

    # Pre-compute continuous-to-discrete transition
    # delta: [B, L, D, 1], A: [1, 1, D, N] -> deltaA: [B, L, D, N]
    deltaA = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, d_in, d_state))
    # delta: [B, L, D, 1], B: [B, L, 1, N] -> deltaB: [B, L, D, N]
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
    deltaB_u = deltaB * u.unsqueeze(-1)

    h = torch.zeros(batch_size, d_in, d_state, device=u.device, dtype=torch.float32)
    ys = []

    for t in range(seq_len):
        h = deltaA[:, t] * h + deltaB_u[:, t]
        y_t = torch.sum(h * C[:, t].unsqueeze(1), dim=-1)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)

    if D is not None:
        y = y + u * D.view(1, 1, d_in).float()

    return y.to(dtype=orig_dtype)


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba Block (AuM / ViM).
    Processes sequence in both forward (t=1..L) and backward (t=L..1) directions.
    Features:
      - Low-rank factorization for delta (dt_rank = ceil(d_model / 16))
      - HiPPO / S4D diagonal initialization for A (A = -exp(A_log))
      - Inverse softplus bias initialization for dt_proj
      - Independent forward and backward SSM streams with parameter exemption from weight decay
      - Multiplicative SiLU gating
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank is None else dt_rank

        # In-projection to SSM branch (u) and gate branch (z)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        # 1D Causal Depthwise Conv on u
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # Forward SSM projections
        self.x_proj_fwd = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj_fwd = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Backward SSM projections
        self.x_proj_bwd = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj_bwd = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S4D / HiPPO A parameter initialization
        A_fwd = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log_fwd = nn.Parameter(torch.log(A_fwd))
        self.A_log_fwd._no_weight_decay = True

        A_bwd = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log_bwd = nn.Parameter(torch.log(A_bwd))
        self.A_log_bwd._no_weight_decay = True

        # D skip parameter
        self.D_fwd = nn.Parameter(torch.ones(self.d_inner))
        self.D_fwd._no_weight_decay = True

        self.D_bwd = nn.Parameter(torch.ones(self.d_inner))
        self.D_bwd._no_weight_decay = True

        # Initialize dt_proj with Inverse Softplus
        self._init_dt_proj(self.dt_proj_fwd, dt_min, dt_max, dt_init_floor)
        self._init_dt_proj(self.dt_proj_bwd, dt_min, dt_max, dt_init_floor)

        # Out-projection back to d_model
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def _init_dt_proj(self, dt_proj: nn.Linear, dt_min: float, dt_max: float, dt_init_floor: float) -> None:
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)

        # Inverse softplus for delta bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        dt_proj.bias._no_weight_decay = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D]
        Returns:
            out: Output tensor [B, L, D]
        """
        B, L, D = x.shape

        # 1. Project to u and z
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)

        # 2. 1D Causal Conv over u
        u_conv = self.conv1d(u.transpose(1, 2))[:, :, :L].transpose(1, 2)
        u_conv = F.silu(u_conv)

        # 3. Forward SSM
        A_fwd = -torch.exp(self.A_log_fwd.float())
        x_proj_f = self.x_proj_fwd(u_conv)
        dt_f, B_f, C_f = torch.split(x_proj_f, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta_f = F.softplus(self.dt_proj_fwd(dt_f))
        y_fwd = selective_scan_pure_pytorch(u_conv, delta_f, A_fwd, B_f, C_f, self.D_fwd)

        # 4. Backward SSM
        u_conv_bwd = torch.flip(u_conv, dims=[1])
        A_bwd = -torch.exp(self.A_log_bwd.float())
        x_proj_b = self.x_proj_bwd(u_conv_bwd)
        dt_b, B_b, C_b = torch.split(x_proj_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta_b = F.softplus(self.dt_proj_bwd(dt_b))
        y_bwd = selective_scan_pure_pytorch(u_conv_bwd, delta_b, A_bwd, B_b, C_b, self.D_bwd)
        y_bwd = torch.flip(y_bwd, dims=[1])

        # 5. Merge bidirectional paths
        y = y_fwd + y_bwd

        # 6. Multiplicative Gating with z
        y = y * F.silu(z)

        # 7. Out-projection
        out = self.out_proj(y)
        return out


class BiMambaLayer(nn.Module):
    """Pre-LayerNorm residual wrapper for BiMambaBlock."""
    def __init__(self, d_model: int, dropout: float = 0.1, **kwargs) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.block = BiMambaBlock(d_model=d_model, **kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.block(self.norm(x)))


class BiMambaAudioBackbone(nn.Module):
    """
    Bidirectional Audio Mamba (AuM) Sequence Backbone (~0.69M params).
    Inspired by KAIST Audio Mamba (AuM, arXiv:2406.03344) & Vision Mamba (ICML 2024).

    Input: [B, T=251, F=2049] (251 time frames, 2049 frequency bins)
      - Linear Stem: Linear(2049 -> 128) + LayerNorm(128) + GELU + Dropout(0.1)
      - 3x BiMambaLayer(d_model=128, d_state=16, d_conv=4, expand=2)
      - Final LayerNorm(128)
      - Global Temporal Mean Pooling: [B, 251, 128] -> [B, 128]
      - Head: Linear(128 -> 224) + LayerNorm(224) -> f_audio [B, 224]
    """
    requires_2d: bool = True

    def __init__(
        self,
        in_features: int = 2049,
        d_model: int = 128,
        embed_dim: int = 224,
        num_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.d_model = d_model
        self.embed_dim = embed_dim

        # 1. Frequency projection stem: 2049 -> d_model
        self.stem = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Stack of BiMamba layers
        self.layers = nn.ModuleList([
            BiMambaLayer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # 3. Output projection head: d_model -> embed_dim (224)
        self.head = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.stem.modules():
            if isinstance(m, nn.Linear):
                init_layer(m)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                init_layer(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spectrogram tensor [B, T=251, F=2049] or [B, 2049]
        Returns:
            f_audio: Embedding [B, embed_dim=224]
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 4:
            x = x.squeeze(1)

        # Stem projection: [B, T, F] -> [B, T, d_model]
        h = self.stem(x)

        # BiMamba sequence modeling
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        # Global Temporal Mean Pooling: [B, T, d_model] -> [B, d_model]
        pooled = h.mean(dim=1)

        # Final projection to 224
        f_audio = self.head(pooled)
        return f_audio


# =========================================================================
# 6. Dual-Path Time-Frequency Mamba (TF-Mamba)
# Authentic Implementation based on Interspeech 2025 (arXiv:2409.05034) & ASCMamba
# =========================================================================

class DualPathTFMambaBlock(nn.Module):
    """
    Dual-Path Time-Frequency Mamba Block.
    Alternates:
      1. Intra-frame Frequency Scan (F-BiMamba): scans across F frequency bins to capture harmonic structure
      2. Inter-frame Temporal Scan (T-BiMamba): scans across T time frames to capture feeding burst dynamics
    """
    def __init__(
        self,
        channels: int = 48,
        d_state: int = 16,
        dt_rank: int = 4,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = channels

        # Frequency Path: intra-frame harmonic modeling
        self.norm_f = nn.GroupNorm(4, channels)
        self.mamba_f = BiMambaBlock(
            d_model=channels,
            d_state=d_state,
            dt_rank=dt_rank,
            d_conv=d_conv,
            expand=expand,
        )

        # Time Path: inter-frame cadence modeling
        self.norm_t = nn.GroupNorm(4, channels)
        self.mamba_t = BiMambaBlock(
            d_model=channels,
            d_state=d_state,
            dt_rank=dt_rank,
            d_conv=d_conv,
            expand=expand,
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 2D feature map [B, C, T, F]
        Returns:
            out: 2D feature map [B, C, T, F]
        """
        B, C, T, F = x.shape

        # 1. Frequency Path (intra-frame):
        # Permute & reshape to (B * T, F, C)
        x_f = x.permute(0, 2, 3, 1).contiguous().view(B * T, F, C)
        x_f_norm = self.norm_f(x_f.transpose(1, 2)).transpose(1, 2)
        out_f = self.drop(self.mamba_f(x_f_norm))
        x = x + out_f.view(B, T, F, C).permute(0, 3, 1, 2)

        # 2. Time Path (inter-frame):
        # Permute & reshape to (B * F, T, C)
        x_t = x.permute(0, 3, 2, 1).contiguous().view(B * F, T, C)
        x_t_norm = self.norm_t(x_t.transpose(1, 2)).transpose(1, 2)
        out_t = self.drop(self.mamba_t(x_t_norm))
        x = x + out_t.view(B, F, T, C).permute(0, 3, 2, 1)

        return x


class TFMambaAudioBackbone(nn.Module):
    """
    Dual-Path Time-Frequency Mamba Audio Backbone (~0.18M params).
    Inspired by Interspeech 2025 TF-Mamba (arXiv:2409.05034).

    Input: [B, 251, 2049] (viewed as [B, 1, 251, 2049])
      - 2D Conv Stem:
          Conv2d(1 -> 32, k=(5, 9), s=(2, 4), p=(2, 4)) + BN + GELU
          Conv2d(32 -> channels, k=(5, 9), s=(2, 4), p=(2, 4)) + BN + GELU
          AdaptiveAvgPool2d((32, 32)) -> [B, channels, 32, 32]
      - 2x DualPathTFMambaBlock(channels=48, d_state=16, dt_rank=4)
      - 2D Global Average Pooling: [B, channels, 32, 32] -> [B, channels]
      - Head: Linear(channels -> 224) + LayerNorm(224) -> f_audio [B, 224]
    """
    requires_2d: bool = True

    def __init__(
        self,
        in_features: int = 2049,
        channels: int = 48,
        embed_dim: int = 224,
        num_stages: int = 2,
        d_state: int = 16,
        dt_rank: int = 4,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.channels = channels
        self.embed_dim = embed_dim

        # 1. 2D Convolutional Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 9), stride=(2, 4), padding=(2, 4), bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, channels, kernel_size=(5, 9), stride=(2, 4), padding=(2, 4), bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((32, 32)),
        )

        # 2. Dual-Path TF-Mamba Stages
        self.stages = nn.ModuleList([
            DualPathTFMambaBlock(
                channels=channels,
                d_state=d_state,
                dt_rank=dt_rank,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_stages)
        ])

        # 3. Global 2D Pooling & Output Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(channels, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.stem.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                init_layer(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spectrogram [B, T=251, F=2049] or [B, 2049]
        Returns:
            f_audio: Joint acoustic embedding [B, embed_dim=224]
        """
        if x.ndim == 2:
            B = x.size(0)
            x = x.view(B, 1, 1, self.in_features)
        elif x.ndim == 3:
            x = x.unsqueeze(1)  # [B, 1, T, F]

        # 1. 2D Conv Stem -> [B, channels, 32, 32]
        feat = self.stem(x)

        # 2. Dual-Path TF-Mamba Stages
        for stage in self.stages:
            feat = stage(feat)

        # 3. Global Pooling & Projection
        pooled = self.pool(feat).flatten(1)  # [B, channels]
        f_audio = self.head(pooled)          # [B, embed_dim=224]
        return f_audio


# =========================================================================
# 7. Audio Backbone Factory
# =========================================================================

def build_audio_backbone(
    name: str = "mlp",
    in_features: int = 2049,
    embed_dim: int = 224,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    Factory function to construct audio backbones for ablation studies:
      - 'mlp': High-Resolution STFT MLP (~1.17M params)
      - 'bcresnet8': Qualcomm BC-ResNet-8 (~0.38M params)
      - 'bigru': CRNN-BiGRU Sequence Baseline (~0.65M params)
      - 'conformer': Conformer Attention-CNN (~0.79M params)
      - 'bimamba': Bidirectional Audio Mamba Sequence Backbone (~0.69M params)
      - 'tfmamba': Dual-Path Time-Frequency Mamba Backbone (~0.18M params)
    """
    norm_name = str(name).lower().strip()
    if norm_name in ("mlp", "stft_mlp"):
        hidden_dim = kwargs.get("hidden_dim", 512)
        return AudioMLPBackbone(
            in_features=in_features,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            dropout=dropout,
            **kwargs
        )
    elif norm_name in ("bcresnet8", "bc_resnet8", "bcresnet", "bc_resnet"):
        base_c = kwargs.get("base_c", 64)
        return BCResNet8AudioBackbone(
            in_features=in_features,
            embed_dim=embed_dim,
            base_c=base_c,
            **kwargs
        )
    elif norm_name in ("bigru", "crnn_bigru", "crnn", "gru"):
        return CRNNBiGRUAudioBackbone(
            in_features=in_features,
            embed_dim=embed_dim,
            dropout=dropout,
            **kwargs
        )
    elif norm_name in ("conformer",):
        return ConformerAudioBackbone(
            in_features=in_features,
            embed_dim=embed_dim,
            dropout=dropout,
            **kwargs
        )
    elif norm_name in ("bimamba", "audio_mamba", "aum", "mamba"):
        d_model = kwargs.get("d_model", 128)
        num_layers = kwargs.get("num_layers", 3)
        return BiMambaAudioBackbone(
            in_features=in_features,
            d_model=d_model,
            embed_dim=embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs
        )
    elif norm_name in ("tfmamba", "tf_mamba", "dual_path_mamba", "time_frequency_mamba"):
        channels = kwargs.get("channels", 48)
        num_stages = kwargs.get("num_stages", 2)
        return TFMambaAudioBackbone(
            in_features=in_features,
            channels=channels,
            embed_dim=embed_dim,
            num_stages=num_stages,
            dropout=dropout,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unsupported audio backbone '{name}'. Must be one of ['mlp', 'bcresnet8', 'bigru', 'conformer', 'bimamba', 'tfmamba']."
        )


__all__ = [
    "init_layer",
    "AudioMLPBackbone",
    "BCResNet8AudioBackbone",
    "CRNNBiGRUAudioBackbone",
    "ConformerAudioBackbone",
    "BiMambaAudioBackbone",
    "TFMambaAudioBackbone",
    "build_audio_backbone",
]
