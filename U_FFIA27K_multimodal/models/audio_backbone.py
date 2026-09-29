import torch
import torch.nn as nn
import torch.nn.functional as F


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
# 5. Audio Backbone Factory
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
    else:
        raise ValueError(
            f"Unsupported audio backbone '{name}'. Must be one of ['mlp', 'bcresnet8', 'bigru', 'conformer']."
        )
