import torch
import torch.nn as nn
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def init_layer(layer: nn.Module) -> None:
    """Initialize a Linear or Convolutional layer with Xavier Uniform."""
    if hasattr(layer, 'weight') and layer.weight is not None:
        nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)


class ModularAudioMLPBackbone(nn.Module):
    """
    Parametric Audio MLP Backbone for Ablation Studies and High-Resolution TKEO-STFT vectors [B, 2049]:
      - Hidden Layers: 0, 1, 2, or 3 layers with pattern Linear -> LayerNorm -> GELU -> Dropout(p)
      - Final Projection: Linear(d_last, embed_dim) -> LayerNorm(embed_dim) -> f_audio [B, embed_dim]
    """
    def __init__(
        self,
        in_features: int = 2049,
        hidden_dims: Optional[List[int]] = None,
        embed_dim: int = 384,
        dropout: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dims = list(hidden_dims) if hidden_dims is not None else [768]
        self.embed_dim = embed_dim
        self.dropout_rate = float(dropout)

        layers = []
        prev_dim = in_features

        # 1. Hidden Layers
        for h_dim in self.hidden_dims:
            fc = nn.Linear(prev_dim, h_dim)
            ln = nn.LayerNorm(h_dim)
            act = nn.GELU()
            drop = nn.Dropout(self.dropout_rate)
            layers.extend([fc, ln, act, drop])
            prev_dim = h_dim

        # 2. Final Projection Layer to embed_dim (Canonical pattern: Linear -> LayerNorm)
        final_fc = nn.Linear(prev_dim, embed_dim)
        final_ln = nn.LayerNorm(embed_dim)
        layers.extend([final_fc, final_ln])

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                init_layer(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: STFT spectral feature vector [B, 2049]

        Returns:
            f_audio: Joint acoustic embedding [B, embed_dim]
        """
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
            if x.size(-1) != self.in_features:
                x = x[:, :self.in_features]

        return self.net(x)


class AudioMLPBackbone(ModularAudioMLPBackbone):
    """
    Legacy 2-layer High-Resolution STFT Audio MLP Backbone (~1.17M params).
    Maintains 100% backward compatibility for existing checkpoints and MultimodalSOTANet.
    """
    def __init__(
        self,
        in_features: int = 2049,
        hidden_dim: int = 512,
        embed_dim: int = 224,
        dropout: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__(
            in_features=in_features,
            hidden_dims=[hidden_dim],
            embed_dim=embed_dim,
            dropout=dropout,
            **kwargs
        )
        self.hidden_dim = hidden_dim

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key_map = {
            f"{prefix}fc1.weight": f"{prefix}net.0.weight",
            f"{prefix}fc1.bias": f"{prefix}net.0.bias",
            f"{prefix}ln1.weight": f"{prefix}net.1.weight",
            f"{prefix}ln1.bias": f"{prefix}net.1.bias",
            f"{prefix}fc2.weight": f"{prefix}net.4.weight",
            f"{prefix}fc2.bias": f"{prefix}net.4.bias",
            f"{prefix}ln2.weight": f"{prefix}net.5.weight",
            f"{prefix}ln2.bias": f"{prefix}net.5.bias",
        }
        for old_k, new_k in key_map.items():
            if old_k in state_dict and new_k not in state_dict:
                state_dict[new_k] = state_dict.pop(old_k)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
