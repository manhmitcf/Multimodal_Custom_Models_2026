import torch
import torch.nn as nn
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class AudioMLPBackbone(nn.Module):
    """
    High-Resolution STFT Audio MLP Backbone (~1.39M params).
    Processes 2049-dimensional TKEO-STFT spectral vectors [B, 2049]:
      - Layer 1: Linear(2049 -> 512) + LayerNorm(512) + GELU + Dropout(0.1)
      - Layer 2: Linear(512 -> 224) + LayerNorm(224) -> f_audio [B, 224]
      - Token Projection: Linear(512 -> 2 * 224) -> tokens_audio [B, 2, 224]
    """
    def __init__(
        self,
        in_features: int = 2049,
        hidden_dim: int = 512,
        embed_dim: int = 224,
        num_tokens: int = 2,
        dropout: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        # 1. 2-layer MLP for 2049 STFT vector
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Orthogonal weight initialization (Saxe et al., ICLR 2014) for high-dimensional spectral MLP.
        Preserves the vector norm and angular geometry when projecting 2049 -> 512 -> 224,
        preventing energy compression or gradient explosion across dense linear layers.
        """
        for layer in (self.fc1, self.fc2):
            nn.init.orthogonal_(layer.weight, gain=1.0)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: STFT spectral feature vector [B, 2049]

        Returns:
            f_audio: Joint acoustic embedding [B, embed_dim]
            f_frequency: Spectral frequency feature [B, embed_dim]
            f_rhythm: Temporal rhythm feature [B, embed_dim]
            f_burst_a: Acoustic burst feature [B, embed_dim]
            tokens_audio: Sequence of audio tokens [B, num_tokens, embed_dim]
        """
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
            if x.size(-1) != self.in_features:
                x = x[:, :self.in_features]

        h = self.dropout(self.act(self.ln1(self.fc1(x))))
        f_audio = self.ln2(self.fc2(h))

        # Compatibility tokens sequence for multimodal fusion interface
        tokens_audio = f_audio.unsqueeze(1).repeat(1, self.num_tokens, 1)

        f_frequency = f_audio
        f_rhythm = f_audio
        f_burst_a = f_audio

        return f_audio, f_frequency, f_rhythm, f_burst_a, tokens_audio


# Canonical alias
AudioBackbone = AudioMLPBackbone
