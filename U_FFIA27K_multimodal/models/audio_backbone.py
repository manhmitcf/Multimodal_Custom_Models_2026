import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AudioHarmonicCRNN(nn.Module):
    """
    Harmonic 1D-CRNN Audio Backbone tailored for High-Resolution TKEO-STFT Spectrograms [B, 2049, T].
    Scaled according to harmonic sub-multiples of embed_dim = 224:
      - Layer 1 (Spectral Compression): Conv1d(2049, 448, kernel_size=1) + BatchNorm1d(448) + GELU
      - Layer 2 (Temporal Convolution): Conv1d(448, 224, kernel_size=3, padding=1) + BatchNorm1d(224) + GELU + MaxPool1d(2)
      - Layer 3 (Bidirectional GRU): nn.GRU(input_size=224, hidden_size=112, bidirectional=True, batch_first=True)
      - Representation Output: LayerNorm(224) over concatenated forward & backward states -> f_audio [B, 224]
      - Tokens Output: Full temporal sequence of BiGRU tokens [B, T // 2, 224]
    Total parameters: ~1.45M params.
    """
    def __init__(
        self,
        in_features: int = 2049,
        mid_dim: int = 448,
        embed_dim: int = 224,
        gru_hidden: int = 112,
        dropout: float = 0.1,
        seed: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.mid_dim = mid_dim
        self.embed_dim = embed_dim
        self.gru_hidden = gru_hidden

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # 1. Spectral Compression: Conv1d(2049 -> 448, k=1)
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_features, mid_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 2. Temporal Convolution: Conv1d(448 -> 224, k=3, padding=1) + Pool
        self.cnn2 = nn.Sequential(
            nn.Conv1d(mid_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # 3. Bidirectional GRU: 112 * 2 = 224
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=gru_hidden,
            bidirectional=True,
            batch_first=True
        )

        # 4. Output Layer Normalization
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Kaiming Normal initialization for Conv1d, Orthogonal initialization for BiGRU recurrent weights,
        Xavier Uniform for BiGRU input weights, and standard normalization scaling.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0.0)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: STFT 2D spectrogram [B, 2049, T] or [B, T, 2049] or 1D vector [B, 2049]
        Returns:
            f_audio: Joint acoustic embedding [B, embed_dim]
            f_frequency: Spectral frequency feature [B, embed_dim]
            f_rhythm: Temporal rhythm dynamics feature [B, embed_dim]
            f_burst_a: Acoustic burst dynamic contrast [B, embed_dim]
            tokens_audio: Sequence of audio tokens [B, T_tokens, embed_dim]
        """
        if x.ndim == 2:
            # 1D vector fallback [B, 2049] -> simulate T=1
            x = x.unsqueeze(-1)
        elif x.ndim == 3 and x.size(1) != self.in_features and x.size(-1) == self.in_features:
            # [B, T, 2049] -> transpose to [B, 2049, T]
            x = x.transpose(1, 2)
        elif x.ndim > 3:
            x = x.flatten(start_dim=2)

        # 1. CNN Stages
        h1 = self.cnn1(x)             # [B, 448, T]
        h2 = self.cnn2(h1)            # [B, 224, T // 2]

        # 2. Transpose for GRU: [B, Seq_Len, Features]
        h_seq = h2.transpose(1, 2)    # [B, T // 2, 224]
        gru_tokens, h_n = self.gru(h_seq) # gru_tokens: [B, T // 2, 224], h_n: [2, B, 112]

        # 3. Concatenate forward and backward final hidden states
        h_last = torch.cat([h_n[0], h_n[1]], dim=-1) # [B, 224]
        f_audio = self.norm(h_last)

        # 4. Rhythm and burst features
        tokens_audio = self.norm(gru_tokens)
        f_frequency = f_audio
        if tokens_audio.size(1) > 1:
            f_rhythm = tokens_audio.std(dim=1)
            f_burst_a = tokens_audio.max(dim=1)[0] - tokens_audio.mean(dim=1)
        else:
            f_rhythm = f_audio
            f_burst_a = f_audio

        return f_audio, f_frequency, f_rhythm, f_burst_a, tokens_audio


# Canonical aliases
AudioBackbone = AudioHarmonicCRNN
