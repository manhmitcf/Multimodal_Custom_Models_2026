import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class TonalStream2D(nn.Module):
    """
    Branch 1: Tonal & Continuous Acoustic Flow Stream.
    Captures continuous harmonic bands, water flow, and aerator background noise
    using narrow-frequency, longer-temporal receptive field convolutions.
    """
    def __init__(self, in_ch: int = 1, out_ch: int = 48) -> None:
        super().__init__()
        self.conv_net = nn.Sequential(
            # Stage 1: [B, 1, T, 2049] -> [B, 24, T, 257]
            nn.Conv2d(in_ch, 24, kernel_size=(5, 5), stride=(1, 8), padding=(2, 2), bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            # Stage 2: [B, 24, T, 257] -> [B, 36, T, 65]
            nn.Conv2d(24, 36, kernel_size=(5, 3), stride=(1, 4), padding=(2, 1), bias=False),
            nn.BatchNorm2d(36),
            nn.SiLU(),
            # Stage 3: [B, 36, T, 65] -> [B, 48, T, 33]
            nn.Conv2d(36, out_ch, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_net(x)


class TransientStream2D(nn.Module):
    """
    Branch 2: Impulsive Transient & Ultrasonic Feeding Click Stream.
    Captures sharp vertical spectral bursts (feeding clicks, cavitation bubbles)
    using wide-frequency receptive fields and multi-scale dilated convolutions.
    """
    def __init__(self, in_ch: int = 1, out_ch: int = 48) -> None:
        super().__init__()
        self.conv_net = nn.Sequential(
            # Stage 1: Wide frequency kernel to catch broadband click stripes
            nn.Conv2d(in_ch, 24, kernel_size=(3, 11), stride=(1, 8), padding=(1, 5), bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            # Stage 2: Dilated convolution across frequency for long-range harmonic correlation
            nn.Conv2d(24, 36, kernel_size=(3, 5), stride=(1, 4), padding=(1, 4), dilation=(1, 2), bias=False),
            nn.BatchNorm2d(36),
            nn.SiLU(),
            # Stage 3: High-frequency transient sharpness extractor
            nn.Conv2d(36, out_ch, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_net(x)


class NanoUnderwaterDualBranch(nn.Module):
    """
    Standalone Nano-Underwater Dual-Branch Audio Classifier (~1.2M params).
    Based on Underwater Acoustic / Hydrophone Research (IEEE JOE / Frontiers in Marine Science 2022-2024).
    - Branch 1 (Tonal): Trích xuất nhiễu dòng chảy, máy sục khí, sóng nước nền.
    - Branch 2 (Transient): Trích xuất xung bùng nổ siêu âm (clicks, tiếng ăn mồi).
    - Cross-Stream Fusion: Hợp nhất 2 dòng đặc trưng qua Pointwise Conv.
    - Transformer Temporal Aggregator: Mô hình hóa quy luật xuất hiện của xung nhịp trên nền nhiễu.
    - Output: 4 classes classification.
    """
    def __init__(
        self,
        num_classes: int = 4,
        d_model: int = 160,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.model_name = "NanoUnderwaterDualBranch"
        self.d_model = d_model

        # 1. Dual-Branch 2D Front-end
        self.tonal_branch = TonalStream2D(in_ch=1, out_ch=48)
        self.transient_branch = TransientStream2D(in_ch=1, out_ch=48)

        # 2. Dynamic Flatten Dimension calculation
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 4, 2049)
            f_tonal = self.tonal_branch(dummy)
            f_trans = self.transient_branch(dummy)
            # Both have shape [B, 48, T, 33] -> concat channel = 96
            flatten_dim = (f_tonal.shape[1] + f_trans.shape[1]) * f_tonal.shape[3]  # 96 * 33 = 3168

        # 3. Channel Fusion & Projection to Transformer Sequence + Positional Encoding
        self.fusion_proj = nn.Sequential(
            nn.Linear(flatten_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.pos_emb = nn.Parameter(torch.randn(1, 500, d_model) * 0.02)

        # 4. Transformer Temporal Aggregator
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.final_norm = nn.LayerNorm(d_model)

        # 5. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, spec_2d: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            spec_2d: [B, 1, T, 2049] or [B, T, 2049].
        """
        if spec_2d.ndim == 3:
            spec_2d = spec_2d.unsqueeze(1)

        # 1. Forward through dual streams
        tonal_feat = self.tonal_branch(spec_2d)        # [B, 48, T, 33]
        transient_feat = self.transient_branch(spec_2d)  # [B, 48, T, 33]

        # 2. Concat across channel axis: [B, 96, T, 33]
        fused_2d = torch.cat([tonal_feat, transient_feat], dim=1)
        B, C, T, F_prime = fused_2d.shape

        # 3. Reshape to temporal sequence: [B, T, C * F_prime] -> [B, T, d_model]
        tokens = fused_2d.permute(0, 2, 1, 3).reshape(B, T, -1)
        tokens = self.fusion_proj(tokens)  # [B, T, d_model]
        tokens = tokens + self.pos_emb[:, :T, :]

        # 4. Transformer Attention over temporal sequence
        tokens = self.transformer_encoder(tokens)
        tokens = self.final_norm(tokens)

        # 5. Dual-Pooling (Mean for continuous tonal level + Max for transient burst peak)
        feat_mean = tokens.mean(dim=1)
        feat_max = tokens.max(dim=1).values
        embedding = feat_mean + feat_max  # [B, d_model]

        # 6. Classification Logits
        logits = self.classifier(embedding)

        return {
            'logits': logits,
            'clipwise_output': logits,
            'embedding': embedding,
            'tokens': tokens
        }
