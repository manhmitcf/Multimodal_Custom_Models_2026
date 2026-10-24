import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any

from features.audio_frontend import AudioFrontend
from features.motion_kinematics import FishMotionKinematics5Ch
from .audio_backbone import AudioMLPBackbone
from .video_backbone import ConvNeXtNanoVideoBackbone


class Flat4ClassMLPHead(nn.Module):
    """
    2-Layer MLP Classification Head for Direct 4-Class Feeding Intensity Prediction.
    Architecture:
        Linear(in_dim -> hidden_dim) + GELU + LayerNorm(hidden_dim) + Dropout(p) + Linear(hidden_dim -> num_classes)
    """
    def __init__(
        self,
        in_dim: int = 384,
        hidden_dim: int = 192,
        num_classes: int = 4,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalFlatNet(nn.Module):
    """
    Ablation Baseline: Flat 4-Class Multimodal Network without Tournament Decision Engine.
    Directly evaluates 3 multimodal fusion mechanisms:
      - 'sum'   : f_fused = LayerNorm(f_video + f_audio) (384-dim)
      - 'concat': f_fused = LayerNorm([f_video || f_audio]) (768-dim)
      - 'gated' : f_fused = LayerNorm(g * f_video + (1-g) * f_audio) where g = sigmoid(W[f_V || f_A]) (384-dim)
    Followed by Flat4ClassMLPHead(in_dim -> 192 -> 4).
    """
    def __init__(
        self,
        image_size: int = 224,
        num_frames: int = 2,
        num_channels: int = 5,
        embed_dim: int = 384,
        classes_num: int = 4,
        fusion_mode: str = "sum",
        temporal_mode: str = "composite_ln",
        video_drop_path: float = 0.1,
        audio_hidden_dims: Tuple[int, ...] = (768, 512),
        dropout: float = 0.1,
        audio_frontend: Optional[nn.Module] = None,
        **kwargs
    ) -> None:
        super().__init__()
        assert fusion_mode in ["sum", "concat", "gated"], (
            f"fusion_mode must be one of ['sum', 'concat', 'gated'], got '{fusion_mode}'"
        )
        self.image_size = image_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.classes_num = classes_num
        self.fusion_mode = fusion_mode

        # 1. Video Pipeline: 5-ch Kinematics -> ConvNeXt-Nano -> 384-dim
        self.motion_kinematics = FishMotionKinematics5Ch(image_size=image_size)
        self.motion_extractor = self.motion_kinematics  # alias for backwards compatibility
        self.video_backbone = ConvNeXtNanoVideoBackbone(
            in_chans=num_channels,
            embed_dim=embed_dim,
            num_frames=num_frames,
            drop_path_rate=video_drop_path,
            temporal_mode=temporal_mode
        )
        self.aux_head_video = nn.Linear(embed_dim, classes_num)

        # 2. Audio Pipeline: TKEO-STFT -> STFT-MLP L2 -> 384-dim
        self.audio_frontend = audio_frontend if audio_frontend is not None else AudioFrontend()
        self.audio_backbone = AudioMLPBackbone(
            in_features=2049,
            hidden_dims=audio_hidden_dims,
            embed_dim=embed_dim,
            dropout=dropout
        )
        self.aux_head_audio = nn.Linear(embed_dim, classes_num)

        # 3. Fusion & Flat 4-Class Classification Head
        if fusion_mode == "sum":
            self.norm_fused = nn.LayerNorm(embed_dim, eps=1e-6)
            self.head = Flat4ClassMLPHead(
                in_dim=embed_dim,
                hidden_dim=192,
                num_classes=classes_num,
                dropout=dropout
            )
            self.gate = None
        elif fusion_mode == "concat":
            self.norm_fused = nn.LayerNorm(embed_dim * 2, eps=1e-6)
            self.head = Flat4ClassMLPHead(
                in_dim=embed_dim * 2,
                hidden_dim=192,
                num_classes=classes_num,
                dropout=dropout
            )
            self.gate = None
        elif fusion_mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, 1),
                nn.Sigmoid()
            )
            self.norm_fused = nn.LayerNorm(embed_dim, eps=1e-6)
            self.head = Flat4ClassMLPHead(
                in_dim=embed_dim,
                hidden_dim=192,
                num_classes=classes_num,
                dropout=dropout
            )

    def forward(
        self,
        video: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        f_video: Optional[torch.Tensor] = None,
        f_audio: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        # --- A. Video Stream Extraction ---
        if f_video is None:
            if video is None:
                raise ValueError("Either 'video' or 'f_video' must be provided.")
            frames_5ch, _ = self.motion_extractor(video)
            f_video = self.video_backbone(frames_5ch)  # [B, 384]

        # --- B. Audio Stream Extraction ---
        if f_audio is None:
            if audio is None:
                raise ValueError("Either 'audio' or 'f_audio' must be provided.")
            stft_features = self.audio_frontend(audio)
            f_audio = self.audio_backbone(stft_features)  # [B, 384]

        # --- C. Auxiliary Unimodal Predictions ---
        logits_video_aux = self.aux_head_video(f_video)  # [B, 4]
        logits_audio_aux = self.aux_head_audio(f_audio)  # [B, 4]

        # --- D. Multimodal Fusion ---
        g = None
        if self.fusion_mode == "sum":
            f_fused = self.norm_fused(f_video + f_audio)  # [B, 384]
        elif self.fusion_mode == "concat":
            combined = torch.cat([f_video, f_audio], dim=-1)  # [B, 768]
            f_fused = self.norm_fused(combined)
        elif self.fusion_mode == "gated":
            combined = torch.cat([f_video, f_audio], dim=-1)  # [B, 768]
            g = self.gate(combined)                           # [B, 1]
            f_fused = self.norm_fused(g * f_video + (1.0 - g) * f_audio)  # [B, 384]

        # --- E. Flat 4-Class Classification Head ---
        logits = self.head(f_fused)                           # [B, 4]
        probabilities = torch.softmax(logits, dim=-1)         # [B, 4]

        # Expected physical intensity on 0..3 ordinal scale
        # None=0, Strong=1, Medium=2, Weak=3
        # Rank mapping: None -> 0.0, Strong -> 3.0, Medium -> 2.0, Weak -> 1.0
        expected_intensity = (
            probabilities[:, 0] * 0.0 +
            probabilities[:, 1] * 3.0 +
            probabilities[:, 2] * 2.0 +
            probabilities[:, 3] * 1.0
        ).unsqueeze(-1)  # [B, 1]

        outputs = {
            "logits": logits,
            "clipwise_output": logits,
            "probabilities": probabilities,
            "expected_intensity": expected_intensity,
            "intensity_score": expected_intensity,
            "f_video": f_video,
            "f_audio": f_audio,
            "f_fused": f_fused,
            "logits_video": logits_video_aux,
            "logits_audio": logits_audio_aux,
            "logits_video_aux": logits_video_aux,
            "logits_audio_aux": logits_audio_aux,
        }
        if g is not None:
            outputs["gate"] = g
            outputs["modality_weights"] = torch.cat([g, 1.0 - g], dim=-1)

        return outputs
