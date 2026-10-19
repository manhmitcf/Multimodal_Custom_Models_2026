import torch
import torch.nn as nn
from typing import Optional, List, Dict
import logging

from features.motion_kinematics import FishMotionKinematics7Ch
from .video_backbone import ConvNeXtNanoVideoBackbone

logger = logging.getLogger(__name__)


class VideoSOTANet(nn.Module):
    """
    Unimodal Video SOTANet Architecture tailored for Fish Feeding Intensity Assessment.
    Supports dynamic input channel slicing across 8 presets and 3 temporal aggregation modes.

    Architecture:
      1. Motion Kinematics Extractor: Differentiable 7-channel physics computation (T=2)
      2. Channel Slicing: Extracts selected subset of channels (K in {3, 4, 5, 6, 7})
      3. ConvNeXt-Nano Video Backbone: Processes selected frames with native 384-dim
      4. Temporal Aggregation: 'sum', 'abs_diff', or 'composite_ln'
      5. Classification Head: Linear(384, 4)
    """
    def __init__(
        self,
        channel_indices: Optional[List[int]] = None,
        temporal_mode: str = "sum",
        embed_dim: int = 384,
        classes_num: int = 4,
        drop_path_rate: float = 0.1,
        image_size: int = 224,
        num_frames: int = 2,
    ) -> None:
        super().__init__()
        if channel_indices is None:
            channel_indices = [0, 1, 2, 3, 4, 5, 6]
        self.channel_indices = sorted(list(channel_indices))
        self.temporal_mode = temporal_mode
        self.embed_dim = embed_dim
        self.classes_num = classes_num
        self.image_size = image_size
        self.num_frames = num_frames

        # Differentiable kinematics buffer (0 trainable parameters)
        self.motion_kinematics = FishMotionKinematics7Ch(image_size=image_size)

        in_chans = len(self.channel_indices)
        self.video_backbone = ConvNeXtNanoVideoBackbone(
            embed_dim=embed_dim,
            in_chans=in_chans,
            num_frames=num_frames,
            drop_path_rate=drop_path_rate,
            temporal_mode=temporal_mode,
        )

        self.classifier = nn.Linear(embed_dim, classes_num)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        video: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass accepting both model(video) and model(video, audio).

        Args:
            video: Video tensor [B, T, 3, H, W]
            audio: Ignored optional audio tensor (for multi-caller compatibility)

        Returns:
            Dict containing:
              - 'clipwise_output': Classification logits [B, classes_num]
              - 'logits_video': Classification logits [B, classes_num]
              - 'f_video': Video representation embedding [B, embed_dim]
        """
        if self.channel_indices == [0, 1, 2]:
            selected_frames = video
        else:
            frames_7ch, _ = self.motion_kinematics(video)
            selected_frames = frames_7ch[:, :, self.channel_indices, :, :]

        f_video = self.video_backbone(selected_frames)
        logits = self.classifier(f_video)

        return {
            "clipwise_output": logits,
            "logits": logits,
            "logits_video": logits,
            "probabilities": torch.softmax(logits, dim=-1),
            "f_video": f_video,
            "video_features": f_video,
        }
