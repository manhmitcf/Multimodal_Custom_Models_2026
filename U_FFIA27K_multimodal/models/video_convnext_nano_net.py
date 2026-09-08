import logging
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from features.motion_kinematics import FishMotionKinematics7Ch
from models.video_backbone import ConvNeXtNanoVideoBackbone

logger = logging.getLogger(__name__)


class VideoConvNeXtNanoNet(nn.Module):
    """
    Dedicated Unimodal Video Architecture for Fish Feeding Intensity Classification.
    Equivalent to the Video Stream in Phase 1, upgraded to 4 frames (T=4).

    Pipeline:
      1. Motion Kinematics 7-Channel Frontend:
         Computes differential spatial-temporal gradients across 4 uniform frames:
         [RGB (3), Optical Flow u, v (2), Velocity Magnitude |V| (1), Fluid Vorticity omega (1)]
      2. ConvNeXt-Nano Video Backbone:
         Multi-stage depthwise 7x7 convolutional feature extraction (~2.70M params).
         Produces spatiotemporal representation f_video (224-dim).
      3. Linear Classifier Head:
         nn.Linear(224, classes_num) mapping spatiotemporal feature directly to 4-class logits.
    """
    def __init__(
        self,
        classes_num: int = 4,
        embed_dim: int = 224,
        image_size: int = 224,
        num_frames: int = 4,
        in_chans: int = 7,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.classes_num = classes_num
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.image_size = image_size
        self.in_chans = in_chans

        # 1. Kinematic Frontend (non-trainable analytical differential filters)
        self.motion_kinematics = FishMotionKinematics7Ch(image_size=image_size)

        # 2. ConvNeXt-Nano Video Backbone (~2.70M params)
        self.video_backbone = ConvNeXtNanoVideoBackbone(
            embed_dim=embed_dim,
            in_chans=in_chans,
            num_frames=num_frames
        )

        # 3. Linear Classifier Head (~900 params)
        self.classifier = nn.Linear(embed_dim, classes_num)

        # Legacy alias for seamless compatibility with multimodal evaluation code
        self.aux_head_video = self.classifier

        self._init_weights()
        logger.info(
            f"Initialized VideoConvNeXtNanoNet: num_frames={num_frames}, "
            f"in_chans={in_chans}, embed_dim={embed_dim}, classes={classes_num}"
        )

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        video_input: torch.Tensor,
        audio_input: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            video_input: Raw RGB frames [B, T, 3, H, W] or precomputed 7-ch tensor [B, T, 7, H, W].
            audio_input: Ignored for unimodal video (maintained for unified API compatibility).

        Returns:
            Dictionary containing logits, probabilities, and latent embeddings.
        """
        # Step 1: Preprocessing & 7-Channel Kinematics
        if video_input.ndim == 5 and video_input.size(2) == 3:
            frames_7ch, kinematics_summary = self.motion_kinematics(video_input)
        else:
            frames_7ch = video_input
            kinematics_summary = torch.zeros(
                video_input.size(0), 4, device=video_input.device, dtype=video_input.dtype
            )

        # Step 2: Spatiotemporal Feature Extraction via ConvNeXt-Nano
        f_video, f_spatial, f_motion, f_burst_v, tokens_video = self.video_backbone(frames_7ch)

        # Step 3: Classification Head
        logits = self.classifier(f_video)
        probabilities = F.softmax(logits, dim=-1)

        return {
            "clipwise_output": logits,
            "logits": logits,
            "probabilities": probabilities,
            "logits_video": logits,
            "prob_video": probabilities,
            "f_video": f_video,
            "f_spatial": f_spatial,
            "f_motion": f_motion,
            "f_burst_v": f_burst_v,
            "tokens_video": tokens_video,
            "kinematics_summary": kinematics_summary,
        }
