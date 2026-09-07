import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from features.motion_kinematics import FishMotionKinematics10Ch
from features.audio_frontend import AudioFrontend
from .video_backbone import MobileViTVideoBackbone
from .audio_backbone import EfficientATAudioBackbone
from .multimodal_fusion import SOTAMultimodalFusion


class MultimodalSOTANet(nn.Module):
    """
    SOTA Unified Multimodal Fish Feeding Intensity Assessment Network (~4.78M Total Parameters):
      - Visual-Kinematic Stream: 10-Channel MobileViT-XS + Inter-frame Transformer (~2.42M)
      - Acoustic-Temporal Stream: EfficientAT (MobileNetV3 + Splash Cadence Convolutions) (~1.72M)
      - Multimodal Fusion Engine: Google MBT + Attentive Reliability Gate + TMC Evidential Uncertainty (~0.64M)

    Invariants:
      1. Zero overlap with dataset authors' papers (Meng Cui et al. / U-FFIA / AV-FFIA).
      2. Strictly maintains total parameters < 5.0 Million (~4.78M).
      3. First 3 RGB channels preserve 100% ImageNet pretrained weights via weight inflation.
      4. Evaluates epistemic uncertainty and dynamic modality reliability under turbid waters and noise.
    """
    def __init__(
        self,
        classes_num: int = 4,
        embed_dim: int = 224,
        num_bottlenecks: int = 4,
        num_heads: int = 4,
        pretrained_video: bool = True,
        audio_frontend: Optional[AudioFrontend] = None,
        image_size: int = 224,
        num_frames: int = 4,
    ) -> None:
        super().__init__()
        self.classes_num = classes_num
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.image_size = image_size

        # 1. Frontends
        self.audio_frontend = audio_frontend if audio_frontend is not None else AudioFrontend()
        self.motion_kinematics = FishMotionKinematics10Ch(image_size=image_size)

        # 2. Backbones (~4.14M)
        self.video_backbone = MobileViTVideoBackbone(
            embed_dim=embed_dim,
            pretrained=pretrained_video,
            num_frames=num_frames
        )
        self.audio_backbone = EfficientATAudioBackbone(
            embed_dim=embed_dim,
            pretrained=False
        )

        # 3. Fusion Engine (~0.64M)
        self.fusion = SOTAMultimodalFusion(
            dim=embed_dim,
            num_bottlenecks=num_bottlenecks,
            num_heads=num_heads,
            classes_num=classes_num
        )

    def forward(
        self,
        video_input: torch.Tensor,
        audio_input: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            video_input: Raw RGB frames [B, T, 3, H, W] or precomputed 10-ch tensor [B, T, 10, H, W]
            audio_input: Raw audio waveforms [B, num_samples] or precomputed Log-Mel Spectrogram [B, 1, Ta, 128]

        Returns:
            Dictionary containing clipwise_output (logits), probabilities, uncertainties, and modality embeddings.
        """
        # Step 1: Preprocessing & Frontend Extraction
        if video_input.ndim == 5 and video_input.size(2) == 3:
            frames_10ch, kinematics_summary = self.motion_kinematics(video_input)
        else:
            frames_10ch = video_input
            kinematics_summary = torch.zeros(video_input.size(0), 4, device=video_input.device, dtype=video_input.dtype)

        if audio_input.ndim == 2:
            mel_spec = self.audio_frontend(audio_input)
        else:
            mel_spec = audio_input

        # Step 2: Unimodal Spatiotemporal Feature Extraction
        f_video, f_spatial, f_motion, tokens_video = self.video_backbone(frames_10ch)
        f_audio, f_frequency, f_rhythm, tokens_audio = self.audio_backbone(mel_spec)

        # Step 3: Multimodal Bottleneck & Evidential Fusion
        fusion_outputs = self.fusion(
            f_video=f_video,
            f_audio=f_audio,
            tokens_video=tokens_video,
            tokens_audio=tokens_audio
        )

        # Step 4: Assemble Comprehensive Output
        outputs = {
            "clipwise_output": fusion_outputs["logits"],
            "logits": fusion_outputs["logits"],
            "probabilities": fusion_outputs["probabilities"],
            "uncertainty": fusion_outputs["uncertainty"],
            "uncertainty_video": fusion_outputs["uncertainty_video"],
            "uncertainty_audio": fusion_outputs["uncertainty_audio"],
            "modality_weights": fusion_outputs["modality_weights"],
            "kinematics_summary": kinematics_summary,
            "f_spatial": f_spatial,
            "f_motion": f_motion,
            "f_frequency": f_frequency,
            "f_rhythm": f_rhythm,
            "f_fused": fusion_outputs["f_fused"],
            "evidence_v": fusion_outputs["evidence_v"],
            "evidence_a": fusion_outputs["evidence_a"],
            "evidence_final": fusion_outputs["evidence_final"],
            "alpha_final": fusion_outputs["alpha_final"],
        }
        return outputs
