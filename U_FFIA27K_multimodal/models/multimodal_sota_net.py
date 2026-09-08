import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from features.motion_kinematics import FishMotionKinematics7Ch, FishMotionKinematics10Ch
from features.audio_frontend import AudioFrontend
from .video_backbone import MobileViTVideoBackbone
from .audio_backbone import EfficientATAudioBackbone
from .multimodal_fusion import MultimodalBoundaryAwareFusion, SOTAMultimodalFusion


class MultimodalBoundaryAwareNet(nn.Module):
    """
    Multimodal Boundary-Aware Network (MultimodalBoundaryAwareNet) (~4.78M Total Parameters).
    Specifically architected to resolve continuous temporal boundary transition ambiguity
    between adjacent fish feeding intensity classes (Strong <-> Medium <-> Weak <-> None):

      1. Visual-Kinematic Stream:
         7-Channel MobileViT-XS (Spatial RGB + Flow (u,v) + Fluid Vorticity omega + Deceleration Delta|V|)
         Dual token output (f_spatial + f_motion) (~2.42M params).
      2. Acoustic Time-Frequency Stream:
         Time-Frequency factorized EfficientAT with Frequency Squeeze-and-Excitation (isolating 2-8 kHz splashes)
         and Temporal Rhythm Depthwise Conv (~1.82M params).
      3. Boundary Transition Multimodal Fusion:
         - Bidirectional Cross-Attention (Bi-CA) across temporal tokens.
         - Boundary Discrepancy Gate (BDG) measuring cross-modal asynchrony Delta_{trans} = |f'_V - f'_A|.
         - Boundary-Aware Channel Routing (BACA) partitioning 224 channels into 128 Core + 96 Boundary.
         - Strictly Monotonic Cumulative Ordinal Decision Head (b_1 < b_2 < b_3) eliminating 50/50 flips (~0.54M params).

    Total Parameters: ~4.78M (Strictly < 5.0M parameter constraint).
    Zero components borrowed from Meng Cui et al. / U-FFIA / AV-FFIA.
    """
    def __init__(
        self,
        classes_num: int = 4,
        embed_dim: int = 224,
        num_bottlenecks: int = 4,  # Kept for config compatibility
        num_heads: int = 4,
        pretrained_video: bool = True,
        audio_frontend: Optional[AudioFrontend] = None,
        image_size: int = 224,
        num_frames: int = 4,
        in_chans: int = 7,
    ) -> None:
        super().__init__()
        self.classes_num = classes_num
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.image_size = image_size
        self.in_chans = in_chans

        # 1. Frontends
        self.audio_frontend = audio_frontend if audio_frontend is not None else AudioFrontend()
        self.motion_kinematics = FishMotionKinematics7Ch(image_size=image_size)

        # 2. Backbones (~4.24M)
        self.video_backbone = MobileViTVideoBackbone(
            embed_dim=embed_dim,
            pretrained=pretrained_video,
            num_frames=num_frames,
            in_chans=in_chans
        )
        self.audio_backbone = EfficientATAudioBackbone(
            embed_dim=embed_dim,
            pretrained=False,
            num_tokens=num_frames
        )

        # 3. Streamlined Boundary-Aware Fusion Engine (~0.54M)
        self.fusion = MultimodalBoundaryAwareFusion(
            dim=embed_dim,
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
            video_input: Raw RGB frames [B, T, 3, H, W] or precomputed 7-ch tensor [B, T, 7, H, W]
            audio_input: Raw audio waveforms [B, num_samples] or precomputed Log-Mel Spectrogram [B, 1, Ta, 128]

        Returns:
            Dictionary containing clipwise_output (logits), probabilities, uncertainties,
            modality weights, boundary weights, and continuous intensity scores.
        """
        # Step 1: Preprocessing & Frontend Extraction
        if video_input.ndim == 5 and video_input.size(2) == 3:
            frames_7ch, kinematics_summary = self.motion_kinematics(video_input)
        else:
            frames_7ch = video_input
            kinematics_summary = torch.zeros(video_input.size(0), 4, device=video_input.device, dtype=video_input.dtype)

        if audio_input.ndim == 2:
            mel_spec = self.audio_frontend(audio_input)
        else:
            mel_spec = audio_input

        # Step 2: Unimodal Spatiotemporal Feature Extraction
        f_video, f_spatial, f_motion, tokens_video = self.video_backbone(frames_7ch)
        f_audio, f_frequency, f_rhythm, tokens_audio = self.audio_backbone(mel_spec)

        # Step 3: Multimodal Boundary-Aware Fusion & Ordinal Decision
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
            "boundary_weights": fusion_outputs["boundary_weights"],
            "intensity_score": fusion_outputs["intensity_score"],
            "expected_intensity": fusion_outputs["expected_intensity"],
            "cutoffs": fusion_outputs["cutoffs"],
            "cum_probs": fusion_outputs["cum_probs"],
            "kinematics_summary": kinematics_summary,
            "f_spatial": f_spatial,
            "f_motion": f_motion,
            "f_frequency": f_frequency,
            "f_rhythm": f_rhythm,
            "f_fused": fusion_outputs["f_fused"],
        }
        return outputs


# Aliases for 100% backwards compatibility with training & evaluation pipelines
MultimodalSOTANet = MultimodalBoundaryAwareNet

