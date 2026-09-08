import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from features.motion_kinematics import FishMotionKinematics7Ch, FishMotionKinematics10Ch
from features.audio_frontend import AudioFrontend
from .video_backbone import ConvNeXtNanoVideoBackbone, MobileViTVideoBackbone
from .audio_backbone import PANNSCNN6AudioBackbone, EfficientATAudioBackbone
from .multimodal_fusion import GatedBilateralBoundaryFusion, MultimodalBoundaryAwareFusion, SOTAMultimodalFusion


class MultimodalBoundaryAwareNet(nn.Module):
    """
    Multimodal Bilateral Boundary Network (BBN-4.5M) (~4.54M Total Parameters).
    Specifically architected to resolve continuous temporal boundary transition ambiguity
    between adjacent fish feeding intensity classes (Strong <-> Medium <-> Weak <-> None):

      1. Visual-Kinematic Stream (~2.70M params):
         7-Channel ConvNeXt-Nano (Spatial RGB + Flow (u,v) + Velocity |V| + Fluid Vorticity omega)
         for T=2 frames.
      2. Acoustic Time-Frequency Stream (~1.76M params):
         TKEO Adaptive Pre-Emphasis + Learnable Frequency Attention + PANNS-CNN6-Pro 4-stage 5x5 Conv
         with Dual Pooling (max+avg).
      3. Gated Bilateral Boundary Fusion (~0.10M params):
         - Dynamic Gated Fusion: g = sigma(W[f_V || f_A]).
         - Bilateral Monotonic CORAL Decision Heads (s_V, b_V & s_A, b_A).
         - Blended Decision: s_final = g*s_V + (1-g)*s_A with strictly monotonic cutoffs.

    Total Parameters: ~4.54M (Strictly < 5.0M parameter constraint).
    """
    def __init__(
        self,
        classes_num: int = 4,
        embed_dim: int = 224,
        num_bottlenecks: int = 4,  # Kept for config compatibility
        num_heads: int = 4,
        pretrained_video: bool = False,
        audio_frontend: Optional[AudioFrontend] = None,
        image_size: int = 224,
        num_frames: int = 2,
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

        # 2. Backbones (~4.46M)
        self.video_backbone = ConvNeXtNanoVideoBackbone(
            embed_dim=embed_dim,
            in_chans=in_chans,
            num_frames=num_frames
        )
        self.audio_backbone = PANNSCNN6AudioBackbone(
            embed_dim=embed_dim,
            num_tokens=num_frames
        )

        # 3. Gated Bilateral Boundary Fusion (~0.10M)
        self.fusion = GatedBilateralBoundaryFusion(
            dim=embed_dim,
            dropout=0.1
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
            modality weights, bilateral scores & cutoffs, and continuous intensity scores.
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
        f_video, f_spatial, f_motion, f_burst_v, tokens_video = self.video_backbone(frames_7ch)
        f_audio, f_frequency, f_rhythm, f_burst_a, tokens_audio = self.audio_backbone(mel_spec)

        # Step 3: Gated Bilateral Boundary Fusion
        fusion_outputs = self.fusion(
            f_video=f_video,
            f_audio=f_audio,
            tokens_video=tokens_video,
            tokens_audio=tokens_audio,
            f_burst_v=f_burst_v,
            f_burst_a=f_burst_a
        )

        # Step 4: Assemble Comprehensive Output
        outputs = {
            "clipwise_output": fusion_outputs["logits"],
            "logits": fusion_outputs["logits"],
            "probabilities": fusion_outputs["probabilities"],
            "uncertainty": fusion_outputs["uncertainty"],
            "modality_weights": fusion_outputs["modality_weights"],
            "intensity_score": fusion_outputs["intensity_score"],
            "expected_intensity": fusion_outputs["expected_intensity"],
            "cutoffs": fusion_outputs["b_final"],
            "b_final": fusion_outputs["b_final"],
            "cutoffs_v": fusion_outputs["cutoffs_v"],
            "cutoffs_a": fusion_outputs["cutoffs_a"],
            "score_v": fusion_outputs["score_v"],
            "score_a": fusion_outputs["score_a"],
            "gate": fusion_outputs["gate"],
            "cum_probs": fusion_outputs["cum_probs"],
            "cum_probs_v": fusion_outputs["cum_probs_v"],
            "cum_probs_a": fusion_outputs["cum_probs_a"],
            "probabilities_v": fusion_outputs["probabilities_v"],
            "probabilities_a": fusion_outputs["probabilities_a"],
            "kinematics_summary": kinematics_summary,
            "f_spatial": f_spatial,
            "f_motion": f_motion,
            "f_burst_v": f_burst_v,
            "f_frequency": f_frequency,
            "f_rhythm": f_rhythm,
            "f_burst_a": f_burst_a,
            "f_fused": fusion_outputs["f_fused"],
        }
        return outputs


# Aliases for 100% backwards compatibility with training & evaluation pipelines
MultimodalSOTANet = MultimodalBoundaryAwareNet
