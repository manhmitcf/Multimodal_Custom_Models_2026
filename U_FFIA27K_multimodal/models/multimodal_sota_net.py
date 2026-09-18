import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from features.motion_kinematics import FishMotionKinematics7Ch
from features.audio_frontend import AudioFrontend
from .video_backbone import ConvNeXtNanoVideoBackbone
from .audio_backbone import AudioMLPBackbone
from .multimodal_fusion import MultimodalTournamentFusion


class MultimodalBoundaryAwareNet(nn.Module):
    """
    Flat 4-Class Round-Robin Tournament Network (~4.08M - 4.23M Total Parameters).
    Specifically architected to resolve fish feeding intensity assessment across 4 classes
    (None, Strong, Medium, Weak) via flat round-robin tournament with 6 Configurable Audio STFT Tie-Breakers:

      1. Visual-Kinematic Stream (~2.70M params):
         7-Channel ConvNeXt-Nano (Spatial RGB + Flow (u,v) + Velocity |V| + Fluid Vorticity omega)
         for T=2 frames.
      2. Acoustic Time-Frequency Stream (~1.17M params):
         High-Resolution TKEO-STFT Audio Frontend (256 kHz, 2049 linear bins)
         + 2-layer MLP Projection (2049 -> 224).
      3. Pairwise Round-Robin Tournament Fusion (~0.20M - 0.36M params):
         - Dynamic Cross-Modal Reliability Gating: g = sigma(W[f_V || f_A]).
         - 6 Pairwise Subspace Expert Heads (B01, B02, B03, B12, B23, B13)
         - 6 Configurable Audio STFT Tie-Breakers.
         - Tournament Borda Voting to derive final calibrated multi-class probabilities.

    Total Parameters: ~4.08M (default 0 tie-breakers) to ~4.23M (all 6 tie-breakers enabled) (< 5.0M).
    """
    model_name: str = "MultimodalSOTANet"

    def __init__(
        self,
        classes_num: int = 4,
        embed_dim: int = 224,
        audio_frontend: Optional[AudioFrontend] = None,
        image_size: int = 224,
        num_frames: int = 2,
        in_chans: int = 7,
        enable_b01: bool = False,
        enable_b02: bool = False,
        enable_b03: bool = False,
        enable_b12: bool = False,
        enable_b23: bool = False,
        enable_b13: bool = False,
        tie_breakers: Optional[Dict[str, bool]] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.classes_num = classes_num
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.image_size = image_size
        self.in_chans = in_chans

        # Extract tie_breaker flags from tie_breakers dict or explicit args
        if tie_breakers is not None:
            self.enable_b01 = bool(tie_breakers.get("enable_b01", enable_b01))
            self.enable_b02 = bool(tie_breakers.get("enable_b02", enable_b02))
            self.enable_b03 = bool(tie_breakers.get("enable_b03", enable_b03))
            self.enable_b12 = bool(tie_breakers.get("enable_b12", enable_b12))
            self.enable_b23 = bool(tie_breakers.get("enable_b23", enable_b23))
            self.enable_b13 = bool(tie_breakers.get("enable_b13", enable_b13))
        else:
            self.enable_b01 = bool(enable_b01)
            self.enable_b02 = bool(enable_b02)
            self.enable_b03 = bool(enable_b03)
            self.enable_b12 = bool(enable_b12)
            self.enable_b23 = bool(enable_b23)
            self.enable_b13 = bool(enable_b13)

        # 1. Frontends
        self.audio_frontend = audio_frontend if audio_frontend is not None else AudioFrontend()
        self.motion_kinematics = FishMotionKinematics7Ch(image_size=image_size)

        # 2. Backbones (~3.87M)
        self.video_backbone = ConvNeXtNanoVideoBackbone(
            embed_dim=embed_dim,
            in_chans=in_chans,
            num_frames=num_frames
        )
        self.audio_backbone = AudioMLPBackbone(
            in_features=2049,
            embed_dim=embed_dim,
            num_tokens=num_frames
        )

        # 3. Multimodal Tournament Fusion with 6 Configurable Audio STFT Tie-Breakers (~0.20M - 0.36M)
        self.fusion = MultimodalTournamentFusion(
            dim=embed_dim,
            dropout=0.1,
            enable_b01=self.enable_b01,
            enable_b02=self.enable_b02,
            enable_b03=self.enable_b03,
            enable_b12=self.enable_b12,
            enable_b23=self.enable_b23,
            enable_b13=self.enable_b13,
        )

        # 4. Auxiliary Unimodal Classifier Heads (~1.8K params)
        # Allows independent supervision, accuracy tracking, and two-phase warmup
        self.aux_head_video = nn.Linear(embed_dim, classes_num)
        self.aux_head_audio = nn.Linear(embed_dim, classes_num)

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

        if audio_input.ndim >= 1 and audio_input.size(-1) > 2049:
            stft_feat = self.audio_frontend(audio_input)
        else:
            stft_feat = audio_input

        # Step 2: Unimodal Spatiotemporal Feature Extraction
        f_video, f_spatial, f_motion, f_burst_v, tokens_video = self.video_backbone(frames_7ch)
        f_audio, f_frequency, f_rhythm, f_burst_a, tokens_audio = self.audio_backbone(stft_feat)

        # Auxiliary Unimodal Logits & Probabilities (for standalone evaluation & Phase 1 warmup)
        logits_video = self.aux_head_video(f_video)
        logits_audio = self.aux_head_audio(f_audio)
        prob_video = F.softmax(logits_video, dim=-1)
        prob_audio = F.softmax(logits_audio, dim=-1)

        # Step 3: Gated Multimodal Tournament Fusion
        fusion_outputs = self.fusion(
            f_video=f_video,
            f_audio=f_audio
        )

        # Step 4: Assemble Comprehensive Output
        outputs = {
            "clipwise_output": fusion_outputs["logits"],
            "logits": fusion_outputs["logits"],
            "probabilities": fusion_outputs["probabilities"],
            "logits_video": logits_video,
            "logits_audio": logits_audio,
            "prob_video": prob_video,
            "prob_audio": prob_audio,
            "uncertainty": fusion_outputs.get("uncertainty"),
            "modality_weights": fusion_outputs.get("modality_weights"),
            "intensity_score": fusion_outputs.get("intensity_score"),
            "expected_intensity": fusion_outputs.get("expected_intensity"),
            "gate": fusion_outputs.get("gate"),
            "f_fused": fusion_outputs.get("f_fused"),
            # Feature diagnostics
            "kinematics_summary": kinematics_summary,
            "f_spatial": f_spatial,
            "f_motion": f_motion,
            "f_burst_v": f_burst_v,
            "f_frequency": f_frequency,
            "f_rhythm": f_rhythm,
            "f_burst_a": f_burst_a,
        }
        # Forward all 6-pair tournament outputs cleanly
        for k, v in fusion_outputs.items():
            if k not in outputs:
                outputs[k] = v

        return outputs


# Aliases for 100% backwards compatibility with training & evaluation pipelines
MultimodalSOTANet = MultimodalBoundaryAwareNet
