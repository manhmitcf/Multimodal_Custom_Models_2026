import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Tuple

from features.motion_kinematics import FishMotionKinematics5Ch
from features.audio_frontend import AudioFrontend
from .video_backbone import ConvNeXtNanoVideoBackbone
from .audio_backbone import AudioMLPBackbone
from .multimodal_fusion import MultimodalTournamentFusion


class MultimodalSOTANet(nn.Module):
    """
    Hierarchical Multimodal Tournament Network with Video Kinematics Tie-Breakers (Unified 384-dim).
    Specifically architected to resolve fish feeding intensity assessment across 4 classes
    (None, Strong, Medium, Weak) via 2-level tournament hierarchy with Configurable Video Kinematics
    Tie-Breakers (B12, B23, B13):

      1. Visual-Kinematic Stream (~2.613M params):
         5-Channel ConvNeXt-Nano (Spatial RGB + Flow (u, v)) with Stem Conv2d(5->48, k=4, s=4)
         and Composite LayerNorm temporal aggregation (composite_ln) producing native 384-dim representation.
      2. Acoustic Time-Frequency Stream (~2.173M params):
         High-Resolution TKEO-STFT Audio Frontend (256 kHz, 2049 linear bins, 4,098 params)
         + 2-hidden-layer AudioMLPBackbone (2049 -> 768 -> 512 -> 384, 2,168,448 params).
      3. Pairwise Tournament Fusion (~0.374M params with B23 enabled):
         - Direct Cross-Modal Reliability Gating: g = sigma(W[f_V || f_A]).
           f_fused = LayerNorm(g * f_V + (1-g) * f_A) -> Directly fed into Tournament Head.
         - Level 1: Feeding Activity Gating Head (None vs Active Feeding, unified 192-dim).
         - Level 2: 3 Specialized Pairwise Subspace Heads on f_fused
             with Configurable Video Kinematics Referees on f_video (B12, B23, B13).
         - Dynamic Tie-Breaker Intervention: logit = logit_base + gamma * u_tie * logit_video.
         - Tournament Borda Voting to derive final calibrated multi-class probabilities.
      4. Auxiliary Unimodal Classifier Heads (3,080 params):
         - Video Aux Head: Linear(384, 4)
         - Audio Aux Head: Linear(384, 4)

    Total Parameters:
      - Default (B23 enabled): 5,163,057 (~5.163M) | 121 parameter tensors.
      - All 3 Referees enabled: 5,312,053 (~5.312M) | 135 parameter tensors.
      - Pure Baseline (0 Referees): 5,088,559 (~5.089M) | 114 parameter tensors.
    """
    model_name: str = "MultimodalSOTANet"

    def __init__(
        self,
        classes_num: int = 4,
        embed_dim: int = 384,
        audio_frontend: Optional[AudioFrontend] = None,
        image_size: int = 224,
        num_frames: int = 2,
        in_chans: int = 5,
        temporal_mode: str = "composite_ln",
        audio_hidden_dims: Tuple[int, ...] = (768, 512),
        tie_breakers: Optional[Any] = None,
        video_drop_path: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__()
        self.classes_num = classes_num
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.image_size = image_size
        self.in_chans = in_chans
        self.temporal_mode = temporal_mode
        self.audio_hidden_dims = audio_hidden_dims
        self.tie_breakers = tie_breakers
        self.video_drop_path = video_drop_path

        # 1. Frontends
        self.audio_frontend = audio_frontend if audio_frontend is not None else AudioFrontend()
        self.motion_kinematics = FishMotionKinematics5Ch(image_size=image_size)

        # 2. Backbones (~4.78M)
        self.video_backbone = ConvNeXtNanoVideoBackbone(
            embed_dim=embed_dim,
            in_chans=in_chans,
            num_frames=num_frames,
            drop_path_rate=video_drop_path,
            temporal_mode=temporal_mode,
        )
        self.audio_backbone = AudioMLPBackbone(
            in_features=2049,
            hidden_dims=audio_hidden_dims,
            embed_dim=embed_dim,
            dropout=0.1,
        )

        # 3. Multimodal Tournament Fusion (Direct Gated, 384-dim)
        self.fusion = MultimodalTournamentFusion(
            dim=embed_dim,
            dropout=0.1,
            tie_breakers=tie_breakers,
            **kwargs
        )

        # 4. Auxiliary Unimodal Classifier Heads (3,080 params)
        # Allows independent supervision, accuracy tracking, and multi-objective optimization
        self.aux_head_video = nn.Linear(embed_dim, classes_num)
        self.aux_head_audio = nn.Linear(embed_dim, classes_num)

    def forward(
        self,
        video_input: torch.Tensor,
        audio_input: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Hierarchical End-to-End Forward Pass:
          Raw Inputs:
            - video_input: [B, 2, 3, 224, 224] (RGB multi-frame clip)
            - audio_input: [B, 512000] (2.0s @ 256 kHz raw 1D acoustic waveform)
          Returns:
            - Full prediction dictionary containing tournament voting probabilities,
              calibrated logits, pairwise boundaries, uncertainty, and unimodal logits.
        """
        # Step 1: Preprocessing & Frontend Extraction
        if video_input.ndim == 5 and video_input.size(2) == 3:
            frames_5ch, _ = self.motion_kinematics(video_input)
        else:
            frames_5ch = video_input

        if audio_input.ndim >= 1 and audio_input.size(-1) > 2049:
            stft_feat = self.audio_frontend(audio_input)
        else:
            stft_feat = audio_input

        # Step 2: Unimodal Spatiotemporal Feature Extraction
        f_video = self.video_backbone(frames_5ch)
        f_audio = self.audio_backbone(stft_feat)

        # Auxiliary Unimodal Logits (for standalone evaluation & multi-objective training)
        logits_video = self.aux_head_video(f_video)
        logits_audio = self.aux_head_audio(f_audio)

        # Step 3: Direct Gated Multimodal Tournament Fusion with Video Tie-Breakers
        fusion_outputs = self.fusion(
            f_video=f_video,
            f_audio=f_audio
        )

        # Step 4: Assemble Comprehensive Output
        outputs = {
            "clipwise_output": fusion_outputs["clipwise_output"],
            "logits": fusion_outputs.get("logits", fusion_outputs["clipwise_output"]),
            "probabilities": fusion_outputs["probabilities"],
            "logits_video": logits_video,
            "logits_audio": logits_audio,
            "uncertainty": fusion_outputs.get("uncertainty"),
            "modality_weights": fusion_outputs.get("modality_weights"),
            "expected_intensity": fusion_outputs.get("expected_intensity"),
            "gate": fusion_outputs.get("gate"),
            "f_fused": fusion_outputs.get("f_fused"),
            # Tournament outputs
            "logit_act": fusion_outputs.get("logit_act"),
            "p_feeding": fusion_outputs.get("p_feeding"),
            "logit_12": fusion_outputs.get("logit_12"),
            "logit_12_base": fusion_outputs.get("logit_12_base"),
            "logit_12_v": fusion_outputs.get("logit_12_v"),
            "u_tie_12": fusion_outputs.get("u_tie_12"),
            "gamma_12": fusion_outputs.get("gamma_12"),
            "gamma_12_v": fusion_outputs.get("gamma_12_v"),
            "logit_23": fusion_outputs.get("logit_23"),
            "logit_23_base": fusion_outputs.get("logit_23_base"),
            "logit_23_v": fusion_outputs.get("logit_23_v"),
            "u_tie_23": fusion_outputs.get("u_tie_23"),
            "gamma_23": fusion_outputs.get("gamma_23"),
            "gamma_23_v": fusion_outputs.get("gamma_23_v"),
            "logit_13": fusion_outputs.get("logit_13"),
            "logit_13_base": fusion_outputs.get("logit_13_base"),
            "logit_13_v": fusion_outputs.get("logit_13_v"),
            "u_tie_13": fusion_outputs.get("u_tie_13"),
            "gamma_13": fusion_outputs.get("gamma_13"),
            "gamma_13_v": fusion_outputs.get("gamma_13_v"),
            # Tournament pairwise winning probabilities & Borda voting scores
            "p_w_over_m": fusion_outputs.get("p_w_over_m"),
            "p_m_over_s": fusion_outputs.get("p_m_over_s"),
            "p_w_over_s": fusion_outputs.get("p_w_over_s"),
            "v_voting": fusion_outputs.get("v_voting"),
        }
        # Filter out None values to maintain clean output dictionary
        return {k: v for k, v in outputs.items() if v is not None}
