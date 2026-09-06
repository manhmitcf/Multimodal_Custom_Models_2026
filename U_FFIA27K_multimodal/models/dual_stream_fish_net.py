import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

from .convnext_video_backbone import FishConvNeXtBackbone
from .video_backbone import FishVideoBackbone
from .audio_backbone import FishAudioBackbone
from .multimodal_fusion import EnhancedFishMultimodalFusion
from .motion_kinematics import FishMotionKinematics


def extract_online_physics_features(video_frames_rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Legacy helper preserved for backward compatibility.
    Computes external physical features on-the-fly from RGB video frames [B, T, 3, H, W].
    """
    B, T, C, H, W = video_frames_rgb.shape
    device = video_frames_rgb.device
    dtype = video_frames_rgb.dtype

    diff = torch.zeros(B, T, 1, H, W, dtype=dtype, device=device)
    if T >= 2:
        diff[:, 1:] = torch.abs(video_frames_rgb[:, 1:] - video_frames_rgb[:, :-1]).mean(dim=2, keepdim=True)
    video_4ch = torch.cat([video_frames_rgb, diff], dim=2)

    is_white = (video_frames_rgb.mean(dim=2) > 0.8).float()
    foam_per_frame = is_white.mean(dim=(-2, -1))
    white_foam_area = foam_per_frame.mean(dim=-1, keepdim=True)

    if T >= 2:
        foam_rate = torch.abs(foam_per_frame[:, 1:] - foam_per_frame[:, :-1]).mean(dim=-1, keepdim=True)
    else:
        foam_rate = torch.zeros(B, 1, dtype=dtype, device=device)

    h_start, h_end = H // 4, 3 * H // 4
    w_start, w_end = W // 4, 3 * W // 4
    center_roi = diff[:, :, :, h_start:h_end, w_start:w_end]
    fish_density = center_roi.mean(dim=(1, 2, 3, 4)).unsqueeze(-1)

    external_physics_feats = torch.cat([white_foam_area, foam_rate, fish_density], dim=-1)
    return video_4ch, external_physics_feats


class DualStreamFishNet(nn.Module):
    """
    DualStreamFishNet-V2: Custom Dual-Stream Multimodal Architecture for Fish Feeding Intensity Assessment.
    
    Architectural Highlights (~4.52M Parameters, < 5.0M strict budget):
    - Video Stream (~2.96M params):
        * FishMotionKinematics extracting:
          1. Foam / Bubble Dynamics: Area ratio A_foam and expansion rate dA/dt
          2. Fish Swimming Velocity: dense velocity magnitude ||v|| = sqrt(vx^2 + vy^2)
          3. Movement Direction & Feeding Flux: unit vectors (cos theta, sin theta) and centripetal flux Phi_feed
        * FishConvNeXtBackbone accepting 6 channels [R, G, B, Velocity_Mag, vx, vy]
        * 7x7 Depthwise Convolutions with TSM (Temporal Shift Module)
        * LayerNorm + GELU + Inverted Bottleneck 4x + Layer Scale (gamma=1e-6)
        * Multi-stage Motion Excitation (ME2, ME3)
        * Extracts: f_spatial (Group 1), f_motion (Group 2), f_v_seq [B, T, 256]
    - Audio Stream (~0.77M params):
        * FishAudioBackbone accepting 128 Log-Mel Spectrogram bins (128kHz SR, 16ms window, 8ms hop)
        * Data-Driven Adaptive Frequency Attention (Dual-Pooling Avg + Max Spectral Energy)
        * 5 deep Depthwise Separable stages (up to 256 channels)
        * 1D Temporal Rhythm Stream capturing pulse repetition rate (cadence)
        * Extracts: f_frequency (Group 3a), f_rhythm (Group 3b), f_a_seq [B, T', 256]
    - Multimodal Fusion Stream (~0.79M params):
        * 8-head Bi-directional Cross-Attention (Video-to-Audio and Audio-to-Video)
        * Temporal Cadence Attention Module (TCAM) resolving pulse density
        * Spectral-Spatial FiLM Modulation (Frequency modulates Foam sensitivity)
        * Temporal Phase-Lag Synchronization Score
        * Physics-Informed Dynamic Reliability Gating (using Foam, dA/dt, Velocity, and Feeding Flux)
        * Adaptive Weak vs. Medium Disambiguation Head (Contrastive Boundary Margin push-pull)
    - Total Model Parameters: ~4.52M Parameters (strictly < 5.0M budget).
    """
    def __init__(
        self,
        classes_num: int = 4,
        embed_dim: int = 256,
        audio_frontend: Optional[nn.Module] = None,
        use_convnext: bool = True,
        n_segment: int = 4
    ) -> None:
        super().__init__()
        self.classes_num = classes_num
        self.embed_dim = embed_dim
        self.n_segment = n_segment
        self.model_name = "dual_stream_fish_net_v2"
        self.audio_frontend = audio_frontend
        self.use_convnext = use_convnext

        # 1. Differentiable Kinematics Extractor (Foam + Velocity Magnitude + Direction)
        self.motion_kinematics = FishMotionKinematics(image_size=224)

        # 2. Custom Video Backbone
        if use_convnext:
            self.video_backbone = FishConvNeXtBackbone(
                in_channels=6,
                embed_dim=embed_dim,
                n_segment=n_segment
            )
        else:
            self.video_backbone = FishVideoBackbone(
                in_channels=4,
                embed_dim=embed_dim,
                n_segment=n_segment
            )

        # 3. Custom Audio Backbone (Log-Mel, Data-Driven Frequency Attention, 1D Rhythm)
        self.audio_backbone = FishAudioBackbone(
            in_channels=1,
            embed_dim=embed_dim
        )

        # 4. Enhanced Multimodal Fusion with TCAM and Weak/Medium Disambiguation
        self.fusion = EnhancedFishMultimodalFusion(
            feat_dim=embed_dim,
            num_classes=classes_num
        )

    def get_name(self) -> str:
        return self.model_name

    def forward(
        self,
        video_input: torch.Tensor,
        audio_input: torch.Tensor,
        external_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            video_input: Video tensor.
                Supported shapes:
                - [B, T, 3, H, W] (RGB sequence) -> 6th channel & kinematics automatically calculated
                - [B, T, 6, H, W] (Pre-assembled 6-channel tensor)
                - [B, 3, H, W] (Single frame, automatically expanded to T=2)
            audio_input: Audio tensor.
                - [B, 1, Ta, 128] (Log-Mel Spectrogram)
                - [B, 256000] or [B, 128000] (Raw waveform, converted via self.audio_frontend)
            external_features: Optional [B, 4] tensor containing precomputed physics scalars
                (a_foam, da_dt, v_mean, phi_feed).
                
        Returns:
            Dictionary containing:
            - 'clipwise_output': Classification logits [B, 4] (with Weak/Medium disambiguation)
            - 'raw_logits': Unadjusted classification logits [B, 4]
            - 'logits': Alias for clipwise_output
            - 'delta_margin': Disambiguation margin between Weak and Medium [B, 1]
            - 'cadence_density': Acoustic burst rate variance [B, 1]
            - 'f_spatial': Group 1 Spatial feature vector [B, embed_dim]
            - 'f_motion': Group 2 Motion feature vector [B, embed_dim]
            - 'f_frequency': Group 3a Audio frequency feature vector [B, embed_dim]
            - 'f_rhythm': Group 3b Audio rhythm feature vector [B, embed_dim]
            - 'f_fused': Joint multimodal representation [B, fused_dim]
            - 'modality_weights': Adaptive modality confidence [B, 2] (w_video, w_audio)
            - 'sync_score': Visual-acoustic temporal synchronization score [B, 1]
            - 'gating_alpha': Primary video reliability factor [B, 1]
            - 'kinematics_feats': Motion kinematics vector [B, 4]
        """
        # Ensure 5D video input [B, T, C, H, W]
        if video_input.dim() == 4:
            video_input = video_input.unsqueeze(1) # [B, 1, C, H, W]
            if video_input.size(1) == 1:
                video_input = video_input.repeat(1, self.n_segment, 1, 1, 1)

        # 1. Preprocess Video & Extract Kinematics (Sủi bọt + Vận tốc + Hướng di chuyển)
        if video_input.size(2) == 3:
            video_6ch, computed_kinematics = self.motion_kinematics(video_input)
            if external_features is None or external_features.numel() == 0:
                external_features = computed_kinematics
        elif video_input.size(2) == 6:
            video_6ch = video_input
            if external_features is None or external_features.numel() == 0:
                external_features = torch.zeros(video_input.size(0), 4, device=video_input.device, dtype=video_input.dtype)
        elif video_input.size(2) == 4 and not self.use_convnext:
            video_6ch = video_input
            if external_features is None or external_features.numel() == 0:
                external_features = torch.zeros(video_input.size(0), 4, device=video_input.device, dtype=video_input.dtype)
        else:
            raise ValueError(f"Expected 3 or 6 channels for video_input (or 4 for legacy), got {video_input.size(2)}")

        # 2. Extract Video Representations (FishConvNeXtBackbone)
        f_video, f_spatial, f_motion, f_v_seq = self.video_backbone(video_6ch)

        # 3. Preprocess & Extract Audio Representations (FishAudioBackbone)
        if audio_input.dim() == 2 and self.audio_frontend is not None:
            audio_mel = self.audio_frontend(audio_input)
        elif audio_input.dim() == 3 and audio_input.size(1) == 1 and audio_input.size(-1) > 128 and self.audio_frontend is not None:
            audio_mel = self.audio_frontend(audio_input.squeeze(1))
        else:
            audio_mel = audio_input

        f_audio, f_frequency, f_rhythm, f_a_seq = self.audio_backbone(audio_mel)

        # 4. Enhanced Multimodal Fusion with Weak vs. Medium Disambiguation
        fusion_res = self.fusion(
            f_v_seq=f_v_seq,
            f_a_seq=f_a_seq,
            f_v_spa=f_spatial,
            f_v_mot=f_motion,
            f_a_frq=f_frequency,
            f_a_rhy=f_rhythm,
            f_audio=f_audio,
            external_physics_feats=external_features
        )

        logits = fusion_res["logits"]
        modality_weights = fusion_res["modality_weights"]
        sync_score = fusion_res["sync_score"]
        w_video = fusion_res["w_video"]

        return {
            "clipwise_output": logits,
            "logits": logits,
            "raw_logits": fusion_res.get("raw_logits", logits),
            "delta_margin": fusion_res.get("delta_margin", torch.zeros(video_input.size(0), 1, device=video_input.device)),
            "cadence_density": fusion_res.get("cadence_density", torch.zeros(video_input.size(0), 1, device=video_input.device)),
            "f_spatial": f_spatial,
            "f_motion": f_motion,
            "f_frequency": f_frequency,
            "f_rhythm": f_rhythm,
            "f_fused": fusion_res["f_fused"],
            "modality_weights": modality_weights,
            "sync_score": sync_score,
            "gating_alpha": w_video,
            "kinematics_feats": external_features
        }
