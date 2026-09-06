import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

from .video_backbone import FishVideoBackbone
from .audio_backbone import FishAudioBackbone
from .multimodal_fusion import EnhancedFishMultimodalFusion


def extract_online_physics_features(video_frames_rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes external physical features on-the-fly from RGB video frames [B, T, 3, H, W]:
    1. Fourth Channel (Temporal Frame Difference): |I_t - I_{t-1}|
    2. White Water Foam Area Ratio: Pixels where RGB mean > 0.8
    3. Foam Expansion & Decay Rate: dA/dt across time
    4. Fish Aggregation Density: Activity in central feeding zone
    Zero learnable parameters. Runs entirely on GPU tensor.
    """
    B, T, C, H, W = video_frames_rgb.shape
    device = video_frames_rgb.device
    dtype = video_frames_rgb.dtype

    # 1. Compute 4th channel: Temporal Frame Difference
    diff = torch.zeros(B, T, 1, H, W, dtype=dtype, device=device)
    if T >= 2:
        diff[:, 1:] = torch.abs(video_frames_rgb[:, 1:] - video_frames_rgb[:, :-1]).mean(dim=2, keepdim=True)
    video_4ch = torch.cat([video_frames_rgb, diff], dim=2) # [B, T, 4, H, W]

    # 2. White Water Foam Ratio
    is_white = (video_frames_rgb.mean(dim=2) > 0.8).float() # [B, T, H, W]
    foam_per_frame = is_white.mean(dim=(-2, -1))           # [B, T]
    white_foam_area = foam_per_frame.mean(dim=-1, keepdim=True) # [B, 1]

    # 3. Foam Expansion Rate dA/dt
    if T >= 2:
        foam_rate = torch.abs(foam_per_frame[:, 1:] - foam_per_frame[:, :-1]).mean(dim=-1, keepdim=True) # [B, 1]
    else:
        foam_rate = torch.zeros(B, 1, dtype=dtype, device=device)

    # 4. Central Feeding Zone Fish Density
    h_start, h_end = H // 4, 3 * H // 4
    w_start, w_end = W // 4, 3 * W // 4
    center_roi = diff[:, :, :, h_start:h_end, w_start:w_end]
    fish_density = center_roi.mean(dim=(1, 2, 3, 4)).unsqueeze(-1) # [B, 1]

    external_physics_feats = torch.cat([white_foam_area, foam_rate, fish_density], dim=-1) # [B, 3]
    return video_4ch, external_physics_feats


class DualStreamFishNet(nn.Module):
    """
    DualStreamFishNet: Custom Dual-Stream Multimodal Architecture for Fish Feeding Intensity Assessment.
    
    Architectural Specification (< 5M Parameters):
    - Video Stream (~1.45M params):
        * FishVideoBackbone accepting 4 channels [R, G, B, Frame_Diff]
        * Temporal Shift Module (TSM) with 0 params / 0 FLOPs for temporal dynamics
        * Motion Excitation (ME) modules highlighting feeding ripples and splashes
        * Extracts: f_spatial (Group 1), f_motion (Group 2), f_v_seq [B, T, D]
    - Audio Stream (~1.28M params):
        * FishAudioBackbone accepting 128 Log-Mel Spectrogram bins
        * Frequency Attention Block prioritizing 2-8 kHz feeding band, suppressing pump noise
        * 4-Stage Depthwise Separable Convolutions
        * 1D Temporal Rhythm Stream capturing pulse repetition rate (cadence)
        * Extracts: f_frequency (Group 3a), f_rhythm (Group 3b), f_a_seq [B, T', D]
    - Multimodal Fusion Stream (~0.35M params):
        * Bi-directional Cross-Attention (Video-to-Audio and Audio-to-Video)
        * Spectral-Spatial FiLM Modulation (Frequency modulates Foam sensitivity)
        * Temporal Phase-Lag Synchronization Score
        * Physics-Informed Dynamic Reliability Gating (using Foam, dA/dt, and Density)
    - Total Model Parameters: ~3.08 Million Parameters (< 5.0M strict budget).
    """
    def __init__(
        self,
        classes_num: int = 4,
        embed_dim: int = 128,
        audio_frontend: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.classes_num = classes_num
        self.embed_dim = embed_dim
        self.model_name = "dual_stream_fish_net"
        self.audio_frontend = audio_frontend

        # 1. Custom Video Backbone (4-channel RGB+Diff, TSM, Motion Excitation)
        self.video_backbone = FishVideoBackbone(
            in_channels=4,
            embed_dim=embed_dim,
            n_segment=2
        )

        # 2. Custom Audio Backbone (Log-Mel, Frequency Attention, 1D Rhythm)
        self.audio_backbone = FishAudioBackbone(
            in_channels=1,
            embed_dim=embed_dim
        )

        # 3. Enhanced Physics-Informed Bi-Directional Fusion
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
                - [B, T, 3, H, W] (RGB sequence) -> 4th channel & physics features automatically calculated
                - [B, T, 4, H, W] (Pre-assembled 4-channel tensor)
                - [B, 3, H, W] (Single frame, automatically expanded to T=2)
            audio_input: Audio tensor.
                - [B, 1, Ta, 128] (Log-Mel Spectrogram)
                - [B, 128000] (Raw waveform, converted via self.audio_frontend if present)
            external_features: Optional [B, 3] tensor containing precomputed physics scalars
                (white_foam_area, foam_rate, fish_density).
                
        Returns:
            Dictionary containing:
            - 'clipwise_output': Classification logits [B, 4]
            - 'logits': Alias for clipwise_output
            - 'f_spatial': Group 1 Spatial feature vector [B, embed_dim]
            - 'f_motion': Group 2 Motion feature vector [B, embed_dim]
            - 'f_frequency': Group 3a Audio frequency feature vector [B, embed_dim]
            - 'f_rhythm': Group 3b Audio rhythm feature vector [B, embed_dim]
            - 'f_fused': Joint multimodal representation [B, fused_dim]
            - 'modality_weights': Adaptive modality confidence [B, 2] (w_video, w_audio)
            - 'sync_score': Visual-acoustic temporal synchronization score [B, 1]
            - 'gating_alpha': Primary video reliability factor [B, 1]
        """
        # Ensure 5D video input [B, T, C, H, W]
        if video_input.dim() == 4:
            video_input = video_input.unsqueeze(1) # [B, 1, C, H, W]
            # Duplicate to T=2 if single frame
            if video_input.size(1) == 1:
                video_input = torch.cat([video_input, video_input], dim=1)

        # 1. Preprocess Video: Prepare 4 channels & Physics Scalars
        if video_input.size(2) == 3:
            video_4ch, computed_physics = extract_online_physics_features(video_input)
            if external_features is None or external_features.numel() == 0:
                external_features = computed_physics
        elif video_input.size(2) == 4:
            video_4ch = video_input
            if external_features is None or external_features.numel() == 0:
                external_features = torch.zeros(video_input.size(0), 3, device=video_input.device, dtype=video_input.dtype)
        else:
            raise ValueError(f"Expected 3 or 4 channels for video_input, got {video_input.size(2)}")

        # 2. Extract Video Representations
        f_video, f_spatial, f_motion, f_v_seq = self.video_backbone(video_4ch)

        # 3. Preprocess & Extract Audio Representations
        if audio_input.dim() == 2 and self.audio_frontend is not None:
            audio_mel = self.audio_frontend(audio_input)
        elif audio_input.dim() == 3 and audio_input.size(1) == 1 and audio_input.size(-1) > 128 and self.audio_frontend is not None:
            audio_mel = self.audio_frontend(audio_input.squeeze(1))
        else:
            audio_mel = audio_input

        f_audio, f_frequency, f_rhythm, f_a_seq = self.audio_backbone(audio_mel)

        # 4. Enhanced Physics-Informed Multimodal Fusion
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
            "f_spatial": f_spatial,
            "f_motion": f_motion,
            "f_frequency": f_frequency,
            "f_rhythm": f_rhythm,
            "f_fused": fusion_res["f_fused"],
            "modality_weights": modality_weights,
            "sync_score": sync_score,
            "gating_alpha": w_video
        }
