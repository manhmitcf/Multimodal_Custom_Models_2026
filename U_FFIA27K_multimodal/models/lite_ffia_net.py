import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .video_backbone import VideoSpatiotemporalBackbone
from .audio_backbone import AudioAcousticBackbone
from .multimodal_fusion import (
    BidirectionalCrossAttentionFusion,
    MultimodalBottleneckFusion,
    AdaptiveModalityGate
)


class LiteFFIANet(nn.Module):
    """
    LiteFFIA-Net v2: High-Performance Lightweight Multimodal Network (< 5.0M Params)
    for Fish Feeding Intensity Assessment.
    
    Architectural Highlights:
    1. Group 1 (Spatial Features): Full MobileNetV2 pretrained 1280-channel representations
       capturing fish density, static foam, and spatial clustering on the latest frame.
    2. Group 2 (Motion Features): Motion Excitation (ME) at stem layer captures temporal
       difference ΔF = |F_end - F_start| with dedicated motion projection.
    3. Group 3 (Audio Frequency & Rhythm): 6-stage Depthwise Inverted Residuals +
       Frequency-SE (2-8 kHz fish chewing focus) + 1D Temporal Rhythm Conv.
    4. SOTA Multimodal Fusion:
       - Bidirectional Multi-Head Cross-Attention (BMCA, IEEE TASE 2024 / ACAF-Net):
         V -> A (Visual queries Audio) and A -> V (Audio queries Visual).
       - Adaptive Modality Reliability Gating (AMRG): dynamically balances visual and acoustic confidence.
    5. Efficiency & Budget:
       - Parameters: 4,806,183 (~4.81M, strictly < 5.0M target).
       - Computation: ~0.91 GFLOPs (< 1.0 GFLOPs) for high-FPS edge deployment.
    """
    def __init__(
        self,
        classes_num: int = 4,
        embed_dim: int = 224,
        num_bottlenecks: int = 4,
        num_heads: int = 4,
        pretrained_video: bool = True,
        audio_frontend: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.classes_num = classes_num
        self.embed_dim = embed_dim
        self.model_name = "lite_ffia_net"
        self.audio_frontend = audio_frontend

        # 1. Visual Spatiotemporal Branch (Full 1280-ch MobileNetV2 + Motion Excitation)
        self.video_backbone = VideoSpatiotemporalBackbone(
            embed_dim=embed_dim,
            pretrained=pretrained_video
        )

        # 2. Acoustic Branch (6-Stage Inverted Residuals + Freq-SE + Rhythm Conv1D)
        self.audio_backbone = AudioAcousticBackbone(
            embed_dim=embed_dim
        )

        # 3. Bidirectional Multi-Head Cross-Attention Fusion with Adaptive Gating
        self.fusion = BidirectionalCrossAttentionFusion(
            dim=embed_dim,
            num_heads=num_heads
        )
        
        # Alias for backward compatibility without duplicating parameters
        self.modality_gate = self.fusion.gate

        # 4. Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 96),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(96, classes_num)
        )

    def get_name(self) -> str:
        return self.model_name

    def forward(
        self,
        video_input: torch.Tensor,
        audio_input: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward Pass of LiteFFIA-Net v2.
        
        Args:
            video_input: Video tensor [B, T, 3, H, W] (T >= 2) or [B, 3, H, W].
            audio_input: Audio Log-Mel Spectrogram [B, 1, Ta, 128] or raw waveform.
            
        Returns:
            Dictionary containing:
            - 'clipwise_output': Classification logits [B, 4]
            - 'logits': Classification logits [B, 4]
            - 'f_spatial': Group 1 Spatial feature vector [B, embed_dim]
            - 'f_motion': Group 2 Motion feature vector [B, embed_dim]
            - 'f_frequency': Group 3a Audio frequency feature vector [B, embed_dim]
            - 'f_rhythm': Group 3b Audio rhythm feature vector [B, embed_dim]
            - 'f_fused': Fused multimodal representation [B, embed_dim]
            - 'gating_alpha': Adaptive visual confidence factor [B, 1]
        """
        # Ensure video input has 5 dimensions [B, T, C, H, W]
        if video_input.dim() == 4:
            video_input = video_input.unsqueeze(1)  # [B, 1, C, H, W]

        # 1. Extract Visual Features (Group 1: Spatial 1280-ch, Group 2: Motion)
        f_video, f_spatial, f_motion = self.video_backbone(video_input)

        # 2. Extract Acoustic Features (Group 3a: Frequency, Group 3b: Rhythm)
        if audio_input.dim() == 2 and self.audio_frontend is not None:
            audio_input = self.audio_frontend(audio_input)
        elif audio_input.dim() == 3 and audio_input.size(1) == 1 and audio_input.size(-1) > 128 and self.audio_frontend is not None:
            audio_input = self.audio_frontend(audio_input.squeeze(1))

        f_audio, f_frequency, f_rhythm = self.audio_backbone(audio_input)

        # 3. SOTA Bidirectional Cross-Modal Attention Fusion + Reliability Gating
        f_fused, alpha = self.fusion(f_video, f_audio)

        # 4. Output Prediction Head
        logits = self.classifier(f_fused)

        return {
            "clipwise_output": logits,
            "logits": logits,
            "f_spatial": f_spatial,
            "f_motion": f_motion,
            "f_frequency": f_frequency,
            "f_rhythm": f_rhythm,
            "f_fused": f_fused,
            "gating_alpha": alpha
        }

