import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .video_backbone import VideoSpatiotemporalBackbone
from .audio_backbone import AudioAcousticBackbone
from .multimodal_fusion import MultimodalBottleneckFusion, AdaptiveModalityGate


class LiteFFIANet(nn.Module):
    """
    LiteFFIA-Net: Ultra-Lightweight Multimodal Network for Fish Feeding Intensity Assessment.
    
    Architectural Highlights:
    1. Group 1 (Spatial Features): MobileNetV2-based spatial feature map captures fish density & static foam.
    2. Group 2 (Motion Features): Motion Excitation (ME) computes temporal difference between consecutive
       frames, modulating spatial features with 0.02M params.
    3. Group 3 (Audio Frequency & Rhythm): Frequency-SE captures 2-8 kHz splash band while 1D Conv captures cadence.
    4. SOTA Multimodal Fusion:
       - Multimodal Bottleneck Transformer (MBT, NeurIPS 2021) with K=4 bottleneck tokens.
       - Adaptive Modality Reliability Gating (AMRG) dynamically balances visual and acoustic confidence.
    5. Efficiency: Strictly < 4.0 Million parameters and ~1.2 GFLOPs for full real-time edge execution.
    """
    def __init__(
        self,
        classes_num: int = 4,
        embed_dim: int = 192,
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

        # 1. Visual Spatiotemporal Branch (Group 1 + Group 2)
        self.video_backbone = VideoSpatiotemporalBackbone(
            embed_dim=embed_dim,
            pretrained=pretrained_video
        )

        # 2. Acoustic Branch (Group 3 Frequency + Rhythm)
        self.audio_backbone = AudioAcousticBackbone(
            embed_dim=embed_dim
        )

        # 3. Multimodal Bottleneck Fusion (MBT)
        self.mbt_fusion = MultimodalBottleneckFusion(
            dim=embed_dim,
            num_bottlenecks=num_bottlenecks,
            num_heads=num_heads
        )

        # 4. Adaptive Modality Reliability Gate
        self.modality_gate = AdaptiveModalityGate(dim=embed_dim)

        # 5. Final Classification Head
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
        Forward Pass of LiteFFIA-Net.
        
        Args:
            video_input: Video tensor.
                Supported shapes:
                - [B, T, 3, H, W] (multi-frame sequence, T >= 2)
                - [B, 3, H, W] (single frame, automatically duplicated)
            audio_input: Audio Log-Mel Spectrogram [B, 1, Ta, 128].
            
        Returns:
            Dictionary containing:
            - 'clipwise_output': Classification logits [B, 4]
            - 'f_spatial': Group 1 Spatial feature vector [B, embed_dim]
            - 'f_motion': Group 2 Motion feature vector [B, embed_dim]
            - 'f_frequency': Group 3a Audio frequency feature vector [B, embed_dim]
            - 'f_rhythm': Group 3b Audio rhythm feature vector [B, embed_dim]
            - 'f_fused': Joint multimodal bottleneck embedding [B, embed_dim]
            - 'gating_alpha': Adaptive visual-acoustic gating factor [B, 1]
        """
        # Ensure video input has 5 dimensions [B, T, C, H, W]
        if video_input.dim() == 4:
            video_input = video_input.unsqueeze(1)  # [B, 1, C, H, W]

        # 1. Extract Visual Features (Group 1: Spatial, Group 2: Motion)
        f_video, f_spatial, f_motion = self.video_backbone(video_input)

        # 2. Extract Acoustic Features (Group 3a: Frequency, Group 3b: Rhythm)
        if audio_input.dim() == 2 and self.audio_frontend is not None:
            audio_input = self.audio_frontend(audio_input)
        elif audio_input.dim() == 3 and audio_input.size(1) == 1 and audio_input.size(-1) > 128 and self.audio_frontend is not None:
            audio_input = self.audio_frontend(audio_input.squeeze(1))

        f_audio, f_frequency, f_rhythm = self.audio_backbone(audio_input)

        # 3. Multimodal Bottleneck Transformer Fusion
        v_token = f_video.unsqueeze(1)  # [B, 1, embed_dim]
        a_token = f_audio.unsqueeze(1)  # [B, 1, embed_dim]
        f_fused = self.mbt_fusion(v_token, a_token)

        # 4. Adaptive Modality Reliability Gating
        alpha = self.modality_gate(f_video, f_audio)  # [B, 1] in [0, 1]
        e_gated = alpha * f_video + (1.0 - alpha) * f_audio
        
        # Residual fusion with LayerNorm
        e_final = F.layer_norm(e_gated + f_fused, (self.embed_dim,))

        # 5. Output Prediction Head
        logits = self.classifier(e_final)

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
