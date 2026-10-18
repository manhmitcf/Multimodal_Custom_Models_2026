import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any

from features.audio_frontend import AudioFrontend
from .audio_backbone import ModularAudioMLPBackbone, init_layer


class AudioSOTANet(nn.Module):
    """
    Pure Unimodal Acoustic SOTA Network for Fish Feeding Intensity Assessment (~1.87M params for W_768).
    Pipeline:
      1. Raw Audio Waveform (2.0s @ 256 kHz = 512,000 samples)
      2. TKEO-STFT Audio Frontend -> Normalized Spectral Vector [B, 2049]
      3. Modular Audio MLP Backbone -> Latent Acoustic Embedding f_audio [B, embed_dim=384]
      4. Linear Classifier Head -> Multi-Class Logits [B, 4]
    """
    def __init__(
        self,
        audio_frontend: Optional[nn.Module] = None,
        hidden_dims: Optional[List[int]] = None,
        embed_dim: int = 384,
        classes_num: int = 4,
        dropout: float = 0.1,
        audio_features_config: Optional[Any] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.classes_num = classes_num

        # 1. High-Resolution TKEO-STFT Audio Frontend (256 kHz, 2049 linear bins)
        if audio_frontend is not None:
            self.audio_frontend = audio_frontend
        else:
            self.audio_frontend = AudioFrontend(config=audio_features_config)

        stft_bins = getattr(self.audio_frontend, 'stft_bins', 2049)

        # 2. Parametric Audio MLP Backbone (2049 -> hidden_dims -> embed_dim=384)
        self.audio_backbone = ModularAudioMLPBackbone(
            in_features=stft_bins,
            hidden_dims=hidden_dims,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        # 3. Dedicated Classifier Head (384 -> 4 classes: None, Strong, Medium, Weak)
        self.classifier = nn.Linear(embed_dim, classes_num)
        init_layer(self.classifier)

    def forward(
        self,
        *args,
        video_input: Optional[torch.Tensor] = None,
        audio_input: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Flexible forward signature supporting both unimodal and multimodal caller protocols:
          - Single argument: model(audio_input)
          - Two arguments: model(dummy_video, audio_input) (used by FLOPs profiling, latency measurement, verify_dry_run)
          - Keyword arguments: model(audio_input=audio)
        """
        if len(args) == 1:
            a_in = args[0]
        elif len(args) >= 2:
            # First is dummy_video, second is audio_input
            a_in = args[1]
        elif audio_input is not None:
            a_in = audio_input
        elif video_input is not None and video_input.ndim == 2 and video_input.size(-1) > 224:
            # Fallback if audio was passed as video_input
            a_in = video_input
        else:
            raise ValueError("Audio input tensor not provided to AudioSOTANet.forward()")

        # 1. Feature extraction through TKEO-STFT frontend
        spec_vector = self.audio_frontend(a_in)  # [B, 2049]

        # 2. Embedding through Modular Audio MLP
        f_audio = self.audio_backbone(spec_vector)  # [B, embed_dim]

        # 3. Classification logits
        logits = self.classifier(f_audio)  # [B, classes_num]

        return {
            "clipwise_output": logits,
            "logits": logits,
            "f_audio": f_audio,
            "spec_vector": spec_vector
        }
