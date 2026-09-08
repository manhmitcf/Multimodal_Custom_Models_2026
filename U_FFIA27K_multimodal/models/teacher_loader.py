import os
import logging
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121

logger = logging.getLogger(__name__)


class ConvBlock5x5(nn.Module):
    """5x5 Convolution block matching PANNs CNN6 architecture."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, input_tensor: torch.Tensor, pool_size: Tuple[int, int] = (2, 2), pool_type: str = 'avg') -> torch.Tensor:
        x = input_tensor
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        return x


class TeacherPANNS_Cnn6(nn.Module):
    """
    Self-contained PANNS CNN6 Audio Teacher model (classes_num=4).
    Accepts Log-Mel Spectrogram [B, 1, Ta, 128].
    """
    def __init__(self, classes_num: int = 4):
        super().__init__()
        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)  # Mean over frequency bins
        (x1, _) = torch.max(x, dim=2)  # Max over time frames
        x2 = torch.mean(x, dim=2)  # Mean over time frames
        x = x1 + x2

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        logits = self.fc_audioset(x)
        return logits


class TeacherDenseNet121(nn.Module):
    """
    Self-contained DenseNet121 Video Teacher model (classes_num=4).
    Accepts RGB image tensor [B, 3, 224, 224].
    """
    def __init__(self, classes_num: int = 4):
        super().__init__()
        self.model = densenet121(weights=None)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, classes_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class OfflineTeacherEnsemble(nn.Module):
    """
    Offline Teacher Ensemble managing frozen Video Teacher (DenseNet121)
    and Audio Teacher (PANNS_Cnn6).

    - Loaded once at training start.
    - Parameters strictly frozen (requires_grad = False).
    - Mode strictly eval().
    - Inferences wrapped inside torch.no_grad().
    - Incur 0 runtime / memory overhead at deployment (not saved with student).
    """
    def __init__(
        self,
        video_ckpt_path: str,
        audio_ckpt_path: str,
        classes_num: int = 4,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.classes_num = classes_num
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Initialize Teacher Models & Frontend
        from features.audio_frontend import AudioFrontend
        from config import AudioFeaturesConfig
        tea_audio_cfg = AudioFeaturesConfig(
            sample_rate=64000,
            window_size=2048,
            hop_size=1024,
            mel_bins=128,
            use_tkeo=False
        )
        self.audio_frontend = AudioFrontend(tea_audio_cfg)
        self.video_teacher = TeacherDenseNet121(classes_num=classes_num)
        self.audio_teacher = TeacherPANNS_Cnn6(classes_num=classes_num)

        # 2. Load Checkpoints
        self._load_video_teacher(video_ckpt_path)
        self._load_audio_teacher(audio_ckpt_path)

        # 3. Freeze all parameters and set to eval mode
        for param in self.video_teacher.parameters():
            param.requires_grad = False
        for param in self.audio_teacher.parameters():
            param.requires_grad = False
        for param in self.audio_frontend.parameters():
            param.requires_grad = False

        self.video_teacher.eval()
        self.audio_teacher.eval()
        self.audio_frontend.eval()
        self.to(self.device)

        logger.info("==================================================")
        logger.info("Initialized OfflineTeacherEnsemble:")
        logger.info(f"  - Video Teacher: DenseNet121 (Loaded from '{video_ckpt_path}')")
        logger.info(f"  - Audio Teacher: PANNS_Cnn6  (Loaded from '{audio_ckpt_path}')")
        logger.info(f"  - State: Frozen (requires_grad=False), Device: {self.device}")
        logger.info("==================================================")

    def _load_video_teacher(self, ckpt_path: str) -> None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Video teacher checkpoint not found at '{ckpt_path}'")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

        # Clean prefix if saved under VideoModel wrapper
        clean_state = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.model.'):
                clean_state[k[len('backbone.model.'):]] = v
            elif k.startswith('backbone.'):
                clean_state[k[len('backbone.'):]] = v
            elif k.startswith('model.'):
                clean_state[k[len('model.'):]] = v
            else:
                clean_state[k] = v

        self.video_teacher.model.load_state_dict(clean_state, strict=True)
        logger.info(f"  [*] Loaded Video Teacher (DenseNet121) weights from: {ckpt_path}")

    def _load_audio_teacher(self, ckpt_path: str) -> None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Audio teacher checkpoint not found at '{ckpt_path}'")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

        # Only keep keys belonging to backbone
        clean_state = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                clean_state[k[len('backbone.'):]] = v
            elif not k.startswith('frontend.'):
                clean_state[k] = v

        self.audio_teacher.load_state_dict(clean_state, strict=True)
        logger.info(f"  [*] Loaded Audio Teacher (PANNS_Cnn6) weights from: {ckpt_path}")

    @torch.no_grad()
    def forward(
        self,
        video_tensor: torch.Tensor,
        audio_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through frozen teachers to generate soft targets.

        Args:
            video_tensor: Student video batch [B, T, C=7, H, W] or [B, C, H, W]
            audio_tensor: Student audio batch [B, num_samples] or [B, 1, Ta, 128]

        Returns:
            teacher_logits_video: [B, classes_num]
            teacher_logits_audio: [B, classes_num]
        """
        self.video_teacher.eval()
        self.audio_teacher.eval()
        self.audio_frontend.eval()

        # 1. Prepare video input: Extract primary RGB frame [B, 3, H, W]
        if video_tensor.dim() == 5:
            # Slices frame 0 RGB channels (0:3)
            rgb_input = video_tensor[:, 0, :3, :, :]
        elif video_tensor.dim() == 4:
            rgb_input = video_tensor[:, :3, :, :]
        else:
            raise ValueError(f"Unexpected video tensor dimension: {video_tensor.shape}")

        # 2. Prepare audio input: Convert raw waveform [B, Num_Samples] to Mel [B, 1, Ta, 128] if needed
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.to(self.device, non_blocking=True)
            audio_mel = self.audio_frontend(audio_tensor)
        elif audio_tensor.dim() == 4:
            audio_mel = audio_tensor.to(self.device, non_blocking=True)
        elif audio_tensor.dim() == 3:
            audio_mel = audio_tensor.unsqueeze(1).to(self.device, non_blocking=True)
        else:
            raise ValueError(f"Unexpected audio tensor dimension: {audio_tensor.shape}")

        rgb_input = rgb_input.to(self.device, non_blocking=True)

        teacher_logits_video = self.video_teacher(rgb_input)
        teacher_logits_audio = self.audio_teacher(audio_mel)

        return teacher_logits_video, teacher_logits_audio
