import os
import logging
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, densenet121

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

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        pooled = x1 + x2

        emb = F.relu(self.fc1(pooled))  # [B, 512]
        logits = self.fc_audioset(emb)
        return logits, emb


class TeacherConvNeXtTiny(nn.Module):
    """
    Self-contained ConvNeXt-Tiny Video Teacher model (classes_num=4).
    Accepts RGB image tensor [B, 3, 224, 224].
    Penultimate feature embedding: 768-dim.
    Spatial feature map: [B, 768, 7, 7].
    """
    def __init__(self, classes_num: int = 4):
        super().__init__()
        self.model = convnext_tiny(weights=None)
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, classes_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat_map = self.model.features(x)  # [B, 768, 7, 7]
        pooled = self.model.avgpool(feat_map)  # [B, 768, 1, 1]
        norm_pooled = self.model.classifier[0](pooled)  # LayerNorm2d [B, 768, 1, 1]
        flattened = self.model.classifier[1](norm_pooled)  # Flatten -> [B, 768]
        logits = self.model.classifier[2](flattened)  # Linear -> [B, 4]
        return logits, flattened, feat_map


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

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat_map = self.model.features(x)  # [B, 1024, 7, 7]
        out = F.relu(feat_map, inplace=False)
        pooled = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)  # [B, 1024]
        logits = self.model.classifier(pooled)
        return logits, pooled, feat_map


# Official HuggingFace artifact URLs for holdout experiment
DEFAULT_VIDEO_TEACHER_URL = (
    "https://huggingface.co/datasets/hoangphihung442004/Results_U_FFIA27K_video/resolve/main/"
    "ConvNeXtTiny_holdout_random_sample_20260729_153012.zip?download=true"
)
DEFAULT_AUDIO_TEACHER_URL = (
    "https://huggingface.co/datasets/hoangphihung442004/Results_U_FFIA27K_audio/resolve/main/"
    "PANNS_Cnn6_holdout_random_sample_20260729_153012.zip?download=true"
)


def download_and_extract_checkpoint(url: str, target_dir: str, expected_filename: str) -> str:
    """
    Downloads zip archive from URL, extracts into target_dir, and returns absolute path
    to the expected checkpoint file.
    """
    os.makedirs(target_dir, exist_ok=True)

    # 1. Quick check if already extracted
    for root, _, files in os.walk(target_dir):
        if expected_filename in files:
            found = os.path.join(root, expected_filename)
            logger.info(f"[*] Found existing teacher checkpoint at: {found}")
            return os.path.abspath(found)

    logger.info("==================================================")
    logger.info(f"[*] Teacher checkpoint '{expected_filename}' not found locally.")
    logger.info(f"[*] Automatically downloading from HuggingFace:")
    logger.info(f"    URL: {url}")
    logger.info(f"    Target Directory: {target_dir}")
    logger.info("==================================================")

    import urllib.request
    import zipfile
    import tempfile

    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    temp_zip_path = temp_zip.name
    temp_zip.close()

    try:
        def _reporthook(block_num, block_size, total_size):
            if total_size > 0 and block_num % 50 == 0:
                percent = min(100.0, block_num * block_size / total_size * 100.0)
                downloaded_mb = block_num * block_size / 1e6
                total_mb = total_size / 1e6
                print(f"\rDownloading teacher checkpoint: {percent:.1f}% ({downloaded_mb:.1f}MB / {total_mb:.1f}MB)", end="", flush=True)

        urllib.request.urlretrieve(url, temp_zip_path, reporthook=_reporthook)
        print()
        logger.info(f"[*] Extracting teacher archive to '{target_dir}'...")
        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        logger.info("[*] Successfully extracted teacher archive.")
    except Exception as exc:
        raise RuntimeError(f"Failed to auto-download and extract teacher from '{url}': {exc}") from exc
    finally:
        if os.path.exists(temp_zip_path):
            try:
                os.remove(temp_zip_path)
            except OSError:
                pass

    for root, _, files in os.walk(target_dir):
        if expected_filename in files:
            found = os.path.join(root, expected_filename)
            logger.info(f"[*] Verified teacher checkpoint at: {found}")
            return os.path.abspath(found)

    raise FileNotFoundError(f"Checkpoint '{expected_filename}' not found after extracting '{url}' into '{target_dir}'.")


def ensure_teacher_checkpoints(
    video_ckpt_path: Optional[str] = None,
    audio_ckpt_path: Optional[str] = None
) -> Tuple[str, str]:
    """
    Ensures that both Video Teacher (ConvNeXt-Tiny) and Audio Teacher (PANNS_Cnn6) checkpoints exist.
    If not found on disk at the requested or standard fallback paths, automatically downloads
    and extracts them from Hugging Face into the 'teachers' directory.
    """
    from pathlib import Path
    current_file = Path(__file__).resolve()
    pkg_dir = current_file.parent.parent       # U_FFIA27K_multimodal
    repo_root = pkg_dir.parent                 # Repository root or workspace root

    # 1. Resolve or auto-download Video Teacher
    resolved_v = None
    v_candidates = []
    if video_ckpt_path:
        v_candidates.extend([
            Path(video_ckpt_path),
            pkg_dir / video_ckpt_path,
            repo_root / video_ckpt_path,
        ])
    v_candidates.extend([
        repo_root / "teachers" / "ConvNeXtTiny" / "DL_video" / "checkpoint" / "convnext_tiny" / "video_best.pt",
        pkg_dir / "teachers" / "ConvNeXtTiny" / "DL_video" / "checkpoint" / "convnext_tiny" / "video_best.pt",
    ])
    for c in v_candidates:
        if c.is_file():
            resolved_v = str(c.resolve())
            break

    if not resolved_v:
        target_v_dir = str((repo_root / "teachers" / "ConvNeXtTiny").resolve())
        resolved_v = download_and_extract_checkpoint(
            url=DEFAULT_VIDEO_TEACHER_URL,
            target_dir=target_v_dir,
            expected_filename="video_best.pt"
        )

    # 2. Resolve or auto-download Audio Teacher
    resolved_a = None
    a_candidates = []
    if audio_ckpt_path:
        a_candidates.extend([
            Path(audio_ckpt_path),
            pkg_dir / audio_ckpt_path,
            repo_root / audio_ckpt_path,
        ])
    a_candidates.extend([
        repo_root / "teachers" / "PANNS_Cnn6" / "DL_audio" / "checkpoint" / "panns_cnn6" / "audio_best.pt",
        pkg_dir / "teachers" / "PANNS_Cnn6" / "DL_audio" / "checkpoint" / "panns_cnn6" / "audio_best.pt",
    ])
    for c in a_candidates:
        if c.is_file():
            resolved_a = str(c.resolve())
            break

    if not resolved_a:
        target_a_dir = str((repo_root / "teachers" / "PANNS_Cnn6").resolve())
        resolved_a = download_and_extract_checkpoint(
            url=DEFAULT_AUDIO_TEACHER_URL,
            target_dir=target_a_dir,
            expected_filename="audio_best.pt"
        )

    return resolved_v, resolved_a


class OfflineTeacherEnsemble(nn.Module):
    """
    Offline Teacher Ensemble managing frozen Video Teacher (ConvNeXt-Tiny)
    and Audio Teacher (PANNS_Cnn6).

    - Automatically downloads & extracts pretrained holdout checkpoints if missing.
    - Loaded once at training start.
    - Parameters strictly frozen (requires_grad = False).
    - Mode strictly eval().
    - Inferences wrapped inside torch.no_grad().
    - Incur 0 runtime / memory overhead at deployment (not saved with student).
    """
    def __init__(
        self,
        video_ckpt_path: Optional[str] = None,
        audio_ckpt_path: Optional[str] = None,
        classes_num: int = 4,
        device: Optional[torch.device] = None,
        auto_download: bool = True
    ):
        super().__init__()
        self.classes_num = classes_num
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Auto-download & resolve checkpoints
        if auto_download:
            video_ckpt_path, audio_ckpt_path = ensure_teacher_checkpoints(video_ckpt_path, audio_ckpt_path)

        # 1. Initialize Teacher Models & Frontend
        from features.audio_frontend import AudioFrontend
        from config import AudioFeaturesConfig
        tea_audio_cfg = AudioFeaturesConfig(
            sample_rate=64000,
            window_size=2048,
            hop_size=1024,
            mel_bins=128,
            fmin=1,
            fmax=32000,
            use_tkeo=False
        )
        self.audio_frontend = AudioFrontend(tea_audio_cfg)
        self.video_teacher = TeacherConvNeXtTiny(classes_num=classes_num)
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

        video_teacher_name = self.video_teacher.__class__.__name__
        logger.info("==================================================")
        logger.info("Initialized OfflineTeacherEnsemble:")
        logger.info(f"  - Video Teacher: {video_teacher_name} (Loaded from '{video_ckpt_path}')")
        logger.info(f"  - Audio Teacher: PANNS_Cnn6  (Loaded from '{audio_ckpt_path}')")
        logger.info(f"  - Multi-Level KD: Logits + Spatial Attention Map (7x7) + Penultimate Embeddings (768-dim)")
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

        # Dynamically switch between ConvNeXt-Tiny and DenseNet121 based on weights structure
        if any('classifier.0.weight' in k or 'features.0.0.weight' in k for k in clean_state):
            if not isinstance(self.video_teacher, TeacherConvNeXtTiny):
                self.video_teacher = TeacherConvNeXtTiny(classes_num=self.classes_num).to(self.device)
            self.video_teacher.model.load_state_dict(clean_state, strict=True)
            logger.info(f"  [*] Loaded Video Teacher (ConvNeXt-Tiny) weights from: {ckpt_path}")
        else:
            if not isinstance(self.video_teacher, TeacherDenseNet121):
                self.video_teacher = TeacherDenseNet121(classes_num=self.classes_num).to(self.device)
            self.video_teacher.model.load_state_dict(clean_state, strict=True)
            logger.info(f"  [*] Loaded Video Teacher (DenseNet121) weights from: {ckpt_path}")

    def _load_audio_teacher(self, ckpt_path: str) -> None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Audio teacher checkpoint not found at '{ckpt_path}'")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

        # Load frontend normalization (bn0) and backbone weights
        frontend_state = {}
        clean_state = {}
        for k, v in state_dict.items():
            if k.startswith('frontend.'):
                frontend_state[k[len('frontend.'):]] = v
            elif k.startswith('backbone.'):
                clean_state[k[len('backbone.'):]] = v
            else:
                clean_state[k] = v

        if frontend_state:
            self.audio_frontend.load_state_dict(frontend_state, strict=False)
            logger.info("  [*] Loaded Audio Teacher frontend normalization (bn0) parameters.")

        self.audio_teacher.load_state_dict(clean_state, strict=True)
        logger.info(f"  [*] Loaded Audio Teacher (PANNS_Cnn6) weights from: {ckpt_path}")

    @torch.no_grad()
    def forward(
        self,
        video_tensor: torch.Tensor,
        audio_tensor: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through frozen teachers to generate multi-level soft targets and features.

        Args:
            video_tensor: Student video batch [B, T, C=7, H, W] or [B, C, H, W]
            audio_tensor: Student audio batch [B, num_samples] or [B, 1, Ta, 128]

        Returns:
            Dict containing:
              - teacher_logits_video: [B, classes_num]
              - teacher_logits_audio: [B, classes_num]
              - teacher_feat_video: [B, 768] (ConvNeXt-Tiny) or [B, 1024] (DenseNet121)
              - teacher_feat_audio: [B, 512]
              - teacher_feat_map_video: [B, 768, 7, 7] or [B, 1024, 7, 7]
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

        teacher_logits_v, teacher_feat_v, teacher_map_v = self.video_teacher.forward_with_features(rgb_input)
        teacher_logits_a, teacher_feat_a = self.audio_teacher.forward_with_features(audio_mel)

        return {
            "teacher_logits_video": teacher_logits_v,
            "teacher_logits_audio": teacher_logits_a,
            "teacher_feat_video": teacher_feat_v,
            "teacher_feat_audio": teacher_feat_a,
            "teacher_feat_map_video": teacher_map_v,
        }

