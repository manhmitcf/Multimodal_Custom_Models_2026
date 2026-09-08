import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_IMAGE_CACHE_ROOT = "/marimo/video_cache"
DEFAULT_AUDIO_CACHE_ROOT = "/marimo/audio_cache"
VALID_CACHE_MODES = ("disk", "ram", "none")


class VideoFeaturesConfig(BaseModel):
    """
    Video frame extraction & spatiotemporal kinematics parameters.
    """
    image_size: int = Field(default=224, description="Target spatial resolution H=W.")
    num_frames: int = Field(default=4, description="Number of uniform frames sampled per video clip (T=4).")
    num_channels: int = Field(default=7, description="7-channel representation (RGB + 4 Kinematics: u, v, vorticity, decel).")


class AudioFeaturesConfig(BaseModel):
    """
    Audio Log-Mel Spectrogram extraction parameters (EfficientAT compatible).
    """
    sample_rate: int = Field(default=64000, description="Audio sampling rate in Hz.")
    window_size: int = Field(default=2048, description="STFT window size in samples.")
    hop_size: int = Field(default=512, description="STFT hop size in samples.")
    mel_bins: int = Field(default=128, description="Number of Mel frequency filter banks.")
    fmin: int = Field(default=50, description="Minimum frequency for Mel filter bank in Hz.")
    fmax: int = Field(default=32000, description="Maximum frequency for Mel filter bank in Hz.")
    time_drop_width: int = Field(default=64, description="SpecAugment max time mask width.")
    time_stripes_num: int = Field(default=2, description="SpecAugment number of time stripes.")
    freq_drop_width: int = Field(default=16, description="SpecAugment max freq mask width.")
    freq_stripes_num: int = Field(default=2, description="SpecAugment number of freq stripes.")
    use_tkeo: bool = Field(default=True, description="Enable Teager-Kaiser Energy Operator Adaptive Pre-Emphasis.")
    alpha_max: float = Field(default=0.99, description="Max pre-emphasis coefficient for TKEO APE.")
    beta: float = Field(default=0.8, description="Temporal smoothing factor for TKEO APE.")


class ModelConfig(BaseModel):
    """
    Configuration for SOTA Multimodal Model (~4.78M parameters).
    Video: MobileViT-XS (10-ch) ~2.30M
    Audio: EfficientAT mn05_as (128 Mel-bins) ~1.80M
    Fusion: Google MBT + Dynamic Gating + TMC Evidential Reasoning ~0.68M
    """
    backbone: str = Field(
        default="MultimodalSOTANet",
        description="Model architecture name exported by U_FFIA27K_multimodal.models."
    )
    embed_dim: int = Field(default=224, description="Common multimodal embedding dimension.")
    num_bottlenecks: int = Field(default=4, description="Number of MBT bottleneck tokens.")
    num_heads: int = Field(default=4, description="Number of cross-attention heads.")
    pretrained_video: bool = Field(default=True, description="Initialize RGB channels from pretrained MobileViT-XS.")
    classes_num: int = Field(default=4, description="Number of output feeding intensity classes (None, Strong, Medium, Weak).")


class SplitterConfig(BaseModel):
    """
    Configuration parameters for dataset splitting.
    """
    dataset_path: str = Field(
        default='/marimo/Fish_Feeding_Intensity_Dataset',
        description="Absolute path to the raw dataset directory."
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for split reproducibility."
    )
    test_sample_per_class: int = Field(
        default=700,
        gt=0,
        description="Number of samples per class designated for test and validation subsets."
    )
    save_results: bool = Field(
        default=True,
        description="Whether to save the splits output results to CSV/JSON files."
    )
    include_video: bool = Field(
        default=True,
        description="Whether to scan video paths alongside audio paths."
    )
    split_strategy: str = Field(
        default="random_sample",
        description="Dataset split strategy: 'random_sample', 'time_series', or 'group_random'."
    )
    evaluation_mode: str = Field(
        default="holdout",
        description="Evaluation mode: 'holdout' or 'cross_validation'."
    )
    num_folds: int = Field(
        default=5,
        gt=1,
        description="Number of outer folds for cross-validation."
    )
    fold_index: Optional[int] = Field(
        default=None,
        description="Current outer fold index for cross-validation."
    )
    cv_val_ratio: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="Validation ratio split from the non-test development folds during cross-validation."
    )


class TrainConfig(BaseModel):
    """
    Master configuration schema for multimodal model training.
    """
    epochs: int = Field(default=400, gt=0, description="Total number of training epochs.")
    batch_size: int = Field(default=32, gt=0, description="Training batch size.")
    learning_rate: float = Field(default=1e-4, gt=0, description="Initial learning rate.")
    weight_decay: float = Field(default=1e-2, ge=0, description="Weight decay factor for AdamW.")
    ckpt_dir: str = Field(default="checkpoint/", description="Directory path to save checkpoints.")
    monitor: str = Field(default="val_f1_macro", description="Metric to monitor for early stopping and best checkpoint.")
    mode: Literal["min", "max"] = Field(default="max", description="Optimization direction for monitored metric.")
    early_stopping: bool = Field(default=True, description="Enable early stopping mechanism.")
    patience: int = Field(default=80, ge=1, description="Early stopping patience in epochs.")
    min_delta: float = Field(default=0.0005, ge=0.0, description="Minimum change threshold in monitored metric.")
    lr_scheduler: str = Field(default="onecycle", description="Learning rate scheduler: 'onecycle', 'cosine' or 'plateau'.")
    use_onecycle: bool = Field(default=True, description="Enable OneCycleLR scheduler.")
    warmup_epochs: int = Field(default=20, ge=0, description="Number of linear warmup epochs.")
    min_lr: float = Field(default=1e-6, gt=0, description="Minimum learning rate.")
    lambda_emd: float = Field(default=0.5, ge=0.0, description="Weight for Squared Earth Mover's Distance loss.")
    lambda_align: float = Field(default=0.2, ge=0.0, description="Weight for Cross-Modal Boundary Alignment loss.")
    num_frames: int = Field(default=4, ge=2, description="Number of frames per video input.")
    image_size: int = Field(default=224, gt=0, description="Video image spatial resolution.")
    sample_rate: int = Field(default=64000, gt=0, description="Audio sampling rate in Hz.")
    cache_mode: str = Field(default="ram", description="Caching mode: 'ram', 'disk', or 'none'.")
    dataloader_workers: int = Field(default=-1, description="Number of worker processes for DataLoader (-1 = auto).")
    prefetch_factor: Optional[int] = Field(default=2, description="Number of batches loaded in advance.")
    save_best_only: bool = Field(default=True, description="Save only the best checkpoint.")
    loss_type: str = Field(default="pairwise_tournament", description="Loss function: 'pairwise_tournament', 'ordinal_wasserstein' or 'clip_ce'.")
    weight_act: float = Field(default=0.5, ge=0.0, description="Weight for Level-1 Activity Gate BCE loss.")
    weight_pairwise: float = Field(default=0.5, ge=0.0, description="Weight for Level-2 Pairwise Boundaries loss.")
    weight_ce: float = Field(default=1.0, ge=0.0, description="Weight for Multi-class CE on Tournament Logits.")
    enable_two_phase_warmup: bool = Field(default=False, description="Enable two-phase warmup training strategy (disabled when KD is used).")
    phase1_warmup_epochs: int = Field(default=200, ge=0, description="Number of epochs for Phase 1 backbone warmup.")
    aux_loss_weight: float = Field(default=0.3, ge=0.0, description="Weight for auxiliary unimodal backbone heads.")
    enable_kd: bool = Field(default=True, description="Enable dual-teacher knowledge distillation into backbones.")
    teacher_video_model: str = Field(default="DenseNet121", description="Video teacher model architecture.")
    teacher_video_ckpt: str = Field(default="teachers/DenseNet121/DL_video/checkpoint/densenet121/fold_00/video_best.pt", description="Path to video teacher checkpoint.")
    teacher_audio_model: str = Field(default="PANNS_Cnn6", description="Audio teacher model architecture.")
    teacher_audio_ckpt: str = Field(default="teachers/PANNS_Cnn6/DL_audio/checkpoint/panns_cnn6/audio_best.pt", description="Path to audio teacher checkpoint.")
    kd_temperature_video: float = Field(default=3.0, ge=0.1, description="Softmax temperature for video KD.")
    kd_temperature_audio: float = Field(default=2.0, ge=0.1, description="Softmax temperature for audio KD.")
    kd_alpha_video: float = Field(default=0.60, ge=0.0, le=1.0, description="KD loss weight alpha for video (40% CE / 60% KD).")
    kd_alpha_audio: float = Field(default=0.60, ge=0.0, le=1.0, description="KD loss weight alpha for audio (40% CE / 60% KD).")
    weight_video_loss: float = Field(default=0.3, ge=0.0, description="Weight for video auxiliary / KD loss in total loss.")
    weight_audio_loss: float = Field(default=0.3, ge=0.0, description="Weight for audio auxiliary / KD loss in total loss.")
    weight_feature_kd: float = Field(default=0.2, ge=0.0, description="Weight for penultimate embedding cosine distillation.")
    weight_at_kd: float = Field(default=0.2, ge=0.0, description="Weight for spatial attention transfer distillation.")
    enable_feature_kd: bool = Field(default=True, description="Enable penultimate embedding feature distillation.")
    enable_at_kd: bool = Field(default=True, description="Enable intermediate spatial attention transfer distillation.")
    adaptive_kd: bool = Field(default=False, description="Enable sample-wise adaptive KD based on teacher confidence and correctness.")
    adaptive_kd_min_alpha: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum alpha weight for incorrect/uncertain teacher predictions.")
    ordinal_sigma: float = Field(default=0.5, gt=0.0, description="Gaussian bandwidth sigma for ordinal soft label smoothing.")
    lambda_ord_start: float = Field(default=0.2, ge=0.0, description="Initial weight for ordinal Wasserstein EMD loss.")
    lambda_ord_end: float = Field(default=2.0, ge=0.0, description="Final weight for ordinal Wasserstein EMD loss after cosine ramp-up.")
    model: ModelConfig = Field(default_factory=ModelConfig, description="Model architecture parameters.")
    dataset_splitter: SplitterConfig = Field(default_factory=SplitterConfig, description="Dataset splitting settings.")
    video_features: VideoFeaturesConfig = Field(default_factory=VideoFeaturesConfig, description="Video preprocessing configuration.")
    audio_features: AudioFeaturesConfig = Field(default_factory=AudioFeaturesConfig, description="Audio preprocessing configuration.")

    @classmethod
    def from_json(cls, path: str = "config/train_config.json") -> "TrainConfig":
        config_path = Path(path)
        if not config_path.is_file():
            pkg_path = Path(__file__).resolve().parent / path
            if pkg_path.is_file():
                config_path = pkg_path
            else:
                pkg_fallback = Path(__file__).resolve().parent / "train_config.json"
                if pkg_fallback.is_file():
                    config_path = pkg_fallback
        logger.info(f"Loading multimodal training configuration from JSON: '{config_path}'")
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


MultimodalTrainConfig = TrainConfig
