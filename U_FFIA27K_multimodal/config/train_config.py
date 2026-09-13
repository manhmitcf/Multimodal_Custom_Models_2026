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
    num_frames: int = Field(default=2, description="Number of temporal frames sampled per video clip (T=2).")
    num_channels: int = Field(default=7, description="7-channel representation (RGB + 4 Kinematics: u, v, |V|, vorticity omega).")


class AudioFeaturesConfig(BaseModel):
    """
    High-Resolution TKEO-STFT Audio Frontend parameters (256 kHz, 2049 linear bins).
    """
    sample_rate: int = Field(default=256000, description="Audio sampling rate in Hz.")
    window_size: int = Field(default=4096, description="STFT window size in samples.")
    hop_size: int = Field(default=2048, description="STFT hop size in samples.")
    mel_bins: int = Field(default=2049, description="Number of STFT linear frequency bins (window_size // 2 + 1).")
    use_tkeo: bool = Field(default=True, description="Enable Teager-Kaiser Energy Operator Adaptive Pre-Emphasis.")
    alpha_max: float = Field(default=0.99, description="Max pre-emphasis coefficient for TKEO APE.")
    use_spectral_aug: bool = Field(default=False, description="Enable 1D Spectral Augmentation for MLP during training.")
    cutout_width: int = Field(default=24, description="Width of 1D frequency cutout band in linear bins.")
    cutout_prob: float = Field(default=0.5, ge=0.0, le=1.0, description="Probability of applying 1D frequency cutout.")
    noise_std: float = Field(default=0.02, ge=0.0, description="Standard deviation of additive Gaussian noise.")




class ModelConfig(BaseModel):
    """
    Configuration for SOTA Multimodal Model.
    Video: ConvNeXt-Nano (7-ch Kinematics)
    Audio: TKEO-STFT-MLP (2049 linear bins, 256 kHz)
    Fusion: Hierarchical Pairwise Cross-Boundary Tournament Engine
    """
    backbone: str = Field(
        default="MultimodalSOTANet",
        description="Model architecture name exported by U_FFIA27K_multimodal.models."
    )
    embed_dim: int = Field(default=224, description="Common multimodal embedding dimension.")
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
    learning_rate: float = Field(default=1e-3, gt=0, description="Initial learning rate.")
    weight_decay: float = Field(default=0.05, ge=0, description="Weight decay factor for AdamW applied to 2D/4D weights (biases and 1D normalization weights are excluded at weight_decay=0.0).")
    max_norm: float = Field(default=5.0, ge=0.1, le=50.0, description="Maximum gradient norm threshold for gradient clipping (torch.nn.utils.clip_grad_norm_).")
    seed: int = Field(default=42, ge=0, description="Master random seed for full reproducibility (PyTorch, NumPy, Python, CUDA, Dataset).")
    ckpt_dir: str = Field(default="checkpoint/", description="Directory path to save checkpoints.")
    monitor: str = Field(default="qwk", description="Metric to monitor for early stopping and best checkpoint: 'qwk' (default), 'val_acc', or 'loss'.")
    mode: Literal["min", "max"] = Field(default="max", description="Optimization direction for monitored metric.")
    early_stopping: bool = Field(default=False, description="Enable early stopping mechanism.")
    patience: int = Field(default=100, ge=1, description="Early stopping patience in epochs.")
    min_delta: float = Field(default=0.0, ge=0.0, description="Minimum change threshold in monitored metric.")
    lr_scheduler: str = Field(default="cosine", description="Learning rate scheduler: 'cosine' (CosineAnnealingLR with optional LinearLR warmup).")
    use_warmup: bool = Field(default=True, description="Enable LinearLR warmup before CosineAnnealingLR. If False, start immediately at learning_rate.")
    warmup_pct: float = Field(default=0.05, ge=0.0, le=1.0, description="Warmup percentage for LinearLR (default: 0.05 = 5% of total epochs).")
    min_lr: float = Field(default=1e-6, gt=0, description="Minimum learning rate.")

    num_frames: int = Field(default=2, ge=2, description="Number of frames per video input.")
    image_size: int = Field(default=224, gt=0, description="Video image spatial resolution.")
    sample_rate: int = Field(default=256000, gt=0, description="Audio sampling rate in Hz.")
    cache_mode: str = Field(default="ram", description="Caching mode: 'ram', 'disk', or 'none'.")
    dataloader_workers: int = Field(default=-1, description="Number of worker processes for DataLoader (-1 = auto).")
    prefetch_factor: Optional[int] = Field(default=2, description="Number of batches loaded in advance.")
    save_best_only: bool = Field(default=True, description="Save only the best checkpoint.")
    loss_type: str = Field(default="pairwise_tournament", description="Loss function: 'pairwise_tournament' or 'clip_ce'.")
    weight_act: float = Field(default=0.5, ge=0.0, description="Weight for Level-1 Activity Gate BCE loss.")
    weight_pairwise: float = Field(default=0.5, ge=0.0, description="Weight for Level-2 Pairwise Boundaries loss.")
    weight_ce: float = Field(default=1.0, ge=0.0, description="Weight for Multi-class CE on Tournament Logits.")
    aux_loss_weight: float = Field(default=0.3, ge=0.0, description="Weight for auxiliary unimodal backbone heads.")
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
