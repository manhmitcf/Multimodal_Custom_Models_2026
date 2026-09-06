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
    Video frame extraction parameters.
    """
    image_size: int = Field(default=224, description="Target width and height for extracted video frames.")
    frame_policy: str = Field(default="end", description="Frame extraction policy: 'quarter', 'center', 'three_quarters', 'end', or 'random'.")
    num_frames: int = Field(default=2, description="Number of consecutive or sampled frames per video clip.")


class AudioFeaturesConfig(BaseModel):
    """
    Audio Log-Mel Spectrogram extraction parameters.
    """
    sample_rate: int = Field(default=64000, description="Target audio sampling rate in Hz.")
    window_size: int = Field(default=2048, description="STFT window length in samples.")
    hop_size: int = Field(default=512, description="STFT hop size in samples.")
    mel_bins: int = Field(default=128, description="Number of Mel frequency filter banks.")
    fmin: int = Field(default=50, description="Minimum frequency for Mel filter bank in Hz.")
    fmax: int = Field(default=32000, description="Maximum frequency for Mel filter bank in Hz.")
    use_tkeo: bool = Field(default=True, description="Enable Teager-Kaiser Energy Operator (TKEO) adaptive pre-emphasis.")
    alpha_max: float = Field(default=0.99, ge=0.0, le=1.0, description="Maximum alpha scaling factor for TKEO filter.")
    beta: float = Field(default=0.8, ge=0.0, le=1.0, description="Smoothing momentum parameter across consecutive frames in TKEO.")
    time_drop_width: int = Field(default=64, description="Maximum time masking width for SpecAugmentation.")
    time_stripes_num: int = Field(default=2, description="Number of masked time stripes for SpecAugmentation.")
    freq_drop_width: int = Field(default=8, description="Maximum frequency masking width for SpecAugmentation.")
    freq_stripes_num: int = Field(default=2, description="Number of masked frequency stripes for SpecAugmentation.")




class ModelConfig(BaseModel):
    """
    Configuration for selecting and tuning the multimodal model.
    """
    backbone: str = Field(
        default="LiteFFIANet",
        description="Model architecture name exported by U_FFIA27K_multimodal.models."
    )
    embed_dim: int = Field(default=192, description="Common embedding dimension for visual and acoustic tokens.")
    num_bottlenecks: int = Field(default=4, description="Number of MBT bottleneck tokens.")
    num_heads: int = Field(default=4, description="Number of cross-attention heads.")
    pretrained_video: bool = Field(default=True, description="Whether to load ImageNet pretrained weights for the video backbone.")
    classes_num: int = Field(default=4, description="Number of output classification classes.")


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
    batch_size: int = Field(default=256, gt=0, description="Training batch size.")
    learning_rate: float = Field(default=1e-4, gt=0, description="Initial learning rate.")
    weight_decay: float = Field(default=1e-4, ge=0, description="Weight decay factor for optimizer.")
    ckpt_dir: str = Field(default="checkpoint/", description="Directory path to save checkpoints.")
    monitor: Literal["loss", "accuracy"] = Field(default="loss", description="Metric to monitor for early stopping and best checkpoint.")
    early_stopping: bool = Field(default=True, description="Enable early stopping mechanism.")
    patience: int = Field(default=40, ge=1, description="Early stopping patience.")
    cache_mode: str = Field(default="ram", description="Caching mode: 'ram', 'disk', or 'none'.")
    dataloader_workers: int = Field(default=1, ge=0, description="Number of background worker processes for DataLoader.")
    prefetch_factor: Optional[int] = Field(default=1, ge=1, description="Number of batches loaded in advance by each worker.")
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


# Alias for naming consistency
MultimodalTrainConfig = TrainConfig
