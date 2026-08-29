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
VALID_CACHE_MODES = ("disk", "ram", "none")


class VideoFeaturesConfig(BaseModel):
    """
    Single image extraction parameters.
    """
    image_size: int = Field(default=224, description="Target width and height for extracted images.")
    frame_policy: str = Field(default="center", description="Frame extraction policy: 'quarter', 'center', 'three_quarters', 'end', or 'random'.")


class ModelConfig(BaseModel):
    """
    Configuration for selecting the video backbone without editing main.py.
    """
    backbone: str = Field(
        default="MobileNetV2",
        description="Backbone class name exported by U_FFIA27K_video.models."
    )
    pretrained: bool = Field(
        default=True,
        description="Whether to request pretrained weights for supported torchvision backbones."
    )


class SplitterConfig(BaseModel):
    """
    Configuration parameters for dataset splitting.
    """
    dataset_path: str = Field(
        default='/marimo/Fish_Feeding_Intensity_Dataset',
        description="Absolute path to the raw dataset directory."
    )
    seed: int = Field(
        default=25,
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
        description="Whether to return video paths in RAM alongside audio paths."
    )
    split_strategy: str = Field(
        default="random_sample",
        description="Dataset split strategy: 'random_sample' preserves the legacy seeded per-class random split; 'time_series' sorts each class by time while keeping the same per-class val/test quotas; 'group_random' splits whole date_session groups to prevent same-session leakage."
    )
    evaluation_mode: str = Field(
        default="holdout",
        description="Evaluation mode: 'holdout' uses one train/val/test split; 'cross_validation' runs stratified K-fold CV."
    )
    num_folds: int = Field(
        default=5,
        gt=1,
        description="Number of outer folds for evaluation_mode='cross_validation'."
    )
    fold_index: Optional[int] = Field(
        default=None,
        description="Current outer fold index for cross-validation. Set automatically by main.py during CV runs."
    )
    cv_val_ratio: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="Stratified validation ratio split from the non-test development folds during cross-validation."
    )


class VideoTrainConfig(BaseModel):
    """
    Single unified Video training configuration class matching the audio side structure.
    Complies with OOP design via Pydantic.
    """
    epochs: int = Field(default=500, description="Maximum training epochs.")
    batch_size: int = Field(default=50, description="Mini-batch size.")
    dataloader_workers: int = Field(default=-1, ge=-1, description="Number of PyTorch DataLoader workers used during training. Use -1 for automatic CPU-based selection.")
    prefetch_factor: Optional[int] = Field(default=None, description="Number of batches prefetched by each DataLoader worker. None uses PyTorch default.")
    learning_rate: float = Field(default=1e-3, description="Optimizer learning rate.")
    ckpt_dir: str = Field(default='checkpoint/', description="Directory to save checkpoints and CSV logs.")
    monitor: str = Field(default='accuracy', description="Metric to monitor for best model saving ('accuracy' or 'loss').")
    early_stopping: bool = Field(default=True, description="Enable/disable early stopping mechanism.")
    patience: int = Field(default=30, description="Early stopping patience epochs.")
    delta: float = Field(default=0.0, description="Minimum change in monitored metric to qualify as improvement.")
    cache_mode: Literal["disk", "ram", "none"] = Field(default="disk", description="Image cache mode: 'disk' uses .pkl cache, 'ram' preloads decoded images into system RAM, 'none' decodes from MP4 on demand.")
    
    # Nested configurations
    model: ModelConfig = Field(default_factory=ModelConfig, description="Video backbone model configuration.")
    dataset_splitter: SplitterConfig = Field(default_factory=SplitterConfig, description="Dataset splitter configurations.")
    video_features: VideoFeaturesConfig = Field(default_factory=VideoFeaturesConfig, description="Image feature extraction configuration.")

    @classmethod
    def from_json(cls, path: str = 'config/train_config.json') -> 'VideoTrainConfig':
        """
        Load single unified video training configuration from a JSON file.
        """
        logger.info(f"Loading single unified video configuration from JSON: '{path}'")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


# Alias for compatibility matching U_FFIA_audio
TrainConfig = VideoTrainConfig
