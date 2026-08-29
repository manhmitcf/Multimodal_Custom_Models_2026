import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import logging
from typing import Optional
from pydantic import BaseModel, Field

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioFeaturesConfig(BaseModel):
    """
    Detailed configuration parameters for GPU Mel-spectrogram feature extraction.
    """
    sample_rate: int = Field(default=64000, description="Sample rate of raw audio waveforms.")
    window_size: int = Field(default=2048, description="FFT window size (n_fft).")
    hop_size: int = Field(default=1024, description="Hop length between FFT windows.")
    mel_bins: int = Field(default=128, description="Number of Mel frequency bins (n_mels).")
    fmin: int = Field(default=1, description="Minimum filter frequency.")
    fmax: int = Field(default=32000, description="Maximum filter frequency.")
    time_drop_width: int = Field(default=64, description="Maximum time masking width.")
    time_stripes_num: int = Field(default=2, description="Number of masked time stripes.")
    freq_drop_width: int = Field(default=8, description="Maximum frequency masking width.")
    freq_stripes_num: int = Field(default=2, description="Number of masked frequency stripes.")


class ModelConfig(BaseModel):
    """
    Configuration for selecting the audio backbone without editing main.py.
    """
    backbone: str = Field(
        default="Cnn14MobileV2",
        description="Backbone class name exported by U_FFIA27K_audio.models."
    )
    pretrained: bool = Field(
        default=False,
        description="Whether to request pretrained weights for backbones that support them."
    )


class SplitterConfig(BaseModel):
    """
    Configuration parameters for dataset splitting.
    """
    dataset_path: str = Field(
        default='/marimo/Fish_Feeding_Intensity_Dataset',
        description="Absolute path to the raw U_FFIA27K dataset directory."
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


class TrainConfig(BaseModel):
    """
    Unified configuration class for training hyperparameters and model structure.
    Complies with OOP design via Pydantic.
    """
    epochs: int = Field(default=100, description="Maximum training epochs.")
    batch_size: int = Field(default=40, description="Mini-batch size.")
    learning_rate: float = Field(default=1e-3, description="Optimizer learning rate.")
    ckpt_dir: str = Field(default='checkpoint/', description="Directory to save checkpoints and CSV logs.")
    monitor: str = Field(default='accuracy', description="Metric to monitor for best model saving ('accuracy' or 'loss').")
    early_stopping: bool = Field(default=True, description="Enable/disable early stopping mechanism.")
    patience: int = Field(default=30, description="Early stopping patience epochs.")
    delta: float = Field(default=0.0, description="Minimum change in monitored metric to qualify as improvement.")
    cache_audio: bool = Field(default=True, description="Preload entire audio dataset to RAM cache.")
    
    # Nested configurations
    model: ModelConfig = Field(default_factory=ModelConfig, description="Audio backbone model configuration.")
    dataset_splitter: SplitterConfig = Field(default_factory=SplitterConfig, description="Dataset splitter configurations.")
    audio_features: AudioFeaturesConfig = Field(default_factory=AudioFeaturesConfig, description="Mel-spectrogram extractor configuration.")

    @classmethod
    def from_json(cls, path: str = 'config/train_config.json') -> 'TrainConfig':
        """
        Load unified training configuration from a JSON file.
        """
        logger.info(f"Loading unified training configuration from JSON: '{path}'")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)
