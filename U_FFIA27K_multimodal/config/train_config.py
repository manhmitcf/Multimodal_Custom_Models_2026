import sys
import json
import logging
from pathlib import Path
from typing import Optional, Literal
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

VALID_CACHE_MODES = ("disk", "ram", "none")


class VideoFeaturesConfig(BaseModel):
    """
    Video frame extraction & spatiotemporal kinematics parameters.
    """
    image_size: int = Field(default=224, description="Target spatial resolution H=W.")
    num_frames: int = Field(default=2, description="Number of uniform frames sampled per video clip (T=2).")
    num_channels: int = Field(default=7, description="7-channel representation (RGB + 4 Kinematics: u, v, vorticity, decel).")


class AudioFeaturesConfig(BaseModel):
    """
    High-Resolution TKEO-STFT Audio Frontend parameters (256 kHz, 2049 linear bins).
    """
    sample_rate: int = Field(default=256000, description="Audio sampling rate in Hz.")
    window_size: int = Field(default=4096, description="STFT window size in samples.")
    hop_size: int = Field(default=2048, description="STFT hop size in samples.")
    mel_bins: int = Field(default=2049, description="Number of STFT frequency bins.")
    fmin: int = Field(default=0, description="Minimum frequency for STFT in Hz.")
    fmax: int = Field(default=128000, description="Maximum frequency for STFT in Hz.")
    use_tkeo: bool = Field(default=True, description="Enable Teager-Kaiser Energy Operator Adaptive Pre-Emphasis.")
    alpha_max: float = Field(default=0.99, description="Max pre-emphasis coefficient for TKEO APE.")
    beta: float = Field(default=0.8, description="Temporal smoothing factor for TKEO APE.")
    use_spectral_aug: bool = Field(default=True, description="Enable 1D Spectral Augmentation (Cutout & Jitter) for STFT.")
    cutout_width: int = Field(default=24, description="Width of contiguous frequency cutout mask in bins.")
    cutout_prob: float = Field(default=0.5, description="Probability of applying frequency cutout per sample.")
    noise_std: float = Field(default=0.02, description="Standard deviation of Gaussian spectral jitter noise.")


class MatchupTieBreakersConfig(BaseModel):
    """
    Configuration for Cross-Modal Referees (Audio STFT & Video Kinematics) on a single pairwise matchup.
    """
    enable_audio: bool = Field(default=True, description="Enable Audio STFT Tie-Breaker for this matchup.")
    enable_video: bool = Field(default=True, description="Enable Video Kinematics Tie-Breaker for this matchup.")


class DualTieBreakersConfig(BaseModel):
    """
    Configuration for Cross-Modal Referee heads on Level 2 pairwise matchups (B12, B23, B13).
    """
    b12: MatchupTieBreakersConfig = Field(
        default_factory=MatchupTieBreakersConfig,
        description="Referees (Audio & Video) for Weak vs Medium (B12)."
    )
    b23: MatchupTieBreakersConfig = Field(
        default_factory=MatchupTieBreakersConfig,
        description="Referees (Audio & Video) for Medium vs Strong (B23)."
    )
    b13: MatchupTieBreakersConfig = Field(
        default_factory=MatchupTieBreakersConfig,
        description="Referees (Audio & Video) for Weak vs Strong (B13)."
    )


class ModelConfig(BaseModel):
    """
    Configuration for Multimodal Tournament Model with Sparse Mixture-of-Referees (SMoR-Net, ~4.21M parameters).
    Video: ConvNeXt-Nano (7-ch Kinematics) ~2.701M
    Audio: TKEO-STFT-MLP (2049 bins @ 256 kHz) ~1.166M
    Audio Frontend: TKEO-STFT LayerNorm ~0.004M
    Fusion: Pairwise Boundary Tournament Decision Head with SMoR Dynamic Routing ~0.339M
    Auxiliary Heads: Video + Audio Aux Heads ~0.002M
    """
    backbone: str = Field(
        default="MultimodalSOTANet",
        description="Model architecture name exported by U_FFIA27K_multimodal.models."
    )
    embed_dim: int = Field(default=224, description="Common multimodal embedding dimension.")
    classes_num: int = Field(default=4, description="Number of output feeding intensity classes (None, Strong, Medium, Weak).")
    tie_breakers: DualTieBreakersConfig = Field(
        default_factory=DualTieBreakersConfig,
        description="Pairwise Cross-Modal Referee configurations for B12, B23, B13."
    )
    use_sparse_moe_routing: bool = Field(
        default=True,
        description="Enable Sparse Mixture-of-Referees (SMoR) dynamic routing."
    )
    router_hidden_dim: int = Field(
        default=32,
        gt=0,
        description="Hidden dimension for Sparse Referee Routers."
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
    weight_decay: float = Field(default=0.05, ge=0, description="Weight decay factor for AdamW.")
    ckpt_dir: str = Field(default="checkpoint/", description="Directory path to save checkpoints.")
    monitor: str = Field(default="val_acc", description="Metric to monitor for best checkpoint: 'val_acc' (or 'accuracy'), 'qwk', or 'both' (dual-track).")
    mode: Literal["min", "max"] = Field(default="max", description="Optimization direction for monitored metric.")
    early_stopping: bool = Field(default=False, description="Enable early stopping mechanism.")
    patience: int = Field(default=100, ge=1, description="Early stopping patience in epochs.")
    min_delta: float = Field(default=0.0, ge=0.0, description="Minimum change threshold in monitored metric.")
    use_onecycle: bool = Field(default=True, description="Enable OneCycleLR scheduler.")
    seed: int = Field(default=42, ge=0, description="Master random seed.")
    cache_mode: str = Field(default="ram", description="Caching mode: 'ram', 'disk', or 'none'.")
    dataloader_workers: int = Field(default=8, description="Number of worker processes for DataLoader (Fixed to 8).")
    prefetch_factor: Optional[int] = Field(default=2, description="Number of batches loaded in advance.")
    save_best_only: bool = Field(default=True, description="Save only the best checkpoint.")
    loss_type: str = Field(default="pairwise_tournament", description="Loss function: 'pairwise_tournament' or 'clip_ce'.")
    weight_act: float = Field(default=0.5, ge=0.0, description="Weight for Level-1 Activity Gate BCE loss.")
    weight_pairwise: float = Field(default=0.5, ge=0.0, description="Weight for Level-2 Pairwise Boundaries loss.")
    weight_ce: float = Field(default=1.0, ge=0.0, description="Weight for Multi-class CE on Tournament Logits.")
    aux_loss_weight: float = Field(default=0.3, ge=0.0, description="Weight for auxiliary unimodal backbone heads.")
    use_sparse_moe_routing: bool = Field(default=True, description="Enable Sparse Mixture-of-Referees (SMoR) dynamic routing.")
    router_hidden_dim: int = Field(default=32, gt=0, description="Hidden dimension for Sparse Referee Routers.")
    lambda_balance: float = Field(default=0.01, ge=0.0, description="Weight for Switch Transformer MoE load balancing loss.")
    lambda_sparse: float = Field(default=0.005, ge=0.0, description="Weight for MoE sparsity regularization penalty.")
    model: ModelConfig = Field(default_factory=ModelConfig, description="Model architecture parameters.")
    dataset_splitter: SplitterConfig = Field(default_factory=SplitterConfig, description="Dataset splitting settings.")
    video_features: VideoFeaturesConfig = Field(default_factory=VideoFeaturesConfig, description="Video preprocessing configuration.")
    audio_features: AudioFeaturesConfig = Field(default_factory=AudioFeaturesConfig, description="Audio preprocessing configuration.")

    @property
    def num_frames(self) -> int:
        return self.video_features.num_frames

    @num_frames.setter
    def num_frames(self, val: int) -> None:
        self.video_features.num_frames = val

    @property
    def image_size(self) -> int:
        return self.video_features.image_size

    @image_size.setter
    def image_size(self, val: int) -> None:
        self.video_features.image_size = val

    @property
    def sample_rate(self) -> int:
        return self.audio_features.sample_rate

    @sample_rate.setter
    def sample_rate(self, val: int) -> None:
        self.audio_features.sample_rate = val

    @property
    def in_chans(self) -> int:
        return self.video_features.num_channels

    @in_chans.setter
    def in_chans(self, val: int) -> None:
        self.video_features.num_channels = val

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
