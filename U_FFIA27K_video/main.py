import os
import sys
import copy
import csv
import json
from pathlib import Path
from datetime import datetime
import zipfile

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import torch
import torch.optim as optim

# Import refactored OOP components
from config import ArtifactUploadConfig, TrainConfig
from dataset import FishVideoDataLoader
from models import (
    ConvNeXtTiny,
    DenseNet121,
    EfficientNetB0,
    MobileNetV2,
    ResNet18,
    ResNet50,
    SwinTiny,
    VideoModel,
    ViTBase16,
    MobileViTXXS,
    MobileViTXS,
    MobileViTv2_050,
    MobileViTv2_075,
)
from tasks import VideoTrainer

# Ensure stdout/stderr UTF-8 encoding on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


MODEL_REGISTRY = {
    "MobileNetV2": MobileNetV2,
    "EfficientNetB0": EfficientNetB0,
    "ResNet18": ResNet18,
    "ResNet50": ResNet50,
    "DenseNet121": DenseNet121,
    "SwinTiny": SwinTiny,
    "ViTBase16": ViTBase16,
    "ConvNeXtTiny": ConvNeXtTiny,
    "MobileViTXXS": MobileViTXXS,
    "MobileViTXS": MobileViTXS,
    "MobileViTv2_050": MobileViTv2_050,
    "MobileViTv2_075": MobileViTv2_075,
}


def validate_backbone_config(config: TrainConfig) -> None:
    backbone_name = config.model.backbone
    if backbone_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown video backbone '{backbone_name}'. Available backbones: {available}.")


def build_backbone(config: TrainConfig):
    validate_backbone_config(config)
    backbone_cls = MODEL_REGISTRY[config.model.backbone]
    return backbone_cls(classes_num=4, pretrained=config.model.pretrained)


def model_cv_dir(base_ckpt_dir: str, model_name: str) -> str:
    base_dir = base_ckpt_dir if base_ckpt_dir else "checkpoint"
    if Path(base_dir).name == model_name:
        return base_dir
    return os.path.join(base_dir, model_name)


def write_runtime_config(config: TrainConfig, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2)
    return str(path)


def read_single_summary_row(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Summary file is empty: {summary_path}")
    return rows[-1]


def aggregate_cv_summaries(model_dir: str, num_folds: int) -> None:
    import statistics

    output_dir = Path(model_dir)
    fold_rows = []
    for fold_index in range(num_folds):
        fold_dir = output_dir / f"fold_{fold_index:02d}"
        summary_path = fold_dir / "summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing fold summary: {summary_path}")
        row = read_single_summary_row(summary_path)
        row = {"fold": str(fold_index), **row}
        fold_rows.append(row)

    metric_columns = [key for key in fold_rows[0].keys() if key != "fold"]
    numeric_columns = []
    for key in metric_columns:
        try:
            [float(row[key]) for row in fold_rows]
            numeric_columns.append(key)
        except (TypeError, ValueError):
            pass

    mean_row = {"fold": "mean"}
    std_row = {"fold": "std"}
    for key in metric_columns:
        if key in numeric_columns:
            values = [float(row[key]) for row in fold_rows]
            mean_row[key] = f"{statistics.mean(values):.6f}"
            std_row[key] = f"{statistics.stdev(values) if len(values) > 1 else 0.0:.6f}"
        else:
            mean_row[key] = ""
            std_row[key] = ""

    csv_path = output_dir / "cv_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["fold"] + metric_columns
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fold_rows)
        writer.writerow(mean_row)
        writer.writerow(std_row)

    json_path = output_dir / "cv_summary.json"
    summary_json = {
        "num_folds": num_folds,
        "folds": fold_rows,
        "mean": mean_row,
        "std": std_row,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    logger.info(f"Saved cross-validation summary CSV to: '{csv_path}'")
    logger.info(f"Saved cross-validation summary JSON to: '{json_path}'")


def safe_filename_part(value: str) -> str:
    safe_chars = []
    for char in str(value):
        if char.isascii() and (char.isalnum() or char in ("-", "_", ".")):
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    safe_value = "".join(safe_chars).strip("._-")
    return safe_value or "unknown"


def build_artifact_filename(config: TrainConfig, timestamp: str, suffix: str = ".zip") -> str:
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    filename_parts = [
        config.model.backbone,
        config.dataset_splitter.evaluation_mode,
        config.dataset_splitter.split_strategy,
        config.video_features.frame_policy,
        timestamp,
    ]
    safe_stem = "_".join(safe_filename_part(part) for part in filename_parts)
    return f"{safe_stem}{suffix}"


def artifact_zip_path(zip_path: str, artifact_filename: str) -> Path:
    return Path(zip_path).with_name(artifact_filename)


def artifact_repo_path(path_in_repo: str, artifact_filename: str) -> str:
    path = Path(path_in_repo)
    parent = path.parent
    if str(parent) == ".":
        return artifact_filename
    return str(parent / artifact_filename).replace("\\", "/")


def zip_directory(source_dir: str, zip_path: str) -> Path:
    source_path = Path(source_dir).resolve()
    output_path = Path(zip_path).resolve()

    if not source_path.is_dir():
        raise FileNotFoundError(f"Artifact source_dir does not exist or is not a directory: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    logger.info("==================================================")
    logger.info(f"Creating artifact zip from: '{source_path}'")
    logger.info(f"Artifact zip path:          '{output_path}'")
    logger.info("==================================================")

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in source_path.rglob("*"):
            if not file_path.is_file():
                continue
            resolved_file = file_path.resolve()
            if resolved_file == output_path:
                continue
            zip_file.write(resolved_file, arcname=resolved_file.relative_to(source_path))

    logger.info(f"Successfully created artifact zip: '{output_path}'")
    return output_path


def upload_artifact_if_enabled(upload_config: ArtifactUploadConfig, config: TrainConfig) -> None:
    if not upload_config.enabled:
        logger.info("Artifact upload is disabled. Skipping Hugging Face upload.")
        return

    if not upload_config.repo_id:
        raise ValueError("artifact_upload.repo_id must be set when artifact_upload.enabled is true.")

    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except ImportError:
            token = None
    if not token:
        raise EnvironmentError("HF_TOKEN environment variable must be set or Hugging Face CLI login must be completed when artifact upload is enabled.")

    try:
        from huggingface_hub import create_repo, upload_file
    except ImportError as exc:
        raise ImportError("huggingface_hub is required for artifact upload. Install it with requirements.txt.") from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = Path(upload_config.path_in_repo).suffix or ".zip"
    artifact_filename = build_artifact_filename(config, timestamp, suffix=suffix)
    output_zip_path = artifact_zip_path(upload_config.zip_path, artifact_filename)
    artifact_zip = zip_directory(upload_config.source_dir, str(output_zip_path))

    if upload_config.create_repo:
        create_repo(
            repo_id=upload_config.repo_id,
            repo_type=upload_config.repo_type,
            token=token,
            exist_ok=True,
        )

    logger.info("==================================================")
    logger.info(f"Uploading artifact to Hugging Face repo: '{upload_config.repo_id}'")
    logger.info(f"Repo type:                            '{upload_config.repo_type}'")
    path_in_repo = artifact_repo_path(upload_config.path_in_repo, artifact_filename)
    logger.info(f"Path in repo:                         '{path_in_repo}'")
    logger.info("==================================================")

    upload_file(
        path_or_fileobj=str(artifact_zip),
        path_in_repo=path_in_repo,
        repo_id=upload_config.repo_id,
        repo_type=upload_config.repo_type,
        token=token,
    )
    logger.info("Successfully uploaded artifact to Hugging Face.")


def run_video_training(config: TrainConfig, train_config_path: str, device: torch.device, fold_index: int = None) -> str:
    """Run one video train/val/test cycle and return the model output directory."""
    logger.info("Initializing DataLoaders...")
    loader_manager = FishVideoDataLoader(
        batch_size=config.batch_size,
        dataloader_workers=config.dataloader_workers,
        prefetch_factor=config.prefetch_factor,
        cache_mode=config.cache_mode,
        image_size=config.video_features.image_size,
        frame_policy=config.video_features.frame_policy,
        splitter_config=config.dataset_splitter
    )

    train_loader = loader_manager.get_dataloader(split='train', shuffle=True, drop_last=False)
    val_loader = loader_manager.get_dataloader(split='val', shuffle=False)
    test_loader = loader_manager.get_dataloader(split='test', shuffle=False)

    # 4. Construct unified VideoModel
    logger.info("Assembling neural network model layers...")
    backbone = build_backbone(config)
    model = VideoModel(backbone=backbone)
    model = model.to(device)

    model_name = backbone.get_name() if hasattr(backbone, "get_name") else backbone.__class__.__name__
    if config.dataset_splitter.evaluation_mode == "cross_validation":
        fold_dir = os.path.join(model_cv_dir(config.ckpt_dir, model_name), f"fold_{fold_index:02d}")
        config.ckpt_dir = fold_dir
        train_config_path = write_runtime_config(config, os.path.join(fold_dir, "fold_train_config.json"))

    # 5. Initialize Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 6. Instantiate VideoTrainer
    trainer = VideoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        optimizer=optimizer,
        train_config_path=train_config_path
    )

    # 7. Start Training & Evaluation process
    trainer.fit()

    if config.dataset_splitter.evaluation_mode == "cross_validation":
        return config.ckpt_dir
    return os.path.join(config.ckpt_dir if config.ckpt_dir else "checkpoint", model_name)


def main():
    # 1. Main configuration file path (override this when running other configs)
    train_config_path = 'config/train_config.json'
    artifact_upload_config_path = 'config/artifact_upload_config.json'

    # Load unified training configurations
    config = TrainConfig.from_json(train_config_path)
    validate_backbone_config(config)

    logger.info("==================================================")
    logger.info("Launching Single-Frame Image Classification Training:")
    logger.info(f"  - Device Name:              {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"  - Max Epochs:               {config.epochs}")
    logger.info(f"  - Batch Size:               {config.batch_size}")
    logger.info(f"  - Learning Rate (LR):       {config.learning_rate}")
    logger.info(f"  - Checkpoint Directory:     '{config.ckpt_dir}'")
    logger.info(f"  - Monitor Metric:           '{config.monitor}'")
    logger.info(f"  - Backbone Model:           '{config.model.backbone}'")
    logger.info(f"  - Pretrained Backbone:      {config.model.pretrained}")
    logger.info(f"  - Evaluation Mode:          '{config.dataset_splitter.evaluation_mode}'")
    logger.info(f"  - Split Strategy:           '{config.dataset_splitter.split_strategy}'")
    logger.info("==================================================")

    # 2. Hardware device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training will run on device: '{device}'")

    if config.dataset_splitter.evaluation_mode == "cross_validation":
        model_dir = None
        for fold_index in range(config.dataset_splitter.num_folds):
            logger.info("==================================================")
            logger.info(f"Starting cross-validation fold {fold_index + 1}/{config.dataset_splitter.num_folds}")
            logger.info("==================================================")
            fold_config = copy.deepcopy(config)
            fold_config.dataset_splitter.fold_index = fold_index
            fold_output_dir = run_video_training(
                config=fold_config,
                train_config_path=train_config_path,
                device=device,
                fold_index=fold_index,
            )
            model_dir = str(Path(fold_output_dir).parent)

        if model_dir is not None:
            aggregate_cv_summaries(model_dir, config.dataset_splitter.num_folds)
    else:
        run_video_training(
            config=config,
            train_config_path=train_config_path,
            device=device,
            fold_index=None,
        )

    artifact_upload_config = ArtifactUploadConfig.from_json(artifact_upload_config_path)
    upload_artifact_if_enabled(artifact_upload_config, config)

    logger.info("Training pipeline finished successfully!")


if __name__ == '__main__':
    main()
