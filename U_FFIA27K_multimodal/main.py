import os
import sys
import copy
import csv
import json
from pathlib import Path
from datetime import datetime
import zipfile
from typing import Optional

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import torch

from config import ArtifactUploadConfig, TrainConfig
from dataset import FishMultimodalDataLoader
from models import LiteFFIANet
from tasks import MultimodalTrainer

# Ensure stdout/stderr UTF-8 encoding on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "LiteFFIANet": LiteFFIANet,
}


def validate_model_config(config: TrainConfig) -> None:
    backbone_name = config.model.backbone
    if backbone_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown multimodal model '{backbone_name}'. Available: {available}.")


from features.audio_frontend import AudioFrontend


def build_model(config: TrainConfig) -> torch.nn.Module:
    validate_model_config(config)
    model_cls = MODEL_REGISTRY[config.model.backbone]
    frontend = AudioFrontend(config.audio_features)
    return model_cls(
        classes_num=config.model.classes_num,
        embed_dim=config.model.embed_dim,
        num_bottlenecks=config.model.num_bottlenecks,
        num_heads=config.model.num_heads,
        pretrained_video=config.model.pretrained_video,
        audio_frontend=frontend
    )


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
        raise ValueError(f"Summary file '{summary_path}' is empty.")
    return rows[-1]


def generate_cv_summary_report(model_dir: Path, num_folds: int) -> Path:
    fold_summaries = []
    for fold_idx in range(num_folds):
        fold_summary_path = model_dir / f"fold_{fold_idx}" / "summary.csv"
        if not fold_summary_path.is_file():
            raise FileNotFoundError(f"Missing summary file for fold {fold_idx}: '{fold_summary_path}'.")
        row = read_single_summary_row(fold_summary_path)
        row["Fold"] = str(fold_idx)
        fold_summaries.append(row)

    metric_keys = [k for k in fold_summaries[0].keys() if k != "Fold"]
    mean_row = {"Fold": "mean"}
    std_row = {"Fold": "std"}

    for key in metric_keys:
        values = []
        for summary in fold_summaries:
            try:
                values.append(float(summary[key]))
            except ValueError:
                pass
        if values:
            mean_row[key] = f"{float(sum(values) / len(values)):.6f}"
            variance = sum((x - float(mean_row[key])) ** 2 for x in values) / len(values)
            std_row[key] = f"{float(variance ** 0.5):.6f}"
        else:
            mean_row[key] = "N/A"
            std_row[key] = "N/A"

    cv_summary_path = model_dir / "cv_summary.csv"
    fieldnames = ["Fold"] + metric_keys
    with cv_summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in fold_summaries:
            writer.writerow(row)
        writer.writerow(mean_row)
        writer.writerow(std_row)

    logger.info(f"Cross-validation summary successfully saved to: '{cv_summary_path}'")
    return cv_summary_path


def build_artifact_filename(config: TrainConfig, timestamp: str, suffix: str = ".zip") -> str:
    model_name = config.model.backbone
    dataset_name = Path(config.dataset_splitter.dataset_path).name or "dataset"
    split_strategy = config.dataset_splitter.split_strategy
    eval_mode = config.dataset_splitter.evaluation_mode
    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return f"{model_name}_{dataset_name}_{split_strategy}_{eval_mode}_{timestamp}{safe_suffix}"


def zip_directory(source_dir: str, output_path: str) -> str:
    source_path = Path(source_dir).resolve()
    target_path = Path(output_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory to zip does not exist: '{source_path}'")

    logger.info(f"Creating artifact zip from '{source_path}' to '{target_path}'...")
    with zipfile.ZipFile(target_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in source_path.rglob("*"):
            if not file_path.is_file():
                continue
            resolved_file = file_path.resolve()
            if resolved_file == target_path:
                continue
            zip_file.write(resolved_file, arcname=resolved_file.relative_to(source_path))

    logger.info(f"Successfully created artifact zip: '{target_path}'")
    return str(target_path)


def upload_artifact_if_enabled(upload_config: ArtifactUploadConfig, config: TrainConfig) -> None:
    if not upload_config.enabled:
        logger.info("Hugging Face upload is disabled. Skipping upload step.")
        return

    if not upload_config.repo_id:
        raise ValueError("artifact_upload.repo_id must be set when upload is enabled.")

    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except ImportError:
            token = None
    if not token:
        logger.warning("HF_TOKEN environment variable not set. Skipping Hugging Face upload.")
        return

    try:
        from huggingface_hub import create_repo, upload_file
    except ImportError as exc:
        raise ImportError("huggingface_hub is required for artifact upload. Install via pip.") from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = Path(upload_config.path_in_repo).suffix or ".zip"
    artifact_filename = build_artifact_filename(config, timestamp, suffix=suffix)
    output_zip_path = os.path.join(upload_config.source_dir, artifact_filename)
    artifact_zip = zip_directory(upload_config.source_dir, output_zip_path)

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
    logger.info("==================================================")

    upload_file(
        path_or_fileobj=artifact_zip,
        path_in_repo=artifact_filename,
        repo_id=upload_config.repo_id,
        repo_type=upload_config.repo_type,
        token=token,
    )
    logger.info("Artifact upload to Hugging Face completed successfully!")


def run_training_session(
    train_config_path: Optional[str] = None,
    artifact_upload_config_path: Optional[str] = None,
) -> None:
    pkg_dir = Path(__file__).resolve().parent
    if train_config_path is None:
        train_config_path = str(pkg_dir / "config" / "train_config.json")
    if artifact_upload_config_path is None:
        artifact_upload_config_path = str(pkg_dir / "config" / "artifact_upload_config.json")

    config = TrainConfig.from_json(train_config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")

    eval_mode = config.dataset_splitter.evaluation_mode
    model_name = config.model.backbone

    if eval_mode == "cross_validation":
        num_folds = config.dataset_splitter.num_folds
        logger.info(f"Starting {num_folds}-Fold Cross-Validation for {model_name}...")
        base_dir = Path(model_cv_dir(config.ckpt_dir, model_name))

        for fold_idx in range(num_folds):
            logger.info(f"===== RUNNING FOLD {fold_idx + 1}/{num_folds} =====")
            fold_config = copy.deepcopy(config)
            fold_config.dataset_splitter.fold_index = fold_idx

            data_loader = FishMultimodalDataLoader(
                batch_size=fold_config.batch_size,
                dataloader_workers=0,
                image_size=fold_config.video_features.image_size,
                frame_policy=fold_config.video_features.frame_policy,
                num_frames=fold_config.video_features.num_frames,
                sample_rate=fold_config.audio_features.sample_rate,
                splitter_config=fold_config.dataset_splitter
            )

            train_loader = data_loader.get_data_loader("train", shuffle=True)
            val_loader = data_loader.get_data_loader("val", shuffle=False)
            test_loader = data_loader.get_data_loader("test", shuffle=False)

            model = build_model(fold_config)
            model.to(device)

            trainer = MultimodalTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                config=fold_config,
                device=device,
                train_config_path=train_config_path
            )
            trainer.train()

        generate_cv_summary_report(base_dir, num_folds)

    else:
        logger.info(f"Starting Holdout Training for {model_name}...")
        data_loader = FishMultimodalDataLoader(
            batch_size=config.batch_size,
            dataloader_workers=0,
            image_size=config.video_features.image_size,
            frame_policy=config.video_features.frame_policy,
            num_frames=config.video_features.num_frames,
            sample_rate=config.audio_features.sample_rate,
            splitter_config=config.dataset_splitter
        )

        train_loader = data_loader.get_data_loader("train", shuffle=True)
        val_loader = data_loader.get_data_loader("val", shuffle=False)
        test_loader = data_loader.get_data_loader("test", shuffle=False)

        model = build_model(config)
        model.to(device)

        trainer = MultimodalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=device,
            train_config_path=train_config_path
        )
        trainer.train()

    # Upload artifacts to Hugging Face if configured
    try:
        artifact_upload_config = ArtifactUploadConfig.from_json(artifact_upload_config_path)
        upload_artifact_if_enabled(artifact_upload_config, config)
    except Exception as exc:
        logger.warning(f"Could not complete Hugging Face upload: {exc}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="U-FFIA27K Multimodal (Video + Audio) Training CLI")
    parser.add_argument("--config", type=str, default=None, help="Path to train_config.json")
    parser.add_argument("--upload-config", type=str, default=None, help="Path to artifact_upload_config.json")
    args = parser.parse_args()

    run_training_session(
        train_config_path=args.config,
        artifact_upload_config_path=args.upload_config
    )
