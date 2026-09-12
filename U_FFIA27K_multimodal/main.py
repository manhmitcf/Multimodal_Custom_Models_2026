import os
import sys
import copy
import csv
import json
import re
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
from models import MultimodalBoundaryAwareNet, MultimodalSOTANet
from tasks import MultimodalTrainer
from utils.profile_model import count_parameters

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
    "MultimodalBoundaryAwareNet": MultimodalBoundaryAwareNet,
    "MultimodalSOTANet": MultimodalSOTANet,
}


def validate_model_config(config: TrainConfig) -> None:
    backbone_name = config.model.backbone
    if backbone_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown multimodal model '{backbone_name}'. Available: {available}.")


def build_model(config: TrainConfig) -> torch.nn.Module:
    validate_model_config(config)
    model_cls = MODEL_REGISTRY[config.model.backbone]
    from features.audio_frontend import AudioFrontend
    frontend = AudioFrontend(config.audio_features)

    return model_cls(
        classes_num=config.model.classes_num,
        embed_dim=config.model.embed_dim,
        audio_frontend=frontend,
        image_size=config.image_size,
        num_frames=config.num_frames,
        in_chans=getattr(config.video_features, "num_channels", 7),
    )



def verify_model_dry_run(config: TrainConfig, device: torch.device, config_path: Optional[str] = None) -> None:
    """
    Run the unified pre-flight test suite (test.py) on the configured model & parameters
    BEFORE loading the dataset into RAM or starting training.
    """
    from test import run_all_tests
    run_all_tests(config_path=config_path, config=config, device=device, verbose=True)


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
            # Exclude target zip, git history, zip archives, temporary files, and local-only secrets
            if (
                resolved_file == target_path
                or ".git" in resolved_file.parts
                or resolved_file.suffix in (".zip", ".tmp")
                or "cli.txt" in resolved_file.name
                or "run_marimo.txt" in resolved_file.name
                or "local_cli" in resolved_file.parts
                or "AGENTS.md" in resolved_file.name
            ):
                continue
            zip_file.write(resolved_file, arcname=resolved_file.relative_to(source_path))

    logger.info(f"Successfully created artifact zip: '{target_path}'")
    return str(target_path)


def discover_hf_token() -> Optional[str]:
    """Find Hugging Face token across environment variables and setup scripts."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()

    try:
        from huggingface_hub import get_token
        hub_token = get_token()
        if hub_token:
            return hub_token.strip()
    except Exception:
        pass

    # Search for token strings in setup files
    token_files = [
        Path("cli.txt"),
        Path("/marimo/cli.txt"),
        Path("run_marimo.txt"),
        Path("/marimo/run_marimo.txt"),
        Path(__file__).resolve().parent.parent / "cli.txt",
        Path(__file__).resolve().parent.parent / "run_marimo.txt"
    ]
    for tf in token_files:
        if tf.is_file():
            try:
                content = tf.read_text(encoding="utf-8")
                match = re.search(r'HF_TOKEN\s*=\s*["\'](hf_[A-Za-z0-9]+)["\']', content)
                if match:
                    return match.group(1).strip()
            except Exception:
                pass
    return None


def upload_artifact_if_enabled(upload_config: ArtifactUploadConfig, config: TrainConfig) -> None:
    if not upload_config.enabled:
        logger.info("Hugging Face upload is disabled. Skipping upload step.")
        return

    if not upload_config.repo_id:
        raise ValueError("artifact_upload.repo_id must be set when upload is enabled.")

    token = discover_hf_token()
    if not token:
        logger.warning("HF_TOKEN not found. Skipping Hugging Face upload.")
        return

    try:
        from huggingface_hub import create_repo, upload_file
    except ImportError as exc:
        raise ImportError("huggingface_hub is required for artifact upload. Install via pip.") from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_filename = build_artifact_filename(config, timestamp)
    source_dir = upload_config.source_dir
    if not Path(source_dir).exists():
        # Fallback to current project root
        source_dir = str(Path(__file__).resolve().parent.parent)

    try:
        if upload_config.zip_path and Path(upload_config.zip_path).parent.exists():
            zip_parent = Path(upload_config.zip_path).parent
        else:
            zip_parent = Path(source_dir).parent
        zip_parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        zip_parent = Path(source_dir)

    output_zip_path = str(zip_parent / artifact_filename)
    artifact_zip = zip_directory(source_dir, output_zip_path)

    if upload_config.create_repo:
        try:
            create_repo(
                repo_id=upload_config.repo_id,
                repo_type=upload_config.repo_type,
                token=token,
                exist_ok=True,
            )
        except Exception as exc:
            logger.warning(f"Could not create or verify HF repository '{upload_config.repo_id}': {exc}")

    logger.info("==================================================")
    logger.info(f"Uploading artifact to Hugging Face repo: '{upload_config.repo_id}'")
    logger.info(f"Repo type:                            '{upload_config.repo_type}'")
    logger.info("==================================================")

    try:
        upload_file(
            path_or_fileobj=artifact_zip,
            path_in_repo=artifact_filename,
            repo_id=upload_config.repo_id,
            repo_type=upload_config.repo_type,
            token=token,
        )
        logger.info("Artifact upload to Hugging Face completed successfully!")
    except Exception as exc:
        logger.error(f"Hugging Face upload encountered an issue: {exc}")
        logger.info(f"All training artifacts and checkpoints remain 100% safely preserved locally at '{source_dir}'.")


def run_training_session(
    train_config_path: Optional[str] = None,
    artifact_upload_config_path: Optional[str] = None,
    dry_run: bool = False,
    device_str: Optional[str] = None,
    lr_scheduler: Optional[str] = None,
) -> None:
    pkg_dir = Path(__file__).resolve().parent
    if train_config_path is None:
        train_config_path = str(pkg_dir / "config" / "train_config.json")
    if artifact_upload_config_path is None:
        artifact_upload_config_path = str(pkg_dir / "config" / "artifact_upload_config.json")

    config = TrainConfig.from_json(train_config_path)
    if lr_scheduler is not None:
        config.lr_scheduler = lr_scheduler

    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")

    eval_mode = config.dataset_splitter.evaluation_mode
    model_name = config.model.backbone

    # =========================================================================
    # UNIFIED PRE-FLIGHT TEST SUITE (Run test.py before preloading RAM)
    # =========================================================================
    verify_model_dry_run(config, device, config_path=train_config_path)

    if dry_run:
        logger.info(">>> Dry-run flag detected: All 5 pre-flight checks passed. Exiting without training.")
        return

    if eval_mode == "cross_validation":
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        num_folds = config.dataset_splitter.num_folds
        logger.info(f"Starting {num_folds}-Fold Cross-Validation for {model_name}...")
        base_dir = Path(model_cv_dir(config.ckpt_dir, model_name))

        for fold_idx in range(num_folds):
            logger.info(f"===== RUNNING FOLD {fold_idx + 1}/{num_folds} =====")
            fold_config = copy.deepcopy(config)
            fold_config.dataset_splitter.fold_index = fold_idx

            data_loader = FishMultimodalDataLoader(
                batch_size=fold_config.batch_size,
                dataloader_workers=fold_config.dataloader_workers,
                prefetch_factor=getattr(fold_config, "prefetch_factor", 2),
                cache_mode=fold_config.cache_mode,
                image_size=fold_config.image_size,
                num_frames=fold_config.num_frames,
                sample_rate=fold_config.sample_rate,
                splitter_config=fold_config.dataset_splitter,
            )

            model = build_model(fold_config).to(device)
            stats = count_parameters(model)
            logger.info(f"Fold {fold_idx} Model Parameters: {stats['total']:,} ({stats['total_million']:.3f} M)")

            trainer = MultimodalTrainer(
                model=model,
                train_loader=data_loader.get_data_loader(split='train'),
                val_loader=data_loader.get_data_loader(split='val'),
                test_loader=data_loader.get_data_loader(split='test'),
                config=fold_config,
                device=device,
                train_config_path=train_config_path
            )
            trainer.train()

        generate_cv_summary_report(base_dir, num_folds)
    else:
        logger.info(f"Starting Holdout Training for {model_name} (Epochs={config.epochs}, Patience={config.patience})...")
        data_loader = FishMultimodalDataLoader(
            batch_size=config.batch_size,
            dataloader_workers=config.dataloader_workers,
            prefetch_factor=getattr(config, "prefetch_factor", 2),
            cache_mode=config.cache_mode,
            image_size=config.image_size,
            num_frames=config.num_frames,
            sample_rate=config.sample_rate,
            splitter_config=config.dataset_splitter,
        )

        model = build_model(config).to(device)

        trainer = MultimodalTrainer(
            model=model,
            train_loader=data_loader.get_data_loader(split='train'),
            val_loader=data_loader.get_data_loader(split='val'),
            test_loader=data_loader.get_data_loader(split='test'),
            config=config,
            device=device,
            train_config_path=train_config_path
        )
        trainer.train()

    # Upload artifacts to Hugging Face
    upload_config = ArtifactUploadConfig.from_json(artifact_upload_config_path)
    upload_artifact_if_enabled(upload_config, config)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Multimodal SOTA Fish Feeding Intensity Assessment")
    parser.add_argument("--config", type=str, default=None, help="Path to train_config.json")
    parser.add_argument("--upload-config", type=str, default=None, help="Path to artifact_upload_config.json")
    parser.add_argument("--device", type=str, default=None, help="Target compute device (cuda or cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Run pre-flight check only without training")
    parser.add_argument("--lr-scheduler", type=str, default=None, choices=["cosine", "onecycle"], help="LR scheduler strategy ('cosine' or 'onecycle')")
    args = parser.parse_args()

    run_training_session(
        train_config_path=args.config,
        artifact_upload_config_path=args.upload_config,
        dry_run=args.dry_run,
        device_str=args.device,
        lr_scheduler=args.lr_scheduler,
    )


if __name__ == "__main__":
    main()

