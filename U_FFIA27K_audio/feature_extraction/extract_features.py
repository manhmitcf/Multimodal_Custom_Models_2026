import csv
import concurrent.futures
import copy
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ArtifactUploadConfig, TrainConfig
from dataset import FishDataSplitter
from dataset.dataloader_melspectrogram import FishVoiceDataLoader
from features import AudioFrontend
from main import artifact_repo_path, artifact_zip_path, build_backbone, safe_filename_part, zip_directory
from models import AudioModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LABEL_TO_NAME = {0: "none", 1: "strong", 2: "medium", 3: "weak"}
SPLIT_ORDER = ("train", "val", "test")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_feature_config() -> Dict[str, Any]:
    config_path = PROJECT_ROOT / "config" / "feature_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing feature config: {config_path}")
    return load_json(config_path)


def load_artifact_upload_config(feature_cfg: Dict[str, Any]) -> ArtifactUploadConfig:
    config_path = PROJECT_ROOT / feature_cfg.get(
        "artifact_upload_config_path", "config/artifact_upload_config.json"
    )
    if not config_path.exists():
        logger.info(f"Artifact upload config not found. Skipping upload: {config_path}")
        return ArtifactUploadConfig(enabled=False)
    return ArtifactUploadConfig.from_json(str(config_path))


def build_feature_artifact_filename(config: TrainConfig, timestamp: str, suffix: str = ".zip") -> str:
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    filename_parts = [
        config.model.backbone,
        config.dataset_splitter.evaluation_mode,
        config.dataset_splitter.split_strategy,
        timestamp,
    ]
    safe_stem = "_".join(safe_filename_part(part) for part in filename_parts)
    return f"Features_{safe_stem}{suffix}"


def upload_features_artifact_if_enabled(upload_config: ArtifactUploadConfig, config: TrainConfig) -> None:
    if not upload_config.enabled:
        logger.info("Feature artifact upload is disabled. Skipping Hugging Face upload.")
        return

    if not upload_config.repo_id:
        raise ValueError("artifact_upload.repo_id must be set when artifact upload is enabled.")

    try:
        from huggingface_hub import create_repo, get_token, upload_file
    except ImportError as exc:
        raise ImportError("huggingface_hub is required for feature artifact upload. Install it with requirements.txt.") from exc

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        try:
            from huggingface_hub import HfFolder

            token = HfFolder.get_token()
        except Exception:
            token = None
    if token:
        logger.info("Hugging Face authentication token detected for artifact upload.")
    else:
        logger.info("No explicit Hugging Face token was detected; relying on the active Hugging Face CLI session.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = Path(upload_config.path_in_repo).suffix or ".zip"
    artifact_filename = build_feature_artifact_filename(config, timestamp, suffix=suffix)
    output_zip_path = artifact_zip_path(upload_config.zip_path, artifact_filename)
    artifact_zip = zip_directory(upload_config.source_dir, str(output_zip_path))

    if upload_config.create_repo:
        create_repo(
            repo_id=upload_config.repo_id,
            repo_type=upload_config.repo_type,
            token=token,
            exist_ok=True,
        )

    path_in_repo = artifact_repo_path(upload_config.path_in_repo, artifact_filename)
    logger.info("==================================================")
    logger.info(f"Uploading feature artifact to Hugging Face repo: '{upload_config.repo_id}'")
    logger.info(f"Repo type:                                    '{upload_config.repo_type}'")
    logger.info(f"Path in repo:                                 '{path_in_repo}'")
    logger.info("==================================================")

    upload_file(
        path_or_fileobj=str(artifact_zip),
        path_in_repo=path_in_repo,
        repo_id=upload_config.repo_id,
        repo_type=upload_config.repo_type,
        token=token,
    )
    logger.info("Successfully uploaded feature artifact to Hugging Face.")


def make_train_config(feature_cfg: Dict[str, Any], mode: str, fold_index: int | None = None) -> TrainConfig:
    train_config_path = PROJECT_ROOT / feature_cfg.get("train_config_path", "config/train_config.json")
    config = TrainConfig.from_json(str(train_config_path))
    model_cfg = feature_cfg.get("model", {})
    if model_cfg:
        config.model.backbone = model_cfg.get("backbone", config.model.backbone)
        if "pretrained" in model_cfg:
            config.model.pretrained = bool(model_cfg["pretrained"])
        if "classes_num" in model_cfg and hasattr(config.model, "classes_num"):
            config.model.classes_num = int(model_cfg["classes_num"])
    config.batch_size = int(feature_cfg.get("batch_size", config.batch_size))
    config.cache_audio = bool(feature_cfg.get("cache_audio", False))
    config.dataset_splitter.seed = int(feature_cfg.get("seed", config.dataset_splitter.seed))
    config.dataset_splitter.test_sample_per_class = int(
        feature_cfg.get("test_sample_per_class", config.dataset_splitter.test_sample_per_class)
    )
    config.dataset_splitter.split_strategy = feature_cfg.get(
        "split_strategy", config.dataset_splitter.split_strategy
    )
    config.dataset_splitter.num_folds = int(feature_cfg.get("num_folds", config.dataset_splitter.num_folds))
    config.dataset_splitter.cv_val_ratio = float(
        feature_cfg.get("cv_val_ratio", config.dataset_splitter.cv_val_ratio)
    )
    config.dataset_splitter.evaluation_mode = mode
    config.dataset_splitter.fold_index = fold_index
    config.dataset_splitter.include_video = True
    config.dataset_splitter.save_results = True
    return config


def sample_key_from_path(path: str) -> str:
    normalized = Path(str(path).replace("\\", "/"))
    parts = normalized.parts
    lowered = [p.lower() for p in parts]
    anchor = None
    for name in ("audio", "video"):
        if name in lowered:
            anchor = lowered.index(name)
            break
    if anchor is not None and anchor + 1 < len(parts):
        rel_parts = list(parts[anchor + 1 :])
    else:
        rel_parts = list(parts[-3:])
    rel_parts[-1] = Path(rel_parts[-1]).stem
    return "/".join(rel_parts)


def entry_paths(entry: List[Any]) -> Tuple[str, str, int]:
    if len(entry) >= 3:
        return str(entry[0]), str(entry[1]), int(entry[2])
    return str(entry[0]), "", int(entry[-1])


def entry_to_base_row(entry: List[Any]) -> Dict[str, Any]:
    audio_path, video_path, label = entry_paths(entry)
    return {
        "sample_key": sample_key_from_path(audio_path),
        "label": LABEL_TO_NAME[label],
        "label_id": label,
        "input_path": audio_path,
        "pair_path": video_path,
    }


def unique_entries(entries: Iterable[List[Any]]) -> List[List[Any]]:
    by_key: Dict[str, List[Any]] = {}
    for entry in entries:
        row = entry_to_base_row(entry)
        by_key[row["sample_key"]] = entry
    return [by_key[key] for key in sorted(by_key)]


def split_entries(feature_cfg: Dict[str, Any], mode: str, fold_index: int | None = None) -> Tuple[List, List, List, TrainConfig]:
    config = make_train_config(feature_cfg, mode=mode, fold_index=fold_index)
    splitter = FishDataSplitter(config=config.dataset_splitter)
    train_entries, test_entries, val_entries = splitter.split_data()
    return train_entries, val_entries, test_entries, config


def csv_value(row: Dict[str, str], names: Tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value:
            return value
    return ""


def load_split_csv(split_csv: Path, modality: str) -> List[List[Any]]:
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing checkpoint split CSV: {split_csv}")

    entries: List[List[Any]] = []
    with split_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row["label"])
            audio_path = csv_value(row, ("audio_path", "input_path", "path"))
            image_path = csv_value(row, ("image_path", "video_path", "pair_path"))
            wave_path = csv_value(row, ("wave_path",))

            if modality == "audio":
                if not audio_path:
                    raise ValueError(f"Missing audio_path in {split_csv}")
                entries.append([audio_path, image_path, wave_path, label] if wave_path else [audio_path, image_path, label])
            else:
                if not image_path:
                    image_path = csv_value(row, ("video_name", "image_name"))
                if not image_path:
                    raise ValueError(f"Missing image/video path in {split_csv}")
                entries.append([audio_path, image_path, wave_path, label] if wave_path else [audio_path, image_path, label])
    return entries


def load_checkpoint_splits(splits_dir: Path, modality: str) -> Tuple[List[List[Any]], List[List[Any]], List[List[Any]]]:
    logger.info(f"Checkpoint split directory selected: {splits_dir}")
    train_entries = load_split_csv(splits_dir / "train.csv", modality=modality)
    val_entries = load_split_csv(splits_dir / "val.csv", modality=modality)
    test_entries = load_split_csv(splits_dir / "test.csv", modality=modality)
    return train_entries, val_entries, test_entries


def resolve_num_workers(value: int) -> int:
    if value >= 0:
        return value
    max_cpu = os.cpu_count()
    if max_cpu is None or max_cpu <= 0:
        return 0
    if max_cpu == 2:
        return max_cpu // 2
    return max_cpu - 1


class AudioFeatureDataset(Dataset):
    def __init__(self, entries: List[List[Any]], sample_rate: int, cache_audio: bool, num_workers: int) -> None:
        self.entries = entries
        self.sample_rate = sample_rate
        self.cache_audio = cache_audio
        self.num_workers = max(int(num_workers), 1)
        self.waveform_cache: List[np.ndarray] | None = None
        if self.cache_audio:
            self._preload_audio()

    def _preload_audio(self) -> None:
        logger.info(f"Feature extraction stage: preloading audio waveforms into RAM (samples={len(self.entries)})")

        def load_one(index: int) -> Tuple[int, np.ndarray]:
            row = entry_to_base_row(self.entries[index])
            waveform = FishVoiceDataLoader.load_audio(row["input_path"], sr=self.sample_rate)
            return index, waveform.numpy()

        cache: List[np.ndarray | None] = [None] * len(self.entries)
        total_bytes = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(load_one, index): index for index in range(len(self.entries))}
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Preloading audio to RAM",
                unit="file",
                dynamic_ncols=True,
            ):
                index, waveform = future.result()
                cache[index] = waveform
                total_bytes += waveform.nbytes

        if any(item is None for item in cache):
            raise RuntimeError("Audio RAM preload did not fill every cache slot.")
        self.waveform_cache = [item for item in cache if item is not None]
        cache_size_mb = total_bytes / (1024 ** 2)
        logger.info(
            f"Feature extraction stage: audio RAM preload completed (samples={len(self.waveform_cache)}, cache_size_mb={cache_size_mb:.1f})"
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = entry_to_base_row(self.entries[index])
        if self.waveform_cache is not None:
            waveform = self.waveform_cache[index]
        else:
            waveform = FishVoiceDataLoader.load_audio(row["input_path"], sr=self.sample_rate).numpy()
        return {**row, "waveform": waveform}


def collate_audio(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "waveform": torch.tensor(np.stack([item["waveform"] for item in batch]), dtype=torch.float32),
        "rows": [{k: item[k] for k in ("sample_key", "label", "label_id", "input_path", "pair_path")} for item in batch],
    }


class ClassifierInputHook:
    def __init__(self, module: torch.nn.Module) -> None:
        self.features: torch.Tensor | None = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        feature = inputs[0]
        self.features = feature.detach()

    def close(self) -> None:
        self.handle.remove()


def build_model(config: TrainConfig, device: torch.device) -> AudioModel:
    frontend = AudioFrontend(config.audio_features)
    backbone = build_backbone(config)
    model = AudioModel(frontend=frontend, backbone=backbone).to(device)
    return model


def default_checkpoint_path(config: TrainConfig, model: AudioModel, fold_index: int | None = None) -> Path:
    model_name = model.backbone.get_name() if hasattr(model.backbone, "get_name") else model.backbone.__class__.__name__
    base_dir = PROJECT_ROOT / (config.ckpt_dir if config.ckpt_dir else "checkpoint")
    model_dir = base_dir if base_dir.name == model_name else base_dir / model_name
    if fold_index is not None:
        return model_dir / f"fold_{fold_index:02d}" / "audio_best.pt"
    return model_dir / "audio_best.pt"


def resolve_checkpoint_path(
    feature_cfg: Dict[str, Any],
    config: TrainConfig,
    model: AudioModel,
    fold_index: int | None = None,
) -> Path:
    if fold_index is None:
        ckpt_value = str(feature_cfg.get("holdout_checkpoint_path", "")).strip()
        checkpoint_path = Path(ckpt_value) if ckpt_value else default_checkpoint_path(config, model, fold_index=None)
    else:
        cv_dir_value = str(feature_cfg.get("cross_validation_checkpoint_dir", "")).strip()
        checkpoint_path = (
            Path(cv_dir_value) / f"fold_{fold_index:02d}" / "audio_best.pt"
            if cv_dir_value
            else default_checkpoint_path(config, model, fold_index=fold_index)
        )

    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    return checkpoint_path


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    cleaned = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    logger.info("Checkpoint compatibility verification: validating exact architecture match.")
    logger.info(f"Checkpoint path: {checkpoint_path}")
    try:
        model.load_state_dict(cleaned, strict=True)
    except RuntimeError as exc:
        logger.error("Checkpoint compatibility verification failed: checkpoint weights do not exactly match the selected architecture.")
        logger.error("Checkpoint keys must match the selected model architecture exactly.")
        raise RuntimeError(
            f"Checkpoint does not match the selected model 100%: {checkpoint_path}"
        ) from exc
    logger.info("Checkpoint compatibility verification passed: checkpoint weights exactly match the selected architecture.")
    logger.info(f"Loaded checkpoint: {checkpoint_path}")


def extract_all_features(
    model: AudioModel,
    entries: List[List[Any]],
    config: TrainConfig,
    feature_cfg: Dict[str, Any],
    device: torch.device,
    run_name: str,
) -> Tuple[np.ndarray, List[Dict[str, Any]], str, int]:
    num_workers = resolve_num_workers(int(feature_cfg.get("num_workers", 0)))
    logger.info(f"Resolved feature extraction workers: {num_workers}")
    dataset = AudioFeatureDataset(
        entries=entries,
        sample_rate=config.audio_features.sample_rate,
        cache_audio=bool(feature_cfg.get("cache_audio", False)),
        num_workers=num_workers,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(feature_cfg.get("batch_size", config.batch_size)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_audio,
        pin_memory=device.type == "cuda",
    )
    target_module = model.backbone.fc_audioset
    hook = ClassifierInputHook(target_module)
    model.eval()
    arrays: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []
    index = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=run_name, unit="batch", dynamic_ncols=True):
            waveform = batch["waveform"].to(device)
            hook.features = None
            _ = model(waveform)
            if hook.features is None:
                raise RuntimeError("Feature hook did not capture classifier input.")
            features = hook.features
            if features.dim() > 2:
                features = torch.flatten(features, start_dim=1)
            feature_np = features.detach().cpu().numpy().astype(np.float32)
            arrays.append(feature_np)
            for row in batch["rows"]:
                metadata.append(
                    {
                        "feature_index": index,
                        "modality": "audio",
                        **row,
                    }
                )
                index += 1
    hook.close()
    features_np = np.concatenate(arrays, axis=0) if arrays else np.empty((0, 0), dtype=np.float32)
    feature_name = "classifier_input_after_fc1_before_fc_audioset"
    feature_dim = int(features_np.shape[1]) if features_np.ndim == 2 and features_np.shape[0] else 0
    return features_np, metadata, feature_name, feature_dim


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_hash(rows: List[Dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{row['split']}|{row['sample_key']}|{row['label']}|{row['feature_index']}" for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize_split(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {split: 0 for split in SPLIT_ORDER}
    class_counts: Dict[str, Dict[str, int]] = {split: {} for split in SPLIT_ORDER}
    for row in rows:
        split = row["split"]
        counts[split] += 1
        class_counts[split][row["label"]] = class_counts[split].get(row["label"], 0) + 1
    return {"counts": counts, "class_counts": class_counts}


def feature_metadata_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    fieldnames = [
        "feature_index",
        "split_id",
        "mode",
        "fold",
        "split",
        "modality",
        "sample_key",
        "label",
        "label_id",
        "input_path",
        "pair_path",
    ]
    if any("wave_path" in row for row in rows):
        fieldnames.append("wave_path")
    return fieldnames


def attach_split_columns(
    metadata: List[Dict[str, Any]],
    mode: str,
    fold: int | None,
    split_entries_map: Dict[str, List[List[Any]]],
) -> List[Dict[str, Any]]:
    split_by_key: Dict[str, str] = {}
    for split in SPLIT_ORDER:
        for entry in split_entries_map[split]:
            base = entry_to_base_row(entry)
            sample_key = base["sample_key"]
            if sample_key in split_by_key:
                raise ValueError(f"Duplicate sample_key across splits for {mode}: {sample_key}")
            split_by_key[sample_key] = split

    split_id = f"{mode}_fold_{fold}" if fold is not None else mode
    rows: List[Dict[str, Any]] = []
    for row in metadata:
        sample_key = row["sample_key"]
        if sample_key not in split_by_key:
            raise KeyError(f"Extracted sample is missing from split map: {sample_key}")
        rows.append(
            {
                "feature_index": row["feature_index"],
                "split_id": split_id,
                "mode": mode,
                "fold": "" if fold is None else fold,
                "split": split_by_key[sample_key],
                "modality": row["modality"],
                "sample_key": row["sample_key"],
                "label": row["label"],
                "label_id": row["label_id"],
                "input_path": row["input_path"],
                "pair_path": row["pair_path"],
                **({"wave_path": row["wave_path"]} if "wave_path" in row else {}),
            }
        )
    return rows


def write_feature_run(
    output_dir: Path,
    mode: str,
    fold: int | None,
    split_entries_map: Dict[str, List[List[Any]]],
    config: TrainConfig,
    feature_cfg: Dict[str, Any],
    device: torch.device,
) -> None:
    combined_entries: List[List[Any]] = []
    for split in SPLIT_ORDER:
        combined_entries.extend(split_entries_map[split])

    logger.info("==================================================")
    logger.info(f"Feature extraction mode: {mode}")
    if fold is not None:
        logger.info(f"FOLD_INDEX: {fold}")
    logger.info(f"Samples to extract for this run: {len(combined_entries)}")
    logger.info("==================================================")

    model = build_model(config, device)
    checkpoint_path = resolve_checkpoint_path(feature_cfg, config, model, fold_index=fold)
    logger.info(f"Selected checkpoint path: {checkpoint_path}")
    load_checkpoint(model, checkpoint_path, device)

    run_name = f"Extracting {mode}" if fold is None else f"Extracting {mode} fold_{fold}"
    logger.info(f"Feature extraction stage: started feature extraction for {run_name}")
    features, metadata, feature_name, feature_dim = extract_all_features(
        model=model,
        entries=combined_entries,
        config=config,
        feature_cfg=feature_cfg,
        device=device,
        run_name=run_name,
    )
    logger.info(f"Feature extraction stage: completed feature extraction for {run_name}")
    rows = attach_split_columns(
        metadata=metadata,
        mode=mode,
        fold=fold,
        split_entries_map=split_entries_map,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Feature extraction stage: saving output files to {output_dir}")
    np.save(output_dir / "features.npy", features)
    write_csv(output_dir / "metadata.csv", rows, feature_metadata_fieldnames(rows))

    summary = summarize_split(rows)
    write_json(
        output_dir / "split_info.json",
        {
            "split_id": f"{mode}_fold_{fold}" if fold is not None else mode,
            "mode": mode,
            "fold": fold,
            "seed": config.dataset_splitter.seed,
            "split_strategy": config.dataset_splitter.split_strategy,
            "test_sample_per_class": config.dataset_splitter.test_sample_per_class,
            "num_folds": config.dataset_splitter.num_folds,
            "cv_val_ratio": config.dataset_splitter.cv_val_ratio,
            "holdout_val_ratio": getattr(config.dataset_splitter, "holdout_val_ratio", None),
            "holdout_test_ratio": getattr(config.dataset_splitter, "holdout_test_ratio", None),
            "split_hash": split_hash(rows),
            **summary,
        },
    )
    feature_info = {
        "modality": rows[0]["modality"] if rows else "",
        "backbone": config.model.backbone,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_match_status": "STRICT_100_PERCENT_MATCH",
        "feature_name": feature_name,
        "feature_dim": feature_dim,
        "num_samples": int(features.shape[0]),
        "feature_shape": list(features.shape),
        "dataset_path": config.dataset_splitter.dataset_path,
    }
    if hasattr(config, "video_features"):
        feature_info["image_size"] = config.video_features.image_size
    write_json(output_dir / "feature_info.json", feature_info)
    logger.info(f"Feature extraction output directory: {output_dir}")
    logger.info("Feature extraction stage: run completed successfully.")


def main() -> None:
    feature_cfg = load_feature_config()
    requested_device = str(feature_cfg.get("device", "cuda"))
    device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
    output_root = PROJECT_ROOT / feature_cfg.get("output_dir", "outputs")
    use_checkpoint_splits = bool(feature_cfg.get("use_checkpoint_splits", True))

    holdout_config = make_train_config(feature_cfg, mode="holdout")
    holdout_model = build_model(holdout_config, device)
    holdout_checkpoint_path = resolve_checkpoint_path(feature_cfg, holdout_config, holdout_model, fold_index=None)
    del holdout_model
    if use_checkpoint_splits:
        holdout_train, holdout_val, holdout_test = load_checkpoint_splits(
            holdout_checkpoint_path.parent / "splits", modality="audio"
        )
    else:
        holdout_train, holdout_val, holdout_test, holdout_config = split_entries(feature_cfg, mode="holdout")
    write_feature_run(
        output_dir=output_root / "holdout",
        mode="holdout",
        fold=None,
        split_entries_map={"train": holdout_train, "val": holdout_val, "test": holdout_test},
        config=holdout_config,
        feature_cfg=feature_cfg,
        device=device,
    )

    cv_root = output_root / "cross_validation"
    for fold_index in range(holdout_config.dataset_splitter.num_folds):
        fold_config = make_train_config(feature_cfg, mode="cross_validation", fold_index=fold_index)
        fold_model = build_model(fold_config, device)
        fold_checkpoint_path = resolve_checkpoint_path(feature_cfg, fold_config, fold_model, fold_index=fold_index)
        del fold_model
        if use_checkpoint_splits:
            train_entries, val_entries, test_entries = load_checkpoint_splits(
                fold_checkpoint_path.parent / "splits", modality="audio"
            )
        else:
            train_entries, val_entries, test_entries, fold_config = split_entries(
                feature_cfg, mode="cross_validation", fold_index=fold_index
            )
        write_feature_run(
            output_dir=cv_root / f"fold_{fold_index}",
            mode="cross_validation",
            fold=fold_index,
            split_entries_map={"train": train_entries, "val": val_entries, "test": test_entries},
            config=fold_config,
            feature_cfg=feature_cfg,
            device=device,
        )

    logger.info(f"Feature extraction completed successfully. Output: {output_root}")
    artifact_upload_config = load_artifact_upload_config(feature_cfg)
    artifact_config = make_train_config(feature_cfg, mode="holdout")
    artifact_config.dataset_splitter.evaluation_mode = "holdout_cross_validation"
    upload_features_artifact_if_enabled(artifact_upload_config, artifact_config)


if __name__ == "__main__":
    main()
