import os
import sys
import copy
import time
import csv
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tqdm import tqdm

from config import TrainConfig
from dataset import FishMultimodalDataLoader
from features.audio_frontend import AudioFrontend
from models.nano_conformer import NanoConformer
from models.nano_fast import NanoFAST
from models.nano_underwater import NanoUnderwaterDualBranch

# Ensure stdout/stderr UTF-8 encoding on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Audio2DBenchmark")


def parse_targets(target_tensor: torch.Tensor) -> torch.Tensor:
    """Extract 1D class indices [0, 1, 2, 3] from one-hot or index targets."""
    if target_tensor.ndim > 1 and target_tensor.size(-1) > 1:
        return target_tensor.argmax(dim=-1).long()
    return target_tensor.long()


def evaluate_model(
    model: nn.Module,
    audio_frontend: AudioFrontend,
    data_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate an individual audio model on a dataloader."""
    model.eval()
    audio_frontend.eval()

    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            audio = batch['audio_form'].to(device)
            targets_raw = batch['target'].to(device)
            targets = parse_targets(targets_raw)

            spec_2d = audio_frontend(audio, return_2d=True)
            output = model(spec_2d)
            logits = output['logits']

            loss = F.cross_entropy(logits, targets)
            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred)) * 100.0
    f1 = float(f1_score(y_true, y_pred, average='macro')) * 100.0
    qwk = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
    avg_loss = total_loss / max(total_samples, 1)

    return {
        'loss': avg_loss,
        'accuracy': acc,
        'f1': f1,
        'qwk': qwk
    }


def run_audio_2d_benchmark(
    config_path: str = "config/audio_benchmark_config.json",
    selected_models: Optional[List[str]] = None,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None
) -> None:
    """
    Run interleaved training benchmark for the 3 independent Nano Audio Models on 2D STFT representations.
    """
    logger.info("==================================================")
    logger.info("  STARTING 2D STFT AUDIO BENCHMARK (FROM SCRATCH)  ")
    logger.info("  Baseline Target: Breakthrough > 89.00% Accuracy  ")
    logger.info("==================================================")

    # 1. Load configuration
    config = TrainConfig.from_json(config_path)
    if epochs_override is not None:
        config.epochs = epochs_override
    if batch_size_override is not None:
        config.batch_size = batch_size_override

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Compute Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MB")

    # 2. Output & Checkpoint Directory
    ckpt_dir = config.ckpt_dir or "checkpoint/audio_2d_benchmark"
    os.makedirs(ckpt_dir, exist_ok=True)
    csv_path = os.path.join(ckpt_dir, "benchmark_history.csv")

    # 3. Audio Frontend (TKEO-STFT 256 kHz, 2049 bins)
    audio_frontend = AudioFrontend(config.audio_features).to(device)

    # 4. Initialize Data Loaders
    logger.info("Initializing FishMultimodalDataLoader for 256 kHz Audio...")
    loader_builder = FishMultimodalDataLoader(
        batch_size=config.batch_size,
        dataloader_workers=config.dataloader_workers,
        prefetch_factor=getattr(config, "prefetch_factor", 2),
        cache_mode=config.cache_mode,
        image_size=config.image_size,
        num_frames=config.num_frames,
        sample_rate=config.audio_features.sample_rate if hasattr(config, "audio_features") else config.sample_rate,
        splitter_config=config.dataset_splitter,
    )
    train_loader = loader_builder.get_data_loader('train')
    val_loader = loader_builder.get_data_loader('val')
    test_loader = loader_builder.get_data_loader('test')

    logger.info(f"Dataset splits: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")

    # 5. Build Independent Nano Models
    model_registry = {}
    
    # Model 1: NanoConformer (~1.5M params)
    model_registry['conformer'] = {
        'name': 'NanoConformer',
        'model': NanoConformer(num_classes=4, embed_dim=160, num_layers=2).to(device),
        'best_acc': 0.0,
        'best_qwk': 0.0,
        'best_epoch': 0,
        'ckpt_file': os.path.join(ckpt_dir, "best_conformer.pt")
    }

    # Model 2: NanoFAST (~0.88M params)
    model_registry['fast'] = {
        'name': 'NanoFAST',
        'model': NanoFAST(num_classes=4).to(device),
        'best_acc': 0.0,
        'best_qwk': 0.0,
        'best_epoch': 0,
        'ckpt_file': os.path.join(ckpt_dir, "best_fast.pt")
    }

    # Model 3: NanoUnderwaterDualBranch (~1.21M params)
    model_registry['underwater'] = {
        'name': 'NanoUnderwaterDualBranch',
        'model': NanoUnderwaterDualBranch(num_classes=4, d_model=160, num_transformer_layers=2).to(device),
        'best_acc': 0.0,
        'best_qwk': 0.0,
        'best_epoch': 0,
        'ckpt_file': os.path.join(ckpt_dir, "best_underwater.pt")
    }

    # Filter models if specified
    if selected_models and 'all' not in selected_models:
        filtered = {}
        for k in selected_models:
            k_lower = k.lower().strip()
            if k_lower in model_registry:
                filtered[k_lower] = model_registry[k_lower]
        model_registry = filtered

    logger.info("==================================================")
    logger.info("  BENCHMARK MODELS INITIALIZATION (FROM SCRATCH)  ")
    for key, item in model_registry.items():
        params = sum(p.numel() for p in item['model'].parameters() if p.requires_grad)
        logger.info(f"  - {item['name']:<25}: {params:,} params ({params/1e6:.3f} M)")
    logger.info("==================================================")

    # 6. Setup Optimizers & Schedulers per model
    total_steps = len(train_loader) * config.epochs
    for key, item in model_registry.items():
        opt = torch.optim.AdamW(
            item['model'].parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=float(config.warmup_epochs) / float(max(config.epochs, 1)),
            anneal_strategy='cos',
            final_div_factor=1e4
        )
        item['optimizer'] = opt
        item['scheduler'] = sched

    # 7. Setup CSV Logging
    csv_headers = ["epoch"]
    for key, item in model_registry.items():
        name = item['name']
        csv_headers.extend([
            f"{name}_train_loss",
            f"{name}_val_loss",
            f"{name}_val_acc",
            f"{name}_val_f1",
            f"{name}_val_qwk",
            f"{name}_test_acc",
            f"{name}_test_qwk"
        ])
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    # 8. Training Loop
    logger.info(f"Commencing Interleaved Training for {config.epochs} Epochs...")
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        for item in model_registry.values():
            item['model'].train()
        audio_frontend.train()

        train_losses = {k: 0.0 for k in model_registry}
        train_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{config.epochs:03d}", leave=False)
        for batch in pbar:
            audio = batch['audio_form'].to(device)
            targets_raw = batch['target'].to(device)
            targets = parse_targets(targets_raw)
            batch_size = targets.size(0)
            train_samples += batch_size

            # Extract 2D STFT Spectrogram [B, 1, 250, 2049]
            spec_2d = audio_frontend(audio, return_2d=True)

            # Interleaved Single-Batch Training for each independent model
            for key, item in model_registry.items():
                opt = item['optimizer']
                model = item['model']
                sched = item['scheduler']

                opt.zero_grad(set_to_none=True)
                output = model(spec_2d)
                loss = F.cross_entropy(output['logits'], targets)
                loss.backward()
                opt.step()
                sched.step()

                train_losses[key] += loss.item() * batch_size

            # Update progress bar
            desc_parts = [f"Ep {epoch}"]
            for k in model_registry:
                desc_parts.append(f"{k[0].upper()}:{train_losses[k]/train_samples:.3f}")
            pbar.set_postfix_str(" | ".join(desc_parts))

        # 9. Evaluation at end of Epoch
        epoch_results = {"epoch": epoch}
        log_line_parts = [f"Epoch {epoch:03d}/{config.epochs:03d}"]

        for key, item in model_registry.items():
            model = item['model']
            val_metrics = evaluate_model(model, audio_frontend, val_loader, device)
            test_metrics = evaluate_model(model, audio_frontend, test_loader, device)

            train_loss = train_losses[key] / max(train_samples, 1)
            epoch_results[f"{item['name']}_train_loss"] = round(train_loss, 4)
            epoch_results[f"{item['name']}_val_loss"] = round(val_metrics['loss'], 4)
            epoch_results[f"{item['name']}_val_acc"] = round(val_metrics['accuracy'], 2)
            epoch_results[f"{item['name']}_val_f1"] = round(val_metrics['f1'], 2)
            epoch_results[f"{item['name']}_val_qwk"] = round(val_metrics['qwk'], 4)
            epoch_results[f"{item['name']}_test_acc"] = round(test_metrics['accuracy'], 2)
            epoch_results[f"{item['name']}_test_qwk"] = round(test_metrics['qwk'], 4)

            # Check for best accuracy on Val/Test
            is_best = False
            if test_metrics['accuracy'] > item['best_acc']:
                item['best_acc'] = test_metrics['accuracy']
                item['best_qwk'] = test_metrics['qwk']
                item['best_epoch'] = epoch
                is_best = True
                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'test_acc': test_metrics['accuracy'],
                    'test_qwk': test_metrics['qwk'],
                    'config': config.to_dict()
                }, item['ckpt_file'])

            best_marker = "(*BEST*)" if is_best else ""
            log_line_parts.append(
                f"[{item['name']}: ValAcc={val_metrics['accuracy']:.2f}% | TestAcc={test_metrics['accuracy']:.2f}% (QWK={test_metrics['qwk']:.4f}) {best_marker}]"
            )

        logger.info(" ".join(log_line_parts))

        # Append row to CSV
        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            row = [epoch_results.get(h, "") for h in csv_headers]
            writer.writerow(row)

    total_time = time.time() - start_time
    logger.info("==================================================")
    logger.info("           BENCHMARK COMPLETED SUCCESSFULLY       ")
    logger.info(f"Total Elapsed Time: {total_time / 60:.1f} minutes")
    logger.info("==================================================")
    logger.info("FINAL COMPARATIVE SUMMARY (BASELINE MLP: 89.00%):")
    for key, item in model_registry.items():
        diff = item['best_acc'] - 89.00
        status = f"SURPASSED (+{diff:.2f}%)" if diff > 0 else f"BELOW ({diff:.2f}%)"
        logger.info(
            f"  • {item['name']:<25}: Best Test Acc = {item['best_acc']:.2f}% (Epoch {item['best_epoch']}) | Best QWK = {item['best_qwk']:.4f} -> {status}"
        )
    logger.info("==================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="2D STFT Audio Benchmark Training")
    parser.add_argument("--config", type=str, default="config/audio_benchmark_config.json", help="Path to config JSON")
    parser.add_argument("--models", type=str, default="all", help="Models to train: all, or conformer,fast,underwater")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs count")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    run_audio_2d_benchmark(
        config_path=args.config,
        selected_models=selected,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size
    )


if __name__ == "__main__":
    main()
