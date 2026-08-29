import os
import sys
import time
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import VideoTrainConfig
from utils import EarlyStopping, HistoryLogger, VideoEvaluator, InferenceTimer

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


class VideoTrainer:
    """
    Unified Trainer class for Fish Video Intensity Classification.
    Identical OOP structure with AudioTrainer: supports HistoryLogger,
    EarlyStopping, InferenceTimer, and automatic experiment checkpointing.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        test_loader: Any,
        config: VideoTrainConfig,
        device: torch.device,
        optimizer: Optional[optim.Optimizer] = None,
        train_config_path: str = 'config/train_config.json'
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device

        # Determine model name for the checkpoint subfolder (matching AudioTrainer)
        model_name = "video_model"
        if hasattr(model, 'get_name'):
            model_name = model.get_name()
        elif hasattr(model, 'backbone') and hasattr(model.backbone, 'get_name'):
            model_name = model.backbone.get_name()

        base_dir = config.ckpt_dir if config.ckpt_dir else "checkpoint"
        base_path = Path(base_dir)
        if base_path.name.startswith("fold_") and base_path.parent.name == model_name:
            self.ckpt_dir = base_dir
        elif not base_dir.rstrip('/\\').endswith(model_name):
            self.ckpt_dir = os.path.join(base_dir, model_name)
        else:
            self.ckpt_dir = base_dir

        os.makedirs(self.ckpt_dir, exist_ok=True)

        try:
            # 1. Copy the full train_config.json
            if os.path.exists(train_config_path):
                shutil.copy(train_config_path, os.path.join(self.ckpt_dir, 'train_config.json'))
            else:
                with open(os.path.join(self.ckpt_dir, 'train_config.json'), 'w', encoding='utf-8') as f:
                    json.dump(self.config.model_dump(), f, indent=2)

            # 2. Extract and save splitter_config.json for compatibility
            splitter_data = self.config.dataset_splitter.model_dump()
            with open(os.path.join(self.ckpt_dir, 'splitter_config.json'), 'w', encoding='utf-8') as f:
                json.dump(splitter_data, f, indent=2)

            dataset_path = Path(self.config.dataset_splitter.dataset_path)
            local_splits_dir = dataset_path / 'splits'
            legacy_splits_dir = dataset_path.parent / 'splits'
            if dataset_path.name in ['audio', 'video'] and legacy_splits_dir.exists() and not local_splits_dir.exists():
                base_splits_dir = legacy_splits_dir
            else:
                base_splits_dir = local_splits_dir

            if getattr(self.config.dataset_splitter, "evaluation_mode", "holdout") == "cross_validation":
                fold_index = int(self.config.dataset_splitter.fold_index)
                splits_dir = base_splits_dir / "cv" / f"fold_{fold_index:02d}"
            else:
                splits_dir = base_splits_dir

            if splits_dir.exists():
                shutil.copytree(splits_dir, os.path.join(self.ckpt_dir, 'splits'), dirs_exist_ok=True)
                logger.info(f"Successfully backed up dataset splits from '{splits_dir}' to checkpoint directory.")
            else:
                logger.warning(f"Warning: Dataset splits directory not found, cannot back it up: '{splits_dir}'")
            logger.info("Successfully backed up active configurations to checkpoint directory.")
        except Exception as e:
            logger.warning(f"Warning: Failed to backup configuration files: {str(e)}")

        self.criterion = nn.CrossEntropyLoss()
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # Early stopping & History logger
        self.early_stopping = config.early_stopping
        if self.early_stopping:
            self.early_stopper = EarlyStopping(patience=config.patience, delta=config.delta, verbose=True)
        else:
            self.early_stopper = None

        self.history_logger = HistoryLogger(log_dir=self.ckpt_dir)
        self.evaluator = VideoEvaluator(model=self.model)

        logger.info("==================================================")
        logger.info("VideoTrainer successfully initialized:")
        logger.info(f"  - Monitor Metric:               '{self.config.monitor}'")
        logger.info(f"  - Early Stopping Enabled:       {self.early_stopping}")
        if self.early_stopping:
            logger.info(f"    * Patience:                   {self.config.patience} epochs")
            logger.info(f"    * Delta:                      {self.config.delta}")
        logger.info(f"  - Checkpoint Dir:               '{self.ckpt_dir}'")
        logger.info("  - Precision:                    FP32")
        logger.info("==================================================")

    def _save_checkpoint(self, path: str, epoch: int, metric_val: float) -> None:
        """Private Method to save model weights, optimizer state, and progress metrics."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ave_precision': metric_val,
        }, path)
        logger.info(f"Saved best model checkpoint to: '{path}' (Monitor value = {metric_val:.5f})")

    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        self.model.train()
        total_loss = 0.0
        train_preds = []
        train_targets = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.epochs}")
        for batch in pbar:
            inputs = batch['video_form'].to(self.device)
            targets = batch['target'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            if isinstance(outputs, dict):
                outputs = outputs.get('clipwise_output', outputs)

            if targets.dim() > 1:
                target_labels = targets.argmax(dim=1)
            else:
                target_labels = targets
            loss = self.criterion(outputs, target_labels)
            loss.backward()
            self.optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val
            train_preds.append(outputs.detach().float().cpu().numpy())
            train_targets.append(targets.detach().float().cpu().numpy())

            pbar.set_postfix({"Loss": f"{loss_val:.4f}"})

        epoch_loss = total_loss / len(self.train_loader)
        train_preds = np.concatenate(train_preds, axis=0)
        train_targets = np.concatenate(train_targets, axis=0)

        target_acc_labels = np.argmax(train_targets, axis=1) if train_targets.ndim > 1 else train_targets
        pred_acc_labels = np.argmax(train_preds, axis=1)
        train_acc = np.mean(target_acc_labels == pred_acc_labels)

        from sklearn import metrics as sklearn_metrics
        train_mAP = np.mean(sklearn_metrics.average_precision_score(train_targets, train_preds, average=None))

        return epoch_loss, train_acc, train_mAP

    def fit(self) -> None:
        logger.info(f"Starting training pipeline (Monitor metric: {self.config.monitor})...")

        best_acc = 0.0
        best_mAP = 0.0
        best_loss = float('inf')
        best_epoch = 0
        best_val_statistics = None

        if self.config.monitor == 'accuracy':
            best_val_metric = 0.0
        else:
            best_val_metric = float('inf')

        if self.early_stopping and self.early_stopper is not None:
            self.early_stopper.reset()

        train_start_time = time.perf_counter()

        for epoch in range(self.config.epochs):
            train_loss, train_acc, train_mAP = self.train_epoch(epoch)

            # Evaluate on validation set
            self.model.eval()
            val_statistics = self.evaluator.evaluate(self.val_loader)
            val_acc = np.mean(val_statistics['accuracy'])
            val_mAP = np.mean(val_statistics['average_precision'])

            # Compute validation loss
            val_loss_sum = 0.0
            with torch.no_grad():
                for val_batch in self.val_loader:
                    val_inputs = val_batch['video_form'].to(self.device)
                    val_targets = val_batch['target'].to(self.device)
                    val_outputs = self.model(val_inputs)
                    if isinstance(val_outputs, dict):
                        val_outputs = val_outputs.get('clipwise_output', val_outputs)
                    val_targ_labels = val_targets.argmax(dim=1) if val_targets.dim() > 1 else val_targets
                    val_loss = self.criterion(val_outputs, val_targ_labels)
                    val_loss_sum += val_loss.item()
            val_loss = val_loss_sum / len(self.val_loader)

            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss = {train_loss:.5f} | Train Acc = {train_acc:.4f} | Train mAP = {train_mAP:.4f} | "
                f"Val Loss = {val_loss:.5f} | Val Acc = {val_acc:.4f} | Val mAP = {val_mAP:.4f}"
            )

            # Save optimal model checkpoint based on monitored metric
            is_best = False
            if self.config.monitor == 'accuracy':
                score = val_acc
                if val_acc > best_val_metric:
                    best_val_metric = val_acc
                    is_best = True
            else:
                score = -val_loss
                if val_loss < best_val_metric:
                    best_val_metric = val_loss
                    is_best = True

            if is_best:
                best_epoch = epoch
                best_acc = val_acc
                best_mAP = val_mAP
                best_loss = val_loss
                best_val_statistics = val_statistics
                save_path = os.path.join(self.ckpt_dir, "video_best.pt")
                self._save_checkpoint(save_path, epoch, val_acc if self.config.monitor == 'accuracy' else val_loss)

            # Record metrics and confusion matrix to history CSV
            self.history_logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                train_mAP=train_mAP,
                val_loss=val_loss,
                val_statistics=val_statistics,
                is_best=is_best
            )

            # Check early stopping conditions
            if self.early_stopping and self.early_stopper is not None:
                if self.early_stopper.step(score):
                    logger.info(f"Early stopping triggered at epoch {epoch}!")
                    break

            logger.info(
                f"Current best: Epoch {best_epoch} | Loss: {best_loss:.5f} | Accuracy: {best_acc:.4f} | mAP: {best_mAP:.4f}"
            )

        # Calculate final training time in seconds
        training_time = time.perf_counter() - train_start_time

        # Generate learning curves plot
        try:
            self.history_logger.plot_history()
        except Exception as e:
            logger.warning(f"Warning: Failed to generate learning curves plot: {str(e)}")

        # Final evaluation on Test dataset using best checkpoint
        logger.info("==================================================")
        logger.info("Training complete. Starting evaluation on Test split...")
        best_ckpt = os.path.join(self.ckpt_dir, "video_best.pt")

        if os.path.exists(best_ckpt):
            checkpoint = torch.load(best_ckpt, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Reloaded best checkpoint model from Epoch {checkpoint['epoch']}...")

            self.model.eval()
            test_statistics = self.evaluator.evaluate(self.test_loader)
            test_mAP = np.mean(test_statistics['average_precision'])
            test_acc = np.mean(test_statistics['accuracy'])
            logger.info(f"TEST Results -> Accuracy: {test_acc:.4f} | mAP: {test_mAP:.4f}")
            logger.info(f"Detailed Classification Report:\n{test_statistics['message']}")

            # Measure model inference latency
            logger.info("Measuring model Inference Latency on device...")
            timer = InferenceTimer(model=self.model, device=self.device)
            img_size = self.config.video_features.image_size
            latency_ms = timer.measure_latency_per_sample(
                input_shape=(1, 3, img_size, img_size),
                warm_up_steps=10,
                num_steps=50
            )

            # Save performance and timing summary to summary.csv
            if best_val_statistics is not None:
                self.history_logger.save_summary(training_time, latency_ms, best_val_statistics, test_statistics)
        else:
            logger.error("Error: Best model checkpoint video_best.pt not found. Cannot evaluate on Test split!")
