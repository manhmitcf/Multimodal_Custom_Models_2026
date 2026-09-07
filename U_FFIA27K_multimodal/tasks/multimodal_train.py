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

from config import MultimodalTrainConfig
from utils import EarlyStopping, HistoryLogger, MultimodalEvaluator, InferenceTimer, ClipCELoss

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultimodalTrainer:
    """
    Unified Trainer class for Multimodal Fish Feeding Intensity Classification.
    Identical OOP structure with VideoTrainer and AudioTrainer: supports HistoryLogger,
    EarlyStopping, InferenceTimer, and automatic experiment checkpointing.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        test_loader: Any,
        config: MultimodalTrainConfig,
        device: torch.device,
        optimizer: Optional[optim.Optimizer] = None,
        train_config_path: str = 'config/train_config.json'
    ) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.train_config_path = train_config_path

        # Setup Loss and Optimizer
        self.loss_fn = ClipCELoss()
        self.optimizer = optimizer if optimizer is not None else optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Learning Rate Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=1e-6
        )

        # Evaluator and Timer
        self.evaluator = MultimodalEvaluator(model=self.model, loss_fn=self.loss_fn)
        self.timer = InferenceTimer(model=self.model, device=self.device)

        # Early stopping setup
        self.early_stopping = EarlyStopping(
            patience=getattr(self.config, "patience", 80),
            delta=getattr(self.config, "delta", 0.0),
            verbose=True
        )

        self._init_logging_and_checkpoints()

    def _init_logging_and_checkpoints(self) -> None:
        model_name = getattr(self.model, "model_name", self.model.__class__.__name__.lower())
        self.run_dir = os.path.join(self.config.ckpt_dir, model_name)
        
        # Handle cross-validation fold subdirectories
        if self.config.dataset_splitter.evaluation_mode == "cross_validation" and self.config.dataset_splitter.fold_index is not None:
            self.run_dir = os.path.join(self.run_dir, f"fold_{self.config.dataset_splitter.fold_index}")

        os.makedirs(self.run_dir, exist_ok=True)
        self.logger = HistoryLogger(log_dir=self.run_dir)
        self.best_checkpoint_path = os.path.join(self.run_dir, 'best_model.pth')
        self.last_checkpoint_path = os.path.join(self.run_dir, 'last_model.pth')

        logger.info("==================================================")
        logger.info("INITIALIZED MULTIMODAL TRAINING EXPERIMENT:")
        logger.info(f"  - Model Architecture:       {model_name}")
        logger.info(f"  - Device:                   {self.device}")
        logger.info(f"  - Max Epochs:               {self.config.epochs}")
        logger.info(f"  - Batch Size:               {self.config.batch_size}")
        logger.info(f"  - Learning Rate:            {self.config.learning_rate}")
        logger.info(f"  - Checkpoint Run Dir:       '{self.run_dir}'")
        logger.info("==================================================")

    def _train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        self.model.train()
        total_loss = 0.0
        train_preds = []
        train_targets = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d}/{self.config.epochs:03d} [Train]")
        for batch_dict in pbar:
            video = batch_dict['video_form'].to(self.device)
            audio = batch_dict['audio_form'].to(self.device)
            targets = batch_dict['target'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(video, audio)

            loss = self.loss_fn(outputs, {'target': targets})
            loss.backward()
            
            # Gradient clipping to ensure stable training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val

            logits = outputs['clipwise_output']
            train_preds.append(logits.detach().cpu().numpy())
            train_targets.append(targets.detach().cpu().numpy())

            pbar.set_postfix({'loss': f"{loss_val:.4f}"})

        epoch_loss = total_loss / max(1, len(self.train_loader))
        train_preds = np.concatenate(train_preds, axis=0)
        train_targets = np.concatenate(train_targets, axis=0)

        target_acc_labels = np.argmax(train_targets, axis=1) if train_targets.ndim > 1 else train_targets
        pred_acc_labels = np.argmax(train_preds, axis=1)
        train_acc = float(np.mean(target_acc_labels == pred_acc_labels))

        try:
            from sklearn import metrics as sklearn_metrics
            train_mAP = float(np.mean(sklearn_metrics.average_precision_score(train_targets, train_preds, average=None)))
        except Exception:
            train_mAP = train_acc

        return epoch_loss, train_acc, train_mAP

    def train(self) -> Dict[str, Any]:
        logger.info(f"Starting training pipeline (Monitor metric: {self.config.monitor})...")
        training_start_time = time.perf_counter()

        best_acc = 0.0
        best_mAP = 0.0
        best_loss = float('inf')
        best_epoch = 1
        best_val_statistics = None

        if self.config.monitor == 'accuracy':
            best_val_metric = 0.0
        else:
            best_val_metric = float('inf')

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc, train_mAP = self._train_epoch(epoch)
            self.scheduler.step()

            # Evaluate on validation split
            self.model.eval()
            val_stats = self.evaluator.evaluate(self.val_loader)
            val_loss = float(val_stats.get('loss', 0.0))
            val_acc = float(np.mean(val_stats['accuracy']))
            val_mAP = float(np.mean(val_stats['average_precision']))

            # Print epoch summary metrics identical to audio and video trainers
            logger.info(
                f"Epoch {epoch:03d}: "
                f"Train Loss = {train_loss:.5f} | Train Acc = {train_acc:.4f} | Train mAP = {train_mAP:.4f} | "
                f"Val Loss = {val_loss:.5f} | Val Acc = {val_acc:.4f} | Val mAP = {val_mAP:.4f}"
            )

            # Determine if this is the best checkpoint
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
                best_val_statistics = val_stats
                torch.save(self.model.state_dict(), self.best_checkpoint_path)
                logger.info(f"[*] New best validation performance! Saved checkpoint: '{self.best_checkpoint_path}' (Monitor value = {val_acc if self.config.monitor == 'accuracy' else val_loss:.5f})")

            logger.info(
                f"Current best: Epoch {best_epoch:03d} | Loss: {best_loss:.5f} | Accuracy: {best_acc:.4f} | mAP: {best_mAP:.4f}"
            )

            # Always save last checkpoint
            torch.save(self.model.state_dict(), self.last_checkpoint_path)

            # Log to history CSV
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                train_mAP=train_mAP,
                val_loss=val_loss,
                val_statistics=val_stats,
                is_best=is_best
            )

            # Early stopping check
            if self.config.early_stopping and self.early_stopping is not None:
                if self.early_stopping.step(score):
                    logger.info(f"Early stopping condition satisfied at epoch {epoch:03d}. Stopping training.")
                    break

        training_duration = time.perf_counter() - training_start_time

        # Generate learning curves plot
        try:
            self.logger.plot_history()
        except Exception as exc:
            logger.warning(f"Failed to generate learning curves plot: {exc}")

        # Final evaluation on Test split
        logger.info("==================================================")
        logger.info("Training complete. Starting evaluation on Test split...")
        if os.path.exists(self.best_checkpoint_path):
            self.model.load_state_dict(torch.load(self.best_checkpoint_path, map_location=self.device, weights_only=True))
            logger.info(f"Reloaded best checkpoint '{self.best_checkpoint_path}' from Epoch {best_epoch:03d}...")

        self.model.eval()
        final_val_stats = best_val_statistics if best_val_statistics is not None else self.evaluator.evaluate(self.val_loader)
        final_test_stats = self.evaluator.evaluate(self.test_loader)

        test_acc = float(np.mean(final_test_stats['accuracy']))
        test_mAP = float(np.mean(final_test_stats['average_precision']))
        logger.info(f"TEST Results -> Accuracy: {test_acc:.4f} | mAP: {test_mAP:.4f}")
        logger.info(f"Detailed Classification Report:\n{final_test_stats.get('message', '')}")
        if 'confu_matrix' in final_test_stats:
            logger.info(f"Confusion Matrix:\n{final_test_stats['confu_matrix']}")

        # Measure inference latency
        logger.info("Measuring model Inference Latency on device...")
        inference_latency_ms = self.timer.measure_latency_per_sample()

        # Save summary report
        self.logger.save_summary(
            training_time=training_duration,
            inference_time_ms=inference_latency_ms,
            val_statistics=final_val_stats,
            test_statistics=final_test_stats
        )

        return {
            'training_time': training_duration,
            'inference_time_ms': inference_latency_ms,
            'val_statistics': final_val_stats,
            'test_statistics': final_test_stats
        }
