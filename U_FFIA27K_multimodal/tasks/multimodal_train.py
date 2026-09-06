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
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
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
        self.evaluator = MultimodalEvaluator(model=self.model)
        self.timer = InferenceTimer(model=self.model, device=self.device)

        # Early stopping setup
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            delta=self.config.delta,
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
        correct_predictions = 0
        total_samples = 0

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

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            logits = outputs['clipwise_output']
            preds = torch.argmax(logits, dim=1)
            gt = torch.argmax(targets, dim=1) if targets.dim() > 1 else targets
            correct_predictions += (preds == gt).sum().item()

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_loss = total_loss / max(1, total_samples)
        epoch_acc = correct_predictions / max(1, total_samples)
        epoch_mAP = epoch_acc  # Proxy metric for training loop speed
        return epoch_loss, epoch_acc, epoch_mAP

    def train(self) -> Dict[str, Any]:
        logger.info("Starting multimodal training session...")
        training_start_time = time.perf_counter()

        best_score = float('inf') if self.config.monitor == 'loss' else -float('inf')

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc, train_mAP = self._train_epoch(epoch)
            self.scheduler.step()

            # Evaluate on validation split
            val_stats = self.evaluator.evaluate(self.val_loader)
            val_loss = 1.0 - val_stats['accuracy']  # Monitor loss proxy
            val_acc = val_stats['accuracy']

            # Determine if this is the best checkpoint
            is_best = False
            current_score = val_loss if self.config.monitor == 'loss' else val_acc
            score_improved = (current_score < best_score) if self.config.monitor == 'loss' else (current_score > best_score)

            if score_improved:
                best_score = current_score
                is_best = True
                torch.save(self.model.state_dict(), self.best_checkpoint_path)
                logger.info(f"[*] New best validation performance! Saved checkpoint: '{self.best_checkpoint_path}'")

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
            stop_score = -val_loss if self.config.monitor == 'loss' else val_acc
            if self.config.early_stopping and self.early_stopping.step(stop_score):
                logger.info(f"Early stopping condition satisfied at epoch {epoch}. Stopping training.")
                break

        training_duration = time.perf_counter() - training_start_time

        # Load best model for final evaluation
        if os.path.exists(self.best_checkpoint_path):
            self.model.load_state_dict(torch.load(self.best_checkpoint_path, map_location=self.device))
            logger.info(f"Loaded best checkpoint '{self.best_checkpoint_path}' for final evaluation.")

        final_val_stats = self.evaluator.evaluate(self.val_loader)
        final_test_stats = self.evaluator.evaluate(self.test_loader)

        # Measure inference latency
        inference_latency_ms = self.timer.measure_latency_per_sample()

        # Save summary report & plot curves
        self.logger.save_summary(
            training_time=training_duration,
            inference_time_ms=inference_latency_ms,
            val_statistics=final_val_stats,
            test_statistics=final_test_stats
        )
        self.logger.plot_history()

        return {
            'training_time': training_duration,
            'inference_time_ms': inference_latency_ms,
            'val_statistics': final_val_stats,
            'test_statistics': final_test_stats
        }
