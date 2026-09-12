import os
import sys
import math
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
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from config import TrainConfig
from utils import (
    EarlyStopping,
    HistoryLogger,
    MultimodalEvaluator,
    InferenceTimer,
    ClipCELoss,
    PairwiseTournamentLoss,
)
from utils.profile_model import count_parameters, measure_flops

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _safe_torch_save(obj: Any, target_path: str) -> bool:
    """
    Atomic fail-safe saving for PyTorch checkpoints.
    Saves to a temporary file first, then atomically renames to target_path.
    Prevents corrupt/truncated files if training is interrupted or if disk is slow.
    """
    tmp_path = f"{target_path}.tmp"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        torch.save(obj, tmp_path)
        if os.path.exists(target_path):
            try:
                os.replace(tmp_path, target_path)
            except OSError:
                os.remove(target_path)
                os.rename(tmp_path, target_path)
        else:
            os.rename(tmp_path, target_path)
        return True
    except Exception as exc:
        logger.error(f"[Checkpoint Save Error] Could not safely write '{target_path}': {exc}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return False


class MultimodalTrainer:
    """
    Unified Trainer class for Pure End-to-End Multimodal Fish Feeding Intensity Classification
    with Dual Auxiliary Heads and CosineAnnealingLR.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        test_loader: Any,
        config: TrainConfig,
        device: torch.device,
        optimizer: Optional[optim.Optimizer] = None,
        train_config_path: str = 'config/train_config.json'
    ) -> None:
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.train_config_path = train_config_path

        # Setup Loss with 2 Auxiliary Heads Supervision (Deep Supervision)
        loss_type = getattr(self.config, "loss_type", "pairwise_tournament")
        self.aux_loss_weight = getattr(self.config, "aux_loss_weight", 0.3)
        if loss_type == "clip_ce":
            self.loss_fn = ClipCELoss()
            logger.info("Configured standard ClipCELoss.")
        else:
            self.loss_fn = PairwiseTournamentLoss(
                weight_act=getattr(self.config, "weight_act", 0.5),
                weight_pairwise=getattr(self.config, "weight_pairwise", 0.5),
                weight_ce=getattr(self.config, "weight_ce", 1.0),
                aux_loss_weight=self.aux_loss_weight,
            ).to(self.device)
            logger.info(f"Configured PairwiseTournamentLoss with Deep Supervision aux heads (aux_loss_weight={self.aux_loss_weight}).")

        # Training Setup
        self.weight_decay = getattr(self.config, "weight_decay", 0.05)
        self.steps_per_epoch = max(1, len(self.train_loader))
        self.lr_scheduler_type = str(getattr(self.config, "lr_scheduler", "cosine")).lower()
        self.lr_step_mode = str(getattr(self.config, "lr_step_mode", "epoch")).lower()
        self.warmup_pct = float(getattr(self.config, "warmup_pct", 0.05))

        # Single unified optimizer for all parameters (Backbones + Aux Heads + Multimodal Fusion)
        self.optimizer = optimizer if optimizer is not None else optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.weight_decay
        )

        min_lr = float(getattr(self.config, "min_lr", 1e-8))
        base_lr = float(self.config.learning_rate)

        if self.lr_scheduler_type == "onecycle":
            total_steps = (self.config.epochs * self.steps_per_epoch) if self.lr_step_mode == "batch" else self.config.epochs
            div_factor = 25.0
            final_div_factor = max(1.0, (base_lr / max(min_lr, 1e-12)) / div_factor)
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=base_lr,
                total_steps=total_steps,
                pct_start=self.warmup_pct,
                div_factor=div_factor,
                final_div_factor=final_div_factor
            )
            logger.info(
                f"Configured OneCycleLR: total_steps={total_steps} (step_mode='{self.lr_step_mode}'), "
                f"max_lr={base_lr}, min_lr={min_lr}, warmup_pct={self.warmup_pct*100:.1f}%."
            )
        else:
            # Default: CosineAnnealingLR
            T_max = (self.config.epochs * self.steps_per_epoch) if self.lr_step_mode == "batch" else self.config.epochs
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=min_lr
            )
            logger.info(
                f"Configured CosineAnnealingLR: T_max={T_max} (step_mode='{self.lr_step_mode}'), "
                f"base_lr={base_lr}, eta_min={min_lr}."
            )


        # Evaluator and Timer
        self.evaluator = MultimodalEvaluator(model=self.model, loss_fn=self.loss_fn)
        self.timer = InferenceTimer(model=self.model, device=self.device)

        # Early stopping setup
        self.early_stopping = EarlyStopping(
            patience=getattr(self.config, "patience", 80),
            delta=getattr(self.config, "delta", 0.0),
            verbose=True
        ) if getattr(self.config, "early_stopping", False) else None

        self._init_logging_and_checkpoints()

    def _init_logging_and_checkpoints(self) -> None:
        model_name = getattr(self.model, "model_name", self.model.__class__.__name__.lower())
        self.run_dir = os.path.join(self.config.ckpt_dir, model_name)

        if self.config.dataset_splitter.evaluation_mode == "cross_validation" and self.config.dataset_splitter.fold_index is not None:
            self.run_dir = os.path.join(self.run_dir, f"fold_{self.config.dataset_splitter.fold_index}")

        os.makedirs(self.run_dir, exist_ok=True)
        self.logger = HistoryLogger(log_dir=self.run_dir)
        self.best_checkpoint_path = os.path.join(self.run_dir, 'best_model.pth')
        self.best_video_path = os.path.join(self.run_dir, 'best_video_backbone.pth')
        self.best_audio_path = os.path.join(self.run_dir, 'best_audio_backbone.pth')
        self.last_checkpoint_path = os.path.join(self.run_dir, 'last_model.pth')

        # Save actual runtime train_config.json copy to checkpoint directory
        try:
            cfg_copy_path = os.path.join(self.run_dir, 'train_config.json')
            if self.train_config_path and os.path.exists(self.train_config_path):
                shutil.copy2(self.train_config_path, cfg_copy_path)
            elif hasattr(self.config, 'model_dump'):
                with open(cfg_copy_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config.model_dump(), f, indent=2)
            logger.info(f"Saved runtime training configuration copy to: '{cfg_copy_path}'")
        except Exception as exc:
            logger.warning(f"Could not save copy of config into checkpoint dir: {exc}")

        # Profile model parameters and inference complexity (GFLOPs)
        self.param_stats = count_parameters(self.model)
        num_frames = getattr(self.config, "num_frames", 2)
        image_size = getattr(self.config, "image_size", 224)
        sample_rate = getattr(self.config.audio_features, "sample_rate", 256000) if hasattr(self.config, "audio_features") else 256000
        audio_samples = int(sample_rate * 2)
        self.model_flops = measure_flops(
            self.model,
            device=self.device,
            num_frames=num_frames,
            image_size=image_size,
            audio_samples=audio_samples
        )

        logger.info("==================================================")
        logger.info("INITIALIZED MULTIMODAL TRAINING EXPERIMENT:")
        logger.info(f"  - Model Architecture:       {model_name}")
        logger.info(f"  - Trainable Parameters:     {self.param_stats['total']:,} ({self.param_stats['total_million']:.3f} M)")
        logger.info(f"    * Video Backbone:         {self.param_stats['video_backbone']:,} ({self.param_stats['video_backbone']/1e6:.3f} M)")
        logger.info(f"    * Audio Backbone:         {self.param_stats['audio_backbone']:,} ({self.param_stats['audio_backbone']/1e6:.3f} M)")
        logger.info(f"    * Tournament Fusion:      {self.param_stats['fusion']:,} ({self.param_stats['fusion']/1e6:.3f} M)")
        if self.model_flops > 0.0:
            logger.info(f"  - Inference Complexity:     {self.model_flops:.3f} GFLOPs (1 sample: {num_frames} frames @ {image_size}x{image_size}, {audio_samples:,} audio samples)")
        logger.info(f"  - Device:                   {self.device}")
        logger.info(f"  - Max Epochs:               {self.config.epochs}")
        logger.info(f"  - Batch Size:               {self.config.batch_size}")
        logger.info(f"  - Learning Rate:            {self.config.learning_rate}")
        lr_sched_name = getattr(self, "lr_scheduler_type", getattr(self.config, "lr_scheduler", "cosine"))
        step_mode_name = getattr(self, "lr_step_mode", "epoch")
        logger.info(f"  - LR Scheduler:             {lr_sched_name} (step_mode='{step_mode_name}', min_lr={getattr(self.config, 'min_lr', 1e-8)})")
        logger.info(f"  - Auxiliary Supervision:    aux_loss_weight = {self.aux_loss_weight} (Video & Audio Aux Heads)")
        logger.info(f"  - Training Strategy:        Pure End-to-End (Unified Optimizer, No Two-Phase)")
        logger.info(f"  - Monitor Metric:           {self.config.monitor} (Default: Validation Accuracy)")
        logger.info(f"  - Early Stopping:           {getattr(self.config, 'early_stopping', False)}")
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

            try:
                loss = self.loss_fn(outputs, {'target': targets}, epoch=epoch)
            except TypeError:
                loss = self.loss_fn(outputs, {'target': targets})
            loss.backward()

            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            self.optimizer.step()

            if self.lr_step_mode == "batch":
                self.scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            logits = outputs['clipwise_output']
            train_preds.append(logits.detach().cpu().numpy())
            train_targets.append(targets.detach().cpu().numpy())

            pbar.set_postfix({
                'loss': f"{loss_val:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.1e}"
            })

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
        monitor_metric = str(getattr(self.config, 'monitor', 'val_acc')).lower()
        logger.info(f"Starting training pipeline (Monitor metric: {monitor_metric})...")
        training_start_time = time.perf_counter()

        best_acc = 0.0
        best_qwk = -1.0
        best_mAP = 0.0
        best_loss = float('inf')
        best_epoch = 1
        best_val_statistics = None

        if monitor_metric == 'loss':
            best_val_metric = float('inf')
        else:
            best_val_metric = -1.0

        for epoch in range(1, self.config.epochs + 1):
            epoch_start_time = time.perf_counter()
            train_loss, train_acc, train_mAP = self._train_epoch(epoch)
            if self.lr_step_mode == "epoch":
                self.scheduler.step()

            # Evaluate on validation split
            self.model.eval()
            val_stats = self.evaluator.evaluate(self.val_loader)
            val_loss = float(val_stats.get('loss', 0.0))
            val_acc = float(np.mean(val_stats['accuracy']))
            val_qwk = float(val_stats.get('qwk', 0.0))
            val_mAP = float(np.mean(val_stats['average_precision']))
            val_acc_v = float(val_stats.get('acc_video', 0.0))
            val_acc_a = float(val_stats.get('acc_audio', 0.0))

            epoch_time = time.perf_counter() - epoch_start_time

            # Extract current learning rate
            lr_current = self.optimizer.param_groups[0]['lr']
            lr_info = f"LR = {lr_current:.2e}"

            logger.info(
                f"Epoch {epoch:03d} ({epoch_time:.1f}s): "
                f"Train Loss = {train_loss:.5f} | Train Acc = {train_acc:.4f} | {lr_info} | "
                f"Val Loss = {val_loss:.5f} | Val Acc = {val_acc:.4f} | Val QWK = {val_qwk:.4f} | "
                f"(Aux Video Acc = {val_acc_v:.4f}, Aux Audio Acc = {val_acc_a:.4f})"
            )

            # Track and save best overall Multimodal checkpoint
            is_best = False
            if monitor_metric == 'loss':
                score = -val_loss
                if val_loss < best_val_metric:
                    best_val_metric = val_loss
                    is_best = True
            elif monitor_metric == 'qwk':
                score = val_qwk
                if val_qwk > best_val_metric:
                    best_val_metric = val_qwk
                    is_best = True
            else:
                # Default & primary monitor: Validation Accuracy (val_acc / accuracy)
                score = val_acc
                if val_acc > best_val_metric:
                    best_val_metric = val_acc
                    is_best = True

            if is_best:
                best_epoch = epoch
                best_acc = val_acc
                best_qwk = val_qwk
                best_mAP = val_mAP
                best_loss = val_loss
                best_val_statistics = val_stats

                # Single unified state_dict snapshot (eliminates ~120 redundant clone calls)
                current_state_dict = self.model.state_dict()

                # 1. Save full Multimodal model
                _safe_torch_save(current_state_dict, self.best_checkpoint_path)

                # 2. Extract & save Video Backbone + Aux Head Video
                v_dict = {k: v for k, v in current_state_dict.items() if k.startswith('video_backbone.') or k.startswith('aux_head_video.')}
                if v_dict:
                    _safe_torch_save(v_dict, self.best_video_path)

                # 3. Extract & save Audio Backbone + Audio Frontend + Aux Head Audio
                a_dict = {k: v for k, v in current_state_dict.items() if k.startswith('audio_backbone.') or k.startswith('audio_frontend.') or k.startswith('aux_head_audio.')}
                if a_dict:
                    _safe_torch_save(a_dict, self.best_audio_path)

                logger.info(f"[*] New best validation performance! Saved checkpoints:")
                logger.info(f"    - Full Multimodal Model:   '{self.best_checkpoint_path}' (Val Acc = {best_acc:.4f}, Val QWK = {best_qwk:.4f})")
                logger.info(f"    - Peak Video Backbone:     '{self.best_video_path}'")
                logger.info(f"    - Peak Audio Backbone:     '{self.best_audio_path}'")

            logger.info(
                f"Current best: Epoch {best_epoch:03d} | Val Acc: {best_acc:.4f} | Val QWK: {best_qwk:.4f} (Loss: {best_loss:.5f})"
            )

            # Always save last checkpoint with full resumption state
            resumption_checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_epoch': best_epoch,
                'best_val_metric': best_val_metric,
                'best_acc': best_acc,
                'best_qwk': best_qwk,
                'val_statistics': val_stats
            }
            _safe_torch_save(resumption_checkpoint, self.last_checkpoint_path)

            # Log to history CSV
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                train_mAP=train_mAP,
                val_loss=val_loss,
                val_statistics=val_stats,
                lr=lr_current,
                epoch_time_seconds=epoch_time,
                is_best=is_best
            )

            # Early stopping check (only if enabled)
            if self.early_stopping is not None:
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
            try:
                state_dict = torch.load(self.best_checkpoint_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                logger.info(f"Reloaded best checkpoint '{self.best_checkpoint_path}' from Epoch {best_epoch:03d}...")
            except Exception as exc:
                logger.warning(f"Failed to load checkpoint with weights_only=True ({exc}), attempting with weights_only=False...")
                try:
                    state_dict = torch.load(self.best_checkpoint_path, map_location=self.device, weights_only=False)
                    self.model.load_state_dict(state_dict)
                    logger.info(f"Reloaded best checkpoint '{self.best_checkpoint_path}' from Epoch {best_epoch:03d}...")
                except Exception as exc2:
                    logger.error(f"Could not reload best checkpoint: {exc2}. Proceeding with current in-memory model weights.")
        else:
            logger.warning(f"Best checkpoint '{self.best_checkpoint_path}' not found. Evaluating in-memory model weights.")

        self.model.eval()
        final_val_stats = best_val_statistics if best_val_statistics is not None else self.evaluator.evaluate(self.val_loader)
        final_test_stats = self.evaluator.evaluate(self.test_loader)

        test_acc = float(np.mean(final_test_stats['accuracy']))
        test_qwk = float(final_test_stats.get('qwk', 0.0))
        test_mAP = float(np.mean(final_test_stats['average_precision']))
        logger.info(f"TEST Results -> Accuracy: {test_acc:.4f} | QWK: {test_qwk:.4f} | mAP: {test_mAP:.4f}")
        logger.info(f"Detailed Classification Report:\n{final_test_stats.get('message', '')}")
        if 'confu_matrix' in final_test_stats:
            logger.info(f"Confusion Matrix:\n{final_test_stats['confu_matrix']}")

        # Measure inference latency
        logger.info("Measuring model Inference Latency on device...")
        try:
            num_frames = getattr(self.config, "num_frames", 2)
            image_size = getattr(self.config, "image_size", 224)
            sample_rate = getattr(getattr(self.config, "audio_features", None), "sample_rate", 256000)
            audio_samples = int(sample_rate * 2)
            inference_latency_ms = self.timer.measure_latency_per_sample(
                video_shape=(1, num_frames, 3, image_size, image_size),
                audio_shape=(1, audio_samples)
            )
        except Exception as exc:
            logger.warning(f"Inference latency measurement failed: {exc}. Defaulting to 0.0 ms.")
            inference_latency_ms = 0.0

        # Save summary report
        try:
            self.logger.save_summary(
                training_time=training_duration,
                inference_time_ms=inference_latency_ms,
                val_statistics=final_val_stats,
                test_statistics=final_test_stats,
                total_params_m=self.param_stats.get('total_million', 0.0),
                gflops=self.model_flops
            )
        except Exception as exc:
            logger.error(f"Failed to export summary report: {exc}")

        # Export consolidated detailed evaluation report (.txt and .json)
        try:
            self.logger.save_detailed_evaluation_report(
                val_statistics=final_val_stats,
                test_statistics=final_test_stats
            )
        except Exception as exc:
            logger.warning(f"Could not export detailed evaluation report: {exc}")

        # Plot test confusion matrices comparison heatmap (.png)
        try:
            self.logger.plot_test_confusion_matrices(
                test_statistics=final_test_stats,
                val_statistics=final_val_stats
            )
        except Exception as exc:
            logger.warning(f"Could not plot test confusion matrices: {exc}")

        return {
            'training_time': training_duration,
            'inference_time_ms': inference_latency_ms,
            'val_statistics': final_val_stats,
            'test_statistics': final_test_stats
        }
