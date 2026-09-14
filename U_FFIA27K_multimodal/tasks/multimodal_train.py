import os
import sys
import time
import logging
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
from utils import (
    EarlyStopping,
    HistoryLogger,
    MultimodalEvaluator,
    InferenceTimer,
    ClipCELoss,
    PairwiseTournamentLoss,
    count_parameters,
    measure_flops,
)

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
        return False


class MultimodalTrainer:
    """
    Unified Trainer class for Multimodal Fish Feeding Intensity Classification.
    Supports Bilateral Boundary Loss, OneCycleLR, QWK monitoring, and Nelder-Mead post-calibration.
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

        # Setup Loss
        loss_type = getattr(self.config, "loss_type", "pairwise_tournament")
        if loss_type == "pairwise_tournament":
            self.loss_fn = PairwiseTournamentLoss(
                weight_act=getattr(self.config, "weight_act", 0.5),
                weight_pairwise=getattr(self.config, "weight_pairwise", 0.5),
                weight_ce=getattr(self.config, "weight_ce", 1.0),
                aux_loss_weight=getattr(self.config, "aux_loss_weight", 0.3),
            ).to(self.device)
            logger.info("Configured PairwiseTournamentLoss (Activity Gate + 3 Pairwise Cross Boundaries B12, B23, B13).")
        else:
            self.loss_fn = ClipCELoss()
            logger.info("Configured standard ClipCELoss.")

        # Training & Optimization settings
        self.weight_decay = getattr(self.config, "weight_decay", 0.05)
        self.steps_per_epoch = max(1, len(self.train_loader))
        self.use_onecycle = getattr(self.config, "use_onecycle", True)

        # Optimizer: Authentic AdamW across full model
        self.optimizer = optimizer if optimizer is not None else optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.weight_decay
        )

        # LR Scheduler: OneCycleLR with cosine annealing
        if self.use_onecycle:
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.epochs,
                steps_per_epoch=self.steps_per_epoch,
                pct_start=0.05,
                anneal_strategy='cos',
                div_factor=25,
                final_div_factor=1000
            )
            logger.info(f"Configured OneCycleLR: max_lr={self.config.learning_rate}, epochs={self.config.epochs}, pct_start=0.05.")
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=1e-6
            )
            logger.info(f"Configured CosineAnnealingLR: T_max={self.config.epochs}, eta_min=1e-6.")

        # Evaluator and Timer
        self.evaluator = MultimodalEvaluator(model=self.model, loss_fn=self.loss_fn)
        self.timer = InferenceTimer(model=self.model, device=self.device)

        # Early stopping setup
        self.early_stopping = EarlyStopping(
            patience=getattr(self.config, "patience", 80),
            delta=getattr(self.config, "min_delta", getattr(self.config, "delta", 0.0)),
            verbose=True
        ) if getattr(self.config, "early_stopping", False) else None

        self._init_logging_and_checkpoints()

    def _init_logging_and_checkpoints(self) -> None:
        model_name = getattr(self.config.model, "backbone", getattr(self.model, "model_name", self.model.__class__.__name__))
        self.run_dir = os.path.join(self.config.ckpt_dir, model_name)

        if self.config.dataset_splitter.evaluation_mode == "cross_validation" and self.config.dataset_splitter.fold_index is not None:
            self.run_dir = os.path.join(self.run_dir, f"fold_{self.config.dataset_splitter.fold_index}")

        os.makedirs(self.run_dir, exist_ok=True)
        self.logger = HistoryLogger(log_dir=self.run_dir)
        self.best_checkpoint_path = os.path.join(self.run_dir, 'best_model.pth')
        self.best_video_path = os.path.join(self.run_dir, 'best_video_backbone.pth')
        self.best_audio_path = os.path.join(self.run_dir, 'best_audio_backbone.pth')
        self.last_checkpoint_path = os.path.join(self.run_dir, 'last_model.pth')

        logger.info("==================================================")
        logger.info("INITIALIZED MULTIMODAL TRAINING EXPERIMENT:")
        logger.info(f"  - Model Architecture:       {model_name}")
        logger.info(f"  - Device:                   {self.device}")
        logger.info(f"  - Max Epochs:               {self.config.epochs}")
        logger.info(f"  - Batch Size:               {self.config.batch_size}")
        logger.info(f"  - Learning Rate:            {self.config.learning_rate}")
        logger.info(f"  - Weight Decay:             {self.weight_decay}")
        logger.info(f"  - Monitor Metric:           {self.config.monitor}")
        logger.info(f"  - Early Stopping:           {getattr(self.config, 'early_stopping', False)}")
        logger.info(f"  - Checkpoint Run Dir:       '{self.run_dir}'")
        logger.info("==================================================")

    def _train_epoch(self, epoch: int) -> Tuple[float, float, float, float]:
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

            if self.use_onecycle:
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

        rank_map = np.array([0, 3, 2, 1])
        try:
            train_mae = float(np.mean(np.abs(rank_map[pred_acc_labels] - rank_map[target_acc_labels])))
        except Exception:
            train_mae = 0.0

        try:
            from sklearn import metrics as sklearn_metrics
            train_mAP = float(np.mean(sklearn_metrics.average_precision_score(train_targets, train_preds, average=None)))
        except Exception:
            train_mAP = train_acc

        return epoch_loss, train_acc, train_mAP, train_mae

    def train(self) -> Dict[str, Any]:
        logger.info(f"Starting training pipeline (Monitor metric: {self.config.monitor})...")
        training_start_time = time.perf_counter()

        best_acc = 0.0
        best_qwk = -1.0
        best_mAP = 0.0
        best_loss = float('inf')
        best_epoch = 1
        best_val_statistics = None

        if self.config.monitor in ('accuracy', 'qwk'):
            best_val_metric = -1.0
        else:
            best_val_metric = float('inf')

        for epoch in range(1, self.config.epochs + 1):
            epoch_start_time = time.perf_counter()
            train_loss, train_acc, train_mAP, train_mae = self._train_epoch(epoch)
            if not self.use_onecycle:
                self.scheduler.step()

            # Evaluate on validation split
            self.model.eval()
            val_stats = self.evaluator.evaluate(self.val_loader)
            val_loss = float(val_stats.get('loss', 0.0))
            val_acc = float(np.mean(val_stats['accuracy']))
            val_qwk = float(val_stats.get('qwk', 0.0))
            val_mAP = float(np.mean(val_stats['average_precision']))
            val_mae = float(val_stats.get('ordinal_mae', 0.0))
            val_acc_v = float(val_stats.get('acc_video', 0.0))
            val_acc_a = float(val_stats.get('acc_audio', 0.0))

            lr_current = self.optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch:03d} [TOURNAMENT E2E]: "
                f"Train Loss = {train_loss:.5f} | Train Acc = {train_acc:.4f} | Train MAE = {train_mae:.4f} | LR = {lr_current:.2e} | "
                f"Val Loss = {val_loss:.5f} | Val Acc Video = {val_acc_v:.4f} | Val Acc Audio = {val_acc_a:.4f} | "
                f"Val Acc Fusion = {val_acc:.4f} | Val QWK = {val_qwk:.4f} | Val MAE = {val_mae:.4f}"
            )

            # Determine if this is the best checkpoint
            is_best = False
            if self.config.monitor == 'qwk':
                score = val_qwk
                if val_qwk > best_val_metric:
                    best_val_metric = val_qwk
                    is_best = True
            elif self.config.monitor == 'accuracy':
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
                best_qwk = val_qwk
                best_mAP = val_mAP
                best_loss = val_loss
                best_val_statistics = val_stats

                current_state_dict = self.model.state_dict()

                # 1. Full Multimodal Model
                _safe_torch_save(current_state_dict, self.best_checkpoint_path)

                # 2. Peak Video Backbone
                v_dict = {k: v for k, v in current_state_dict.items() if k.startswith('video_backbone.') or k.startswith('aux_head_video.')}
                if v_dict:
                    _safe_torch_save(v_dict, self.best_video_path)

                # 3. Peak Audio Backbone
                a_dict = {k: v for k, v in current_state_dict.items() if k.startswith('audio_backbone.') or k.startswith('audio_frontend.') or k.startswith('aux_head_audio.')}
                if a_dict:
                    _safe_torch_save(a_dict, self.best_audio_path)

                logger.info(f"[*] New best validation performance! Saved checkpoints:")
                logger.info(f"    - Full Multimodal Model:   '{self.best_checkpoint_path}' (Val Acc = {best_acc:.4f}, Val QWK = {best_qwk:.4f})")
                logger.info(f"    - Peak Video Backbone:     '{self.best_video_path}'")
                logger.info(f"    - Peak Audio Backbone:     '{self.best_audio_path}'")

            logger.info(
                f"Current best: Epoch {best_epoch:03d} | Loss: {best_loss:.5f} | Accuracy: {best_acc:.4f} | QWK: {best_qwk:.4f} | mAP: {best_mAP:.4f}"
            )

            # 4. Always save last checkpoint with full resumption state
            resumption_checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                'best_epoch': best_epoch,
                'best_val_metric': best_val_metric,
                'best_acc': best_acc,
                'best_qwk': best_qwk,
                'val_statistics': val_stats
            }
            _safe_torch_save(resumption_checkpoint, self.last_checkpoint_path)

            epoch_time_seconds = time.perf_counter() - epoch_start_time

            # Log to history CSV
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                train_mAP=train_mAP,
                val_loss=val_loss,
                val_statistics=val_stats,
                lr=lr_current,
                epoch_time_seconds=epoch_time_seconds,
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

        # Measure model complexity, FLOPs, and inference latency
        logger.info("Profiling model Parameters, FLOPs, and Inference Latency...")
        param_stats = count_parameters(self.model)
        total_params_m = float(param_stats.get('total_million', 0.0))
        try:
            device_str = self.device.type
            gflops = measure_flops(
                self.model,
                device=device_str,
                num_frames=getattr(self.config, "num_frames", 2)
            )
            logger.info(f"Model Parameters: {total_params_m:.3f} M | Complexity: {gflops:.4f} GFLOPs")
        except Exception as exc:
            logger.warning(f"FLOPs measurement failed: {exc}. Defaulting to 0.0.")
            gflops = 0.0

        try:
            inference_latency_ms = self.timer.measure_latency_per_sample(
                video_shape=(1, getattr(self.config, "num_frames", 2), 3, getattr(self.config, "image_size", 224), getattr(self.config, "image_size", 224)),
                audio_shape=(1, 512000)
            )
        except Exception as exc:
            logger.warning(f"Inference latency measurement failed: {exc}. Defaulting to 0.0 ms.")
            inference_latency_ms = 0.0

        # Save summary report
        self.logger.save_summary(
            training_time=training_duration,
            inference_time_ms=inference_latency_ms,
            val_statistics=final_val_stats,
            test_statistics=final_test_stats,
            total_params_m=total_params_m,
            gflops=gflops
        )

        # Export consolidated detailed evaluation report (.txt and .json)
        try:
            self.logger.save_detailed_evaluation_report(
                val_statistics=final_val_stats,
                test_statistics=final_test_stats,
                total_params_m=total_params_m,
                gflops=gflops
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
