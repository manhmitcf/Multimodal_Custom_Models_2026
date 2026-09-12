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
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score

from config import MultimodalTrainConfig
from utils import (
    EarlyStopping,
    HistoryLogger,
    MultimodalEvaluator,
    InferenceTimer,
    ClipCELoss,
    BilateralBoundaryLoss,
    PairwiseTournamentLoss,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_flat_cosine_scheduler(
    optimizer: optim.Optimizer,
    total_steps: int,
    flat_steps: int,
    min_lr_ratio: float = 1e-3
) -> optim.lr_scheduler.LambdaLR:
    """
    Creates a Flat-Cosine LR Scheduler:
    - Step 0 -> flat_steps: LR stays constant at initial LR (multiplier 1.0).
    - flat_steps -> total_steps: LR decays via cosine curve down to min_lr_ratio.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < flat_steps:
            return 1.0
        decay_steps = max(1, total_steps - flat_steps)
        progress = min(1.0, max(0.0, (current_step - flat_steps) / decay_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


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
        if loss_type == "pairwise_tournament":
            self.loss_fn = PairwiseTournamentLoss(
                weight_act=getattr(self.config, "weight_act", 0.5),
                weight_pairwise=getattr(self.config, "weight_pairwise", 0.5),
                weight_ce=getattr(self.config, "weight_ce", 1.0),
                aux_loss_weight=self.aux_loss_weight,
            ).to(self.device)
            logger.info(f"Configured PairwiseTournamentLoss with Deep Supervision aux heads (aux_loss_weight={self.aux_loss_weight}).")
        elif loss_type in ("bilateral_boundary", "ordinal_wasserstein"):
            self.loss_fn = BilateralBoundaryLoss(
                lambda_emd=getattr(self.config, "lambda_emd", 0.5),
                lambda_align=getattr(self.config, "lambda_align", 0.2),
                aux_loss_weight=self.aux_loss_weight,
            ).to(self.device)
            logger.info("Configured BilateralBoundaryLoss.")
        else:
            self.loss_fn = ClipCELoss()
            logger.info("Configured standard ClipCELoss.")

        # Unified End-to-End Training Setup (No Two-Phase Warmup)
        self.enable_two_phase_warmup = False
        self.phase1_warmup_epochs = 0
        self.current_phase = 2
        self.weight_decay = getattr(self.config, "weight_decay", 0.05)
        self.steps_per_epoch = max(1, len(self.train_loader))
        self.lr_scheduler_type = getattr(self.config, "lr_scheduler", "cosine")
        self.use_onecycle = getattr(self.config, "use_onecycle", False) or (self.lr_scheduler_type == "onecycle")
        self.is_stepwise_scheduler = (self.lr_scheduler_type in ("flat_cosine", "onecycle")) or self.use_onecycle

        # Single unified optimizer for all parameters (Backbones + Aux Heads + Multimodal Fusion)
        self.optimizer = optimizer if optimizer is not None else optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.weight_decay
        )

        min_lr = getattr(self.config, "min_lr", 1e-8)
        base_lr = self.config.learning_rate

        if self.lr_scheduler_type == "flat_cosine":
            total_steps = self.config.epochs * self.steps_per_epoch
            flat_steps = int(getattr(self.config, "flat_pct", 0.05) * total_steps)
            min_lr_ratio = min_lr / base_lr if base_lr > 0 else 1e-3
            self.scheduler = create_flat_cosine_scheduler(
                self.optimizer,
                total_steps=total_steps,
                flat_steps=flat_steps,
                min_lr_ratio=min_lr_ratio
            )
            logger.info(f"Configured Flat-Cosine Scheduler: total_steps={total_steps}, flat_steps={flat_steps}, lr={base_lr} -> min_lr={min_lr}.")
        elif self.use_onecycle:
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
            logger.info(f"Configured OneCycleLR: max_lr={self.config.learning_rate}, epochs={self.config.epochs}.")
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=min_lr
            )
            logger.info(f"Configured CosineAnnealingLR: T_max={self.config.epochs}, eta_min={min_lr}.")

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
        self.phase1_checkpoint_path = os.path.join(self.run_dir, 'best_phase1_backbone.pth')
        self.last_checkpoint_path = os.path.join(self.run_dir, 'last_model.pth')

        logger.info("==================================================")
        logger.info("INITIALIZED MULTIMODAL TRAINING EXPERIMENT:")
        logger.info(f"  - Model Architecture:       {model_name}")
        logger.info(f"  - Device:                   {self.device}")
        logger.info(f"  - Max Epochs:               {self.config.epochs}")
        logger.info(f"  - Batch Size:               {self.config.batch_size}")
        logger.info(f"  - Learning Rate:            {self.config.learning_rate}")
        logger.info(f"  - LR Scheduler:             {self.lr_scheduler_type} (T_max={self.config.epochs}, min_lr={getattr(self.config, 'min_lr', 1e-8)})")
        logger.info(f"  - Auxiliary Supervision:    aux_loss_weight = {self.aux_loss_weight} (Video & Audio Aux Heads)")
        logger.info(f"  - Training Strategy:        Pure End-to-End (Unified Optimizer, No Two-Phase)")
        logger.info(f"  - Monitor Metric:           {self.config.monitor} (Validation Accuracy)")
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

            if self.is_stepwise_scheduler:
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

    def nelder_mead_calibrate(self) -> Dict[str, Any]:
        """
        Nelder-Mead Post-Calibration on Validation split to optimize cutoffs without gradients.
        """
        if not hasattr(self.model.fusion, 'head_v') or not hasattr(self.model.fusion.head_v, 'get_cutoffs'):
            logger.info("Tournament architecture uses direct Pairwise Voting (no 1D cutoffs needed). Skipping Nelder-Mead.")
            return {}

        logger.info("Starting Nelder-Mead post-training calibration on validation set...")
        self.model.eval()

        s_v_list, s_a_list, g_list, y_list = [], [], [], []
        with torch.no_grad():
            for batch in self.val_loader:
                video = batch['video_form'].to(self.device)
                audio = batch['audio_form'].to(self.device)
                targets = batch['target'].to(self.device)
                outputs = self.model(video, audio)

                s_v_list.append(outputs['score_v'].cpu().numpy())
                s_a_list.append(outputs['score_a'].cpu().numpy())
                g_list.append(outputs['gate'].squeeze(-1).cpu().numpy())
                y_raw = targets.argmax(dim=-1) if targets.ndim > 1 else targets
                y_list.append(y_raw.cpu().numpy())

        s_v = np.concatenate(s_v_list)
        s_a = np.concatenate(s_a_list)
        g = np.concatenate(g_list)
        y = np.concatenate(y_list)

        raw_to_rank = np.array([0, 3, 2, 1])
        y_rank = raw_to_rank[y]

        b_v_init = [c.item() for c in self.model.fusion.head_v.get_cutoffs()]
        b_a_init = [c.item() for c in self.model.fusion.head_a.get_cutoffs()]
        x0 = np.array(b_v_init + b_a_init, dtype=float)

        def objective(params):
            bv = params[:3]
            ba = params[3:]
            if bv[1] <= bv[0] + 0.1 or bv[2] <= bv[1] + 0.1:
                return 10.0
            if ba[1] <= ba[0] + 0.1 or ba[2] <= ba[1] + 0.1:
                return 10.0

            s = g * s_v + (1.0 - g) * s_a
            b1 = g * bv[0] + (1.0 - g) * ba[0]
            b2 = g * bv[1] + (1.0 - g) * ba[1]
            b3 = g * bv[2] + (1.0 - g) * ba[2]

            pred_rank = np.zeros_like(s, dtype=int)
            pred_rank[s >= b1] = 1
            pred_rank[s >= b2] = 2
            pred_rank[s >= b3] = 3

            qwk = cohen_kappa_score(y_rank, pred_rank, weights='quadratic')
            return -float(qwk)

        init_qwk = -objective(x0)
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-3})
        opt_qwk = -res.fun

        logger.info(f"Nelder-Mead Calibration: Initial Val QWK = {init_qwk:.4f} -> Calibrated Val QWK = {opt_qwk:.4f} (+{opt_qwk - init_qwk:.4f})")

        calibrated_cutoffs = {
            "initial_cutoffs_v": b_v_init,
            "initial_cutoffs_a": b_a_init,
            "calibrated_cutoffs_v": [float(x) for x in res.x[:3]],
            "calibrated_cutoffs_a": [float(x) for x in res.x[3:]],
            "initial_qwk": float(init_qwk),
            "calibrated_qwk": float(opt_qwk),
        }

        calibrated_path = os.path.join(self.run_dir, 'calibrated_cutoffs.json')
        with open(calibrated_path, 'w', encoding='utf-8') as f:
            json.dump(calibrated_cutoffs, f, indent=2)
        logger.info(f"Saved calibrated cutoffs to: '{calibrated_path}'")

        # Update model cutoffs with calibrated values
        try:
            self.model.fusion.head_v.set_cutoffs(tuple(res.x[:3]))
            self.model.fusion.head_a.set_cutoffs(tuple(res.x[3:]))
            logger.info("Updated model boundary heads with calibrated cutoffs.")
        except Exception as exc:
            logger.warning(f"Could not directly update head cutoffs: {exc}")

        return calibrated_cutoffs

    def train(self) -> Dict[str, Any]:
        monitor_metric = str(getattr(self.config, 'monitor', 'val_acc')).lower()
        logger.info(f"Starting training pipeline (Monitor metric: {monitor_metric} [Validation Accuracy])...")
        training_start_time = time.perf_counter()

        best_acc = 0.0
        best_mAP = 0.0
        best_loss = float('inf')
        best_epoch = 1
        best_val_statistics = None

        if monitor_metric == 'loss':
            best_val_metric = float('inf')
        else:
            best_val_metric = -1.0

        best_val_video_acc = -1.0
        best_val_audio_acc = -1.0
        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc, train_mAP, train_mae = self._train_epoch(epoch)
            if not self.is_stepwise_scheduler:
                self.scheduler.step()

            # Evaluate on validation split
            self.model.eval()
            val_stats = self.evaluator.evaluate(self.val_loader)
            val_loss = float(val_stats.get('loss', 0.0))
            val_acc = float(np.mean(val_stats['accuracy']))
            val_mAP = float(np.mean(val_stats['average_precision']))
            val_mae = float(val_stats.get('ordinal_mae', 0.0))
            val_acc_v = float(val_stats.get('acc_video', 0.0))
            val_acc_a = float(val_stats.get('acc_audio', 0.0))

            # Extract current learning rate
            lr_current = self.optimizer.param_groups[0]['lr']
            lr_info = f"LR = {lr_current:.2e}"

            logger.info(
                f"Epoch {epoch:03d} [END-TO-END + AUX HEADS]: "
                f"Train Loss = {train_loss:.5f} | Train Acc = {train_acc:.4f} | Train MAE = {train_mae:.4f} | {lr_info} | "
                f"Val Loss = {val_loss:.5f} | Val Acc Video = {val_acc_v:.4f} (Peak: {max(best_val_video_acc, val_acc_v):.4f}) | "
                f"Val Acc Audio = {val_acc_a:.4f} (Peak: {max(best_val_audio_acc, val_acc_a):.4f}) | "
                f"Val Acc Fusion = {val_acc:.4f} | Val MAE = {val_mae:.4f}"
            )

            # 1. Track and save PEAK Video Backbone independently
            if val_acc_v > best_val_video_acc:
                best_val_video_acc = val_acc_v
                v_keys = [k for k in self.model.state_dict().keys() if k.startswith('video_backbone.') or k.startswith('aux_head_video.')]
                torch.save({k: self.model.state_dict()[k] for k in v_keys}, self.best_video_path)
                logger.info(f"[*] New PEAK Video Backbone! Saved: '{self.best_video_path}' (Val Acc = {best_val_video_acc:.4f})")

            # 2. Track and save PEAK Audio Backbone independently
            if val_acc_a > best_val_audio_acc:
                best_val_audio_acc = val_acc_a
                a_keys = [k for k in self.model.state_dict().keys() if k.startswith('audio_backbone.') or k.startswith('audio_frontend.') or k.startswith('aux_head_audio.')]
                torch.save({k: self.model.state_dict()[k] for k in a_keys}, self.best_audio_path)
                logger.info(f"[*] New PEAK Audio Backbone! Saved: '{self.best_audio_path}' (Val Acc = {best_val_audio_acc:.4f})")

            # 3. Track and save best overall Multimodal checkpoint (Monitored by Validation Accuracy)
            is_best = False
            if monitor_metric == 'loss':
                score = -val_loss
                if val_loss < best_val_metric:
                    best_val_metric = val_loss
                    is_best = True
            else:
                # Default & primary monitor: Validation Accuracy (acc val)
                score = val_acc
                if val_acc > best_val_metric:
                    best_val_metric = val_acc
                    is_best = True

            if is_best:
                best_epoch = epoch
                best_acc = val_acc
                best_mAP = val_mAP
                best_loss = val_loss
                best_val_statistics = val_stats
                torch.save(self.model.state_dict(), self.best_checkpoint_path)
                logger.info(f"[*] New best validation performance! Saved checkpoint: '{self.best_checkpoint_path}' (Val Acc = {best_acc:.4f})")

            logger.info(
                f"Current best: Epoch {best_epoch:03d} | Val Acc: {best_acc:.4f} (Loss: {best_loss:.5f}) | Best Video: {best_val_video_acc:.4f} | Best Audio: {best_val_audio_acc:.4f}"
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

        # Post-training Nelder-Mead calibration
        try:
            self.nelder_mead_calibrate()
        except Exception as exc:
            logger.warning(f"Nelder-Mead calibration encountered an error: {exc}")

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
