import os
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import ClipCELoss, AudioEvaluator, EarlyStopping, HistoryLogger, InferenceTimer

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


class BaseTrainer:
    """
    Abstract Base Class (ABC / Interface) standardizing model training pipelines.
    Complies with OOP design guidelines.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        ckpt_dir: str
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        # Determine model name for the checkpoint directory
        model_name = "model"
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'get_name'):
            model_name = model.backbone.get_name()
        
        # Ensure base directory is not empty
        base_dir = ckpt_dir if ckpt_dir else "checkpoint"
        
        base_path = Path(base_dir)

        # CV runs pass checkpoint/<model_name>/fold_xx and should not append model_name again.
        if base_path.name.startswith("fold_") and base_path.parent.name == model_name:
            self.ckpt_dir = base_dir
        elif not base_dir.rstrip('/\\').endswith(model_name):
            self.ckpt_dir = os.path.join(base_dir, model_name)
        else:
            self.ckpt_dir = base_dir
            
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def train(self, train_loader: Any, val_loader: Any, test_loader: Any, max_epoch: int) -> None:
        """
        Run training pipeline. Subclasses must override this method.
        """
        raise NotImplementedError("Method 'train' must be implemented in subclasses.")


class AudioTrainer(BaseTrainer):
    """
    OOP compliant trainer class for audio classification models (AudioTrainer).
    Integrates best model checkpoint saving, early stopping, and CSV history logging.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        ckpt_dir: str,
        monitor: str = 'accuracy',
        early_stopping: bool = True,
        patience: int = 30,
        delta: float = 0.0,
        train_config_path: str = 'config/train_config.json'
    ) -> None:
        super(AudioTrainer, self).__init__(model, optimizer, device, ckpt_dir)
        # Initialize OOP loss and evaluator components
        self.loss_fn = ClipCELoss()
        self.evaluator = AudioEvaluator(model=self.model)
        
        # Validate monitored performance metric
        assert monitor in ['accuracy', 'loss'], "Error: Monitored metric must be 'accuracy' or 'loss'!"
        self.monitor = monitor
        
        # Configure early stopping
        self.early_stopping = early_stopping
        if self.early_stopping:
            self.early_stopper = EarlyStopping(patience=patience, delta=delta, verbose=True)
        else:
            self.early_stopper = None

        # Initialize HistoryLogger for exporting training progression (using model-specific self.ckpt_dir)
        self.history_logger = HistoryLogger(log_dir=self.ckpt_dir)

        # Automatically copy config files to checkpoint directory for experiment tracking
        import shutil
        import json
        try:
            # 1. Copy the full train_config.json
            shutil.copy(train_config_path, os.path.join(self.ckpt_dir, 'train_config.json'))
            
            # 2. Extract and save splitter_config.json for compatibility
            from config import TrainConfig
            config_obj = TrainConfig.from_json(train_config_path)
            splitter_data = config_obj.dataset_splitter.model_dump()
            with open(os.path.join(self.ckpt_dir, 'splitter_config.json'), 'w', encoding='utf-8') as f:
                json.dump(splitter_data, f, indent=2)

            dataset_path = Path(config_obj.dataset_splitter.dataset_path)
            local_splits_dir = dataset_path / 'splits'
            legacy_splits_dir = dataset_path.parent / 'splits'
            if dataset_path.name in ['audio', 'video'] and legacy_splits_dir.exists() and not local_splits_dir.exists():
                base_splits_dir = legacy_splits_dir
            else:
                base_splits_dir = local_splits_dir

            if getattr(config_obj.dataset_splitter, "evaluation_mode", "holdout") == "cross_validation":
                fold_index = int(config_obj.dataset_splitter.fold_index)
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

        logger.info("==================================================")
        logger.info("AudioTrainer successfully initialized:")
        logger.info(f"  - Monitor Metric:               '{self.monitor}'")
        logger.info(f"  - Early Stopping Enabled:       {self.early_stopping}")
        if self.early_stopping:
            logger.info(f"    * Patience:                   {patience} epochs")
            logger.info(f"    * Delta:                      {delta}")
        logger.info(f"  - Checkpoint Dir:               '{ckpt_dir}'")
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

    def train(self, train_loader: Any, val_loader: Any, test_loader: Any, max_epoch: int) -> None:
        """
        Execute forward, backward updates and validation cycles.
        Overrides BaseTrainer's train method.
        """
        import time
        logger.info(f"Starting training pipeline (Monitor metric: {self.monitor})...")
        best_acc = 0.0
        best_mAP = 0.0
        best_loss = float('inf')
        best_epoch = 0
        best_val_statistics = None

        train_start_time = time.perf_counter()

        # Reset early stopper internal state
        if self.early_stopping:
            self.early_stopper.reset()

        # Initialize tracking metric values
        if self.monitor == 'accuracy':
            best_val_metric = 0.0
        else:  # loss
            best_val_metric = float('inf')

        for epoch in range(max_epoch):
            mean_loss = 0.0
            self.model.train()
            
            # Lists to accumulate training predictions and targets for metric computation
            train_preds = []
            train_targets = []
            
            if hasattr(train_loader.dataset, 'epoch'):
                train_loader.dataset.epoch = epoch

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epoch}")
            for data_dict in pbar:
                # 1. Transfer tensors to GPU device
                waveform = data_dict['waveform'].to(self.device)
                target = data_dict['target'].to(self.device)
                
                # 2. Forward execution
                output_dict = self.model(waveform)
                target_dict = {'target': target}
                
                # 3. Loss calculation
                loss = self.loss_fn(output_dict, target_dict)
                
                # 4. Backpropagation & optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update progress bar
                loss_val = loss.item()
                mean_loss += loss_val
                pbar.set_postfix({"Loss": f"{loss_val:.4f}"})

                # Accumulate predictions and targets (detached from graph)
                train_preds.append(output_dict['clipwise_output'].detach().cpu().numpy())
                train_targets.append(target.cpu().numpy())

            epoch_loss = mean_loss / len(train_loader)
            
            # Compute epoch-level training metrics
            train_preds = np.concatenate(train_preds, axis=0)
            train_targets = np.concatenate(train_targets, axis=0)
            train_acc = np.mean(np.argmax(train_targets, axis=1) == np.argmax(train_preds, axis=1))
            from sklearn import metrics as sklearn_metrics
            train_mAP = np.mean(sklearn_metrics.average_precision_score(train_targets, train_preds, average=None))

            # 5. Evaluate model performance on validation set
            self.model.eval()
            val_statistics = self.evaluator.evaluate(val_loader)
            
            val_mAP = np.mean(val_statistics['average_precision'])
            val_acc = np.mean(val_statistics['accuracy'])

            # Compute validation loss
            val_loss_sum = 0.0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_wave = val_batch['waveform'].to(self.device)
                    val_targ = val_batch['target'].to(self.device)
                    val_out = self.model(val_wave)
                    val_loss_sum += self.loss_fn(val_out, {'target': val_targ}).item()
            val_loss = val_loss_sum / len(val_loader)

            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss = {epoch_loss:.5f} | Train Acc = {train_acc:.4f} | Train mAP = {train_mAP:.4f} | "
                f"Val Loss = {val_loss:.5f} | Val Acc = {val_acc:.4f} | Val mAP = {val_mAP:.4f}"
            )

            # Save optimal model checkpoint based on monitored metric
            is_best = False
            if self.monitor == 'accuracy':
                score = val_acc
                if val_acc > best_val_metric:
                    best_val_metric = val_acc
                    is_best = True
            elif self.monitor == 'loss':
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
                best_model_path = os.path.join(self.ckpt_dir, 'audio_best.pt')
                
                self._save_checkpoint(best_model_path, epoch, val_acc if self.monitor == 'accuracy' else val_loss)

            # Record metrics and confusion matrix to history CSV
            self.history_logger.log_epoch(
                epoch=epoch,
                train_loss=epoch_loss,
                train_acc=train_acc,
                train_mAP=train_mAP,
                val_loss=val_loss,
                val_statistics=val_statistics,
                is_best=is_best
            )

            # Check early stopping conditions
            if self.early_stopping:
                if self.early_stopper.step(score):
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

        # 6. Run final evaluation on test set using best checkpoint
        logger.info("==================================================")
        logger.info("Training complete. Starting evaluation on Test split...")
        best_model_path = os.path.join(self.ckpt_dir, 'audio_best.pt')
        
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Reloaded best checkpoint model from Epoch {checkpoint['epoch']}...")
            
            self.model.eval()
            test_statistics = self.evaluator.evaluate(test_loader)
            test_mAP = np.mean(test_statistics['average_precision'])
            test_acc = np.mean(test_statistics['accuracy'])
            logger.info(f"TEST Results -> Accuracy: {test_acc:.4f} | mAP: {test_mAP:.4f}")
            logger.info(f"Detailed Classification Report:\n{test_statistics['message']}")
            
            # Measure inference latency and throughput
            logger.info("Measuring model Inference Latency on device...")
            timer = InferenceTimer(model=self.model, device=self.device)
            latency_ms = timer.measure_latency_per_sample(
                sample_length=128000, 
                warm_up_steps=10, 
                num_steps=50
            )

            # Save performance and timing summary to CSV
            if best_val_statistics is not None:
                self.history_logger.save_summary(training_time, latency_ms, best_val_statistics, test_statistics)
        else:
            logger.error("Error: Best model checkpoint audio_best.pt not found. Cannot evaluate on Test split!")
