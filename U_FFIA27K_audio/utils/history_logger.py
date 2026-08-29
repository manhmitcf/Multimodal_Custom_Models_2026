import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import csv
import numpy as np
import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoryLogger:
    """
    OOP compliant HistoryLogger to automatically log training performance history.
    Automatically records evaluation metrics and flattened confusion matrices to a CSV file.
    """
    def __init__(self, log_dir: str) -> None:
        """
        Initialize HistoryLogger.

        Args:
            log_dir (str): Path to directory where logs and CSV files are stored.
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.history_csv_path = os.path.join(self.log_dir, 'history.csv')
        self._write_history_header()

    def _history_headers(self) -> list:
        return [
            'epoch',
            'train_loss',
            'train_accuracy',
            'train_mAP',
            'val_loss',
            'val_accuracy',
            'val_mAP',
            'val_auc_class_none', 'val_auc_class_strong', 'val_auc_class_medium', 'val_auc_class_weak',
            'val_ap_class_none', 'val_ap_class_strong', 'val_ap_class_medium', 'val_ap_class_weak',

            # 16 flattened confusion matrix columns (Actual vs Predicted)
            'cm_none_none', 'cm_none_strong', 'cm_none_medium', 'cm_none_weak',
            'cm_strong_none', 'cm_strong_strong', 'cm_strong_medium', 'cm_strong_weak',
            'cm_medium_none', 'cm_medium_strong', 'cm_medium_medium', 'cm_medium_weak',
            'cm_weak_none', 'cm_weak_strong', 'cm_weak_medium', 'cm_weak_weak'
        ]

    def _write_history_header(self) -> None:
        # A new trainer instance represents a new run, so do not append epochs from previous runs.
        with open(self.history_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self._history_headers())

    def _save_confusion_matrix_csv(self, path: str, matrix: np.ndarray) -> None:
        """
        Private Method to save 2D confusion matrix to a labeled CSV file.
        """
        labels = ['none', 'strong', 'medium', 'weak']
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header row (predicted labels)
            writer.writerow(['Actual\\Predicted'] + labels)
            # Write rows (actual labels)
            for idx, label in enumerate(labels):
                writer.writerow([label] + list(matrix[idx]))

    def log_epoch(self, epoch: int, train_loss: float, train_acc: float, train_mAP: float, val_loss: float, val_statistics: dict, is_best: bool = False) -> None:
        """
        Record performance metrics for the current epoch and append to the history CSV file.

        Args:
            epoch (int): Current epoch number.
            train_loss (float): Mean training loss value.
            train_acc (float): Training accuracy value.
            train_mAP (float): Training mean Average Precision.
            val_loss (float): Validation loss value.
            val_statistics (dict): Dictionary returned by AudioEvaluator.
            is_best (bool): True if current epoch represents the best validation performance. Default: False.
        """
        val_acc = np.mean(val_statistics['accuracy'])
        val_mAP = np.mean(val_statistics['average_precision'])
        val_auc = val_statistics['auc']               # Class AUC array
        val_ap = val_statistics['average_precision']   # Class AP array
        cm = val_statistics['confu_matrix']           # 2D confusion matrix

        # Flatten 2D confusion matrix to 1D list
        cm_flat = list(cm.flatten())

        # Write new row to history.csv
        row_data = [
            epoch,
            f"{train_loss:.6f}",
            f"{train_acc:.6f}",
            f"{train_mAP:.6f}",
            f"{val_loss:.6f}",
            f"{val_acc:.6f}",
            f"{val_mAP:.6f}",
            f"{val_auc[0]:.6f}", f"{val_auc[1]:.6f}", f"{val_auc[2]:.6f}", f"{val_auc[3]:.6f}",
            f"{val_ap[0]:.6f}", f"{val_ap[1]:.6f}", f"{val_ap[2]:.6f}", f"{val_ap[3]:.6f}"
        ] + [int(val) for val in cm_flat]

        with open(self.history_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

        # Update best confusion matrix CSV if this is the best epoch
        if is_best:
            best_cm_path = os.path.join(self.log_dir, 'confusion_matrix_best.csv')
            self._save_confusion_matrix_csv(best_cm_path, cm)

    def save_summary(self, training_time: float, inference_time_ms: float, val_statistics: dict, test_statistics: dict) -> None:
        """
        Save final summary report (Val & Test performance metrics, training and inference times) to summary.csv.
        """
        summary_csv_path = os.path.join(self.log_dir, 'summary.csv')
        
        # Check if file exists to write headers
        file_exists = os.path.exists(summary_csv_path)
        
        # Compute mAP Val and Test
        val_mAP = np.mean(val_statistics['average_precision'])
        test_mAP = np.mean(test_statistics['average_precision'])

        headers = [
            'Training Time (s)',
            'Inference Time (ms/sample)',
            'Precision Val (Weighted)', 'Recall Val (Weighted)', 'F1-score Val (Weighted)', 'Accuracy Val', 'mAP Val',
            'Precision Val (Macro)', 'Recall Val (Macro)', 'F1-score Val (Macro)',
            'Precision Test (Weighted)', 'Recall Test (Weighted)', 'F1-score Test (Weighted)', 'Accuracy Test', 'mAP Test',
            'Precision Test (Macro)', 'Recall Test (Macro)', 'F1-score Test (Macro)'
        ]
        
        row_data = [
            f"{training_time:.2f}",
            f"{inference_time_ms:.3f}",
            f"{val_statistics['prec_weighted']:.6f}",
            f"{val_statistics['rec_weighted']:.6f}",
            f"{val_statistics['f1_weighted']:.6f}",
            f"{val_statistics['accuracy']:.6f}",
            f"{val_mAP:.6f}",
            f"{val_statistics['prec_macro']:.6f}",
            f"{val_statistics['rec_macro']:.6f}",
            f"{val_statistics['f1_macro']:.6f}",
            f"{test_statistics['prec_weighted']:.6f}",
            f"{test_statistics['rec_weighted']:.6f}",
            f"{test_statistics['f1_weighted']:.6f}",
            f"{test_statistics['accuracy']:.6f}",
            f"{test_mAP:.6f}",
            f"{test_statistics['prec_macro']:.6f}",
            f"{test_statistics['rec_macro']:.6f}",
            f"{test_statistics['f1_macro']:.6f}"
        ]
        
        with open(summary_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row_data)
        logger.info(f"Successfully exported Summary Report to: '{summary_csv_path}'")

    def plot_history(self) -> None:
        """
        Generate and save a 3-panel learning curve plot (Loss, Accuracy, mAP) from the history CSV.
        """
        if not os.path.exists(self.history_csv_path):
            logger.warning(f"Warning: History file '{self.history_csv_path}' does not exist. Cannot plot curves.")
            return

        epochs = []
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        train_maps, val_maps = [], []

        # Read history data from CSV
        with open(self.history_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    epochs.append(int(row['epoch']))
                    train_losses.append(float(row['train_loss']))
                    val_losses.append(float(row['val_loss']))
                    train_accs.append(float(row['train_accuracy']))
                    val_accs.append(float(row['val_accuracy']))
                    train_maps.append(float(row['train_mAP']))
                    val_maps.append(float(row['val_mAP']))
                except KeyError as e:
                    logger.warning(f"Warning: Missing column in history CSV when plotting: {str(e)}")
                    return
                except ValueError:
                    continue  # Skip row on format errors

        if not epochs:
            logger.warning("Warning: No epoch data found to plot.")
            return

        # If an older history.csv contains multiple runs, keep only the latest epoch sequence.
        start_idx = 0
        for idx in range(1, len(epochs)):
            if epochs[idx] <= epochs[idx - 1]:
                start_idx = idx
        if start_idx:
            epochs = epochs[start_idx:]
            train_losses = train_losses[start_idx:]
            val_losses = val_losses[start_idx:]
            train_accs = train_accs[start_idx:]
            val_accs = val_accs[start_idx:]
            train_maps = train_maps[start_idx:]
            val_maps = val_maps[start_idx:]

        # Setup headless matplotlib backend for remote/docker compatibility
        import matplotlib
        matplotlib.use('Agg')
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        import matplotlib.pyplot as plt

        # Create a beautiful 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Fish Feeding Intensity Model Learning History', fontsize=16, fontweight='bold', y=0.98)

        # 1. Loss Panel
        axes[0].plot(epochs, train_losses, label='Train Loss', color='#1f77b4', linewidth=2, linestyle='--')
        axes[0].plot(epochs, val_losses, label='Val Loss', color='#ff7f0e', linewidth=2)
        axes[0].set_title('Loss Curves', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=10)
        axes[0].set_ylabel('Loss', fontsize=10)
        axes[0].grid(True, linestyle=':', alpha=0.6)
        axes[0].legend(frameon=True)

        # 2. Accuracy Panel
        axes[1].plot(epochs, train_accs, label='Train Acc', color='#2ca02c', linewidth=2, linestyle='--')
        axes[1].plot(epochs, val_accs, label='Val Acc', color='#d62728', linewidth=2)
        axes[1].set_title('Accuracy Curves', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=10)
        axes[1].set_ylabel('Accuracy', fontsize=10)
        axes[1].grid(True, linestyle=':', alpha=0.6)
        axes[1].legend(frameon=True)

        # 3. mAP Panel
        axes[2].plot(epochs, train_maps, label='Train mAP', color='#9467bd', linewidth=2, linestyle='--')
        axes[2].plot(epochs, val_maps, label='Val mAP', color='#8c564b', linewidth=2)
        axes[2].set_title('Mean Average Precision (mAP)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Epoch', fontsize=10)
        axes[2].set_ylabel('mAP', fontsize=10)
        axes[2].grid(True, linestyle=':', alpha=0.6)
        axes[2].legend(frameon=True)

        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'learning_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Successfully generated and saved learning curves to: '{plot_path}'")
