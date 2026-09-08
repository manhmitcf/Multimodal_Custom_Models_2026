import os
import sys
import csv
import logging
from pathlib import Path
import numpy as np

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoryLogger:
    """
    OOP compliant HistoryLogger to automatically log training performance history.
    Records evaluation metrics and flattened confusion matrices to CSV and plots learning curves.
    """
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.history_csv_path = os.path.join(self.log_dir, 'history.csv')
        self._write_history_header()

    def _history_headers(self) -> list:
        return [
            'epoch',
            'train_loss',
            'train_accuracy',
            'train_acc_video',
            'train_acc_audio',
            'train_mAP',
            'val_loss',
            'val_accuracy',
            'val_acc_video',
            'val_acc_audio',
            'val_qwk',
            'val_mAP',
            'val_auc_class_none', 'val_auc_class_strong', 'val_auc_class_medium', 'val_auc_class_weak',
            'val_ap_class_none', 'val_ap_class_strong', 'val_ap_class_medium', 'val_ap_class_weak',
            'cm_none_none', 'cm_none_strong', 'cm_none_medium', 'cm_none_weak',
            'cm_strong_none', 'cm_strong_strong', 'cm_strong_medium', 'cm_strong_weak',
            'cm_medium_none', 'cm_medium_strong', 'cm_medium_medium', 'cm_medium_weak',
            'cm_weak_none', 'cm_weak_strong', 'cm_weak_medium', 'cm_weak_weak'
        ]

    def _write_history_header(self) -> None:
        with open(self.history_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self._history_headers())

    def _save_confusion_matrix_csv(self, path: str, matrix: np.ndarray) -> None:
        labels = ['none', 'strong', 'medium', 'weak']
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Actual\\Predicted'] + labels)
            for idx, label in enumerate(labels):
                writer.writerow([label] + list(matrix[idx]))

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        train_mAP: float,
        val_loss: float,
        val_statistics: dict,
        train_acc_video: float = 0.0,
        train_acc_audio: float = 0.0,
        is_best: bool = False
    ) -> None:
        val_acc = np.mean(val_statistics['accuracy'])
        val_acc_v = float(val_statistics.get('acc_video', 0.0))
        val_acc_a = float(val_statistics.get('acc_audio', 0.0))
        val_qwk = float(val_statistics.get('qwk', 0.0))
        val_mAP = np.mean(val_statistics['average_precision'])
        val_auc = val_statistics['auc']
        val_ap = val_statistics['average_precision']
        cm = val_statistics['confu_matrix']

        cm_flat = list(cm.flatten())

        row_data = [
            epoch,
            f"{train_loss:.6f}",
            f"{train_acc:.6f}",
            f"{train_acc_video:.6f}",
            f"{train_acc_audio:.6f}",
            f"{train_mAP:.6f}",
            f"{val_loss:.6f}",
            f"{val_acc:.6f}",
            f"{val_acc_v:.6f}",
            f"{val_acc_a:.6f}",
            f"{val_qwk:.6f}",
            f"{val_mAP:.6f}",
            f"{val_auc[0]:.6f}", f"{val_auc[1]:.6f}", f"{val_auc[2]:.6f}", f"{val_auc[3]:.6f}",
            f"{val_ap[0]:.6f}", f"{val_ap[1]:.6f}", f"{val_ap[2]:.6f}", f"{val_ap[3]:.6f}"
        ] + [int(val) for val in cm_flat]

        with open(self.history_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

        if is_best:
            best_cm_path = os.path.join(self.log_dir, 'confusion_matrix_best.csv')
            self._save_confusion_matrix_csv(best_cm_path, cm)

    def save_summary(self, training_time: float, inference_time_ms: float, val_statistics: dict, test_statistics: dict) -> None:
        summary_csv_path = os.path.join(self.log_dir, 'summary.csv')
        file_exists = os.path.exists(summary_csv_path)

        val_mAP = np.mean(val_statistics['average_precision'])
        test_mAP = np.mean(test_statistics['average_precision'])

        headers = [
            'Training Time (s)',
            'Inference Time (ms/sample)',
            'Val Acc Fusion', 'Val Acc Video', 'Val Acc Audio', 'Val QWK', 'mAP Val',
            'Precision Val (Weighted)', 'Recall Val (Weighted)', 'F1-score Val (Weighted)',
            'Precision Val (Macro)', 'Recall Val (Macro)', 'F1-score Val (Macro)',
            'Test Acc Fusion', 'Test Acc Video', 'Test Acc Audio', 'Test QWK', 'mAP Test',
            'Precision Test (Weighted)', 'Recall Test (Weighted)', 'F1-score Test (Weighted)',
            'Precision Test (Macro)', 'Recall Test (Macro)', 'F1-score Test (Macro)'
        ]

        row_data = [
            f"{training_time:.2f}",
            f"{inference_time_ms:.3f}",
            f"{val_statistics['accuracy']:.6f}",
            f"{val_statistics.get('acc_video', 0.0):.6f}",
            f"{val_statistics.get('acc_audio', 0.0):.6f}",
            f"{val_statistics.get('qwk', 0.0):.6f}",
            f"{val_mAP:.6f}",
            f"{val_statistics['prec_weighted']:.6f}",
            f"{val_statistics['rec_weighted']:.6f}",
            f"{val_statistics['f1_weighted']:.6f}",
            f"{val_statistics['prec_macro']:.6f}",
            f"{val_statistics['rec_macro']:.6f}",
            f"{val_statistics['f1_macro']:.6f}",
            f"{test_statistics['accuracy']:.6f}",
            f"{test_statistics.get('acc_video', 0.0):.6f}",
            f"{test_statistics.get('acc_audio', 0.0):.6f}",
            f"{test_statistics.get('qwk', 0.0):.6f}",
            f"{test_mAP:.6f}",
            f"{test_statistics['prec_weighted']:.6f}",
            f"{test_statistics['rec_weighted']:.6f}",
            f"{test_statistics['f1_weighted']:.6f}",
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
        if not os.path.exists(self.history_csv_path):
            logger.warning(f"Warning: History file '{self.history_csv_path}' does not exist. Cannot plot curves.")
            return

        epochs = []
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        val_acc_v_list, val_acc_a_list = [], []
        train_maps, val_maps = [], []

        with open(self.history_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    epochs.append(int(row['epoch']))
                    train_losses.append(float(row['train_loss']))
                    val_losses.append(float(row['val_loss']))
                    train_accs.append(float(row['train_accuracy']))
                    val_accs.append(float(row['val_accuracy']))
                    val_acc_v_list.append(float(row.get('val_acc_video', 0.0)))
                    val_acc_a_list.append(float(row.get('val_acc_audio', 0.0)))
                    train_maps.append(float(row['train_mAP']))
                    val_maps.append(float(row['val_mAP']))
                except (KeyError, ValueError):
                    continue

        if not epochs:
            return

        import matplotlib
        matplotlib.use('Agg')
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Multimodal Fish Feeding Intensity Model Learning History (Student Backbones + Fusion)', fontsize=15, fontweight='bold', y=0.98)

        axes[0].plot(epochs, train_losses, label='Train Loss', color='#1f77b4', linewidth=2, linestyle='--')
        axes[0].plot(epochs, val_losses, label='Val Loss', color='#ff7f0e', linewidth=2)
        axes[0].set_title('Loss Curves', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, linestyle=':', alpha=0.6)
        axes[0].legend(frameon=True)

        axes[1].plot(epochs, train_accs, label='Train Acc (Fusion)', color='#2ca02c', linewidth=2, linestyle='--')
        axes[1].plot(epochs, val_accs, label='Val Acc (Fusion)', color='#d62728', linewidth=2)
        if any(v > 0 for v in val_acc_v_list):
            axes[1].plot(epochs, val_acc_v_list, label='Val Acc (Video Backbone)', color='#17becf', linewidth=1.5, linestyle=':')
        if any(a > 0 for a in val_acc_a_list):
            axes[1].plot(epochs, val_acc_a_list, label='Val Acc (Audio Backbone)', color='#bcbd22', linewidth=1.5, linestyle=':')
        axes[1].set_title('Accuracy Curves (Backbones + Fusion)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True, linestyle=':', alpha=0.6)
        axes[1].legend(frameon=True)

        axes[2].plot(epochs, train_maps, label='Train mAP', color='#9467bd', linewidth=2, linestyle='--')
        axes[2].plot(epochs, val_maps, label='Val mAP', color='#8c564b', linewidth=2)
        axes[2].set_title('Mean Average Precision (mAP)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('mAP')
        axes[2].grid(True, linestyle=':', alpha=0.6)
        axes[2].legend(frameon=True)

        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'learning_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Successfully generated learning curves plot at: '{plot_path}'")
