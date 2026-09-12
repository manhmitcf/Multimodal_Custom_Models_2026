import os
import sys
import csv
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
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
            'lr',
            'epoch_time_seconds',
            'train_loss',
            'train_accuracy',
            'train_mAP',
            'val_loss',
            'val_accuracy',
            'val_acc_video',
            'val_acc_audio',
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
        lr: Optional[float] = None,
        epoch_time_seconds: Optional[float] = None,
        is_best: bool = False
    ) -> None:
        val_acc = float(np.mean(val_statistics['accuracy']))
        val_mAP = float(np.mean(val_statistics['average_precision']))
        val_acc_v = float(val_statistics.get('acc_video', 0.0))
        val_acc_a = float(val_statistics.get('acc_audio', 0.0))
        val_auc = val_statistics['auc']
        val_ap = val_statistics['average_precision']
        cm = val_statistics['confu_matrix']

        cm_flat = list(cm.flatten())
        lr_str = f"{lr:.8e}" if lr is not None else "N/A"
        time_str = f"{epoch_time_seconds:.2f}" if epoch_time_seconds is not None else "N/A"

        row_data = [
            epoch,
            lr_str,
            time_str,
            f"{train_loss:.6f}",
            f"{train_acc:.6f}",
            f"{train_mAP:.6f}",
            f"{val_loss:.6f}",
            f"{val_acc:.6f}",
            f"{val_acc_v:.6f}",
            f"{val_acc_a:.6f}",
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

    def save_summary(
        self,
        training_time: float,
        inference_time_ms: float,
        val_statistics: dict,
        test_statistics: dict,
        total_params_m: Optional[float] = None,
        gflops: Optional[float] = None
    ) -> None:
        summary_csv_path = os.path.join(self.log_dir, 'summary.csv')
        file_exists = os.path.exists(summary_csv_path)

        val_mAP = np.mean(val_statistics['average_precision'])
        test_mAP = np.mean(test_statistics['average_precision'])

        headers = [
            'Total Parameters (M)',
            'Inference Complexity (GFLOPs)',
            'Training Time (s)',
            'Inference Time (ms/sample)',
            'Precision Val (Weighted)', 'Recall Val (Weighted)', 'F1-score Val (Weighted)', 'Accuracy Val', 'mAP Val',
            'Precision Val (Macro)', 'Recall Val (Macro)', 'F1-score Val (Macro)',
            'Precision Test (Weighted)', 'Recall Test (Weighted)', 'F1-score Test (Weighted)', 'Accuracy Test', 'mAP Test',
            'Precision Test (Macro)', 'Recall Test (Macro)', 'F1-score Test (Macro)'
        ]

        params_str = f"{total_params_m:.3f}" if total_params_m is not None else "N/A"
        gflops_str = f"{gflops:.3f}" if gflops is not None else "N/A"

        row_data = [
            params_str,
            gflops_str,
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
        if not os.path.exists(self.history_csv_path):
            logger.warning(f"Warning: History file '{self.history_csv_path}' does not exist. Cannot plot curves.")
            return

        epochs = []
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
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
        fig.suptitle('Multimodal Fish Feeding Intensity Model Learning History', fontsize=16, fontweight='bold', y=0.98)

        axes[0].plot(epochs, train_losses, label='Train Loss', color='#1f77b4', linewidth=2, linestyle='--')
        axes[0].plot(epochs, val_losses, label='Val Loss', color='#ff7f0e', linewidth=2)
        axes[0].set_title('Loss Curves', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, linestyle=':', alpha=0.6)
        axes[0].legend(frameon=True)

        axes[1].plot(epochs, train_accs, label='Train Acc', color='#2ca02c', linewidth=2, linestyle='--')
        axes[1].plot(epochs, val_accs, label='Val Acc', color='#d62728', linewidth=2)
        axes[1].set_title('Accuracy Curves', fontsize=12, fontweight='bold')
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

    def save_detailed_evaluation_report(
        self,
        val_statistics: Dict[str, Any],
        test_statistics: Dict[str, Any]
    ) -> str:
        """
        Consolidates Confusion Matrices and Classification Reports for all 3 branches
        (Multimodal Fusion, Video Aux Head, Audio Aux Head) across both Validation (best epoch)
        and Test splits into a single comprehensive report file (.txt and .json).
        """
        report_txt_path = os.path.join(self.log_dir, 'evaluation_detailed_report.txt')
        report_json_path = os.path.join(self.log_dir, 'evaluation_detailed_report.json')

        def format_cm(cm: Optional[np.ndarray]) -> str:
            if cm is None:
                return "  [No confusion matrix available]\n"
            labels = ['None', 'Strong', 'Medium', 'Weak']
            header = f"  {'Actual\\Pred':<14}" + "".join(f"{lbl:>10}" for lbl in labels) + "\n"
            lines = [header]
            for idx, lbl in enumerate(labels):
                row = cm[idx] if idx < len(cm) else [0] * len(labels)
                row_str = f"  {lbl:<14}" + "".join(f"{int(val):>10d}" for val in row) + "\n"
                lines.append(row_str)
            return "".join(lines)

        content = []
        content.append("=" * 80)
        content.append("          COMPREHENSIVE MULTIMODAL & DUAL BACKBONE EVALUATION REPORT")
        content.append("=" * 80)
        content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"Directory:    {self.log_dir}\n")

        # PART 1: VALIDATION
        content.append("-" * 80)
        content.append("PART 1: VALIDATION SPLIT (AT BEST FUSION MODEL CHECKPOINT)")
        content.append("-" * 80)

        # 1.1 Fusion
        content.append("\n[1.1] MULTIMODAL TOURNAMENT FUSION (OVERALL MODEL)")
        content.append(f"  - Accuracy:       {float(np.mean(val_statistics.get('accuracy', 0.0))):.4f}")
        content.append(f"  - mAP:            {float(np.mean(val_statistics.get('average_precision', 0.0))):.4f}")
        content.append(f"  - Macro F1:       {float(val_statistics.get('f1_macro', 0.0)):.4f}")
        content.append(f"  - Weighted F1:    {float(val_statistics.get('f1_weighted', 0.0)):.4f}")
        content.append("\n  Confusion Matrix (Actual \\ Predicted):")
        content.append(format_cm(val_statistics.get('confu_matrix')))
        content.append("  Classification Report:")
        content.append(val_statistics.get('message', '  N/A\n'))

        # 1.2 Video
        content.append("\n[1.2] VIDEO AUXILIARY HEAD (ConvNeXt-Nano 7-Channel Kinematics)")
        content.append(f"  - Accuracy:       {float(val_statistics.get('acc_video', 0.0)):.4f}")
        content.append(f"  - Macro F1:       {float(val_statistics.get('f1_macro_video', 0.0)):.4f}")
        content.append(f"  - Weighted F1:    {float(val_statistics.get('f1_weighted_video', 0.0)):.4f}")
        content.append("\n  Confusion Matrix (Actual \\ Predicted):")
        content.append(format_cm(val_statistics.get('confu_matrix_video')))
        content.append("  Classification Report:")
        content.append(val_statistics.get('message_video', '  N/A\n'))

        # 1.3 Audio
        content.append("\n[1.3] AUDIO AUXILIARY HEAD (TKEO-STFT-MLP 256 kHz)")
        content.append(f"  - Accuracy:       {float(val_statistics.get('acc_audio', 0.0)):.4f}")
        content.append(f"  - Macro F1:       {float(val_statistics.get('f1_macro_audio', 0.0)):.4f}")
        content.append(f"  - Weighted F1:    {float(val_statistics.get('f1_weighted_audio', 0.0)):.4f}")
        content.append("\n  Confusion Matrix (Actual \\ Predicted):")
        content.append(format_cm(val_statistics.get('confu_matrix_audio')))
        content.append("  Classification Report:")
        content.append(val_statistics.get('message_audio', '  N/A\n'))

        # PART 2: TEST
        content.append("\n" + "-" * 80)
        content.append("PART 2: TEST SPLIT (FINAL INDEPENDENT EVALUATION)")
        content.append("-" * 80)

        # 2.1 Fusion
        content.append("\n[2.1] MULTIMODAL TOURNAMENT FUSION (OVERALL MODEL)")
        content.append(f"  - Accuracy:       {float(np.mean(test_statistics.get('accuracy', 0.0))):.4f}")
        content.append(f"  - mAP:            {float(np.mean(test_statistics.get('average_precision', 0.0))):.4f}")
        content.append(f"  - Macro F1:       {float(test_statistics.get('f1_macro', 0.0)):.4f}")
        content.append(f"  - Weighted F1:    {float(test_statistics.get('f1_weighted', 0.0)):.4f}")
        content.append("\n  Confusion Matrix (Actual \\ Predicted):")
        content.append(format_cm(test_statistics.get('confu_matrix')))
        content.append("  Classification Report:")
        content.append(test_statistics.get('message', '  N/A\n'))

        # 2.2 Video
        content.append("\n[2.2] VIDEO AUXILIARY HEAD (ConvNeXt-Nano 7-Channel Kinematics)")
        content.append(f"  - Accuracy:       {float(test_statistics.get('acc_video', 0.0)):.4f}")
        content.append(f"  - Macro F1:       {float(test_statistics.get('f1_macro_video', 0.0)):.4f}")
        content.append(f"  - Weighted F1:    {float(test_statistics.get('f1_weighted_video', 0.0)):.4f}")
        content.append("\n  Confusion Matrix (Actual \\ Predicted):")
        content.append(format_cm(test_statistics.get('confu_matrix_video')))
        content.append("  Classification Report:")
        content.append(test_statistics.get('message_video', '  N/A\n'))

        # 2.3 Audio
        content.append("\n[2.3] AUDIO AUXILIARY HEAD (TKEO-STFT-MLP 256 kHz)")
        content.append(f"  - Accuracy:       {float(test_statistics.get('acc_audio', 0.0)):.4f}")
        content.append(f"  - Macro F1:       {float(test_statistics.get('f1_macro_audio', 0.0)):.4f}")
        content.append(f"  - Weighted F1:    {float(test_statistics.get('f1_weighted_audio', 0.0)):.4f}")
        content.append("\n  Confusion Matrix (Actual \\ Predicted):")
        content.append(format_cm(test_statistics.get('confu_matrix_audio')))
        content.append("  Classification Report:")
        content.append(test_statistics.get('message_audio', '  N/A\n'))

        content.append("=" * 80 + "\n")

        with open(report_txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(content))
        logger.info(f"Successfully exported Consolidated Detailed Report to: '{report_txt_path}'")

        # Also serialize to structured JSON
        def sanitize_for_json(d: Any) -> Any:
            if isinstance(d, dict):
                return {str(k): sanitize_for_json(v) for k, v in d.items()}
            elif isinstance(d, (list, tuple)):
                return [sanitize_for_json(v) for v in d]
            elif isinstance(d, np.ndarray):
                return d.tolist()
            elif isinstance(d, (np.floating, np.integer)):
                return d.item()
            elif isinstance(d, (int, float, str, bool)) or d is None:
                return d
            return str(d)

        json_payload = {
            "val_evaluation": sanitize_for_json(val_statistics),
            "test_evaluation": sanitize_for_json(test_statistics)
        }
        with open(report_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_payload, f, indent=2)

        return report_txt_path

    def plot_test_confusion_matrices(self, test_statistics: Dict[str, Any]) -> str:
        """Plot a 1x3 heatmap comparison of Confusion Matrices on Test set: Video, Audio, Fusion."""
        import matplotlib
        matplotlib.use('Agg')
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        import matplotlib.pyplot as plt

        acc_v = float(test_statistics.get('acc_video', 0.0))
        acc_a = float(test_statistics.get('acc_audio', 0.0))
        acc_f = float(np.mean(test_statistics.get('accuracy', 0.0)))

        cms = [
            (test_statistics.get('confu_matrix_video'), f"Video Aux Head (Acc: {acc_v:.4f})"),
            (test_statistics.get('confu_matrix_audio'), f"Audio Aux Head (Acc: {acc_a:.4f})"),
            (test_statistics.get('confu_matrix'), f"Multimodal Fusion (Acc: {acc_f:.4f})"),
        ]

        class_names = ['None', 'Strong', 'Medium', 'Weak']
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        fig.suptitle('Test Set Confusion Matrices: Video Aux vs Audio Aux vs Multimodal Fusion', fontsize=15, fontweight='bold')

        for idx, (cm, title) in enumerate(cms):
            ax = axes[idx]
            if cm is not None and isinstance(cm, np.ndarray) and cm.size > 0:
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.set_title(title, fontsize=12, fontweight='bold')
                tick_marks = np.arange(len(class_names))
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                ax.set_xticklabels(class_names, rotation=30)
                ax.set_yticklabels(class_names)
                ax.set_ylabel('Actual Label', fontweight='bold')
                ax.set_xlabel('Predicted Label', fontweight='bold')

                thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
                for i in range(cm.shape[0]):
                    row_total = np.sum(cm[i, :])
                    for j in range(cm.shape[1]):
                        count = int(cm[i, j])
                        pct = (count / row_total * 100.0) if row_total > 0 else 0.0
                        color = "white" if count > thresh else "black"
                        ax.text(j, i, f"{count}\n({pct:.1f}%)",
                                horizontalalignment="center",
                                verticalalignment="center",
                                color=color, fontsize=10, fontweight='bold')
            else:
                ax.text(0.5, 0.5, "Data Not Available", horizontalalignment='center', verticalalignment='center')
                ax.set_title(title, fontsize=12)

        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'confusion_matrices_test.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Successfully exported Test Confusion Matrices plot to: '{plot_path}'")
        return plot_path
