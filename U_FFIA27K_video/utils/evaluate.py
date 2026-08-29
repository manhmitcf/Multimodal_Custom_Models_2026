import os
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

# Ensure stdout/stderr UTF-8 encoding on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


class BaseEvaluator:
    """
    Abstract Base Class (ABC / Interface) standardizing evaluation metrics computation.
    Complies with OOP design guidelines.
    """
    def __init__(self, model: nn.Module) -> None:
        """
        Initialize base evaluator.
        """
        self.model = model
        # Dynamically retrieve execution device (CPU/GPU) from model parameters
        self.device = next(model.parameters()).device

    def evaluate(self, data_loader: Any) -> Dict[str, Any]:
        """
        Run evaluation on data loader. Subclasses must override this method.
        """
        raise NotImplementedError("Method 'evaluate' must be implemented in subclasses.")


class VideoEvaluator(BaseEvaluator):
    """
    Specialized VideoEvaluator class, inheriting from BaseEvaluator.
    Encapsulates raw video model inference, postprocessing, and classification metrics calculation.
    """
    def __init__(self, model: nn.Module) -> None:
        super(VideoEvaluator, self).__init__(model=model)

    def _move_data_to_device(self, x: Any) -> torch.Tensor:
        """
        Private Method to move data tensors to CPU/GPU device.
        """
        if 'float' in str(x.dtype):
            x = torch.Tensor(x)
        elif 'int' in str(x.dtype):
            x = torch.LongTensor(x)
        else:
            return x
        return x.to(self.device)

    def _append_to_dict(self, data_dict: Dict[str, list], key: str, value: Any) -> None:
        """
        Private Method to append batch predictions to accumulation lists.
        """
        if key in data_dict:
            data_dict[key].append(value)
        else:
            data_dict[key] = [value]

    def _forward_video(self, data_loader: Any) -> Dict[str, np.ndarray]:
        """
        Private Method to run video model inference across the entire dataset.
        Returns accumulated predictions and ground truth labels as NumPy arrays.
        """
        output_dict = {}
        pbar = tqdm(data_loader, desc="Running video model evaluation...")

        for batch_data_dict in pbar:
            batch_video = self._move_data_to_device(batch_data_dict['video_form'])
            
            with torch.no_grad():
                self.model.eval()
                batch_output = self.model(batch_video)
                if isinstance(batch_output, dict):
                    batch_logits = batch_output.get('clipwise_output', batch_output)
                else:
                    batch_logits = batch_output

            self._append_to_dict(output_dict, 'video_name', batch_data_dict['video_name'])
            self._append_to_dict(
                output_dict, 
                'clipwise_output', 
                batch_logits.detach().float().cpu().numpy()
            )
            
            if 'target' in batch_data_dict:
                self._append_to_dict(output_dict, 'target', batch_data_dict['target'])

        # Concatenate mini-batch lists into unified arrays
        for key in output_dict.keys():
            output_dict[key] = np.concatenate(output_dict[key], axis=0)

        return output_dict

    def evaluate(self, data_loader: Any) -> Dict[str, Any]:
        """
        Evaluate video model on the entire dataset and compute metrics (Accuracy, AP, AUC, Confusion Matrix).
        Overrides abstract evaluate method from BaseEvaluator.
        """
        # 1. Run inference to collect predictions
        output_dict = self._forward_video(data_loader)

        clipwise_output = output_dict['clipwise_output']  # Shape: [videos_num, classes_num]
        target = output_dict['target']                    # Shape: [videos_num, classes_num]

        # 2. Compute Average Precision (AP) for each class
        average_precision = metrics.average_precision_score(
            target, clipwise_output, average=None
        )

        # 3. Compute Area Under the ROC Curve (AUC) for each class
        auc = metrics.roc_auc_score(target, clipwise_output, average=None)

        # 4. Compute overall accuracy
        target_acc = np.argmax(target, axis=1) if target.ndim > 1 else target
        clipwise_output_acc = np.argmax(clipwise_output, axis=1)
        acc = accuracy_score(target_acc, clipwise_output_acc)

        # 5. Compute confusion matrix
        cm = confusion_matrix(target_acc, clipwise_output_acc)

        # 6. Generate detailed text classification report
        message = classification_report(target_acc, clipwise_output_acc, digits=4, zero_division=0)
        message = '\n' + message

        # 7. Compute weighted and macro precision, recall, f1-score for reporting
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            target_acc, clipwise_output_acc, average='weighted', zero_division=0
        )
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            target_acc, clipwise_output_acc, average='macro', zero_division=0
        )

        statistics = {
            'average_precision': average_precision,
            'accuracy': acc,
            'auc': auc,
            'message': message,
            'confu_matrix': cm,
            'prec_weighted': prec_weighted,
            'rec_weighted': rec_weighted,
            'f1_weighted': f1_weighted,
            'prec_macro': prec_macro,
            'rec_macro': rec_macro,
            'f1_macro': f1_macro
        }

        return statistics
