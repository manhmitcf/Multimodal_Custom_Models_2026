import os
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

# Ensure stdout/stderr UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


class BaseEvaluator:
    """
    Abstract Base Class standardizing evaluation metrics computation.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.device = next(model.parameters()).device

    def evaluate(self, data_loader: Any) -> Dict[str, Any]:
        raise NotImplementedError("Method 'evaluate' must be implemented in subclasses.")


class MultimodalEvaluator(BaseEvaluator):
    """
    MultimodalEvaluator evaluating both Video and Audio inputs.
    Calculates Accuracy, Average Precision (AP), AUC, Confusion Matrix, and F1-Scores.
    """
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model=model)

    def _move_data_to_device(self, x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        if 'float' in str(getattr(x, 'dtype', '')):
            x = torch.Tensor(x)
        elif 'int' in str(getattr(x, 'dtype', '')):
            x = torch.LongTensor(x)
        else:
            x = torch.as_tensor(x)
        return x.to(self.device)

    def _append_to_dict(self, data_dict: Dict[str, list], key: str, value: Any) -> None:
        if key in data_dict:
            data_dict[key].append(value)
        else:
            data_dict[key] = [value]

    def _forward_multimodal(self, data_loader: Any) -> Dict[str, np.ndarray]:
        output_dict = {}
        pbar = tqdm(data_loader, desc="Running multimodal model evaluation...")

        for batch_data_dict in pbar:
            batch_video = self._move_data_to_device(batch_data_dict['video_form'])
            batch_audio = self._move_data_to_device(batch_data_dict['audio_form'])

            with torch.no_grad():
                self.model.eval()
                batch_output = self.model(batch_video, batch_audio)
                if isinstance(batch_output, dict):
                    batch_logits = batch_output.get('clipwise_output', batch_output)
                else:
                    batch_logits = batch_output

            if 'clip_name' in batch_data_dict:
                self._append_to_dict(output_dict, 'clip_name', batch_data_dict['clip_name'])
            self._append_to_dict(
                output_dict,
                'clipwise_output',
                batch_logits.detach().float().cpu().numpy()
            )

            if 'target' in batch_data_dict:
                tgt = batch_data_dict['target']
                if hasattr(tgt, 'detach'):
                    tgt = tgt.detach().cpu().numpy()
                elif hasattr(tgt, 'numpy'):
                    tgt = tgt.numpy()
                self._append_to_dict(output_dict, 'target', tgt)

        for key in output_dict.keys():
            output_dict[key] = np.concatenate(output_dict[key], axis=0)

        return output_dict

    def evaluate(self, data_loader: Any) -> Dict[str, Any]:
        output_dict = self._forward_multimodal(data_loader)

        clipwise_output = output_dict['clipwise_output']
        target = output_dict['target']

        average_precision = metrics.average_precision_score(
            target, clipwise_output, average=None
        )

        try:
            auc = metrics.roc_auc_score(target, clipwise_output, average=None)
        except Exception:
            auc = np.zeros(clipwise_output.shape[1])

        target_acc = np.argmax(target, axis=1) if target.ndim > 1 else target
        clipwise_output_acc = np.argmax(clipwise_output, axis=1)
        acc = accuracy_score(target_acc, clipwise_output_acc)

        cm = confusion_matrix(target_acc, clipwise_output_acc)

        message = classification_report(target_acc, clipwise_output_acc, digits=4, zero_division=0)
        message = '\n' + message

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
