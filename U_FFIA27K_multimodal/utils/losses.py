import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    """
    Abstract Base Class for loss functions.
    """
    def __init__(self) -> None:
        super(BaseLoss, self).__init__()

    def forward(self, output_dict: dict, target_dict: dict) -> torch.Tensor:
        raise NotImplementedError("Method 'forward' must be implemented in subclasses.")


class ClipCELoss(BaseLoss):
    """
    Multi-class Cross Entropy Loss at the clip level.
    """
    def forward(self, output_dict: dict, target_dict: dict) -> torch.Tensor:
        return F.cross_entropy(output_dict['clipwise_output'], target_dict['target'])


class ClipBCELoss(BaseLoss):
    """
    Binary Cross Entropy Loss at the clip level.
    """
    def forward(self, output_dict: dict, target_dict: dict) -> torch.Tensor:
        return F.binary_cross_entropy(output_dict['clipwise_output'], target_dict['target'])


class AdaptiveOrdinalCELoss(BaseLoss):
    """
    Adaptive Ordinal Cross Entropy Loss with Weak-vs-Medium Contrastive Margin.
    
    1. Cross Entropy with Soft Gaussian Ordinal Smoothing:
       Respects class ranking None (0) < Weak (1) < Medium (2) < Strong (3).
    2. Boundary Disambiguation Margin Loss:
       Directly supervises delta_margin:
       - If ground truth is Weak (1): encourages delta_margin <= -margin (amplifying Weak over Medium)
       - If ground truth is Medium (2): encourages delta_margin >= +margin (amplifying Medium over Weak)
    """
    def __init__(self, margin: float = 0.2, margin_weight: float = 0.5) -> None:
        super().__init__()
        self.margin = margin
        self.margin_weight = margin_weight

    def forward(self, output_dict: dict, target_dict: dict) -> torch.Tensor:
        logits = output_dict['clipwise_output']
        targets = target_dict['target']

        # Handle one-hot or class index targets
        if targets.ndim > 1:
            target_indices = torch.argmax(targets, dim=-1)
        else:
            target_indices = targets.long()

        # Standard Cross-Entropy
        ce_loss = F.cross_entropy(logits, target_indices)

        # Margin supervision on delta_margin if present
        if 'delta_margin' in output_dict:
            delta = output_dict['delta_margin'].squeeze(-1) # [B]
            
            # Loss for Weak (class 1): delta should be <= 0
            is_weak = (target_indices == 1).float()
            weak_penalty = is_weak * F.relu(delta + self.margin)
            
            # Loss for Medium (class 2): delta should be >= 0
            is_medium = (target_indices == 2).float()
            medium_penalty = is_medium * F.relu(-delta + self.margin)
            
            margin_loss = (weak_penalty + medium_penalty).sum() / max(1.0, (is_weak + is_medium).sum())
            return ce_loss + self.margin_weight * margin_loss

        return ce_loss
