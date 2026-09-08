import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional


class BaseLoss(nn.Module):
    """
    Abstract Base Class for loss functions.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, output_dict: dict, target_dict: dict, epoch: int = 1) -> torch.Tensor:
        raise NotImplementedError("Method 'forward' must be implemented in subclasses.")


class ClipCELoss(BaseLoss):
    """
    Standard Multi-class Cross Entropy Loss at the clip level.
    """
    def forward(self, output_dict: dict, target_dict: dict, epoch: int = 1) -> torch.Tensor:
        logits = output_dict['clipwise_output']
        targets = target_dict['target']
        if targets.ndim > 1 and targets.size(-1) > 1:
            return F.cross_entropy(logits, targets)
        return F.cross_entropy(logits, targets.long())


class PairwiseTournamentLoss(BaseLoss):
    """
    Hierarchical Pairwise Cross-Boundary Tournament Loss.
    Optimizes 2-level tournament decision:
      Level 1: Activity Gating Loss (None vs Active Feeding).
      Level 2: 3 Pairwise Cross-Boundary Losses:
               - B12: Weak vs Medium
               - B23: Medium vs Strong
               - B13: Weak vs Strong (Cross-boundary jumping protection)
      Level 3: End-to-End Multi-Class Cross Entropy on final tournament voting probabilities.
    """
    def __init__(
        self,
        weight_act: float = 0.5,
        weight_pairwise: float = 0.5,
        weight_ce: float = 1.0,
        **kwargs
    ) -> None:
        super().__init__()
        self.weight_act = float(kwargs.get("weight_act", weight_act))
        self.weight_pairwise = float(kwargs.get("weight_pairwise", weight_pairwise))
        self.weight_ce = float(kwargs.get("weight_ce", weight_ce))

    def _get_raw_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim > 1 and targets.size(-1) > 1:
            y_raw = targets.argmax(dim=-1)
        else:
            y_raw = targets.long().squeeze()
            if y_raw.ndim == 0:
                y_raw = y_raw.unsqueeze(0)
        return y_raw

    def forward(self, output_dict: dict, target_dict: dict, epoch: int = 1) -> torch.Tensor:
        targets = target_dict['target']
        y_raw = self._get_raw_targets(targets)  # [B]: 0=None, 1=Strong, 2=Medium, 3=Weak

        # 1. Level 1: Feeding Activity Gating Loss (None=0 vs Feeding=1)
        # Active feeding: Strong (1), Medium (2), Weak (3) -> Label 1.0
        # Inactive: None (0) -> Label 0.0
        logit_act = output_dict.get("logit_act")
        if logit_act is not None:
            target_act = (y_raw != 0).float()
            loss_act = F.binary_cross_entropy_with_logits(logit_act, target_act)
        else:
            loss_act = torch.tensor(0.0, device=targets.device)

        # 2. Level 2: 3 Pairwise Cross-Boundary Head Losses
        logit_12 = output_dict.get("logit_12")  # Positive -> Weak, Negative -> Medium
        logit_23 = output_dict.get("logit_23")  # Positive -> Medium, Negative -> Strong
        logit_13 = output_dict.get("logit_13")  # Positive -> Weak, Negative -> Strong

        # B12: Weak (3) vs Medium (2)
        mask_12 = (y_raw == 3) | (y_raw == 2)
        if mask_12.sum() > 0 and logit_12 is not None:
            target_12 = (y_raw[mask_12] == 3).float()  # 1.0 if Weak, 0.0 if Medium
            loss_12 = F.binary_cross_entropy_with_logits(logit_12[mask_12], target_12)
        else:
            loss_12 = torch.tensor(0.0, device=targets.device)

        # B23: Medium (2) vs Strong (1)
        mask_23 = (y_raw == 2) | (y_raw == 1)
        if mask_23.sum() > 0 and logit_23 is not None:
            target_23 = (y_raw[mask_23] == 2).float()  # 1.0 if Medium, 0.0 if Strong
            loss_23 = F.binary_cross_entropy_with_logits(logit_23[mask_23], target_23)
        else:
            loss_23 = torch.tensor(0.0, device=targets.device)

        # B13: Weak (3) vs Strong (1) [Cross-Protection Boundary]
        mask_13 = (y_raw == 3) | (y_raw == 1)
        if mask_13.sum() > 0 and logit_13 is not None:
            target_13 = (y_raw[mask_13] == 3).float()  # 1.0 if Weak, 0.0 if Strong
            loss_13 = F.binary_cross_entropy_with_logits(logit_13[mask_13], target_13)
        else:
            loss_13 = torch.tensor(0.0, device=targets.device)

        loss_pairwise = (loss_12 + loss_23 + loss_13) / 3.0

        # 3. Level 3: Multi-class Cross Entropy on Final Logits
        logits = output_dict.get("clipwise_output", output_dict.get("logits"))
        if logits is not None:
            loss_ce = F.cross_entropy(logits, y_raw)
        else:
            loss_ce = torch.tensor(0.0, device=targets.device)

        # Total Composite Tournament Loss
        total_loss = (
            self.weight_ce * loss_ce +
            self.weight_act * loss_act +
            self.weight_pairwise * loss_pairwise
        )

        return total_loss


# Backward compatibility aliases
BilateralBoundaryLoss = PairwiseTournamentLoss
OrdinalWassersteinEvidentialLoss = PairwiseTournamentLoss
TMCEvidentialLoss = PairwiseTournamentLoss
