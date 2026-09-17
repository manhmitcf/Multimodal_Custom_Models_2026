import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Flat 4-Class Round-Robin Tournament Loss.
    Optimizes 6 pairwise cross-boundary expert heads + Global CE + Auxiliary supervision.
      - B01: None (0) vs Strong (1) [w = 0.05]
      - B02: None (0) vs Medium (2) [w = 0.10]
      - B03: None (0) vs Weak (3)   [w = 0.25]
      - B12: Strong (1) vs Medium (2) [w = 0.25]
      - B23: Medium (2) vs Weak (3)   [w = 0.25]
      - B13: Strong (1) vs Weak (3)   [w = 0.10]
    
    Total Loss:
      L_total = weight_ce * L_ce + weight_pairwise * L_pairwise + aux_loss_weight * (L_v + L_a)
    """
    def __init__(
        self,
        weight_pairwise: float = 1.0,
        weight_ce: float = 1.0,
        aux_loss_weight: float = 0.3,
        only_backbones: bool = False,
        w_03: float = 0.25,
        w_23: float = 0.25,
        w_12: float = 0.25,
        w_02: float = 0.10,
        w_13: float = 0.10,
        w_01: float = 0.05,
        **kwargs
    ) -> None:
        super().__init__()
        self.weight_pairwise = float(kwargs.get("weight_pairwise", weight_pairwise))
        self.weight_ce = float(kwargs.get("weight_ce", weight_ce))
        self.aux_loss_weight = float(kwargs.get("aux_loss_weight", aux_loss_weight))
        self.only_backbones = bool(kwargs.get("only_backbones", only_backbones))

        # Pairwise matchup weights
        self.w_03 = float(kwargs.get("w_03", w_03))
        self.w_23 = float(kwargs.get("w_23", w_23))
        self.w_12 = float(kwargs.get("w_12", w_12))
        self.w_02 = float(kwargs.get("w_02", w_02))
        self.w_13 = float(kwargs.get("w_13", w_13))
        self.w_01 = float(kwargs.get("w_01", w_01))

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

        # Auxiliary Unimodal Backbone Losses
        logits_v = output_dict.get("logits_video")
        logits_a = output_dict.get("logits_audio")
        loss_v = F.cross_entropy(logits_v, y_raw) if logits_v is not None else torch.tensor(0.0, device=targets.device)
        loss_a = F.cross_entropy(logits_a, y_raw) if logits_a is not None else torch.tensor(0.0, device=targets.device)

        if self.only_backbones:
            # Phase 1: train only unimodal backbones with clean auxiliary supervision
            return loss_v + loss_a

        # ----------------------------------------------------------------------
        # 6 Pairwise Cross-Boundary Head Losses (Each sample trains exactly 3 heads)
        # ----------------------------------------------------------------------
        # B01: None (0) vs Strong (1)
        logit_01 = output_dict.get("logit_01")
        mask_01 = (y_raw == 0) | (y_raw == 1)
        if mask_01.sum() > 0 and logit_01 is not None:
            target_01 = (y_raw[mask_01] == 0).float()  # 1.0 if None, 0.0 if Strong
            loss_01 = F.binary_cross_entropy_with_logits(logit_01[mask_01], target_01)
        else:
            loss_01 = torch.tensor(0.0, device=targets.device)

        # B02: None (0) vs Medium (2)
        logit_02 = output_dict.get("logit_02")
        mask_02 = (y_raw == 0) | (y_raw == 2)
        if mask_02.sum() > 0 and logit_02 is not None:
            target_02 = (y_raw[mask_02] == 0).float()  # 1.0 if None, 0.0 if Medium
            loss_02 = F.binary_cross_entropy_with_logits(logit_02[mask_02], target_02)
        else:
            loss_02 = torch.tensor(0.0, device=targets.device)

        # B03: None (0) vs Weak (3) [Critical Entry Boundary]
        logit_03 = output_dict.get("logit_03")
        mask_03 = (y_raw == 0) | (y_raw == 3)
        if mask_03.sum() > 0 and logit_03 is not None:
            target_03 = (y_raw[mask_03] == 0).float()  # 1.0 if None, 0.0 if Weak
            loss_03 = F.binary_cross_entropy_with_logits(logit_03[mask_03], target_03)
        else:
            loss_03 = torch.tensor(0.0, device=targets.device)

        # B12: Strong (1) vs Medium (2) [Critical Burst Boundary]
        logit_12 = output_dict.get("logit_12")
        mask_12 = (y_raw == 1) | (y_raw == 2)
        if mask_12.sum() > 0 and logit_12 is not None:
            target_12 = (y_raw[mask_12] == 1).float()  # 1.0 if Strong, 0.0 if Medium
            loss_12 = F.binary_cross_entropy_with_logits(logit_12[mask_12], target_12)
        else:
            loss_12 = torch.tensor(0.0, device=targets.device)

        # B23: Medium (2) vs Weak (3) [Critical Transition Boundary]
        logit_23 = output_dict.get("logit_23")
        mask_23 = (y_raw == 2) | (y_raw == 3)
        if mask_23.sum() > 0 and logit_23 is not None:
            target_23 = (y_raw[mask_23] == 2).float()  # 1.0 if Medium, 0.0 if Weak
            loss_23 = F.binary_cross_entropy_with_logits(logit_23[mask_23], target_23)
        else:
            loss_23 = torch.tensor(0.0, device=targets.device)

        # B13: Strong (1) vs Weak (3) [Anchor Protection Boundary]
        logit_13 = output_dict.get("logit_13")
        mask_13 = (y_raw == 1) | (y_raw == 3)
        if mask_13.sum() > 0 and logit_13 is not None:
            target_13 = (y_raw[mask_13] == 1).float()  # 1.0 if Strong, 0.0 if Weak
            loss_13 = F.binary_cross_entropy_with_logits(logit_13[mask_13], target_13)
        else:
            loss_13 = torch.tensor(0.0, device=targets.device)

        # Boundary-weighted Pairwise Loss
        loss_pairwise = (
            self.w_03 * loss_03 +
            self.w_23 * loss_23 +
            self.w_12 * loss_12 +
            self.w_02 * loss_02 +
            self.w_13 * loss_13 +
            self.w_01 * loss_01
        )

        # Global Multi-class Cross Entropy on Final Logits
        logits = output_dict.get("clipwise_output", output_dict.get("logits"))
        if logits is not None:
            loss_ce = F.cross_entropy(logits, y_raw)
        else:
            loss_ce = torch.tensor(0.0, device=targets.device)

        # Total Composite Tournament Loss + Auxiliary Regularization
        total_loss = (
            self.weight_ce * loss_ce +
            self.weight_pairwise * loss_pairwise +
            self.aux_loss_weight * (loss_v + loss_a)
        )

        return total_loss
