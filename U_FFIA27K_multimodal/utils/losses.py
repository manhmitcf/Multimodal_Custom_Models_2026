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


class BilateralBoundaryLoss(BaseLoss):
    """
    Bilateral Boundary Loss for continuous feeding intensity assessment:
      1. Bilateral CORAL Ordinal Loss: Evaluates cumulative binary cross-entropy on cutoffs
         for Video, Audio, and Blended decision heads.
      2. Squared Earth Mover's Distance (EMD) Loss: Penalizes rank jumps quadratically
         (jumping 2 ranks penalized 4x, jumping 3 ranks penalized 9x).
      3. Cross-Modal Boundary Alignment Loss: Synchronizes Video and Audio boundary manifolds:
         L_align = mean_k (tanh(s_V - b_Vk) - tanh(s_A - b_Ak))^2.
    """
    def __init__(
        self,
        lambda_emd: float = 0.5,
        lambda_align: float = 0.2,
        lambda_unimodal: float = 0.5
    ) -> None:
        super().__init__()
        self.lambda_emd = lambda_emd
        self.lambda_align = lambda_align
        self.lambda_unimodal = lambda_unimodal

        # Raw dataset class index to Physical Ordinal Rank:
        # Raw 0: None   -> Rank 0
        # Raw 1: Strong -> Rank 3
        # Raw 2: Medium -> Rank 2
        # Raw 3: Weak   -> Rank 1
        self.register_buffer(
            "raw_to_rank",
            torch.tensor([0, 3, 2, 1], dtype=torch.long)
        )

    def _get_rank_targets(self, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts targets (indices or one-hot) into ordinal ranks and cumulative binary indicators.
        """
        if targets.ndim > 1 and targets.size(-1) > 1:
            y_raw = targets.argmax(dim=-1)
        else:
            y_raw = targets.long().squeeze()
            if y_raw.ndim == 0:
                y_raw = y_raw.unsqueeze(0)

        # Map to physical ordinal rank: [0, 1, 2, 3]
        rank = self.raw_to_rank[y_raw]

        # Cumulative binary exceedance targets: [B, 3]
        # k=1 (Rank >= 1), k=2 (Rank >= 2), k=3 (Rank >= 3)
        y_ge_1 = (rank >= 1).float()
        y_ge_2 = (rank >= 2).float()
        y_ge_3 = (rank >= 3).float()
        cum_targets = torch.stack([y_ge_1, y_ge_2, y_ge_3], dim=-1)

        return rank, cum_targets

    def forward(self, output_dict: dict, target_dict: dict, epoch: int = 1) -> torch.Tensor:
        targets = target_dict['target']
        rank, cum_targets = self._get_rank_targets(targets)

        # 1. Blended CORAL Exceedance BCE Loss
        cum_probs = output_dict.get("cum_probs")  # [B, 3]
        if cum_probs is not None:
            # Binary Cross Entropy on cumulative probabilities
            loss_coral_blended = F.binary_cross_entropy(
                torch.clamp(cum_probs, 1e-7, 1.0 - 1e-7),
                cum_targets
            )
        else:
            loss_coral_blended = torch.tensor(0.0, device=targets.device)

        # 2. Unimodal CORAL Losses for Video and Audio
        s_v = output_dict.get("score_v")
        s_a = output_dict.get("score_a")
        b_v = output_dict.get("cutoffs_v")
        b_a = output_dict.get("cutoffs_a")

        if s_v is not None and b_v is not None:
            # s_v: [B], b_v: [3] -> logits: [B, 3]
            logits_v = s_v.unsqueeze(-1) - b_v.unsqueeze(0)
            loss_coral_v = F.binary_cross_entropy_with_logits(logits_v, cum_targets)
        else:
            loss_coral_v = torch.tensor(0.0, device=targets.device)

        if s_a is not None and b_a is not None:
            logits_a = s_a.unsqueeze(-1) - b_a.unsqueeze(0)
            loss_coral_a = F.binary_cross_entropy_with_logits(logits_a, cum_targets)
        else:
            loss_coral_a = torch.tensor(0.0, device=targets.device)

        # 3. Squared Earth Mover's Distance (EMD) Loss
        probs = output_dict.get("probabilities")  # [B, 4] in raw order [None, Strong, Medium, Weak]
        if probs is not None:
            # Convert to ordinal rank probabilities: [None, Weak, Medium, Strong]
            p_rank = torch.stack([probs[:, 0], probs[:, 3], probs[:, 2], probs[:, 1]], dim=-1)
            # Predicted CDF across ranks 0, 1, 2
            cdf_pred = torch.cumsum(p_rank[:, :3], dim=-1)  # [B, 3]

            # Target one-hot & target CDF
            t_onehot = F.one_hot(rank, num_classes=4).float()
            cdf_target = torch.cumsum(t_onehot[:, :3], dim=-1)  # [B, 3]

            # Squared distance between CDFs
            loss_emd = torch.mean(torch.sum((cdf_pred - cdf_target) ** 2, dim=-1))
        else:
            loss_emd = torch.tensor(0.0, device=targets.device)

        # 4. Cross-Modal Boundary Alignment Loss
        if s_v is not None and s_a is not None and b_v is not None and b_a is not None:
            d_v = s_v.unsqueeze(-1) - b_v.unsqueeze(0)  # [B, 3]
            d_a = s_a.unsqueeze(-1) - b_a.unsqueeze(0)  # [B, 3]
            loss_align = torch.mean((torch.tanh(d_v) - torch.tanh(d_a)) ** 2)
        else:
            loss_align = torch.tensor(0.0, device=targets.device)

        # 5. Composite Bilateral Boundary Loss
        total_loss = (
            loss_coral_blended
            + self.lambda_unimodal * (loss_coral_v + loss_coral_a)
            + self.lambda_emd * loss_emd
            + self.lambda_align * loss_align
        )

        return total_loss


# Aliases for 100% backwards compatibility
OrdinalWassersteinEvidentialLoss = BilateralBoundaryLoss
TMCEvidentialLoss = BilateralBoundaryLoss
