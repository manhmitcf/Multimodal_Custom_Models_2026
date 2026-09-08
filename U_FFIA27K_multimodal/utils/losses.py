import torch
import torch.nn as nn
import torch.nn.functional as F


from .ordinal_loss import OrdinalWassersteinEvidentialLoss


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
            # Targets provided as one-hot or probability distribution
            return F.cross_entropy(logits, targets)
        return F.cross_entropy(logits, targets.long())


class TMCEvidentialLoss(BaseLoss):
    """
    Evidential Dirichlet Loss for Trusted Multi-View Classification (TMC, Han et al., NeurIPS 2021).
    Optimizes evidential distribution and epistemic uncertainty across Video, Audio, and Fused views.
    """
    def __init__(self, annealing_epochs: int = 20, lambda_epochs: int = 50) -> None:
        super().__init__()
        self.annealing_epochs = annealing_epochs
        self.lambda_epochs = lambda_epochs

    def _dirichlet_loss(self, alpha: torch.Tensor, targets: torch.Tensor, epoch: int = 1) -> torch.Tensor:
        # alpha: [B, K] > 1, targets: [B, K] one-hot
        S = torch.sum(alpha, dim=-1, keepdim=True)
        # Expected cross-entropy loss under Dirichlet
        loss_ace = torch.sum(targets * (torch.digamma(S) - torch.digamma(alpha)), dim=-1, keepdim=True)

        # KL divergence regularizer for non-target evidence
        alpha_tilde = targets + (1.0 - targets) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=-1, keepdim=True)
        K = alpha.size(-1)

        first_term = (
            torch.lgamma(S_tilde)
            - torch.sum(torch.lgamma(alpha_tilde), dim=-1, keepdim=True)
            - torch.lgamma(torch.tensor(float(K), device=alpha.device))
            + torch.sum(torch.lgamma(torch.ones(1, K, device=alpha.device)), dim=-1, keepdim=True)
        )
        second_term = torch.sum(
            (alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)),
            dim=-1,
            keepdim=True
        )
        kl = first_term + second_term
        annealing_coef = min(1.0, float(epoch) / float(self.lambda_epochs))
        return torch.mean(loss_ace + annealing_coef * kl)

    def forward(self, output_dict: dict, target_dict: dict, epoch: int = 1) -> torch.Tensor:
        targets = target_dict['target']
        if targets.ndim == 1 or (targets.ndim == 2 and targets.size(-1) == 1):
            targets = F.one_hot(targets.long().squeeze(), num_classes=4).float()

        # Standard CE Loss on final logits
        ce_loss = F.cross_entropy(output_dict['clipwise_output'], targets)

        # Evidential losses on alpha parameters
        if "alpha_final" in output_dict:
            alpha_f = output_dict["alpha_final"]
            ev_loss = self._dirichlet_loss(alpha_f, targets, epoch)
            return ce_loss + 0.5 * ev_loss

        return ce_loss
