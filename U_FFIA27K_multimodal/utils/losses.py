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
