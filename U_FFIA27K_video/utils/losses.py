import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseLoss(nn.Module):
    """
    Abstract Base Class (ABC / Interface) standardizing custom loss functions in the project.
    Complies with PyTorch OOP design guidelines.
    """
    def __init__(self) -> None:
        super(BaseLoss, self).__init__()

    def forward(self, output_dict: dict, target_dict: dict) -> torch.Tensor:
        """
        Compute loss value. Subclasses must override this method.
        """
        raise NotImplementedError("Method 'forward' must be implemented in subclasses.")


class ClipCELoss(BaseLoss):
    """
    Multi-class Cross Entropy Loss at the clip level.
    Inputs:
      - output_dict: Dict containing predicted classification logits 'clipwise_output' [Batch, Num_Classes]
      - target_dict: Dict containing one-hot encoded ground truth targets 'target' [Batch, Num_Classes]
    """
    def forward(self, output_dict: dict, target_dict: dict) -> torch.Tensor:
        return F.cross_entropy(output_dict['clipwise_output'], target_dict['target'])


class ClipBCELoss(BaseLoss):
    """
    Binary Cross Entropy Loss at the clip level.
    Inputs:
      - output_dict: Dict containing predicted classification logits 'clipwise_output' [Batch, Num_Classes]
      - target_dict: Dict containing one-hot encoded ground truth targets 'target' [Batch, Num_Classes]
    """
    def forward(self, output_dict: dict, target_dict: dict) -> torch.Tensor:
        return F.binary_cross_entropy(output_dict['clipwise_output'], target_dict['target'])
