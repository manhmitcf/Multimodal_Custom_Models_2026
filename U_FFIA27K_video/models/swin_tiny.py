import torch.nn as nn

try:
    from torchvision.models import Swin_T_Weights, swin_t
except ImportError:
    from torchvision.models import swin_t
    Swin_T_Weights = None


class SwinTiny(nn.Module):
    """
    Swin-Tiny image classifier with optional ImageNet pretrained weights.
    """
    def __init__(self, classes_num: int = 4, pretrained: bool = True) -> None:
        super().__init__()

        if Swin_T_Weights is None:
            self.model = swin_t(pretrained=pretrained)
        else:
            weights = Swin_T_Weights.DEFAULT if pretrained else None
            self.model = swin_t(weights=weights)

        self.model.head = nn.Linear(self.model.head.in_features, classes_num)
        self.model_name = "swin_tiny"

    def get_name(self) -> str:
        return self.model_name

    def forward(self, x):
        return self.model(x)
