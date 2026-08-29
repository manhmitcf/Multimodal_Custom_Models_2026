import torch.nn as nn

try:
    from torchvision.models import ViT_B_16_Weights, vit_b_16
except ImportError:
    from torchvision.models import vit_b_16
    ViT_B_16_Weights = None


class ViTBase16(nn.Module):
    """
    ViT-Base/16 image classifier with optional ImageNet pretrained weights.
    """
    def __init__(self, classes_num: int = 4, pretrained: bool = True) -> None:
        super().__init__()

        if ViT_B_16_Weights is None:
            self.model = vit_b_16(pretrained=pretrained)
        else:
            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            self.model = vit_b_16(weights=weights)

        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, classes_num)
        self.model_name = "vit_base_16"

    def get_name(self) -> str:
        return self.model_name

    def forward(self, x):
        return self.model(x)
