import torch.nn as nn

try:
    from torchvision.models import ResNet50_Weights, resnet50
except ImportError:
    from torchvision.models import resnet50
    ResNet50_Weights = None


class ResNet50(nn.Module):
    """
    ResNet50 image classifier with optional ImageNet pretrained weights.
    """
    def __init__(self, classes_num: int = 4, pretrained: bool = True) -> None:
        super().__init__()

        if ResNet50_Weights is None:
            self.model = resnet50(pretrained=pretrained)
        else:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.model = resnet50(weights=weights)

        self.model.fc = nn.Linear(self.model.fc.in_features, classes_num)
        self.model_name = "resnet50"

    def get_name(self) -> str:
        return self.model_name

    def forward(self, x):
        return self.model(x)
