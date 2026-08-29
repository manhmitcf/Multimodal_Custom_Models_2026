import torch.nn as nn

_CONVNEXT_IMPORT_ERROR = None

try:
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
except ImportError:
    ConvNeXt_Tiny_Weights = None
    try:
        from torchvision.models import convnext_tiny
    except ImportError as exc:
        convnext_tiny = None
        _CONVNEXT_IMPORT_ERROR = exc


class ConvNeXtTiny(nn.Module):
    """
    ConvNeXt-Tiny image classifier with optional ImageNet pretrained weights.
    """
    def __init__(self, classes_num: int = 4, pretrained: bool = True) -> None:
        super().__init__()

        if convnext_tiny is None:
            raise ImportError(
                "ConvNeXt-Tiny requires a torchvision version that provides "
                "torchvision.models.convnext_tiny."
            ) from _CONVNEXT_IMPORT_ERROR

        if ConvNeXt_Tiny_Weights is None:
            self.model = convnext_tiny(pretrained=pretrained)
        else:
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.model = convnext_tiny(weights=weights)

        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, classes_num)
        self.model_name = "convnext_tiny"

    def get_name(self) -> str:
        return self.model_name

    def forward(self, x):
        return self.model(x)
