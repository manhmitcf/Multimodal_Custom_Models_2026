import torch.nn as nn
import timm

class MobileViTXS(nn.Module):
    """
    MobileViT-XS image classifier using timm.
    """
    def __init__(self, classes_num: int = 4, pretrained: bool = True) -> None:
        super().__init__()
        
        self.model = timm.create_model("mobilevit_xs", pretrained=pretrained, num_classes=classes_num)
        self.model_name = "mobilevit_xs"

    def get_name(self) -> str:
        return self.model_name

    def forward(self, x):
        return self.model(x)
