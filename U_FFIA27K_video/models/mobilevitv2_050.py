import torch.nn as nn
import timm

class MobileViTv2_050(nn.Module):
    """
    MobileViTv2-0.50 image classifier using timm.
    """
    def __init__(self, classes_num: int = 4, pretrained: bool = True) -> None:
        super().__init__()
        
        self.model = timm.create_model("mobilevitv2_050", pretrained=pretrained, num_classes=classes_num)
        self.model_name = "mobilevitv2_050"

    def get_name(self) -> str:
        return self.model_name

    def forward(self, x):
        return self.model(x)
