import random
from typing import List, Union, Optional
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class ImageToPIL:
    """Convert one RGB image in [C, H, W] or [H, W, C] format to PIL."""
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            if image.ndim == 4:
                # Array of frames [T, H, W, C]
                return [self.__call__(f) for f in image]

            if image.ndim != 3:
                raise ValueError(f"Expected image with 3 dimensions, got shape {tuple(image.shape)}")

            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            elif image.shape[-1] != 3:
                raise ValueError(f"Expected RGB channel dimension with size 3, got shape {tuple(image.shape)}")

            return TF.to_pil_image(image)

        if isinstance(image, torch.Tensor):
            if image.ndim == 4:
                # Tensor of frames [T, C, H, W]
                return [self.__call__(f) for f in image]

            if image.ndim != 3:
                raise ValueError(f"Expected image with 3 dimensions, got shape {tuple(image.shape)}")

            if image.shape[0] != 3 and image.shape[-1] == 3:
                image = image.permute(2, 0, 1)
            elif image.shape[0] != 3:
                raise ValueError(f"Expected RGB channel dimension with size 3, got shape {tuple(image.shape)}")

            return TF.to_pil_image(image)

        return image


class VideoTransform:
    """
    Video transform pipeline applied uniformly across video frames.
    Identical to multimodal branches.

    Input:  np.ndarray [H, W, C] or [C, H, W] uint8
    Output: torch.Tensor [C, H, W] float32 normalized with ImageNet statistics.
    """
    def __init__(self, transform: transforms.Compose) -> None:
        self.transform = transform

    def __call__(self, image: Union[np.ndarray, torch.Tensor, List]) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(image, (list, tuple)):
            return [self.transform(f) for f in image]
        return self.transform(image)

    @staticmethod
    def get_transforms(image_size: int = 224):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        train_transform = [
            ImageToPIL(),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

        eval_transform = [
            ImageToPIL(),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

        return {
            "train": VideoTransform(transforms.Compose(train_transform)),
            "val": VideoTransform(transforms.Compose(eval_transform)),
            "test": VideoTransform(transforms.Compose(eval_transform)),
        }

