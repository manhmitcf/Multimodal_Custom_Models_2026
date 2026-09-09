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
    VideoTransform for multi-frame clips with ZERO data augmentation (No-Aug).
    Applies pure deterministic preprocessing across all frames:
      1. Resize to (image_size, image_size)
      2. ToTensor()
      3. ImageNet normalization
    Preserves exact pixel values, true optical flow, and pristine temporal differences.
    """
    def __init__(
        self,
        is_train: bool = False,
        image_size: int = 224,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225)
    ) -> None:
        self.is_train = is_train
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.to_pil = ImageToPIL()

    def transform_clip(self, frames: List[Union[np.ndarray, torch.Tensor]]) -> List[torch.Tensor]:
        """
        Pure deterministic transform of video frames with NO augmentation.
        """
        if not frames:
            return []

        pil_frames = [self.to_pil(f) if not hasattr(f, 'convert') else f for f in frames]

        transformed_tensors = []
        for img in pil_frames:
            # 1. Resize to target resolution
            img = TF.resize(img, (self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR)

            # 2. ToTensor & Normalize with ImageNet stats (NO AGU)
            tensor = TF.to_tensor(img)
            tensor = TF.normalize(tensor, mean=self.mean, std=self.std)
            transformed_tensors.append(tensor)

        return transformed_tensors

    def __call__(self, image: Union[np.ndarray, torch.Tensor, List]) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(image, (list, tuple)):
            return self.transform_clip(list(image))
        if isinstance(image, np.ndarray) and image.ndim == 4:
            return self.transform_clip(list(image))
        return self.transform_clip([image])[0]

    @staticmethod
    def get_transforms(image_size: int = 224):
        return {
            "train": VideoTransform(is_train=False, image_size=image_size),
            "val": VideoTransform(is_train=False, image_size=image_size),
            "test": VideoTransform(is_train=False, image_size=image_size),
        }

