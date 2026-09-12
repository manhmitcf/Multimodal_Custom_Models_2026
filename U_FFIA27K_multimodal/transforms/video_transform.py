import random
from typing import Union, List, Optional
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class ConsistentVideoTransform:
    """
    Clip-Consistent Video Transform Pipeline.
    Guarantees temporal coherence across all T frames in a video clip:
      - Random Horizontal Flip: Decided once per clip, applied identically to all frames.
      - Random Rotation: Angle sampled once per clip, applied identically to all frames.
      - Color Jitter (Brightness/Contrast): Factors sampled once per clip.
      - Bilinear Resize: Standardized HxW.
      - ImageNet Normalization.

    Accepts:
      - 4D np.ndarray [T, H, W, 3] or [T, 3, H, W]
      - List of 3D frames [frame_0, frame_1, ...]
      - Single 3D frame [H, W, 3] or [3, H, W] (backward compatibility)
    Returns:
      - torch.Tensor [T, 3, H, W] or [3, H, W]
    """
    def __init__(
        self,
        image_size: int = 224,
        is_train: bool = True,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
    ) -> None:
        self.image_size = int(image_size)
        self.is_train = bool(is_train)
        self.mean = mean
        self.std = std

    def _to_pil(self, frame: Union[np.ndarray, torch.Tensor]):
        if isinstance(frame, np.ndarray):
            if frame.ndim == 3 and frame.shape[0] == 3:
                frame = frame.transpose(1, 2, 0)
            return TF.to_pil_image(frame)
        elif isinstance(frame, torch.Tensor):
            if frame.ndim == 3 and frame.shape[-1] == 3:
                frame = frame.permute(2, 0, 1)
            return TF.to_pil_image(frame)
        return frame

    def __call__(
        self,
        frames: Union[np.ndarray, List[np.ndarray], torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        is_single_frame = False
        if isinstance(frames, (list, tuple)):
            frame_list = list(frames)
        elif isinstance(frames, (np.ndarray, torch.Tensor)):
            if frames.ndim == 3:
                is_single_frame = True
                frame_list = [frames]
            elif frames.ndim == 4:
                frame_list = [frames[i] for i in range(frames.shape[0])]
            else:
                raise ValueError(f"Expected 3D or 4D video tensor/array, got ndim={frames.ndim}")
        else:
            is_single_frame = True
            frame_list = [frames]

        # Sample augmentation parameters ONCE per video clip (temporal synchronization)
        if self.is_train:
            do_flip = (random.random() < 0.5)
            rot_angle = random.uniform(-15.0, 15.0)
            brightness_factor = random.uniform(0.85, 1.15)
            contrast_factor = random.uniform(0.85, 1.15)

        transformed_tensors = []
        for frame in frame_list:
            pil_img = self._to_pil(frame)
            pil_img = TF.resize(
                pil_img,
                (self.image_size, self.image_size),
                interpolation=InterpolationMode.BILINEAR
            )
            if self.is_train:
                pil_img = TF.adjust_brightness(pil_img, brightness_factor)
                pil_img = TF.adjust_contrast(pil_img, contrast_factor)
                if do_flip:
                    pil_img = TF.hflip(pil_img)
                pil_img = TF.rotate(
                    pil_img,
                    rot_angle,
                    interpolation=InterpolationMode.BILINEAR
                )

            t_img = TF.to_tensor(pil_img)
            t_img = TF.normalize(t_img, self.mean, self.std)
            transformed_tensors.append(t_img)

        if is_single_frame:
            return transformed_tensors[0]
        return torch.stack(transformed_tensors, dim=0)

    @classmethod
    def get_transforms(cls, image_size: int = 224):
        return {
            "train": cls(image_size=image_size, is_train=True),
            "val": cls(image_size=image_size, is_train=False),
            "test": cls(image_size=image_size, is_train=False),
        }


# Canonical alias for 100% backward compatibility
VideoTransform = ConsistentVideoTransform
