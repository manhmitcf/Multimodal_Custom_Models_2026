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
        from PIL import Image
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, np.ndarray):
            if image.ndim == 4 and image.shape[0] == 1:
                image = image.squeeze(0)
            if image.ndim != 3:
                raise ValueError(f"Expected image with 3 dimensions, got shape {tuple(image.shape)}")

            if image.shape[0] == 3 and image.shape[-1] != 3:
                image = image.transpose(1, 2, 0)
            elif image.shape[-1] != 3:
                raise ValueError(f"Expected RGB channel dimension with size 3, got shape {tuple(image.shape)}")

            return TF.to_pil_image(image)

        if isinstance(image, torch.Tensor):
            if image.ndim == 4 and image.shape[0] == 1:
                image = image.squeeze(0)
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
    VideoTransform supporting both single-frame and spatiotemporally consistent multi-frame clips.
    
    For multi-frame sequences (T frames):
    Applies IDENTICAL spatial augmentation (flip, rotation, affine shift) across all frames
    in the clip to prevent destroying temporal frame difference (It = It - It-1) and optical flow.
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

    def transform_clip(self, frames: Union[List, tuple, np.ndarray, torch.Tensor]) -> List[torch.Tensor]:
        """
        Transform a sequence of video frames with spatiotemporal consistency.
        Supports:
        - List or Tuple of 3D frames [H, W, C] or [C, H, W]
        - 4D NumPy array [T, H, W, C] or [T, C, H, W]
        - 4D PyTorch tensor [T, C, H, W] or [T, H, W, C]
        """
        if frames is None:
            return []

        # Unpack 4D tensor/array into list of 3D frames
        if hasattr(frames, "ndim") and frames.ndim == 4:
            frame_list = [frames[i] for i in range(frames.shape[0])]
        elif isinstance(frames, (list, tuple)):
            frame_list = list(frames)
        else:
            frame_list = [frames]

        if not frame_list:
            return []

        # Convert all frames to PIL Image
        pil_frames = [self.to_pil(f) for f in frame_list]

        # In train mode, sample geometric and photometric augmentation parameters ONCE per clip
        if self.is_train:
            do_flip = random.random() < 0.5
            angle = random.uniform(-15.0, 15.0)
            max_trans = int(0.1 * self.image_size)
            translate = (random.randint(-max_trans, max_trans), random.randint(-max_trans, max_trans))
            bright_factor = random.uniform(0.85, 1.15)
        else:
            do_flip = False
            angle = 0.0
            translate = (0, 0)
            bright_factor = 1.0

        transformed_tensors = []
        for img in pil_frames:
            # 1. Resize to target resolution
            img = TF.resize(img, (self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR)

            # 2. Consistent Photometric Augmentation
            if self.is_train and bright_factor != 1.0:
                img = TF.adjust_brightness(img, bright_factor)

            # 3. Consistent Geometric Augmentation
            if do_flip:
                img = TF.hflip(img)
            if angle != 0.0 or translate != (0, 0):
                img = TF.affine(
                    img,
                    angle=angle,
                    translate=translate,
                    scale=1.0,
                    shear=0.0,
                    interpolation=InterpolationMode.BILINEAR
                )

            # 4. ToTensor & Normalize with ImageNet stats
            tensor = TF.to_tensor(img)
            tensor = TF.normalize(tensor, mean=self.mean, std=self.std)
            transformed_tensors.append(tensor)

        return transformed_tensors

    def __call__(self, image: Union[np.ndarray, torch.Tensor, List, tuple]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Supports:
        - List or Tuple of frames -> returns List[torch.Tensor]
        - 4D array or tensor [T, H, W, C] / [T, C, H, W] -> returns List[torch.Tensor]
        - Single 3D image (np.ndarray [H, W, C] or torch.Tensor [C, H, W]) -> returns torch.Tensor [C, H, W]
        """
        if isinstance(image, (list, tuple)):
            return self.transform_clip(image)
        if hasattr(image, "ndim") and image.ndim == 4:
            return self.transform_clip(image)
        return self.transform_clip([image])[0]

    @staticmethod
    def get_transforms(image_size: int = 224):
        return {
            "train": VideoTransform(is_train=True, image_size=image_size),
            "val": VideoTransform(is_train=False, image_size=image_size),
            "test": VideoTransform(is_train=False, image_size=image_size),
        }
