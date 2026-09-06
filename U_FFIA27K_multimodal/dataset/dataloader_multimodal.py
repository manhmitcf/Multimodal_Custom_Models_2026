import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    DEFAULT_IMAGE_CACHE_ROOT,
    DEFAULT_AUDIO_CACHE_ROOT,
    VALID_CACHE_MODES,
    SplitterConfig,
)
from dataset.data_split import FishDataSplitter
from transforms.video_transform import VideoTransform
from features.audio_frontend import AudioFrontend

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FishMultimodalDataLoader:
    """
    Unified DataLoader Manager for Multimodal Fish Feeding Intensity Assessment.
    Synchronously supplies paired Video Frames [B, T, 3, H, W] and Audio Spectrograms [B, 1, Ta, 128].
    """
    def __init__(
        self,
        batch_size: int = 32,
        dataloader_workers: int = -1,
        prefetch_factor: Optional[int] = None,
        cache_mode: str = "ram",
        image_size: int = 224,
        frame_policy: str = "end",
        num_frames: int = 2,
        sample_rate: int = 64000,
        splitter_config: Optional[SplitterConfig] = None,
    ) -> None:
        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers
        self.prefetch_factor = prefetch_factor
        self.cache_mode = cache_mode.lower()
        if self.cache_mode not in VALID_CACHE_MODES:
            raise ValueError(f"Invalid cache_mode='{cache_mode}'. Expected one of {sorted(VALID_CACHE_MODES)}.")

        self.image_size = image_size
        self.frame_policy = frame_policy
        self.num_frames = num_frames
        self.sample_rate = sample_rate

        if self.dataloader_workers == -1:
            max_cpu = os.cpu_count()
            if max_cpu is None or max_cpu <= 0:
                self.dataloader_workers = 0
            elif max_cpu == 2:
                self.dataloader_workers = 1
            else:
                self.dataloader_workers = (max_cpu // 2) + 1

        self.splitter_config = splitter_config if splitter_config is not None else SplitterConfig(include_video=True)
        self.splitter_config.include_video = True
        
        try:
            self.splitter = FishDataSplitter(config=self.splitter_config)
            self.train_dict, self.test_dict, self.val_dict = self.splitter.split_data()
        except Exception as exc:
            logger.warning(f"Could not load dataset from '{self.splitter_config.dataset_path}' ({exc}). Mock splits initialized.")
            self.train_dict, self.test_dict, self.val_dict = [], [], []

        logger.info("==================================================")
        logger.info("Initializing FishMultimodalDataLoader:")
        logger.info(f"  - Batch Size:               {self.batch_size}")
        logger.info(f"  - DataLoader Workers:       {self.dataloader_workers}")
        logger.info(f"  - Cache Mode:               {self.cache_mode}")
        logger.info(f"  - Image Resolution:         {self.image_size}x{self.image_size}")
        logger.info(f"  - Number of Frames:         {self.num_frames}")
        logger.info(f"  - Audio Sample Rate:        {self.sample_rate} Hz")
        logger.info("==================================================")

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        clip_names = [data['clip_name'] for data in batch]
        targets = [data['target'] for data in batch]

        video_tensors = torch.stack([data['video_form'] for data in batch])
        audio_tensors = torch.stack([data['audio_form'] for data in batch])
        targets_tensor = torch.FloatTensor(np.array(targets))

        return {
            'clip_name': clip_names,
            'video_form': video_tensors,
            'audio_form': audio_tensors,
            'target': targets_tensor
        }

    class _InnerDataset(Dataset):
        def __init__(self, parent: 'FishMultimodalDataLoader', split: str) -> None:
            self.parent = parent
            self.split = split
            
            if self.split == 'train':
                self.data_dict = parent.train_dict
            elif self.split == 'test':
                self.data_dict = parent.test_dict
            elif self.split == 'val':
                self.data_dict = parent.val_dict
            else:
                raise ValueError(f"Invalid split value '{self.split}'.")

            transforms = VideoTransform.get_transforms(image_size=parent.image_size)
            self.transform = transforms[self.split]

        def __len__(self) -> int:
            return len(self.data_dict)

        def _decode_video_frames(self, video_path: str) -> torch.Tensor:
            import cv2
            frames = []
            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    if self.parent.num_frames == 2:
                        # First frame (0) and Last frame (total_frames - 1)
                        indices = [0, max(0, total_frames - 1)]
                    else:
                        indices = np.linspace(0, total_frames - 1, self.parent.num_frames).astype(int)

                    for idx in indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            tensor = self.transform(frame)
                            frames.append(tensor)
                cap.release()

            while len(frames) < self.parent.num_frames:
                frames.append(torch.zeros(3, self.parent.image_size, self.parent.image_size, dtype=torch.float32))

            return torch.stack(frames[:self.parent.num_frames])  # [T, 3, H, W]

        def _decode_audio_waveform(self, audio_path: str) -> torch.Tensor:
            target_length = self.parent.sample_rate * 2  # 2 seconds (128,000 samples)
            if os.path.exists(audio_path):
                try:
                    import torchaudio
                    waveform, sr = torchaudio.load(audio_path)
                    if sr != self.parent.sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.parent.sample_rate)
                        waveform = resampler(waveform)
                    if waveform.ndim == 2 and waveform.size(0) > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    y = waveform.squeeze(0).to(torch.float32)
                    if y.numel() > target_length:
                        y = y[:target_length]
                    elif y.numel() < target_length:
                        y = torch.nn.functional.pad(y, (0, target_length - y.numel()))
                    return y
                except Exception as exc:
                    logger.debug(f"Could not load audio '{audio_path}': {exc}")

            return torch.zeros(target_length, dtype=torch.float32)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            item = self.data_dict[idx]
            
            # data_split.py generates [audio_path, video_path, label]
            if isinstance(item, (list, tuple)):
                if len(item) == 3:
                    audio_path = str(item[0])
                    video_path = str(item[1])
                    target = item[2]
                elif len(item) == 2:
                    p0 = str(item[0])
                    target = item[1]
                    if "_audio_" in p0 or p0.endswith(".wav"):
                        audio_path = p0
                        video_path = ""
                    else:
                        video_path = p0
                        audio_path = ""
                else:
                    audio_path, video_path, target = "", "", 0
            elif isinstance(item, dict):
                audio_path = str(item.get('audio_path', ''))
                video_path = str(item.get('video_path', ''))
                target = item.get('label', item.get('target', 0))
            else:
                audio_path, video_path, target = "", "", 0

            # Fallback path pairing if one is missing
            if not audio_path and video_path:
                cand = video_path.replace("/video/", "/audio/").replace("\\video\\", "\\audio\\").replace("_video_", "_audio_")
                if cand.endswith(".mp4"):
                    cand = cand[:-4] + ".wav"
                if os.path.exists(cand):
                    audio_path = cand
            if not video_path and audio_path:
                cand = audio_path.replace("/audio/", "/video/").replace("\\audio\\", "\\video\\").replace("_audio_", "_video_")
                if cand.endswith(".wav"):
                    cand = cand[:-4] + ".mp4"
                if os.path.exists(cand):
                    video_path = cand

            # Convert target to one-hot encoding [4]
            if isinstance(target, (int, np.integer)):
                target_onehot = np.zeros(4, dtype=np.float32)
                if 0 <= int(target) < 4:
                    target_onehot[int(target)] = 1.0
            elif isinstance(target, str):
                target_str = target.strip().lower()
                class_to_idx = {"none": 0, "strong": 1, "medium": 2, "weak": 3}
                target_onehot = np.zeros(4, dtype=np.float32)
                if target_str in class_to_idx:
                    target_onehot[class_to_idx[target_str]] = 1.0
                else:
                    try:
                        idx = int(target_str)
                        if 0 <= idx < 4:
                            target_onehot[idx] = 1.0
                    except ValueError:
                        pass
            else:
                try:
                    arr = np.array(target, dtype=np.float32)
                    if arr.size == 4:
                        target_onehot = arr.reshape(4)
                    else:
                        target_onehot = np.zeros(4, dtype=np.float32)
                        target_onehot[int(arr.item())] = 1.0
                except Exception:
                    target_onehot = np.zeros(4, dtype=np.float32)

            video_tensor = self._decode_video_frames(video_path)
            audio_tensor = self._decode_audio_waveform(audio_path)

            return {
                'clip_name': os.path.basename(video_path or audio_path or f"sample_{idx}"),
                'video_form': video_tensor,
                'audio_form': audio_tensor,
                'target': target_onehot
            }

    def get_data_loader(self, split: str, shuffle: bool = False) -> DataLoader:
        dataset = self._InnerDataset(self, split)
        loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': shuffle,
            'num_workers': self.dataloader_workers,
            'collate_fn': self.collate_fn,
            'pin_memory': True,
        }
        if self.dataloader_workers > 0 and self.prefetch_factor is not None:
            loader_kwargs['prefetch_factor'] = self.prefetch_factor

        return DataLoader(dataset, **loader_kwargs)
