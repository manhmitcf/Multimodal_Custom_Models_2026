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


def _decode_video_frames_raw(video_path: str, image_size: int = 224, num_frames: int = 2) -> np.ndarray:
    """Decode video frames into uint8 NumPy array [num_frames, image_size, image_size, 3] RGB."""
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, width=image_size, height=image_size, ctx=cpu(0))
        total = len(vr)
        if total > 0:
            indices = [0, max(0, total - 1)] if num_frames == 2 else np.linspace(0, total - 1, num_frames).astype(int).tolist()
            return vr.get_batch(indices).asnumpy()
    except Exception:
        pass

    import cv2
    frames = []
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            indices = [0, max(0, total - 1)] if num_frames == 2 else np.linspace(0, total - 1, num_frames).astype(int).tolist()
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if frame.shape[0] != image_size or frame.shape[1] != image_size:
                        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                    frames.append(frame)
        cap.release()

    while len(frames) < num_frames:
        frames.append(np.zeros((image_size, image_size, 3), dtype=np.uint8))

    return np.stack(frames[:num_frames])


def _decode_audio_waveform_raw(audio_path: str, sample_rate: int = 128000) -> np.ndarray:
    """Load raw audio waveform into float32 NumPy array [sample_rate * 2]."""
    target_len = sample_rate * 2
    if audio_path and os.path.exists(audio_path):
        try:
            import torchaudio
            waveform, sr = torchaudio.load(audio_path)
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                waveform = resampler(waveform)
            if waveform.ndim == 2 and waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            y = waveform.squeeze(0).to(torch.float32)
            if y.numel() > target_len:
                y = y[:target_len]
            elif y.numel() < target_len:
                y = torch.nn.functional.pad(y, (0, target_len - y.numel()))
            return y.numpy()
        except Exception:
            pass
    return np.zeros(target_len, dtype=np.float32)


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
        sample_rate: int = 128000,
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
            self.ram_cache = None

            if self.parent.cache_mode == "ram" and len(self.data_dict) > 0:
                self._preload_to_ram()

        def _preload_to_ram(self) -> None:
            import concurrent.futures
            try:
                from tqdm import tqdm
                has_tqdm = True
            except ImportError:
                has_tqdm = False

            preload_threads = max(1, self.parent.dataloader_workers * 2 if self.parent.dataloader_workers > 0 else 4)
            logger.info("==================================================")
            logger.info(f"Starting Multimodal -> RAM preload for split '{self.split}' ({len(self.data_dict)} samples)...")
            logger.info(f"Using ThreadPoolExecutor with {preload_threads} workers for RAM preload.")

            def load_single_entry(idx_and_item):
                idx, item = idx_and_item
                if isinstance(item, (list, tuple)):
                    if len(item) == 3:
                        audio_p, video_p, tgt = str(item[0]), str(item[1]), item[2]
                    elif len(item) == 2:
                        p0, tgt = str(item[0]), item[1]
                        if "_audio_" in p0 or p0.endswith(".wav"):
                            audio_p, video_p = p0, ""
                        else:
                            audio_p, video_p = "", p0
                    else:
                        audio_p, video_p, tgt = "", "", 0
                elif isinstance(item, dict):
                    audio_p = str(item.get('audio_path', ''))
                    video_p = str(item.get('video_path', ''))
                    tgt = item.get('label', item.get('target', 0))
                else:
                    audio_p, video_p, tgt = "", "", 0

                if not audio_p and video_p:
                    cand = video_p.replace("/video/", "/audio/").replace("\\video\\", "\\audio\\").replace("_video_", "_audio_")
                    if cand.endswith(".mp4"):
                        cand = cand[:-4] + ".wav"
                    if os.path.exists(cand):
                        audio_p = cand
                if not video_p and audio_p:
                    cand = audio_p.replace("/audio/", "/video/").replace("\\audio\\", "\\video\\").replace("_audio_", "_video_")
                    if cand.endswith(".wav"):
                        cand = cand[:-4] + ".mp4"
                    if os.path.exists(cand):
                        video_p = cand

                if isinstance(tgt, (int, np.integer)):
                    tgt_onehot = np.zeros(4, dtype=np.float32)
                    if 0 <= int(tgt) < 4:
                        tgt_onehot[int(tgt)] = 1.0
                elif isinstance(tgt, str):
                    tgt_str = tgt.strip().lower()
                    class_map = {"none": 0, "strong": 1, "medium": 2, "weak": 3}
                    tgt_onehot = np.zeros(4, dtype=np.float32)
                    if tgt_str in class_map:
                        tgt_onehot[class_map[tgt_str]] = 1.0
                else:
                    try:
                        arr = np.array(tgt, dtype=np.float32)
                        tgt_onehot = arr.reshape(4) if arr.size == 4 else np.zeros(4, dtype=np.float32)
                    except Exception:
                        tgt_onehot = np.zeros(4, dtype=np.float32)

                video_raw = _decode_video_frames_raw(video_p, self.parent.image_size, self.parent.num_frames)
                audio_raw = _decode_audio_waveform_raw(audio_p, self.parent.sample_rate)
                clip_name = os.path.basename(video_p or audio_p or f"sample_{idx}")

                return idx, (video_raw, audio_raw, tgt_onehot, clip_name)

            cache = [None] * len(self.data_dict)
            total_bytes = 0
            indexed_items = list(enumerate(self.data_dict))

            with concurrent.futures.ThreadPoolExecutor(max_workers=preload_threads) as executor:
                futures = {
                    executor.submit(load_single_entry, item): item[0]
                    for item in indexed_items
                }
                iterator = concurrent.futures.as_completed(futures)
                if has_tqdm:
                    pbar = tqdm(total=len(futures), desc=f"Preloading {self.split} to RAM")
                    for future in iterator:
                        idx = futures[future]
                        try:
                            result_idx, sample = future.result()
                            cache[result_idx] = sample
                            total_bytes += sample[0].nbytes + sample[1].nbytes
                        except Exception as exc:
                            logger.error(f"Error preloading sample {idx}: {exc}")
                        pbar.update(1)
                    pbar.close()
                else:
                    for future in iterator:
                        idx = futures[future]
                        try:
                            result_idx, sample = future.result()
                            cache[result_idx] = sample
                            total_bytes += sample[0].nbytes + sample[1].nbytes
                        except Exception as exc:
                            logger.error(f"Error preloading sample {idx}: {exc}")

            self.ram_cache = cache
            cache_mb = total_bytes / (1024 ** 2)
            logger.info(f"Successfully cached '{self.split}' split to RAM: {len(self.ram_cache)} samples ({cache_mb:.1f} MB)")
            logger.info("==================================================")

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
            if self.ram_cache is not None and self.ram_cache[idx] is not None:
                video_raw, audio_raw, target_onehot, clip_name = self.ram_cache[idx]
                frames = [self.transform(frame) for frame in video_raw]
                video_tensor = torch.stack(frames[:self.parent.num_frames])
                audio_tensor = torch.from_numpy(audio_raw).to(torch.float32)
                return {
                    'clip_name': clip_name,
                    'video_form': video_tensor,
                    'audio_form': audio_tensor,
                    'target': target_onehot
                }

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

    def get_dataloader(self, split: str, shuffle: bool = False) -> DataLoader:
        return self.get_data_loader(split, shuffle=shuffle)

