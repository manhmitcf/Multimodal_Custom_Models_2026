import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from dataset import SplitterConfig, FishDataSplitter

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FishVoiceDataLoader:
    """
    Unified manager class for the fish voice raw audio dataset (FishVoiceDataLoader).
    Provides raw waveforms to the model for Mel-spectrogram feature extraction on GPU.
    Supports flexible configuration paths.
    """
    _fallback_warning_shown = False

    def __init__(
        self,
        sample_rate: int = 64000,
        batch_size: int = 40,
        num_workers: int = -1,
        cache_audio: bool = True,
        splitter_config: Optional[SplitterConfig] = None
    ) -> None:
        """
        Initialize FishVoiceDataLoader.

        Args:
            sample_rate (int): Audio target sample rate. Default: 64000.
            batch_size (int): Data loader batch size. Default: 40.
            num_workers (int): Number of parallel CPU loader workers (-1 for auto-detect). Default: -1.
            cache_audio (bool): Enable preloading the entire dataset into RAM cache. Default: True.
            splitter_config (Optional[SplitterConfig]): Pre-built SplitterConfig object. Default: None (uses SplitterConfig defaults).
        """
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_audio = cache_audio

        # Auto-detect optimal CPU worker threads if set to -1
        if self.num_workers == -1:
            max_cpu = os.cpu_count()
            if max_cpu is None or max_cpu <= 0:
                self.num_workers = 0
            elif max_cpu == 2:
                self.num_workers = max_cpu // 2
            else:
                self.num_workers = (max_cpu // 2) + 1

        logger.info("==================================================")
        logger.info("Initializing FishVoiceDataLoader settings:")
        logger.info(f"  - Sample Rate:              {self.sample_rate} Hz")
        logger.info(f"  - Batch Size:               {self.batch_size}")
        logger.info(f"  - Num Workers:              {self.num_workers} (Auto-calculated from {os.cpu_count()} CPU cores)")
        logger.info(f"  - Cache audio in RAM:       {self.cache_audio}")
        logger.info("==================================================")

        # 1. Load splitter configurations and initialize the data splitter
        if splitter_config is None:
            self.splitter_config = SplitterConfig()
        else:
            self.splitter_config = splitter_config
        self.splitter = FishDataSplitter(config=self.splitter_config)

        # 2. Split dataset into train, val, and test partitions
        self.train_dict, self.test_dict, self.val_dict = self.splitter.split_data()

        logger.info("Successfully initialized FishVoiceDataLoader.")

    @staticmethod
    def load_audio(path: str, sr: int = 64000) -> torch.Tensor:
        """
        Load audio file using torchaudio, resample to the target sample rate,
        and force a fixed length of 2 seconds (clipping/padding).
        """
        import torchaudio

        waveform, original_sr = torchaudio.load(path)

        # Convert multi-channel audio to mono while keeping shape [1, num_samples].
        if waveform.ndim == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if original_sr != sr:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=sr)
            waveform = resampler(waveform)

        y = waveform.squeeze(0).to(torch.float32)
        target_length = sr * 2  # 2 seconds
        
        # Clip if longer, zero-pad if shorter
        if y.numel() > target_length:
            y = y[:target_length]
        elif y.numel() < target_length:
            y = torch.nn.functional.pad(y, (0, target_length - y.numel()))
            
        return y

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate single raw waveform samples into unified mini-batch tensors.
        """
        wav_names = [data['audio_name'] for data in batch]
        wav = [data['waveform'] for data in batch]
        target = [data['target'] for data in batch]

        waveforms = torch.FloatTensor(np.array(wav))
        targets_tensor = torch.FloatTensor(np.array(target))

        return {
            'audio_name': wav_names,
            'waveform': waveforms,
            'target': targets_tensor
        }

    class _InnerDataset(Dataset):
        """
        Internal PyTorch Dataset wrapper matching standard API.
        """
        def __init__(self, parent: 'FishVoiceDataLoader', split: str) -> None:
            self.parent = parent
            self.split = split
            self.cache_audio = parent.cache_audio
            self.waveform_cache = None
            self.cache_size_mb = 0.0

            if self.split == 'train':
                self.data_dict = parent.train_dict
            elif self.split == 'test':
                self.data_dict = parent.test_dict
            elif self.split == 'val':
                self.data_dict = parent.val_dict
            else:
                raise ValueError(f"Invalid split value '{self.split}'. Must be one of ['train', 'test', 'val'].")

            # Preload to RAM cache if enabled
            if self.cache_audio:
                self._preload_audio()

        def _preload_audio(self) -> None:
            """
            Preload raw audio waveforms into RAM cache using multi-threading to speed up training.
            """
            import concurrent.futures

            logger.info(f"Starting dataset preload for '{self.split}' split to RAM...")

            # Helper function to load a single raw waveform
            def load_single_audio(item: List[Any]) -> np.ndarray:
                wav_name = item[0]
                wav = FishVoiceDataLoader.load_audio(wav_name, sr=self.parent.sample_rate)
                wav_1d = wav.squeeze(0) if wav.ndim > 1 else wav
                return wav_1d.numpy()

            num_threads = self.parent.num_workers
            if num_threads <= 0:
                num_threads = 1

            logger.info(f"Preloading audio files parallel using {num_threads} threads...")

            cache = [None] * len(self.data_dict)
            total_bytes = 0

            # Safe check for tqdm library import
            try:
                from tqdm import tqdm
                has_tqdm = True
            except ImportError:
                has_tqdm = False

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {
                    executor.submit(load_single_audio, item): idx 
                    for idx, item in enumerate(self.data_dict)
                }

                if has_tqdm:
                    pbar = tqdm(total=len(self.data_dict), desc=f"Preloading {self.split} to RAM")
                    for future in concurrent.futures.as_completed(futures):
                        idx = futures[future]
                        wav_numpy = future.result()
                        cache[idx] = wav_numpy
                        total_bytes += wav_numpy.nbytes
                        pbar.update(1)
                    pbar.close()
                else:
                    for future in concurrent.futures.as_completed(futures):
                        idx = futures[future]
                        wav_numpy = future.result()
                        cache[idx] = wav_numpy
                        total_bytes += wav_numpy.nbytes

            self.waveform_cache = cache
            self.cache_size_mb = total_bytes / (1024 ** 2)
            logger.info(
                f"Cached {self.split} split to RAM: {len(cache)} files, {self.cache_size_mb:.1f} MB"
            )

        def __len__(self) -> int:
            return len(self.data_dict)

        def __getitem__(self, index: int) -> Dict[str, Any]:
            item = self.data_dict[index]
            wav_name = item[0]
            target = item[-1]

            # 1. Fetch waveform from cache, fallback to disk load
            if self.waveform_cache is not None:
                wav_numpy = self.waveform_cache[index]
            else:
                wav = FishVoiceDataLoader.load_audio(wav_name, sr=self.parent.sample_rate)
                wav_1d = wav.squeeze(0) if wav.ndim > 1 else wav
                wav_numpy = wav_1d.numpy()

            # 2. Transform target labels into one-hot vectors
            target_onehot = np.eye(4)[target]

            return {
                'audio_name': wav_name,
                'waveform': wav_numpy,
                'target': target_onehot
            }

    def get_dataloader(
        self,
        split: str,
        shuffle: bool = False,
        drop_last: bool = False
    ) -> DataLoader:
        """
        Instantiate and return standard PyTorch DataLoader for the requested dataset split.
        """
        dataset = self._InnerDataset(parent=self, split=split)
        
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )
