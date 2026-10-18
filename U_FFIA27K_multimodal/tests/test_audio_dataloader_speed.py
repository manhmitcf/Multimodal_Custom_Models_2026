import sys
from pathlib import Path
import unittest
import numpy as np
import torch

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import SplitterConfig
from dataset.dataloader_multimodal import FishMultimodalDataLoader
from models.audio_sota_net import AudioSOTANet


class TestAudioDataLoaderSpeed(unittest.TestCase):
    def test_include_video_config_flag(self):
        """Verify that setting include_video=False properly sets the flag on the config."""
        splitter_cfg = SplitterConfig(
            dataset_path="/marimo/Fish_Feeding_Intensity_Dataset",
            include_video=False,
            save_results=False
        )
        self.assertTrue(hasattr(splitter_cfg, "include_video"))
        self.assertFalse(splitter_cfg.include_video)

    def test_inner_dataset_lightweight_dummy_video(self):
        """Verify that _InnerDataset with include_video=False returns 1D dummy tensor (4 bytes) instead of full 5D video."""
        class MockParent:
            include_video = False
            num_frames = 2
            image_size = 224
            sample_rate = 256000

        # Construct _InnerDataset with mock parent
        inner_ds = FishMultimodalDataLoader._InnerDataset.__new__(FishMultimodalDataLoader._InnerDataset)
        inner_ds.parent = MockParent()
        inner_ds.split = 'train'
        inner_ds.data_dict = [{'audio_path': 'fake.wav', 'video_path': 'fake.mp4', 'target': 1}]

        # Mock ram_cache with preloaded raw data
        fake_audio_raw = np.zeros(512000, dtype=np.float32)
        fake_video_raw = np.zeros((2, 1, 1, 3), dtype=np.uint8)
        fake_target = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        inner_ds.ram_cache = [(fake_video_raw, fake_audio_raw, fake_target, "test_clip")]

        sample = inner_ds[0]
        self.assertEqual(sample['video_form'].shape, torch.Size([1]))
        self.assertEqual(sample['audio_form'].shape, torch.Size([512000]))
        self.assertEqual(sample['clip_name'], "test_clip")

        # Test batch collation
        batch = FishMultimodalDataLoader.collate_fn([sample, sample])
        self.assertEqual(batch['video_form'].shape, torch.Size([2, 1]))
        self.assertEqual(batch['audio_form'].shape, torch.Size([2, 512000]))
        self.assertEqual(batch['target'].shape, torch.Size([2, 4]))

        # Test forward pass with AudioSOTANet
        model = AudioSOTANet(hidden_dims=[512], embed_dim=384, classes_num=4)
        out = model(batch['video_form'], batch['audio_form'])
        self.assertEqual(out['clipwise_output'].shape, torch.Size([2, 4]))


if __name__ == "__main__":
    unittest.main()
