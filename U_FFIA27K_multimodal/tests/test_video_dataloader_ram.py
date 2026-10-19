import unittest
from unittest.mock import patch
import sys
from pathlib import Path
import numpy as np
import torch

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.train_config import SplitterConfig
from dataset.dataloader_multimodal import FishMultimodalDataLoader


class TestVideoDataLoaderRAM(unittest.TestCase):
    def test_include_audio_false_bypasses_decode(self):
        splitter_cfg = SplitterConfig(
            dataset_path="dummy_path",
            include_video=True,
            include_audio=False
        )

        with patch("dataset.dataloader_multimodal.FishDataSplitter") as mock_splitter_cls:
            mock_splitter = mock_splitter_cls.return_value
            # Mock 2 dummy samples: (video_p, audio_p, target)
            mock_splitter.split_data.return_value = (
                [("dummy_audio.wav", "dummy_video.mp4", 1)],
                [("dummy_audio.wav", "dummy_video.mp4", 1)],
                [("dummy_audio.wav", "dummy_video.mp4", 1)],
            )

            with patch("dataset.dataloader_multimodal._decode_audio_waveform_raw") as mock_decode_audio:
                loader = FishMultimodalDataLoader(
                    batch_size=1,
                    cache_mode="none",
                    image_size=224,
                    num_frames=2,
                    splitter_config=splitter_cfg,
                    dataloader_workers=0
                )
                dataset = loader._InnerDataset(loader, split="train")
                sample = dataset[0]

                # Verify audio decode was never called
                mock_decode_audio.assert_not_called()

                # Verify tensor shapes
                self.assertEqual(sample["video_form"].shape, (2, 3, 224, 224))
                self.assertEqual(sample["audio_form"].shape, (1,))

                # Test collation
                batch = loader.collate_fn([sample])
                self.assertEqual(batch["video_form"].shape, (1, 2, 3, 224, 224))
                self.assertEqual(batch["audio_form"].shape, (1, 1))
                self.assertEqual(batch["target"].shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
