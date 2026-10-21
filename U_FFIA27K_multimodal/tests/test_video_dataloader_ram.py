import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
import numpy as np
import torch

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.train_config import SplitterConfig
from dataset.dataloader_multimodal import FishMultimodalDataLoader, _decode_video_frames_raw


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

    def test_frame_index_starts_at_1_decord(self):
        mock_vr = MagicMock()
        mock_vr.__len__.return_value = 50
        mock_vr.get_batch.return_value.asnumpy.return_value = np.zeros((2, 224, 224, 3), dtype=np.uint8)

        with patch("decord.VideoReader", return_value=mock_vr):
            res = _decode_video_frames_raw("dummy.mp4", image_size=224, num_frames=2)
            indices = mock_vr.get_batch.call_args[0][0]
            self.assertEqual(indices, [1, 49])
            self.assertEqual(res.shape, (2, 224, 224, 3))

    def test_frame_index_starts_at_1_cv2_fallback(self):
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = mock_cap_cls.return_value
            mock_cap.get.return_value = 50.0
            mock_cap.read.return_value = (True, np.zeros((224, 224, 3), dtype=np.uint8))
            with patch("os.path.exists", return_value=True):
                with patch.dict("sys.modules", {"decord": None}):
                    res = _decode_video_frames_raw("dummy.mp4", image_size=224, num_frames=2)
                    pos_calls = [call.args[1] for call in mock_cap.set.call_args_list]
                    self.assertEqual(pos_calls, [1, 49])
                    self.assertEqual(res.shape, (2, 224, 224, 3))


if __name__ == "__main__":
    unittest.main()
