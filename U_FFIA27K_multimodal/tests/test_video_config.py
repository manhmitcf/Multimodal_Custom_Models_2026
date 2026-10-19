import unittest
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.train_config import VideoChannelAblationConfig, TrainConfig


class TestVideoConfig(unittest.TestCase):
    def setUp(self):
        self.expected_presets = {
            "C3_rgb": [0, 1, 2],
            "C4_rgb_mag": [0, 1, 2, 5],
            "C4_rgb_vorticity": [0, 1, 2, 6],
            "C5_rgb_flow": [0, 1, 2, 3, 4],
            "C5_rgb_scalars": [0, 1, 2, 5, 6],
            "C6_rgb_flow_mag": [0, 1, 2, 3, 4, 5],
            "C6_rgb_flow_vorticity": [0, 1, 2, 3, 4, 6],
            "C7_full": [0, 1, 2, 3, 4, 5, 6],
        }

    def test_default_presets_presence(self):
        cfg = VideoChannelAblationConfig()
        self.assertEqual(len(cfg.presets), 8)
        for preset_name, indices in self.expected_presets.items():
            self.assertIn(preset_name, cfg.presets)
            self.assertEqual(cfg.presets[preset_name], indices)

    def test_get_active_channel_indices(self):
        cfg = VideoChannelAblationConfig()
        for preset_name, indices in self.expected_presets.items():
            cfg.active_config = preset_name
            self.assertEqual(cfg.get_active_channel_indices(), indices)

    def test_temporal_mode_validation(self):
        for mode in ["sum", "abs_diff", "composite_ln"]:
            cfg = VideoChannelAblationConfig(temporal_mode=mode)
            self.assertEqual(cfg.temporal_mode, mode)

        with self.assertRaises(ValueError):
            VideoChannelAblationConfig(temporal_mode="invalid_mode")

    def test_train_config_parsing(self):
        cfg = TrainConfig.from_json()
        self.assertIsNotNone(getattr(cfg, "video_ablation", None))
        self.assertEqual(cfg.model.embed_dim, 384)
        self.assertFalse(cfg.dataset_splitter.include_audio)


if __name__ == "__main__":
    unittest.main()
