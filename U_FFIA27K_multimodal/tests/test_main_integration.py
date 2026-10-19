import unittest
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.train_config import TrainConfig
from main import build_model, build_artifact_filename, MODEL_REGISTRY
from models.video_sota_net import VideoSOTANet


class TestMainIntegration(unittest.TestCase):
    def test_model_registry_contains_video_sota_net(self):
        self.assertIn("VideoSOTANet", MODEL_REGISTRY)

    def test_build_model_returns_video_sota_net(self):
        cfg = TrainConfig.from_json()
        cfg.model.backbone = "VideoSOTANet"
        cfg.video_ablation.active_config = "C5_rgb_scalars"
        cfg.video_ablation.temporal_mode = "abs_diff"

        model = build_model(cfg)
        self.assertIsInstance(model, VideoSOTANet)
        self.assertEqual(model.channel_indices, [0, 1, 2, 5, 6])
        self.assertEqual(model.temporal_mode, "abs_diff")
        self.assertEqual(model.embed_dim, 384)

    def test_build_artifact_filename(self):
        cfg = TrainConfig.from_json()
        cfg.model.backbone = "VideoSOTANet"
        cfg.video_ablation.active_config = "C4_rgb_mag"
        cfg.video_ablation.temporal_mode = "composite_ln"

        filename = build_artifact_filename(cfg, timestamp="20261019_120000")
        self.assertTrue(filename.startswith("Video_C4_rgb_mag_composite_ln_"))

    def test_run_training_session_cli_overrides_dry_run(self):
        import shutil
        import tempfile
        from main import run_training_session

        orig_json = Path(project_root) / "config" / "train_config.json"
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            shutil.copyfile(orig_json, tmp_path)
            run_training_session(
                train_config_path=tmp_path,
                preset="C4_rgb_vorticity",
                temporal_mode="abs_diff",
                dry_run=True,
                device_str="cpu"
            )
            reloaded = TrainConfig.from_json(tmp_path)
            self.assertEqual(reloaded.video_ablation.active_config, "C4_rgb_vorticity")
            self.assertEqual(reloaded.video_ablation.temporal_mode, "abs_diff")
        except Exception as e:
            self.fail(f"run_training_session with CLI overrides failed with: {e}")
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()


if __name__ == "__main__":
    unittest.main()
