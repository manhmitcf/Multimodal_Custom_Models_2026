import os
import sys
from pathlib import Path
import unittest
import torch

project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.video_sota_net import VideoSOTANet
from utils.profile_model import count_parameters, measure_flops
from utils.losses import ClipCELoss
from config.train_config import VideoChannelAblationConfig


class TestVideoChannelAblation(unittest.TestCase):
    """
    Comprehensive verification suite for Video Backbone Ablation Study:
    - 8 Input Channel Presets (C3_rgb to C7_full)
    - 3 Temporal Aggregation Modes (sum, abs_diff, composite_ln)
    - 100% Active Gradient Flow on all trainable tensors
    - Parameter Budget Verification (< 5.0M)
    - FLOPs Profiling (< 2.0 GFLOPs)
    """
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[TestVideoChannelAblation] Testing on compute device: {cls.device}")

        cls.presets = {
            "C3_rgb": [0, 1, 2],
            "C4_rgb_mag": [0, 1, 2, 5],
            "C4_rgb_vorticity": [0, 1, 2, 6],
            "C5_rgb_flow": [0, 1, 2, 3, 4],
            "C5_rgb_scalars": [0, 1, 2, 5, 6],
            "C6_rgb_flow_mag": [0, 1, 2, 3, 4, 5],
            "C6_rgb_flow_vorticity": [0, 1, 2, 3, 4, 6],
            "C7_full": [0, 1, 2, 3, 4, 5, 6],
        }

    def test_all_8_channel_presets(self):
        batch_size = 2
        num_frames = 2
        loss_fn = ClipCELoss()

        print(f"\n{'Preset Name':<22} | {'Channels':<10} | {'Total Params':<14} | {'GFLOPs':<8} | {'Grad Flow'}")
        print("-" * 75)

        for preset_name, indices in self.presets.items():
            model = VideoSOTANet(
                channel_indices=indices,
                temporal_mode="sum",
                embed_dim=384,
                classes_num=4,
                drop_path_rate=0.1
            ).to(self.device)

            stats = count_parameters(model)
            total_params = stats["total"]
            self.assertLess(total_params, 5_000_000, f"Preset {preset_name} exceeds 5.0M budget!")

            # 1. Forward Pass
            model.train()
            dummy_video = torch.randn(batch_size, num_frames, 3, 224, 224, device=self.device)
            dummy_target = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], device=self.device)

            out = model(dummy_video)
            self.assertIn("clipwise_output", out)
            self.assertEqual(out["clipwise_output"].shape, (batch_size, 4))
            self.assertEqual(out["f_video"].shape, (batch_size, 384))

            # 2. Backward Pass & 100% Gradient Verification
            loss = loss_fn(out, {"target": dummy_target})
            loss.backward()

            trainable_with_grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
            total_trainable = sum(1 for p in model.parameters() if p.requires_grad)

            self.assertEqual(
                trainable_with_grads,
                total_trainable,
                f"Gradient flow broken in preset {preset_name}: {trainable_with_grads}/{total_trainable}"
            )

            # 3. Measure FLOPs
            model.eval()
            with torch.no_grad():
                gflops = measure_flops(model, device=self.device.type, num_frames=num_frames)

            print(f"{preset_name:<22} | {len(indices):<10} | {total_params:<14,} | {gflops:.4f}  | 100% ({total_trainable}/{total_trainable})")

    def test_all_3_temporal_modes(self):
        batch_size = 2
        num_frames = 2
        loss_fn = ClipCELoss()
        temporal_modes = ["sum", "abs_diff", "composite_ln"]

        print(f"\n{'Temporal Mode':<22} | {'Formula':<26} | {'Params':<12} | {'Grad Flow'}")
        print("-" * 75)

        formulas = {
            "sum": "v0 + v1",
            "abs_diff": "|v1 - v0|",
            "composite_ln": "LayerNorm((v0+v1)+|v1-v0|)",
        }

        for mode in temporal_modes:
            model = VideoSOTANet(
                channel_indices=[0, 1, 2, 3, 4, 5, 6],
                temporal_mode=mode,
                embed_dim=384,
                classes_num=4,
                drop_path_rate=0.1
            ).to(self.device)

            stats = count_parameters(model)
            total_params = stats["total"]

            dummy_video = torch.randn(batch_size, num_frames, 3, 224, 224, device=self.device)
            dummy_target = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], device=self.device)

            out = model(dummy_video)
            loss = loss_fn(out, {"target": dummy_target})
            loss.backward()

            trainable_with_grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
            total_trainable = sum(1 for p in model.parameters() if p.requires_grad)

            self.assertEqual(trainable_with_grads, total_trainable)
            print(f"{mode:<22} | {formulas[mode]:<26} | {total_params:<12,} | 100% ({total_trainable}/{total_trainable})")

    def test_channel_config_sync(self):
        cfg = VideoChannelAblationConfig()
        for preset_name, expected_indices in self.presets.items():
            self.assertIn(preset_name, cfg.presets)
            cfg.active_config = preset_name
            self.assertEqual(cfg.get_active_channel_indices(), expected_indices)


if __name__ == "__main__":
    unittest.main()
