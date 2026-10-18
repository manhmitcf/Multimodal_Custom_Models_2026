import os
import sys
from pathlib import Path
import unittest
import torch

project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.audio_sota_net import AudioSOTANet
from features.audio_frontend import AudioFrontend
from utils.profile_model import count_parameters, measure_flops
from utils.losses import ClipCELoss
from config.train_config import TrainConfig, AudioMLPAblationConfig


class TestAudioMLPAblations(unittest.TestCase):
    """
    Comprehensive test suite verifying all 10 Audio MLP ablation configurations:
    Checking parameter counts, forward pass, 100% gradient propagation, and FLOPs profiling.
    """
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[TestAudioMLPAblations] Testing on compute device: {cls.device}")

        # The 10 ablation presets from Table 3.2
        cls.presets = {
            "L0_linear": [],
            "W_256_bottleneck": [256],
            "W_512_moderate": [512],
            "W_768_balanced": [768],
            "W_1024_wide": [1024],
            "W_2048_expansion": [2048],
            "L2_pyramidal_1024_512": [1024, 512],
            "L2_compact_768_512": [768, 512],
            "L2_wide_1024_768": [1024, 768],
            "L3_deep_1024_768_512": [1024, 768, 512],
        }

    def test_all_10_presets_forward_backward(self):
        batch_size = 2
        raw_audio_len = 512000  # 2.0s @ 256 kHz
        loss_fn = ClipCELoss()

        print(f"\n{'Preset Name':<26} | {'Hidden Dims':<18} | {'Total Params':<14} | {'GFLOPs':<8} | {'Grad Flow'}")
        print("-" * 80)

        for preset_name, hidden_dims in self.presets.items():
            model = AudioSOTANet(
                hidden_dims=hidden_dims,
                embed_dim=384,
                classes_num=4,
                dropout=0.1
            ).to(self.device)

            stats = count_parameters(model)
            total_params = stats["total"]
            self.assertLess(total_params, 5_000_000, f"Preset {preset_name} exceeds 5.0M budget!")

            # 1. Forward Pass
            model.train()
            dummy_audio = torch.randn(batch_size, raw_audio_len, device=self.device)
            dummy_target = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], device=self.device)

            out = model(dummy_audio)
            self.assertIn("clipwise_output", out)
            self.assertEqual(out["clipwise_output"].shape, (batch_size, 4))
            self.assertEqual(out["f_audio"].shape, (batch_size, 384))

            # Also check dual-argument signature compatibility
            dummy_video = torch.randn(batch_size, 2, 3, 224, 224, device=self.device)
            out_dual = model(dummy_video, dummy_audio)
            self.assertEqual(out_dual["clipwise_output"].shape, (batch_size, 4))

            # 2. Loss & Backward Pass
            loss = loss_fn(out, {"target": dummy_target})
            loss.backward()

            trainable_with_grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
            total_trainable = sum(1 for p in model.parameters() if p.requires_grad)

            self.assertEqual(
                trainable_with_grads,
                total_trainable,
                f"Gradient flow broken in preset {preset_name}: {trainable_with_grads}/{total_trainable}"
            )

            # 3. Eval forward
            model.eval()
            with torch.no_grad():
                out_eval = model(dummy_audio)
                self.assertEqual(out_eval["clipwise_output"].shape, (batch_size, 4))

            # 4. Measure FLOPs
            gflops = measure_flops(model, device=self.device.type)

            print(f"{preset_name:<26} | {str(hidden_dims):<18} | {total_params:<14,} | {gflops:.4f}  | 100% ({total_trainable}/{total_trainable})")

    def test_ablation_config_presets_sync(self):
        """Verify that AudioMLPAblationConfig contains all expected presets."""
        cfg = AudioMLPAblationConfig()
        for preset_name, expected_dims in self.presets.items():
            self.assertIn(preset_name, cfg.presets)
            cfg.active_config = preset_name
            self.assertEqual(cfg.get_active_hidden_dims(), expected_dims)


if __name__ == "__main__":
    unittest.main()
