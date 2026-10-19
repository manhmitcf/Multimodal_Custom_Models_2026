import unittest
import sys
from pathlib import Path
import torch

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.video_sota_net import VideoSOTANet
from utils.profile_model import count_parameters


class TestVideoSOTANet(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

    def test_c3_rgb_forward_backward(self):
        B, T = 2, 2
        model = VideoSOTANet(
            channel_indices=[0, 1, 2],
            temporal_mode="sum",
            embed_dim=384,
            classes_num=4
        ).to(self.device)

        dummy_video = torch.randn(B, T, 3, 224, 224, device=self.device)
        dummy_audio = torch.randn(B, 512000, device=self.device)

        # Single argument call
        out = model(dummy_video)
        self.assertIn("clipwise_output", out)
        self.assertEqual(out["clipwise_output"].shape, (B, 4))
        self.assertEqual(out["f_video"].shape, (B, 384))

        # Dual argument call (for trainer and profiler compatibility)
        out_dual = model(dummy_video, dummy_audio)
        self.assertEqual(out_dual["clipwise_output"].shape, (B, 4))

        # Backward & 100% gradient flow
        loss = out["clipwise_output"].sum()
        loss.backward()

        trainable_with_grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        self.assertEqual(trainable_with_grads, total_trainable)

        # Parameter count
        stats = count_parameters(model)
        self.assertLess(stats["total"], 5_000_000)

    def test_c7_full_forward_backward(self):
        B, T = 2, 2
        model = VideoSOTANet(
            channel_indices=[0, 1, 2, 3, 4, 5, 6],
            temporal_mode="composite_ln",
            embed_dim=384,
            classes_num=4
        ).to(self.device)

        dummy_video = torch.randn(B, T, 3, 224, 224, device=self.device)
        out = model(dummy_video)
        self.assertEqual(out["clipwise_output"].shape, (B, 4))
        self.assertEqual(out["f_video"].shape, (B, 384))

        loss = out["clipwise_output"].sum()
        loss.backward()

        trainable_with_grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        self.assertEqual(trainable_with_grads, total_trainable)


if __name__ == "__main__":
    unittest.main()
