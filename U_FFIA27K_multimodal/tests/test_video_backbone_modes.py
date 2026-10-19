import unittest
import sys
from pathlib import Path
import torch

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.video_backbone import ConvNeXtNanoVideoBackbone


class TestVideoBackboneModes(unittest.TestCase):
    def test_dynamic_in_chans(self):
        B, T = 2, 2
        for in_chans in [3, 4, 5, 6, 7]:
            model = ConvNeXtNanoVideoBackbone(
                in_chans=in_chans,
                embed_dim=384,
                temporal_mode="sum"
            )
            x = torch.randn(B, T, in_chans, 224, 224)
            out = model(x)
            self.assertEqual(out.shape, (B, 384))

    def test_three_temporal_modes(self):
        B, T, C = 2, 2, 7
        x = torch.randn(B, T, C, 224, 224)

        for mode in ["sum", "abs_diff", "composite_ln"]:
            model = ConvNeXtNanoVideoBackbone(
                in_chans=C,
                embed_dim=384,
                temporal_mode=mode
            )
            self.assertEqual(getattr(model, "temporal_mode", None), mode)
            # Verify exact mathematical formulation in eval mode (DropPath = Identity)
            model.eval()
            with torch.no_grad():
                out_eval = model(x)
                flat_x = x.reshape(B * T, C, 224, 224)
                features = model.forward_features(flat_x).view(B, T, 384)
                v0, v1 = features[:, 0], features[:, 1]

                if mode == "sum":
                    expected = v0 + v1
                    self.assertTrue(torch.allclose(out_eval, expected, atol=1e-5))
                elif mode == "abs_diff":
                    expected = torch.abs(v1 - v0)
                    self.assertTrue(torch.allclose(out_eval, expected, atol=1e-5))
                elif mode == "composite_ln":
                    expected = model.norm_composite((v0 + v1) + torch.abs(v1 - v0))
                    self.assertTrue(torch.allclose(out_eval, expected, atol=1e-5))

            # Backward test in train mode
            model.train()
            out = model(x)
            loss = out.sum()
            loss.backward()
            trainable_grads = [p.grad for p in model.parameters() if p.requires_grad]
            self.assertTrue(all(g is not None for g in trainable_grads))

    def test_backward_compatibility_multimodal(self):
        B, T, C = 2, 2, 7
        x = torch.randn(B, T, C, 224, 224)
        model = ConvNeXtNanoVideoBackbone(
            in_chans=C,
            embed_dim=224,
            temporal_mode="multimodal"
        )
        out = model(x)
        self.assertEqual(out.shape, (B, 224))


if __name__ == "__main__":
    unittest.main()
