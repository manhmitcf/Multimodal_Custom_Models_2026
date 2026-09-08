"""
Unit test for VideoConvNeXtNanoNet with 4 uniform frames (T=4).
Verifies:
  1. Analytical kinematics extraction (7 channels across 4 frames).
  2. ConvNeXt-Nano Video Backbone spatiotemporal feature extraction (~2.70M params).
  3. Classification logits and probability outputs.
  4. Backward pass and end-to-end gradient propagation.
"""
import sys
from pathlib import Path

# Add project root to sys.path
pkg_dir = Path(__file__).resolve().parent
if str(pkg_dir) not in sys.path:
    sys.path.insert(0, str(pkg_dir))

import torch
import torch.nn.functional as F

from features.motion_kinematics import FishMotionKinematics7Ch
from models.video_convnext_nano_net import VideoConvNeXtNanoNet
from utils.losses import ClipCELoss
from utils.profile_model import count_parameters


def test_kinematics_4frames():
    print(">>> Testing FishMotionKinematics7Ch with T=4 frames...")
    kinematics = FishMotionKinematics7Ch(image_size=224)
    dummy_rgb = torch.rand(2, 4, 3, 224, 224, dtype=torch.float32)
    frames_7ch, summary = kinematics(dummy_rgb)

    assert frames_7ch.shape == (2, 4, 7, 224, 224), f"Unexpected frames_7ch shape: {frames_7ch.shape}"
    assert summary.shape == (2, 4), f"Unexpected summary shape: {summary.shape}"
    print(f"    [PASS] Kinematics 7-channel shape: {frames_7ch.shape}")


def test_video_convnext_model():
    print(">>> Testing VideoConvNeXtNanoNet Architecture...")
    model = VideoConvNeXtNanoNet(
        classes_num=4,
        embed_dim=224,
        image_size=224,
        num_frames=4,
        in_chans=7
    )

    # 1. Parameter Audit
    stats = count_parameters(model)
    total_params = stats["total"]
    total_m = stats["total_million"]
    print(f"    Model Trainable Parameters: {total_params:,} ({total_m:.3f} M)")
    assert 2_600_000 <= total_params <= 2_800_000, f"Expected ~2.70M params, got {total_params}"

    # 2. Forward pass with raw RGB video [B, 4, 3, 224, 224]
    model.train()
    dummy_video = torch.randn(2, 4, 3, 224, 224, dtype=torch.float32)
    dummy_targets = torch.tensor([1, 2], dtype=torch.long)

    outputs = model(dummy_video)
    logits = outputs["clipwise_output"]
    probs = outputs["probabilities"]

    assert logits.shape == (2, 4), f"Unexpected logits shape: {logits.shape}"
    assert probs.shape == (2, 4), f"Unexpected probs shape: {probs.shape}"
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5), "Softmax probabilities do not sum to 1.0"
    print(f"    [PASS] Logits shape: {logits.shape}, Probs shape: {probs.shape}")

    # 3. Backward pass & gradient propagation
    loss_fn = ClipCELoss()
    loss = loss_fn(outputs, {"target": dummy_targets})
    loss.backward()

    trainable_with_grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"    Gradient Flow: {trainable_with_grads}/{total_trainable} parameters with gradients.")
    assert trainable_with_grads == total_trainable, "Some parameters did not receive gradients!"
    print(f"    [PASS] Full gradient flow confirmed. Loss = {loss.item():.4f}")


def test_eval_mode():
    print(">>> Testing Evaluation Mode (torch.no_grad)...")
    model = VideoConvNeXtNanoNet(
        classes_num=4,
        embed_dim=224,
        image_size=224,
        num_frames=4,
        in_chans=7
    )
    model.eval()
    with torch.no_grad():
        dummy_video = torch.randn(2, 4, 3, 224, 224, dtype=torch.float32)
        outputs = model(dummy_video)
        assert outputs["logits"].shape == (2, 4)
    print("    [PASS] Eval mode forward completed cleanly.")


if __name__ == "__main__":
    test_kinematics_4frames()
    test_video_convnext_model()
    test_eval_mode()
    print("==================================================")
    print("ALL TESTS PASSED SUCCESSFULLY FOR VideoConvNeXtNanoNet (4 FRAMES)!")
    print("==================================================")
