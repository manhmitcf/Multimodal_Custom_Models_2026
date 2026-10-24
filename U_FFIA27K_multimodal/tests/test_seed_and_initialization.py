import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import torch
import torch.nn as nn
from models.multimodal_flat_net import MultimodalFlatNet
from models.multimodal_sota_net import MultimodalSOTANet
from utils.seed import seed_everything
from config.train_config import TrainConfig


def test_seed_deterministic_model_weights():
    """Verify that seed_everything(42) creates bit-for-bit identical weights for MultimodalFlatNet and MultimodalSOTANet."""
    for mode in ["sum", "concat", "gated"]:
        # Run 1: seed 42
        seed_everything(42)
        model_1 = MultimodalFlatNet(num_frames=2, fusion_mode=mode)

        # Run 2: seed 42
        seed_everything(42)
        model_2 = MultimodalFlatNet(num_frames=2, fusion_mode=mode)

        # Verify all parameter tensors are strictly identical bit-for-bit
        for (name1, p1), (name2, p2) in zip(model_1.named_parameters(), model_2.named_parameters()):
            assert name1 == name2
            assert torch.equal(p1, p2), f"Parameter {name1} mismatch under identical seed 42 for mode '{mode}'"

        # Run 3: different seed 43 -> parameters must diverge
        seed_everything(43)
        model_3 = MultimodalFlatNet(num_frames=2, fusion_mode=mode)
        has_divergence = any(
            not torch.equal(p1, p3)
            for (name1, p1), (name3, p3) in zip(model_1.named_parameters(), model_3.named_parameters())
        )
        assert has_divergence, f"Parameters did not diverge with different seed for mode '{mode}'"


def test_seed_deterministic_forward_pass():
    """Verify identical seed produces identical forward pass outputs and probability scores."""
    for mode in ["sum", "concat", "gated"]:
        # Run 1
        seed_everything(42)
        model_1 = MultimodalFlatNet(num_frames=2, fusion_mode=mode)
        model_1.eval()

        # Fixed deterministic input
        torch.manual_seed(100)
        video = torch.randn(2, 2, 3, 224, 224)
        audio = torch.randn(2, 512000)

        with torch.no_grad():
            out_1 = model_1(video, audio)

        # Run 2
        seed_everything(42)
        model_2 = MultimodalFlatNet(num_frames=2, fusion_mode=mode)
        model_2.eval()

        with torch.no_grad():
            out_2 = model_2(video, audio)

        assert torch.allclose(out_1["logits"], out_2["logits"], atol=1e-7), f"Logits mismatch in mode '{mode}'"
        assert torch.allclose(out_1["probabilities"], out_2["probabilities"], atol=1e-7)
        assert torch.allclose(out_1["expected_intensity"], out_2["expected_intensity"], atol=1e-7)


def test_video_backbone_no_custom_init_weights():
    """Verify invariant rule: Video backbone relies on default PyTorch initialization without calling _init_weights."""
    model = MultimodalFlatNet(num_frames=2, fusion_mode="sum")
    assert not hasattr(model.video_backbone, "_init_weights"), "video_backbone must NOT have _init_weights method."


def test_config_seed_lock():
    """Verify that config loads master seed 42 from train_config.json."""
    config_path = str(Path(project_root) / "config" / "train_config.json")
    cfg = TrainConfig.from_json(config_path)

    assert cfg.system.seed == 42, f"Expected system.seed == 42, got {cfg.system.seed}"
    assert cfg.dataset_splitter.seed == 42, f"Expected dataset_splitter.seed == 42, got {cfg.dataset_splitter.seed}"
    assert cfg.seed == 42, f"Expected property cfg.seed == 42, got {cfg.seed}"


def test_no_nan_or_inf_in_initialized_weights():
    """Verify that all initial weights across all models are finite numbers (no NaN, no Inf)."""
    for mode in ["sum", "concat", "gated"]:
        model = MultimodalFlatNet(num_frames=2, fusion_mode=mode)
        for name, param in model.named_parameters():
            assert not torch.isnan(param).any(), f"NaN detected in {name} of mode '{mode}'"
            assert not torch.isinf(param).any(), f"Inf detected in {name} of mode '{mode}'"
