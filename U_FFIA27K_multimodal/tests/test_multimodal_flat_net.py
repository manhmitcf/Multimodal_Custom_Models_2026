import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from models.multimodal_flat_net import MultimodalFlatNet, Flat4ClassMLPHead


def test_flat_4class_mlp_head_dimensions():
    # Test 384 in_dim
    head_384 = Flat4ClassMLPHead(in_dim=384, hidden_dim=192, num_classes=4)
    x_384 = torch.randn(2, 384)
    out_384 = head_384(x_384)
    assert out_384.shape == (2, 4)
    assert sum(p.numel() for p in head_384.parameters()) == 75_076

    # Test 768 in_dim
    head_768 = Flat4ClassMLPHead(in_dim=768, hidden_dim=192, num_classes=4)
    x_768 = torch.randn(2, 768)
    out_768 = head_768(x_768)
    assert out_768.shape == (2, 4)
    assert sum(p.numel() for p in head_768.parameters()) == 148_804


def test_multimodal_flat_net_sum():
    model = MultimodalFlatNet(num_frames=2, embed_dim=384, fusion_mode="sum")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert params == 4_864_878, f"Expected 4,864,878 params for sum mode, got {params}"

    v = torch.randn(2, 2, 3, 224, 224)
    a = torch.randn(2, 512000)
    out = model(v, a)

    assert out["logits"].shape == (2, 4)
    assert out["probabilities"].shape == (2, 4)
    assert out["f_fused"].shape == (2, 384)
    assert out.get("gate") is None
    assert "logits_video_aux" in out and "logits_audio_aux" in out


def test_multimodal_flat_net_concat():
    model = MultimodalFlatNet(num_frames=2, embed_dim=384, fusion_mode="concat")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert params == 4_939_374, f"Expected 4,939_374 params for concat mode, got {params}"

    v = torch.randn(2, 2, 3, 224, 224)
    a = torch.randn(2, 512000)
    out = model(v, a)

    assert out["logits"].shape == (2, 4)
    assert out["probabilities"].shape == (2, 4)
    assert out["f_fused"].shape == (2, 768)
    assert out.get("gate") is None


def test_multimodal_flat_net_gated():
    model = MultimodalFlatNet(num_frames=2, embed_dim=384, fusion_mode="gated")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert params == 4_865_647, f"Expected 4,865,647 params for gated mode, got {params}"

    v = torch.randn(2, 2, 3, 224, 224)
    a = torch.randn(2, 512000)
    out = model(v, a)

    assert out["logits"].shape == (2, 4)
    assert out["probabilities"].shape == (2, 4)
    assert out["f_fused"].shape == (2, 384)
    assert out.get("gate") is not None
    assert out["gate"].shape == (2, 1)
