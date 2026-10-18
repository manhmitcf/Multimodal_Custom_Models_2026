import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from models.audio_backbone import ModularAudioMLPBackbone, AudioMLPBackbone


def test_modular_audio_mlp_backbone_zero_hidden():
    """Test 0 hidden layers: 2049 -> 384"""
    model = ModularAudioMLPBackbone(in_features=2049, hidden_dims=[], embed_dim=384)
    x = torch.randn(2, 2049, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 384), f"Expected shape (2, 384), got {out.shape}"
    loss = out.sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient!"
    print("PASS: test_modular_audio_mlp_backbone_zero_hidden")


def test_modular_audio_mlp_backbone_one_hidden():
    """Test 1 hidden layer: 2049 -> 768 -> 384"""
    model = ModularAudioMLPBackbone(in_features=2049, hidden_dims=[768], embed_dim=384, dropout=0.1)
    x = torch.randn(4, 2049)
    out = model(x)
    assert out.shape == (4, 384), f"Expected shape (4, 384), got {out.shape}"
    loss = out.sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient!"
    print("PASS: test_modular_audio_mlp_backbone_one_hidden")


def test_modular_audio_mlp_backbone_two_hidden():
    """Test 2 hidden layers: 2049 -> 1024 -> 512 -> 384"""
    model = ModularAudioMLPBackbone(in_features=2049, hidden_dims=[1024, 512], embed_dim=384)
    x = torch.randn(2, 2049)
    out = model(x)
    assert out.shape == (2, 384), f"Expected shape (2, 384), got {out.shape}"
    loss = out.sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient!"
    print("PASS: test_modular_audio_mlp_backbone_two_hidden")


def test_audio_mlp_backbone_backward_compatibility():
    """Test AudioMLPBackbone maintains backward compatibility with legacy 224 dim"""
    model = AudioMLPBackbone(in_features=2049, hidden_dim=512, embed_dim=224)
    x = torch.randn(2, 2049)
    out = model(x)
    assert out.shape == (2, 224), f"Expected shape (2, 224), got {out.shape}"
    print("PASS: test_audio_mlp_backbone_backward_compatibility")


def test_legacy_checkpoint_key_loading():
    """Test that legacy state_dict with fc1, ln1, fc2, ln2 keys loads cleanly into AudioMLPBackbone."""
    legacy_state_dict = {
        "fc1.weight": torch.randn(512, 2049),
        "fc1.bias": torch.randn(512),
        "ln1.weight": torch.ones(512),
        "ln1.bias": torch.zeros(512),
        "fc2.weight": torch.randn(224, 512),
        "fc2.bias": torch.randn(224),
        "ln2.weight": torch.ones(224),
        "ln2.bias": torch.zeros(224),
    }
    expected_w = legacy_state_dict["fc1.weight"].clone()
    model = AudioMLPBackbone(in_features=2049, hidden_dim=512, embed_dim=224)
    model.load_state_dict(legacy_state_dict, strict=True)
    assert torch.equal(model.net[0].weight, expected_w)
    print("PASS: test_legacy_checkpoint_key_loading")


if __name__ == "__main__":
    test_modular_audio_mlp_backbone_zero_hidden()
    test_modular_audio_mlp_backbone_one_hidden()
    test_modular_audio_mlp_backbone_two_hidden()
    test_audio_mlp_backbone_backward_compatibility()
    test_legacy_checkpoint_key_loading()
    print("ALL TESTS IN test_audio_backbone.py PASSED!")
