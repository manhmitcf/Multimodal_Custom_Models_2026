import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from models.audio_sota_net import AudioSOTANet


def test_audio_sota_net_single_arg():
    """Test AudioSOTANet called with model(audio_input)"""
    model = AudioSOTANet(hidden_dims=[768], embed_dim=384, classes_num=4)
    dummy_audio = torch.randn(2, 512000)
    out = model(dummy_audio)
    assert isinstance(out, dict), "Output must be a dict"
    assert "clipwise_output" in out, "Missing 'clipwise_output' key"
    assert out["clipwise_output"].shape == (2, 4), f"Expected shape (2, 4), got {out['clipwise_output'].shape}"
    assert "f_audio" in out
    assert out["f_audio"].shape == (2, 384), f"Expected shape (2, 384), got {out['f_audio'].shape}"

    # Backward pass
    loss = out["clipwise_output"].sum()
    loss.backward()
    trainable_with_grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert trainable_with_grads == total_trainable, f"Gradient flow broken: {trainable_with_grads}/{total_trainable}"
    print("PASS: test_audio_sota_net_single_arg")


def test_audio_sota_net_two_args():
    """Test AudioSOTANet called with model(dummy_video, dummy_audio) for profiler compatibility"""
    model = AudioSOTANet(hidden_dims=[512], embed_dim=384, classes_num=4)
    dummy_video = torch.randn(2, 2, 3, 224, 224)
    dummy_audio = torch.randn(2, 512000)
    out = model(dummy_video, dummy_audio)
    assert "clipwise_output" in out
    assert out["clipwise_output"].shape == (2, 4)
    print("PASS: test_audio_sota_net_two_args")


def test_audio_sota_net_keyword_args():
    """Test AudioSOTANet called with keyword arguments"""
    model = AudioSOTANet(hidden_dims=[], embed_dim=384, classes_num=4)
    dummy_audio = torch.randn(2, 512000)
    out = model(audio_input=dummy_audio)
    assert out["clipwise_output"].shape == (2, 4)
    print("PASS: test_audio_sota_net_keyword_args")


if __name__ == "__main__":
    test_audio_sota_net_single_arg()
    test_audio_sota_net_two_args()
    test_audio_sota_net_keyword_args()
    print("ALL TESTS IN test_audio_sota_net.py PASSED!")
