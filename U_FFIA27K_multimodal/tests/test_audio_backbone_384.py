import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from models.audio_backbone import AudioMLPBackbone

def test_audio_mlp_l2_compact_768_512_384():
    model = AudioMLPBackbone(in_features=2049, hidden_dims=(768, 512), embed_dim=384, dropout=0.1)
    dummy_stft = torch.randn(2, 2049)
    out = model(dummy_stft)
    assert out.shape == (2, 384)
    params = sum(p.numel() for p in model.parameters())
    assert params == 2_168_448  # Khớp chính xác tính toán lý thuyết

def test_audio_mlp_multi_dim_input():
    model = AudioMLPBackbone(in_features=2049, hidden_dims=(768, 512), embed_dim=384, dropout=0.1)
    dummy_stft_3d = torch.randn(2, 1, 2049)
    out = model(dummy_stft_3d)
    assert out.shape == (2, 384)
