import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from models.video_backbone import ConvNeXtNanoVideoBackbone

def test_convnext_nano_5ch_384_composite_ln():
    model = ConvNeXtNanoVideoBackbone(in_chans=5, embed_dim=384, temporal_mode="composite_ln")
    dummy_input = torch.randn(2, 2, 5, 224, 224)
    out = model(dummy_input)
    assert out.shape == (2, 384)
    params = sum(p.numel() for p in model.parameters())
    assert params == 2_613_408  # Khớp chính xác con số kiểm toán vi mô
    assert not hasattr(model, "proj")  # Không còn Linear(384 -> 224)

def test_convnext_nano_single_frame_guard():
    model = ConvNeXtNanoVideoBackbone(in_chans=5, embed_dim=384, temporal_mode="composite_ln")
    single_frame = torch.randn(2, 1, 5, 224, 224)
    out = model(single_frame)
    assert out.shape == (2, 384)

def test_convnext_nano_temporal_modes():
    dummy_input = torch.randn(2, 2, 5, 224, 224)
    for mode in ["sum", "abs_diff"]:
        m = ConvNeXtNanoVideoBackbone(in_chans=5, embed_dim=384, temporal_mode=mode)
        out = m(dummy_input)
        assert out.shape == (2, 384)
