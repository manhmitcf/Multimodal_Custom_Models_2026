import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import pytest
from models.multimodal_sota_net import MultimodalSOTANet


def test_multimodal_sota_net_full_pipeline():
    model = MultimodalSOTANet(
        embed_dim=384,
        in_chans=5,
        temporal_mode="composite_ln"
    )
    model.eval()
    v = torch.randn(2, 2, 3, 224, 224)
    a = torch.randn(2, 512000)
    with torch.no_grad():
        out = model(v, a)

    assert out["clipwise_output"].shape == (2, 4)
    assert out["probabilities"].shape == (2, 4)
    assert out["logits_video"].shape == (2, 4)
    assert out["logits_audio"].shape == (2, 4)
    assert out["f_fused"].shape == (2, 384)

    # Verify Borda voting score invariant
    assert "v_voting" in out
    v_sum = out["v_voting"].sum(dim=-1)
    assert torch.allclose(v_sum, torch.tensor([3.0, 3.0]), atol=1e-5)

    # Exact parameter count for default configuration (B23 enabled):
    total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_p == 5_163_057, f"Total params mismatch: {total_p} != 5,163,057"


def test_multimodal_sota_net_all_three_referees():
    model = MultimodalSOTANet(
        embed_dim=384,
        in_chans=5,
        temporal_mode="composite_ln",
        enable_b12=True,
        enable_b23=True,
        enable_b13=True
    )
    total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_p == 5_312_053, f"Total params mismatch with 3 referees: {total_p} != 5,312,053"
