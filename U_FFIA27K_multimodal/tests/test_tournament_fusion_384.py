import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from models.multimodal_fusion import MultimodalTournamentFusion

def test_tournament_fusion_direct_gated_384():
    fusion = MultimodalTournamentFusion(dim=384, enable_b12=False, enable_b23=True, enable_b13=False)
    f_v = torch.randn(4, 384)
    f_a = torch.randn(4, 384)
    out = fusion(f_video=f_v, f_audio=f_a)

    assert out["clipwise_output"].shape == (4, 4)
    assert out["probabilities"].shape == (4, 4)
    assert out["f_fused"].shape == (4, 384)
    assert not hasattr(fusion, "proj_joint")  # Đã loại bỏ proj_joint

    th = fusion.tournament_head
    assert th.activity_head[0].out_features == 192
    assert isinstance(th.activity_head[2], torch.nn.LayerNorm)

    # Kiểm tra Borda count invariant: V_W + V_M + V_S = 3.0
    v_sum = out["v_voting"].sum(dim=-1)
    assert torch.allclose(v_sum, torch.tensor([3.0] * 4), atol=1e-5)

def test_tournament_fusion_all_3_referees_params():
    fusion_all3 = MultimodalTournamentFusion(dim=384, enable_b12=True, enable_b23=True, enable_b13=True)
    params_all3 = sum(p.numel() for p in fusion_all3.parameters())
    assert params_all3 == 523_019  # Khớp chính xác tính toán kiểm toán QA
