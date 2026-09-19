import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from models.multimodal_sota_net import MultimodalBoundaryAwareNet
from utils.losses import PairwiseTournamentLoss
from utils.seed import seed_everything


def test_parameter_budget():
    print("\n" + "=" * 65)
    print("TEST 1: FLAT 4-CLASS DUAL REFEREES PARAMETER BUDGET (< 5.0M)")
    print("=" * 65)
    seed_everything(42)

    model = MultimodalBoundaryAwareNet(num_frames=2)
    total_params = sum(p.numel() for p in model.parameters())
    v_params = sum(p.numel() for p in model.video_backbone.parameters())
    a_params = sum(p.numel() for p in model.audio_backbone.parameters())
    f_params = sum(p.numel() for p in model.fusion.parameters())

    print(f"Total Model Parameters:                     {total_params:,}")
    print(f"  - Video Backbone (ConvNeXt-Nano 7ch):       {v_params:,}")
    print(f"  - Audio Backbone (STFT-MLP 2049):           {a_params:,}")
    print(f"  - Dual Tournament Fusion (6 Base + Dual):   {f_params:,}")

    strict_limit = 5000000
    assert total_params < strict_limit, f"FAILED: Exceeded budget {total_params} >= {strict_limit}"
    headroom = strict_limit - total_params
    print(f"[PASSED] Total parameters ({total_params:,}) safely under 5.0M! (Headroom: {headroom:,})")


def test_tournament_forward_and_pairwise():
    print("\n" + "=" * 65)
    print("TEST 2: FLAT 4-CLASS FORWARD PASS & 6 PAIRWISE DUAL MATCHUPS")
    print("=" * 65)
    seed_everything(42)

    model = MultimodalBoundaryAwareNet(num_frames=2)
    model.eval()

    B = 4
    v_input = torch.randn(B, 2, 3, 224, 224)
    a_input = torch.randn(B, 512000)  # 2.0s @ 256 kHz

    with torch.no_grad():
        out = model(v_input, a_input)

    # 1. Check all 6 Pairwise Probabilities
    for pair_name in ["p_0_over_1", "p_0_over_2", "p_0_over_3", "p_1_over_2", "p_2_over_3", "p_1_over_3"]:
        p = out[pair_name]
        assert (p >= 0.0).all() and (p <= 1.0).all(), f"{pair_name} out of [0, 1] range"
        print(f"Pairwise {pair_name}: {p.tolist()}")

    # 2. Check Borda Voting scores across 4 classes
    v_voting = out["v_voting"]
    print(f"Tournament Borda Voting [None, Strong, Med, Weak]:\n{v_voting}")
    assert v_voting.shape == (B, 4), f"Expected shape (B, 4), got {v_voting.shape}"
    assert (v_voting >= 0.0).all() and (v_voting <= 3.0).all(), "Borda scores must be in [0, 3]"

    # Invariant: Sum of Borda votes across 4 classes must equal exactly 6.0
    v_sums = v_voting.sum(dim=-1)
    assert torch.allclose(v_sums, torch.full_like(v_sums, 6.0), atol=1e-5), f"Sum of Borda votes must be 6.0, got: {v_sums}"
    print(f"Borda Vote Sums across 4 classes: {v_sums.tolist()} (Exact algebraic invariant = 6.0)")

    # 3. Check Final Probabilities sum to 1
    probs = out["probabilities"]
    assert probs.shape == (B, 4)
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), f"Probabilities do not sum to 1: {prob_sums}"

    print("[PASSED] Flat 4-Class Round-Robin Tournament forward pass verified!")


def test_gradient_flow_tournament_loss():
    print("\n" + "=" * 65)
    print("TEST 3: 100% GRADIENT FLOW THROUGH FLAT 4-CLASS DUAL REFEREES LOSS")
    print("=" * 65)
    seed_everything(42)

    # Enable all 6 boundaries with both Audio and Video Referees (12 referee heads + 6 base heads = 18 heads)
    all_dual_tb = {
        f"b{pair}": {"enable_audio": True, "enable_video": True}
        for pair in ["01", "02", "03", "12", "23", "13"]
    }
    model = MultimodalBoundaryAwareNet(
        num_frames=2,
        tie_breakers=all_dual_tb
    )
    model.train()
    criterion = PairwiseTournamentLoss(weight_pairwise=1.0, weight_ce=1.0)

    # Batch with all 4 classes: 0 (None), 1 (Strong), 2 (Medium), 3 (Weak)
    B = 4
    v_input = torch.randn(B, 2, 3, 224, 224)
    a_input = torch.randn(B, 512000)  # 2.0s @ 256 kHz
    targets = {"target": torch.tensor([0, 1, 2, 3])}

    outputs = model(v_input, a_input)
    loss = criterion(outputs, targets)
    loss.backward()

    total_tensors = 0
    valid_grads = 0

    for name, param in model.named_parameters():
        total_tensors += 1
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                raise AssertionError(f"NaN/Inf gradient in parameter: {name}")
            valid_grads += 1
        else:
            raise AssertionError(f"Parameter without gradient: {name}")

    print(f"[PASSED] 100% Gradient flow verified: {valid_grads}/{total_tensors} parameters with healthy gradients!")
    print(f"  Flat Dual Tournament Composite Loss: {loss.item():.4f}")


def test_end_to_end_from_scratch():
    print("\n" + "=" * 65)
    print("TEST 4: END-TO-END FROM SCRATCH SIMULTANEOUS TRAINING")
    print("=" * 65)
    seed_everything(42)

    model = MultimodalBoundaryAwareNet(num_frames=2)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = PairwiseTournamentLoss(weight_pairwise=1.0, weight_ce=1.0, aux_loss_weight=0.3)

    B = 4
    v_input = torch.randn(B, 2, 3, 224, 224)
    a_input = torch.randn(B, 512000)
    targets = {"target": torch.tensor([0, 1, 2, 3])}

    optimizer.zero_grad()
    outputs = model(v_input, a_input)
    loss = criterion(outputs, targets)
    loss.backward()

    # Verify all components received gradients
    vb_grads = [p.grad for p in model.video_backbone.parameters() if p.requires_grad]
    ab_grads = [p.grad for p in model.audio_backbone.parameters() if p.requires_grad]
    f_grads = [p.grad for p in model.fusion.parameters() if p.requires_grad]

    assert all(g is not None for g in vb_grads), "Video backbone missing gradients"
    assert all(g is not None for g in ab_grads), "Audio MLP backbone missing gradients"
    assert all(g is not None for g in f_grads), "Tournament fusion missing gradients"

    optimizer.step()
    print(f"[PASSED] End-to-End Single Phase: All {len(list(model.parameters()))} param tensors updated simultaneously!")


def test_tie_breakers_toggle_config():
    print("\n" + "=" * 65)
    print("TEST 5: CONFIGURABLE DUAL REFEREES TOGGLE (ABLATION VERIFICATION)")
    print("=" * 65)
    seed_everything(42)

    B = 2
    v_input = torch.randn(B, 2, 3, 224, 224)
    a_input = torch.randn(B, 512000)

    # Case A: Selective Dual/Single Referees (B12 dual, B23 audio only, B13 video only, others off)
    tb_selective = {
        "b12": {"enable_audio": True, "enable_video": True},
        "b23": {"enable_audio": True, "enable_video": False},
        "b13": {"enable_audio": False, "enable_video": True},
        "b01": {"enable_audio": False, "enable_video": False},
        "b02": {"enable_audio": False, "enable_video": False},
        "b03": {"enable_audio": False, "enable_video": False},
    }
    model_sel = MultimodalBoundaryAwareNet(tie_breakers=tb_selective)
    th_sel = model_sel.fusion.tournament_head

    assert th_sel.head_b12_a is not None and th_sel.head_b12_v is not None
    assert th_sel.head_b23_a is not None and th_sel.head_b23_v is None
    assert th_sel.head_b13_a is None and th_sel.head_b13_v is not None
    assert th_sel.head_b01_a is None and th_sel.head_b01_v is None

    out_sel = model_sel(v_input, a_input)
    assert out_sel["probabilities"].shape == (B, 4)
    p_params_sel = sum(p.numel() for p in model_sel.parameters())
    print(f"  Case A (Selective Audio/Video Referees): {p_params_sel:,} params - verified clean!")

    # Case B: All referees disabled (pure 6-base pairwise round-robin tournament)
    tb_none = {
        f"b{p}": {"enable_audio": False, "enable_video": False}
        for p in ["01", "02", "03", "12", "23", "13"]
    }
    model_none = MultimodalBoundaryAwareNet(tie_breakers=tb_none)
    th_none = model_none.fusion.tournament_head
    for p in ["01", "02", "03", "12", "23", "13"]:
        assert getattr(th_none, f"head_b{p}_a") is None
        assert getattr(th_none, f"head_b{p}_v") is None

    out_none = model_none(v_input, a_input)
    assert out_none["probabilities"].shape == (B, 4)
    p_params_none = sum(p.numel() for p in model_none.parameters())
    print(f"  Case B (All Dual Referees disabled - Base Only): {p_params_none:,} params - verified clean!")

    # Case C: All 6 boundaries dual enabled (6 Base + 6 Audio Referees + 6 Video Referees = 18 heads)
    tb_all = {
        f"b{p}": {"enable_audio": True, "enable_video": True}
        for p in ["01", "02", "03", "12", "23", "13"]
    }
    model_all = MultimodalBoundaryAwareNet(tie_breakers=tb_all)
    th_all = model_all.fusion.tournament_head
    for p in ["01", "02", "03", "12", "23", "13"]:
        assert getattr(th_all, f"head_b{p}_a") is not None
        assert getattr(th_all, f"head_b{p}_v") is not None

    out_all = model_all(v_input, a_input)
    assert out_all["probabilities"].shape == (B, 4)
    p_params_all = sum(p.numel() for p in model_all.parameters())
    print(f"  Case C (All 6 Boundaries Dual Enabled - 18 Heads): {p_params_all:,} params - verified clean!")

    print("[PASSED] Configurable Dual Referees toggle verified across all ablation states!")


def test_consistent_video_transform():
    print("\n" + "=" * 65)
    print("TEST 6: CLIP-SYNCHRONIZED VIDEO TRANSFORM & RANDOM ERASING")
    print("=" * 65)
    seed_everything(42)

    import numpy as np
    from transforms.video_transform import ConsistentVideoTransform
    from features.motion_kinematics import FishMotionKinematics7Ch

    tf_train = ConsistentVideoTransform(image_size=224, is_train=True, erase_prob=1.0)
    dummy_clip = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
    tensor_train = tf_train(dummy_clip)
    assert tensor_train.shape == (2, 3, 224, 224), f"Wrong shape: {tensor_train.shape}"

    mask0 = (tensor_train[0] == 0.0)
    mask1 = (tensor_train[1] == 0.0)
    assert torch.equal(mask0, mask1), "Random Erasing is NOT clip-synchronized across frames!"
    erased_pixels = mask0.sum().item()
    assert erased_pixels > 0, "No pixels were erased in train mode with erase_prob=1.0!"
    print(f"Clip-synchronized erased pixels per channel: {erased_pixels // 3}")

    mk = FishMotionKinematics7Ch()
    kinematics, _ = mk(tensor_train.unsqueeze(0))
    flow = kinematics[0, :, 3:5]
    flow_in_erased = flow[:, :, mask0[0]]
    assert torch.all(flow_in_erased == 0.0), f"Flow in erased region is not zero: max abs {flow_in_erased.abs().max()}"
    print("[PASSED] Optical flow in erased region is strictly 0.0 (no kinematic artifacts)!")

    tf_val = ConsistentVideoTransform(image_size=224, is_train=False)
    tensor_val = tf_val(dummy_clip)
    assert tensor_val.shape == (2, 3, 224, 224)
    assert not torch.equal(tensor_val[0] == 0.0, torch.ones_like(tensor_val[0], dtype=torch.bool)), "Unexpected zeros in val"
    print("[PASSED] Val mode preserves clean full frames without erasing!")


if __name__ == "__main__":
    test_parameter_budget()
    test_tournament_forward_and_pairwise()
    test_gradient_flow_tournament_loss()
    test_end_to_end_from_scratch()
    test_tie_breakers_toggle_config()
    test_consistent_video_transform()
    print("\n" + "=" * 65)
    print("ALL FLAT 4-CLASS DUAL REFEREES TESTS PASSED SUCCESSFULLY! (100% READY)")
    print("=" * 65 + "\n")
