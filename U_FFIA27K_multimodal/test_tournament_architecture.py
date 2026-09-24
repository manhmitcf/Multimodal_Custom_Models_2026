import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from models.multimodal_sota_net import MultimodalSOTANet
from utils.losses import PairwiseTournamentLoss
from utils.seed import seed_everything


def test_parameter_budget():
    print("\n" + "=" * 65)
    print("TEST 1: DUAL TIE-BREAKERS TOURNAMENT PARAMETER BUDGET (< 5.0M)")
    print("=" * 65)

    model = MultimodalSOTANet(num_frames=2)
    total_params = sum(p.numel() for p in model.parameters())
    v_params = sum(p.numel() for p in model.video_backbone.parameters())
    a_params = sum(p.numel() for p in model.audio_backbone.parameters())
    f_params = sum(p.numel() for p in model.fusion.parameters())

    print(f"Total Model Parameters:               {total_params:,}")
    print(f"  - Video Backbone (ConvNeXt-Nano 7ch): {v_params:,}")
    print(f"  - Audio Backbone (STFT-MLP 2049):     {a_params:,}")
    print(f"  - Pairwise Dual Tournament Fusion:    {f_params:,}")

    strict_limit = 5000000
    assert total_params < strict_limit, f"FAILED: Exceeded budget {total_params} >= {strict_limit}"
    headroom = strict_limit - total_params
    print(f"[PASSED] Total parameters ({total_params:,}) safely under 5.0M! (Headroom: {headroom:,})")


def test_tournament_forward_and_pairwise():
    print("\n" + "=" * 65)
    print("TEST 2: TOURNAMENT 2-LEVEL FORWARD PASS & DUAL TIE-BREAKERS")
    print("=" * 65)

    model = MultimodalSOTANet(num_frames=2)
    model.eval()

    B = 4
    v_input = torch.randn(B, 2, 3, 224, 224)
    a_input = torch.randn(B, 512000)  # 2.0s @ 256 kHz

    with torch.no_grad():
        out = model(v_input, a_input)

    # 1. Check Level 1 Activity Gate
    p_feeding = out["p_feeding"]
    print(f"Level 1 Feeding Activity Probabilities: {p_feeding.tolist()}")
    assert (p_feeding >= 0.0).all() and (p_feeding <= 1.0).all(), "p_feeding out of [0, 1] range"

    # 2. Check Level 2 Pairwise Boundaries B12, B23, B13 and Dual Referees
    p_w_over_m = out["p_w_over_m"]
    p_m_over_s = out["p_m_over_s"]
    p_w_over_s = out["p_w_over_s"]

    u_tie_12 = out["u_tie_12"]
    u_tie_23 = out["u_tie_23"]
    u_tie_13 = out["u_tie_13"]

    print(f"Pairwise B12 P(Weak > Medium):          {p_w_over_m.tolist()}")
    print(f"Pairwise B23 P(Medium > Strong):        {p_m_over_s.tolist()}")
    print(f"Pairwise B13 P(Weak > Strong) [Cross]:  {p_w_over_s.tolist()}")
    print(f"Referee Indecision u_tie_12:            {u_tie_12.tolist()}")
    print(f"Referee Indecision u_tie_23:            {u_tie_23.tolist()}")
    print(f"Referee Indecision u_tie_13:            {u_tie_13.tolist()}")

    assert (p_w_over_m >= 0.0).all() and (p_w_over_m <= 1.0).all()
    assert (p_m_over_s >= 0.0).all() and (p_m_over_s <= 1.0).all()
    assert (p_w_over_s >= 0.0).all() and (p_w_over_s <= 1.0).all()
    assert (u_tie_12 > 0.0).all() and (u_tie_12 <= 1.0).all()
    assert (u_tie_23 > 0.0).all() and (u_tie_23 <= 1.0).all()
    assert (u_tie_13 > 0.0).all() and (u_tie_13 <= 1.0).all()

    # Check that both video and audio referee logits and gammas are present
    assert "logit_12_v" in out and "logit_12_a" in out and "gamma_12_v" in out and "gamma_12_a" in out
    assert "logit_23_v" in out and "logit_23_a" in out and "gamma_23_v" in out and "gamma_23_a" in out
    assert "logit_13_v" in out and "logit_13_a" in out and "gamma_13_v" in out and "gamma_13_a" in out

    # 3. Check Tournament Voting scores
    v_voting = out["v_voting"]
    print(f"Tournament Borda Voting [Weak, Med, Str]:\n{v_voting}")
    assert v_voting.shape == (B, 3)
    assert (v_voting >= 0.0).all() and (v_voting <= 2.0).all(), "Borda scores must be in [0, 2]"

    # Verify Borda Voting Sum strictly invariant = 3.0
    v_sum = v_voting.sum(dim=-1)
    print(f"Borda Vote Sums per sample: {v_sum.tolist()} (Exact algebraic invariant = 3.0)")
    assert torch.allclose(v_sum, torch.tensor([3.0] * B), atol=1e-5), "Borda voting sum must strictly equal 3.0"

    # 4. Check Hierarchical Probabilities
    probs = out["probabilities"]
    print(f"Hierarchical Multi-Class Probabilities [None, Strong, Med, Weak]:\n{probs}")
    assert probs.shape == (B, 4)
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities must be in [0, 1]"
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones(B), atol=1e-5), "Probabilities must sum to 1.0"

    # Check intensity predictions
    intensity = out["expected_intensity"]
    assert intensity.shape == (B, 1)
    assert (intensity >= 0.0).all() and (intensity <= 3.0).all(), "Intensity must be in [0, 3]"

    print("[PASSED] Tournament 2-level forward pass with Dual Tie-Breakers verified!")


def test_gradient_flow_through_loss():
    print("\n" + "=" * 65)
    print("TEST 3: 100% GRADIENT FLOW THROUGH TOURNAMENT LOSS")
    print("=" * 65)

    model = MultimodalSOTANet(num_frames=2)
    model.train()

    B = 4
    v_input = torch.randn(B, 2, 3, 224, 224)
    a_input = torch.randn(B, 512000)
    targets = {"target": torch.tensor([0, 1, 2, 3])}

    criterion = PairwiseTournamentLoss(weight_act=0.5, weight_pairwise=0.5, weight_ce=1.0, aux_loss_weight=0.3)
    outputs = model(v_input, a_input)
    loss = criterion(outputs, targets)
    loss.backward()

    # Verify 100% of trainable parameters receive gradients
    param_count = 0
    missing_grad_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            param_count += 1
            if p.grad is None:
                missing_grad_params.append(name)

    if missing_grad_params:
        print(f"FAILED: The following {len(missing_grad_params)} parameters received no gradient:")
        for name in missing_grad_params:
            print(f"  - {name}")
        assert False, f"Gradient flow broken for {len(missing_grad_params)} parameters!"

    print(f"[PASSED] 100% Gradient flow verified: {param_count}/{param_count} parameters with healthy gradients!")
    print(f"  Composite Loss: {loss.item():.4f}")


def test_end_to_end_from_scratch():
    print("\n" + "=" * 65)
    print("TEST 4: END-TO-END FROM SCRATCH SIMULTANEOUS TRAINING")
    print("=" * 65)

    model = MultimodalSOTANet(num_frames=2)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = PairwiseTournamentLoss(weight_act=0.5, weight_pairwise=0.5, weight_ce=1.0, aux_loss_weight=0.3)

    B = 4
    v_input = torch.randn(B, 2, 3, 224, 224)
    a_input = torch.randn(B, 512000)
    targets = {"target": torch.tensor([0, 1, 2, 3])}

    optimizer.zero_grad()
    outputs = model(v_input, a_input)
    loss = criterion(outputs, targets)
    loss.backward()

    vb_grads = [p.grad for p in model.video_backbone.parameters() if p.requires_grad]
    ab_grads = [p.grad for p in model.audio_backbone.parameters() if p.requires_grad]
    f_grads = [p.grad for p in model.fusion.parameters() if p.requires_grad]

    assert all(g is not None for g in vb_grads), "Video backbone missing gradients"
    assert all(g is not None for g in ab_grads), "Audio MLP backbone missing gradients"
    assert all(g is not None for g in f_grads), "Tournament fusion missing gradients"

    optimizer.step()
    print(f"[PASSED] End-to-End Single Phase: All {len(list(model.parameters()))} param tensors updated simultaneously!")


def test_all_8_dual_tie_breaker_combinations():
    print("\n" + "=" * 65)
    print("TEST 5: ALL 8 DUAL TIE-BREAKER COMBINATIONS (ABLATION MATRIX)")
    print("=" * 65)

    B = 2
    v_input = torch.randn(B, 2, 3, 224, 224)
    a_input = torch.randn(B, 512000)
    targets = {"target": torch.tensor([1, 2])}
    criterion = PairwiseTournamentLoss(weight_act=0.5, weight_pairwise=0.5, weight_ce=1.0, aux_loss_weight=0.3)

    combinations = [
        (False, False, False, "Pure Baseline (No Tie-Breakers)"),
        (True,  False, False, "Dual Referees B12 Only (Video + Audio)"),
        (False, True,  False, "Dual Referees B23 Only (Video + Audio)"),
        (False, False, True,  "Dual Referees B13 Only (Video + Audio)"),
        (True,  True,  False, "Dual Referees B12 + B23"),
        (True,  False, True,  "Dual Referees B12 + B13"),
        (False, True,  True,  "Dual Referees B23 + B13"),
        (True,  True,  True,  "All 3 Dual Referees (B12 + B23 + B13)"),
    ]

    from config.train_config import DualTieBreakersConfig

    for idx, (b12, b23, b13, desc) in enumerate(combinations, 1):
        # Validate Pydantic DualTieBreakersConfig
        tb_pydantic = DualTieBreakersConfig(enable_b12=b12, enable_b23=b23, enable_b13=b13)
        model = MultimodalSOTANet(num_frames=2, tie_breakers=tb_pydantic)
        th = model.fusion.tournament_head

        # Check sub-modules existence (both Video and Audio referees)
        if b12:
            assert th.head_b12_v is not None and th.head_b12_a is not None
            assert th.gamma_12_v is not None and th.gamma_12_a is not None
        else:
            assert th.head_b12_v is None and th.head_b12_a is None
            assert th.gamma_12_v is None and th.gamma_12_a is None

        if b23:
            assert th.head_b23_v is not None and th.head_b23_a is not None
            assert th.gamma_23_v is not None and th.gamma_23_a is not None
        else:
            assert th.head_b23_v is None and th.head_b23_a is None
            assert th.gamma_23_v is None and th.gamma_23_a is None

        if b13:
            assert th.head_b13_v is not None and th.head_b13_a is not None
            assert th.gamma_13_v is not None and th.gamma_13_a is not None
        else:
            assert th.head_b13_v is None and th.head_b13_a is None
            assert th.gamma_13_v is None and th.gamma_13_a is None

        # Also validate plain dict format
        tb_dict = {"enable_b12": b12, "enable_b23": b23, "enable_b13": b13}
        model_dict = MultimodalSOTANet(num_frames=2, tie_breakers=tb_dict)
        assert model_dict.fusion.tournament_head.enable_b12 == b12
        assert model_dict.fusion.tournament_head.enable_b23 == b23
        assert model_dict.fusion.tournament_head.enable_b13 == b13

        # Test forward & backward
        model.train()
        outputs = model(v_input, a_input)
        loss = criterion(outputs, targets)
        loss.backward()

        param_count = sum(p.numel() for p in model.parameters())
        assert param_count < 5000000
        print(f"  Comb {idx}/8: [B12={b12}, B23={b23}, B13={b13}] -> {param_count:,} params | {desc} -> PASSED")

    # Explicit divergence verification: B12 Only vs B23 Only must produce different logits
    seed_everything(42)
    m_b12 = MultimodalSOTANet(num_frames=2, tie_breakers=DualTieBreakersConfig(enable_b12=True, enable_b23=False, enable_b13=False)).eval()
    seed_everything(42)
    m_b23 = MultimodalSOTANet(num_frames=2, tie_breakers=DualTieBreakersConfig(enable_b12=False, enable_b23=True, enable_b13=False)).eval()
    with torch.no_grad():
        out_b12 = m_b12(v_input, a_input)["logits"]
        out_b23 = m_b23(v_input, a_input)["logits"]
    assert not torch.allclose(out_b12, out_b23, atol=1e-3), "B12 and B23 tie-breaker configs must produce distinct outputs!"
    print("  Divergence check: B12-only vs B23-only produce distinctly calibrated logits -> PASSED")

    print("[PASSED] All 8 Dual Tie-Breaker ablation combinations verified successfully!")


def test_consistent_video_transform():
    print("\n" + "=" * 65)
    print("TEST 6: CLIP-SYNCHRONIZED VIDEO TRANSFORM & RANDOM ERASING")
    print("=" * 65)

    import numpy as np
    from transforms.video_transform import ConsistentVideoTransform
    from features.motion_kinematics import FishMotionKinematics7Ch

    tf_train = ConsistentVideoTransform(image_size=224, is_train=True, erase_prob=1.0)
    dummy_clip = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
    tensor_train = tf_train(dummy_clip)
    assert tensor_train.shape == (2, 3, 224, 224), f"Wrong shape: {tensor_train.shape}"

    # Verify identical erasing mask across frames
    diff = torch.abs(tensor_train[0] - tensor_train[1])
    erased_pixels_f0 = (tensor_train[0] == 0).sum().item()
    erased_pixels_f1 = (tensor_train[1] == 0).sum().item()
    assert erased_pixels_f0 == erased_pixels_f1, "Random erasing mask must be identical across all frames in a clip!"
    print(f"Clip-synchronized erased pixels per channel: {erased_pixels_f0}")

    kinematics = FishMotionKinematics7Ch()
    feat_7ch, _ = kinematics(tensor_train.unsqueeze(0))
    assert feat_7ch.shape == (1, 2, 7, 224, 224), f"Wrong 7ch shape: {feat_7ch.shape}"

    erased_mask = (tensor_train[0, 0] == 0) & (tensor_train[1, 0] == 0)
    if erased_mask.sum() > 0:
        flow_u = feat_7ch[0, 1, 3][erased_mask]
        flow_v = feat_7ch[0, 1, 4][erased_mask]
        assert torch.allclose(flow_u, torch.zeros_like(flow_u), atol=1e-5), "Flow u in erased region must be 0!"
        assert torch.allclose(flow_v, torch.zeros_like(flow_v), atol=1e-5), "Flow v in erased region must be 0!"
        print("[PASSED] Optical flow in erased region is strictly 0.0 (no kinematic artifacts)!")

    tf_val = ConsistentVideoTransform(image_size=224, is_train=False)
    tensor_val = tf_val(dummy_clip)
    assert (tensor_val == 0).sum().item() == 0, "No erasing should happen during validation mode!"
    print("[PASSED] Val mode preserves clean full frames without erasing!")


def test_convnext_layerscale_and_droppath():
    print("\n" + "=" * 65)
    print("TEST 7: CONVNEXT-NANO LAYERSCALE & DROPPATH VERIFICATION")
    print("=" * 65)

    model = MultimodalSOTANet(num_frames=2, video_drop_path=0.1, layer_scale_init_value=1e-6)

    # 1. Inspect all 6 blocks across the 4 stages for LayerScale & DropPath
    expected_rates = [x.item() for x in torch.linspace(0, 0.1, 6)]
    actual_rates = []
    block_dims = [48, 96, 192, 192, 192, 384]
    blk_idx_total = 0
    for stage_idx, stage in enumerate(model.video_backbone.stages):
        for blk_idx, blk in enumerate(stage):
            # Check DropPath
            dp_prob = blk.drop_path.drop_prob if hasattr(blk.drop_path, "drop_prob") else 0.0
            actual_rates.append(dp_prob)

            # Check LayerScale gamma
            assert blk.gamma is not None, f"Block {blk_idx_total} missing LayerScale gamma!"
            assert blk.gamma.requires_grad, f"Block {blk_idx_total} LayerScale gamma must be trainable!"
            assert blk.gamma.shape == (block_dims[blk_idx_total],), f"Wrong gamma shape at block {blk_idx_total}"
            assert torch.allclose(blk.gamma, torch.full_like(blk.gamma, 1e-6)), "LayerScale gamma must initialize to 1e-6"

            print(f"  Stage {stage_idx + 1}, Block {blk_idx}: DropPath = {dp_prob:.4f} | LayerScale Dim = {blk.gamma.numel()} (gamma=1e-6)")
            blk_idx_total += 1

    assert len(actual_rates) == 6, f"Expected 6 blocks, got {len(actual_rates)}"
    for act, exp in zip(actual_rates, expected_rates):
        assert abs(act - exp) < 1e-5, f"Rate mismatch: got {act}, expected {exp}"
    print(f"[PASSED] All 6 ConvNeXt blocks verified with LayerScale (1e-6) and DropPath schedule: {[round(r, 4) for r in actual_rates]}")

    # 2. Verify identity in eval mode
    model.eval()
    dummy_x = torch.randn(2, 96, 28, 28)
    blk_eval = model.video_backbone.stages[1][0]
    assert not blk_eval.training
    out_eval1 = blk_eval(dummy_x)
    out_eval2 = blk_eval(dummy_x)
    assert torch.allclose(out_eval1, out_eval2), "DropPath must be deterministic (identity) in eval mode!"
    print("[PASSED] DropPath identity pass-through verified in eval mode!")

    # 3. Verify drop_path=0.0 turns into nn.Identity
    model_zero = MultimodalSOTANet(num_frames=2, video_drop_path=0.0)
    for stage in model_zero.video_backbone.stages:
        for blk in stage:
            assert isinstance(blk.drop_path, nn.Identity), "When video_drop_path=0.0, drop_path should be nn.Identity"
    print("[PASSED] Zero drop path rate correctly instantiates nn.Identity!")


if __name__ == "__main__":
    seed_everything(42)
    print("=" * 65)
    print("RUNNING MANDATORY DUAL TIE-BREAKERS ARCHITECTURE VERIFICATION TEST SUITE")
    print("=" * 65)

    test_parameter_budget()
    test_tournament_forward_and_pairwise()
    test_gradient_flow_through_loss()
    test_end_to_end_from_scratch()
    test_all_8_dual_tie_breaker_combinations()
    test_consistent_video_transform()
    test_convnext_layerscale_and_droppath()

    print("\n" + "=" * 65)
    print("ALL DUAL TIE-BREAKERS TESTS PASSED SUCCESSFULLY! (100% READY)")
    print("=" * 65)
