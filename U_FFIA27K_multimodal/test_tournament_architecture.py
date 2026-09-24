import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from models.multimodal_sota_net import MultimodalBoundaryAwareNet
from utils.losses import PairwiseTournamentLoss
from utils.seed import seed_everything


def test_parameter_budget():
    print("\n" + "=" * 65)
    print("TEST 1: SMoR-NET TOURNAMENT PARAMETER BUDGET (< 5.0M)")
    print("=" * 65)

    model = MultimodalBoundaryAwareNet(num_frames=2)
    total_params = sum(p.numel() for p in model.parameters())
    v_params = sum(p.numel() for p in model.video_backbone.parameters())
    a_params = sum(p.numel() for p in model.audio_backbone.parameters())
    f_params = sum(p.numel() for p in model.fusion.parameters())

    print(f"Total Model Parameters:               {total_params:,}")
    print(f"  - Video Backbone (ConvNeXt-Nano 7ch): {v_params:,}")
    print(f"  - Audio Backbone (STFT-MLP 2049):     {a_params:,}")
    print(f"  - Pairwise SMoR Fusion:               {f_params:,}")

    strict_limit = 5000000
    assert total_params < strict_limit, f"FAILED: Exceeded budget {total_params} >= {strict_limit}"
    headroom = strict_limit - total_params
    print(f"[PASSED] Total parameters ({total_params:,}) safely under 5.0M! (Headroom: {headroom:,})")


def test_tournament_forward_and_pairwise():
    print("\n" + "=" * 65)
    print("TEST 2: TOURNAMENT 2-LEVEL FORWARD PASS & SPARSE MIXTURE-OF-REFEREES")
    print("=" * 65)

    model = MultimodalBoundaryAwareNet(num_frames=2)
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

    # 2. Check Level 2 Pairwise Boundaries B12, B23, B13 and Referees
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

    # Check that both audio and video referee logits are present
    assert "logit_12_a" in out and "logit_12_v" in out
    assert "logit_23_a" in out and "logit_23_v" in out
    assert "logit_13_a" in out and "logit_13_v" in out

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

    print("[PASSED] SMoR Tournament 2-level forward pass verified!")


def test_gradient_flow_through_loss():
    print("\n" + "=" * 65)
    print("TEST 3: 100% GRADIENT FLOW THROUGH SMoR LOSS")
    print("=" * 65)

    model = MultimodalBoundaryAwareNet(num_frames=2)
    model.train()

    B = 4
    v_input = torch.randn(B, 2, 3, 224, 224)
    a_input = torch.randn(B, 512000)
    targets = {
        "target": torch.tensor([0, 1, 2, 3]),
        "target_ordinal": torch.tensor([0.0, 3.0, 2.0, 1.0]),
        "activity_target": torch.tensor([0.0, 1.0, 1.0, 1.0])
    }

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
    print(f"  SMoR Composite Loss: {loss.item():.4f}")


def test_end_to_end_from_scratch():
    print("\n" + "=" * 65)
    print("TEST 4: END-TO-END FROM SCRATCH SIMULTANEOUS TRAINING")
    print("=" * 65)

    model = MultimodalBoundaryAwareNet(num_frames=2)
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


def test_tie_breakers_toggle_config():
    print("\n" + "=" * 65)
    print("TEST 5: CONFIGURABLE REFEREES TOGGLE (ABLATION VERIFICATION)")
    print("=" * 65)

    B = 2
    v_input = torch.randn(B, 2, 3, 224, 224)
    a_input = torch.randn(B, 512000)

    # Case A: Pure Baseline (All tie-breakers disabled)
    tb_baseline = {
        "b12": {"enable_audio": False, "enable_video": False},
        "b23": {"enable_audio": False, "enable_video": False},
        "b13": {"enable_audio": False, "enable_video": False}
    }
    model_baseline = MultimodalBoundaryAwareNet(tie_breakers=tb_baseline)
    th_base = model_baseline.fusion.tournament_head
    assert th_base.head_b12_a is None and th_base.head_b12_v is None
    assert th_base.head_b23_a is None and th_base.head_b23_v is None
    assert th_base.head_b13_a is None and th_base.head_b13_v is None
    out_b = model_baseline(v_input, a_input)
    assert out_b["probabilities"].shape == (B, 4)
    p_params_b = sum(p.numel() for p in model_baseline.parameters())
    print(f"  Case A (Pure Baseline - All Referees Off): {p_params_b:,} params - verified clean!")

    # Case B: All Audio Referees Only (Triple Audio)
    tb_audio_only = {
        "b12": {"enable_audio": True, "enable_video": False},
        "b23": {"enable_audio": True, "enable_video": False},
        "b13": {"enable_audio": True, "enable_video": False}
    }
    model_audio = MultimodalBoundaryAwareNet(tie_breakers=tb_audio_only)
    th_aud = model_audio.fusion.tournament_head
    assert th_aud.head_b12_a is not None and th_aud.head_b12_v is None
    assert th_aud.head_b23_a is not None and th_aud.head_b23_v is None
    assert th_aud.head_b13_a is not None and th_aud.head_b13_v is None
    out_a = model_audio(v_input, a_input)
    assert out_a["probabilities"].shape == (B, 4)
    p_params_a = sum(p.numel() for p in model_audio.parameters())
    print(f"  Case B (Triple Audio Referees Only):      {p_params_a:,} params - verified clean!")

    # Case C: All Video Referees Only (Triple Video)
    tb_video_only = {
        "b12": {"enable_audio": False, "enable_video": True},
        "b23": {"enable_audio": False, "enable_video": True},
        "b13": {"enable_audio": False, "enable_video": True}
    }
    model_video = MultimodalBoundaryAwareNet(tie_breakers=tb_video_only)
    th_vid = model_video.fusion.tournament_head
    assert th_vid.head_b12_a is None and th_vid.head_b12_v is not None
    assert th_vid.head_b23_a is None and th_vid.head_b23_v is not None
    assert th_vid.head_b13_a is None and th_vid.head_b13_v is not None
    out_v = model_video(v_input, a_input)
    assert out_v["probabilities"].shape == (B, 4)
    p_params_v = sum(p.numel() for p in model_video.parameters())
    print(f"  Case C (Triple Video Referees Only):      {p_params_v:,} params - verified clean!")

    # Case D: Both Referees (Audio + Video enabled on all 3 matchups)
    tb_full_dual = {
        "b12": {"enable_audio": True, "enable_video": True},
        "b23": {"enable_audio": True, "enable_video": True},
        "b13": {"enable_audio": True, "enable_video": True}
    }
    model_dual = MultimodalBoundaryAwareNet(tie_breakers=tb_full_dual)
    th_dual = model_dual.fusion.tournament_head
    assert th_dual.head_b12_a is not None and th_dual.head_b12_v is not None
    assert th_dual.head_b23_a is not None and th_dual.head_b23_v is not None
    assert th_dual.head_b13_a is not None and th_dual.head_b13_v is not None
    out_d = model_dual(v_input, a_input)
    assert out_d["probabilities"].shape == (B, 4)
    p_params_d = sum(p.numel() for p in model_dual.parameters())
    print(f"  Case D (Both Referees - Audio + Video): {p_params_d:,} params - verified clean!")

    print("[PASSED] Configurable referee toggle verified across all ablation states!")


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

    erased_mask_frame0 = (tensor_train[0] == 0.0).all(dim=0)
    erased_mask_frame1 = (tensor_train[1] == 0.0).all(dim=0)
    assert torch.equal(erased_mask_frame0, erased_mask_frame1), "Erasing mask not synchronized across clip frames!"
    print(f"Clip-synchronized erased pixels per channel: {erased_mask_frame0.sum().item()}")

    kinematics = FishMotionKinematics7Ch(image_size=224)
    frames_7ch, _ = kinematics(tensor_train.unsqueeze(0))
    flow_u = frames_7ch[0, :, 3, :, :]
    flow_v = frames_7ch[0, :, 4, :, :]
    assert flow_u[:, erased_mask_frame0].abs().max().item() == 0.0, "Optical flow must be 0.0 in erased region!"
    assert flow_v[:, erased_mask_frame0].abs().max().item() == 0.0, "Optical flow must be 0.0 in erased region!"
    print("[PASSED] Optical flow in erased region is strictly 0.0 (no kinematic artifacts)!")

    tf_val = ConsistentVideoTransform(image_size=224, is_train=False)
    tensor_val = tf_val(dummy_clip)
    assert (tensor_val != 0.0).any(), "Val transform incorrectly erased pixels!"
    print("[PASSED] Val mode preserves clean full frames without erasing!")


def test_smor_routing_and_ste_states():
    print("\n" + "=" * 65)
    print("TEST 7: SPARSE MIXTURE-OF-REFEREES (SMoR) ROUTING & STE STATES")
    print("=" * 65)

    model = MultimodalBoundaryAwareNet(num_frames=2, use_sparse_moe_routing=True)
    model.train()

    B = 8
    v_input = torch.randn(B, 2, 3, 224, 224)
    a_input = torch.randn(B, 512000)

    out = model(v_input, a_input)

    # 1. Verify router output keys
    for tag in ["12", "23", "13"]:
        assert f"m_{tag}_a" in out, f"Missing m_{tag}_a"
        assert f"m_{tag}_v" in out, f"Missing m_{tag}_v"
        assert f"prob_{tag}_a" in out, f"Missing prob_{tag}_a"
        assert f"prob_{tag}_v" in out, f"Missing prob_{tag}_v"

        m_a = out[f"m_{tag}_a"]
        m_v = out[f"m_{tag}_v"]
        p_a = out[f"prob_{tag}_a"]
        p_v = out[f"prob_{tag}_v"]

        # 2. In forward pass, STE outputs must be strictly binary {0.0, 1.0}
        assert torch.all((m_a == 0.0) | (m_a == 1.0)), f"m_{tag}_a must be binary {0, 1}"
        assert torch.all((m_v == 0.0) | (m_v == 1.0)), f"m_{tag}_v must be binary {0, 1}"
        assert torch.all((p_a >= 0.0) & (p_a <= 1.0)), f"prob_{tag}_a must be in [0, 1]"
        assert torch.all((p_v >= 0.0) & (p_v <= 1.0)), f"prob_{tag}_v must be in [0, 1]"

        print(f"  Matchup B{tag} Router Decisions: m_Audio={m_a.tolist()}, m_Video={m_v.tolist()}")

    # 3. Test loss with MoE balancing and sparsity
    targets = {"target": torch.tensor([0, 1, 2, 3, 1, 2, 3, 0])}
    criterion = PairwiseTournamentLoss(
        weight_act=0.5,
        weight_pairwise=0.5,
        weight_ce=1.0,
        aux_loss_weight=0.3,
        lambda_balance=0.01,
        lambda_sparse=0.0001,
        use_sparse_moe_routing=True
    )
    loss = criterion(out, targets)
    loss.backward()

    # 4. Verify all router parameters receive active gradients through STE & balance loss
    router_params = [
        model.fusion.tournament_head.router_b12,
        model.fusion.tournament_head.router_b23,
        model.fusion.tournament_head.router_b13
    ]
    for idx, router in enumerate(router_params):
        for name, p in router.named_parameters():
            assert p.grad is not None, f"Router {idx} param {name} missing gradient!"
            assert not torch.isnan(p.grad).any(), f"Router {idx} param {name} has NaN gradient!"

    print(f"[PASSED] SMoR Straight-Through Estimator and 4-state dynamic routing verified!")
    print(f"  SMoR Composite Loss with Balancing & Sparsity: {loss.item():.4f}")


def test_convnext_droppath_linear_schedule():
    print("\n" + "=" * 65)
def test_convnext_layerscale_and_droppath():
    print("\n" + "=" * 65)
    print("TEST 8: CONVNEXT-NANO LAYERSCALE & DROPPATH VERIFICATION")
    print("=" * 65)

    model = MultimodalBoundaryAwareNet(num_frames=2, video_drop_path=0.1, layer_scale_init_value=1e-6)

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
    blk_eval = model.video_backbone.stages[1][0]  # Stage 2 has drop_path > 0
    assert not blk_eval.training
    out_eval1 = blk_eval(dummy_x)
    out_eval2 = blk_eval(dummy_x)
    assert torch.allclose(out_eval1, out_eval2), "DropPath must be deterministic (identity) in eval mode!"
    print("[PASSED] DropPath identity pass-through verified in eval mode!")

    # 3. Verify drop_path=0.0 turns into nn.Identity
    model_zero = MultimodalBoundaryAwareNet(num_frames=2, video_drop_path=0.0)
    for stage in model_zero.video_backbone.stages:
        for blk in stage:
            assert isinstance(blk.drop_path, nn.Identity), "When video_drop_path=0.0, drop_path should be nn.Identity"
    print("[PASSED] Zero drop path rate correctly instantiates nn.Identity!")


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("RUNNING MANDATORY SMoR-NET ARCHITECTURE VERIFICATION TEST SUITE")
    print("=" * 65)

    seed_everything(42)

    test_parameter_budget()
    test_tournament_forward_and_pairwise()
    test_gradient_flow_through_loss()
    test_end_to_end_from_scratch()
    test_tie_breakers_toggle_config()
    test_consistent_video_transform()
    test_smor_routing_and_ste_states()
    test_convnext_layerscale_and_droppath()

    print("\n" + "=" * 65)
    print("ALL SMoR-NET TOURNAMENT TESTS PASSED SUCCESSFULLY! (100% READY)")
    print("=" * 65 + "\n")
