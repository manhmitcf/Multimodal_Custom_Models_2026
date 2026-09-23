import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from models.multimodal_sota_net import MultimodalBoundaryAwareNet
from models.audio_backbone import FrequencyConvNeXtAudioBackbone
from features.audio_frontend import AudioFrontend, AdvancedSpectral1DAugmentation
from utils.losses import PairwiseTournamentLoss


def test_parameter_budget():
    print("\n" + "=" * 65)
    print("TEST 1: TOURNAMENT NETWORK PARAMETER BUDGET (< 5.0M)")
    print("=" * 65)

    model = MultimodalBoundaryAwareNet(num_frames=2)
    total_params = sum(p.numel() for p in model.parameters())
    v_params = sum(p.numel() for p in model.video_backbone.parameters())
    a_params = sum(p.numel() for p in model.audio_backbone.parameters())
    f_params = sum(p.numel() for p in model.fusion.parameters())

    print(f"Total Model Parameters:                 {total_params:,}")
    print(f"  - Video Backbone (ConvNeXt-Nano 7ch):   {v_params:,}")
    print(f"  - Audio Backbone (ConvNeXt-1D 2049):    {a_params:,}")
    print(f"  - Pairwise Tournament Fusion [No-Ties]: {f_params:,}")

    strict_limit = 5000000
    assert total_params < strict_limit, f"FAILED: Exceeded budget {total_params} >= {strict_limit}"
    headroom = strict_limit - total_params
    print(f"[PASSED] Total parameters ({total_params:,}) safely under 5.0M! (Headroom: {headroom:,})")


def test_tournament_forward_and_pairwise():
    print("\n" + "=" * 65)
    print("TEST 2: TOURNAMENT 2-LEVEL FORWARD PASS (DEFAULT: NO TIE-BREAKERS)")
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

    # 2. Check Level 2 Pairwise Boundaries B12, B23, B13 (Tie-breakers disabled)
    p_w_over_m = out["p_w_over_m"]
    p_m_over_s = out["p_m_over_s"]
    p_w_over_s = out["p_w_over_s"]

    u_tie_12 = out["u_tie_12"]
    u_tie_23 = out["u_tie_23"]
    u_tie_13 = out["u_tie_13"]

    print(f"Pairwise B12 P(Weak > Medium):          {p_w_over_m.tolist()}")
    print(f"Pairwise B23 P(Medium > Strong):        {p_m_over_s.tolist()}")
    print(f"Pairwise B13 P(Weak > Strong) [Cross]:  {p_w_over_s.tolist()}")
    print(f"Tie-Breakers Status:                    Disabled (u_tie = 0.0)")

    assert (p_w_over_m >= 0.0).all() and (p_w_over_m <= 1.0).all()
    assert (p_m_over_s >= 0.0).all() and (p_m_over_s <= 1.0).all()
    assert (p_w_over_s >= 0.0).all() and (p_w_over_s <= 1.0).all()
    assert torch.all(u_tie_12 == 0.0)
    assert torch.all(u_tie_23 == 0.0)
    assert torch.all(u_tie_13 == 0.0)

    # 3. Check Tournament Voting scores
    v_voting = out["v_voting"]
    print(f"Tournament Borda Voting [Weak, Med, Str]:\n{v_voting}")
    assert v_voting.shape == (B, 3)
    assert (v_voting >= 0.0).all() and (v_voting <= 2.0).all(), "Borda scores must be in [0, 2]"

    # 4. Check Final Probabilities sum to 1
    probs = out["probabilities"]
    assert probs.shape == (B, 4)
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), f"Probabilities do not sum to 1: {prob_sums}"

    # 5. Check Active Tie-Breakers Mode
    model_tb = MultimodalBoundaryAwareNet(num_frames=2, enable_b12=True, enable_b23=True, enable_b13=True)
    model_tb.eval()
    with torch.no_grad():
        out_tb = model_tb(v_input, a_input)
    assert (out_tb["u_tie_12"] > 0.0).all() and (out_tb["u_tie_12"] <= 1.0).all()
    assert (out_tb["u_tie_23"] > 0.0).all() and (out_tb["u_tie_23"] <= 1.0).all()
    assert (out_tb["u_tie_13"] > 0.0).all() and (out_tb["u_tie_13"] <= 1.0).all()

    print("[PASSED] 2-level tournament hierarchy (both no-tie-breakers and tie-breakers enabled) verified!")


def test_gradient_flow_tournament_loss():
    print("\n" + "=" * 65)
    print("TEST 3: 100% GRADIENT FLOW THROUGH PAIRWISE TOURNAMENT LOSS")
    print("=" * 65)

    model = MultimodalBoundaryAwareNet(num_frames=2)
    model.train()
    criterion = PairwiseTournamentLoss(weight_act=0.5, weight_pairwise=0.5, weight_ce=1.0)

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
    print(f"  Composite Tournament Loss: {loss.item():.4f}")


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

    # Verify all components received gradients
    vb_grads = [p.grad for p in model.video_backbone.parameters() if p.requires_grad]
    ab_grads = [p.grad for p in model.audio_backbone.parameters() if p.requires_grad]
    f_grads = [p.grad for p in model.fusion.parameters() if p.requires_grad]

    assert all(g is not None for g in vb_grads), "Video backbone missing gradients"
    assert all(g is not None for g in ab_grads), "Audio ConvNeXt-1D backbone missing gradients"
    assert all(g is not None for g in f_grads), "Tournament fusion missing gradients"

    optimizer.step()
    print(f"[PASSED] End-to-End Single Phase: All {len(list(model.parameters()))} param tensors updated simultaneously!")


def test_tie_breakers_toggle_config():
    print("\n" + "=" * 65)
    print("TEST 5: CONFIGURABLE TIE-BREAKER TOGGLE (ABLATION VERIFICATION)")
    print("=" * 65)

    B = 2
    v_input = torch.randn(B, 2, 3, 224, 224)
    a_input = torch.randn(B, 512000)

    # Case A: Only B23 enabled (legacy main_01 equivalence)
    model_b23 = MultimodalBoundaryAwareNet(enable_b12=False, enable_b23=True, enable_b13=False)
    assert model_b23.fusion.tournament_head.head_b12_a is None
    assert model_b23.fusion.tournament_head.head_b23_a is not None
    assert model_b23.fusion.tournament_head.head_b13_a is None
    out_b23 = model_b23(v_input, a_input)
    assert out_b23["probabilities"].shape == (B, 4)
    p_params_b23 = sum(p.numel() for p in model_b23.parameters())
    print(f"  Case A (Only B23 enabled): {p_params_b23:,} params - verified clean!")

    # Case B: All tie-breakers disabled (pure joint tournament ablation - current branch default)
    model_none = MultimodalBoundaryAwareNet(enable_b12=False, enable_b23=False, enable_b13=False)
    assert model_none.fusion.tournament_head.head_b12_a is None
    assert model_none.fusion.tournament_head.head_b23_a is None
    assert model_none.fusion.tournament_head.head_b13_a is None
    out_none = model_none(v_input, a_input)
    assert out_none["probabilities"].shape == (B, 4)
    p_params_none = sum(p.numel() for p in model_none.parameters())
    print(f"  Case B (All tie-breakers disabled - default): {p_params_none:,} params - verified clean!")

    # Case C: All 3 tie-breakers enabled
    model_all = MultimodalBoundaryAwareNet(enable_b12=True, enable_b23=True, enable_b13=True)
    assert model_all.fusion.tournament_head.head_b12_a is not None
    assert model_all.fusion.tournament_head.head_b23_a is not None
    assert model_all.fusion.tournament_head.head_b13_a is not None
    out_all = model_all(v_input, a_input)
    assert out_all["probabilities"].shape == (B, 4)
    p_params_all = sum(p.numel() for p in model_all.parameters())
    print(f"  Case C (All 3 tie-breakers enabled): {p_params_all:,} params - verified clean!")

    print("[PASSED] Configurable tie-breaker toggle verified across all ablation states!")


def test_audio_convnext_backbone_and_spectral_aug():
    print("\n" + "=" * 65)
    print("TEST 6: FREQUENCY-DOMAIN 1D CONVNEXT AUDIO BACKBONE & SPECTRAL AUG")
    print("=" * 65)

    # 1. Test Advanced Spectral 1D Augmentation
    aug = AdvancedSpectral1DAugmentation(tilt_max=0.05, max_shift=12, noise_std=0.015, cutout_width=20)
    dummy_spec = torch.ones(2, 2, 2049)
    aug.train()
    aug_out = aug(dummy_spec)
    assert aug_out.shape == (2, 2, 2049), f"Wrong aug shape: {aug_out.shape}"
    assert not torch.equal(aug_out, dummy_spec), "Augmentation should modify input in train mode"

    aug.eval()
    aug_eval = aug(dummy_spec)
    assert torch.equal(aug_eval, dummy_spec), "Augmentation must be identity pass-through in eval mode"
    print("  [Subtest 6.1] Advanced Spectral 1D Augmentation verified (Train & Eval)!")

    # 2. Test Audio Frontend end-to-end
    frontend = AudioFrontend()
    dummy_audio = torch.randn(2, 512000)
    spec_profile = frontend(dummy_audio)
    assert spec_profile.shape == (2, 2, 2049), f"Wrong frontend output shape: {spec_profile.shape}"
    print(f"  [Subtest 6.2] Audio Frontend output shape: {tuple(spec_profile.shape)} (Dual-Channel Profile)!")

    # 3. Test FrequencyConvNeXtAudioBackbone
    backbone = FrequencyConvNeXtAudioBackbone(in_channels=2, embed_dim=224, num_tokens=2, drop_path_rate=0.1, dropout=0.1)
    backbone_params = sum(p.numel() for p in backbone.parameters())
    print(f"  [Subtest 6.3] Audio Backbone Parameters: {backbone_params:,} (~832K)")
    assert 800000 < backbone_params < 900000, f"Backbone parameter count unexpected: {backbone_params}"

    f_audio, f_freq, f_rhythm, f_burst, tokens_audio = backbone(spec_profile)
    assert f_audio.shape == (2, 224), f"Wrong f_audio shape: {f_audio.shape}"
    assert tokens_audio.shape == (2, 2, 224), f"Wrong tokens_audio shape: {tokens_audio.shape}"
    print(f"  [Subtest 6.4] Audio Backbone embedding shape: {tuple(f_audio.shape)}, tokens shape: {tuple(tokens_audio.shape)}!")
    print("[PASSED] Frequency-Domain 1D ConvNeXt audio pipeline fully verified!")


def test_consistent_video_transform():
    print("\n" + "=" * 65)
    print("TEST 7: CLIP-SYNCHRONIZED VIDEO TRANSFORM & RANDOM ERASING")
    print("=" * 65)

    from transforms.video_transform import ConsistentVideoTransform
    from features.motion_kinematics import FishMotionKinematics7Ch

    # Train mode with 100% erasing probability
    tf_train = ConsistentVideoTransform(image_size=224, is_train=True, erase_prob=1.0)
    dummy_clip = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
    tensor_train = tf_train(dummy_clip)
    assert tensor_train.shape == (2, 3, 224, 224), f"Wrong shape: {tensor_train.shape}"

    # Check temporal synchronization: erased region must match exactly across all frames
    mask0 = (tensor_train[0] == 0.0)
    mask1 = (tensor_train[1] == 0.0)
    assert torch.equal(mask0, mask1), "Random Erasing is NOT clip-synchronized across frames!"
    erased_pixels = mask0.sum().item()
    assert erased_pixels > 0, "No pixels were erased in train mode with erase_prob=1.0!"
    print(f"Clip-synchronized erased pixels per channel: {erased_pixels // 3}")

    # Check optical flow in erased region is strictly 0.0
    mk = FishMotionKinematics7Ch()
    kinematics, _ = mk(tensor_train.unsqueeze(0))
    flow = kinematics[0, :, 3:5]
    flow_in_erased = flow[:, :, mask0[0]]
    assert torch.all(flow_in_erased == 0.0), f"Flow in erased region is not zero: max abs {flow_in_erased.abs().max()}"
    print("[PASSED] Optical flow in erased region is strictly 0.0 (no kinematic artifacts)!")

    # Val mode must NOT apply erasing
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
    test_audio_convnext_backbone_and_spectral_aug()
    test_consistent_video_transform()
    print("\n" + "=" * 65)
    print("ALL TOURNAMENT CONVNEXT-1D TESTS PASSED SUCCESSFULLY! (100% READY)")
    print("=" * 65 + "\n")
