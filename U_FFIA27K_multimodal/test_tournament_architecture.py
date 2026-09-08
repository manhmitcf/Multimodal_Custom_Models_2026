import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.multimodal_sota_net import MultimodalBoundaryAwareNet
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

    print(f"Total Model Parameters:               {total_params:,}")
    print(f"  - Video Backbone (ConvNeXt-Nano 7ch): {v_params:,}")
    print(f"  - Audio Backbone (STFT-MLP 2049):     {a_params:,}")
    print(f"  - Pairwise Tournament Fusion:         {f_params:,}")

    strict_limit = 5000000
    assert total_params < strict_limit, f"FAILED: Exceeded budget {total_params} >= {strict_limit}"
    headroom = strict_limit - total_params
    print(f"[PASSED] Total parameters ({total_params:,}) safely under 5.0M! (Headroom: {headroom:,})")


def test_tournament_forward_and_pairwise():
    print("\n" + "=" * 65)
    print("TEST 2: TOURNAMENT 2-LEVEL FORWARD PASS & PAIRWISE CROSS BOUNDARIES")
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

    # 2. Check Level 2 Pairwise Boundaries B12, B23, B13
    p_w_over_m = out["p_w_over_m"]
    p_m_over_s = out["p_m_over_s"]
    p_w_over_s = out["p_w_over_s"]

    print(f"Pairwise B12 P(Weak > Medium):          {p_w_over_m.tolist()}")
    print(f"Pairwise B23 P(Medium > Strong):        {p_m_over_s.tolist()}")
    print(f"Pairwise B13 P(Weak > Strong) [Cross]:  {p_w_over_s.tolist()}")

    assert (p_w_over_m >= 0.0).all() and (p_w_over_m <= 1.0).all()
    assert (p_m_over_s >= 0.0).all() and (p_m_over_s <= 1.0).all()
    assert (p_w_over_s >= 0.0).all() and (p_w_over_s <= 1.0).all()

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

    print("[PASSED] 2-level tournament hierarchy, pairwise cross boundaries, and Borda voting verified!")


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


def test_phase1_warmup():
    print("\n" + "=" * 65)
    print("TEST 4: PHASE 1 BACKBONE WARMUP (FUSION FROZEN, SEPARATE GRADIENTS)")
    print("=" * 65)

    model = MultimodalBoundaryAwareNet(num_frames=2)
    model.train()

    # Freeze fusion in Phase 1
    for p in model.fusion.parameters():
        p.requires_grad = False

    B = 4
    v_input = torch.randn(B, 2, 3, 224, 224)
    a_input = torch.randn(B, 512000)
    y_raw = torch.tensor([0, 1, 2, 3])

    # Unimodal Video step
    outputs = model(v_input, a_input)
    loss_v = F.cross_entropy(outputs['logits_video'], y_raw)
    loss_v.backward(retain_graph=True)

    vb_grads = [p.grad for p in model.video_backbone.parameters()]
    assert all(g is not None for g in vb_grads), "Video backbone missing gradients in Phase 1"
    f_grads = [p.grad for p in model.fusion.parameters()]
    assert all(g is None for g in f_grads), "Fusion received gradients during Phase 1 warmup (should be frozen)!"

    # Unimodal Audio step
    loss_a = F.cross_entropy(outputs['logits_audio'], y_raw)
    loss_a.backward()

    ab_grads = [p.grad for p in model.audio_backbone.parameters()]
    assert all(g is not None for g in ab_grads), "Audio MLP backbone missing gradients in Phase 1"

    print("[PASSED] Phase 1 Warmup: Video & Audio backbones trained independently, Fusion safely FROZEN!")


def test_phase2_unfreeze_last_stages():
    print("\n" + "=" * 65)
    print("TEST 5: PHASE 2 UNFREEZE LAST STAGES & TOURNAMENT FUSION")
    print("=" * 65)

    model = MultimodalBoundaryAwareNet(num_frames=2)
    model.train()

    # Unfreeze fusion
    for p in model.fusion.parameters():
        p.requires_grad = True

    # Video: freeze stem, stage0, stage1; unfreeze stage2, stage3, proj
    vb = model.video_backbone
    for p in vb.stem.parameters():
        p.requires_grad = False
    for p in vb.downsample_layers[0].parameters():
        p.requires_grad = False
    for p in vb.stages[0].parameters():
        p.requires_grad = False
    for p in vb.stages[1].parameters():
        p.requires_grad = False
    for p in vb.stages[2:].parameters():
        p.requires_grad = True
    for p in vb.downsample_layers[1:].parameters():
        p.requires_grad = True
    for p in vb.norm_final.parameters():
        p.requires_grad = True
    for p in vb.proj.parameters():
        p.requires_grad = True

    # Audio MLP: freeze fc1, ln1; unfreeze fc2, ln2
    ab = model.audio_backbone
    for p in ab.fc1.parameters():
        p.requires_grad = False
    for p in ab.ln1.parameters():
        p.requires_grad = False
    for p in ab.fc2.parameters():
        p.requires_grad = True
    for p in ab.ln2.parameters():
        p.requires_grad = True

    # Aux heads frozen
    for p in model.aux_head_video.parameters():
        p.requires_grad = False
    for p in model.aux_head_audio.parameters():
        p.requires_grad = False

    criterion = PairwiseTournamentLoss(weight_act=0.5, weight_pairwise=0.5, weight_ce=1.0)

    B = 4
    v_input = torch.randn(B, 2, 3, 224, 224)
    a_input = torch.randn(B, 512000)
    targets = {"target": torch.tensor([0, 1, 2, 3])}

    outputs = model(v_input, a_input)
    loss = criterion(outputs, targets)
    loss.backward()

    # Verify frozen parameters have NO grad
    assert vb.stem[0].weight.grad is None, "Video stem should be frozen in Phase 2!"
    assert ab.fc1.weight.grad is None, "Audio fc1 should be frozen in Phase 2!"

    # Verify unfrozen parameters HAVE healthy grad
    assert vb.proj[0].weight.grad is not None, "Video proj should have grad in Phase 2!"
    assert ab.fc2.weight.grad is not None, "Audio fc2 should have grad in Phase 2!"
    assert model.fusion.tournament_head.head_b12[0].weight.grad is not None, "Tournament fusion should have grad in Phase 2!"

    print("[PASSED] Phase 2: Early stages & fc1 safely FROZEN; Late stages, fc2 & Tournament Fusion actively learning!")


if __name__ == "__main__":
    test_parameter_budget()
    test_tournament_forward_and_pairwise()
    test_gradient_flow_tournament_loss()
    test_phase1_warmup()
    test_phase2_unfreeze_last_stages()
    print("\n" + "=" * 65)
    print("ALL TOURNAMENT STFT-MLP 2-PHASE TESTS PASSED! (100% READY)")
    print("=" * 65 + "\n")
