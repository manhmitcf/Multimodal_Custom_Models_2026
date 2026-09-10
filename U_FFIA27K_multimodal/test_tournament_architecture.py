import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
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

    print(f"Total Model Parameters:                 {total_params:,}")
    print(f"  - Video Backbone (ConvNeXt-Nano 7ch):   {v_params:,}")
    print(f"  - Audio Backbone (Dual-Axis 1D Conv):   {a_params:,}")
    print(f"  - Pairwise Tournament Fusion:           {f_params:,}")

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
    assert all(g is not None for g in ab_grads), "Audio backbone missing gradients"
    assert all(g is not None for g in f_grads), "Tournament fusion missing gradients"

    optimizer.step()
    print(f"[PASSED] End-to-End Single Phase: All {len(list(model.parameters()))} param tensors updated simultaneously!")


def test_dual_axis_convnet_engine():
    print("\n" + "=" * 65)
    print("TEST 5: FACTORIZED DUAL-AXIS 1D CONVNET AUDIO ENGINE")
    print("=" * 65)

    from features.audio_frontend import AudioFrontend, AudioFrontendOutput
    from models.audio_backbone import FactorizedDualAxisAudioBackbone, SpectralAxis1DConvEngine, TemporalAxis1DConvEngine

    frontend = AudioFrontend()
    backbone = FactorizedDualAxisAudioBackbone(embed_dim=224, num_tokens=2)

    B = 2
    raw_audio = torch.randn(B, 512000)  # 2.0s @ 256 kHz

    out_frontend = frontend(raw_audio)
    assert isinstance(out_frontend, AudioFrontendOutput), "Frontend must return AudioFrontendOutput"
    assert out_frontend.spectrogram.shape == (B, 1, 251, 2049), f"Expected spectrogram [B, 1, 251, 2049], got {out_frontend.spectrogram.shape}"
    assert out_frontend.spec_vector.shape == (B, 2049), f"Expected spec_vector [B, 2049], got {out_frontend.spec_vector.shape}"

    print(f"Frontend outputs:")
    print(f"  - spectrogram shape:    {list(out_frontend.spectrogram.shape)}")
    print(f"  - spec_vector shape:    {list(out_frontend.spec_vector.shape)}")

    f_audio, f_freq, f_rhythm, f_burst_a, tokens_audio = backbone(out_frontend)

    assert f_audio.shape == (B, 224), f"Expected f_audio [B, 224], got {f_audio.shape}"
    assert f_freq.shape == (B, 224), f"Expected f_freq [B, 224], got {f_freq.shape}"
    assert f_rhythm.shape == (B, 224), f"Expected f_rhythm [B, 224], got {f_rhythm.shape}"
    assert f_burst_a.shape == (B, 224), f"Expected f_burst_a [B, 224], got {f_burst_a.shape}"
    assert tokens_audio.shape == (B, 2, 224), f"Expected tokens_audio [B, 2, 224], got {tokens_audio.shape}"

    # Test backward pass on audio backbone alone
    loss = (f_audio.sum() + f_freq.sum() + f_rhythm.sum() + f_burst_a.sum())
    loss.backward()

    for name, param in backbone.named_parameters():
        assert param.grad is not None, f"Backbone param {name} did not receive gradient"

    print(f"[PASSED] Factorized Dual-Axis 1D ConvNet forward & backward verified!")


if __name__ == "__main__":
    test_parameter_budget()
    test_tournament_forward_and_pairwise()
    test_gradient_flow_tournament_loss()
    test_end_to_end_from_scratch()
    test_dual_axis_convnet_engine()
    print("\n" + "=" * 65)
    print("ALL FACTORIZED DUAL-AXIS CONV1D TOURNAMENT TESTS PASSED! (100% READY)")
    print("=" * 65 + "\n")
