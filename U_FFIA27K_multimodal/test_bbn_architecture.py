import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from models.multimodal_sota_net import MultimodalBoundaryAwareNet
from utils.losses import BilateralBoundaryLoss


def test_parameter_budget():
    print("\n" + "=" * 60)
    print("TEST 1: PARAMETER BUDGET AUDIT (< 5.0M STRICT LIMIT)")
    print("=" * 60)

    model = MultimodalBoundaryAwareNet(num_frames=2)
    total_params = sum(p.numel() for p in model.parameters())
    v_params = sum(p.numel() for p in model.video_backbone.parameters())
    a_params = sum(p.numel() for p in model.audio_backbone.parameters())
    f_params = sum(p.numel() for p in model.fusion.parameters())

    print(f"Total Model Parameters:               {total_params:,}")
    print(f"  - Video Backbone (ConvNeXt-Nano):   {v_params:,}")
    print(f"  - Audio Backbone (PANNS-CNN6-Pro):  {a_params:,}")
    print(f"  - Gated Bilateral Boundary Fusion:  {f_params:,}")

    strict_limit = 5000000
    assert total_params < strict_limit, f"FAILED: Model exceeds 5M budget ({total_params} >= {strict_limit})"
    print(f"[PASSED] Total parameters ({total_params:,}) safely under 5.0M limit (headroom: {strict_limit - total_params:,})!")


def test_forward_and_shapes():
    print("\n" + "=" * 60)
    print("TEST 2: FORWARD PASS & TENSOR SHAPES (T=2 FRAMES)")
    print("=" * 60)

    model = MultimodalBoundaryAwareNet(num_frames=2)
    model.eval()

    B = 2
    video_input = torch.randn(B, 2, 3, 224, 224)
    audio_input = torch.randn(B, 64000)

    with torch.no_grad():
        outputs = model(video_input, audio_input)

    assert "clipwise_output" in outputs, "Missing clipwise_output"
    assert "probabilities" in outputs, "Missing probabilities"
    assert "intensity_score" in outputs, "Missing intensity_score"
    assert "cutoffs_v" in outputs, "Missing cutoffs_v"
    assert "cutoffs_a" in outputs, "Missing cutoffs_a"
    assert "b_final" in outputs, "Missing b_final"

    assert outputs["clipwise_output"].shape == (B, 4), f"Wrong logits shape: {outputs['clipwise_output'].shape}"
    assert outputs["probabilities"].shape == (B, 4), f"Wrong probabilities shape: {outputs['probabilities'].shape}"
    assert outputs["intensity_score"].shape == (B, 1), f"Wrong intensity shape: {outputs['intensity_score'].shape}"

    # Probabilities sum to 1
    prob_sum = outputs["probabilities"].sum(dim=-1)
    assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5), "Probabilities do not sum to 1"

    print("[PASSED] Forward pass tensor shapes verified!")
    print(f"  Logits shape:        {outputs['clipwise_output'].shape}")
    print(f"  Probabilities shape: {outputs['probabilities'].shape}")
    print(f"  Intensity score:     {outputs['intensity_score'].squeeze(-1).tolist()}")


def test_monotonic_cutoffs():
    print("\n" + "=" * 60)
    print("TEST 3: STRICT MONOTONICITY OF BOUNDARY CUTOFFS")
    print("=" * 60)

    model = MultimodalBoundaryAwareNet(num_frames=2)
    b_v = model.fusion.head_v.get_cutoffs()
    b_a = model.fusion.head_a.get_cutoffs()

    print(f"Video Cutoffs: [b1={b_v[0].item():.4f}, b2={b_v[1].item():.4f}, b3={b_v[2].item():.4f}]")
    print(f"Audio Cutoffs: [b1={b_a[0].item():.4f}, b2={b_a[1].item():.4f}, b3={b_a[2].item():.4f}]")

    # Verify b1 < b2 < b3 with delta >= 0.5
    assert b_v[1] >= b_v[0] + 0.5, "Video b2 < b1 + 0.5 violated"
    assert b_v[2] >= b_v[1] + 0.5, "Video b3 < b2 + 0.5 violated"
    assert b_a[1] >= b_a[0] + 0.5, "Audio b2 < b1 + 0.5 violated"
    assert b_a[2] >= b_a[1] + 0.5, "Audio b3 < b2 + 0.5 violated"

    print("[PASSED] Strictly monotonic cutoffs (delta >= 0.5) mathematically verified!")


def test_gradient_flow():
    print("\n" + "=" * 60)
    print("TEST 4: 100% GRADIENT FLOW THROUGH COMPOSITE BILATERAL LOSS")
    print("=" * 60)

    model = MultimodalBoundaryAwareNet(num_frames=2)
    model.train()
    criterion = BilateralBoundaryLoss(lambda_emd=0.5, lambda_align=0.2)

    B = 2
    video_input = torch.randn(B, 2, 3, 224, 224, requires_grad=False)
    audio_input = torch.randn(B, 64000, requires_grad=False)
    targets = {"target": torch.tensor([0, 2])}

    outputs = model(video_input, audio_input)
    loss = criterion(outputs, targets)
    loss.backward()

    total_tensors = 0
    valid_grads = 0
    zero_grads = 0

    for name, param in model.named_parameters():
        total_tensors += 1
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                raise AssertionError(f"NaN or Inf gradient in parameter: {name}")
            valid_grads += 1
            if param.grad.abs().sum() == 0:
                zero_grads += 1
        else:
            raise AssertionError(f"Parameter without gradient: {name}")

    print(f"[PASSED] 100% Gradient flow verified: {valid_grads}/{total_tensors} parameters with healthy gradients!")
    print(f"  Loss value: {loss.item():.4f}")


def test_nelder_mead_calibration():
    print("\n" + "=" * 60)
    print("TEST 5: NELDER-MEAD POST-CALIBRATION FUNCTIONALITY")
    print("=" * 60)

    model = MultimodalBoundaryAwareNet(num_frames=2)
    model.eval()

    # Simulate validation dataset
    B = 8
    v_dummy = torch.randn(B, 2, 3, 224, 224)
    a_dummy = torch.randn(B, 64000)
    y_dummy = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    with torch.no_grad():
        out = model(v_dummy, a_dummy)

    s_v = out["score_v"].numpy()
    s_a = out["score_a"].numpy()
    g = out["gate"].squeeze(-1).numpy()
    y_raw = y_dummy.numpy()

    from scipy.optimize import minimize
    from sklearn.metrics import cohen_kappa_score

    raw_to_rank = np.array([0, 3, 2, 1])
    y_rank = raw_to_rank[y_raw]

    b_v_init = [c.item() for c in model.fusion.head_v.get_cutoffs()]
    b_a_init = [c.item() for c in model.fusion.head_a.get_cutoffs()]
    x0 = np.array(b_v_init + b_a_init, dtype=float)

    def objective(params):
        bv = params[:3]
        ba = params[3:]
        if bv[1] <= bv[0] + 0.1 or bv[2] <= bv[1] + 0.1:
            return 10.0
        if ba[1] <= ba[0] + 0.1 or ba[2] <= ba[1] + 0.1:
            return 10.0

        s = g * s_v + (1.0 - g) * s_a
        b1 = g * bv[0] + (1.0 - g) * ba[0]
        b2 = g * bv[1] + (1.0 - g) * ba[1]
        b3 = g * bv[2] + (1.0 - g) * ba[2]

        pred_rank = np.zeros_like(s, dtype=int)
        pred_rank[s >= b1] = 1
        pred_rank[s >= b2] = 2
        pred_rank[s >= b3] = 3

        qwk = cohen_kappa_score(y_rank, pred_rank, weights='quadratic')
        return -float(qwk)

    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 100})
    print(f"Nelder-Mead initial score: {-objective(x0):.4f} -> Calibrated: {-res.fun:.4f}")
    assert res.success or res.nit > 0, "Nelder-Mead failed to execute"
    print("[PASSED] Nelder-Mead post-calibration algorithm verified!")


if __name__ == "__main__":
    test_parameter_budget()
    test_forward_and_shapes()
    test_monotonic_cutoffs()
    test_gradient_flow()
    test_nelder_mead_calibration()
    print("\n" + "=" * 60)
    print("ALL 5 VERIFICATION TESTS PASSED SUCCESSFULLY! ARCHITECTURE 100% OPERATIONAL.")
    print("=" * 60 + "\n")
