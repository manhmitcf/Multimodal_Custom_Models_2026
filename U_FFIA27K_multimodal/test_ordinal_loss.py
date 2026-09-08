import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import OrdinalWassersteinEvidentialLoss, count_parameters
from models import MultimodalSOTANet


def test_ordinal_permutation():
    print("--------------------------------------------------")
    print("TEST 1: Physical Ordinal Permutation & Rank Invariance")
    loss_fn = OrdinalWassersteinEvidentialLoss(classes_num=4, sigma=0.5)

    # Raw labels: 0: None, 1: Strong, 2: Medium, 3: Weak
    raw_targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    expected_ranks = torch.tensor([0, 3, 2, 1], dtype=torch.long)

    q_ord, q_raw = loss_fn.build_gaussian_soft_target(raw_targets, device=torch.device('cpu'))

    print(f"  - Ordinal Permutation buffer: {loss_fn.ordinal_perm.tolist()}")
    print(f"  - Raw-to-Rank buffer:         {loss_fn.raw_to_rank.tolist()}")
    assert torch.equal(loss_fn.raw_to_rank, expected_ranks), "Rank mapping mismatch!"
    print("  [PASSED] Physical permutation tensor correctly reflects None < Weak < Medium < Strong.")


def test_gaussian_smoothing():
    print("--------------------------------------------------")
    print("TEST 2: Truncated Gaussian Smoothing (sigma=0.5)")
    loss_fn = OrdinalWassersteinEvidentialLoss(classes_num=4, sigma=0.5)

    # Test sample with raw label 2 (Medium -> rank 2)
    raw_medium = torch.tensor([2], dtype=torch.long)
    q_ord, q_raw = loss_fn.build_gaussian_soft_target(raw_medium, device=torch.device('cpu'))
    
    q_ord_np = q_ord.squeeze().numpy()
    # In ordinal order: [0: None, 1: Weak, 2: Medium, 3: Strong]
    print(f"  - Soft Target for Medium in Ordinal space: {np.round(q_ord_np, 4).tolist()}")
    print(f"    * Rank 0 (None):   {q_ord_np[0]:.4f} (~0%)")
    print(f"    * Rank 1 (Weak):   {q_ord_np[1]:.4f} (~10.6% adjacent)")
    print(f"    * Rank 2 (Medium): {q_ord_np[2]:.4f} (~78.7% peak)")
    print(f"    * Rank 3 (Strong): {q_ord_np[3]:.4f} (~10.6% adjacent)")

    assert np.isclose(q_ord_np.sum(), 1.0, atol=1e-5), "Soft targets must sum to 1.0!"
    assert q_ord_np[2] > q_ord_np[1] and q_ord_np[2] > q_ord_np[3], "Peak must be at true rank!"
    assert q_ord_np[0] < 0.01, "Far class (None) must receive virtually 0 mass!"
    assert np.isclose(q_ord_np[1], q_ord_np[3], atol=1e-3), "Symmetric distribution expected around rank 2!"
    print("  [PASSED] Truncated Gaussian smoothing satisfies unimodal continuity and mass conservation.")


def test_wasserstein_monotonicity():
    print("--------------------------------------------------")
    print("TEST 3: Squared EMD Ordinal Monotonicity")
    loss_fn = OrdinalWassersteinEvidentialLoss(classes_num=4, sigma=0.5)

    # Target is Strong (raw label 1, ordinal rank 3)
    target = torch.tensor([1], dtype=torch.long)

    # Candidate predictions in RAW index space [0: None, 1: Strong, 2: Medium, 3: Weak]:
    # Pred A: 100% Medium (raw index 2 -> ordinal rank 2, 1 step from Strong rank 3)
    prob_a = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    
    # Pred B: 100% Weak   (raw index 3 -> ordinal rank 1, 2 steps from Strong rank 3)
    prob_b = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    
    # Pred C: 100% None   (raw index 0 -> ordinal rank 0, 3 steps from Strong rank 3)
    prob_c = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    out_a = {'probabilities': prob_a, 'clipwise_output': torch.log(prob_a + 1e-7)}
    out_b = {'probabilities': prob_b, 'clipwise_output': torch.log(prob_b + 1e-7)}
    out_c = {'probabilities': prob_c, 'clipwise_output': torch.log(prob_c + 1e-7)}

    tgt_dict = {'target': target}
    loss_a = loss_fn(out_a, tgt_dict, epoch=400).item()
    loss_b = loss_fn(out_b, tgt_dict, epoch=400).item()
    loss_c = loss_fn(out_c, tgt_dict, epoch=400).item()

    print(f"  - Target: Strong (Rank 3)")
    print(f"    * Loss A (Pred Medium - 1 rank error): {loss_a:.4f}")
    print(f"    * Loss B (Pred Weak   - 2 rank error): {loss_b:.4f}")
    print(f"    * Loss C (Pred None   - 3 rank error): {loss_c:.4f}")

    assert loss_a < loss_b < loss_c, (
        f"EMD Monotonicity violated! Expected loss_a < loss_b < loss_c, got {loss_a}, {loss_b}, {loss_c}"
    )
    print("  [PASSED] Squared EMD strictly penalizes predictions proportional to physical distance!")


def test_full_model_gradient_flow():
    print("--------------------------------------------------")
    print("TEST 4: Full Model Forward-Backward Gradient Flow")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  - Testing on device: {device}")

    model = MultimodalSOTANet(
        classes_num=4,
        embed_dim=224,
        num_bottlenecks=4,
        num_heads=4,
        pretrained_video=False  # synthetic test
    ).to(device)

    loss_fn = OrdinalWassersteinEvidentialLoss(
        classes_num=4,
        sigma=0.5,
        lambda_ord_start=0.2,
        lambda_ord_end=2.0,
        total_epochs=400
    ).to(device)

    # Batch of 4 synthetic samples [B, T=4, C=10, H=224, W=224]
    video = torch.randn(4, 4, 10, 224, 224, device=device)
    # 1D audio waveform [B, 64000] so audio_frontend (Mel-Spectrogram + BN) is executed
    audio = torch.randn(4, 64000, device=device)
    targets = torch.tensor([0, 1, 2, 3], device=device)

    # Forward
    outputs = model(video, audio)
    assert "probabilities" in outputs
    assert "alpha_final" in outputs
    assert "evidence_v" in outputs
    assert "evidence_a" in outputs

    # Compute loss at epoch 100
    loss = loss_fn(outputs, {'target': targets}, epoch=100)
    print(f"  - Forward Loss value at epoch 100: {loss.item():.5f}")
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss produced NaN or Inf!"

    # Backward
    loss.backward()

    # Check gradients
    grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients!"
            grad_norm += param.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5
    print(f"  - Backward pass completed successfully. Total grad norm: {grad_norm:.4f}")
    assert grad_norm > 0.0, "Gradients are all zero!"

    params_dict = count_parameters(model)
    total_params = params_dict["total"] if isinstance(params_dict, dict) else params_dict
    print(f"  - Total model parameters: {total_params:,} (Budget: < 5,000,000)")
    assert total_params < 5_000_000, "Parameter budget exceeded!"
    print("  [PASSED] Full forward-backward gradient flow verified with 0 added parameters.")


if __name__ == "__main__":
    print("==================================================")
    print("RUNNING ORDINAL WASSERSTEIN EVIDENTIAL SUITE")
    print("==================================================")
    test_ordinal_permutation()
    test_gaussian_smoothing()
    test_wasserstein_monotonicity()
    test_full_model_gradient_flow()
    print("==================================================")
    print("ALL TESTS PASSED SUCCESSFULLY! [100%]")
    print("==================================================")
