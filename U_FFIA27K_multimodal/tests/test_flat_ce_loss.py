import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import torch
import torch.nn.functional as F
from utils.losses import FlatCELoss


def test_flat_ce_loss_computation():
    """Verify that FlatCELoss computes global CE + aux_loss_weight * (loss_v + loss_a)."""
    torch.manual_seed(42)
    criterion = FlatCELoss(weight_ce=1.0, aux_loss_weight=0.3)

    B, C = 4, 4
    logits = torch.randn(B, C, requires_grad=True)
    logits_v = torch.randn(B, C, requires_grad=True)
    logits_a = torch.randn(B, C, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    output_dict = {
        "clipwise_output": logits,
        "logits_video": logits_v,
        "logits_audio": logits_a,
    }
    target_dict = {"target": targets}

    loss = criterion(output_dict, target_dict)

    # Compute expected loss manually
    expected_ce = F.cross_entropy(logits, targets)
    expected_v = F.cross_entropy(logits_v, targets)
    expected_a = F.cross_entropy(logits_a, targets)
    expected_loss = 1.0 * expected_ce + 0.3 * (expected_v + expected_a)

    assert torch.isclose(loss, expected_loss, atol=1e-5), f"Expected {expected_loss}, got {loss}"

    # Verify backward pass produces non-zero gradients for all heads
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert logits_v.grad is not None and logits_v.grad.abs().sum() > 0
    assert logits_a.grad is not None and logits_a.grad.abs().sum() > 0


def test_flat_ce_loss_without_aux():
    """Verify that FlatCELoss works gracefully if aux logits are None or missing."""
    criterion = FlatCELoss(weight_ce=1.0, aux_loss_weight=0.3)

    B, C = 2, 4
    logits = torch.randn(B, C, requires_grad=True)
    targets = torch.tensor([1, 2], dtype=torch.long)

    output_dict = {"clipwise_output": logits}
    target_dict = {"target": targets}

    loss = criterion(output_dict, target_dict)
    expected_ce = F.cross_entropy(logits, targets)

    assert torch.isclose(loss, expected_ce, atol=1e-5)


def test_flat_ce_loss_one_hot_targets():
    """Verify that FlatCELoss handles one-hot / 2D target tensors."""
    criterion = FlatCELoss(weight_ce=1.0, aux_loss_weight=0.3)

    B, C = 2, 4
    logits = torch.randn(B, C, requires_grad=True)
    # One-hot targets: class 0 and class 3
    targets = torch.zeros(B, C)
    targets[0, 0] = 1.0
    targets[1, 3] = 1.0

    output_dict = {"clipwise_output": logits}
    target_dict = {"target": targets}

    loss = criterion(output_dict, target_dict)
    expected_ce = F.cross_entropy(logits, torch.tensor([0, 3], dtype=torch.long))

    assert torch.isclose(loss, expected_ce, atol=1e-5)
