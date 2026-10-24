import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from models.multimodal_flat_net import MultimodalFlatNet
from utils.losses import FlatCELoss
from utils.seed import seed_everything
from utils.profile_model import measure_flops


def test_parameter_budgets():
    print("\n" + "=" * 65)
    print("TEST 1: FLAT 4-CLASS MULTIMODAL PARAMETER BUDGETS")
    print("=" * 65)

    expected_budgets = {
        "sum": 4_864_878,
        "concat": 4_939_374,
        "gated": 4_865_647,
    }

    for mode, expected in expected_budgets.items():
        model = MultimodalFlatNet(num_frames=2, fusion_mode=mode)
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        v_params = sum(p.numel() for p in model.video_backbone.parameters() if p.requires_grad)
        a_params = sum(p.numel() for p in model.audio_backbone.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)

        print(f"[{mode.upper()} Mode]:")
        print(f"  - Video Backbone: {v_params:,}")
        print(f"  - Audio Backbone: {a_params:,}")
        print(f"  - Flat MLP Head:  {head_params:,}")
        print(f"  - Total Params:   {total:,} (Expected: {expected:,})")

        assert total == expected, f"Parameter mismatch for {mode}: got {total:,}, expected {expected:,}"
        print(f"  [PASSED] {mode} exactly matches budget: {total:,} params")


def test_forward_pass_all_modes():
    print("\n" + "=" * 65)
    print("TEST 2: FORWARD PASS ACROSS ALL 3 FUSION MODES")
    print("=" * 65)

    B = 2
    video = torch.randn(B, 2, 3, 224, 224)
    audio = torch.randn(B, 512000)

    for mode in ["sum", "concat", "gated"]:
        model = MultimodalFlatNet(num_frames=2, fusion_mode=mode)
        model.eval()

        with torch.no_grad():
            out = model(video, audio)

        assert "clipwise_output" in out and out["clipwise_output"].shape == (B, 4)
        assert "logits" in out and out["logits"].shape == (B, 4)
        assert "probabilities" in out and out["probabilities"].shape == (B, 4)
        assert "expected_intensity" in out and out["expected_intensity"].shape == (B, 1)
        assert "logits_video" in out and out["logits_video"].shape == (B, 4)
        assert "logits_audio" in out and out["logits_audio"].shape == (B, 4)

        probs_sum = out["probabilities"].sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)

        if mode == "concat":
            assert out["f_fused"].shape == (B, 768)
        else:
            assert out["f_fused"].shape == (B, 384)

        if mode == "gated":
            assert "gate" in out and out["gate"].shape == (B, 1)
            assert (out["gate"] >= 0.0).all() and (out["gate"] <= 1.0).all()
            assert "modality_weights" in out and out["modality_weights"].shape == (B, 2)

        print(f"  [PASSED] Forward pass for mode '{mode}' verified.")


def test_gradient_flow_all_modes():
    print("\n" + "=" * 65)
    print("TEST 3: FULL BACKWARD GRADIENT FLOW VERIFICATION")
    print("=" * 65)

    B = 2
    for mode in ["sum", "concat", "gated"]:
        seed_everything(42)
        model = MultimodalFlatNet(num_frames=2, fusion_mode=mode)
        model.train()
        criterion = FlatCELoss(weight_ce=1.0, aux_loss_weight=0.3)

        video = torch.randn(B, 2, 3, 224, 224)
        audio = torch.randn(B, 512000)
        targets = torch.tensor([0, 2], dtype=torch.long)

        out = model(video, audio)
        loss = criterion(out, {"target": targets})
        loss.backward()

        trainable_tensors = [p for p in model.parameters() if p.requires_grad]
        with_grads = [p for p in trainable_tensors if p.grad is not None and p.grad.abs().sum() > 0]

        total_tensors = len(trainable_tensors)
        grad_tensors = len(with_grads)

        print(f"[{mode.upper()} Mode]: {grad_tensors}/{total_tensors} parameter tensors received non-zero gradients.")
        assert grad_tensors == total_tensors, f"Gradient failure in mode {mode}: {grad_tensors}/{total_tensors}"
        print(f"  [PASSED] 100% gradient propagation verified for '{mode}'.")


def test_flop_complexity():
    print("\n" + "=" * 65)
    print("TEST 4: INFERENCE FLOPs PROFILING (< 2.0 GFLOPs)")
    print("=" * 65)

    for mode in ["sum", "concat", "gated"]:
        model = MultimodalFlatNet(num_frames=2, fusion_mode=mode)
        gflops = measure_flops(model, device="cpu", num_frames=2)
        print(f"  [{mode.upper()} Mode] FLOPs: {gflops:.4f} GFLOPs")
        assert gflops < 2.0, f"FLOPs exceeded limit: {gflops:.4f} >= 2.0 GFLOPs"
        print(f"  [PASSED] FLOPs complexity strictly within budget (< 2.0 GFLOPs)!")


def main():
    print("RUNNING COMPREHENSIVE FLAT 4-CLASS FUSION ABLATION VERIFICATION SUITE...")
    test_parameter_budgets()
    test_forward_pass_all_modes()
    test_gradient_flow_all_modes()
    test_flop_complexity()
    print("\n" + "=" * 65)
    print(">>> ALL ARCHITECTURAL TESTS PASSED FOR FLAT FUSION ABLATION! <<<")
    print("=" * 65)


if __name__ == "__main__":
    main()
