import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from models.multimodal_sota_net import MultimodalSOTANet
from utils.losses import PairwiseTournamentLoss
from utils.seed import seed_everything
from config.train_config import VideoTieBreakersConfig


def test_audio_backbones_forward_and_params():
    print("\n" + "=" * 65)
    print("TEST 1: AUDIO BACKBONES PARAMETER BUDGET & FORWARD OUTPUT SHAPES")
    print("=" * 65)

    backbones = ["mlp", "bcresnet8", "bigru", "conformer", "bimamba", "tfmamba"]
    tb_off = VideoTieBreakersConfig(enable_b12=False, enable_b23=False, enable_b13=False)

    B = 2
    v_input = torch.randn(B, 2, 3, 224, 224)
    a_input = torch.randn(B, 512000)

    for name in backbones:
        seed_everything(42)
        model = MultimodalSOTANet(num_frames=2, audio_backbone=name, tie_breakers=tb_off)
        model.eval()

        with torch.no_grad():
            out = model(v_input, a_input)

        probs = out["probabilities"]
        assert probs.shape == (B, 4), f"{name}: Expected probabilities shape (2, 4), got {probs.shape}"
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B), atol=1e-5), f"{name}: Probabilities must sum to 1.0"

        logits_a = out["logits_audio"]
        assert logits_a.shape == (B, 4), f"{name}: Expected logits_audio shape (2, 4), got {logits_a.shape}"

        total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        audio_p = sum(p.numel() for p in model.audio_backbone.parameters() if p.requires_grad)

        assert total_p < 5_000_000, f"{name}: Parameters {total_p} exceed 5.0M budget!"
        print(f"  [PASSED] {name:<10s} -> Audio: {audio_p:,} ({audio_p/1e6:.3f}M) | Total: {total_p:,} ({total_p/1e6:.3f}M) | Shape: {tuple(probs.shape)}")


def test_audio_backbones_gradient_flow():
    print("\n" + "=" * 65)
    print("TEST 2: 100% ACTIVE GRADIENT FLOW FOR ALL 6 AUDIO BACKBONES")
    print("=" * 65)

    backbones = ["mlp", "bcresnet8", "bigru", "conformer", "bimamba", "tfmamba"]
    tb_off = VideoTieBreakersConfig(enable_b12=False, enable_b23=False, enable_b13=False)
    criterion = PairwiseTournamentLoss(weight_act=0.5, weight_pairwise=0.5, weight_ce=1.0, aux_loss_weight=0.3)

    B = 4
    v_input = torch.randn(B, 2, 3, 224, 224)
    a_input = torch.randn(B, 512000)
    targets = {"target": torch.tensor([0, 1, 2, 3])}

    for name in backbones:
        seed_everything(42)
        model = MultimodalSOTANet(num_frames=2, audio_backbone=name, tie_breakers=tb_off)
        model.train()

        out = model(v_input, a_input)
        loss = criterion(out, targets)
        loss.backward()

        dead_params = []
        trainable_count = 0
        for p_name, param in model.named_parameters():
            if param.requires_grad:
                trainable_count += 1
                if param.grad is None:
                    dead_params.append(p_name)

        assert len(dead_params) == 0, f"{name}: Broken gradient flow for {len(dead_params)} tensors: {dead_params}"
        print(f"  [PASSED] {name:<10s} -> 100% Gradient Flow ({trainable_count}/{trainable_count} tensors active, Loss={loss.item():.4f})")


def test_audio_frontend_augmentation():
    print("\n" + "=" * 65)
    print("TEST 3: AUDIO FRONTEND 1D AND 2D AUGMENTATION CONSISTENCY")
    print("=" * 65)

    from features.audio_frontend import AudioFrontend

    frontend = AudioFrontend()
    waveform = torch.randn(2, 512000)

    # 1. Eval mode identity / determinism
    frontend.eval()
    with torch.no_grad():
        out_eval1 = frontend(waveform, return_2d=True)
        out_eval2 = frontend(waveform, return_2d=True)
    assert torch.allclose(out_eval1, out_eval2), "Eval mode must be deterministic!"
    print("  [PASSED] Eval mode is 100% deterministic (no stochastic jitter/cutout in evaluation).")

    # 2. Train mode shape preservation
    frontend.train()
    out_train_1d = frontend(waveform, return_2d=False)
    out_train_2d = frontend(waveform, return_2d=True)
    assert out_train_1d.shape == (2, 2049)
    assert out_train_2d.shape == (2, 251, 2049)
    print(f"  [PASSED] Train mode shapes: 1D={tuple(out_train_1d.shape)}, 2D={tuple(out_train_2d.shape)}")


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("RUNNING AUDIO BACKBONES ABLATION VERIFICATION TEST SUITE")
    print("=" * 65)

    test_audio_backbones_forward_and_params()
    test_audio_backbones_gradient_flow()
    test_audio_frontend_augmentation()

    print("\n" + "=" * 65)
    print("ALL AUDIO BACKBONE ABLATION TESTS PASSED SUCCESSFULLY! (100% READY)")
    print("=" * 65 + "\n")
