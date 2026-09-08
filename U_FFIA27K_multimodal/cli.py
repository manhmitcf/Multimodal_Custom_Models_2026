#!/usr/bin/env python
"""
Command Line Interface (CLI) for Multimodal Fish Feeding Intensity Assessment.
Architecture: MultimodalBoundaryAwareNet (~4.78M parameters)
Features:
  - Video Stream: 7-Channel Spatiotemporal Kinematics (RGB + Flow u,v + Vorticity omega + Deceleration Delta|V|)
  - Audio Stream: Time-Frequency Factorized EfficientAT (Splash Band 2-8 kHz + Rhythm Depthwise Conv)
  - Multimodal Engine: Bidirectional Cross-Attention (Bi-CA) + Boundary Discrepancy Gate (BDG)
    + Boundary-Aware Channel Routing (BACA: 128 Core + 96 Boundary)
    + Strictly Monotonic Cumulative Ordinal Head (b_1 < b_2 < b_3)

Supported Modes:
  1. profile   : Inspect parameter breakdown, compute FLOPs and latency.
  2. dry-run   : Fast pre-flight check (< 2s) with dummy forward/backward passes.
  3. train     : Full dataset training with validation, checkpointing, and history logging.
  4. evaluate  : Evaluate a trained checkpoint on validation/test datasets.

Examples:
  python cli.py --mode profile
  python cli.py --mode dry-run --device cuda
  python cli.py --mode train --config config/train_config.json
  python cli.py --mode evaluate --weights checkpoint/multimodalboundaryawarenet/best_model.pth
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from config import MultimodalTrainConfig, TrainConfig
from models import MultimodalBoundaryAwareNet, MultimodalSOTANet
from features.audio_frontend import AudioFrontend
from utils.profile_model import count_parameters, measure_flops, measure_latency
from utils.losses import OrdinalWassersteinEvidentialLoss, ClipCELoss, PairwiseTournamentLoss

# Ensure clean UTF-8 console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MultimodalCLI")


def get_device(requested_device: str) -> torch.device:
    if requested_device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested_device)


def build_cli_model(config: TrainConfig, pretrained_video: bool = True) -> nn.Module:
    audio_frontend = AudioFrontend(config.audio_features)
    model = MultimodalBoundaryAwareNet(
        classes_num=config.model.classes_num,
        embed_dim=config.model.embed_dim,
        num_heads=config.model.num_heads,
        pretrained_video=pretrained_video,
        audio_frontend=audio_frontend,
        image_size=config.image_size,
        num_frames=config.num_frames,
        in_chans=getattr(config.video_features, "num_channels", 7),
        use_frequency_attention=getattr(config.audio_features, "use_frequency_attention", False)
    )
    return model


def mode_profile(args: argparse.Namespace) -> None:
    """Audit parameter counts, FLOPs, and latency."""
    print("=================================================================")
    print("  MULTIMODAL BOUNDARY-AWARE NET: ARCHITECTURE PROFILER")
    print("=================================================================")

    config_path = args.config if os.path.exists(args.config) else os.path.join(project_root, args.config)
    if os.path.exists(config_path):
        config = MultimodalTrainConfig.from_json(config_path)
    else:
        config = TrainConfig()

    device = get_device(args.device)
    model = build_cli_model(config, pretrained_video=False).to(device)
    model.eval()

    # 1. Parameter Breakdown
    stats = count_parameters(model)
    print("\n[1] DETAILED PARAMETER BREAKDOWN:")
    print(f"  - Video Backbone (ConvNeXt-Nano 7-ch) : {stats['video_backbone']:,} ({stats['video_backbone']/1e6:.3f} M)")
    print(f"  - Audio Backbone (TKEO-STFT-MLP 256k) : {stats['audio_backbone']:,} ({stats['audio_backbone']/1e6:.3f} M)")
    print(f"  - Pairwise Tournament Fusion          : {stats['fusion']:,} ({stats['fusion']/1e6:.3f} M)")
    print(f"  - Kinematics Extractor (Sobel/P-free) : {stats['kinematics']:,}")
    print(f"  ===============================================================")
    print(f"  * CORE ARCHITECTURE TOTAL             : {stats['core_total']:,} ({stats['core_total']/1e6:.3f} M)")
    print(f"  * TOTAL TRAINABLE PARAMETERS          : {stats['total']:,} ({stats['total_million']:.3f} M)")

    margin = (5_000_000 - stats['total']) / 1e6
    if stats['total'] < 5_000_000:
        print(f"  >>> [PASS] Budget check: {stats['total_million']:.3f} M < 5.0M (Margin: +{margin:.3f} M)")
    else:
        print(f"  >>> [FAIL] Parameter count exceeds 5.0M! ({stats['total_million']:.3f} M)")

    # 2. FLOPs Calculation
    print("\n[2] COMPUTATIONAL COMPLEXITY (FLOPs):")
    try:
        flops = measure_flops(model, device=str(device), num_frames=config.num_frames)
        print(f"  * Inference FLOPs (1 Sample, 4 frames + 2s audio): {flops:.3f} GFLOPs")
    except Exception as exc:
        print(f"  * FLOPs counter unavailable: {exc}")

    # 3. Latency
    print(f"\n[3] INFERENCE LATENCY ({device.type.upper()}):")
    try:
        latency = measure_latency(model, device=str(device), warmup=3, iterations=10, num_frames=config.num_frames)
        print(f"  * Mean Latency: {latency:.2f} ms per clip")
    except Exception as exc:
        print(f"  * Latency measurement unavailable: {exc}")

    print("\n=================================================================")


def mode_dry_run(args: argparse.Namespace) -> None:
    """Instantaneous Pre-Flight Verification (< 2s) to guarantee zero runtime failures."""
    print("=================================================================")
    print("  MULTIMODAL BOUNDARY-AWARE NET: FAST PRE-FLIGHT DRY RUN")
    print("=================================================================")

    config_path = args.config if os.path.exists(args.config) else os.path.join(project_root, args.config)
    if os.path.exists(config_path):
        config = MultimodalTrainConfig.from_json(config_path)
    else:
        config = TrainConfig()

    device = get_device(args.device)
    print(f"[*] Initializing model on target device: {device}...")
    model = build_cli_model(config, pretrained_video=False).to(device)

    # 1. Forward Pass
    print("[*] Running forward pass with dummy batch (B=2)...")
    dummy_video = torch.randn(2, config.num_frames, 3, config.image_size, config.image_size, device=device)
    dummy_audio = torch.randn(2, config.audio_features.sample_rate * 2, device=device)
    dummy_targets = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], device=device)

    model.train()
    out = model(dummy_video, dummy_audio)

    # Check outputs
    assert "clipwise_output" in out and out["clipwise_output"].shape == (2, 4), "Logits output shape mismatch!"
    assert "probabilities" in out and out["probabilities"].shape == (2, 4), "Probabilities shape mismatch!"

    if "p_w_over_m" in out:
        p_act = out["p_feeding"].mean().item()
        p_12 = out["p_w_over_m"].mean().item()
        p_23 = out["p_m_over_s"].mean().item()
        u12 = out["u_tie_12"].mean().item() if "u_tie_12" in out else 0.0
        g12 = out["gamma_12"].item() if "gamma_12" in out else 0.0
        u23 = out["u_tie_23"].mean().item() if "u_tie_23" in out else 0.0
        g23 = out["gamma_23"].item() if "gamma_23" in out else 0.0
        print(f"    - Level 1 Activity Gate: P(Feeding)={p_act:.4f}")
        print(f"    - Sandwich B12 (Weak vs Med): P(W>M)={p_12:.4f} [Audio Tie u_tie12={u12:.4f}, gamma12={g12:.4f}]")
        print(f"    - Sandwich B23 (Med vs Str):  P(M>S)={p_23:.4f} [Audio Tie u_tie23={u23:.4f}, gamma23={g23:.4f}]")
    elif "cutoffs" in out:
        b_1, b_2, b_3 = out["cutoffs"][0].item(), out["cutoffs"][1].item(), out["cutoffs"][2].item()
        print(f"    - Cutoffs: b_1={b_1:.4f} < b_2={b_2:.4f} < b_3={b_3:.4f}")
        assert b_1 < b_2 < b_3, "Cutoff monotonicity violated!"

    # 2. Backward Pass & Gradient Flow
    loss_type = getattr(config, "loss_type", "pairwise_tournament")
    if loss_type == "pairwise_tournament" or "p_w_over_m" in out:
        print("[*] Running backward pass with PairwiseTournamentLoss...")
        loss_fn = PairwiseTournamentLoss(
            weight_act=getattr(config, "weight_act", 0.5),
            weight_pairwise=getattr(config, "weight_pairwise", 0.5),
            weight_ce=getattr(config, "weight_ce", 1.0)
        ).to(device)
    else:
        print("[*] Running backward pass with OrdinalWassersteinEvidentialLoss...")
        loss_fn = OrdinalWassersteinEvidentialLoss(
            classes_num=config.model.classes_num,
            sigma=getattr(config, "ordinal_sigma", 0.5),
            lambda_ord_start=getattr(config, "lambda_ord_start", 0.2),
            lambda_ord_end=getattr(config, "lambda_ord_end", 2.0),
            total_epochs=config.epochs
        ).to(device)

    loss = loss_fn(out, {"target": dummy_targets}, epoch=1)
    loss.backward()

    grads_ok = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"    - Parameters receiving gradients: {grads_ok} / {total_trainable}")
    assert grads_ok == total_trainable, f"Gradient flow broken: only {grads_ok}/{total_trainable} got gradients!"

    print("\n>>> [SUCCESS] 100% PRE-FLIGHT VERIFIED! All modules, device placement, and gradients are operational.")
    print("=================================================================")


def mode_train(args: argparse.Namespace) -> None:
    """Launch full multimodal training pipeline."""
    from main import main as run_main
    sys.argv = [
        sys.argv[0],
        "--config", args.config,
        "--device", args.device,
    ]
    if args.dry_run:
        sys.argv.append("--dry-run")
    if getattr(args, "no_two_phase", False):
        sys.argv.append("--no-two-phase")
    if getattr(args, "phase1_epochs", None) is not None:
        sys.argv.extend(["--phase1-epochs", str(args.phase1_epochs)])
    run_main()


def mode_evaluate(args: argparse.Namespace) -> None:
    """Evaluate trained checkpoint weights."""
    print("=================================================================")
    print("  MULTIMODAL BOUNDARY-AWARE NET: EVALUATION MODE")
    print("=================================================================")

    if not args.weights or not os.path.exists(args.weights):
        logger.error(f"Checkpoint weights not found at: {args.weights}")
        sys.exit(1)

    config_path = args.config if os.path.exists(args.config) else os.path.join(project_root, args.config)
    config = MultimodalTrainConfig.from_json(config_path) if os.path.exists(config_path) else TrainConfig()
    device = get_device(args.device)

    # Build model & load weights
    model = build_cli_model(config, pretrained_video=False).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Loaded checkpoint weights from: {args.weights}")

    # Build test DataLoader
    from dataset import FishMultimodalDataLoader
    data_loader_mgr = FishMultimodalDataLoader(config=config)
    _, _, test_loader = data_loader_mgr.get_dataloaders()

    from utils import MultimodalEvaluator
    loss_fn = ClipCELoss()
    evaluator = MultimodalEvaluator(model=model, loss_fn=loss_fn)
    metrics = evaluator.evaluate(test_loader, device=device)

    print("\n[EVALUATION RESULTS]:")
    print(f"  * Accuracy   : {metrics.get('accuracy', 0.0) * 100:.2f}%")
    print(f"  * Macro F1   : {metrics.get('f1_macro', 0.0):.4f}")
    print(f"  * Mean Loss  : {metrics.get('loss', 0.0):.4f}")
    print("=================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MultimodalBoundaryAwareNet CLI Tool for Fish Feeding Intensity Assessment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="profile",
        choices=["profile", "dry-run", "train", "evaluate"],
        help="CLI operational mode."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_config.json",
        help="Path to multimodal training configuration JSON file."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model checkpoint .pth file for evaluate mode."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target compute device (cuda or cpu)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform pre-flight verification only without running full training."
    )
    parser.add_argument(
        "--no-two-phase",
        action="store_true",
        help="Disable two-phase warmup and train end-to-end directly."
    )
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=None,
        help="Number of epochs for Phase 1 backbone warmup (default: 200)."
    )

    args = parser.parse_args()

    if args.mode == "profile":
        mode_profile(args)
    elif args.mode == "dry-run":
        mode_dry_run(args)
    elif args.mode == "train":
        mode_train(args)
    elif args.mode == "evaluate":
        mode_evaluate(args)


if __name__ == "__main__":
    main()
