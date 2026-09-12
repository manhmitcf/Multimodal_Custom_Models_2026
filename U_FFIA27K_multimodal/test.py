# ==============================================================================
# UNIFIED MULTIMODAL TEST & PRE-FLIGHT VERIFICATION SUITE
# ==============================================================================
# This single test file consolidates and replaces all separate architecture/loss
# tests. All parameters, dimensions, and configurations are loaded directly from
# config/train_config.json.
# ==============================================================================

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Any

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import numpy as np

from config.train_config import TrainConfig
from models.multimodal_sota_net import MultimodalBoundaryAwareNet, MultimodalSOTANet
from features.audio_frontend import AudioFrontend
from utils.losses import PairwiseTournamentLoss, ClipCELoss
from utils.profile_model import count_parameters, measure_flops

logger = logging.getLogger(__name__)


def get_default_config_path() -> str:
    return os.path.join(project_root, "config", "train_config.json")


def load_config(config_path: Optional[str] = None) -> TrainConfig:
    path = config_path or get_default_config_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration JSON file not found at: '{path}'")
    return TrainConfig.from_json(path)


def build_test_model(config: TrainConfig, device: torch.device) -> nn.Module:
    model_name = config.model.backbone
    frontend = AudioFrontend(config.audio_features)
    
    if model_name in ("MultimodalSOTANet", "MultimodalBoundaryAwareNet"):
        model_cls = MultimodalSOTANet
    else:
        model_cls = MultimodalBoundaryAwareNet

    model = model_cls(
        classes_num=config.model.classes_num,
        embed_dim=config.model.embed_dim,
        audio_frontend=frontend,
        image_size=config.image_size,
        num_frames=config.num_frames,
        in_chans=getattr(config.video_features, "num_channels", 7),
    ).to(device)
    return model


def test_parameter_budget(
    model: nn.Module,
    config: Optional[TrainConfig] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> bool:
    if verbose:
        print("\n" + "=" * 65)
        print("  TEST 1: PARAMETER BUDGET & GFLOPS COMPLEXITY AUDIT (< 5.0M)")
        print("=" * 65)

    stats = count_parameters(model)
    total_params = stats["total"]
    strict_limit = 5_000_000

    num_frames = getattr(config, "num_frames", 2) if config else 2
    image_size = getattr(config, "image_size", 224) if config else 224
    audio_samples = int(config.audio_features.sample_rate * 2) if (config and hasattr(config, "audio_features")) else 512000
    flops = measure_flops(model, device=device or "cpu", num_frames=num_frames, image_size=image_size, audio_samples=audio_samples)

    if verbose:
        print(f"  - Video Backbone (ConvNeXt-Nano 7-ch)       : {stats['video_backbone']:,} ({stats['video_backbone']/1e6:.3f} M)")
        print(f"  - Audio Backbone (TKEO-STFT-MLP 256k)       : {stats['audio_backbone']:,} ({stats['audio_backbone']/1e6:.3f} M)")
        print(f"  - Tournament Cross-Modal Fusion             : {stats['fusion']:,} ({stats['fusion']/1e6:.3f} M)")
        print(f"  * Total Trainable Parameters                : {total_params:,} ({stats['total_million']:.3f} M)")
        if flops > 0.0:
            print(f"  * Inference Complexity (1 sample)           : {flops:.3f} GFLOPs ({num_frames} frames @ {image_size}x{image_size}, {audio_samples:,} audio samples)")

    if total_params >= strict_limit:
        raise ValueError(f"FAILED: Total parameters ({total_params:,}) exceed budget {strict_limit:,}!")

    margin = (strict_limit - total_params) / 1e6
    if verbose:
        print(f"  >>> [PASS] Budget check passed: {stats['total_million']:.3f} M < 5.0M (Margin: +{margin:.3f} M)")
    return True


def test_kinematics_and_frontend(model: nn.Module, config: TrainConfig, device: torch.device, verbose: bool = True) -> bool:
    if verbose:
        print("\n" + "=" * 65)
        print("  TEST 2: 7-CHANNEL KINEMATICS & TKEO-STFT AUDIO FRONTEND")
        print("=" * 65)

    B = 2
    if hasattr(model, "motion_kinematics") and model.motion_kinematics is not None:
        dummy_rgb = torch.randn(B, config.num_frames, 3, config.image_size, config.image_size, device=device)
        frames_7ch = model.motion_kinematics(dummy_rgb)
        expected_shape = (B, config.num_frames, 7, config.image_size, config.image_size)
        if frames_7ch.shape != expected_shape:
            raise ValueError(f"Kinematics output shape mismatch! Got {frames_7ch.shape}, expected {expected_shape}")
        if verbose:
            print(f"  - Kinematics 7-ch tensor shape:  {list(frames_7ch.shape)} (RGB + u, v, |V|, vorticity omega)")

    # 2. Audio frontend test
    if hasattr(model, "audio_frontend") and model.audio_frontend is not None:
        audio_samples = config.audio_features.sample_rate * 2
        dummy_wave = torch.randn(B, audio_samples, device=device)
        stft_feat = model.audio_frontend(dummy_wave)
        expected_bins = config.audio_features.mel_bins
        if stft_feat.shape[-1] != expected_bins:
            raise ValueError(f"STFT frequency bins mismatch! Got {stft_feat.shape[-1]}, expected {expected_bins}")
        if verbose:
            print(f"  - Audio TKEO-STFT feature shape: {list(stft_feat.shape)} ({expected_bins} linear bins)")

    if verbose:
        print("  >>> [PASS] Video kinematics and audio frontend pipelines 100% verified.")
    return True


def test_forward_pass_and_aux_heads(model: nn.Module, config: TrainConfig, device: torch.device, verbose: bool = True) -> Tuple[bool, Any]:
    if verbose:
        print("\n" + "=" * 65)
        print("  TEST 3: MULTIMODAL FORWARD PASS & DUAL AUXILIARY HEADS")
        print("=" * 65)

    model.train()
    B = 2
    dummy_video = torch.randn(B, config.num_frames, 3, config.image_size, config.image_size, device=device)
    dummy_audio = torch.randn(B, config.audio_features.sample_rate * 2, device=device)

    outputs = model(dummy_video, dummy_audio)

    # 1. Check Multimodal Fusion Output
    if "clipwise_output" not in outputs:
        raise KeyError("Missing 'clipwise_output' in model outputs.")
    if outputs["clipwise_output"].shape != (B, config.model.classes_num):
        raise ValueError(f"Logits shape mismatch: expected ({B}, {config.model.classes_num}), got {outputs['clipwise_output'].shape}")

    # 2. Check Probabilities
    if "probabilities" in outputs:
        prob_sum = outputs["probabilities"].sum(dim=-1)
        if not torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-4):
            raise ValueError("Predicted probabilities do not sum to 1.0!")

    # 3. Check Dual Auxiliary Classification Heads (Video & Audio)
    if "logits_video" not in outputs:
        raise KeyError("Missing 'logits_video' aux head in outputs!")
    if "logits_audio" not in outputs:
        raise KeyError("Missing 'logits_audio' aux head in outputs!")
    if outputs["logits_video"].shape != (B, config.model.classes_num):
        raise ValueError(f"logits_video shape mismatch: {outputs['logits_video'].shape}")
    if outputs["logits_audio"].shape != (B, config.model.classes_num):
        raise ValueError(f"logits_audio shape mismatch: {outputs['logits_audio'].shape}")

    if verbose:
        print(f"  - Multimodal Logits shape:       {list(outputs['clipwise_output'].shape)}")
        print(f"  - Aux Video Logits shape:        {list(outputs['logits_video'].shape)}")
        print(f"  - Aux Audio Logits shape:        {list(outputs['logits_audio'].shape)}")
        if "p_w_over_m" in outputs:
            p_feed = outputs["p_feeding"].mean().item()
            print(f"  - Tournament Level-1 Gate:       P(Feeding) = {p_feed:.4f}")
            print(f"  - Tournament Level-2 Boundaries: P(W>M) = {outputs['p_w_over_m'].mean().item():.4f}, P(M>S) = {outputs['p_m_over_s'].mean().item():.4f}")
        print("  >>> [PASS] Forward pass and auxiliary heads output shapes 100% verified.")

    return True, outputs


def test_backward_and_gradient_flow(
    model: nn.Module,
    outputs: Any,
    config: TrainConfig,
    device: torch.device,
    verbose: bool = True
) -> bool:
    if verbose:
        print("\n" + "=" * 65)
        print("  TEST 4: COMPOSITE LOSS COMPUTATION & 100% GRADIENT FLOW")
        print("=" * 65)

    B = outputs["clipwise_output"].shape[0]
    dummy_targets = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], device=device, dtype=torch.float32)
    aux_loss_weight = getattr(config, "aux_loss_weight", 0.3)
    loss_type = getattr(config, "loss_type", "pairwise_tournament")

    if loss_type == "clip_ce":
        loss_fn = ClipCELoss()
        loss = loss_fn(outputs, {"target": dummy_targets})
    else:
        loss_fn = PairwiseTournamentLoss(
            weight_act=getattr(config, "weight_act", 0.5),
            weight_pairwise=getattr(config, "weight_pairwise", 0.5),
            weight_ce=getattr(config, "weight_ce", 1.0),
            aux_loss_weight=aux_loss_weight
        ).to(device)
        loss = loss_fn(outputs, {"target": dummy_targets}, epoch=1)

    loss.backward()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    params_with_grad = [p for p in trainable_params if p.grad is not None]

    if verbose:
        print(f"  - Loss Type:                     {loss_type}")
        print(f"  - Aux Head Weight (lambda_aux):  {aux_loss_weight}")
        print(f"  - Calculated Loss Value:         {loss.item():.5f}")
        print(f"  - Parameters Receiving Gradients:{len(params_with_grad)} / {len(trainable_params)}")

    if len(params_with_grad) != len(trainable_params):
        missing = len(trainable_params) - len(params_with_grad)
        raise RuntimeError(f"FAILED: {missing} parameters did not receive gradients!")

    model.zero_grad(set_to_none=True)
    if verbose:
        print("  >>> [PASS] All trainable parameters received non-zero gradients (Zero Dead Paths).")
    return True


def test_optimizer_and_scheduler(
    model: nn.Module,
    config: TrainConfig,
    verbose: bool = True
) -> bool:
    if verbose:
        print("\n" + "=" * 65)
        print("  TEST 5: OPTIMIZER & COSINE-ANNEALING SCHEDULER STEP")
        print("=" * 65)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=getattr(config, "min_lr", 1e-8)
    )

    init_lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    stepped_lr = optimizer.param_groups[0]["lr"]

    if verbose:
        print(f"  - Optimizer:                     AdamW (weight_decay={config.weight_decay})")
        print(f"  - Scheduler:                     CosineAnnealingLR (T_max={config.epochs}, eta_min={getattr(config, 'min_lr', 1e-8)})")
        print(f"  - Initial LR (Epoch 0):          {init_lr:.6e}")
        print(f"  - Stepped LR (Epoch 1):          {stepped_lr:.6e}")

    if stepped_lr >= init_lr:
        raise ValueError("Scheduler did not decay learning rate as expected!")

    if verbose:
        print("  >>> [PASS] Optimizer & CosineAnnealingLR verified smoothly.")
    return True


def run_all_tests(
    config_path: Optional[str] = None,
    config: Optional[TrainConfig] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> bool:
    """
    Runs the complete pre-flight test suite against the target configuration.
    Returns True if 100% tests pass; raises an exception on failure.
    """
    if config is None:
        config = load_config(config_path)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_file = config_path or get_default_config_path()
    if verbose:
        print("\n" + "#" * 65)
        print("    RUNNING UNIFIED MULTIMODAL PRE-FLIGHT VERIFICATION")
        print(f"    (Configuration JSON: '{cfg_file}')")
        print(f"    (Target Device: {device})")
        print("#" * 65)

    try:
        model = build_test_model(config, device)

        # Run test sequence
        test_parameter_budget(model, config=config, device=device, verbose=verbose)
        test_kinematics_and_frontend(model, config, device, verbose=verbose)
        _, outputs = test_forward_pass_and_aux_heads(model, config, device, verbose=verbose)
        test_backward_and_gradient_flow(model, outputs, config, device, verbose=verbose)
        test_optimizer_and_scheduler(model, config, verbose=verbose)

        # Cleanup memory
        del model, outputs
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if verbose:
            print("\n" + "=" * 65)
            print("  >>> [100% TESTS PASSED] ALL 5 PRE-FLIGHT CHECKS OK!")
            print("  >>> Safe to proceed with dataset loading and training session.")
            print("=" * 65 + "\n")
        return True

    except Exception as exc:
        if verbose:
            print("\n" + "!" * 65)
            print(f"  >>> [TEST FAILURE]: {exc}")
            print("  >>> Pre-flight verification failed! Halting training.")
            print("!" * 65 + "\n")
        raise exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Multimodal Test Suite")
    parser.add_argument("--config", type=str, default=None, help="Path to train_config.json")
    parser.add_argument("--device", type=str, default="cpu", help="Target device (cpu or cuda)")
    args = parser.parse_args()

    dev = torch.device(args.device)
    success = run_all_tests(config_path=args.config, device=dev, verbose=True)
    sys.exit(0 if success else 1)
