#!/usr/bin/env python
"""
CLI for 2D STFT Audio Benchmark:
Compares 3 independent Nano Backbones trained from scratch:
  1. NanoConformer (~1.5M parameters)
  2. NanoFAST (~0.88M parameters)
  3. NanoUnderwaterDualBranch (~1.21M parameters)

Usage:
  python cli_audio_benchmark.py --mode profile
  python cli_audio_benchmark.py --mode dry-run --device cuda
  python cli_audio_benchmark.py --mode train --epochs 400 --batch_size 32
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nano_conformer import NanoConformer
from models.nano_fast import NanoFAST
from models.nano_underwater import NanoUnderwaterDualBranch
from features.audio_frontend import AudioFrontend
from config import TrainConfig
from tasks.train_audio_2d_benchmark import run_audio_2d_benchmark

# Ensure clean UTF-8 console output on Windows with immediate flushing
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AudioBenchmarkCLI")


def get_models(num_classes: int = 4) -> dict:
    return {
        "NanoConformer": NanoConformer(num_classes=num_classes, embed_dim=160, num_layers=2),
        "NanoFAST": NanoFAST(num_classes=num_classes),
        "NanoUnderwaterDualBranch": NanoUnderwaterDualBranch(num_classes=num_classes, d_model=160, num_transformer_layers=2)
    }


def mode_profile(device: str = "cpu") -> None:
    print("=================================================================")
    print("      2D STFT NANO AUDIO BENCHMARK: ARCHITECTURE PROFILER        ")
    print("=================================================================")

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    models = get_models()
    x = torch.randn(1, 1, 250, 2049, device=dev)  # 2 seconds @ 256 kHz

    print(f"Device: {dev} | Input 2D Spectrogram shape: {list(x.shape)} (2.0s audio)\n")
    print(f"{'Model Name':<28} | {'Params':<12} | {'FLOPs (2s Audio)':<18} | {'Status':<15}")
    print("-" * 80)

    for name, model in models.items():
        model = model.to(dev).eval()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        try:
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(display=False) as fcm:
                _ = model(x)
            flops = fcm.get_total_flops()
            flops_str = f"{flops / 1e9:.4f} GFLOPs"
        except Exception:
            flops_str = "N/A"

        status = "PASSED (<1.8M)" if params <= 1_800_000 else "EXCEEDS"
        print(f"{name:<28} | {params:>9,} ({params/1e6:.2f}M) | {flops_str:>16} | {status:<15}")

    print("=" * 80)
    print("Baseline Comparison Target: Previous STFT MLP (~1.39M params, 89.00% accuracy)")
    print("All 3 models process full 2D representation without temporal mean.\n")


def mode_dry_run(device: str = "cuda") -> None:
    print("=================================================================")
    print("    2D STFT AUDIO BENCHMARK: PRE-FLIGHT DRY-RUN (4GB VRAM CHECK)  ")
    print("=================================================================")

    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"Running Dry-run on: {dev}")

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"Initial VRAM Allocated: {mem_before:.1f} MB")

    batch_size = 32
    audio_dummy = torch.randn(batch_size, 512000, device=dev)
    targets_dummy = torch.randint(0, 4, (batch_size,), device=dev)

    frontend = AudioFrontend().to(dev)
    models = get_models()

    print(f"Batch Size: {batch_size} (32 x 2.0s audio waveforms @ 256 kHz)")
    print("Extracting 2D STFT Spectrograms...")
    with torch.no_grad():
        spec_2d = frontend(audio_dummy, return_2d=True)
    print(f"Spectrogram Output Shape: {spec_2d.shape}")

    print("\nSimulating Interleaved Forward + Backward passes:")
    for name, model in models.items():
        model = model.to(dev).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        optimizer.zero_grad(set_to_none=True)
        out = model(spec_2d)
        loss = F.cross_entropy(out['logits'], targets_dummy)
        loss.backward()
        optimizer.step()

        if dev.type == "cuda":
            curr_mem = torch.cuda.memory_allocated() / (1024 ** 2)
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"  ✓ {name:<25}: Loss = {loss.item():.4f} | VRAM: {curr_mem:.1f} MB (Peak: {peak_mem:.1f} MB)")
        else:
            print(f"  ✓ {name:<25}: Loss = {loss.item():.4f} [CPU mode]")

    if dev.type == "cuda":
        peak_total = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print("\n=================================================================")
        print(f"TOTAL PEAK VRAM CONSUMPTION: {peak_total:.1f} MB / 4096 MB ({peak_total / 4096 * 100:.1f}%)")
        if peak_total < 3000:
            print(">>> [PASS] VRAM check: Peak memory is safely under 3.0 GB! Zero risk of OOM.")
        else:
            print(">>> [WARN] VRAM check: Approaching 4.0 GB limit.")
        print("=================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="2D STFT Audio Benchmark CLI")
    parser.add_argument("--mode", type=str, choices=["profile", "dry-run", "train"], default="profile")
    parser.add_argument("--config", type=str, default="config/audio_benchmark_config.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--models", type=str, default="all")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "profile":
        mode_profile(device=args.device)
    elif args.mode == "dry-run":
        mode_dry_run(device=args.device)
    elif args.mode == "train":
        selected = [m.strip() for m in args.models.split(",") if m.strip()]
        run_audio_2d_benchmark(
            config_path=args.config,
            selected_models=selected,
            epochs_override=args.epochs,
            batch_size_override=args.batch_size
        )


if __name__ == "__main__":
    main()
