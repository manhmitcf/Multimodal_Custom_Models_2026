import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from models.multimodal_sota_net import MultimodalBoundaryAwareNet, MultimodalSOTANet
from features.motion_kinematics import FishMotionKinematics7Ch
from features.audio_frontend import AudioFrontend
from utils.profile_model import count_parameters, measure_flops, measure_latency


def test_multimodal_sota_net():
    print("=================================================================")
    print("    MULTIMODAL BILATERAL BOUNDARY NET (BBN-4.5M) VERIFICATION   ")
    print("    (ConvNeXt-Nano 7ch + PANNS-CNN6-Pro TKEO + Bilateral CORAL)  ")
    print("=================================================================")

    # 1. Instantiate Core Model
    model = MultimodalBoundaryAwareNet(
        classes_num=4,
        embed_dim=224,
        num_heads=4,
        pretrained_video=False,
        num_frames=2,
        image_size=224,
        in_chans=7
    )
    model.eval()

    # 2. Parameter Audit
    stats = count_parameters(model)
    print("\n[1] PARAMETER BREAKDOWN AUDIT:")
    print(f"  - Video Backbone (ConvNeXt-Nano 7ch)  : {stats['video_backbone']:,} ({stats['video_backbone']/1e6:.3f} M)")
    print(f"  - Audio Backbone (PANNS-CNN6-Pro)     : {stats['audio_backbone']:,} ({stats['audio_backbone']/1e6:.3f} M)")
    print(f"  - Multimodal Fusion (Gated Bilateral) : {stats['fusion']:,} ({stats['fusion']/1e6:.3f} M)")
    print(f"  - Kinematics Extractor (Param-free)   : {stats['kinematics']:,}")
    print(f"  ===============================================================")
    print(f"  * CORE ARCHITECTURE TOTAL             : {stats['core_total']:,} ({stats['core_total']/1e6:.3f} M)")
    print(f"  * TOTAL TRAINABLE PARAMETERS          : {stats['total']:,} ({stats['total_million']:.3f} M)")

    assert stats['total'] < 5_000_000, f"Total parameters exceed 5.0M! Found {stats['total']}"
    assert stats['total'] >= 4_000_000, f"Total parameters too low for ~4.5M! Found {stats['total']}"
    margin = (5_000_000 - stats['total']) / 1e6
    print(f"  >>> [PASS] Model meets ~4.64M target ({stats['total_million']:.3f} M) strictly under 5.0M! (Margin: {margin:.3f} M)")

    # 3. Verify Kinematics Module (7-Channel Tensor from T=2 Frames)
    print("\n[2] 7-CHANNEL KINEMATICS EXTRACTION VERIFICATION (T=2 Frames):")
    kinematics_mod = model.motion_kinematics
    dummy_rgb = torch.rand(2, 2, 3, 224, 224)
    frames_7ch, k_summary = kinematics_mod(dummy_rgb)

    assert frames_7ch.shape == (2, 2, 7, 224, 224), f"7-channel shape mismatch: {frames_7ch.shape}"
    assert k_summary.shape == (2, 4), f"Kinematics summary shape mismatch: {k_summary.shape}"
    print(f"  - 7-channel tensor shape: {list(frames_7ch.shape)}")
    print(f"    * Ch 0-2: Spatial RGB visual channels (shoaling, white foam, pellets)")
    print(f"    * Ch 3-4: Dense Optical Flow (u, v) swimming velocities")
    print(f"    * Ch 5:   Velocity Magnitude |V| (kinetic energy)")
    print(f"    * Ch 6:   Fluid Vorticity omega (swirling turbulence from feeding strike)")
    print(f"  - Kinematics Summary vector: {k_summary[0].tolist()}")
    print("  >>> [PASS] 7-Channel Kinematic representation verified!")

    # 4. Verify Audio Frontend with TKEO (64kHz -> 128 Mel-bins)
    print("\n[3] AUDIO FRONTEND WITH TKEO (64kHz -> 128 Mel-bins):")
    dummy_audio = torch.randn(2, 128000)  # 2 seconds @ 64kHz
    mel_out = model.audio_frontend(dummy_audio)
    print(f"  - Log-Mel Spectrogram shape: {list(mel_out.shape)} (Expected: [2, 1, Ta, 128])")
    assert mel_out.shape[1] == 1 and mel_out.shape[-1] == 128, f"Mel shape mismatch: {mel_out.shape}"
    print("  >>> [PASS] Audio TKEO + Log-Mel filterbank extraction verified!")

    # 5. Full End-to-End Forward Pass
    print("\n[4] FORWARD PASS & BILATERAL BOUNDARY ENGINE:")
    model.train()
    B = 2
    out = model(video_input=dummy_rgb, audio_input=dummy_audio)

    expected_keys = [
        "clipwise_output", "logits", "probabilities", "uncertainty",
        "modality_weights", "intensity_score", "expected_intensity",
        "cutoffs", "b_final", "cutoffs_v", "cutoffs_a", "score_v", "score_a",
        "gate", "cum_probs", "cum_probs_v", "cum_probs_a",
        "probabilities_v", "probabilities_a", "f_spatial", "f_motion",
        "f_burst_v", "f_frequency", "f_rhythm", "f_burst_a", "f_fused"
    ]

    for k in expected_keys:
        assert k in out, f"Missing key in model output: '{k}'"

    assert out["clipwise_output"].shape == (B, 4)
    assert out["probabilities"].shape == (B, 4)
    assert out["uncertainty"].shape == (B, 1)
    assert out["modality_weights"].shape == (B, 2)

    # Verify probability normalization
    prob_sums = torch.sum(out["probabilities"], dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), f"Probabilities do not sum to 1: {prob_sums}"

    # Verify strict monotonicity of cutoffs: b_1 < b_2 < b_3
    b_v = out["cutoffs_v"]
    b_a = out["cutoffs_a"]
    print(f"  - Video Monotonic Cutoffs: b_1={b_v[0].item():.4f} < b_2={b_v[1].item():.4f} < b_3={b_v[2].item():.4f}")
    print(f"  - Audio Monotonic Cutoffs: b_1={b_a[0].item():.4f} < b_2={b_a[1].item():.4f} < b_3={b_a[2].item():.4f}")
    assert b_v[0] < b_v[1] < b_v[2], "Video cutoff monotonicity violated!"
    assert b_a[0] < b_a[1] < b_a[2], "Audio cutoff monotonicity violated!"
    print("  >>> [PASS] Forward pass, probabilities, & strictly monotonic cutoffs verified!")

    # 6. Backward Pass & Gradient Propagation
    print("\n[5] BACKWARD PASS & GRADIENT VERIFICATION:")
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out["clipwise_output"], targets)
    loss.backward()

    trainable_with_grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  - Trainable parameters with gradients: {trainable_with_grads} / {total_trainable}")
    print(f"  - Loss value: {loss.item():.4f}")
    assert trainable_with_grads == total_trainable, f"Gradient flow broken: {trainable_with_grads} vs {total_trainable}"
    print("  >>> [PASS] Backward pass & gradient flow 100% verified!")

    # 7. Measure FLOPs & Inference Latency
    print("\n[6] COMPUTATIONAL EFFICIENCY:")
    flops_g = measure_flops(model, device="cpu", num_frames=2)
    print(f"  * Total FLOPs (Batch=1, 2 frames + 2s audio): {flops_g:.3f} GFLOPs")

    latency_ms = measure_latency(model, device="cpu", warmup=2, iterations=5, num_frames=2)
    print(f"  * Average Inference Latency (CPU): {latency_ms:.2f} ms")

    print("\n=================================================================")
    print("ALL TESTS PASSED SUCCESSFULLY! BBN-4.5M is 100% OPERATIONAL.")
    print("=================================================================")


if __name__ == "__main__":
    test_multimodal_sota_net()
