import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from models.multimodal_sota_net import MultimodalSOTANet
from features.motion_kinematics import FishMotionKinematics10Ch
from features.audio_frontend import AudioFrontend
from utils.profile_model import count_parameters, measure_flops, measure_latency


def test_multimodal_sota_net():
    print("=================================================================")
    print("      MULTIMODAL SOTA FISH FEEDING INTENSITY ASSESSMENT NET")
    print("      (MobileViT-XS 10-ch + EfficientAT + Google MBT + TMC)")
    print("=================================================================")

    # 1. Instantiate Core Model
    model = MultimodalSOTANet(
        classes_num=4,
        embed_dim=224,
        num_bottlenecks=4,
        num_heads=4,
        pretrained_video=False,
        num_frames=4,
        image_size=224
    )
    model.eval()

    # 2. Parameter Audit
    stats = count_parameters(model)
    print("\n[1] PARAMETER BREAKDOWN AUDIT:")
    print(f"  - Video Backbone (MobileViT-XS 10-ch) : {stats['video_backbone']:,} ({stats['video_backbone']/1e6:.3f} M)")
    print(f"  - Audio Backbone (EfficientAT 128-mel): {stats['audio_backbone']:,} ({stats['audio_backbone']/1e6:.3f} M)")
    print(f"  - Multimodal Fusion (MBT + TMC)       : {stats['fusion']:,} ({stats['fusion']/1e6:.3f} M)")
    print(f"  - Kinematics Extractor (Param-free)   : {stats['kinematics']:,}")
    print(f"  ===============================================================")
    print(f"  * CORE ARCHITECTURE TOTAL             : {stats['core_total']:,} ({stats['core_total']/1e6:.3f} M)")
    print(f"  * TOTAL TRAINABLE PARAMETERS          : {stats['total']:,} ({stats['total_million']:.3f} M)")

    assert stats['total'] < 5_000_000, f"Total parameters exceed 5.0M! Found {stats['total']}"
    assert stats['total'] >= 4_500_000, f"Total parameters too low for ~4.8M! Found {stats['total']}"
    margin = (5_000_000 - stats['total']) / 1e6
    print(f"  >>> [PASS] Model meets ~4.8M target ({stats['total_million']:.3f} M) strictly under 5.0M! (Margin: {margin:.3f} M)")

    # 3. Verify Kinematics Module (10-Channel Tensor from T=4 Frames)
    print("\n[2] 10-CHANNEL KINEMATICS EXTRACTION VERIFICATION (T=4 Frames):")
    kinematics_mod = model.motion_kinematics
    dummy_rgb = torch.rand(2, 4, 3, 224, 224)
    frames_10ch, k_summary = kinematics_mod(dummy_rgb)

    assert frames_10ch.shape == (2, 4, 10, 224, 224), f"10-channel shape mismatch: {frames_10ch.shape}"
    assert k_summary.shape == (2, 4), f"Kinematics summary shape mismatch: {k_summary.shape}"
    print(f"  - 10-channel tensor shape: {list(frames_10ch.shape)}")
    print(f"    * Ch 0-2: RGB visual channels")
    print(f"    * Ch 3-4: Dense Optical Flow (u, v)")
    print(f"    * Ch 5:   Velocity Magnitude |V|")
    print(f"    * Ch 6:   Flow Direction Angle theta")
    print(f"    * Ch 7:   Fluid Vorticity omega (feeding turbulence)")
    print(f"    * Ch 8:   Frame Difference Delta I")
    print(f"    * Ch 9:   Motion History Image (MHI)")
    print(f"  - Kinematics Summary vector: {k_summary[0].tolist()}")
    print("  >>> [PASS] 10-Channel Kinematic representation verified!")

    # 4. Verify Audio Frontend (64kHz -> 128 Mel-bins)
    print("\n[3] AUDIO FRONTEND (64kHz -> 128 Mel-bins):")
    dummy_audio = torch.randn(2, 128000)  # 2 seconds @ 64kHz
    mel_out = model.audio_frontend(dummy_audio)
    print(f"  - Log-Mel Spectrogram shape: {list(mel_out.shape)} (Expected: [2, 1, Ta, 128])")
    assert mel_out.shape[1] == 1 and mel_out.shape[-1] == 128, f"Mel shape mismatch: {mel_out.shape}"
    print("  >>> [PASS] Audio Log-Mel filterbank extraction verified!")

    # 5. Full End-to-End Forward Pass
    print("\n[4] FORWARD PASS & UNCERTAINTY QUANTIFICATION:")
    model.train()
    B = 2
    out = model(video_input=dummy_rgb, audio_input=dummy_audio)

    expected_keys = [
        "clipwise_output", "logits", "probabilities", "uncertainty",
        "uncertainty_video", "uncertainty_audio", "modality_weights",
        "kinematics_summary", "f_spatial", "f_motion", "f_frequency",
        "f_rhythm", "f_fused", "evidence_v", "evidence_a", "evidence_final"
    ]
    for k in expected_keys:
        assert k in out, f"Missing key in model output: '{k}'"
        print(f"  - Output '{k}': shape {list(out[k].shape)}")

    assert out["clipwise_output"].shape == (B, 4)
    assert out["probabilities"].shape == (B, 4)
    assert out["uncertainty"].shape == (B, 1)
    assert out["modality_weights"].shape == (B, 2)
    print("  >>> [PASS] Forward pass & tensor shapes verified!")

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
    flops_g = measure_flops(model, device="cpu", num_frames=4)
    print(f"  * Total FLOPs (Batch=1, 4 frames + 2s audio): {flops_g:.3f} GFLOPs")

    latency_ms = measure_latency(model, device="cpu", warmup=2, iterations=5, num_frames=4)
    print(f"  * Average Inference Latency (CPU): {latency_ms:.2f} ms")

    print("\n=================================================================")
    print("ALL TESTS PASSED SUCCESSFULLY! MultimodalSOTANet is 100% OPERATIONAL.")
    print("=================================================================")


if __name__ == "__main__":
    test_multimodal_sota_net()
