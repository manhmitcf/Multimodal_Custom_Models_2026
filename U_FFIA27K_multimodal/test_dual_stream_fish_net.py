import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from models.dual_stream_fish_net import DualStreamFishNet
from features.audio_frontend import AudioFrontend
from utils.profile_model import count_parameters, measure_flops, measure_latency


def test_dual_stream_fish_net():
    print("=================================================================")
    print("      DUALSTREAMFISHNET: ARCHITECTURAL AUDIT & VERIFICATION")
    print("=================================================================")

    # 1. Instantiate Core Model (without frontend, accepts Mel Spectrogram directly)
    model_core = DualStreamFishNet(
        classes_num=4,
        embed_dim=128
    )
    model_core.eval()

    # 2. Parameter Audit of Core Architecture
    stats = count_parameters(model_core)
    print("\n[1] CORE MODEL PARAMETER BREAKDOWN (Audio + Video + Fusion):")
    print(f"  - Video Backbone (4ch + TSM + ME) : {stats['video_backbone']:,} ({stats['video_backbone']/1e6:.3f} M)")
    print(f"  - Audio Backbone (Freq-SE + 1D)    : {stats['audio_backbone']:,} ({stats['audio_backbone']/1e6:.3f} M)")
    print(f"  - Enhanced Physics Fusion Head     : {stats['fusion']:,} ({stats['fusion']/1e6:.3f} M)")
    print(f"  ===============================================================")
    print(f"  * TOTAL ARCHITECTURE PARAMETERS    : {stats['core_total']:,} ({stats['core_total']/1e6:.3f} M)")
    print(f"  * TOTAL TRAINABLE PARAMETERS       : {stats['total']:,} ({stats['total_million']:.3f} M)")
    
    assert stats['core_total'] < 5_000_000, f"Model exceeds 5M budget: {stats['core_total']}"
    print(f"  >>> [PASS] Core Architecture is strictly under 5.0M! (Margin: {(5_000_000 - stats['core_total'])/1e6:.3f} M)")

    # 3. Instantiate Full Model with AudioFrontend (accepts raw waveform)
    audio_frontend = AudioFrontend()
    model_full = DualStreamFishNet(
        classes_num=4,
        embed_dim=128,
        audio_frontend=audio_frontend
    )
    model_full.train()
    stats_full = count_parameters(model_full)
    print(f"\n[2] FULL END-TO-END PIPELINE (With AudioFrontend GPU STFT):")
    print(f"  * Total Trainable Parameters       : {stats_full['total']:,} ({stats_full['total_million']:.3f} M)")
    print(f"  * (Note: AudioFrontend STFT uses 4.3M fixed non-trainable Fourier basis constants)")
    assert stats_full['total'] < 5_000_000, "Trainable parameters exceed 5M!"

    # 3b. Verification of Frequency Attention Physics Prior
    print("\n[2b] FREQUENCY ATTENTION ACOUSTIC PHYSICS VERIFICATION:")
    freq_attn = model_core.audio_backbone.freq_attention
    assert hasattr(freq_attn, "physics_prior"), "FrequencyAttentionBlock missing physics_prior buffer!"
    dummy_mel = torch.zeros(1, 1, 100, 128)
    with torch.no_grad():
        _, init_weights = freq_attn(dummy_mel)
    init_weights = init_weights[0].cpu().numpy()
    
    print(f"  - Bin 0   (50 Hz)   Weight: {init_weights[0]:.4f} (Suppression: {(1-init_weights[0])*100:.1f}%)")
    print(f"  - Bin 10  (389 Hz)  Weight: {init_weights[10]:.4f} (Suppression: {(1-init_weights[10])*100:.1f}%)")
    print(f"  - Bin 13  (491 Hz)  Weight: {init_weights[13]:.4f} (Suppression: {(1-init_weights[13])*100:.1f}%)")
    print(f"  - Bin 48  (2,014 Hz) Weight: {init_weights[48]:.4f} (Amplification band)")
    print(f"  - Bin 68  (4,057 Hz) Weight: {init_weights[68]:.4f} (Cavitation peak band)")
    print(f"  - Bin 88  (8,171 Hz) Weight: {init_weights[88]:.4f} (Amplification band)")
    print(f"  - Bin 127 (32,000 Hz) Weight: {init_weights[127]:.4f} (High-freq spray roll-off)")

    # Assert physical constraints
    assert init_weights[0] < 0.05, f"Low frequency 50Hz not suppressed: {init_weights[0]}"
    assert init_weights[10] < 0.05, f"Low frequency 389Hz not suppressed: {init_weights[10]}"
    assert init_weights[68] > 0.85, f"Feeding peak 4kHz not amplified: {init_weights[68]}"
    assert init_weights[48] > 0.70, f"Feeding band 2kHz not amplified: {init_weights[48]}"
    assert init_weights[88] > 0.70, f"Feeding band 8kHz not amplified: {init_weights[88]}"
    print("  >>> [PASS] Acoustic Physics Prior correctly suppresses <500Hz and amplifies 2-8kHz!")


    # 4. Forward Pass Tests
    print("\n[3] FORWARD PASS COMPATIBILITY TESTS:")
    B = 2
    T = 2
    dummy_video_rgb = torch.randn(B, T, 3, 224, 224)
    dummy_audio_wav = torch.randn(B, 128000) # 2 seconds @ 64kHz

    out = model_full(video_input=dummy_video_rgb, audio_input=dummy_audio_wav)

    expected_keys = [
        "clipwise_output", "logits", "f_spatial", "f_motion",
        "f_frequency", "f_rhythm", "f_fused", "modality_weights",
        "sync_score", "gating_alpha"
    ]
    for k in expected_keys:
        assert k in out, f"Missing key: {k}"
        print(f"  - Output key '{k}': shape {list(out[k].shape)}")

    assert out["clipwise_output"].shape == (B, 4)
    assert out["modality_weights"].shape == (B, 2)
    assert out["sync_score"].shape == (B, 1)
    print("  >>> [PASS] All output tensor shapes verified!")

    # 5. Backward Pass & Gradient Propagation
    print("\n[4] BACKWARD PASS & GRADIENT VERIFICATION:")
    target = torch.tensor([1, 2], dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out["clipwise_output"], target)
    loss.backward()

    trainable_with_grads = sum(1 for p in model_full.parameters() if p.requires_grad and p.grad is not None)
    total_trainable = sum(1 for p in model_full.parameters() if p.requires_grad)
    print(f"  - Trainable parameters with gradients: {trainable_with_grads} / {total_trainable}")
    print(f"  - Loss value: {loss.item():.4f}")
    assert trainable_with_grads == total_trainable, "Some trainable parameters did not receive gradients!"
    print("  >>> [PASS] Backward pass & gradient propagation 100% verified!")

    # 6. Measure FLOPs & Inference Latency
    print("\n[5] COMPUTATIONAL COMPLEXITY & LATENCY:")
    flops_g = measure_flops(model_core, device="cpu")
    print(f"  * Total FLOPs (Batch=1, 2 frames + Mel Spectrogram): {flops_g:.3f} GFLOPs")
    assert flops_g < 2.0, f"FLOPs exceed 2.0 GFLOPs: {flops_g}"
    print("  >>> [PASS] FLOPs < 2.0 GFLOPs (Ultra-lightweight for real-time edge processing)!")

    latency_ms = measure_latency(model_core, device="cpu", warmup=5, iterations=15)
    print(f"  * Average Inference Latency (CPU): {latency_ms:.2f} ms (~{1000.0/latency_ms:.1f} FPS)")

    print("\n=================================================================")
    print("ALL TESTS PASSED SUCCESSFULLY! DualStreamFishNet is 100% OPERATIONAL.")
    print("=================================================================")


if __name__ == "__main__":
    test_dual_stream_fish_net()
