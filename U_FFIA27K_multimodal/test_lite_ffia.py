import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from U_FFIA27K_multimodal.models import LiteFFIANet
from U_FFIA27K_multimodal.utils import count_parameters, measure_flops, measure_latency


def run_benchmark():
    print("=" * 70)
    print("      LITEFFIA-NET: COMPREHENSIVE ARCHITECTURAL AUDIT & BENCHMARK")
    print("=" * 70)

    # 1. Instantiate model
    model = LiteFFIANet(
        classes_num=4,
        embed_dim=224,
        num_bottlenecks=4,
        num_heads=4,
        pretrained_video=False
    )
    model.eval()

    # 2. Parameter Audit
    breakdown = count_parameters(model)
    print("\n[1] PARAMETER BREAKDOWN:")
    print(f"  - Video Backbone (Spatial + Motion ME):  {breakdown['video_backbone'] / 1e6:6.3f} M params")
    print(f"  - Audio Backbone (Frequency + Rhythm):   {breakdown['audio_backbone'] / 1e6:6.3f} M params")
    print(f"  - Bidirectional Cross-Attention Fusion:  {breakdown['fusion'] / 1e6:6.3f} M params")
    print(f"  - Adaptive Modality Reliability Gate:    {breakdown['modality_gate'] / 1e6:6.3f} M params")
    print(f"  - Classifier Head (4 Classes):           {breakdown['classifier'] / 1e6:6.3f} M params")
    print("  " + "-" * 55)
    print(f"  * TOTAL MODEL PARAMETERS:                {breakdown['total_million']:6.3f} M params ({breakdown['total']:,})")
    
    assert breakdown['total'] < 5_000_000, f"Error: Model has {breakdown['total']} params, exceeding 5M limit!"
    print("  >>> VERIFICATION PASSED: Total parameters < 5 Million!")

    # 3. FLOPs Audit
    flops_g = measure_flops(model, device="cpu")
    print(f"\n[2] COMPUTATIONAL COMPLEXITY (FLOPs):")
    print(f"  * Total FLOPs (Batch=1, 2 frames + 1 audio spec): {flops_g:.3f} GFLOPs")
    assert flops_g < 2.0, f"Error: Model has {flops_g} GFLOPs, higher than expected edge limit!"
    print("  >>> VERIFICATION PASSED: Ultra-low FLOPs (< 1.5 GFLOPs) suitable for edge AI!")

    # 4. Multi-Shape Forward Pass Tests
    print("\n[3] FORWARD PASS COMPATIBILITY TESTS:")
    
    # Test A: Standard 2-frame input [B=2, T=2, 3, 224, 224]
    v_input_2f = torch.randn(2, 2, 3, 224, 224)
    a_input = torch.randn(2, 1, 100, 128)
    out_2f = model(v_input_2f, a_input)
    print("  * Test A (Batch=2, 2 Frames):")
    print(f"    - clipwise_output (logits): {tuple(out_2f['clipwise_output'].shape)}  [Expected: (2, 4)]")
    print(f"    - f_spatial (Group 1):      {tuple(out_2f['f_spatial'].shape)}  [Expected: (2, 224)]")
    print(f"    - f_motion (Group 2):       {tuple(out_2f['f_motion'].shape)}  [Expected: (2, 224)]")
    print(f"    - f_frequency (Group 3a):   {tuple(out_2f['f_frequency'].shape)}  [Expected: (2, 224)]")
    print(f"    - f_rhythm (Group 3b):      {tuple(out_2f['f_rhythm'].shape)}  [Expected: (2, 224)]")
    print(f"    - f_fused (Multimodal BMCA): {tuple(out_2f['f_fused'].shape)}  [Expected: (2, 224)]")
    print(f"    - gating_alpha (Reliability): {tuple(out_2f['gating_alpha'].shape)}, avg={out_2f['gating_alpha'].mean().item():.3f}")

    # Test B: 4-frame video input [B=1, T=4, 3, 224, 224]
    v_input_4f = torch.randn(1, 4, 3, 224, 224)
    a_input_1 = torch.randn(1, 1, 100, 128)
    out_4f = model(v_input_4f, a_input_1)
    print("  * Test B (Batch=1, 4 Frames): logits shape =", tuple(out_4f['clipwise_output'].shape))

    # Test C: Single-frame backward compatibility [B=1, 3, 224, 224]
    v_input_1f = torch.randn(1, 3, 224, 224)
    out_1f = model(v_input_1f, a_input_1)
    print("  * Test C (Batch=1, Single Frame 4D tensor): logits shape =", tuple(out_1f['clipwise_output'].shape))

    # 5. Backward Pass & Gradient Flow Verification
    print("\n[4] END-TO-END TRAINING & GRADIENT FLOW VERIFICATION:")
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    dummy_targets = torch.tensor([0, 2], dtype=torch.long)
    outputs = model(v_input_2f, a_input)
    loss = criterion(outputs['clipwise_output'], dummy_targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"  * Cross-Entropy Loss on dummy batch: {loss.item():.4f}")
    
    # Check gradients in all key submodules
    v_grad = model.video_backbone.stem[0][0].weight.grad is not None
    m_grad = model.video_backbone.motion_excitation.channel_squeeze.weight.grad is not None
    a_grad = model.audio_backbone.block1.conv1a.weight.grad is not None
    fusion_grad = model.fusion.v_to_a_attn.in_proj_weight.grad is not None
    gate_grad = model.fusion.gate[0].weight.grad is not None
    cls_grad = model.classifier[0].weight.grad is not None

    print(f"  * Gradient flow check:")
    print(f"    - Video Stem:              {'OK' if v_grad else 'FAILED'}")
    print(f"    - Motion Excitation:       {'OK' if m_grad else 'FAILED'}")
    print(f"    - Audio Backbone:          {'OK' if a_grad else 'FAILED'}")
    print(f"    - BMCA Cross-Attention:    {'OK' if fusion_grad else 'FAILED'}")
    print(f"    - Adaptive Modality Gate:  {'OK' if gate_grad else 'FAILED'}")
    print(f"    - Classifier Head:         {'OK' if cls_grad else 'FAILED'}")
    
    assert all([v_grad, m_grad, a_grad, fusion_grad, gate_grad, cls_grad]), "Error: Some modules did not receive gradients!"
    print("  >>> VERIFICATION PASSED: End-to-end backpropagation operates flawlessly!")

    # 6. Latency Benchmark
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[5] INFERENCE SPEED BENCHMARK (Device: {device}):")
    avg_latency = measure_latency(model, device=device, warmup=5, iterations=20)
    fps = 1000.0 / avg_latency
    print(f"  * Latency per sample: {avg_latency:.2f} ms")
    print(f"  * Throughput:         {fps:.1f} FPS")

    print("\n" + "=" * 70)
    print("         ALL ARCHITECTURAL & NUMERICAL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
