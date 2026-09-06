import time
import torch
import torch.nn as nn
from typing import Dict, Any

try:
    from torch.utils.flop_counter import FlopCounterMode
except ImportError:
    FlopCounterMode = None


def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """Calculate parameter breakdown for LiteFFIANet."""
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    fusion_module = getattr(model, "fusion", getattr(model, "mbt_fusion", None))
    fusion_params = sum(p.numel() for p in fusion_module.parameters() if p.requires_grad) if fusion_module else 0
    gate_module = getattr(model, "modality_gate", None)
    gate_params = sum(p.numel() for p in gate_module.parameters() if p.requires_grad) if gate_module else 0

    breakdown = {
        "video_backbone": sum(p.numel() for p in model.video_backbone.parameters() if p.requires_grad) if hasattr(model, "video_backbone") else 0,
        "audio_backbone": sum(p.numel() for p in model.audio_backbone.parameters() if p.requires_grad) if hasattr(model, "audio_backbone") else 0,
        "fusion": fusion_params,
        "mbt_fusion": fusion_params,
        "modality_gate": gate_params,
        "classifier": sum(p.numel() for p in model.classifier.parameters() if p.requires_grad) if hasattr(model, "classifier") else 0,
        "total": total_trainable,
        "total_million": total_trainable / 1e6
    }
    return breakdown


def measure_flops(model: nn.Module, device: str = "cpu") -> float:
    """Measure total FLOPs for 1 sample inference using native PyTorch FlopCounterMode."""
    if FlopCounterMode is None:
        return 0.0
        
    model.eval()
    dummy_video = torch.randn(1, 2, 3, 224, 224, device=device)
    dummy_audio = torch.randn(1, 1, 100, 128, device=device)
    
    with FlopCounterMode(display=False) as flop_counter:
        with torch.no_grad():
            _ = model(dummy_video, dummy_audio)
            
    total_flops = flop_counter.get_total_flops()
    return total_flops / 1e9  # in GFLOPs


def measure_latency(
    model: nn.Module,
    device: str = "cpu",
    warmup: int = 10,
    iterations: int = 50
) -> float:
    """Measure average inference latency in milliseconds."""
    model.to(device)
    model.eval()
    
    dummy_video = torch.randn(1, 2, 3, 224, 224, device=device)
    dummy_audio = torch.randn(1, 1, 100, 128, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_video, dummy_audio)
            
    if device == "cuda":
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_video, dummy_audio)
            if device == "cuda":
                torch.cuda.synchronize()
                
    elapsed = time.perf_counter() - start_time
    avg_latency_ms = (elapsed / iterations) * 1000.0
    return avg_latency_ms
