import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from features.motion_kinematics import FishMotionKinematics8Ch
from models.multimodal_sota_net import MultimodalBoundaryAwareNet


def test_8ch_kinematics_extractor():
    print('=' * 65)
    print('TEST 1: 8-CHANNEL MOTION KINEMATICS EXTRACTOR (T=2 FRAMES)')
    print('=' * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kinematics = FishMotionKinematics8Ch(image_size=224).to(device)

    B, T = 4, 2
    dummy_rgb = torch.rand(B, T, 3, 224, 224, device=device, requires_grad=True)

    frames_8ch, summary = kinematics(dummy_rgb)

    print(f'Output frames_8ch shape: {list(frames_8ch.shape)}')
    print(f'Output summary shape:    {list(summary.shape)}')

    assert frames_8ch.shape == (B, T, 8, 224, 224), f'Shape mismatch: {frames_8ch.shape}'
    assert summary.shape == (B, 4), f'Summary mismatch: {summary.shape}'

    ch_names = [
        'Ch 0: R',
        'Ch 1: G',
        'Ch 2: B',
        'Ch 3: Flow u',
        'Ch 4: Flow v',
        'Ch 5: Velocity |V|',
        'Ch 6: Vorticity omega',
        'Ch 7: Convective Acc |a|'
    ]

    for c in range(8):
        ch_slice = frames_8ch[:, :, c]
        c_min = ch_slice.min().item()
        c_max = ch_slice.max().item()
        c_mean = ch_slice.mean().item()
        assert not torch.isnan(ch_slice).any(), f'NaN found in channel {c} ({ch_names[c]})'
        assert not torch.isinf(ch_slice).any(), f'Inf found in channel {c} ({ch_names[c]})'
        print(f'  [{c}] {ch_names[c]:<25}: min={c_min:+.4f}, max={c_max:+.4f}, mean={c_mean:+.4f}')

    loss = frames_8ch.sum()
    loss.backward()
    assert dummy_rgb.grad is not None, 'Gradient did not flow back to input RGB frames!'
    print('\
[PASSED] Differentiability verified: Gradients successfully propagated to input RGB!')


def test_full_model_8ch():
    print('\
' + '=' * 65)
    print('TEST 2: FULL MULTIMODAL MODEL WITH 8-CH CONVNEXT-NANO & TOURNAMENT')
    print('=' * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalBoundaryAwareNet(in_chans=8, num_frames=2).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    v_params = sum(p.numel() for p in model.video_backbone.parameters())
    a_params = sum(p.numel() for p in model.audio_backbone.parameters())
    f_params = sum(p.numel() for p in model.fusion.parameters())

    print(f'Total Parameters:            {total_params:,} ({total_params/1e6:.3f} M)')
    print(f'  - ConvNeXt-Nano 8-ch Video: {v_params:,} ({v_params/1e6:.3f} M)')
    print(f'  - TKEO-STFT-MLP 256k Audio: {a_params:,} ({a_params/1e6:.3f} M)')
    print(f'  - Tournament Fusion:        {f_params:,} ({f_params/1e6:.3f} M)')

    assert total_params < 5_000_000, f'Exceeds 5.0M: {total_params}'
    headroom = 5_000_000 - total_params
    print(f'Headroom under 5.0M limit:   {headroom:,} params')

    B = 2
    v_in = torch.randn(B, 2, 3, 224, 224, device=device)
    a_in = torch.randn(B, 512000, device=device)

    with torch.no_grad():
        out = model(v_in, a_in)

    assert 'probabilities' in out
    assert out['probabilities'].shape == (B, 4)
    print(f'[PASSED] Full forward pass verified! Output keys: {list(out.keys())}')
    print('=' * 65)


if __name__ == '__main__':
    test_8ch_kinematics_extractor()
    test_full_model_8ch()
