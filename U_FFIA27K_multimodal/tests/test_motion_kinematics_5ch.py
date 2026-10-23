import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from features.motion_kinematics import FishMotionKinematics5Ch

def test_motion_kinematics_5ch_shape_and_channels():
    kinematics = FishMotionKinematics5Ch(image_size=224)
    dummy_video = torch.randn(2, 2, 3, 224, 224)
    frames_5ch, summary = kinematics(dummy_video)
    assert frames_5ch.shape == (2, 2, 5, 224, 224)
    assert summary.shape == (2, 4)
    assert not torch.isnan(frames_5ch).any()

def test_motion_kinematics_5ch_zero_input():
    kinematics = FishMotionKinematics5Ch(image_size=224)
    zero_video = torch.zeros(2, 2, 3, 224, 224)
    frames_5ch, _ = kinematics(zero_video)
    assert not torch.isnan(frames_5ch).any()
    assert not torch.isinf(frames_5ch).any()

def test_motion_kinematics_5ch_single_frame():
    kinematics = FishMotionKinematics5Ch(image_size=224)
    single_frame = torch.randn(2, 1, 3, 224, 224)
    frames_5ch, summary = kinematics(single_frame)
    assert frames_5ch.shape == (2, 1, 5, 224, 224)
    assert summary.shape == (2, 4)
    # Optical flow should be zero for single frame
    assert torch.equal(frames_5ch[:, :, 3:], torch.zeros_like(frames_5ch[:, :, 3:]))
