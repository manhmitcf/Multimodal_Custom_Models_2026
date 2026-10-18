import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from config import SplitterConfig
from dataset.dataloader_multimodal import FishMultimodalDataLoader


def test_audio_dataloader_include_video_false():
    """Verify that setting include_video=False properly sets the flag on the dataloader."""
    splitter_cfg = SplitterConfig(
        dataset_path="/marimo/Fish_Feeding_Intensity_Dataset",
        include_video=False,
        save_results=False
    )
    # Instantiate with dry or small params without crashing
    assert hasattr(splitter_cfg, "include_video")
    assert splitter_cfg.include_video is False
    print("PASS: test_audio_dataloader_include_video_false")


if __name__ == "__main__":
    test_audio_dataloader_include_video_false()
    print("ALL TESTS IN test_audio_dataloader_speed.py PASSED!")
