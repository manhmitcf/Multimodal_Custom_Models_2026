import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.train_config import load_train_config


def test_new_architecture_train_config():
    cfg = load_train_config()
    assert cfg.model.embed_dim == 384
    assert cfg.video_features.num_channels == 5
    assert cfg.video_features.temporal_mode == "composite_ln"
    assert getattr(cfg.model, "audio_hidden_dims", (768, 512)) == (768, 512)
    assert cfg.in_chans == 5
    assert cfg.temporal_mode == "composite_ln"
