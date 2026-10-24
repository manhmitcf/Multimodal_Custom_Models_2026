import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
from pydantic import ValidationError
from config.train_config import ModelConfig, LossConfig, TrainConfig


def test_model_config_fusion_mode_default():
    """Verify default fusion_mode is 'sum'."""
    cfg = ModelConfig()
    assert hasattr(cfg, "fusion_mode")
    assert cfg.fusion_mode == "sum"


def test_model_config_fusion_mode_valid_values():
    """Verify valid fusion modes ('sum', 'concat', 'gated') are accepted."""
    for mode in ["sum", "concat", "gated"]:
        cfg = ModelConfig(fusion_mode=mode)
        assert cfg.fusion_mode == mode


def test_model_config_fusion_mode_invalid_value():
    """Verify invalid fusion mode raises ValidationError."""
    with pytest.raises(ValidationError):
        ModelConfig(fusion_mode="invalid_fusion")


def test_loss_config_flat_ce():
    """Verify LossConfig accepts 'flat_ce'."""
    loss_cfg = LossConfig(loss_type="flat_ce", weight_ce=1.0, aux_loss_weight=0.3)
    assert loss_cfg.loss_type == "flat_ce"
    assert loss_cfg.weight_ce == 1.0
    assert loss_cfg.aux_loss_weight == 0.3
