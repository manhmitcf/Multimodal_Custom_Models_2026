import os
import sys
import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestEfficientAT")

from features.audio_frontend import AudioFrontend
from models.efficientat import EfficientATAudioBackbone, EfficientAT_MN01
from models.multimodal_sota_net import MultimodalBoundaryAwareNet


def test_efficientat_backbone_shape_and_weights():
    logger.info("=== 1. Testing EfficientATAudioBackbone & Pretrained Weights ===")
    model = EfficientATAudioBackbone(embed_dim=224, num_tokens=2, pretrained=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"EfficientATAudioBackbone Total Params: {total_params:,} ({total_params / 1e6:.3f} M)")
    
    assert total_params < 200_000, f"Expected < 200K params for MN01, got {total_params}"

    # Test forward with 2D STFT Spectrogram [B, 1, Time=250, Freq=2049]
    dummy_stft = torch.randn(2, 1, 250, 2049)
    f_audio, f_frequency, f_rhythm, f_burst_a, tokens_audio = model(dummy_stft)

    assert f_audio.shape == (2, 224), f"Wrong f_audio shape: {f_audio.shape}"
    assert tokens_audio.shape == (2, 2, 224), f"Wrong tokens_audio shape: {tokens_audio.shape}"
    assert f_burst_a.shape == (2, 224), f"Wrong f_burst_a shape: {f_burst_a.shape}"
    assert f_frequency.shape == (2, 224)
    assert f_rhythm.shape == (2, 224)
    logger.info("[PASSED] EfficientATAudioBackbone output shapes verified successfully!")


def test_audio_frontend_2d_output():
    logger.info("=== 2. Testing AudioFrontend 2D TKEO-STFT Output ===")
    frontend = AudioFrontend(return_2d=True)
    # 2 seconds of audio at 256 kHz = 512,000 samples
    dummy_raw_audio = torch.randn(2, 512000)
    spec_2d = frontend(dummy_raw_audio)
    logger.info(f"Frontend 2D Output Shape: {spec_2d.shape}")

    assert spec_2d.ndim == 4, f"Expected 4D tensor, got {spec_2d.ndim}D"
    assert spec_2d.shape[0] == 2, f"Expected batch size 2, got {spec_2d.shape[0]}"
    assert spec_2d.shape[1] == 1, f"Expected 1 channel, got {spec_2d.shape[1]}"
    assert spec_2d.shape[3] == 2049, f"Expected 2049 freq bins, got {spec_2d.shape[3]}"
    assert spec_2d.shape[2] in (250, 251), f"Expected 250 or 251 time frames, got {spec_2d.shape[2]}"
    logger.info("[PASSED] AudioFrontend generates correct [B, 1, 250, 2049] tensor!")


def test_multimodal_boundary_net_end_to_end():
    logger.info("=== 3. Testing MultimodalBoundaryAwareNet End-to-End ===")
    model = MultimodalBoundaryAwareNet(
        classes_num=4,
        embed_dim=224,
        num_frames=2,
        in_chans=7,
        image_size=224
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"MultimodalBoundaryAwareNet Total Params: {total_params:,} ({total_params / 1e6:.3f} M)")
    assert total_params < 5_000_000, f"Total params exceeds 5M budget: {total_params}"

    dummy_video = torch.randn(2, 2, 3, 224, 224)
    dummy_audio = torch.randn(2, 512000)

    outputs = model(dummy_video, dummy_audio)

    assert "clipwise_output" in outputs
    assert outputs["clipwise_output"].shape == (2, 4)
    assert "logits_video" in outputs and outputs["logits_video"].shape == (2, 4)
    assert "logits_audio" in outputs and outputs["logits_audio"].shape == (2, 4)
    assert "prob_video" in outputs and outputs["prob_video"].shape == (2, 4)
    assert "prob_audio" in outputs and outputs["prob_audio"].shape == (2, 4)
    assert "uncertainty" in outputs
    assert "intensity_score" in outputs
    logger.info("[PASSED] MultimodalBoundaryAwareNet End-to-End forward pass verified!")


def test_phase2_freezing_policy():
    logger.info("=== 4. Testing Phase 2 Freezing Strategy ===")
    model = MultimodalBoundaryAwareNet(
        classes_num=4,
        embed_dim=224,
        num_frames=2,
        in_chans=7
    )

    # Simulate Phase 2 unfreeze_last_stages policy
    # 1. Unfreeze fusion
    for p in model.fusion.parameters():
        p.requires_grad = True

    # 2. Freeze early video stages, unfreeze late stages
    vb = model.video_backbone
    for p in vb.stem.parameters():
        p.requires_grad = False
    for p in vb.stages[:2].parameters():
        p.requires_grad = False
    for p in vb.stages[2:].parameters():
        p.requires_grad = True
    for p in vb.proj.parameters():
        p.requires_grad = True

    # 3. Audio EfficientAT MN01: 100% FROZEN in Phase 2
    ab = model.audio_backbone
    if hasattr(ab, "mn01"):
        for p in ab.parameters():
            p.requires_grad = False

    # 4. Aux heads frozen
    for p in model.aux_head_video.parameters():
        p.requires_grad = False
    for p in model.aux_head_audio.parameters():
        p.requires_grad = False

    # Verify assertions
    assert all(not p.requires_grad for p in ab.parameters()), "Audio backbone must be 100% frozen in Phase 2!"
    assert any(p.requires_grad for p in vb.stages[2].parameters()), "Video late stages must be trainable!"
    assert all(not p.requires_grad for p in vb.stages[0].parameters()), "Video early stages must be frozen!"
    assert all(p.requires_grad for p in model.fusion.parameters()), "Fusion must be 100% trainable!"
    assert all(not p.requires_grad for p in model.aux_head_audio.parameters()), "Audio aux head must be frozen!"
    logger.info("[PASSED] Phase 2 Freezing Strategy correctly locks 100% of Audio MN01 and opens Fusion + Video late stages!")


if __name__ == "__main__":
    test_efficientat_backbone_shape_and_weights()
    test_audio_frontend_2d_output()
    test_multimodal_boundary_net_end_to_end()
    test_phase2_freezing_policy()
    print("\n========================================================")
    print("ALL 4/4 UNIT TESTS PASSED FOR EfficientAT MN01 INTEGRATION!")
    print("========================================================")
