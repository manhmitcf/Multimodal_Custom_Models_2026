import os
import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
multimodal_root = Path(__file__).resolve().parent
if str(multimodal_root) not in sys.path:
    sys.path.insert(0, str(multimodal_root))

repo_root = multimodal_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MultimodalTrainConfig, TrainConfig
from models.multimodal_sota_net import MultimodalBoundaryAwareNet
from models.teacher_loader import OfflineTeacherEnsemble, TeacherDenseNet121, TeacherPANNS_Cnn6
from utils.losses import PairwiseTournamentLoss, KDLoss
from utils.profile_model import count_parameters


class TestKDTournamentPipeline(unittest.TestCase):
    """
    Automated Test Suite for Dual-Teacher Knowledge Distillation (KD)
    into Pairwise Tournament Multimodal Boundary Network (< 5.0M params).
    """
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.v_ckpt = str(repo_root / "teachers/DenseNet121/DL_video/checkpoint/densenet121/fold_00/video_best.pt")
        cls.a_ckpt = str(repo_root / "teachers/PANNS_Cnn6/DL_audio/checkpoint/panns_cnn6/audio_best.pt")

    def test_01_teacher_models_and_checkpoints_loading(self):
        """Verify standalone loading of DenseNet121 and PANNS_Cnn6 checkpoints with zero missing keys."""
        if not os.path.exists(self.v_ckpt) or not os.path.exists(self.a_ckpt):
            self.skipTest("Teacher checkpoints not found on local disk. Skipping.")

        teachers = OfflineTeacherEnsemble(
            video_ckpt_path=self.v_ckpt,
            audio_ckpt_path=self.a_ckpt,
            classes_num=4,
            device=self.device
        )
        self.assertFalse(any(p.requires_grad for p in teachers.parameters()), "Teacher parameters must be completely frozen!")
        self.assertFalse(teachers.video_teacher.training, "Video teacher must be in eval() mode!")
        self.assertFalse(teachers.audio_teacher.training, "Audio teacher must be in eval() mode!")

    def test_02_teacher_ensemble_forward_flexibility(self):
        """Verify OfflineTeacherEnsemble forward pass with raw waveforms and multi-channel video."""
        if not os.path.exists(self.v_ckpt) or not os.path.exists(self.a_ckpt):
            self.skipTest("Teacher checkpoints not found on local disk. Skipping.")

        teachers = OfflineTeacherEnsemble(
            video_ckpt_path=self.v_ckpt,
            audio_ckpt_path=self.a_ckpt,
            classes_num=4,
            device=self.device
        )

        B = 2
        # Student video batch [B, T=2, C=7, 224, 224]
        dummy_video_7ch = torch.randn(B, 2, 7, 224, 224, device=self.device)
        # Raw waveform [B, 128000]
        dummy_audio_raw = torch.randn(B, 128000, device=self.device)

        with torch.no_grad():
            t_out = teachers(dummy_video_7ch, dummy_audio_raw)

        self.assertIsInstance(t_out, dict, "Teachers forward must return a dictionary of multi-level targets!")
        self.assertEqual(t_out["teacher_logits_video"].shape, (B, 4), "Video teacher logits must be [B, 4]")
        self.assertEqual(t_out["teacher_logits_audio"].shape, (B, 4), "Audio teacher logits must be [B, 4]")
        self.assertEqual(t_out["teacher_feat_video"].shape, (B, 1024), "Video teacher penultimate feat must be [B, 1024]")
        self.assertEqual(t_out["teacher_feat_audio"].shape, (B, 512), "Audio teacher penultimate feat must be [B, 512]")
        self.assertEqual(t_out["teacher_feat_map_video"].shape, (B, 1024, 7, 7), "Video teacher feat map must be [B, 1024, 7, 7]")

        # Also verify with precomputed Mel Spectrogram [B, 1, 100, 128]
        dummy_audio_mel = torch.randn(B, 1, 100, 128, device=self.device)
        with torch.no_grad():
            t_out2 = teachers(dummy_video_7ch, dummy_audio_mel)
        self.assertEqual(t_out2["teacher_logits_audio"].shape, (B, 4), "Audio teacher logits from Mel must be [B, 4]")

    def test_03_kd_loss_and_independent_gradient_flow(self):
        """Verify PairwiseTournamentLoss with Multi-Level KD (Logits 35/65 + Feature Cosine + Spatial AT)."""
        loss_fn = PairwiseTournamentLoss(
            weight_act=0.5,
            weight_pairwise=0.5,
            weight_ce=1.0,
            weight_video_loss=0.3,
            weight_audio_loss=0.3,
            enable_kd=True,
            kd_temperature_video=3.0,
            kd_temperature_audio=2.0,
            kd_alpha_video=0.65,
            kd_alpha_audio=0.65,
            weight_feature_kd=0.2,
            weight_at_kd=0.2,
            enable_feature_kd=True,
            enable_at_kd=True,
        ).to(self.device)

        B = 4
        student_out = {
            'clipwise_output': torch.randn(B, 4, device=self.device, requires_grad=True),
            'logit_act': torch.randn(B, device=self.device, requires_grad=True),
            'logit_12': torch.randn(B, device=self.device, requires_grad=True),
            'logit_23': torch.randn(B, device=self.device, requires_grad=True),
            'logit_13': torch.randn(B, device=self.device, requires_grad=True),
            'logits_video': torch.randn(B, 4, device=self.device, requires_grad=True),
            'logits_audio': torch.randn(B, 4, device=self.device, requires_grad=True),
            'feat_video': torch.randn(B, 224, device=self.device, requires_grad=True),
            'feat_audio': torch.randn(B, 224, device=self.device, requires_grad=True),
            'feat_map_video': torch.randn(B, 384, 7, 7, device=self.device, requires_grad=True),
        }

        target_dict = {
            'target': torch.tensor([0, 1, 2, 3], device=self.device),
            'teacher_logits_video': torch.randn(B, 4, device=self.device),
            'teacher_logits_audio': torch.randn(B, 4, device=self.device),
            'teacher_feat_video': torch.randn(B, 1024, device=self.device),
            'teacher_feat_audio': torch.randn(B, 512, device=self.device),
            'teacher_feat_map_video': torch.randn(B, 1024, 7, 7, device=self.device),
        }

        total_loss = loss_fn(student_out, target_dict, epoch=1)
        self.assertGreater(total_loss.item(), 0.0, "Total loss must be positive")

        total_loss.backward()

        self.assertIsNotNone(student_out['logits_video'].grad, "logits_video must receive gradients!")
        self.assertIsNotNone(student_out['logits_audio'].grad, "logits_audio must receive gradients!")
        self.assertIsNotNone(student_out['feat_video'].grad, "feat_video must receive gradients from Feature KD!")
        self.assertIsNotNone(student_out['feat_audio'].grad, "feat_audio must receive gradients from Feature KD!")
        self.assertIsNotNone(student_out['feat_map_video'].grad, "feat_map_video must receive gradients from AT!")
        self.assertIsNotNone(student_out['clipwise_output'].grad, "clipwise_output must receive gradients!")

    def test_04_student_parameter_budget_under_5m(self):
        """Ensure student MultimodalBoundaryAwareNet remains strictly < 5.0M trainable parameters."""
        config_path = str(multimodal_root / "config/train_config.json")
        config = MultimodalTrainConfig.from_json(config_path)

        from features.audio_frontend import AudioFrontend
        frontend = AudioFrontend(config.audio_features)
        model = MultimodalBoundaryAwareNet(
            classes_num=4,
            embed_dim=224,
            num_heads=4,
            audio_frontend=frontend,
            image_size=224,
            num_frames=2,
            in_chans=7
        )

        stats = count_parameters(model)
        print(f"\n[Test Budget] Total Trainable Parameters: {stats['total']:,} ({stats['total_million']:.3f} M)")
        self.assertLess(stats['total'], 5_000_000, f"Model parameters ({stats['total']:,}) must be < 5.0M!")
        self.assertEqual(stats['total'], 4_681_893, "Parameters must match exact architecture budget 4,681,893")

    def test_05_adaptive_kd_confidence_and_correctness_gating(self):
        """Verify Instance-Level Adaptive KD adjusts alpha by teacher confidence and zeros out when teacher is wrong."""
        loss_fn_adaptive = PairwiseTournamentLoss(
            enable_kd=True,
            adaptive_kd=True,
            kd_alpha_video=0.65,
            kd_alpha_audio=0.65,
            adaptive_kd_min_alpha=0.0,
            enable_feature_kd=False,
            enable_at_kd=False,
        ).to(self.device)

        # Batch of 2 samples, both true target = 1 (Strong)
        targets = torch.tensor([1, 1], device=self.device)

        student_out = {
            'clipwise_output': torch.zeros(2, 4, device=self.device, requires_grad=True),
            'logits_video': torch.zeros(2, 4, device=self.device, requires_grad=True),
            'logits_audio': torch.zeros(2, 4, device=self.device, requires_grad=True),
        }

        # Teacher Video Logits:
        # Sample 0: Teacher correctly predicts class 1 with extreme confidence (~99.9%)
        # Sample 1: Teacher wrongly predicts class 0 with extreme confidence (~99.9%)
        t_logits_v = torch.tensor([
            [-5.0, 10.0, -5.0, -5.0],  # Correct (Class 1)
            [10.0, -5.0, -5.0, -5.0],  # Wrong (Class 0 instead of 1)
        ], device=self.device)

        t_logits_a = t_logits_v.clone()

        target_dict = {
            'target': targets,
            'teacher_logits_video': t_logits_v,
            'teacher_logits_audio': t_logits_a,
        }

        loss = loss_fn_adaptive(student_out, target_dict)
        loss.backward()

        # Check that mean adaptive alpha reflects: Sample 0 (~0.65) + Sample 1 (0.0) -> mean ~0.325
        self.assertAlmostEqual(loss_fn_adaptive.last_mean_alpha_v, 0.325, delta=0.02,
                               msg="Mean alpha must be ~0.325 because sample 1 teacher was wrong!")
        self.assertAlmostEqual(loss_fn_adaptive.last_mean_alpha_a, 0.325, delta=0.02,
                               msg="Mean audio alpha must be ~0.325 because sample 1 teacher was wrong!")

        # Verify fixed KD (adaptive=False) gives strictly 0.65
        loss_fn_fixed = PairwiseTournamentLoss(
            enable_kd=True,
            adaptive_kd=False,
            kd_alpha_video=0.65,
            kd_alpha_audio=0.65,
            enable_feature_kd=False,
            enable_at_kd=False,
        ).to(self.device)

        _ = loss_fn_fixed(student_out, target_dict)
        self.assertEqual(loss_fn_fixed.last_mean_alpha_v, 0.65, "Fixed KD must yield exactly 0.65 alpha")


if __name__ == "__main__":
    unittest.main()
