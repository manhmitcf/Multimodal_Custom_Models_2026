import unittest
import subprocess
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config.train_config import TrainConfig
from tasks.multimodal_train import MultimodalTrainer


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 4)

    def forward(self, video=None, audio=None):
        x = torch.zeros(video.shape[0], 10, device=video.device)
        logits = self.fc(x)
        return {
            'clipwise_output': logits,
            'gate_output': logits[:, :2],
            'pairwise_outputs': {'b12': logits[:, 0], 'b23': logits[:, 1], 'b13': logits[:, 2]},
            'aux_video_output': logits,
            'aux_audio_output': logits,
        }


class TestLRSchedulerMechanisms(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.base_cfg_dict = {
            "dataset": {
                "dataset_dir": "tests/dummy_data",
                "label_map": {"None": 0, "Strong": 1, "Medium": 2, "Weak": 3}
            },
            "model": {
                "backbone": "MultimodalSOTANet",
                "loss_type": "ce"
            },
            "training": {
                "epochs": 50,
                "batch_size": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.05,
                "use_onecycle": True,
                "lr_scheduler": "onecycle"
            },
            "system": {
                "dataloader_workers": 0
            }
        }

    def _create_mock_loaders(self):
        x_vid = torch.randn(8, 2, 7, 224, 224)
        x_aud = torch.randn(8, 2049)
        y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
        
        class MockDataset(torch.utils.data.Dataset):
            def __len__(self): return 8
            def __getitem__(self, idx):
                return {
                    'video_form': x_vid[idx],
                    'audio_form': x_aud[idx],
                    'target': y[idx]
                }
        loader = DataLoader(MockDataset(), batch_size=2, shuffle=False)
        return loader, loader, loader

    def test_config_lr_scheduler_property(self):
        cfg = TrainConfig.model_validate(self.base_cfg_dict)
        self.assertEqual(cfg.lr_scheduler, "onecycle")
        self.assertTrue(cfg.use_onecycle)

        cfg.lr_scheduler = "constant"
        self.assertEqual(cfg.lr_scheduler, "constant")
        self.assertFalse(cfg.use_onecycle)

        cfg.use_onecycle = True
        self.assertEqual(cfg.lr_scheduler, "onecycle")

        cfg.use_onecycle = False
        self.assertEqual(cfg.lr_scheduler, "cosine")

    def test_backward_compatibility_when_scheduler_empty(self):
        legacy_dict = dict(self.base_cfg_dict)
        legacy_dict["training"] = {
            "epochs": 50,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.05,
            "use_onecycle": False
        }
        cfg = TrainConfig.model_validate(legacy_dict)
        self.assertEqual(cfg.lr_scheduler, "cosine")
        self.assertFalse(cfg.use_onecycle)

        train_l, val_l, test_l = self._create_mock_loaders()
        trainer = MultimodalTrainer(
            model=DummyModel(),
            train_loader=train_l,
            val_loader=val_l,
            test_loader=test_l,
            config=cfg,
            device=self.device
        )
        self.assertEqual(trainer.lr_scheduler_type, "cosine")
        self.assertIsNotNone(trainer.scheduler)

    def test_constant_lr_scheduler_keeps_lr_fixed(self):
        cfg_dict = dict(self.base_cfg_dict)
        cfg_dict["training"] = dict(self.base_cfg_dict["training"])
        cfg_dict["training"]["lr_scheduler"] = "constant"
        cfg_dict["training"]["learning_rate"] = 0.0005
        cfg = TrainConfig.model_validate(cfg_dict)

        train_l, val_l, test_l = self._create_mock_loaders()
        trainer = MultimodalTrainer(
            model=DummyModel(),
            train_loader=train_l,
            val_loader=val_l,
            test_loader=test_l,
            config=cfg,
            device=self.device
        )

        self.assertIsNone(trainer.scheduler)
        self.assertEqual(trainer.lr_scheduler_type, "constant")
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]['lr'], 0.0005, places=6)

        # Run 1 epoch of training
        loss, acc, acc_v, acc_a, mAP = trainer._train_epoch(epoch=1)
        # Verify LR has NOT changed at all
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]['lr'], 0.0005, places=6)

    def test_none_alias_for_constant_lr(self):
        cfg_dict = dict(self.base_cfg_dict)
        cfg_dict["training"] = dict(self.base_cfg_dict["training"])
        cfg_dict["training"]["lr_scheduler"] = "none"
        cfg_dict["training"]["learning_rate"] = 0.002
        cfg = TrainConfig.model_validate(cfg_dict)

        train_l, val_l, test_l = self._create_mock_loaders()
        trainer = MultimodalTrainer(
            model=DummyModel(),
            train_loader=train_l,
            val_loader=val_l,
            test_loader=test_l,
            config=cfg,
            device=self.device
        )

        self.assertIsNone(trainer.scheduler)
        self.assertEqual(trainer.optimizer.param_groups[0]['lr'], 0.002)

    def test_onecycle_scheduler_steps(self):
        cfg_dict = dict(self.base_cfg_dict)
        cfg_dict["training"] = dict(self.base_cfg_dict["training"])
        cfg_dict["training"]["lr_scheduler"] = "onecycle"
        cfg_dict["training"]["learning_rate"] = 0.001
        cfg = TrainConfig.model_validate(cfg_dict)

        train_l, val_l, test_l = self._create_mock_loaders()
        trainer = MultimodalTrainer(
            model=DummyModel(),
            train_loader=train_l,
            val_loader=val_l,
            test_loader=test_l,
            config=cfg,
            device=self.device
        )

        self.assertIsNotNone(trainer.scheduler)
        self.assertEqual(trainer.lr_scheduler_type, "onecycle")
        initial_lr = trainer.optimizer.param_groups[0]['lr']
        self.assertAlmostEqual(initial_lr, 0.001 / 25.0, places=6)

        trainer._train_epoch(epoch=1)
        stepped_lr = trainer.optimizer.param_groups[0]['lr']
        self.assertNotEqual(initial_lr, stepped_lr)

    def test_cli_lr_scheduler_override(self):
        res = subprocess.run(
            [sys.executable, "main.py", "--dry-run", "--lr-scheduler", "constant"],
            capture_output=True,
            text=True
        )
        self.assertEqual(res.returncode, 0, f"CLI error: {res.stderr}")
        cfg = TrainConfig.from_json("config/train_config.json")
        self.assertEqual(cfg.lr_scheduler, "constant")
        self.assertFalse(cfg.use_onecycle)

        # Restore to onecycle
        res_restore = subprocess.run(
            [sys.executable, "main.py", "--dry-run", "--lr-scheduler", "onecycle"],
            capture_output=True,
            text=True
        )
        self.assertEqual(res_restore.returncode, 0)
        cfg_restored = TrainConfig.from_json("config/train_config.json")
        self.assertEqual(cfg_restored.lr_scheduler, "onecycle")
        self.assertTrue(cfg_restored.use_onecycle)


if __name__ == '__main__':
    unittest.main()
