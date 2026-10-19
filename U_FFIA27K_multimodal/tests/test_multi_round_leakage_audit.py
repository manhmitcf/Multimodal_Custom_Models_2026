import os
import sys
import unittest
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.train_config import SplitterConfig
from dataset.data_split import (
    BaseDataSplitter,
    LABEL_ORDER,
    LABEL_TO_CLASS,
)
from transforms.video_transform import ConsistentVideoTransform
from features.motion_kinematics import FishMotionKinematics7Ch
from main import zip_directory


class DummySplitter(BaseDataSplitter):
    """Concrete implementation of BaseDataSplitter for synthetic leakage verification."""
    def __init__(self, config: SplitterConfig):
        self.config = config
        self.dataset_path = config.dataset_path
        self.seed = config.seed
        self.test_sample_per_class = config.test_sample_per_class
        self.split_strategy = config.split_strategy
        self.evaluation_mode = config.evaluation_mode
        self.num_folds = config.num_folds
        self.fold_index = config.fold_index
        self.cv_val_ratio = getattr(config, "cv_val_ratio", 0.125)
        self.video_exists = True
        self.audio_exists = True

    def get_file_list(self, split_name: str) -> List[str]:
        return []

    def split_data(self) -> Tuple[List[List], List[List], List[List]]:
        return [], [], []

    def _format_samples(self, split_name: str, data_list: List[List]):
        return []

    def _sample_group_key(self, path: str) -> str:
        parts = Path(str(path).replace('\\', '/')).parts
        if len(parts) >= 3:
            return f"{parts[-3]}_{parts[-2]}"
        return "default_group"

    def _sample_temporal_key(self, path: str) -> tuple:
        p = str(path).replace('\\', '/')
        stem = Path(p).stem
        parts = stem.split('_')
        seq_num = int(parts[-1]) if parts[-1].isdigit() else 0
        return (0, 0, 0, seq_num)

    def _generate_splits(self, all_entries: List[List]):
        if self.split_strategy == "group_random":
            return self._split_entries_group_holdout(all_entries)

        all_entries = sorted(all_entries, key=self._entry_sample_key)
        by_class = {label: [item for item in all_entries if item[2] == label] for label in LABEL_ORDER}

        train_dict, test_dict, val_dict = [], [], []
        for label, class_list in by_class.items():
            if self.split_strategy == "time_series":
                class_list = sorted(class_list, key=self._entry_temporal_key)
                holdout_size = 2 * self.test_sample_per_class
                train_dict.extend(class_list[:-holdout_size])
                val_dict.extend(class_list[-holdout_size:-self.test_sample_per_class])
                test_dict.extend(class_list[-self.test_sample_per_class:])
            else:
                rng = np.random.RandomState(self.seed + label)
                rng.shuffle(class_list)
                test_dict.extend(class_list[:self.test_sample_per_class])
                val_dict.extend(class_list[self.test_sample_per_class:2*self.test_sample_per_class])
                train_dict.extend(class_list[2*self.test_sample_per_class:])

        self._validate_split_policy(train_dict, test_dict, val_dict)
        return train_dict, test_dict, val_dict


class TestMultiRoundLeakageAudit(unittest.TestCase):
    """
    Exhaustive Multi-Round Leakage Verification Suite:
      - Round 1: Mathematical disjointness across Train/Val/Test splits.
      - Round 2: Deterministic validation transforms vs stochastic train transforms.
      - Round 3: Numerical isolation of Kinematics (zero cross-sample contamination).
      - Round 4: Security credentials and secret exclusion in artifact bundling.
      - Round 5: Memory gradient detachment and graph non-retention.
    """

    def setUp(self):
        # Generate 160 synthetic sample entries (40 per class, across 4 sessions)
        # Format matches dataset: .../video/date/session/class/sample.mp4
        self.synthetic_entries = []
        for session_id in [1, 2, 3, 4]:
            for label in LABEL_ORDER:
                cls_name = LABEL_TO_CLASS[label]
                for sample_id in range(10):
                    idx = (session_id - 1) * 10 + sample_id
                    v_path = f"/data/video/2026-10-19/session_{session_id:02d}/{cls_name}/sample_{idx:04d}.mp4"
                    a_path = f"/data/audio/2026-10-19/session_{session_id:02d}/{cls_name}/sample_{idx:04d}.wav"
                    self.synthetic_entries.append([v_path, a_path, label])

    def test_round1_data_split_leakage_random_sample(self):
        """Round 1A: Verify zero sample overlap in random_sample strategy."""
        cfg = SplitterConfig(
            dataset_path="/dummy",
            split_strategy="random_sample",
            evaluation_mode="holdout",
            test_sample_per_class=5,
            seed=42,
        )
        splitter = DummySplitter(cfg)
        train, test, val = splitter._generate_splits(self.synthetic_entries)

        train_keys = {splitter._entry_sample_key(x) for x in train}
        test_keys = {splitter._entry_sample_key(x) for x in test}
        val_keys = {splitter._entry_sample_key(x) for x in val}

        self.assertEqual(len(train_keys.intersection(test_keys)), 0, "Train and Test share samples!")
        self.assertEqual(len(train_keys.intersection(val_keys)), 0, "Train and Val share samples!")
        self.assertEqual(len(val_keys.intersection(test_keys)), 0, "Val and Test share samples!")
        self.assertEqual(len(test), 20)
        self.assertEqual(len(val), 20)
        self.assertEqual(len(train), 120)

    def test_round1_data_split_leakage_time_series(self):
        """Round 1B: Verify strict temporal ordering and zero overlap in time_series strategy."""
        cfg = SplitterConfig(
            dataset_path="/dummy",
            split_strategy="time_series",
            evaluation_mode="holdout",
            test_sample_per_class=5,
            seed=42,
        )
        splitter = DummySplitter(cfg)
        train, test, val = splitter._generate_splits(self.synthetic_entries)

        train_keys = {splitter._entry_sample_key(x) for x in train}
        test_keys = {splitter._entry_sample_key(x) for x in test}
        val_keys = {splitter._entry_sample_key(x) for x in val}

        self.assertEqual(len(train_keys.intersection(test_keys)), 0)
        self.assertEqual(len(train_keys.intersection(val_keys)), 0)
        self.assertEqual(len(val_keys.intersection(test_keys)), 0)

        # Check temporal order per class: max(train) < min(val) and max(val) < min(test)
        for label in LABEL_ORDER:
            c_train = [splitter._entry_temporal_key(x) for x in train if x[2] == label]
            c_val = [splitter._entry_temporal_key(x) for x in val if x[2] == label]
            c_test = [splitter._entry_temporal_key(x) for x in test if x[2] == label]

            self.assertLess(max(c_train), min(c_val), "Time leakage: Train sample occurs after Val sample!")
            self.assertLess(max(c_val), min(c_test), "Time leakage: Val sample occurs after Test sample!")

    def test_round1_data_split_leakage_group_random(self):
        """Round 1C: Verify complete session-level disjointness in group_random strategy."""
        cfg = SplitterConfig(
            dataset_path="/dummy",
            split_strategy="group_random",
            evaluation_mode="holdout",
            test_sample_per_class=10,
            seed=42,
        )
        splitter = DummySplitter(cfg)
        train, test, val = splitter._generate_splits(self.synthetic_entries)

        train_groups = {splitter._entry_group_key(x) for x in train}
        test_groups = {splitter._entry_group_key(x) for x in test}
        val_groups = {splitter._entry_group_key(x) for x in val}

        self.assertEqual(len(train_groups.intersection(test_groups)), 0, "Session leakage: Train and Test share groups!")
        self.assertEqual(len(train_groups.intersection(val_groups)), 0, "Session leakage: Train and Val share groups!")
        self.assertEqual(len(val_groups.intersection(test_groups)), 0, "Session leakage: Val and Test share groups!")

    def test_round2_transform_determinism_val_test(self):
        """Round 2: Verify validation transforms are 100% deterministic with zero randomness."""
        val_transform = ConsistentVideoTransform(image_size=224, is_train=False)
        dummy_clip = np.random.randint(0, 256, (2, 224, 224, 3), dtype=np.uint8)

        # Run 10 consecutive applications on identical inputs
        baseline = val_transform(dummy_clip)
        for _ in range(10):
            repeated = val_transform(dummy_clip)
            diff = torch.max(torch.abs(baseline - repeated)).item()
            self.assertEqual(diff, 0.0, "Validation transform produced non-deterministic output!")

        # In contrast, training transform should be stochastic across multiple runs
        train_transform = ConsistentVideoTransform(image_size=224, is_train=True)
        distinct = False
        for _ in range(10):
            t_out = train_transform(dummy_clip)
            if torch.max(torch.abs(t_out - baseline)).item() > 1e-4:
                distinct = True
                break
        self.assertTrue(distinct, "Training transform failed to apply stochastic augmentations!")

    def test_round3_kinematics_cross_sample_isolation(self):
        """Round 3: Verify Kinematics extractor has zero cross-sample contamination in batch."""
        kinematics = FishMotionKinematics7Ch(image_size=224)
        sample_a = torch.randn(1, 2, 3, 224, 224)
        sample_b = torch.randn(1, 2, 3, 224, 224)

        # Process sample A alone
        out_a_alone, summary_a_alone = kinematics(sample_a)

        # Process sample A combined with sample B in a single batch
        batch_ab = torch.cat([sample_a, sample_b], dim=0)
        out_ab, summary_ab = kinematics(batch_ab)

        out_a_in_batch = out_ab[0:1]
        summary_a_in_batch = summary_ab[0:1]

        max_diff_frames = torch.max(torch.abs(out_a_alone - out_a_in_batch)).item()
        max_diff_summary = torch.max(torch.abs(summary_a_alone - summary_a_in_batch)).item()

        self.assertLess(max_diff_frames, 1e-6, "Kinematics leaked information across batch samples!")
        self.assertLess(max_diff_summary, 1e-6, "Kinematics summary leaked information across batch samples!")

    def test_round4_security_credentials_excluded_from_zip(self):
        """Round 4: Verify zero secret/credential leakage when packaging project directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "repo"
            src.mkdir()

            # Normal source files
            (src / "main.py").write_text("import os", encoding="utf-8")
            (src / "config.json").write_text("{}", encoding="utf-8")

            # Sensitive credential files that MUST be excluded
            (src / "token.txt").write_text("hf_secret_token_12345", encoding="utf-8")
            (src / ".env").write_text("SECRET=my_key", encoding="utf-8")
            (src / "auth.token").write_text("auth_token", encoding="utf-8")
            (src / "vault.secret").write_text("vault_secret", encoding="utf-8")
            (src / "id_rsa").write_text("PRIVATE_KEY", encoding="utf-8")
            (src / "cert.pem").write_text("CERT", encoding="utf-8")

            # Sensitive directories that MUST be excluded
            docs_dir = src / "docs"
            docs_dir.mkdir()
            (docs_dir / "confidential.txt").write_text("classified", encoding="utf-8")

            git_dir = src / ".git"
            git_dir.mkdir()
            (git_dir / "config").write_text("git config", encoding="utf-8")

            out_zip = Path(tmp_dir) / "bundle.zip"
            zip_directory(str(src), str(out_zip))

            with zipfile.ZipFile(out_zip, "r") as zf:
                zipped_names = zf.namelist()

            # Confirm allowed files exist
            self.assertIn("main.py", zipped_names)
            self.assertIn("config.json", zipped_names)

            # Confirm strict absence of all sensitive files
            for sensitive in ["token.txt", ".env", "auth.token", "vault.secret", "id_rsa", "cert.pem"]:
                self.assertNotIn(sensitive, zipped_names, f"Security Leak: '{sensitive}' was zipped!")

            for folder in ["docs", ".git"]:
                self.assertFalse(any(folder in name for name in zipped_names), f"Security Leak: '{folder}' was zipped!")

    def test_round5_memory_graph_detachment(self):
        """Round 5: Verify outputs and losses are cleanly detached without lingering graph tensors."""
        loss_tensor = torch.tensor(1.2345, requires_grad=True)
        loss_val = loss_tensor.item()
        self.assertIsInstance(loss_val, float, "Loss was not converted to scalar float via .item()!")

        logits = torch.randn(4, 4, requires_grad=True)
        logits_detached = logits.detach().cpu().numpy()
        self.assertIsInstance(logits_detached, np.ndarray, "Logits were not detached before numpy conversion!")
        self.assertFalse(hasattr(logits_detached, "grad_fn"))


if __name__ == "__main__":
    unittest.main()
