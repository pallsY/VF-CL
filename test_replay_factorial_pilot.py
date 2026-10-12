import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from calibration_split import manifest_indices
from data_utils import VFLDataset
from run_replay_factorial_pilot import (
    PilotDataset, StopBeforeTest, build_holdout_manifest, derive_config,
    validate_training_only_artifacts,
)


class ReplayFactorialPilotTests(unittest.TestCase):
    def source(self):
        return {
            "seed": 42, "data": "tabvfl", "num_classes": 26,
            "num_tasks": 13, "num_parties": 4,
            "proto_lambda_a": 0.05, "distill_weight": 0.10,
            "feat_distill_weight": 0.02,
            "head_consolidation_enabled": 1,
            "head_consolidation_mode": "adaptive_dual_branch",
            "head_consolidation_samples_per_class": 40,
            "formal_deferred_evaluation": True,
            "vector_npz": "/data/isolet_vfl.npz",
            "party_drift_telemetry": 0,
            "results_dir": "/old", "output_dir": "/old/run",
            "exp_name": "adaptive40",
        }

    def test_four_arms_only_change_two_factors_and_paths(self):
        source = self.source()
        with tempfile.TemporaryDirectory() as folder:
            arms = {
                "P1R1": (1, 1, "adaptive_dual_branch"),
                "P0R1": (0, 1, "adaptive_dual_branch"),
                "P1R0": (1, 0, "full_classifier"),
                "P0R0": (0, 0, "full_classifier"),
            }
            for arm, expected in arms.items():
                config = derive_config(source, Path(folder), "isolet", 54, arm)
                self.assertEqual(
                    (config["proto_replay_gradient_enabled"],
                     config["head_consolidation_enabled"],
                     config["head_consolidation_mode"]), expected)
                self.assertEqual(config["head_consolidation_samples_per_class"], 20)
                self.assertEqual(
                    (config["proto_lambda_a"], config["distill_weight"],
                     config["feat_distill_weight"]), (0.05, 0.10, 0.02))
                self.assertIs(config["formal_deferred_evaluation"], True)
            self.assertEqual(source["head_consolidation_samples_per_class"], 40)

    def test_source_dataset_identity_is_checked(self):
        source = self.source()
        source["vector_npz"] = "/data/upmc_food101_vfl.npz"
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'dataset'):
                derive_config(source, Path(folder), "isolet", 54, "P1R1")

    def test_linux_result_root_stays_linux_when_configured_on_windows(self):
        config = derive_config(
            self.source(), "/home/runner/results/replay-factorial",
            "isolet", 54, "P1R1")
        self.assertEqual(
            config["output_dir"],
            "/home/runner/results/replay-factorial/isolet_seed54_P1R1")

    def test_holdout_excludes_gate_and_calibration(self):
        labels = [0] * 100 + [1] * 100
        excluded = set(range(10)) | set(range(100, 110))
        manifest, indices = build_holdout_manifest(
            labels, excluded, 10, 20261013, "toy")
        self.assertEqual(indices, manifest_indices(manifest))
        self.assertEqual(len(indices), 20)
        self.assertFalse(indices & excluded)
        self.assertEqual(sum(i < 100 for i in indices), 10)

    def test_test_authorization_stops_before_real_access(self):
        dataset = PilotDataset.__new__(PilotDataset)
        with mock.patch.object(VFLDataset, "authorize_formal_access",
                               side_effect=AssertionError("test accessed")):
            with self.assertRaises(StopBeforeTest):
                dataset.authorize_formal_access(
                    split="test", phase="final_test_post_install",
                    event_idx=1, task_id=1, timeline_step="event_1_CIL",
                    classes=[0, 1])

    def test_artifact_gate_rejects_any_formal_test_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / 'checkpoints').mkdir()
            (run / 'checkpoints' / 'formal_final.pt').write_bytes(b'checkpoint')
            (run / 'FORMAL_STATE_FROZEN.json').write_text('{}')
            (run / 'FORMAL_EVALUATION_PENDING.json').write_text('{}')
            (run / 'data_flow_audit.jsonl').write_text(
                '{"split":"train"}\n{"split":"validation"}\n')
            validate_training_only_artifacts(run)
            (run / 'FORMAL_EVALUATION_CONSUMING.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'test access'):
                validate_training_only_artifacts(run)

    def test_holdout_readout_uses_deterministic_validation_view(self):
        dataset = PilotDataset.__new__(PilotDataset)
        dataset.trainset = object()
        dataset.validationset = object()
        dataset.holdout_indices = {2, 3}
        dataset.get_class_indices = lambda _view, _classes: [1, 2, 3]
        seen = []
        dataset._loader = lambda view, indices, key, shuffle: seen.append(
            (view, indices, key, shuffle))
        dataset.holdout_loader([0, 1])
        self.assertIs(seen[0][0], dataset.validationset)
        self.assertEqual(seen[0][1], [2, 3])
