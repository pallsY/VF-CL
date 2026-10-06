import unittest
import os
import tempfile

import numpy as np

from analyze_output_bias import (
    deterministic_environment_mismatches,
    final_checkpoint_path,
    task_logit_summary,
    task_parameter_summary,
    validate_sum_decomposition,
    validate_run_contract,
)


class OutputBiasHelpersTest(unittest.TestCase):
    def test_deterministic_environment_reports_each_missing_contract_value(self):
        expected = {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONHASHSEED": "42",
        }
        self.assertEqual(deterministic_environment_mismatches(42, expected), [])

        errors = deterministic_environment_mismatches(42, {})

        self.assertEqual(len(errors), 4)
        self.assertTrue(any("CUBLAS_WORKSPACE_CONFIG" in error for error in errors))

    def test_final_checkpoint_path_requires_full_model_state(self):
        with tempfile.TemporaryDirectory() as run_dir:
            audit_dir = os.path.join(run_dir, "party_kd_audit")
            os.makedirs(audit_dir)
            open(os.path.join(audit_dir, "event_9_CIL.pt"), "wb").close()

            with self.assertRaisesRegex(FileNotFoundError, "full CIL checkpoint"):
                final_checkpoint_path(run_dir)

            checkpoint_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(checkpoint_dir)
            expected = os.path.join(checkpoint_dir, "event_9_CIL.pt")
            open(expected, "wb").close()

            self.assertEqual(final_checkpoint_path(run_dir), expected)

    def test_parameter_summary_groups_classifier_rows_by_task(self):
        weight = np.array([[3.0, 4.0], [0.0, 2.0], [6.0, 8.0], [0.0, 1.0]])
        actual = task_parameter_summary(
            weight, np.array([1.0, 3.0, 5.0, 7.0]), classes_per_task=2
        )

        self.assertEqual(actual["mean_row_norm"], [3.5, 5.5])
        self.assertEqual(actual["mean_classifier_bias"], [2.0, 6.0])

    def test_decomposition_uses_classifier_bias_and_rejects_error(self):
        party = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        full = party.sum(axis=1) + np.array([[0.5, -0.5]])

        validate_sum_decomposition(full, party, np.array([0.5, -0.5]))

        with self.assertRaisesRegex(ValueError, "decomposition mismatch"):
            validate_sum_decomposition(full, party, np.zeros(2))

    def test_decomposition_accepts_fp32_accumulation_scale_roundoff(self):
        party = np.full((1, 4, 1), 1.0e-6, dtype=np.float32)
        error = validate_sum_decomposition(
            np.zeros((1, 1), dtype=np.float32), party, np.zeros(1, dtype=np.float32)
        )

        self.assertAlmostEqual(error, 4.0e-6, places=11)

    def test_party_summary_is_grouped_by_true_task(self):
        labels = np.array([0, 1, 2, 3])
        party = np.array(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]],
                [[0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 3.0]],
                [[0.0, 0.0, 4.0, 0.0], [0.0, 0.0, 0.0, 4.0]],
            ]
        )

        actual = task_logit_summary(labels, party.sum(axis=1), party, classes_per_task=2)

        self.assertEqual(actual["mean_party_logit_by_true_task"][0], [0.375, 0.375])
        self.assertEqual(actual["mean_party_logit_by_true_task"][1], [0.875, 0.875])

    def test_run_contract_rejects_wrong_aggregation(self):
        config = {
            "deterministic": 1,
            "data": "cifar100",
            "num_tasks": 10,
            "classes_per_task": 10,
            "num_parties": 4,
            "model_type": "resnet18",
            "aggregation": "sum",
            "cosine_head": False,
        }

        validate_run_contract(config)
        config["aggregation"] = "concat"

        with self.assertRaisesRegex(ValueError, "aggregation"):
            validate_run_contract(config)


if __name__ == "__main__":
    unittest.main()
