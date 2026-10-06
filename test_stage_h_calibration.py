import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from stage_h_calibration import (
    CANDIDATES,
    cross_fit,
    final_checkpoint_for_config,
    metrics_with_replaced_final,
    select_candidate,
    stratified_fold_ids,
    task_classes_from_config,
)


class StageHCalibrationTests(unittest.TestCase):
    def test_custom_tasks_support_nonuniform_final_task(self):
        config = {
            "custom_tasks": "0,1,2|3,4|5,6",
            "num_classes": 7,
            "num_tasks": 3,
            "classes_per_task": 3,
        }
        self.assertEqual(task_classes_from_config(config), {
            0: [0, 1, 2], 1: [3, 4], 2: [5, 6],
        })

    def test_final_checkpoint_uses_configured_task_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoints = Path(tmp) / "checkpoints"
            checkpoints.mkdir()
            expected = checkpoints / "event_12_CIL.pt"
            expected.touch()
            (checkpoints / "event_9_CIL.pt").touch()
            actual = final_checkpoint_for_config(tmp, {"num_tasks": 13})
            self.assertEqual(actual, expected)

    def test_stratified_folds_are_complete_balanced_and_deterministic(self):
        labels = np.repeat(np.arange(4), 9)
        first = stratified_fold_ids(labels)
        second = stratified_fold_ids(labels)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(set(first.tolist()), {0, 1})
        for class_id in range(4):
            counts = [
                int(np.sum((labels == class_id) & (first == fold)))
                for fold in (0, 1)
            ]
            self.assertLessEqual(abs(counts[0] - counts[1]), 1)

    def test_candidate_budget_is_fixed_unique_and_includes_identity(self):
        names = [item["name"] for item in CANDIDATES]
        self.assertEqual(len(names), 8)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[0], "identity")

    def test_identity_cross_fit_reproduces_scores(self):
        torch.manual_seed(7)
        logits = torch.randn(40, 4)
        labels = torch.tensor([0, 1, 2, 3] * 10)
        folds = stratified_fold_ids(labels.numpy())
        scores, fits = cross_fit(
            CANDIDATES[0], logits, labels,
            {0: [0, 1], 1: [2, 3]}, folds,
        )
        self.assertTrue(torch.equal(scores, logits))
        self.assertEqual(len(fits), 2)

    def test_final_replacement_uses_repository_bwt_definition(self):
        results = {"task_acc_history": [
            {
                "step": "event_0_CIL",
                "per_task_accs": {"task_0": 0.8},
                "per_task_accs_taskil": {"task_0": 0.9},
                "overall_acc": 0.8,
            },
            {
                "step": "event_1_CIL",
                "per_task_accs": {"task_0": 0.7, "task_1": 0.8},
                "per_task_accs_taskil": {"task_0": 0.9, "task_1": 0.9},
                "overall_acc": 0.75,
            },
        ]}
        final = {
            "overall_accuracy": 0.85,
            "per_task_accuracy": {"task_0": 0.8, "task_1": 0.9},
            "task_il": {"task_0": 0.95, "task_1": 0.95},
        }
        metrics = metrics_with_replaced_final(results, final)
        self.assertEqual(metrics["AA_final"], 0.85)
        self.assertEqual(metrics["BWT"], 0.0)
        self.assertEqual(metrics["AA_final_taskil"], 0.95)

    def test_selection_applies_bwt_and_task_il_before_accuracy(self):
        records = [
            {"candidate": "bad_bwt", "metrics": {
                "AA_final": 0.95, "BWT": 0.1, "AA_final_taskil": 0.95,
            }},
            {"candidate": "bad_task_il", "metrics": {
                "AA_final": 0.94, "BWT": 0.13, "AA_final_taskil": 0.90,
            }},
            {"candidate": "eligible", "metrics": {
                "AA_final": 0.86, "BWT": 0.13, "AA_final_taskil": 0.95,
            }},
        ]
        selection = select_candidate(
            records,
            {"er_bwt": 0.12, "der_pp_aa_final": 0.85},
            raw_task_il=0.95,
        )
        self.assertEqual(selection["selected_candidate"], "eligible")
        self.assertTrue(selection["passed"])


if __name__ == "__main__":
    unittest.main()
