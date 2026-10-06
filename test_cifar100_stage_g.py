import json
import tempfile
import unittest
from pathlib import Path

import torch

from cifar100_stage_f import stage_b_jobs as stage_f_b_jobs
from cifar100_stage_g import CANDIDATES, expected_config, stage_a_jobs
from cl_methods.proto_evolve import effective_supcon_weight, supervised_contrastive_loss


class SupervisedContrastiveTests(unittest.TestCase):
    def test_late_task_weight_schedule(self):
        actual = [effective_supcon_weight(0.05, 5, task) for task in range(10)]
        self.assertEqual(actual, [0.0] * 5 + [0.05] * 5)

    def test_loss_is_zero_without_positive_pairs(self):
        features = torch.eye(4, requires_grad=True)
        loss = supervised_contrastive_loss(features, torch.arange(4))
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_clustered_same_class_pairs_have_lower_loss(self):
        labels = torch.tensor([0, 0, 1, 1])
        clustered = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
        crossed = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]])
        self.assertLess(
            float(supervised_contrastive_loss(clustered, labels)),
            float(supervised_contrastive_loss(crossed, labels)),
        )

    def test_loss_backward_is_finite(self):
        features = torch.randn(12, 8, requires_grad=True)
        labels = torch.arange(12) // 3
        loss = supervised_contrastive_loss(features, labels, 0.1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(features.grad).all())


class StageGProtocolTests(unittest.TestCase):
    def test_stage_a_has_three_seed_42_jobs(self):
        self.assertEqual(stage_a_jobs(), [f"{candidate}:42" for candidate in CANDIDATES])

    def test_candidates_only_change_supcon_weight(self):
        low = expected_config("supcon_002:42")
        high = expected_config("supcon_010:42")
        self.assertEqual(low["distill_weight_schedule"], "stage_f_decay_005")
        self.assertEqual(low["current_supcon_start_task"], 5)
        self.assertEqual(low["current_supcon_temperature"], 0.1)
        changed = {key for key in low if low[key] != high[key]}
        self.assertEqual(changed, {"current_supcon_weight"})

    def test_stage_f_empty_promotion_has_no_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps({
                "selection_source": "cifar100-train-validation",
                "test_used_for_selection": False,
                "promoted_candidates": [],
            }), encoding="utf-8")
            self.assertEqual(stage_f_b_jobs(path), [])


if __name__ == "__main__":
    unittest.main()
