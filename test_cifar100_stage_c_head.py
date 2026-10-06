import unittest

import torch

from cifar100_stage_c_head import (
    candidate_name,
    candidate_specs,
    hierarchical_task_scores,
    parse_candidate,
)


class StageCHeadTests(unittest.TestCase):
    def test_candidate_grid_is_unique_and_contains_incumbent(self):
        names = [candidate_name(*spec) for spec in candidate_specs()]
        self.assertEqual(len(names), 96)
        self.assertEqual(len(set(names)), len(names))
        self.assertIn("tcb_c0.001_t0_g1", names)

    def test_candidate_name_round_trips(self):
        values = (0.0003, 0.001, 1.15)
        self.assertEqual(parse_candidate(candidate_name(*values)), values)

    def test_unit_task_weight_preserves_predictions(self):
        torch.manual_seed(7)
        scores = torch.randn(25, 6)
        task_classes = {0: [0, 1, 2], 1: [3, 4, 5]}
        adjusted = hierarchical_task_scores(scores, task_classes, 1.0)
        self.assertTrue(torch.equal(scores.argmax(1), adjusted.argmax(1)))

    def test_task_reweighting_preserves_within_task_predictions(self):
        torch.manual_seed(11)
        scores = torch.randn(20, 6)
        task_classes = {0: [0, 1, 2], 1: [3, 4, 5]}
        adjusted = hierarchical_task_scores(scores, task_classes, 1.5)
        for classes in task_classes.values():
            self.assertTrue(torch.equal(
                scores[:, classes].argmax(1),
                adjusted[:, classes].argmax(1),
            ))


if __name__ == "__main__":
    unittest.main()
