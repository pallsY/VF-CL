import unittest

import torch

from analyze_isolet_head_branches import score_branch


class ScoreBranchTests(unittest.TestCase):
    def test_separates_cross_task_and_within_task_errors(self):
        task_classes = {0: [0, 2], 1: [1, 3]}
        labels = torch.tensor([0, 2, 1, 3])
        logits = torch.tensor([
            [5.0, 0.0, 1.0, 0.0],
            [0.0, 5.0, 4.0, 0.0],
            [0.0, 3.0, 0.0, 5.0],
            [0.0, 1.0, 0.0, 5.0],
        ], dtype=torch.float64)
        scores = score_branch(torch.log_softmax(logits, dim=1), labels, task_classes)
        self.assertEqual(scores['cil'], 0.5)
        self.assertEqual(scores['task_id'], 0.75)
        self.assertEqual(scores['task_il'], 0.75)
        self.assertEqual(scores['old_cil'], 0.5)
        self.assertEqual(scores['new_cil'], 0.5)
        self.assertEqual(scores['old_to_new_rate'], 0.5)
        self.assertEqual(scores['new_to_old_rate'], 0.0)
        self.assertEqual(scores['task_confusion'], [[1, 1], [0, 2]])


if __name__ == '__main__':
    unittest.main()
