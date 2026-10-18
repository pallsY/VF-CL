import unittest

import torch
import torch.nn.functional as F

from development_task_bias import apply_task_bias, fit_task_bias, fit_and_score


class TaskBiasTests(unittest.TestCase):
    def test_fits_task_offsets_without_truth_at_inference(self):
        logits = torch.tensor([[2.0, 0.0], [1.2, 1.0]], dtype=torch.float64)
        labels = torch.tensor([0, 1])
        task_classes = {0: [0], 1: [1]}
        offsets = fit_task_bias(logits, labels, task_classes)
        calibrated = apply_task_bias(logits, offsets, task_classes)
        self.assertLess(F.cross_entropy(calibrated, labels),
                        F.cross_entropy(logits, labels))
        self.assertEqual(calibrated.argmax(1).tolist(), [0, 1])

    def test_holdout_labels_cannot_change_fitted_offsets(self):
        validation = torch.tensor([[2.0, 0.0], [1.2, 1.0]],
                                  dtype=torch.float64)
        holdout = torch.tensor([[0.5, 0.4], [0.3, 0.8]],
                               dtype=torch.float64)
        tasks = {0: [0], 1: [1]}
        first = fit_and_score(validation, torch.tensor([0, 1]),
                              holdout, torch.tensor([0, 1]), tasks)
        second = fit_and_score(validation, torch.tensor([0, 1]),
                               holdout, torch.tensor([1, 0]), tasks)
        self.assertTrue(torch.equal(first['offsets'], second['offsets']))


if __name__ == '__main__':
    unittest.main()
