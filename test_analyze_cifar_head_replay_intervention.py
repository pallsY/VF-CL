import unittest

import torch

from analyze_cifar_head_replay_intervention import readout_metrics


class HeadReplayReadoutTests(unittest.TestCase):
    def test_task_il_can_be_correct_when_class_il_is_wrong(self):
        logits = torch.full((2, 100), -10.0)
        logits[0, 0] = 2.0
        logits[0, 90] = 3.0
        logits[1, 90] = 4.0
        log_probabilities = logits.log_softmax(dim=1)
        result = readout_metrics(log_probabilities, torch.tensor([0, 90]))
        self.assertEqual(result['cil']['correct'], 1)
        self.assertEqual(result['til']['correct'], 2)
        self.assertEqual(result['old_cil']['correct'], 0)
        self.assertEqual(result['new_cil']['correct'], 1)
        self.assertEqual(result['per_class']['0']['total'], 1)
        self.assertAlmostEqual(result['nll']['sum'],
                               float(-log_probabilities[0, 0] - log_probabilities[1, 90]))

    def test_rejects_malformed_probabilities_and_labels(self):
        with self.assertRaisesRegex(ValueError, 'shape'):
            readout_metrics(torch.zeros(2, 99), torch.tensor([0, 90]))
        with self.assertRaisesRegex(ValueError, 'label'):
            readout_metrics(torch.zeros(2, 100), torch.tensor([0, 100]))


if __name__ == '__main__':
    unittest.main()
