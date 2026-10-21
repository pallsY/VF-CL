import unittest

import numpy as np
import torch

from sampleonly_train_replay import BalancedOldReplay


class BalancedOldReplayTests(unittest.TestCase):
    def test_exact_64_and_independent_rng(self):
        data = {class_id: torch.full((20, 3), float(class_id))
                for class_id in range(3)}
        np.random.seed(123)
        before = np.random.get_state()
        first = BalancedOldReplay(data, seed=9)
        batch, labels, duplicates = first.sample()
        self.assertEqual(batch.shape, (64, 3))
        self.assertEqual(labels.bincount(minlength=3).tolist(), [22, 21, 21])
        self.assertGreater(duplicates, 0)
        second = BalancedOldReplay(data, seed=9)
        other, other_labels, other_duplicates = second.sample()
        self.assertTrue(torch.equal(batch, other))
        self.assertTrue(torch.equal(labels, other_labels))
        self.assertEqual(duplicates, other_duplicates)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        self.assertTrue(np.array_equal(before[1], after[1]))
        self.assertEqual(before[2:], after[2:])

    def test_more_old_classes_than_batch_slots_rotates(self):
        data = {class_id: torch.full((20, 2), float(class_id))
                for class_id in range(90)}
        sampler = BalancedOldReplay(data, seed=11)
        seen = set()
        for _ in range(90):
            batch, labels, _ = sampler.sample()
            self.assertEqual(len(batch), 64)
            self.assertEqual(len(labels), 64)
            self.assertEqual(len(set(labels.tolist())), 64)
            seen.update(labels.tolist())
        self.assertEqual(seen, set(range(90)))

    def test_rejects_empty_class(self):
        with self.assertRaisesRegex(ValueError, 'empty'):
            BalancedOldReplay({0: torch.empty((0, 2))}, seed=1)


if __name__ == '__main__':
    unittest.main()
