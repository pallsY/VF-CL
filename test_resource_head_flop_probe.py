"""Synthetic head inputs match frozen class and sample dimensions."""

import unittest

from resource_head_flop_probe import synthetic_replay_embeddings


class HeadFlopProbeTests(unittest.TestCase):
    def test_each_class_has_only_the_declared_raw_replay_count(self):
        replay = synthetic_replay_embeddings([0, 1, 2], 20, 4)
        self.assertEqual(sorted(replay), [0, 1, 2])
        self.assertEqual([tuple(value.shape) for value in replay.values()],
                         [(20, 4)] * 3)
        self.assertTrue(all(value.is_floating_point()
                            for value in replay.values()))

    def test_empty_or_duplicate_class_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'class'):
            synthetic_replay_embeddings([], 20, 4)
        with self.assertRaisesRegex(ValueError, 'class'):
            synthetic_replay_embeddings([0, 0], 20, 4)


if __name__ == '__main__':
    unittest.main()
