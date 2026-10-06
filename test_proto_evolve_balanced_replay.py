import unittest

import torch

from cl_methods.proto_evolve import reduce_proto_replay_loss


class ProtoEvolveBalancedReplayTests(unittest.TestCase):
    def test_sample_mean_is_invariant_to_old_class_count(self):
        losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
        first = reduce_proto_replay_loss(losses, 10, "sample_mean")
        later = reduce_proto_replay_loss(losses, 90, "sample_mean")
        self.assertEqual(float(first), 2.5)
        self.assertEqual(float(later), 2.5)

    def test_legacy_mode_preserves_historical_scaling(self):
        losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
        actual = reduce_proto_replay_loss(
            losses, 5, "legacy_class_normalized"
        )
        self.assertEqual(float(actual), 2.0)

    def test_party_weights_keep_weighted_sample_mean(self):
        losses = torch.tensor([1.0, 2.0, 3.0])
        weights = torch.tensor([0.5, 1.0, 1.5])
        actual = reduce_proto_replay_loss(
            losses, 20, "sample_mean", class_weights=weights
        )
        self.assertAlmostEqual(float(actual), 7.0 / 3.0, places=6)

    def test_invalid_reduction_is_rejected(self):
        with self.assertRaises(ValueError):
            reduce_proto_replay_loss(torch.ones(2), 10, "unknown")


if __name__ == "__main__":
    unittest.main()
