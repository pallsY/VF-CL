import unittest

import torch

from analyze_online_offline_herding_gap import normalized_centroid_error


class OnlineOfflineHerdingGapTests(unittest.TestCase):
    def test_balanced_pair_matches_class_center(self):
        full = torch.tensor([[1., 0.], [1., 0.],
                             [0., 1.], [0., 1.]])
        balanced = torch.tensor([[1., 0.], [0., 1.]])
        one_sided = torch.tensor([[1., 0.], [1., 0.]])
        self.assertAlmostEqual(normalized_centroid_error(full, balanced), 0.0)
        self.assertGreater(normalized_centroid_error(full, one_sided), 0.7)


if __name__ == '__main__':
    unittest.main()
