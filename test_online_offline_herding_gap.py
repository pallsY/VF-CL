import unittest

import torch

from analyze_online_offline_herding_gap import normalized_centroid_error, summarize_rows


class OnlineOfflineHerdingGapTests(unittest.TestCase):
    def test_balanced_pair_matches_class_center(self):
        full = torch.tensor([[1., 0.], [1., 0.],
                             [0., 1.], [0., 1.]])
        balanced = torch.tensor([[1., 0.], [0., 1.]])
        one_sided = torch.tensor([[1., 0.], [1., 0.]])
        self.assertAlmostEqual(normalized_centroid_error(full, balanced), 0.0)
        self.assertGreater(normalized_centroid_error(full, one_sided), 0.7)

    def test_summary_preserves_paired_party_comparison(self):
        rows = [{
            'online_aggregate_error': 2.0,
            'offline_aggregate_error': 1.0,
            'party_online_errors': [2.0, 1.0, 1.0, 1.0],
            'party_offline_errors': [1.0, 2.0, 1.0, 1.0],
        } for _ in range(100)]
        summary = summarize_rows(rows)
        self.assertEqual(summary['mean_difference'], 1.0)
        self.assertEqual(summary['party_mean_differences'], [1.0, -1.0, 0.0, 0.0])
        self.assertEqual(summary['party_fraction_online_worse'], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(summary['party_online_medians'], [2.0, 1.0, 1.0, 1.0])


if __name__ == '__main__':
    unittest.main()
