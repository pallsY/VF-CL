import unittest

import torch

from party_balanced_herding_dev import party_balanced_features


class PartyBalancedFeaturesTests(unittest.TestCase):
    def test_equalizes_per_party_row_norm(self):
        first = torch.tensor([[100.0, 0.0], [0.0, 200.0]])
        second = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
        features = party_balanced_features([first, second])
        self.assertEqual(features.shape, (2, 4))
        self.assertTrue(torch.allclose(features[:, :2].norm(dim=1),
                                       features[:, 2:].norm(dim=1)))

    def test_rejects_mismatched_rows(self):
        with self.assertRaisesRegex(ValueError, 'party embeddings'):
            party_balanced_features([
                torch.zeros(2, 2), torch.zeros(3, 2)])


if __name__ == '__main__':
    unittest.main()
