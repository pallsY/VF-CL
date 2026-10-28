import unittest

import torch

from diagnose_party_kd_fim_readonly import drift_metrics, frozen_fraction


class ReadOnlyDiagnosticMathTests(unittest.TestCase):
    def test_identical_embeddings_have_zero_drift_and_no_scatter_change(self):
        old = torch.tensor([[2.0, 0.0], [2.2, 0.0], [0.0, 2.0], [0.0, 2.2]])
        labels = torch.tensor([0, 0, 1, 1])
        result = drift_metrics(old, old.clone(), labels)
        self.assertAlmostEqual(result['relative_l2_mean'], 0.0)
        self.assertAlmostEqual(result['cosine_mean'], 1.0)
        self.assertAlmostEqual(result['scatter_log_change'], 0.0)

    def test_frozen_fraction_uses_parameter_count(self):
        state = {'weight': torch.zeros(10, 10), 'bias': torch.zeros(1)}
        self.assertAlmostEqual(frozen_fraction(state, {'weight': False, 'bias': True}), 1 / 101)


if __name__ == '__main__':
    unittest.main()
