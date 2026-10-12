import unittest

import torch
from types import SimpleNamespace

from cl_methods.ocar import OCARCL, kfac_precondition
from models import TopModel


class OCARKFACTest(unittest.TestCase):
    def test_low_rank_kfac_matches_dense_kronecker_factor_solve(self):
        activations = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        output_covariance = torch.tensor([[2.0, 0.2], [0.2, 1.0]])
        gradient = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        damping = 0.5
        input_covariance = activations.T @ activations / len(activations)
        expected = torch.linalg.solve(
            output_covariance + damping * torch.eye(2), gradient)
        expected = torch.linalg.solve(
            input_covariance + damping * torch.eye(2), expected.T).T
        actual = kfac_precondition(
            gradient, activations, output_covariance, damping)
        torch.testing.assert_close(actual, expected)

    def test_true_fisher_factors_cover_convolution_and_restore(self):
        convolution = torch.nn.Conv2d(3, 4, 3, padding=1, bias=False)
        bottom = torch.nn.Sequential(
            convolution, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        trainer = SimpleNamespace(
            bottoms=[bottom], top_model=TopModel(4, 2),
            _aggregate=lambda values: values[0],
            bytes_transmitted=0, comm_rounds=0,
        )
        args = SimpleNamespace(
            data='cifar100', num_parties=1, img_size=32, device='cpu',
            lr=0.01, er_per_class=2, er_batch=2, batch_size=2,
            ocar_ema=0.1, ocar_delta_tau=1e-5, ocar_max_rows=8,
        )
        method = OCARCL(trainer, args)
        method.current_batch = torch.randn(2, 3, 32, 32)
        method._update_factors()
        self.assertIn(convolution, method.factors)
        rows, covariance = method.factors[convolution]
        self.assertEqual(rows.shape, (8, 27))
        self.assertEqual(covariance.shape, (4, 4))
        self.assertTrue(torch.isfinite(covariance).all())
        restored = OCARCL(trainer, args)
        restored.load_state(method.get_state())
        torch.testing.assert_close(restored.factors[convolution][1], covariance)


if __name__ == '__main__':
    unittest.main()
