"""Numerical regression tests for AdaGauss inference."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from cl_methods.adagauss import AdaGaussCL, _Adapter


class AdaGaussNumericsTests(unittest.TestCase):
    @staticmethod
    def _args(**changes):
        values = {
            'device': 'cpu', 'epochs_per_task': 3,
            'adagauss_lambda_ac': .2, 'adagauss_lambda_pkd': 1.,
            'adagauss_shrinkage': .1, 'adagauss_adapter_epochs': 1,
            'adagauss_n_samples': 4, 'data': 'synthvfl',
            'party_col_ranges': [(0, 2)],
        }
        values.update(changes)
        return SimpleNamespace(**values)

    def test_train_task_requests_official_unit_gradient_clip(self):
        trainer = Mock()
        trainer.top_model.classifier.in_features = 2
        trainer.train_task.return_value = ([], 0.0)
        method = AdaGaussCL(trainer, self._args())

        method.train_task([], 1)

        trainer.train_task.assert_called_once_with(
            [], 3, extra_loss_fn=method._adagauss_loss,
            grad_clip_norm=1.0,
        )

    def test_adapter_rejects_nonfinite_loss_before_parameter_mutation(self):
        trainer = SimpleNamespace(
            bottoms=[torch.nn.Identity()],
            top_model=SimpleNamespace(
                classifier=SimpleNamespace(in_features=2)),
            _aggregate=lambda parts: parts[0],
            evaluate=Mock(),
        )
        method = AdaGaussCL(trainer, self._args())
        method.old_bottoms = [torch.nn.Identity()]
        adapter = _Adapter(2)
        before = [parameter.detach().clone()
                  for parameter in adapter.parameters()]
        loader = [(
            torch.tensor([[float('nan'), 0.0], [1.0, 2.0]]),
            torch.tensor([0, 1]),
        )]

        with patch('cl_methods.adagauss._Adapter', return_value=adapter), \
                self.assertRaisesRegex(FloatingPointError, 'non-finite'):
            method._train_adapter(loader)

        self.assertTrue(all(
            torch.equal(left, right)
            for left, right in zip(before, adapter.parameters())
        ))

    def test_anti_collapse_factors_covariance_in_float64(self):
        method = AdaGaussCL.__new__(AdaGaussCL)
        features = torch.randn(64, 512, requires_grad=True)
        original = torch.linalg.cholesky
        seen = []

        def platform_sensitive(matrix):
            seen.append(matrix.dtype)
            if matrix.dtype == torch.float32:
                return torch.full_like(matrix, float('nan'))
            return original(matrix)

        with patch(
                'cl_methods.adagauss.torch.linalg.cholesky',
                side_effect=platform_sensitive):
            loss = method._loss_ac(features)

        self.assertEqual([torch.float64], seen)
        self.assertEqual(torch.float32, loss.dtype)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_scale_aware_shrink_repairs_large_covariance(self):
        method = AdaGaussCL.__new__(AdaGaussCL)
        method.shrinkage = .1
        covariance = torch.diag(torch.tensor(
            [5e5, 2e5, -0.15], dtype=torch.float32))

        repaired = method._shrink(covariance)

        expected_shift = .1 * torch.diagonal(covariance.double()).mean()
        self.assertTrue(torch.allclose(
            repaired.double(),
            covariance.double() + expected_shift * torch.eye(3),
            rtol=1e-6, atol=1e-3,
        ))
        self.assertGreater(
            float(torch.linalg.eigvalsh(repaired.double()).min()), 0)

    def test_old_gaussian_adaptation_repairs_scaled_indefinite_covariance(self):
        method = AdaGaussCL.__new__(AdaGaussCL)
        method.args = SimpleNamespace(device='cpu')
        method.shrinkage = .1
        method.n_samples = 32
        method.dim = 3
        method.gaussians = {0: {
            'mean': torch.zeros(3),
            'cov': torch.diag(torch.tensor([5e5, 2e5, -0.15])),
        }}

        method._adapt_old_gaussians(torch.nn.Identity())

        covariance = method.gaussians[0]['cov'].double()
        self.assertTrue(torch.isfinite(covariance).all())
        self.assertGreater(float(torch.linalg.eigvalsh(covariance).min()), 0)

    def test_bayes_evaluation_factors_ill_conditioned_positive_covariance(self):
        original_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            samples = torch.randn(
                2, 8, generator=torch.Generator().manual_seed(0),
            ) * 1000.0
            covariance = 0.9 * torch.cov(samples.T) + 0.1 * torch.eye(8)
            self.assertGreater(torch.linalg.eigvalsh(covariance.double()).min(), 0)
            original_covariance = covariance.clone()

            trainer = SimpleNamespace(
                bottoms=[torch.nn.Identity()],
                top_model=torch.nn.Identity(),
                _aggregate=lambda parts: parts[0],
            )
            method = AdaGaussCL.__new__(AdaGaussCL)
            method.trainer = trainer
            method.args = SimpleNamespace(
                data='synthvfl', party_col_ranges=[(0, 8)],
                device='cpu', num_classes=1,
            )
            method.dim = 8
            method.gaussians = {0: {
                'mean': torch.zeros(8), 'cov': covariance,
            }}
            values = [(torch.zeros(2, 8), torch.zeros(2, dtype=torch.long))]

            accuracy, probabilities, labels = method._bayes_evaluate(values)

            self.assertEqual(1.0, accuracy)
            self.assertTrue(torch.isfinite(probabilities).all())
            self.assertEqual((2, 1), tuple(probabilities.shape))
            self.assertEqual([0, 0], labels.tolist())
            self.assertTrue(torch.equal(covariance, original_covariance))
        finally:
            torch.set_num_threads(original_threads)


if __name__ == '__main__':
    unittest.main()
