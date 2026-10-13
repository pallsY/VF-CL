import unittest

import torch
from types import SimpleNamespace

from cl_methods.balanced_contrast import (
    BalancedContrastCL, batch_balance_weights, proxy_contrastive_loss,
)


class BalancedContrastTest(unittest.TestCase):
    def test_paper_weighted_proxy_loss_matches_analytic_zero_logits(self):
        features = torch.zeros(3, 2, requires_grad=True)
        proxies = torch.zeros(2, 2, requires_grad=True)
        labels = torch.tensor([0, 0, 1])
        loss = proxy_contrastive_loss(
            features, proxies, labels,
            task_number=1, gamma=1.0, temperature=1.0,
        )
        expected = torch.tensor(2.0 / 3.0) * torch.log(torch.tensor(3.0))
        torch.testing.assert_close(loss, expected)

    def test_rare_class_gets_larger_attraction_weight(self):
        labels = torch.tensor([0, 0, 0, 1])
        weights = batch_balance_weights(labels, task_number=2, gamma=4.0)
        self.assertAlmostEqual(float(weights[0]), 3 ** (-1 / 8))
        self.assertEqual(float(weights[-1]), 1.0)
        self.assertGreater(weights[-1], weights[0])

    def test_proxy_loss_has_finite_gradient_for_features_and_proxies(self):
        features = torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 2.0]],
                                requires_grad=True)
        proxies = torch.tensor([[1.0, 0.0], [0.0, 1.0]],
                               requires_grad=True)
        labels = torch.tensor([0, 0, 1])
        loss = proxy_contrastive_loss(features, proxies, labels,
                                      task_number=1, gamma=4.0,
                                      temperature=1.0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertTrue(torch.isfinite(proxies.grad).all())
        self.assertGreater(float(proxies.grad.abs().sum()), 0.0)

    def test_checkpoint_restores_task_weighting_stage(self):
        args = SimpleNamespace(er_per_class=2, er_batch=2, batch_size=2,
                               balanced_gamma=4.0, balanced_temperature=0.1)
        first = BalancedContrastCL(SimpleNamespace(), args)
        first.task_number = 3
        second = BalancedContrastCL(SimpleNamespace(), args)
        second.load_state(first.get_state())
        self.assertEqual(second.task_number, 3)


if __name__ == '__main__':
    unittest.main()
