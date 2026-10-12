import unittest

import torch
from types import SimpleNamespace

from cl_methods.hpcr import HPCRCL, proxy_supcon_loss, similarity_distillation_loss
from models import TopModel


class HPCRLossTest(unittest.TestCase):
    def test_first_unreplayed_batch_only_populates_memory(self):
        bottom = torch.nn.Linear(1, 2)
        top = TopModel(2, 2)
        trainer = SimpleNamespace(
            bottoms=[bottom], top_model=top,
            _aggregate=lambda values: values[0],
            bytes_transmitted=0, comm_rounds=0,
        )
        args = SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 1)],
            device='cpu', num_classes=2, er_per_class=2, er_batch=2,
        )
        method = HPCRCL(trainer, args)
        loss = method._hpcr_loss(
            [bottom], top, torch.tensor([[1.0], [2.0]]),
            torch.tensor([0, 1]))
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertEqual(sum(map(len, method.memory.values())), 2)

    def test_proxy_contrastive_gradient_reaches_features_and_proxies(self):
        features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
                                requires_grad=True)
        proxies = torch.tensor([[1.0, 0.0], [0.0, 1.0]],
                               requires_grad=True)
        loss = proxy_supcon_loss(features, proxies, torch.tensor([0, 0, 1]),
                                 temperature=0.09)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(features.grad.abs().sum()), 0.0)
        self.assertGreater(float(proxies.grad.abs().sum()), 0.0)

    def test_similarity_distillation_is_stationary_at_stored_geometry(self):
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
                                requires_grad=True)
        loss = similarity_distillation_loss(features, features.detach(),
                                            temperature=0.09)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(features.grad.abs().max()), 1e-4)

    def test_buffer_caps_examples_and_restores_stored_targets(self):
        args = SimpleNamespace(er_per_class=2, er_batch=2, num_classes=3)
        first = HPCRCL(SimpleNamespace(), args)
        first._insert(torch.arange(3.0)[:, None], torch.tensor([0, 0, 0]),
                      torch.ones(3, 2), torch.arange(6.0).reshape(3, 2))
        self.assertEqual(len(first.memory[0]), 2)
        self.assertEqual(first.seen_count[0], 3)
        self.assertTrue(torch.isnan(first.memory[0][0][2][2]))
        second = HPCRCL(SimpleNamespace(), args)
        second.load_state(first.get_state())
        self.assertEqual(len(second.memory[0]), 2)
        torch.testing.assert_close(second.memory[0][0][3],
                                   first.memory[0][0][3])


if __name__ == '__main__':
    unittest.main()
