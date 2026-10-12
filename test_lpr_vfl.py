import unittest

import torch
from types import SimpleNamespace

from cl_methods.lpr import LPRCL, precondition_affine_gradient
from models import TopModel
from vfl_trainer import VFLTrainer


class LPRPreconditionerTest(unittest.TestCase):
    def test_matches_paper_layerwise_inverse(self):
        activations = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        weight_gradient = torch.tensor([[2.0, 4.0], [3.0, 5.0]])
        bias_gradient = torch.tensor([1.0, 2.0])
        omega = 0.5
        augmented = torch.cat((activations, torch.ones(2, 1)), dim=1)
        gradient = torch.cat((weight_gradient, bias_gradient[:, None]), dim=1)
        p = torch.eye(3) + omega / 2 * augmented.T @ augmented
        expected = torch.linalg.solve(p, gradient.T).T

        actual_weight, actual_bias = precondition_affine_gradient(
            activations, weight_gradient, bias_gradient, omega)
        torch.testing.assert_close(actual_weight, expected[:, :2])
        torch.testing.assert_close(actual_bias, expected[:, 2])

    def test_trainer_applies_precondition_hooks_before_optimizer_steps(self):
        args = SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 1)], aggregation='concat',
            device='cpu', optimizer='sgd', lr=0.1, bottom_lr_scale=1.0,
            momentum=0.0, weight_decay=0.0, own_concentrate_weight=0.0,
            data_flow_audit=0,
        )
        bottom = torch.nn.Linear(1, 1, bias=False)
        top = TopModel(1, 2)
        trainer = VFLTrainer([bottom], top, args)
        before = [parameter.detach().clone()
                  for module in (bottom, top) for parameter in module.parameters()]
        called = []

        def clear_gradients(modules):
            called.append(len(modules))
            for module in modules:
                for parameter in module.parameters():
                    if parameter.grad is not None:
                        parameter.grad.zero_()

        trainer.gradient_transform_top = clear_gradients
        trainer.gradient_transform_bottom = clear_gradients
        optimizer = trainer._create_optimizers()
        trainer.train_epoch([(torch.tensor([[1.0], [-1.0]]),
                              torch.tensor([0, 1]))], optimizer)
        after = [parameter.detach() for module in (bottom, top)
                 for parameter in module.parameters()]
        self.assertEqual(called, [1, 1])
        for old, new in zip(before, after):
            torch.testing.assert_close(old, new)

    def test_convolution_uses_replay_patch_activations(self):
        convolution = torch.nn.Conv2d(3, 4, 3, padding=1, bias=False)
        bottom = torch.nn.Sequential(
            convolution, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        top = TopModel(4, 2)
        trainer = SimpleNamespace(
            bottoms=[bottom], top_model=top,
            _aggregate=lambda parts: parts[0],
            bytes_transmitted=0, comm_rounds=0,
        )
        args = SimpleNamespace(
            data='cifar100', num_parties=1, img_size=32, device='cpu',
            er_per_class=2, er_batch=2, batch_size=2,
            lpr_omega0=1.0, lpr_beta=1.0, lpr_interval=10,
            lpr_reference_size=2, lpr_max_rows=8,
        )
        method = LPRCL(trainer, args)
        examples = torch.randn(2, 3, 32, 32)
        method.buffer.add_batch(examples, torch.tensor([0, 1]))
        method._refresh()
        self.assertIn(convolution, method.references)
        convolution.weight.grad = torch.ones_like(convolution.weight)
        method._transform_modules([bottom])
        self.assertTrue(torch.isfinite(convolution.weight.grad).all())
        self.assertFalse(torch.equal(convolution.weight.grad,
                                     torch.ones_like(convolution.weight)))


if __name__ == '__main__':
    unittest.main()
