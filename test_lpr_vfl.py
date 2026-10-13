import unittest

import torch
from types import SimpleNamespace

from cl_methods.lpr import LPRCL, precondition_affine_gradient
from models import TopModel
from vfl_trainer import VFLTrainer


class LPRPreconditionerTest(unittest.TestCase):
    def test_batchnorm_jointly_preconditions_scale_and_shift(self):
        bn = torch.nn.BatchNorm2d(1, eps=0.0)
        bn.eval()
        with torch.no_grad():
            bn.running_mean.fill_(0.5)
            bn.running_var.fill_(4.0)
        args = SimpleNamespace(
            batch_size=1, er_per_class=1, er_batch=1,
            lpr_omega0=1.0, lpr_beta=0.0,
            lpr_interval=1, lpr_reference_size=1, lpr_max_rows=8,
        )
        method = LPRCL(SimpleNamespace(), args)
        method.reference_count = 1
        raw = torch.tensor([[[[2.5, 4.5]]]])
        method._record_inputs(bn, (raw,))
        bn.weight.grad = torch.tensor([2.0])
        bn.bias.grad = torch.tensor([0.0])
        method._transform_modules([bn])

        normalized = torch.tensor([[1.0, 1.0], [2.0, 1.0]])
        system = torch.eye(2) + normalized.T @ normalized
        expected = torch.linalg.solve(system, torch.tensor([2.0, 0.0]))
        torch.testing.assert_close(bn.weight.grad, expected[:1])
        torch.testing.assert_close(bn.bias.grad, expected[1:])
        self.assertLess(float(bn.bias.grad), 0.0)

    def test_patch_row_cap_preserves_full_covariance_mass(self):
        convolution = torch.nn.Conv2d(1, 1, 1, bias=True)
        args = SimpleNamespace(
            batch_size=1, er_per_class=1, er_batch=1,
            lpr_omega0=1.0, lpr_beta=1.0,
            lpr_interval=1, lpr_reference_size=1, lpr_max_rows=4,
        )
        method = LPRCL(SimpleNamespace(), args)
        method.reference_count = 1
        method._record_inputs(convolution, (torch.full((1, 1, 4, 4), 2.0),))
        convolution.weight.grad = torch.tensor([[[[1.0]]]])
        convolution.bias.grad = torch.tensor([0.0])
        method._transform_modules([convolution])

        full_rows = torch.tensor([[2.0, 1.0]]).repeat(16, 1)
        system = torch.eye(2) + full_rows.T @ full_rows / 16
        expected = torch.linalg.solve(system, torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(convolution.weight.grad.flatten(), expected[:1])
        torch.testing.assert_close(convolution.bias.grad, expected[1:])

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

    def test_four_party_lpr_step_refreshes_top_and_all_bottoms(self):
        args = SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 1), (1, 2),
                                             (2, 3), (3, 4)],
            aggregation='sum', device='cpu', optimizer='sgd', lr=0.05,
            bottom_lr_scale=1.0, momentum=0.0, weight_decay=0.0,
            own_concentrate_weight=0.0, data_flow_audit=0,
            batch_size=2, epochs_per_task=1, er_per_class=2, er_batch=2,
            er_alpha=1.0, lpr_omega0=1.0, lpr_beta=1.0,
            lpr_interval=10, lpr_reference_size=2, lpr_max_rows=4,
        )
        bottoms = [torch.nn.Linear(1, 1) for _ in range(4)]
        top = TopModel(1, 2)
        trainer = VFLTrainer(bottoms, top, args)
        method = LPRCL(trainer, args)
        examples = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                                 [-1.0, -2.0, -3.0, -4.0]])
        labels = torch.tensor([0, 1])
        method.buffer.add_batch(examples, labels)
        method.before_task(1, [1], [0, 1])
        before = [parameter.detach().clone()
                  for module in (*bottoms, top)
                  for parameter in module.parameters()]

        method.train_task([(examples, labels)], 1)

        self.assertEqual(method.steps, 1)
        self.assertTrue(all(bottom in method.references for bottom in bottoms))
        self.assertIn(top.classifier, method.references)
        after = [parameter.detach() for module in (*bottoms, top)
                 for parameter in module.parameters()]
        self.assertTrue(any(not torch.equal(old, new)
                            for old, new in zip(before, after)))
        self.assertFalse(hasattr(trainer, 'gradient_transform_top'))
        self.assertFalse(hasattr(trainer, 'gradient_transform_bottom'))

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
