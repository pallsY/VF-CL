import unittest

import torch
from types import SimpleNamespace

from cl_methods.ocar import OCARCL, kfac_precondition
from models import TopModel
from vfl_trainer import VFLTrainer


class OCARKFACTest(unittest.TestCase):
    def test_legacy_factor_state_is_rejected(self):
        trainer = SimpleNamespace(bottoms=[], top_model=TopModel(1, 2))
        args = SimpleNamespace(device='cpu', lr=0.1, batch_size=1,
                               er_per_class=1, er_batch=1)
        method = OCARCL(trainer, args)
        old_state = method.get_state()
        old_state.pop('factor_schema_version', None)
        with self.assertRaisesRegex(ValueError, 'factor schema'):
            OCARCL(trainer, args).load_state(old_state)

    def test_four_party_step_builds_all_curvature_factors(self):
        args = SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 1), (1, 2),
                                             (2, 3), (3, 4)],
            aggregation='sum', device='cpu', optimizer='sgd', lr=0.05,
            bottom_lr_scale=1.0, momentum=0.0, weight_decay=0.0,
            own_concentrate_weight=0.0, data_flow_audit=0,
            batch_size=2, epochs_per_task=1, er_per_class=2, er_batch=2,
            ocar_ema=0.1, ocar_delta_tau=0.0, ocar_max_rows=4,
        )
        bottoms = [torch.nn.Linear(1, 1) for _ in range(4)]
        top = TopModel(1, 2)
        trainer = VFLTrainer(bottoms, top, args)
        method = OCARCL(trainer, args)
        old = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                            [-1.0, -2.0, -3.0, -4.0]])
        labels = torch.tensor([0, 1])
        method.buffer.add_batch(old, labels)
        method.before_task(1, [1], [0, 1])
        method.train_task([(old, labels)], 1)
        self.assertEqual(method.steps, 1)
        self.assertTrue(all(bottom in method.factors for bottom in bottoms))
        self.assertIn(top.classifier, method.factors)
        for rows, covariance in method.factors.values():
            self.assertTrue(torch.isfinite(rows).all())
            self.assertTrue(torch.isfinite(covariance).all())

    def test_model_preconditioner_uses_tau_as_block_damping(self):
        bottom = torch.nn.Linear(1, 1, bias=False)
        trainer = SimpleNamespace(bottoms=[bottom], top_model=TopModel(1, 1))
        args = SimpleNamespace(device='cpu', lr=0.25, batch_size=1,
                               er_per_class=1, er_batch=1)
        method = OCARCL(trainer, args)
        method.factors[bottom] = (torch.tensor([[2.0]]), torch.tensor([[3.0]]))
        bottom.weight.grad = torch.ones_like(bottom.weight)
        method._precondition([bottom])
        torch.testing.assert_close(
            bottom.weight.grad, torch.tensor([[1 / (3 * 4 + 0.25)]])
        )

    def test_old_class_curvature_weight_counts_only_old_classes(self):
        top = TopModel(1, 10)
        trainer = SimpleNamespace(bottoms=[], top_model=top)
        args = SimpleNamespace(device='cpu', lr=0.1, batch_size=2,
                               er_per_class=2, er_batch=2)
        method = OCARCL(trainer, args)
        method.before_task(1, list(range(10, 20)), list(range(20)))
        self.assertEqual(method.lambda_buffer, 1.0)

    def test_factor_ema_preserves_covariance_under_rank_cap(self):
        bottom = torch.nn.Linear(2, 2, bias=False)
        trainer = SimpleNamespace(
            bottoms=[bottom], top_model=TopModel(2, 2),
            _aggregate=lambda values: values[0],
            bytes_transmitted=0, comm_rounds=0,
        )
        args = SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 2)],
            device='cpu', lr=0.1, batch_size=2,
            er_per_class=2, er_batch=2,
            ocar_ema=0.25, ocar_max_rows=2,
            ocar_delta_tau=0.0,
        )
        method = OCARCL(trainer, args)
        first = torch.eye(2)
        second = 2 * torch.eye(2)
        method.current_batch = first
        method._update_factors()
        first_rows, _ = method.factors[bottom]
        first_covariance = first.T @ first / len(first)
        torch.testing.assert_close(first_rows.T @ first_rows, first_covariance)

        method.current_batch = second
        method._update_factors()
        rows, _ = method.factors[bottom]
        expected = (0.75 * first_covariance
                    + 0.25 * second.T @ second / len(second))
        self.assertEqual(len(rows), 2)
        torch.testing.assert_close(rows.T @ rows, expected)

    def test_low_rank_kfac_matches_dense_damped_block_solve(self):
        activations = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        output_covariance = torch.tensor([[2.0, 0.2], [0.2, 1.0]])
        gradient = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        damping = 0.5
        input_covariance = activations.T @ activations
        system = torch.kron(output_covariance, input_covariance)
        system += damping * torch.eye(gradient.numel())
        expected = torch.linalg.solve(system, gradient.flatten()).reshape_as(gradient)
        actual = kfac_precondition(
            gradient, activations, output_covariance, damping)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            output_covariance @ actual @ input_covariance + damping * actual,
            gradient,
        )

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
