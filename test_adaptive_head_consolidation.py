import dataclasses
import inspect
import json
import math
import unittest
from unittest import mock

import torch

import adaptive_head_consolidation as adaptive
from adaptive_head_consolidation import (
    class_balanced_mixture_nll,
    fixed_half_gate_record,
    mix_log_probabilities,
    solve_global_mixture_weight,
    solve_sample_mean_ablation,
)
from models import TopModel


def log_probabilities(rows):
    return torch.log(torch.tensor(rows, dtype=torch.float64))


class AdaptiveBranchFitTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260814)
        self.pre_top = TopModel(2, 5, cosine=False)
        self.replay = {
            3: torch.tensor([[2.0, 0.0], [1.8, -0.1]]),
            1: torch.tensor([[-2.0, 0.0], [-1.8, 0.1]]),
        }
        self.prototypes = {
            class_id: {
                'mean': values.mean(0),
                'std': values.std(0),
            }
            for class_id, values in self.replay.items()
        }
        self.task_classes = {9: [3], 4: [1]}

    def test_fits_distinct_frozen_branches_from_one_origin(self):
        fit = getattr(adaptive, 'fit_adaptive_candidates', None)
        self.assertIsNotNone(fit, 'adaptive branch fitting API is missing')
        records = {}
        full_audit = {'mode': 'full-audit'}
        bias_audit = {'mode': 'bias-audit'}

        def full_spy(top_model, prototypes, regularization, steps, lr,
                     samples_per_class, seed, device, **kwargs):
            records['full'] = {
                'model': top_model,
                'entry_hash': adaptive.hash_top_state(top_model),
                'prototypes': prototypes,
                'regularization': regularization,
                'steps': steps,
                'lr': lr,
                'samples_per_class': samples_per_class,
                'seed': seed,
                'device': device,
                'kwargs': kwargs,
            }
            with torch.no_grad():
                top_model.classifier.weight.add_(10.0)
            return full_audit

        def bias_spy(top_model, replay_embeddings, task_classes,
                     class_regularization, task_regularization, task_weight,
                     steps, lr, samples_per_class, device, **kwargs):
            records['bias'] = {
                'model': top_model,
                'entry_hash': adaptive.hash_top_state(top_model),
                'replay_embeddings': replay_embeddings,
                'task_classes': task_classes,
                'class_regularization': class_regularization,
                'task_regularization': task_regularization,
                'task_weight': task_weight,
                'steps': steps,
                'lr': lr,
                'samples_per_class': samples_per_class,
                'device': device,
                'kwargs': kwargs,
            }
            with torch.no_grad():
                top_model.classifier.bias.add_(2.0)
            return bias_audit

        with mock.patch.object(
                adaptive, 'consolidate_classifier', side_effect=full_spy
        ), mock.patch.object(
                adaptive, 'consolidate_task_class_bias', side_effect=bias_spy
        ):
            candidates = fit(
                self.pre_top, self.replay, self.prototypes, self.task_classes,
                4, 123, 'cpu',
            )

        self.assertIsNot(records['full']['model'], self.pre_top)
        self.assertIsNot(records['bias']['model'], self.pre_top)
        self.assertIsNot(records['full']['model'], records['bias']['model'])
        self.assertEqual(records['full']['entry_hash'], candidates.pre_head_sha256)
        self.assertEqual(records['bias']['entry_hash'], candidates.pre_head_sha256)
        self.assertEqual(
            adaptive.hash_top_state(self.pre_top), candidates.pre_head_sha256
        )
        self.assertEqual(candidates.ordered_classes, (1, 3))
        self.assertEqual(candidates.full_audit, full_audit)
        self.assertEqual(candidates.bias_audit, bias_audit)
        self.assertEqual(
            candidates.full_head_sha256,
            adaptive.hash_top_state(records['full']['model']),
        )
        self.assertEqual(
            candidates.bias_head_sha256,
            adaptive.hash_top_state(records['bias']['model']),
        )

        self.assertIs(records['full']['prototypes'], self.prototypes)
        self.assertEqual(records['full']['regularization'], 0.01)
        self.assertEqual(records['full']['steps'], 500)
        self.assertEqual(records['full']['lr'], 0.01)
        self.assertEqual(records['full']['samples_per_class'], 20)
        self.assertEqual(records['full']['seed'], 123)
        self.assertEqual(records['full']['device'], 'cpu')
        self.assertIs(records['full']['kwargs']['replay_embeddings'], self.replay)
        self.assertEqual(records['full']['kwargs'], {
            'replay_embeddings': self.replay,
            'replay_source': 'balanced_current_encoder_raw_replay',
            'persistent_raw_example_count': 4,
        })
        self.assertIs(records['bias']['replay_embeddings'], self.replay)
        self.assertIs(records['bias']['task_classes'], self.task_classes)
        self.assertEqual(records['bias']['class_regularization'], 0.01)
        self.assertEqual(records['bias']['task_regularization'], 0.01)
        self.assertEqual(records['bias']['task_weight'], 1.3)
        self.assertEqual(records['bias']['steps'], 600)
        self.assertEqual(records['bias']['lr'], 0.03)
        self.assertEqual(records['bias']['samples_per_class'], 20)
        self.assertEqual(records['bias']['device'], 'cpu')
        self.assertEqual(records['bias']['kwargs'], {
            'persistent_raw_example_count': 4,
            'replay_source': 'balanced_current_encoder_raw_replay',
        })

        for state, model in (
                (candidates.full_state, records['full']['model']),
                (candidates.bias_state, records['bias']['model'])):
            with self.assertRaises(TypeError):
                state['new'] = torch.tensor(1.0)
            for name, value in state.items():
                live_value = model.state_dict()[name]
                self.assertEqual(value.device.type, 'cpu')
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad_fn)
                self.assertIsNot(value, live_value)
                if value.numel():
                    self.assertNotEqual(value.data_ptr(), live_value.data_ptr())
        self.assertFalse(any(
            isinstance(value, torch.nn.Module)
            for value in vars(candidates).values()
        ))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            candidates.ordered_classes = ()

    def test_rejects_origin_mutation_during_branch_fitting(self):
        fit = getattr(adaptive, 'fit_adaptive_candidates', None)
        self.assertIsNotNone(fit, 'adaptive branch fitting API is missing')

        def mutate_origin(*args, **kwargs):
            with torch.no_grad():
                self.pre_top.classifier.weight.add_(1.0)
            return {}

        with mock.patch.object(
                adaptive, 'consolidate_classifier', side_effect=mutate_origin
        ), mock.patch.object(
                adaptive, 'consolidate_task_class_bias', return_value={}
        ), self.assertRaisesRegex(
                RuntimeError,
                'pre-consolidation head changed during branch fitting'):
            fit(
                self.pre_top, self.replay, self.prototypes, self.task_classes,
                4, 123, 'cpu',
            )

    def test_fixed_endpoint_fits_only_selected_branch_and_keeps_inactive_empty(self):
        fit = getattr(adaptive, 'fit_fixed_endpoint_candidate', None)
        install = getattr(adaptive, 'install_fixed_endpoint_and_reload_verify', None)
        self.assertIsNotNone(fit, 'fixed endpoint fit API is missing')
        self.assertIsNotNone(install, 'fixed endpoint install API is missing')
        for branch, gate in (('full', 1.0), ('bias', 0.0)):
            with self.subTest(branch=branch):
                with mock.patch.object(
                        adaptive, 'consolidate_classifier', return_value={
                            'mode': 'full_classifier',
                        }) as full_fit, mock.patch.object(
                        adaptive, 'consolidate_task_class_bias', return_value={
                            'mode': 'task_class_bias',
                        }) as bias_fit:
                    candidates = fit(
                        self.pre_top, branch, self.replay, self.prototypes,
                        self.task_classes, 4, 123, 'cpu',
                    )
                self.assertEqual(full_fit.call_count, branch == 'full')
                self.assertEqual(bias_fit.call_count, branch == 'bias')
                inactive = (candidates.bias_state if branch == 'full'
                            else candidates.full_state)
                selected = (candidates.full_state if branch == 'full'
                            else candidates.bias_state)
                self.assertEqual(dict(inactive), {})
                self.assertIn('classifier.weight', selected)
                record = {
                    'gate_rule': f'fixed_{branch}', 'is_primary': False,
                    'g': gate, 'converged': True,
                }
                installed = install(self.pre_top, candidates, record)
                self.assertTrue(bool(installed._adaptive_enabled))
                self.assertEqual(float(installed._adaptive_gate), gate)
                self.assertEqual(installed._adaptive_full_weight.numel(), 0)
                restored = TopModel(2, 5, cosine=False)
                restored.load_state_dict(installed.state_dict(), strict=True)
                self.assertEqual(restored._adaptive_full_weight.numel(), 0)

    def test_signature_excludes_selection_and_evaluation_inputs(self):
        fit = getattr(adaptive, 'fit_adaptive_candidates', None)
        self.assertIsNotNone(fit, 'adaptive branch fitting API is missing')
        self.assertEqual(list(inspect.signature(fit).parameters), [
            'pre_top', 'replay_embeddings', 'prototypes', 'task_classes',
            'persistent_raw_example_count', 'seed', 'device',
            'samples_per_class',
        ])
        source = inspect.getsource(fit)
        self.assertNotIn('validation_loader', source)
        self.assertNotIn('test_loader', source)
        self.assertNotIn('solve_global_mixture_weight', source)


class ProbabilityMixtureTest(unittest.TestCase):
    def setUp(self):
        self.full = log_probabilities([[0.9, 0.1], [0.2, 0.8]])
        self.bias = log_probabilities([[0.6, 0.4], [0.7, 0.3]])

    def test_exact_endpoints_clone_the_selected_branch(self):
        at_zero = mix_log_probabilities(self.full, self.bias, 0.0)
        at_one = mix_log_probabilities(self.full, self.bias, 1.0)

        self.assertTrue(torch.equal(at_zero, self.bias))
        self.assertTrue(torch.equal(at_one, self.full))
        self.assertNotEqual(at_zero.data_ptr(), self.bias.data_ptr())
        self.assertNotEqual(at_one.data_ptr(), self.full.data_ptr())

    def test_interior_probability_mixture_is_normalized_float64(self):
        mixed = mix_log_probabilities(self.full, self.bias, 0.25)
        expected = 0.25 * self.full.exp() + 0.75 * self.bias.exp()

        self.assertEqual(mixed.dtype, torch.float64)
        self.assertTrue(torch.allclose(mixed.exp(), expected, atol=1e-15, rtol=0.0))
        self.assertTrue(torch.allclose(
            torch.logsumexp(mixed, dim=1),
            torch.zeros(2, dtype=torch.float64),
            atol=1e-15,
            rtol=0.0,
        ))

    def test_extreme_finite_log_probabilities_stay_finite(self):
        full = torch.log_softmax(
            torch.tensor([[0.0, -1000.0], [-1000.0, 0.0]], dtype=torch.float64),
            dim=1,
        )
        bias = torch.log_softmax(
            torch.tensor([[-1000.0, 0.0], [0.0, -1000.0]], dtype=torch.float64),
            dim=1,
        )

        mixed = mix_log_probabilities(full, bias, 0.5)

        self.assertTrue(torch.isfinite(mixed).all())
        self.assertTrue(torch.allclose(
            torch.logsumexp(mixed, dim=1),
            torch.zeros(2, dtype=torch.float64),
            atol=1e-15,
            rtol=0.0,
        ))

    def test_invalid_probability_inputs_and_gate_are_rejected(self):
        with self.assertRaises(ValueError):
            mix_log_probabilities(self.full[:, :1], self.bias, 0.5)
        with self.assertRaises(TypeError):
            mix_log_probabilities(self.full.float(), self.bias.float(), 0.5)
        with self.assertRaises(ValueError):
            mix_log_probabilities(self.full.masked_fill(
                torch.tensor([[True, False], [False, False]]), float('nan')
            ), self.bias, 0.5)
        for gate in (-0.1, 1.1, float('nan')):
            with self.subTest(gate=gate), self.assertRaises(ValueError):
                mix_log_probabilities(self.full, self.bias, gate)

    def test_probability_rows_must_be_normalized_with_float64_tolerance(self):
        within_tolerance = self.full + 0.5e-12
        mix_log_probabilities(within_tolerance, self.bias, 0.5)
        mix_log_probabilities(self.full, self.bias + 0.5e-12, 0.5)

        beyond_tolerance = self.full.clone()
        beyond_tolerance[0] += 2.0e-12
        with self.assertRaises(ValueError):
            mix_log_probabilities(beyond_tolerance, self.bias, 0.5)
        with self.assertRaises(ValueError):
            mix_log_probabilities(self.full, beyond_tolerance, 0.5)

    def test_class_balanced_loss_maps_ordered_class_ids_to_columns(self):
        probabilities = log_probabilities([
            [0.8, 0.2],
            [0.4, 0.6],
            [0.6, 0.4],
        ])
        labels = torch.tensor([10, 20, 20])

        actual = class_balanced_mixture_nll(
            probabilities, probabilities, labels, [10, 20], 0.0
        )
        expected = (-math.log(0.8) + (-math.log(0.6) - math.log(0.4)) / 2.0) / 2.0

        self.assertAlmostEqual(actual.item(), expected, places=15)

    def test_invalid_label_shape_missing_class_and_class_order_are_rejected(self):
        labels = torch.tensor([10, 20])
        invalid_calls = (
            (labels.reshape(2, 1), [10, 20]),
            (labels, [20, 10]),
            (labels, [10, 10]),
            (torch.tensor([10, 30]), [10, 20]),
            (torch.tensor([10, 10]), [10, 20]),
        )
        for bad_labels, classes in invalid_calls:
            with self.subTest(labels=bad_labels.tolist(), classes=classes):
                with self.assertRaises(ValueError):
                    class_balanced_mixture_nll(
                        self.full, self.bias, bad_labels, classes, 0.5
                    )


class GlobalGateSolverTest(unittest.TestCase):
    def test_left_and_right_boundary_optima_are_exact(self):
        labels = torch.tensor([0, 1])
        left = solve_global_mixture_weight(
            log_probabilities([[0.4, 0.6], [0.6, 0.4]]),
            log_probabilities([[0.8, 0.2], [0.2, 0.8]]),
            labels,
            [0, 1],
        )
        right = solve_global_mixture_weight(
            log_probabilities([[0.8, 0.2], [0.2, 0.8]]),
            log_probabilities([[0.4, 0.6], [0.6, 0.4]]),
            labels,
            [0, 1],
        )

        self.assertEqual(left['g'], 0.0)
        self.assertEqual(right['g'], 1.0)
        self.assertEqual(left['final_interval'], [0.0, 0.0])
        self.assertEqual(right['final_interval'], [1.0, 1.0])

    def test_zero_boundary_derivative_returns_unique_convex_boundary(self):
        labels = torch.tensor([0, 1])
        full = log_probabilities([[0.6, 0.4], [0.6, 0.4]])
        bias = log_probabilities([[0.4, 0.6], [0.2, 0.8]])

        left = solve_global_mixture_weight(full, bias, labels, [0, 1])
        right = solve_global_mixture_weight(bias, full, labels, [0, 1])

        self.assertLessEqual(abs(left['boundary_derivatives'][0]), 1e-12)
        self.assertGreater(left['boundary_derivatives'][1], 1e-12)
        self.assertEqual(left['g'], 0.0)
        self.assertEqual(left['final_interval'], [0.0, 0.0])
        self.assertLess(right['boundary_derivatives'][0], -1e-12)
        self.assertLessEqual(abs(right['boundary_derivatives'][1]), 1e-12)
        self.assertEqual(right['g'], 1.0)
        self.assertEqual(right['final_interval'], [1.0, 1.0])

    def test_interior_solution_is_bounded_converged_and_deterministic(self):
        full = log_probabilities([[0.9, 0.1], [0.8, 0.2]])
        bias = log_probabilities([[0.4, 0.6], [0.2, 0.8]])
        labels = torch.tensor([0, 1])

        first = solve_global_mixture_weight(full, bias, labels, [0, 1])
        second = solve_global_mixture_weight(full, bias, labels, [0, 1])

        self.assertEqual(first, second)
        self.assertGreater(first['g'], 0.0)
        self.assertLess(first['g'], 1.0)
        self.assertLessEqual(abs(first['g'] - 4.0 / 15.0), 1e-12)
        self.assertLessEqual(first['final_interval_width'], 1e-12)
        self.assertLessEqual(first['iterations'], 80)
        self.assertTrue(first['converged'])
        self.assertEqual(first['tolerance'], 1e-12)
        self.assertEqual(first['max_iterations'], 80)
        self.assertEqual(len(first['boundary_derivatives']), 2)
        json.dumps(first, allow_nan=False)

    def test_flat_objective_with_different_non_targets_selects_half(self):
        full = log_probabilities([
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.3, 0.2, 0.5],
        ])
        bias = log_probabilities([
            [0.5, 0.1, 0.4],
            [0.4, 0.5, 0.1],
            [0.1, 0.4, 0.5],
        ])
        result = solve_global_mixture_weight(
            full, bias, torch.tensor([10, 20, 30]), [10, 20, 30]
        )

        self.assertEqual(result['g'], 0.5)
        self.assertEqual(result['boundary_derivatives'], [0.0, 0.0])
        self.assertEqual(result['final_interval'], [0.5, 0.5])
        self.assertEqual(result['final_interval_width'], 0.0)

    def test_extreme_opposite_logits_produce_strict_json_record(self):
        full = torch.log_softmax(
            torch.tensor([[0.0, -1000.0], [-1000.0, 0.0]], dtype=torch.float64),
            dim=1,
        )
        bias = torch.log_softmax(
            torch.tensor([[-1000.0, 0.0], [0.0, -1000.0]], dtype=torch.float64),
            dim=1,
        )

        result = solve_global_mixture_weight(
            full, bias, torch.tensor([0, 1]), [0, 1]
        )

        self.assertEqual(result['g'], 1.0)
        json.dumps(result, allow_nan=False)

    def test_ablation_records_are_explicitly_non_primary(self):
        full, bias, labels, classes = self._imbalanced_fixture()

        primary = solve_global_mixture_weight(full, bias, labels, classes)
        sample_mean = solve_sample_mean_ablation(full, bias, labels, classes)
        fixed = fixed_half_gate_record(full, bias, labels, classes)

        self.assertEqual(primary['gate_rule'], 'class_balanced')
        self.assertTrue(primary['is_primary'])
        self.assertEqual(sample_mean['gate_rule'], 'sample_mean_ablation')
        self.assertFalse(sample_mean['is_primary'])
        self.assertEqual(fixed['gate_rule'], 'fixed_half_ablation')
        self.assertFalse(fixed['is_primary'])
        self.assertEqual(fixed['g'], 0.5)
        self.assertNotEqual(primary['g'], sample_mean['g'])
        self.assertNotEqual(
            primary['full_branch_nll'], sample_mean['full_branch_nll']
        )
        self.assertEqual(
            list(inspect.signature(solve_global_mixture_weight).parameters),
            ['log_p_full', 'log_p_bias', 'labels', 'classes'],
        )
        json.dumps(sample_mean, allow_nan=False)
        json.dumps(fixed, allow_nan=False)

    @staticmethod
    def _imbalanced_fixture():
        labels = torch.tensor([10, 20, 20, 20])
        full_true = [0.9, 0.4, 0.4, 0.4]
        bias_true = [0.1, 0.6, 0.6, 0.6]
        full_rows = [[full_true[0], 1.0 - full_true[0]]]
        bias_rows = [[bias_true[0], 1.0 - bias_true[0]]]
        full_rows += [[1.0 - p, p] for p in full_true[1:]]
        bias_rows += [[1.0 - p, p] for p in bias_true[1:]]
        return (
            log_probabilities(full_rows),
            log_probabilities(bias_rows),
            labels,
            [10, 20],
        )

    def test_fixed_synthetic_properties(self):
        generator = torch.Generator().manual_seed(20260814)
        labels = torch.tensor([10, 20, 30] * 4)
        classes = [10, 20, 30]
        for fixture in range(20):
            full = torch.log_softmax(
                torch.randn(12, 3, dtype=torch.float64, generator=generator), dim=1
            )
            bias = torch.log_softmax(
                torch.randn(12, 3, dtype=torch.float64, generator=generator), dim=1
            )
            result = solve_global_mixture_weight(full, bias, labels, classes)
            loss = lambda g: class_balanced_mixture_nll(
                full, bias, labels, classes, g
            ).item()

            with self.subTest(fixture=fixture, property='boundaries'):
                self.assertLessEqual(loss(result['g']), min(loss(0.0), loss(1.0)) + 1e-12)
            values = [loss(step / 10.0) for step in range(11)]
            for step in range(1, 10):
                with self.subTest(fixture=fixture, property='convexity', step=step):
                    self.assertGreaterEqual(
                        values[step - 1] - 2.0 * values[step] + values[step + 1],
                        -1e-10,
                    )


if __name__ == '__main__':
    unittest.main()
