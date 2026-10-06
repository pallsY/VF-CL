from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import torch

import adaptive_consolidation_audit as audit
import adaptive_head_consolidation as head
import adaptive_tinyimagenet_heldout as heldout
import cl_methods.proto_evolve as proto
import data_utils
import three_dataset_formal_runtime as runtime


def runtime_function_identities():
    return (
        id(head.solve_global_mixture_weight),
        id(proto.solve_global_mixture_weight),
        id(audit.solve_global_mixture_weight),
        id(head.install_and_reload_verify),
        id(proto.install_and_reload_verify),
        id(audit.install_and_reload_verify),
        id(head._validate_primary_gate),
        id(proto.fit_adaptive_candidates),
        id(head.FULL_BRANCH_CONFIG),
        id(head.BIAS_BRANCH_CONFIG),
        id(proto.FULL_BRANCH_CONFIG),
        id(proto.BIAS_BRANCH_CONFIG),
    )


class ThreeDatasetFormalRuntimeTest(unittest.TestCase):
    def test_adaptive_is_identity_and_unknown_variant_fails_closed(self):
        originals = runtime_function_identities()
        with runtime.variant_runtime('adaptive'):
            self.assertEqual(originals, runtime_function_identities())
        self.assertEqual(originals, runtime_function_identities())
        with self.assertRaisesRegex(ValueError, 'unknown internal variant'):
            with runtime.variant_runtime('unknown'):
                pass
        self.assertEqual(originals, runtime_function_identities())

    def test_fixed_endpoint_fits_only_selected_sparse_branch(self):
        replay = {0: torch.zeros(1, 2), 1: torch.ones(1, 2)}
        for variant, selected, inactive in (
                ('fixed_full', 'full', 'bias'),
                ('fixed_bias', 'bias', 'full')):
            with self.subTest(variant=variant):
                pre = torch.nn.Linear(2, 2)
                with mock.patch.object(
                        head, 'consolidate_classifier',
                        return_value={'mode': 'full'}) as full_fit, \
                        mock.patch.object(
                            head, 'consolidate_task_class_bias',
                            return_value={'mode': 'bias'}) as bias_fit:
                    with runtime.variant_runtime(variant):
                        candidates = proto.fit_adaptive_candidates(
                            pre, replay, replay, ((0,), (1,)), 0, 42, 'cpu')
                calls = {'full': full_fit.call_count, 'bias': bias_fit.call_count}
                self.assertEqual(1, calls[selected])
                self.assertEqual(0, calls[inactive])
                self.assertEqual({}, dict(getattr(candidates, f'{inactive}_state')))
                self.assertEqual(
                    {'mode': 'inactive', 'parameters': 0, 'skipped': True},
                    getattr(candidates, f'{inactive}_audit'),
                )

    def test_no_consolidation_skips_all_candidate_fit(self):
        class PreHead:
            def state_dict(self):
                return {'weight': torch.ones(1)}

        with mock.patch.object(
                head, 'fit_fixed_endpoint_candidate',
                side_effect=AssertionError('fixed fit must not run')) as fixed, \
                mock.patch.object(
                    proto, 'fit_adaptive_candidates',
                    side_effect=AssertionError('dual fit must not run')) as dual:
            with runtime.variant_runtime('no_consolidation'):
                candidates = proto.fit_adaptive_candidates(
                    PreHead(), {0: torch.zeros(1)}, {}, ((0,),), 0, 42, 'cpu')
        fixed.assert_not_called()
        dual.assert_not_called()
        self.assertEqual('no_consolidation', candidates.full_audit['mode'])
        self.assertEqual('no_consolidation', candidates.bias_audit['mode'])

    def test_explanation_solvers_are_exact_and_called_once(self):
        full = torch.log_softmax(torch.tensor([
            [4.0, 1.0], [2.0, 1.0], [1.0, 4.0],
        ], dtype=torch.float64), dim=1)
        bias = torch.log_softmax(torch.tensor([
            [2.0, 1.0], [1.0, 2.0], [1.0, 2.0],
        ], dtype=torch.float64), dim=1)
        labels = torch.tensor([0, 0, 1])
        cases = (
            ('fixed_half', 'fixed_half_gate_record',
             'fixed_half_ablation', 0.5),
            ('sample_mean_nll', 'solve_sample_mean_ablation',
             'sample_mean_ablation', None),
        )
        for variant, solver_name, rule, expected_g in cases:
            with self.subTest(variant=variant), mock.patch.object(
                    head, solver_name, wraps=getattr(head, solver_name)) as solver:
                with runtime.variant_runtime(variant):
                    gate = proto.solve_global_mixture_weight(
                        full, bias, labels, (0, 1))
                self.assertEqual(1, solver.call_count)
                self.assertEqual(rule, gate['gate_rule'])
                self.assertFalse(gate['is_primary'])
                if expected_g is not None:
                    self.assertEqual(expected_g, gate['g'])

    def test_runtime_restores_after_normal_raised_and_nested_exits(self):
        originals = runtime_function_identities()
        for variant in (
                'no_consolidation', 'fixed_full', 'fixed_bias', 'adaptive',
                'fixed_half', 'sample_mean_nll'):
            with self.subTest(variant=variant):
                with runtime.variant_runtime(variant):
                    pass
                self.assertEqual(originals, runtime_function_identities())
                with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                    with runtime.variant_runtime(variant):
                        raise RuntimeError('fixture failure')
                self.assertEqual(originals, runtime_function_identities())

        with runtime.variant_runtime('fixed_full'):
            outer = runtime_function_identities()
            with runtime.variant_runtime('fixed_full'):
                self.assertNotEqual(outer, runtime_function_identities())
            self.assertEqual(outer, runtime_function_identities())
        self.assertEqual(originals, runtime_function_identities())

    def test_isolated_runtime_does_not_access_dataset_or_heldout_state(self):
        sentinel = object()
        with mock.patch.object(
                data_utils, 'VFLDataset', side_effect=AssertionError), \
                mock.patch.object(
                    heldout, '_load_freeze', side_effect=AssertionError), \
                mock.patch.object(
                    heldout, '_validate_runtime_freeze',
                    side_effect=AssertionError), \
                mock.patch.object(
                    head, 'fit_fixed_endpoint_candidate',
                    return_value=sentinel):
            with runtime.variant_runtime('fixed_full'):
                result = proto.fit_adaptive_candidates(
                    'pre', 'replay', 'prototypes', 'tasks', 0, 42, 'cpu')
        self.assertIs(sentinel, result)

    def test_execute_variant_main_runs_inside_context_and_restores(self):
        originals = runtime_function_identities()

        def inspect_context(path, run_name):
            self.assertEqual('main.py', path)
            self.assertEqual('__main__', run_name)
            self.assertNotEqual(originals, runtime_function_identities())
            return {'ok': True}

        with mock.patch.object(
                runtime.runpy, 'run_path', side_effect=inspect_context) as run:
            self.assertIsNone(runtime.execute_variant_main('fixed_full'))
        run.assert_called_once_with('main.py', run_name='__main__')
        self.assertEqual(originals, runtime_function_identities())

        with mock.patch.object(
                runtime.runpy, 'run_path',
                side_effect=RuntimeError('main failed')):
            with self.assertRaisesRegex(RuntimeError, 'main failed'):
                runtime.execute_variant_main('sample_mean_nll')
        self.assertEqual(originals, runtime_function_identities())

    def test_fixed_wrappers_bootstrap_canonical_config_before_runtime(self):
        self.assertEqual(
            {'mode': 'inactive', 'parameters': 0},
            head.INACTIVE_BRANCH_CONFIG,
        )
        for variant in ('fixed_full', 'fixed_bias'):
            with self.subTest(variant=variant):
                wrapper = (
                    'from three_dataset_formal_runtime import '
                    'execute_variant_main; '
                    f'execute_variant_main({variant!r})'
                )
                completed = subprocess.run(
                    [sys.executable, '-c', wrapper, '--help'],
                    cwd=Path(__file__).resolve().parent,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                self.assertEqual(0, completed.returncode, completed.stdout)
                for option in (
                        '--head_full_lr', '--head_full_steps',
                        '--head_bias_lr', '--head_bias_steps'):
                    self.assertIn(option, completed.stdout)
