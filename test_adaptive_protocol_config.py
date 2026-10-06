import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

from adaptive_head_consolidation import (
    ADAPTIVE_METHOD_VERSION,
    BIAS_BRANCH_CONFIG,
    FULL_BRANCH_CONFIG,
    SOLVER_MAX_ITERATIONS,
    SOLVER_TOLERANCE,
)
from config import get_config


class AdaptiveProtocolConfigTest(unittest.TestCase):
    adaptive_args = [
        '--head_consolidation_enabled', '1',
        '--head_consolidation_mode', 'adaptive_dual_branch',
        '--lambda_validation_enabled', '1',
    ]

    def parse_config(self, arguments, ablation_authorization=None):
        with tempfile.TemporaryDirectory() as results_dir:
            argv = [
                'config.py', '--results_dir', results_dir,
                '--exp_name', 'adaptive-contract', *arguments,
            ]
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop('VFCL_REVIEWED_ADAPTIVE_ABLATION', None)
                if ablation_authorization is not None:
                    os.environ['VFCL_REVIEWED_ADAPTIVE_ABLATION'] = ablation_authorization
                with mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                    return get_config()

    def assert_config_rejected(self, arguments, ablation_authorization=None):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                self.parse_config(arguments, ablation_authorization)
        self.assertEqual(error.exception.code, 2)

    def test_adaptive_defaults_are_frozen(self):
        args = self.parse_config(self.adaptive_args)

        self.assertEqual(ADAPTIVE_METHOD_VERSION, 1)
        self.assertEqual(FULL_BRANCH_CONFIG,
                         {'mode': 'full_classifier', 'lr': 0.01, 'steps': 500})
        self.assertEqual(BIAS_BRANCH_CONFIG,
                         {'mode': 'task_class_bias', 'lr': 0.03, 'steps': 600})
        self.assertEqual(SOLVER_TOLERANCE, 1e-12)
        self.assertEqual(SOLVER_MAX_ITERATIONS, 80)
        self.assertEqual((args.head_full_lr, args.head_full_steps), (0.01, 500))
        self.assertEqual((args.head_bias_lr, args.head_bias_steps), (0.03, 600))
        self.assertEqual(args.lambda_validation_split_seed, 20260729)
        self.assertEqual(args.head_gate_rule, 'class_balanced')
        self.assertEqual(args.head_gate_solver_tolerance, 1e-12)
        self.assertEqual(args.head_gate_solver_max_iterations, 80)

    def test_validation_seed_is_dataset_specific(self):
        development = self.parse_config(
            self.adaptive_args + ['--data', 'cifar100']
        )
        self.assertEqual(development.lambda_validation_split_seed, 20260729)
        self.assert_config_rejected(
            self.adaptive_args + [
                '--data', 'cifar100',
                '--lambda_validation_split_seed', '20260813',
            ]
        )
        for vector_name in ('isolet_vfl.npz', 'upmc_food101_vfl.npz'):
            canonical = self.parse_config(self.adaptive_args + [
                '--data', 'tabvfl', '--vector_npz', vector_name,
                '--lambda_validation_split_seed', '20260809',
            ])
            self.assertEqual(canonical.lambda_validation_split_seed, 20260809)
            self.assert_config_rejected(self.adaptive_args + [
                '--data', 'tabvfl', '--vector_npz', vector_name,
                '--lambda_validation_split_seed', '20260729',
            ])
        generic = self.parse_config(self.adaptive_args + [
            '--data', 'tabvfl', '--vector_npz', 'generic_vfl.npz',
            '--lambda_validation_split_seed', '20260729',
        ])
        self.assertEqual(generic.lambda_validation_split_seed, 20260729)
        self.assert_config_rejected(self.adaptive_args + [
            '--data', 'tabvfl', '--vector_npz', 'generic_vfl.npz',
            '--lambda_validation_split_seed', '20260809',
        ])
        tiny = self.parse_config(self.adaptive_args + [
            '--data', 'tinyimagenet', '--lambda_validation_per_class', '50',
            '--lambda_validation_split_seed', '20260813',
        ])
        self.assertEqual(tiny.lambda_validation_split_seed, 20260813)
        self.assert_config_rejected(
            self.adaptive_args + ['--data', 'tinyimagenet']
        )

    def test_adaptive_mode_rejects_incompatible_protocol(self):
        incompatible_options = [
            ('--head_consolidation_schedule', 'every'),
            ('--head_consolidation_samples_per_class', '19'),
            ('--head_consolidation_regularization', '0.02'),
            ('--head_consolidation_class_regularization', '0.02'),
            ('--head_consolidation_task_regularization', '0.02'),
            ('--head_consolidation_task_weight', '1.2'),
            ('--lambda_validation_enabled', '0'),
            ('--lambda_validation_split_seed', '20260730'),
            ('--head_full_lr', '0.02'),
            ('--head_full_steps', '499'),
            ('--head_bias_lr', '0.02'),
            ('--head_bias_steps', '599'),
            ('--head_gate_solver_tolerance', '1e-10'),
            ('--head_gate_solver_max_iterations', '79'),
        ]

        for option, value in incompatible_options:
            with self.subTest(option=option, value=value):
                self.assert_config_rejected(self.adaptive_args + [option, value])

    def test_gate_rules_require_reviewed_ablation_authorization(self):
        args = self.parse_config(self.adaptive_args)
        self.assertEqual(args.head_gate_rule, 'class_balanced')

        for rule in ('fixed_half_ablation', 'sample_mean_ablation'):
            arguments = self.adaptive_args + ['--head_gate_rule', rule]
            with self.subTest(rule=rule, authorization='missing'):
                self.assert_config_rejected(arguments)
            for authorization in ('0', 'reviewed'):
                with self.subTest(rule=rule, authorization=authorization):
                    self.assert_config_rejected(arguments, authorization)
            with self.subTest(rule=rule, authorization='1'):
                args = self.parse_config(arguments, '1')
                self.assertEqual(args.head_gate_rule, rule)

    def test_dataset_specific_adaptive_option_is_rejected(self):
        self.assert_config_rejected(
            self.adaptive_args + ['--adaptive_cifar100_head_full_lr', '0.01']
        )


if __name__ == '__main__':
    unittest.main()
