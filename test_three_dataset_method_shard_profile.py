import os
import unittest
from unittest import mock

import three_dataset_formal_registry as registry
import three_dataset_formal_driver as driver


class MethodShardProfileTests(unittest.TestCase):
    def test_selected_method_has_exact_three_seed_specs(self):
        env = {
            registry.PROFILE_ENV: 'single-method-formal-v1',
            registry.DATASET_ENV: 'cifar100',
            'VFCL_FORMAL_METHOD': 'adagauss',
            'VFCL_PYTHON': str(registry._REVIEWED_VFCL_PYTHON),
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual((3, 0), registry.profile_cardinality())
            self.assertEqual('cifar100', registry.selected_formal_dataset())
            self.assertEqual('adagauss', registry.selected_formal_method())
            self.assertEqual(
                [('cifar100', 'adagauss', seed) for seed in (42, 43, 44)],
                [(spec.dataset, spec.method, spec.seed)
                 for spec in registry.formal_specs()],
            )
            self.assertEqual((), registry.explanation_specs())
            command = registry.command_for(
                registry.formal_specs()[0], 'cuda:0', '/tmp/method-shard')
            options = registry._option_map(command)
            self.assertNotIn('expected_party_kd_variant', options)
            self.assertEqual('0', options['dep_tracking_enabled'])
            self.assertEqual('0', options['party_kd_enabled'])

    def test_driver_plan_binds_method_and_exact_three_jobs(self):
        env = {
            registry.PROFILE_ENV: 'single-method-formal-v1',
            registry.DATASET_ENV: 'cifar100',
            registry.METHOD_ENV: 'adagauss',
        }
        with mock.patch.dict(os.environ, env, clear=True):
            census = driver.build_census({})
            plan = driver.build_plan(census)
        expected = [f'cifar100:adagauss:{seed}' for seed in (42, 43, 44)]
        self.assertEqual('cifar100', plan['formal_dataset'])
        self.assertEqual('adagauss', plan['formal_method'])
        self.assertEqual(expected, plan['formal_cells'])
        self.assertEqual(expected, plan['missing_jobs'])
        with mock.patch.dict(os.environ, env, clear=True):
            driver._validate_completed_plan_identity(plan)

    def test_method_shard_has_distinct_success_markers(self):
        self.assertEqual(
            registry.METHOD_SHARD_PROFILE,
            driver._SUCCESS_MARKER_PROFILES['METHOD_SHARD_SUCCESS'])
        self.assertEqual(
            {'method_shard_success'},
            driver._MARKER_KINDS['METHOD_SHARD_SUCCESS'])

    def test_method_selector_fails_closed(self):
        base = {
            registry.PROFILE_ENV: 'single-method-formal-v1',
            registry.DATASET_ENV: 'cifar100',
        }
        for method in (None, '', 'unknown'):
            env = dict(base)
            if method is not None:
                env['VFCL_FORMAL_METHOD'] = method
            with self.subTest(method=method), \
                    mock.patch.dict(os.environ, env, clear=True), \
                    self.assertRaises(ValueError):
                registry.experiment_profile()

    def test_method_selector_is_forbidden_outside_shard_profile(self):
        with mock.patch.dict(os.environ, {
                registry.PROFILE_ENV: registry.SINGLE_DATASET_PROFILE,
                registry.DATASET_ENV: 'cifar100',
                'VFCL_FORMAL_METHOD': 'adagauss',
        }, clear=True), self.assertRaises(ValueError):
            registry.experiment_profile()


if __name__ == '__main__':
    unittest.main()
