import copy
import os
import unittest
from pathlib import Path
from unittest import mock

import three_dataset_formal_driver as driver
from three_dataset_cifar_continuation_reconcile import build_bundle, origin_keys
from three_dataset_cifar_continuation_profile import reuse_census


ORIGIN_ROOT = Path(
    '/home/c3080/YangXiaoXiang/VF-CL/results/'
    'formal-cifar3080-dual-20g-20260928-v8'
)


class CifarContinuationProjectionTests(unittest.TestCase):
    def test_only_missing_twenty_four_are_planned(self):
        old_env = {'VFCL_EXPERIMENT_PROFILE': 'single-dataset-full-matrix',
                   'VFCL_FORMAL_DATASET': 'cifar100',
                   'VFCL_PYTHON': '/home/c3080/YangXiaoXiang/envs/vfcl/bin/python'}
        new_env = {**old_env,
                   'VFCL_EXPERIMENT_PROFILE':
                   'single-dataset-verified-continuation-v1'}
        with mock.patch.dict(os.environ, old_env):
            bundle = build_bundle(ORIGIN_ROOT, Path(__file__).resolve().parent)
        with mock.patch.dict(os.environ, new_env):
            census = reuse_census(bundle)
            statuses = {row['spec_key']: row['status']
                        for row in census['records']}
            self.assertEqual(len(statuses), 42)
            self.assertEqual({key for key, status in statuses.items()
                              if status == 'REUSABLE'}, set(origin_keys()))
            missing = driver.build_plan(census)['missing_jobs']
            self.assertEqual(len(missing), 24)
            self.assertEqual(missing[0], 'cifar100:gpm:42')
            self.assertTrue(set(missing).isdisjoint(origin_keys()))
            duplicate = copy.deepcopy(bundle)
            duplicate['admitted'].append(copy.deepcopy(duplicate['admitted'][0]))
            with self.assertRaises(ValueError):
                reuse_census(duplicate)
            gpm = copy.deepcopy(bundle)
            gpm['admitted'][0]['spec_key'] = 'cifar100:gpm:42'
            with self.assertRaises(ValueError):
                reuse_census(gpm)
        with mock.patch.dict(os.environ, old_env), self.assertRaises(ValueError):
            driver._validate_census(census)
