import tempfile
import unittest

from run_resource_lite_pilot import expected_tasks
from run_selected_baseline_dev_pilot import (
    derive_config, summarize_icarl_readouts,
)


class SelectedBaselineDevPilotTests(unittest.TestCase):
    def test_icarl_uses_exemplar_mean_counts_not_global_head(self):
        readouts = {
            'class_il_pred_task_counts': {
                'task_0': {'correct': 7, 'total': 10},
                'task_1': {'correct': 9, 'total': 10},
            },
            'class_il_global': {'task_0': 0.1, 'task_1': 0.2},
            'task_il_oracle': {'task_0': 0.9, 'task_1': 1.0},
            'task_prediction': {'task_0': 0.8, 'task_1': 0.9},
        }
        metrics = summarize_icarl_readouts(readouts, expected_count=20)
        self.assertEqual(metrics['cil'], 0.8)
        self.assertEqual(metrics['per_task_cil'], [0.7, 0.9])
        self.assertEqual(metrics['readout_rule'], 'icarl_nearest_exemplar_mean')
        self.assertEqual(metrics['global_head_per_task'], [0.1, 0.2])

    def source(self, dataset):
        specs = {
            'isolet': ('icarl', 26, 13, 'isolet/isolet_vfl.npz', 40, 20260809),
            'upmc_food101': (
                'der_pp', 101, 10,
                'upmc_food101_image_clean_4party_v1/upmc_food101_vfl.npz',
                64, 20260809,
            ),
            'cifar100': ('er_ace', 100, 10, '', 25, 20260729),
        }
        method, classes, tasks, vector, count, split_seed = specs[dataset]
        return {
            'seed': 42,
            'data': 'cifar100' if dataset == 'cifar100' else 'tabvfl',
            'num_classes': classes, 'num_tasks': tasks, 'num_parties': 4,
            'custom_tasks': '' if dataset == 'cifar100' else expected_tasks(dataset),
            'classes_per_task': 10 if dataset == 'cifar100' else 2,
            'cl_method': method,
            'vector_npz': '/brf/users/yangxx/VF-CL/data/' + vector if vector else '',
            'head_consolidation_enabled': 0,
            'lambda_validation_enabled': 1,
            'lambda_validation_per_class': count,
            'lambda_validation_split_seed': split_seed,
            'bic_enabled': 1 if dataset == 'cifar100' else 0,
            'formal_deferred_evaluation': True,
            'epochs_per_task': 20 if dataset == 'upmc_food101' else 50,
            'results_dir': '/old', 'output_dir': '/old/run',
        }

    def test_selected_baselines_preserve_data_and_training_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            for dataset in ('isolet', 'upmc_food101', 'cifar100'):
                with self.subTest(dataset=dataset):
                    source = self.source(dataset)
                    config = derive_config(source, folder, dataset, 54)
                    self.assertEqual(config['num_parties'], 4)
                    self.assertEqual(config['custom_tasks'], source['custom_tasks'])
                    self.assertEqual(config['epochs_per_task'],
                                     source['epochs_per_task'])
                    self.assertEqual(config['seed'], 54)
                    self.assertEqual(source['seed'], 42)

    def test_two_party_or_wrong_method_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.source('upmc_food101')
            with self.assertRaisesRegex(ValueError, 'identity'):
                derive_config({**source, 'num_parties': 2},
                              folder, 'upmc_food101', 54)
            with self.assertRaisesRegex(ValueError, 'method'):
                derive_config({**source, 'cl_method': 'er'},
                              folder, 'upmc_food101', 54)
            with self.assertRaisesRegex(ValueError, 'seed'):
                derive_config(source, folder, 'upmc_food101', 42)

    def test_extra_validation_or_training_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.source('isolet')
            for change in ({'lambda_validation_per_class': 41},
                           {'epochs_per_task': 10},
                           {'head_consolidation_enabled': 1}):
                with self.subTest(change=change):
                    with self.assertRaises(ValueError):
                        derive_config({**source, **change},
                                      folder, 'isolet', 54)

    def test_smoke_preserves_two_task_prefix_and_uses_cpu(self):
        with tempfile.TemporaryDirectory() as folder:
            config = derive_config(self.source('isolet'), folder,
                                   'isolet', 54, smoke=True)
        self.assertEqual(config['custom_tasks'], '0,1|2,3')
        self.assertEqual(config['num_tasks'], 2)
        self.assertEqual(config['epochs_per_task'], 1)
        self.assertEqual(config['device'], 'cpu')
        self.assertTrue(config['pilot_smoke'])


if __name__ == '__main__':
    unittest.main()
