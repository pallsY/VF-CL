import tempfile
import unittest
from pathlib import Path

from launch_cifar_head_replay_intervention import (
    TrainingOnlyStop, derive_config, stop_before_deferred_evaluation,
)


class HeadReplayLauncherTests(unittest.TestCase):
    def test_fresh_bic_holdout_and_equal_memory(self):
        source = {
            'data': 'cifar100', 'seed': 42, 'num_tasks': 10,
            'classes_per_task': 10, 'custom_tasks': '',
            'lambda_validation_enabled': 1,
            'lambda_validation_per_class': 25,
            'lambda_validation_split_seed': 20260729,
            'formal_deferred_evaluation': True,
            'head_consolidation_enabled': 1,
            'head_consolidation_mode': 'adaptive_dual_branch',
            'head_consolidation_schedule': 'final',
            'head_consolidation_samples_per_class': 20,
            'bic_enabled': 1, 'bic_per_class': 25,
            'bic_split_seed': 20260722,
            'unlearn_after_tasks': [10],
            'deterministic': 1, 'data_flow_audit': 1,
            'results_dir': '/formal', 'output_dir': '/formal/seed42',
            'exp_name': 'formal_seed42',
        }
        with tempfile.TemporaryDirectory() as scratch:
            config, changed = derive_config(source, Path(scratch) / 'pilot')
        self.assertEqual(set(changed), {
            'seed', 'bic_split_seed', 'formal_deferred_evaluation',
            'results_dir', 'output_dir', 'exp_name',
        })
        self.assertEqual(config['seed'], 50)
        self.assertEqual(config['bic_split_seed'], 20261011)
        self.assertEqual(config['head_consolidation_schedule'], 'final')
        self.assertEqual(config['head_consolidation_samples_per_class'], 20)
        self.assertEqual(config['lambda_validation_split_seed'], 20260729)

    def test_explicit_stop_precedes_deferred_test_evaluation(self):
        with self.assertRaises(TrainingOnlyStop):
            stop_before_deferred_evaluation()


if __name__ == '__main__':
    unittest.main()
