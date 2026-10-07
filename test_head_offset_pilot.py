import copy
import tempfile
import unittest
from pathlib import Path

from launch_head_offset_pilot import derive_config


class HeadOffsetPilotConfigTests(unittest.TestCase):
    def test_derivation_changes_only_preregistered_fields(self):
        source = {
            'data': 'cifar100', 'seed': 42, 'num_tasks': 10,
            'classes_per_task': 10, 'custom_tasks': '',
            'unlearn_after_tasks': [999],
            'lambda_validation_enabled': 1,
            'lambda_validation_per_class': 25,
            'lambda_validation_split_seed': 20260729,
            'formal_deferred_evaluation': True,
            'head_consolidation_enabled': 1,
            'head_consolidation_mode': 'adaptive_dual_branch',
            'dep_tracking_enabled': 1, 'party_kd_enabled': 1,
            'party_kd_mode': 'uniform', 'save_task_checkpoints': 3,
            'results_dir': '/old/root', 'output_dir': '/old/root/run',
            'exp_name': 'old', 'lr': 0.001,
        }
        unchanged = copy.deepcopy(source)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'new-root'
            config, changed = derive_config(source, root)
        self.assertEqual(source, unchanged)
        self.assertEqual(config['seed'], 45)
        self.assertEqual(config['lambda_validation_split_seed'], 20261007)
        self.assertEqual(config['head_consolidation_enabled'], 0)
        self.assertEqual(config['head_consolidation_mode'], 'full_classifier')
        self.assertIs(config['formal_deferred_evaluation'], False)
        self.assertEqual(config['lr'], source['lr'])
        self.assertEqual(set(changed), {
            'seed', 'lambda_validation_split_seed',
            'formal_deferred_evaluation', 'head_consolidation_enabled',
            'head_consolidation_mode', 'results_dir', 'output_dir', 'exp_name',
        })
        self.assertEqual(config['output_dir'], str(root / 'seed_45_baseline'))


if __name__ == '__main__':
    unittest.main()
