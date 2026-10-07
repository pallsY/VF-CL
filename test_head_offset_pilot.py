import copy
import tempfile
import unittest
from pathlib import Path

from launch_head_offset_pilot import derive_config
from analyze_head_offset_pilot import (
    fit_scalar_offset, select_calibration_indices, pilot_passes,
)


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
            'deterministic': 1, 'data_flow_audit': 1,
            'num_workers': 2, 'bic_enabled': 1,
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

    def test_train_only_calibration_indices_are_fixed_and_disjoint(self):
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        indices = select_calibration_indices(labels, {0, 4}, 2)
        self.assertEqual(indices, [1, 2, 5, 6])
        self.assertFalse(set(indices) & {0, 4})

    def test_scalar_fit_corrects_new_class_underprediction(self):
        logits = [[2.0, 0.0], [2.0, 0.0]] * 20
        labels = [0, 1] * 20
        delta, nll = fit_scalar_offset(logits, labels, [1])
        self.assertAlmostEqual(delta, 2.0, places=3)
        self.assertLess(nll, 0.7)

    def test_scalar_fit_accepts_interior_optimum_near_bound(self):
        logits = [[9.995, 0.0], [9.995, 0.0]] * 20
        labels = [0, 1] * 20
        delta, _ = fit_scalar_offset(logits, labels, [1])
        self.assertAlmostEqual(delta, 9.995, places=3)
        with self.assertRaises(ValueError):
            fit_scalar_offset([[11.0, 0.0], [11.0, 0.0]] * 20,
                              labels, [1])

    def test_pilot_gate_requires_new_gain_without_old_regression(self):
        baseline = dict(examples=2500, old_examples=2250, new_examples=250,
                        correct=40, old_correct=40, new_correct=0,
                        taskil_correct=1750)
        improved = dict(examples=2500, old_examples=2250, new_examples=250,
                        correct=65, old_correct=27, new_correct=38,
                        taskil_correct=1750)
        self.assertTrue(pilot_passes(baseline, improved))
        regressed = dict(improved, correct=65, old_correct=10, new_correct=55)
        self.assertFalse(pilot_passes(baseline, regressed))


if __name__ == '__main__':
    unittest.main()
