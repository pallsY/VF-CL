import unittest
from pathlib import Path

import run_party_drift_control_preflight as pilot


class ControlPreflightTests(unittest.TestCase):
    def test_prepare_locks_two_tasks_seed_and_cifar_bic_boundary(self):
        self.assertTrue(hasattr(pilot, 'prepare_control_config'))
        base = {
            'data': 'cifar100', 'num_tasks': 10, 'custom_tasks': '',
            'bic_enabled': 1, 'party_drift_enabled': 1,
            'pilot_dataset': 'cifar100', 'seed': 73,
            'epochs_per_task': 1, 'device': 'cuda:1', 'num_workers': 2,
        }
        config = pilot.prepare_control_config(
            base, 'cifar100', 'cmcdr_budget', '/results')
        self.assertEqual(config['seed'], 74)
        self.assertEqual(config['num_tasks'], 2)
        self.assertEqual(config['bic_enabled'], 0)
        self.assertEqual(config['device'], 'cuda:1')
        self.assertEqual(config['party_replay_arm'], 'cmcdr_budget')
        self.assertEqual(config['party_replay_alpha'], 1.0)
        self.assertEqual(config['party_drift_preflight'], 1)
        self.assertEqual(config['output_dir'],
                         '/results/cifar100_cmcdr_budget_seed74')
        self.assertIn('0,1,2,3,4,5,6,7,8,9', config['custom_tasks'])
        self.assertNotIn('20,21,22,23,24,25,26,27,28,29',
                         config['custom_tasks'])

    def test_prepare_refuses_unlisted_or_candidate_arm(self):
        self.assertTrue(hasattr(pilot, 'prepare_control_config'))
        for arm in ('drift', 'bogus'):
            with self.assertRaises(ValueError):
                pilot.prepare_control_config({}, 'isolet', arm,
                                             Path('/results'))


if __name__ == '__main__':
    unittest.main()
