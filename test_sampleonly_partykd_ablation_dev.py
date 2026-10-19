import unittest

from run_sampleonly_partykd_ablation_dev import APPROVED_HASHES, derive_config


class ConfigDerivationTests(unittest.TestCase):
    def test_only_changes_partykd_and_output_identity(self):
        source = {
            'pilot_dataset': 'isolet', 'seed': 54,
            'num_parties': 4, 'sample_replay_only': 1,
            'resource_lite': 0, 'head_consolidation_samples_per_class': 20,
            'head_consolidation_mode': 'adaptive_dual_branch',
            'party_kd_enabled': 1, 'dep_tracking_enabled': 1,
            'party_kd_mode': 'uniform',
            'expected_party_kd_variant': 'uniform',
            'output_dir': '/old', 'results_dir': '/old',
            'exp_name': 'reference', 'epochs_per_task': 50,
        }
        derived = derive_config(source, APPROVED_HASHES['isolet', 54],
                                '/new')
        changed = {key for key in source if source[key] != derived[key]}
        self.assertEqual(changed, {
            'party_kd_enabled', 'dep_tracking_enabled',
            'expected_party_kd_variant', 'output_dir', 'results_dir',
            'exp_name',
        })
        self.assertEqual(derived['pilot_variant'], 'no_partykd20')
        with self.assertRaisesRegex(ValueError, 'source config hash'):
            derive_config(source, 'wrong', '/new')

    def test_task_marginal_keeps_party_kd_enabled(self):
        source = {
            'pilot_dataset': 'isolet', 'seed': 54,
            'num_parties': 4, 'sample_replay_only': 1,
            'resource_lite': 0, 'head_consolidation_samples_per_class': 20,
            'head_consolidation_mode': 'adaptive_dual_branch',
            'party_kd_enabled': 1, 'dep_tracking_enabled': 1,
            'party_kd_mode': 'uniform',
            'expected_party_kd_variant': 'uniform',
            'output_dir': '/old', 'results_dir': '/old',
            'exp_name': 'reference', 'epochs_per_task': 50,
        }
        derived = derive_config(source, APPROVED_HASHES['isolet', 54],
                                '/new', mode='task_marginal')
        self.assertEqual(derived['party_kd_enabled'], 1)
        self.assertEqual(derived['dep_tracking_enabled'], 1)
        self.assertEqual(derived['party_kd_mode'], 'task_marginal')
        self.assertEqual(derived['expected_party_kd_variant'],
                         'task_marginal')
        source.update(custom_tasks='0,1|2,3|4,5', num_tasks=3,
                      device='cuda:0')
        smoke = derive_config(source, APPROVED_HASHES['isolet', 54],
                              '/new', smoke=True, mode='task_marginal')
        self.assertEqual(smoke['num_tasks'], 3)
        self.assertEqual(smoke['custom_tasks'], source['custom_tasks'])
        self.assertEqual(smoke['epochs_per_task'], 1)
        self.assertEqual(smoke['device'], 'cpu')

    def test_fim_off_keeps_uniform_party_kd(self):
        source = {
            'pilot_dataset': 'isolet', 'seed': 54,
            'num_parties': 4, 'sample_replay_only': 1,
            'resource_lite': 0, 'head_consolidation_samples_per_class': 20,
            'head_consolidation_mode': 'adaptive_dual_branch',
            'party_kd_enabled': 1, 'dep_tracking_enabled': 1,
            'party_kd_mode': 'uniform',
            'expected_party_kd_variant': 'uniform',
            'fim_freeze_frac': 0.0,
            'output_dir': '/old', 'results_dir': '/old',
            'exp_name': 'reference', 'epochs_per_task': 50,
        }
        derived = derive_config(source, APPROVED_HASHES['isolet', 54],
                                '/new', mode='fim_off')
        self.assertEqual(derived['fim_freeze_frac'], 0.0)
        self.assertEqual(derived['party_kd_enabled'], 1)
        self.assertEqual(derived['expected_party_kd_variant'], 'uniform')
        self.assertEqual({key for key in source
                          if source[key] != derived[key]}, {
                              'output_dir', 'results_dir', 'exp_name',
                          })


if __name__ == '__main__':
    unittest.main()
