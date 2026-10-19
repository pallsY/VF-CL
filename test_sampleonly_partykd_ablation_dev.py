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


if __name__ == '__main__':
    unittest.main()
