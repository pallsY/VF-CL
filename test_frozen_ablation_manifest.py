import unittest

from frozen_ablation_manifest import (auxiliary_data_paths, derive_config,
                                       expected_keys, semantic_diff)


class FrozenManifestTests(unittest.TestCase):
    def setUp(self):
        self.source = {
            'seed': 42, 'cl_method': 'proto_evolve',
            'formal_deferred_evaluation': True,
            'sample_replay_only': 1,
            'head_consolidation_samples_per_class': 20,
            'epochs_per_task': 50,
            'party_kd_enabled': 1,
            'dep_tracking_enabled': 1,
            'expected_party_kd_variant': 'uniform',
            'fim_freeze_frac': 0.0,
            'results_dir': '/original/isolet/adaptive/42',
            'output_dir': '/original/isolet/adaptive/42/run',
            'exp_name': 'run',
            'device': 'cuda:0',
        }

    def test_exact_eighteen_job_keys(self):
        self.assertEqual(len(expected_keys()), 18)
        self.assertIn('isolet/no_partykd/42', expected_keys())
        self.assertIn('cifar100/fim_off/44', expected_keys())

    def test_no_party_kd_has_only_three_semantic_differences(self):
        target = derive_config(self.source, 'isolet', 'no_partykd', 42)
        self.assertEqual(semantic_diff(self.source, target), {
            'party_kd_enabled': [1, 0],
            'dep_tracking_enabled': [1, 0],
            'expected_party_kd_variant': ['uniform', 'base'],
        })
        self.assertEqual(target['device'], 'cuda:0')
        self.assertEqual(self.source['party_kd_enabled'], 1)

    def test_fim_off_only_adds_explicit_switch(self):
        target = derive_config(self.source, 'isolet', 'fim_off', 42)
        self.assertEqual(semantic_diff(self.source, target), {
            'fim_apply_mask': [None, 0],
        })
        self.assertEqual(target['fim_freeze_frac'], 0.0)
        with self.assertRaises(ValueError):
            derive_config({**self.source, 'seed': 43}, 'isolet', 'fim_off', 42)

    def test_formal_data_inventory_includes_test_and_metadata(self):
        cifar = auxiliary_data_paths({'data_path': '/data'}, 'cifar100')
        self.assertEqual(set(cifar), {'meta', 'test'})
        isolet = auxiliary_data_paths(
            {'vector_npz': '/data/isolet/isolet_vfl.npz'}, 'isolet')
        self.assertEqual(isolet['metadata'].name,
                         'isolet_vfl.metadata.json')


if __name__ == '__main__':
    unittest.main()
