import unittest

import torch

from analyze_isolet_online_head_memory import (
    memory_stats, passes_capacity, validate_provenance_fields,
)


class OnlineHeadMemoryTests(unittest.TestCase):
    def test_counts_persistent_raw_bytes_and_locked_gain_rule(self):
        replay = {0: torch.zeros(4, 3), 1: torch.ones(4, 3)}
        self.assertEqual(memory_stats(replay), {
            'classes': 2, 'examples_per_class': 4,
            'total_examples': 8, 'raw_bytes': 96,
        })
        self.assertTrue(passes_capacity([
            {'cil_delta': 0.012, 'new_cil_delta': -0.005},
            {'cil_delta': 0.009, 'new_cil_delta': 0.0},
        ]))
        self.assertFalse(passes_capacity([
            {'cil_delta': 0.02, 'new_cil_delta': -0.02},
            {'cil_delta': 0.01, 'new_cil_delta': 0.0},
        ]))
        self.assertFalse(passes_capacity([
            {'cil_delta': 0.02, 'new_cil_delta': 0.0},
            {'cil_delta': -0.001, 'new_cil_delta': 0.0},
        ]))

    def test_rejects_wrong_producer_and_test_access(self):
        protocol = {
            'source_commit': 'a575bbf446ae501cf8e580ba62c2cc5492a25f30',
            'source_config_sha256': '5702b8846ca8e9d279728a725550c1b053b8580fec832a9695b56d6ea7f2af34',
            'source_record_sha256': '7dc5bf930600d097134948d6ee166fd55490327d16cde1fe2496ac374a73299b',
            'launcher_sha256': '419dc05641a90bc4834ae9ca4f4bbb8079a8226df1226c7b72019a691cd3043f',
            'data_sha256': {
                'data:isolet/isolet_vfl.npz': 'd34312670de93198afcd2b126c95b79bae2b4cffeb30d480f097faf046b69514',
                'data:isolet/isolet_vfl.metadata.json': '79396dea1751b6094a5769f2dd789ad58b3ea9582a12c8c98d3ec07d8d1eb3cd',
            },
            'holdout_split_seed': 20261012, 'selector': 'herding',
            'raw_replay_capacity_per_class': 40,
            'adaptive_contract_capacity_override': True,
        }
        done = {'status': 'training_only_before_test'}
        config = {
            'seed': 51, 'head_consolidation_samples_per_class': 40,
            'proto_lambda_a': 0.05, 'distill_weight': 0.10,
            'feat_distill_weight': 0.02, 'formal_deferred_evaluation': False,
        }
        validate_provenance_fields(protocol, done, config, 51, 40,
                                   [{'split': 'train'}])
        with self.assertRaises(ValueError):
            validate_provenance_fields({**protocol, 'source_commit': 'other'},
                                       done, config, 51, 40, [])
        with self.assertRaises(ValueError):
            validate_provenance_fields(protocol, done, config, 51, 40,
                                       [{'split': 'test'}])
if __name__ == '__main__':
    unittest.main()
