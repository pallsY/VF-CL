import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from replay_fim_formal import authorized_policy, build_formal_config, formal_job_keys


class FormalGateTests(unittest.TestCase):
    def test_formal_allowlist_is_exactly_nine_jobs(self):
        keys = formal_job_keys('f1')
        self.assertEqual(len(keys), 9)
        self.assertIn('cifar100/f1/0.25/42', keys)
        self.assertIn('upmc_food101/f1/0.25/44', keys)
        with self.assertRaises(ValueError):
            formal_job_keys('f2')

    def test_formal_config_uses_full_data_and_locked_budget(self):
        source = {
            'seed': 43, 'cl_method': 'proto_evolve',
            'formal_deferred_evaluation': True, 'sample_replay_only': 1,
            'epochs_per_task': 50, 'fim_freeze_frac': 0.0,
            'party_kd_enabled': 1, 'party_kd_lambda': 1.0,
            'head_consolidation_samples_per_class': 20,
            'proto_replay_gradient_enabled': 0, 'replay_mode': 'prototype',
            'formal_source_provenance': {'old': 'value'},
        }
        config = build_formal_config(source, 'isolet', 'f1', 43,
                                     '/brf/users/yangxx/VFCL-results/allparty-fim-final-attempt-v1-20261029/formal', 1)
        self.assertEqual(config['fim_mask_policy'], 'fresh')
        self.assertEqual(config['party_replay_arm'], 'all_party')
        self.assertEqual(config['party_replay_alpha'], 0.25)
        self.assertEqual(config['epochs_per_task'], 50)
        self.assertEqual(config['seed'], 43)
        self.assertEqual(config['study_physical_gpu'], 1)
        self.assertNotIn('formal_source_provenance', config)
        self.assertNotIn('development_holdout_manifest', config)
        with self.assertRaises(ValueError):
            build_formal_config(source, 'isolet', 'f1', 43,
                                '/tmp/formal', 1)

    def test_both_audited_decisions_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = root / 'A.json'
            b = root / 'B.json'
            a.write_text(json.dumps({
                'stage': 'A', 'status': 'selection_passed',
                'selected_policy': 'f1', 'official_test_accessed': False,
            }), encoding='utf-8')
            b.write_text(json.dumps({
                'stage': 'B', 'status': 'party_kd_supported',
                'selected_policy': 'f1', 'official_test_accessed': False,
                'stage_a_decision_sha256': hashlib.sha256(a.read_bytes()).hexdigest(),
            }), encoding='utf-8')
            self.assertEqual(authorized_policy(a, b), 'f1')
            b.write_text(b.read_text(encoding='utf-8').replace(
                'party_kd_supported', 'direction_closed'), encoding='utf-8')
            with self.assertRaises(ValueError):
                authorized_policy(a, b)
            b.write_text(b.read_text(encoding='utf-8').replace(
                'direction_closed', 'party_kd_supported'), encoding='utf-8')
            a.write_text(a.read_text(encoding='utf-8') + ' ', encoding='utf-8')
            with self.assertRaises(ValueError):
                authorized_policy(a, b)


if __name__ == '__main__':
    unittest.main()
