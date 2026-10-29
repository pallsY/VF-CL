import unittest
import hashlib
import json
import tempfile
from pathlib import Path

import replay_fim_manifest as m


class ReplayFIMManifestTests(unittest.TestCase):
    def test_exact_stage_allowlists(self):
        first = m.expected_job_keys('A')
        self.assertEqual(len(first), 65)
        self.assertIn('isolet/f1/0.25/84', first)
        self.assertIn('upmc_food101/derpp/none/80', first)
        self.assertNotIn('isolet/no_partykd_f1/0.25/84', first)
        second = m.expected_job_keys('B', selected_policy='f1')
        self.assertEqual(len(second), 15)
        self.assertEqual(second, {
            f'{dataset}/no_partykd_f1/0.25/{seed}'
            for dataset in m.DATASETS for seed in m.SEEDS
        })

    def test_fim_variants_only_change_mask_policy(self):
        source = {
            'seed': 42, 'cl_method': 'proto_evolve',
            'sample_replay_only': 1, 'formal_deferred_evaluation': True,
            'party_kd_enabled': 1, 'dep_tracking_enabled': 1,
            'expected_party_kd_variant': 'uniform',
            'epochs_per_task': 50, 'vector_npz': '/data/isolet.npz',
            'fim_freeze_frac': 0.0, 'party_kd_lambda': 1.0,
            'head_consolidation_samples_per_class': 20,
            'proto_replay_gradient_enabled': 0,
            'replay_mode': 'prototype',
        }
        variants = [m.build_job_config(
            source, dataset='isolet', method=method, seed=80,
            root='/new', source_config='/source.json',
            holdout_manifest='/holdout.json',
            paired_validation_manifest='/validation.json',
        ) for method in ('f0', 'f1', 'f_off')]
        for config, policy in zip(variants, ('cumulative', 'fresh', 'off')):
            self.assertEqual(config['fim_mask_policy'], policy)
            self.assertEqual(config['party_replay_arm'], 'all_party')
            self.assertEqual(config['party_replay_alpha'], 0.25)
            self.assertEqual(config['device'], 'cuda:0')
            self.assertEqual(config['pilot_holdout_per_class'], 30)
        excluded = {'development_job_key', 'output_dir', 'results_dir',
                    'exp_name', 'fim_mask_policy', 'pilot_variant'}
        self.assertEqual({k: v for k, v in variants[0].items()
                          if k not in excluded},
                         {k: v for k, v in variants[1].items()
                          if k not in excluded})

    def test_no_party_kd_keeps_selected_fim_and_replay(self):
        source = {
            'seed': 42, 'cl_method': 'proto_evolve',
            'sample_replay_only': 1, 'formal_deferred_evaluation': True,
            'party_kd_enabled': 1, 'dep_tracking_enabled': 1,
            'expected_party_kd_variant': 'uniform',
            'epochs_per_task': 50, 'vector_npz': '/data/isolet.npz',
            'fim_freeze_frac': 0.0, 'party_kd_lambda': 1.0,
            'head_consolidation_samples_per_class': 20,
            'proto_replay_gradient_enabled': 0,
            'replay_mode': 'prototype',
        }
        config = m.build_job_config(
            source, dataset='isolet', method='no_partykd_f1', seed=81,
            root='/new', source_config='/source.json',
            holdout_manifest='/holdout.json',
            paired_validation_manifest='/validation.json',
        )
        self.assertEqual(config['fim_mask_policy'], 'fresh')
        self.assertEqual(config['party_replay_arm'], 'all_party')
        self.assertEqual(config['party_kd_enabled'], 0)
        self.assertEqual(config['expected_party_kd_variant'], 'base')
        self.assertEqual(config['dep_tracking_enabled'], 0)

    def test_upmc_keeps_20_epochs_and_rejects_50(self):
        source = {
            'seed': 42, 'cl_method': 'proto_evolve',
            'sample_replay_only': 1, 'formal_deferred_evaluation': True,
            'party_kd_enabled': 1, 'party_kd_lambda': 1.0,
            'fim_freeze_frac': 0.0, 'epochs_per_task': 20,
            'head_consolidation_samples_per_class': 20,
            'proto_replay_gradient_enabled': 0,
            'replay_mode': 'prototype', 'vector_npz': '/data/upmc.npz',
        }
        arguments = dict(dataset='upmc_food101', method='f0', seed=80,
                         root='/new', source_config='/source.json',
                         holdout_manifest='/holdout.json',
                         paired_validation_manifest='/validation.json')
        self.assertEqual(m.build_job_config(source, **arguments)['epochs_per_task'], 20)
        with self.assertRaises(ValueError):
            m.build_job_config({**source, 'epochs_per_task': 50}, **arguments)

    def test_load_requires_exact_65_configs_and_external_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configs = root / 'configs'
            configs.mkdir()
            rows = []
            for index, key in enumerate(sorted(m.expected_job_keys('A'))):
                dataset, method, _, seed = key.split('/')
                config = {'development_job_key': key, 'seed': int(seed),
                          'pilot_protocol': m.PROTOCOL,
                          'party_replay_arm': ('all_party' if method in m.POLICIES
                                               else 'off'),
                          'party_replay_alpha': (0.25 if method in m.POLICIES
                                                 else 1.0),
                          'fim_mask_policy': m.POLICIES.get(method, 'cumulative'),
                          'party_kd_enabled': 1,
                          'study_physical_gpu': index % 2}
                path = configs / f'{index}.json'
                path.write_text(json.dumps(config), encoding='utf-8')
                rows.append({'job_key': key, 'dataset': dataset,
                             'method': method, 'seed': int(seed),
                             'stage': 'A', 'physical_gpu': index % 2,
                             'code_commit': 'a' * 40,
                             'source_commit': m.SOURCE_COMMIT,
                             'config_relpath': f'configs/{index}.json',
                             'config_sha256': hashlib.sha256(
                                 path.read_bytes()).hexdigest(),
                             **{field: 'b' * 64 for field in (
                                 'source_config_sha256', 'source_data_sha256',
                                 'holdout_sha256',
                                 'validation_manifest_sha256')}})
            manifest = root / 'manifest.jsonl'
            manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows),
                                encoding='utf-8')
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            self.assertEqual(m.load_approved_job(
                rows[0]['job_key'], manifest, digest, code_commit='a' * 40,
                stage='A')['job_key'], rows[0]['job_key'])
            with self.assertRaises(ValueError):
                m.load_approved_job('isolet/f1/0.25/99', manifest, digest,
                                    code_commit='a' * 40, stage='A')
            (configs / '0.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                m.load_approved_job(rows[0]['job_key'], manifest, digest,
                                    code_commit='a' * 40, stage='A')
            manifest.write_text(manifest.read_text() + '\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                m.load_approved_job(rows[0]['job_key'], manifest, digest,
                                    code_commit='a' * 40, stage='A')


if __name__ == '__main__':
    unittest.main()
