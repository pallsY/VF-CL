import unittest
import json
import hashlib
import tempfile
from pathlib import Path

from f_off_formal_manifest import ROOT, build_config, expected_keys, validate_stage_a, load_approved_job


class FormalManifestTests(unittest.TestCase):
    def test_stage_a_history_must_be_closed_and_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'decision.json'
            path.write_text(json.dumps({'stage': 'A', 'status': 'direction_closed',
                                        'selected_policy': None,
                                        'official_test_accessed': False}), encoding='utf-8')
            from f_off_formal_manifest import digest
            self.assertEqual(validate_stage_a(path, digest(path)), digest(path))
            path.write_text(path.read_text(encoding='utf-8').replace(
                'direction_closed', 'selection_passed'), encoding='utf-8')
            with self.assertRaises(ValueError):
                validate_stage_a(path, digest(path))

    def test_exact_nine_jobs(self):
        keys = expected_keys()
        self.assertEqual(len(keys), 9)
        self.assertIn('cifar100/f_off/0.25/42', keys)
        self.assertIn('isolet/f_off/0.25/43', keys)
        self.assertIn('upmc_food101/f_off/0.25/44', keys)

    def test_unlisted_job_is_rejected(self):
        import f_off_formal_manifest as m
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / 'manifest.jsonl'
            rows = [{'job_key': key, 'stage': 'formal_followup',
                     'code_commit': 'a' * 40,
                     'stage_a_decision_sha256': m.STAGE_A_SHA}
                    for key in sorted(expected_keys())]
            manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows),
                                encoding='utf-8')
            with self.assertRaises(ValueError):
                load_approved_job('isolet/f0/0.25/42', manifest,
                                  m.digest(manifest), 'a' * 40)

    def test_sealed_manifest_rejects_tampered_config(self):
        import f_off_formal_manifest as m
        from party_drift_manifest import _canonical_bytes, write_manifest_files
        with tempfile.TemporaryDirectory() as folder:
            entries = []
            for index, key in enumerate(sorted(expected_keys())):
                dataset, _, _, seed_text = key.split('/')
                seed = int(seed_text)
                source = {
                    'seed': seed, 'cl_method': 'proto_evolve',
                    'formal_deferred_evaluation': True, 'sample_replay_only': 1,
                    'num_parties': 4, 'expected_party_kd_variant': 'uniform',
                    'epochs_per_task': m.EPOCHS[dataset], 'fim_freeze_frac': 0.0,
                    'party_kd_enabled': 1, 'party_kd_lambda': 1.0,
                    'head_consolidation_samples_per_class': 20,
                    'proto_replay_gradient_enabled': 0, 'replay_mode': 'prototype',
                }
                config = build_config(source, dataset, seed, index % 2)
                row = {
                    'job_key': key, 'stage': 'formal_followup',
                    'code_commit': 'a' * 40,
                    'stage_a_decision_sha256': m.STAGE_A_SHA,
                    'dataset': dataset, 'seed': seed, 'physical_gpu': index % 2,
                    'config_relpath': 'configs/' + key.replace('/', '_').replace('.', 'p') + '.json',
                    'config_sha256': hashlib.sha256(_canonical_bytes(config)).hexdigest(),
                }
                entries.append((row, config))
            output = Path(folder) / 'manifest'
            write_manifest_files(output, entries)
            manifest = output / 'manifest.jsonl'
            key = sorted(expected_keys())[0]
            row = load_approved_job(key, manifest, m.digest(manifest), 'a' * 40)
            config_path = output / row['config_relpath']
            config_path.write_text(config_path.read_text(encoding='utf-8') + ' ',
                                   encoding='utf-8')
            with self.assertRaises(ValueError):
                load_approved_job(key, manifest, m.digest(manifest), 'a' * 40)

    def test_full_training_config_keeps_frozen_budget(self):
        source = {
            'seed': 43, 'cl_method': 'proto_evolve',
            'formal_deferred_evaluation': True, 'sample_replay_only': 1,
            'num_parties': 4, 'expected_party_kd_variant': 'uniform',
            'epochs_per_task': 50, 'fim_freeze_frac': 0.0,
            'party_kd_enabled': 1, 'party_kd_lambda': 1.0,
            'head_consolidation_samples_per_class': 20,
            'proto_replay_gradient_enabled': 0, 'replay_mode': 'prototype',
            'formal_source_provenance': {'old': 'value'},
        }
        config = build_config(source, 'isolet', 43, 0)
        self.assertEqual(config['fim_mask_policy'], 'off')
        self.assertEqual(config['party_replay_arm'], 'all_party')
        self.assertEqual(config['party_replay_alpha'], 0.25)
        self.assertEqual(config['epochs_per_task'], 50)
        self.assertEqual(config['output_dir'], str(ROOT / 'formal/isolet/f_off/0.25/43'))
        self.assertNotIn('formal_source_provenance', config)
        self.assertNotIn('development_holdout_manifest', config)
        source['epochs_per_task'] = 20
        with self.assertRaises(ValueError):
            build_config(source, 'isolet', 43, 0)


if __name__ == '__main__':
    unittest.main()
