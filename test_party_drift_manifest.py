import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import party_drift_manifest as manifest


class DevelopmentManifestTests(unittest.TestCase):
    def test_source_paths_preserve_each_baseline_method(self):
        self.assertTrue(hasattr(manifest, 'source_config_path'))
        own = manifest.source_config_path('/sample', '/baseline',
                                          'cifar100', 'drift')
        self.assertEqual(str(own).replace('\\', '/'),
                         '/sample/full/cifar100/adaptive/42/run/config.json')
        der = manifest.source_config_path('/sample', '/baseline',
                                          'upmc_food101', 'derpp')
        self.assertEqual(str(der).replace('\\', '/'),
                         '/baseline/full-3080/upmc_food101/der_pp/42/run/config.json')

    def test_writer_hashes_exact_config_and_manifest_bytes(self):
        self.assertTrue(hasattr(manifest, 'write_manifest_files'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'new-manifest'
            config = {'development_job_key': 'isolet/drift/0.5/70',
                      'seed': 70}
            encoded = (json.dumps(config, sort_keys=True,
                                  separators=(',', ':')) + '\n').encode()
            row = {'job_key': config['development_job_key'],
                   'config_relpath': 'configs/isolet_drift_0p5_70.json',
                   'config_sha256': hashlib.sha256(encoded).hexdigest()}
            digest = manifest.write_manifest_files(root, [(row, config)])
            self.assertEqual(hashlib.sha256(
                (root / 'manifest.jsonl').read_bytes()).hexdigest(), digest)
            self.assertEqual((root / row['config_relpath']).read_bytes(), encoded)
            self.assertEqual((root / 'manifest.sha256').read_text().strip(), digest)
            with self.assertRaises(FileExistsError):
                manifest.write_manifest_files(root, [(row, config)])

    def test_job_config_derives_from_the_correct_frozen_method(self):
        self.assertTrue(hasattr(manifest, 'build_job_config'))
        own = manifest.build_job_config(
            {'seed': 42, 'cl_method': 'proto_evolve',
             'sample_replay_only': 1, 'epochs_per_task': 20,
             'vector_npz': '/data/upmc.npz',
             'formal_deferred_evaluation': True},
            dataset='upmc_food101', method='drift', alpha=0.5, seed=70,
            root='/results', source_config='/source.json',
            holdout_manifest='/holdout.json',
            paired_validation_manifest='/validation.json',
        )
        self.assertEqual(own['seed'], 70)
        self.assertEqual(own['party_replay_arm'], 'drift')
        self.assertEqual(own['party_replay_alpha'], 0.5)
        self.assertEqual(own['party_drift_enabled'], 1)
        self.assertEqual(own['pilot_holdout_seed'], 20261023)
        self.assertEqual(own['pilot_holdout_per_class'], 30)
        self.assertEqual(own['device'], 'cuda:0')
        baseline = manifest.build_job_config(
            {'seed': 42, 'cl_method': 'icarl', 'epochs_per_task': 20,
             'vector_npz': '/data/upmc.npz',
             'formal_deferred_evaluation': True},
            dataset='upmc_food101', method='icarl', alpha=None, seed=71,
            root='/results', source_config='/icarl.json',
            holdout_manifest='/holdout.json',
            paired_validation_manifest='/validation.json',
        )
        self.assertEqual(baseline['cl_method'], 'icarl')
        self.assertEqual(baseline['party_drift_enabled'], 0)
        self.assertEqual(baseline['party_replay_arm'], 'off')
        self.assertEqual(baseline['device'], 'cuda:1')
        with self.assertRaises(ValueError):
            manifest.build_job_config(
                {'seed': 42, 'cl_method': 'proto_evolve'},
                dataset='upmc_food101', method='drift', alpha=None, seed=70,
                root='/results', source_config='/source.json',
                holdout_manifest='/holdout.json',
                paired_validation_manifest='/validation.json')

    def test_expected_matrix_has_183_jobs_and_paired_baselines(self):
        self.assertTrue(hasattr(manifest, 'expected_job_keys'))
        keys = manifest.expected_job_keys()
        self.assertEqual(len(keys), 183)
        self.assertEqual(len(set(keys)), 183)
        self.assertIn('cifar100/er_ace/none/70', keys)
        self.assertIn('isolet/icarl/none/72', keys)
        self.assertIn('upmc_food101/derpp/none/71', keys)
        self.assertIn('upmc_food101/icarl/none/71', keys)
        self.assertIn('upmc_food101/drift/0.5/72', keys)

    def test_allowlist_rejects_changed_config_manifest_and_unlisted_key(self):
        self.assertTrue(hasattr(manifest, 'load_approved_job'))
        key = 'upmc_food101/drift/0.5/70'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'configs' / 'chosen.json'
            config.parent.mkdir()
            config.write_text(json.dumps({
                'development_job_key': key, 'seed': 70,
                'pilot_dataset': 'upmc_food101',
                'party_replay_arm': 'drift', 'party_replay_alpha': 0.5,
            }), encoding='utf-8')
            config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
            rows = []
            for job in sorted(manifest.expected_job_keys()):
                dataset, method, alpha, seed = job.split('/')
                rows.append({
                    'job_key': job,
                    'dataset': dataset, 'method': method,
                    'alpha': None if alpha == 'none' else float(alpha),
                    'seed': int(seed),
                    'config_relpath': 'configs/chosen.json' if job == key
                                      else 'configs/unused.json',
                    'config_sha256': config_hash if job == key else '0' * 64,
                    'source_config_sha256': '1' * 64,
                    'source_data_sha256': '2' * 64,
                    'holdout_sha256': '3' * 64,
                    'validation_manifest_sha256': '4' * 64,
                    'production_commit': 'a' * 40,
                })
            path = root / 'manifest.jsonl'
            path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n'
                                    for row in rows), encoding='utf-8')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            approved = manifest.load_approved_job(key, path, digest,
                                                  production_commit='a' * 40)
            self.assertEqual(approved['job_key'], key)
            with self.assertRaises(ValueError):
                manifest.load_approved_job('unlisted', path, digest,
                                           production_commit='a' * 40)
            config.write_text(config.read_text() + ' ', encoding='utf-8')
            with self.assertRaises(ValueError):
                manifest.load_approved_job(key, path, digest,
                                           production_commit='a' * 40)
            config.write_text(json.dumps({
                'development_job_key': key, 'seed': 70,
                'pilot_dataset': 'upmc_food101',
                'party_replay_arm': 'drift', 'party_replay_alpha': 0.5,
            }), encoding='utf-8')
            path.write_text(path.read_text() + '\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                manifest.load_approved_job(key, path, digest,
                                           production_commit='a' * 40)
            selected = next(row for row in rows if row['job_key'] == key)
            selected['config_relpath'] = '../outside.json'
            path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n'
                                    for row in rows), encoding='utf-8')
            changed_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                manifest.load_approved_job(key, path, changed_digest,
                                           production_commit='a' * 40)
            selected['config_relpath'] = 'configs/chosen.json'
            rows[-1] = rows[0]
            path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n'
                                    for row in rows), encoding='utf-8')
            changed_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                manifest.load_approved_job(key, path, changed_digest,
                                           production_commit='a' * 40)


if __name__ == '__main__':
    unittest.main()
