import unittest
import hashlib
import json
import tempfile
from pathlib import Path

import torch

import run_party_drift_development as dev


class DevelopmentReadoutTests(unittest.TestCase):
    def test_attempt_changes_only_output_location_not_locked_config(self):
        self.assertTrue(hasattr(dev, 'attempt_args'))
        config = {'output_dir': '/results/isolet/drift/0.5/70',
                  'seed': 70, 'party_replay_alpha': 0.5}
        first = dev.attempt_args(config, 1)
        second = dev.attempt_args(config, 2)
        self.assertEqual(first.output_dir,
                         '/results/isolet/drift/0.5/70/attempt-01')
        self.assertEqual(second.output_dir,
                         '/results/isolet/drift/0.5/70/attempt-02')
        self.assertEqual(first.seed, second.seed)
        self.assertEqual(first.party_replay_alpha, second.party_replay_alpha)
        self.assertEqual(config['output_dir'], '/results/isolet/drift/0.5/70')
        with self.assertRaises(ValueError):
            dev.attempt_args(config, 0)

    def test_source_files_are_hashed_before_training(self):
        self.assertTrue(hasattr(dev, 'validate_job_sources'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.json'
            data = root / 'data.bin'
            holdout = root / 'holdout.json'
            paired = root / 'paired_validation.json'
            source.write_text('{"seed":42}', encoding='utf-8')
            data.write_bytes(b'exact training data')
            holdout.write_text('{"sha256":"a"}', encoding='utf-8')
            paired.write_text('{"sha256":"b"}', encoding='utf-8')
            sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            config = {
                'development_source_config_path': str(source),
                'development_source_data_path': str(data),
                'development_holdout_manifest': str(holdout),
                'pilot_paired_validation_manifest': str(paired),
                'development_job_key': 'isolet/drift/0.5/70',
                'pilot_protocol': 'party-drift-development-v1',
            }
            row = {'job_key': config['development_job_key'],
                   'source_config_sha256': sha(source),
                   'source_data_sha256': sha(data),
                   'holdout_sha256': sha(holdout),
                   'validation_manifest_sha256': sha(paired)}
            dev.validate_job_sources(config, row)
            data.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                dev.validate_job_sources(config, row)

    def test_icarl_uses_nearest_means_not_linear_head(self):
        self.assertTrue(hasattr(dev, 'score_icarl_holdout'))

        class Method:
            def _class_means(self):
                return torch.tensor([0, 1]), torch.tensor([[1.0, 0.0],
                                                          [-1.0, 0.0]])

            def _features(self, examples, record_comm=False):
                return examples

        class Dataset:
            def holdout_loader(self, classes):
                yield (torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
                       torch.tensor([0, 1]))

        metrics = dev.score_icarl_holdout(Method(), Dataset(),
                                          {0: [0], 1: [1]}, 2)
        self.assertEqual(metrics['cil'], 1.0)
        self.assertEqual(metrics['count'], 2)
        self.assertEqual(metrics['cil_correct'], 2)


if __name__ == '__main__':
    unittest.main()
