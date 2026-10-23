import unittest
import hashlib
import json
import tempfile
from pathlib import Path

import torch

import run_party_drift_development as dev


class DevelopmentReadoutTests(unittest.TestCase):
    def test_worker_rejects_unplanned_or_training_anomaly_retry(self):
        self.assertTrue(hasattr(dev, 'authorize_queue_attempt'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = 'isolet/drift/0.5/70'
            job = root / key
            job.mkdir(parents=True)
            (root / 'QUEUE_LOCK.json').write_text(json.dumps({
                'pid': 1234, 'manifest_sha256': 'm',
                'production_commit': 'c'}))
            events = root / 'ATTEMPTS.jsonl'
            events.write_text('')
            with self.assertRaises(ValueError):
                dev.authorize_queue_attempt(job, key, 1, 'm', 'c', 'h',
                                            queue_pid=1234)
            (job / 'attempt-01.plan.json').write_text(json.dumps({
                'job_key': key, 'attempt': 1}))
            events.write_text(json.dumps({
                'kind': 'start', 'job_key': key, 'attempt': 1,
                'config_sha256': 'h'}) + '\n')
            dev.authorize_queue_attempt(job, key, 1, 'm', 'c', 'h',
                                        queue_pid=1234)
            with self.assertRaises(ValueError):
                dev.authorize_queue_attempt(job, key, 1, 'm', 'c', 'h',
                                            queue_pid=9999)
            events.write_text(events.read_text() + json.dumps({
                'kind': 'finish', 'job_key': key, 'attempt': 1,
                'classification': 'training_anomaly'}) + '\n' +
                json.dumps({'kind': 'start', 'job_key': key, 'attempt': 2,
                            'config_sha256': 'h'}) + '\n')
            (job / 'attempt-02.plan.json').write_text(json.dumps({
                'job_key': key, 'attempt': 2}))
            with self.assertRaises(ValueError):
                dev.authorize_queue_attempt(job, key, 2, 'm', 'c', 'h',
                                            queue_pid=1234)
            events.write_text(events.read_text().replace(
                'training_anomaly', 'infrastructure'))
            (job / 'attempt-02.resume.json').write_text(json.dumps({
                'job_key': key, 'attempt': 2, 'manifest_sha256': 'm',
                'reason': 'host reboot'}))
            dev.authorize_queue_attempt(job, key, 2, 'm', 'c', 'h',
                                        queue_pid=1234)

    def test_bic_fit_loader_uses_method_specific_reserved_split(self):
        self.assertTrue(hasattr(dev, 'bic_fit_loader'))

        class Dataset:
            validationset = object()
            calibrationset = object()
            validation_indices = {1, 3}
            calibration_indices = {2, 4}
            holdout_indices = {5}
            def get_class_indices(self, view, classes):
                return [1, 2, 3, 4, 5]
            def _loader(self, view, indices, key, shuffle, audit=False):
                return (view, indices, key, shuffle, audit)

        from types import SimpleNamespace
        dataset = Dataset()
        adaptive = SimpleNamespace(cl_method='proto_evolve',
                                   bic_enabled=1, lambda_validation_enabled=1,
                                   lambda_validation_per_class=1,
                                   bic_per_class=1, num_classes=2)
        baseline = SimpleNamespace(**{**vars(adaptive), 'cl_method': 'er_ace'})
        loader, source = dev.bic_fit_loader(adaptive, dataset)
        self.assertEqual(source, 'validation')
        self.assertEqual(loader[0], dataset.validationset)
        self.assertEqual(loader[1], [1, 3])
        self.assertTrue(loader[4])
        loader, source = dev.bic_fit_loader(baseline, dataset)
        self.assertEqual(source, 'calibration')
        self.assertEqual(loader[0], dataset.calibrationset)
        self.assertEqual(loader[1], [2, 4])

    def test_cifar_bic_readout_fits_frozen_joint_rule_on_calibration_only(self):
        self.assertTrue(hasattr(dev, 'score_bic_logits'))
        calibration_logits = torch.tensor([[1.0, 2.0], [1.0, 2.0],
                                           [0.0, 3.0], [0.0, 3.0]])
        calibration_labels = torch.tensor([0, 0, 1, 1])
        holdout_logits = calibration_logits.clone()
        holdout_labels = calibration_labels.clone()
        result = dev.score_bic_logits(
            calibration_logits, calibration_labels,
            holdout_logits, holdout_labels, {0: [0], 1: [1]},
            lr=0.05, steps=200)
        self.assertEqual(result['raw']['cil_correct'], 2)
        self.assertEqual(result['calibrated']['cil_correct'], 4)
        self.assertEqual(result['calibrated']['count'], 4)

    def test_runtime_config_matches_attempt_path_and_preserves_approved_bytes(self):
        self.assertTrue(hasattr(dev, 'write_attempt_configs'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'job'
            approved = Path(tmp) / 'approved.json'
            approved.write_text(json.dumps({
                'output_dir': str(root), 'seed': 70,
            }), encoding='utf-8')
            args = dev.attempt_args(json.loads(approved.read_text()), 2)
            run = Path(args.output_dir)
            run.mkdir(parents=True)
            dev.write_attempt_configs(run, approved, args)
            self.assertEqual((run / 'APPROVED_CONFIG.json').read_bytes(),
                             approved.read_bytes())
            runtime = json.loads((run / 'config.json').read_text())
            self.assertEqual(runtime['output_dir'], str(run))
            self.assertEqual(runtime['results_dir'], str(root))
            self.assertEqual(runtime['seed'], 70)

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
