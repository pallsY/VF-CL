"""Read-only adaptive evidence audit against the isolated full-schedule pilot."""

import unittest
import tempfile
import json
from pathlib import Path

from adaptive9_evidence import (
    verify_run, verify_pinned_job, validate_runtime,
    verify_adaptive_access_flow)


RUN = Path('/brf/users/yangxx/VFCL-results/'
           'sampleonly-upmc_food101-preflight-20261017/'
           'upmc_food101/adaptive/42/run')
ISOLET_5090_RUN = Path('/brf/users/yangxx/VFCL-results/'
                       'sampleonly-isolet-preflight-20261017/'
                       'isolet/adaptive/42/run')
CIFAR_5090_RUN = Path('/brf/users/yangxx/VFCL-results/'
                      'sampleonly-cifar100-preflight-20261017/'
                      'cifar100/adaptive/42/run')


class AdaptiveRunEvidenceTests(unittest.TestCase):
    def test_validation_access_marker_must_match_the_flow(self):
        with tempfile.TemporaryDirectory() as scratch:
            run = Path(scratch)
            access_dir = run / 'formal_access'
            access_dir.mkdir()
            classes = [0, 1]
            common = {
                'event': 'first_iteration', 'event_idx': 0,
                'task_id': 0, 'timeline_step': 'event_0_CIL',
                'classes': classes,
            }
            validation = dict(
                common, loader_key=repr(('lambda_validation',
                                         tuple(classes))),
                split='validation', phase='final_validation_pre_install')
            test = dict(common, loader_key=repr(('test', tuple(classes))),
                        split='test', phase='final_test_post_install')
            for name, access in (('validation', validation), ('test', test)):
                (access_dir / f'{name}.consumed.json').write_text(
                    json.dumps({'schema_version': 1, 'status': 'consumed',
                                'access': access}), encoding='utf-8')
            flow = run / 'data_flow_audit.jsonl'
            flow.write_text(''.join(json.dumps(row) + '\n'
                                    for row in (validation, test)),
                            encoding='utf-8')
            self.assertEqual(
                verify_adaptive_access_flow(run, 1, 2),
                {'validation': validation, 'test': test})
            extra = access_dir / 'calibration.consumed.json'
            extra.write_text(json.dumps({
                'schema_version': 1, 'status': 'consumed',
                'access': dict(validation, split='calibration')}),
                encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'access inventory'):
                verify_adaptive_access_flow(run, 1, 2)
            extra.unlink()
            flow.write_text(json.dumps(test) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'validation access'):
                verify_adaptive_access_flow(run, 1, 2)
            flow.write_text(''.join(json.dumps(row) + '\n' for row in (
                dict(validation, phase='wrong'), test)), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'validation access'):
                verify_adaptive_access_flow(run, 1, 2)

    def test_cifar_adaptive_bic_uses_preinstall_validation_cache(self):
        if not (CIFAR_5090_RUN / 'results.json').is_file():
            self.skipTest('full-task CIFAR pilot is not complete')
        checked = verify_run(
            CIFAR_5090_RUN, dataset='cifar100', seed=42,
            expected_epochs=1,
            producer_commit='e2a559dc4945729848225cf21fdc95451762c231',
            producer_root=Path('/tmp/vfcl-sampleonly-producer-preflight-v1'),
            data_root=Path('/brf/users/yangxx/VF-CL/data'))
        self.assertEqual(checked['admission'], 'pilot')
        self.assertGreater(checked['flow']['batches'], 0)

    def test_one_gpu_source_restores_on_two_gpu_audit_host(self):
        if not ISOLET_5090_RUN.is_dir():
            self.skipTest('cross-host ISOLET pilot is not mounted')
        checked = verify_run(
            ISOLET_5090_RUN, dataset='isolet', seed=42,
            expected_epochs=1,
            producer_commit='e2a559dc4945729848225cf21fdc95451762c231',
            producer_root=Path('/tmp/vfcl-sampleonly-producer-preflight-v1'),
            data_root=Path('/brf/users/yangxx/VF-CL/data'))
        self.assertEqual(checked['admission'], 'pilot')
        self.assertEqual(len(checked['flow']['party_col_ranges']), 4)

    def test_remote_runtime_requires_pinned_host_versions_and_interpreter(self):
        runtime = {
            'host': 'c5090.upcbrfed.xyz', 'python': '3.11.15',
            'torch': '2.5.1+cu124', 'torchvision': '0.20.1+cu124',
            'numpy': '1.24.4', 'scikit_learn': '1.9.0',
            'pillow': '11.3.0', 'cuda_runtime': '12.4',
        }
        interpreter = '/var/tmp/vfcl-adaptive9-yangxx/venv/bin/python'
        self.assertEqual(validate_runtime(runtime, interpreter), runtime)
        with self.assertRaisesRegex(ValueError, 'runtime'):
            validate_runtime(dict(runtime, numpy='1.26.4'), interpreter)
        with self.assertRaisesRegex(ValueError, 'runtime'):
            validate_runtime(runtime, '/tmp/unreviewed-python')

    def test_formal_job_must_be_under_pinned_root(self):
        with tempfile.TemporaryDirectory() as scratch:
            original = Path(scratch) / 'other/isolet/adaptive/42'
            original.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'pinned'):
                verify_pinned_job(
                    original, results_root=Path(scratch) / 'formal',
                    data_root=Path(scratch) / 'data',
                    producer_root=Path(scratch) / 'producer',
                    producer_commit='0' * 40)

    def test_one_epoch_pilot_cannot_claim_formal_budget(self):
        if not RUN.is_dir():
            self.skipTest('isolated Linux pilot is not mounted')
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            verify_run(
                RUN, dataset='upmc_food101', seed=42,
                expected_epochs=20,
                producer_commit='e2a559dc4945729848225cf21fdc95451762c231',
                producer_root=Path('/tmp/vfcl-sampleonly-producer-preflight-v1'),
                data_root=Path('/brf/users/yangxx/VF-CL/data'))

    def test_four_party_pilot_replays_frozen_evidence_and_batches(self):
        if not RUN.is_dir():
            self.skipTest('isolated Linux pilot is not mounted')
        checked = verify_run(
            RUN, dataset='upmc_food101', seed=42, expected_epochs=1,
            producer_commit='e2a559dc4945729848225cf21fdc95451762c231',
            producer_root=Path('/tmp/vfcl-sampleonly-producer-preflight-v1'),
            data_root=Path('/brf/users/yangxx/VF-CL/data'))
        self.assertEqual(checked['admission'], 'pilot')
        self.assertEqual(checked['flow']['party_col_ranges'],
                         [(0, 256), (256, 512),
                          (512, 1024), (1024, 1536)])
        self.assertGreater(checked['flow']['batches'], 0)


if __name__ == '__main__':
    unittest.main()
