"""Read-only adaptive evidence audit against the isolated full-schedule pilot."""

import unittest
import tempfile
from pathlib import Path

from adaptive9_evidence import (
    verify_run, verify_pinned_job, validate_runtime)


RUN = Path('/brf/users/yangxx/VFCL-results/'
           'adaptive9-v2-fullschedule-preflight-20261015/run')
ISOLET_5090_RUN = Path('/brf/users/yangxx/VFCL-results/'
                       'adaptive9-isolet-5090-preflight-20261015/run')
CIFAR_5090_RUN = Path('/brf/users/yangxx/VFCL-results/'
                      'adaptive9-cifar-fullschedule-5090-preflight-20261015/run')


class AdaptiveRunEvidenceTests(unittest.TestCase):
    def test_cifar_adaptive_bic_uses_preinstall_validation_cache(self):
        if not (CIFAR_5090_RUN / 'results.json').is_file():
            self.skipTest('full-task CIFAR pilot is not complete')
        checked = verify_run(
            CIFAR_5090_RUN, dataset='cifar100', seed=42,
            expected_epochs=1,
            producer_commit='9973018295da270fd6098de07dcf36b9fe65ff61',
            producer_root=Path('/tmp/vfcl-adaptive9-v3'),
            data_root=Path('/brf/users/yangxx/VF-CL/data'))
        self.assertEqual(checked['admission'], 'pilot')
        self.assertGreater(checked['flow']['batches'], 0)

    def test_one_gpu_source_restores_on_two_gpu_audit_host(self):
        if not ISOLET_5090_RUN.is_dir():
            self.skipTest('cross-host ISOLET pilot is not mounted')
        checked = verify_run(
            ISOLET_5090_RUN, dataset='isolet', seed=42,
            expected_epochs=1,
            producer_commit='9973018295da270fd6098de07dcf36b9fe65ff61',
            producer_root=Path('/tmp/vfcl-adaptive9-v3'),
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
                producer_commit='bff98e74e9037e6676a0cf26da9ebc5420ebc6a9',
                producer_root=Path('/tmp/vfcl-adaptive9-v2'),
                data_root=Path('/brf/users/yangxx/VF-CL/data'))

    def test_four_party_pilot_replays_frozen_evidence_and_batches(self):
        if not RUN.is_dir():
            self.skipTest('isolated Linux pilot is not mounted')
        checked = verify_run(
            RUN, dataset='upmc_food101', seed=42, expected_epochs=1,
            producer_commit='bff98e74e9037e6676a0cf26da9ebc5420ebc6a9',
            producer_root=Path('/tmp/vfcl-adaptive9-v2'),
            data_root=Path('/brf/users/yangxx/VF-CL/data'))
        self.assertEqual(checked['admission'], 'pilot')
        self.assertEqual(checked['flow']['party_col_ranges'],
                         [(0, 256), (256, 512),
                          (512, 1024), (1024, 1536)])
        self.assertGreater(checked['flow']['batches'], 0)


if __name__ == '__main__':
    unittest.main()
