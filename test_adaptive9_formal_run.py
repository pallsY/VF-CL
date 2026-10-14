"""Read-only adaptive evidence audit against the isolated full-schedule pilot."""

import unittest
import tempfile
from pathlib import Path

from adaptive9_evidence import verify_run, verify_pinned_job


RUN = Path('/brf/users/yangxx/VFCL-results/'
           'adaptive9-v2-fullschedule-preflight-20261015/run')


class AdaptiveRunEvidenceTests(unittest.TestCase):
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
