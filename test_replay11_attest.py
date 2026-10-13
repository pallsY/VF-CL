"""Audit ledger tests: failures remain visible and seed 42 is gated."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import replay11_attest as attest

TREE_A = {'sha256': 'a' * 64, 'files': 5, 'bytes': 100, 'error': None}
TREE_B = {'sha256': 'b' * 64, 'files': 5, 'bytes': 100, 'error': None}


class AuditLedgerTests(unittest.TestCase):
    def test_append_keeps_failure_and_hash_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / 'attestations.jsonl'
            first = attest.append_event(ledger, {'status': 'failed', 'job': 'isolet/mir/42'})
            second = attest.append_event(ledger, {'status': 'passed', 'job': 'isolet/mir/42'})
            self.assertEqual([row['status'] for row in attest.read_ledger(ledger)],
                             ['failed', 'passed'])
            self.assertEqual(second['previous_event_sha256'], first['event_sha256'])
            self.assertEqual(len(ledger.read_text().splitlines()), 2)
            changed = ledger.read_text().replace('failed', 'passed', 1)
            ledger.write_text(changed)
            with self.assertRaisesRegex(ValueError, 'ledger hash'):
                attest.read_ledger(ledger)

    def test_numeric_is_not_run_when_basic_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / 'attestations.jsonl'
            with patch.object(attest, 'LEDGER_PATH', ledger), \
                    patch.object(attest, '_audit_commit', return_value='a' * 40), \
                    patch.object(attest.evidence, 'verify_v3_formal_job',
                                 side_effect=ValueError('bad frozen state')), \
                    patch.object(attest, '_tree_snapshot', return_value=TREE_A), \
                    patch.object(attest.numeric, 'audit_job') as numeric:
                with self.assertRaisesRegex(ValueError, 'bad frozen state'):
                    attest.audit_one(('isolet', 'mir', 42), Path('/producer'))
                numeric.assert_not_called()
            rows = attest.read_ledger(ledger)
            self.assertEqual([(row['stage'], row['status']) for row in rows],
                             [('basic', 'failed')])

    def test_only_preselected_seed_gets_numeric_ledger_event(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / 'attestations.jsonl'
            with patch.object(attest, 'LEDGER_PATH', ledger), \
                    patch.object(attest, '_audit_commit', return_value='a' * 40), \
                    patch.object(attest.evidence, 'verify_v3_formal_job',
                                 return_value={'admission': 'formal'}), \
                    patch.object(attest, '_tree_snapshot', return_value=TREE_A), \
                    patch.object(attest.numeric, 'audit_job',
                                 return_value={'basic_admission': 'formal'}) as numeric:
                attest.audit_one(('isolet', 'mir', 42), Path('/producer'))
                attest.audit_one(('isolet', 'mir', 43), Path('/producer'))
            self.assertEqual([row['stage'] for row in attest.read_ledger(ledger)],
                             ['basic', 'numeric', 'basic'])
            self.assertEqual(attest.read_ledger(ledger)[0]['original_tree_sha256'],
                             'a' * 64)
            numeric.assert_called_once()

    def test_seed_43_waits_for_latest_seed_42_numeric_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / 'attestations.jsonl'
            with patch.object(attest, 'LEDGER_PATH', ledger), \
                    patch.object(attest, '_audit_commit', return_value='a' * 40), \
                    patch.object(attest.evidence, 'verify_v3_formal_job') as basic:
                with self.assertRaisesRegex(ValueError, 'seed 42 numeric'):
                    attest.audit_one(('isolet', 'mir', 43), Path('/producer'))
                basic.assert_not_called()
            self.assertEqual([(row['stage'], row['status'])
                              for row in attest.read_ledger(ledger)],
                             [('prerequisite', 'failed')])

    def test_preflight_failure_is_appended(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / 'attestations.jsonl'
            with patch.object(attest, 'LEDGER_PATH', ledger), \
                    patch.object(attest, '_audit_commit',
                                 side_effect=ValueError('dirty checkout')):
                with self.assertRaisesRegex(ValueError, 'dirty checkout'):
                    attest.audit_one(('isolet', 'mir', 42), Path('/producer'))
            row = attest.read_ledger(ledger)[0]
            self.assertEqual((row['stage'], row['status']), ('preflight', 'failed'))

    def test_tree_change_during_numeric_is_failed_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / 'attestations.jsonl'
            with patch.object(attest, 'LEDGER_PATH', ledger), \
                    patch.object(attest, '_audit_commit', return_value='a' * 40), \
                    patch.object(attest.evidence, 'verify_v3_formal_job',
                                 return_value={'admission': 'formal'}), \
                    patch.object(attest, '_tree_snapshot',
                                 side_effect=[TREE_A, TREE_A, TREE_A, TREE_B]), \
                    patch.object(attest.numeric, 'audit_job',
                                 return_value={'basic_admission': 'formal'}):
                with self.assertRaisesRegex(ValueError, 'tree changed'):
                    attest.audit_one(('isolet', 'mir', 42), Path('/producer'))
            rows = attest.read_ledger(ledger)
            self.assertEqual([row['status'] for row in rows], ['passed', 'failed'])
            self.assertEqual(rows[-1]['original_tree_sha256'], 'a' * 64)
            self.assertEqual(rows[-1]['original_tree_after_sha256'], 'b' * 64)

    def test_tree_change_between_basic_and_numeric_blocks_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / 'attestations.jsonl'
            with patch.object(attest, 'LEDGER_PATH', ledger), \
                    patch.object(attest, '_audit_commit', return_value='a' * 40), \
                    patch.object(attest.evidence, 'verify_v3_formal_job',
                                 return_value={'admission': 'formal'}), \
                    patch.object(attest, '_tree_snapshot',
                                 side_effect=[TREE_A, TREE_A, TREE_B, TREE_B]), \
                    patch.object(attest.numeric, 'audit_job') as numeric:
                with self.assertRaisesRegex(ValueError, 'tree changed since basic'):
                    attest.audit_one(('isolet', 'mir', 42), Path('/producer'))
                numeric.assert_not_called()
            self.assertEqual([row['status'] for row in attest.read_ledger(ledger)],
                             ['passed', 'failed'])

    def test_concurrent_seed42_failure_blocks_seed43_at_append(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / 'attestations.jsonl'
            attest.append_event(ledger, {
                'job': 'isolet/mir/42', 'stage': 'numeric', 'status': 'passed',
                'audit_commit': 'a' * 40,
                'producer_commit': attest.evidence.CORRECTED_COMMIT})

            def fail_seed42_during_basic(_job, **_kwargs):
                attest.append_event(ledger, {
                    'job': 'isolet/mir/42', 'stage': 'basic', 'status': 'failed',
                    'audit_commit': 'a' * 40,
                    'producer_commit': attest.evidence.CORRECTED_COMMIT})
                return {'admission': 'formal'}

            with patch.object(attest, 'LEDGER_PATH', ledger), \
                    patch.object(attest, '_audit_commit', return_value='a' * 40), \
                    patch.object(attest, '_tree_snapshot', return_value=TREE_A), \
                    patch.object(attest.evidence, 'verify_v3_formal_job',
                                 side_effect=fail_seed42_during_basic):
                with self.assertRaisesRegex(ValueError, 'seed 42 numeric'):
                    attest.audit_one(('isolet', 'mir', 43), Path('/producer'))
            self.assertEqual([(row['stage'], row['status'])
                              for row in attest.read_ledger(ledger)][-2:],
                             [('basic', 'failed'), ('prerequisite', 'failed')])

    def test_missing_original_is_not_claimed_as_hashed_empty_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = attest._tree_snapshot(Path(directory) / 'missing')
        self.assertIsNone(snapshot['sha256'])
        self.assertIn('missing', snapshot['error'])

    def test_finalize_requires_latest_pass_for_every_basic_and_seed42_numeric(self):
        commit = 'a' * 40
        jobs = [('isolet', 'mir', 42), ('isolet', 'mir', 43)]

        def row(job, stage, status='passed'):
            path = attest.evidence.FROZEN_RESULTS_ROOT / job[0] / job[1] / str(job[2])
            return {'job': '/'.join(map(str, job)), 'stage': stage,
                    'status': status, 'audit_commit': commit,
                    'producer_commit': attest.evidence.CORRECTED_COMMIT,
                    'source_kind': 'original', 'source_path': str(path),
                    'original_tree_sha256': 'f' * 64,
                    'original_tree_after_sha256': 'f' * 64,
                    'original_tree_error': None,
                    'file_sha256': {'JOB_SUCCESS.json': 'e' * 64},
                    'result': ({'admission': 'formal'} if stage == 'basic'
                               else {'basic_admission': 'formal'}),
                    'event_sha256': 'd' * 64}

        records = [row(jobs[0], 'basic'), row(jobs[0], 'numeric'),
                   row(jobs[1], 'basic')]
        summary = attest.finalize_records(records, commit, jobs=jobs)
        self.assertEqual((summary['basic_passed'], summary['numeric_passed']),
                         (2, 1))
        with self.assertRaisesRegex(ValueError, 'numeric'):
            attest.finalize_records([records[0], records[2]], commit, jobs=jobs)
        with self.assertRaisesRegex(ValueError, 'latest'):
            attest.finalize_records(records + [row(jobs[0], 'basic', 'failed')],
                                    commit, jobs=jobs)


if __name__ == '__main__':
    unittest.main()
