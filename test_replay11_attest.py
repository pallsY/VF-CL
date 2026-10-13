"""Audit ledger tests: failures remain visible and seed 42 is gated."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import replay11_attest as attest


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
                    patch.object(attest.evidence, '_tree_identity',
                                 return_value=(5, 100, 'f' * 64)), \
                    patch.object(attest.numeric, 'audit_job',
                                 return_value={'basic_admission': 'formal'}) as numeric:
                attest.audit_one(('isolet', 'mir', 42), Path('/producer'))
                attest.audit_one(('isolet', 'mir', 43), Path('/producer'))
            self.assertEqual([row['stage'] for row in attest.read_ledger(ledger)],
                             ['basic', 'numeric', 'basic'])
            self.assertEqual(attest.read_ledger(ledger)[0]['original_tree_sha256'],
                             'f' * 64)
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
            first = (5, 100, 'a' * 64)
            changed = (5, 100, 'b' * 64)
            with patch.object(attest, 'LEDGER_PATH', ledger), \
                    patch.object(attest, '_audit_commit', return_value='a' * 40), \
                    patch.object(attest.evidence, 'verify_v3_formal_job',
                                 return_value={'admission': 'formal'}), \
                    patch.object(attest.evidence, '_tree_identity',
                                 side_effect=[first, first, first, changed]), \
                    patch.object(attest.numeric, 'audit_job',
                                 return_value={'basic_admission': 'formal'}):
                with self.assertRaisesRegex(ValueError, 'tree changed'):
                    attest.audit_one(('isolet', 'mir', 42), Path('/producer'))
            rows = attest.read_ledger(ledger)
            self.assertEqual([row['status'] for row in rows], ['passed', 'failed'])
            self.assertEqual(rows[-1]['original_tree_sha256'], 'a' * 64)
            self.assertEqual(rows[-1]['original_tree_after_sha256'], 'b' * 64)


if __name__ == '__main__':
    unittest.main()
