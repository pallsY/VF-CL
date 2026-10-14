"""Final gate over nine proposed-method originals and three replays."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from adaptive9_attest import (
    finalize_records, audit_one, numeric_visibility, next_audit_stage)
import replay11_attest as ledger


PRODUCER = '9973018295da270fd6098de07dcf36b9fe65ff61'
AUDIT = 'b' * 40
ROOT = '/results/full'


def passed(job, stage):
    count = 13 if job.startswith('isolet/') else 10
    totals = {f'task_{index}': 100 for index in range(count)}
    correct = {name: {key: 50 for key in totals}
               for name in ('diagonal', 'final', 'task_il')}
    metrics = {'aa_final_cil': 0.5, 'bwt_cil': 0.0,
               'task_il_final': 0.5}
    result = ({'admission': 'formal', 'metrics': metrics}
              if stage == 'basic' else
              {'basic_admission': 'formal', 'job': job,
               'producer_commit': PRODUCER, 'metrics': metrics,
               'task_totals': totals, 'correct': correct})
    return {
        'job': job, 'stage': stage, 'status': 'passed',
        'source_kind': 'original', 'source_path': ROOT + '/' + job,
        'producer_commit': PRODUCER, 'audit_commit': AUDIT,
        'original_tree_sha256': 'c' * 64,
        'original_tree_after_sha256': 'c' * 64,
        'result': result,
        'event_sha256': 'd' * 64,
    }


class AdaptiveFinalGateTests(unittest.TestCase):
    def test_retry_appends_new_basic_after_failed_basic(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / 'ledger.jsonl'
            ledger.append_event(path, {
                'job': 'isolet/adaptive/43', 'stage': 'basic',
                'status': 'failed'})
            snapshot = {'sha256': 'c' * 64, 'error': None,
                        'files': 1, 'bytes': 1}
            with patch.object(ledger, '_audit_commit', return_value=AUDIT), \
                 patch('adaptive9_attest._snapshot', return_value=snapshot), \
                 patch('adaptive9_attest.evidence.verify_pinned_job',
                       return_value={'admission': 'formal'}):
                audit_one(
                    'isolet', 43, results_root=Path(scratch) / 'full',
                    data_root='/data', producer_root='/producer',
                    producer_commit=PRODUCER, ledger_path=path,
                    audit_commit=AUDIT)
            self.assertEqual(
                [row['status'] for row in ledger.read_ledger(path)],
                ['failed', 'passed'])

    def test_retry_appends_new_basic_and_numeric_after_failed_numeric(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / 'ledger.jsonl'
            ledger.append_event(path, {
                'job': 'isolet/adaptive/42', 'stage': 'basic',
                'status': 'passed'})
            ledger.append_event(path, {
                'job': 'isolet/adaptive/42', 'stage': 'numeric',
                'status': 'failed'})
            snapshot = {'sha256': 'c' * 64, 'error': None,
                        'files': 1, 'bytes': 1}
            basic = {'admission': 'formal', 'metrics': {'aa_final_cil': 0.5}}
            numeric = {'metrics': {'aa_final_cil': 0.5}}
            with patch.object(ledger, '_audit_commit', return_value=AUDIT), \
                 patch('adaptive9_attest._snapshot', return_value=snapshot), \
                 patch('adaptive9_attest.evidence.verify_pinned_job',
                       return_value=basic), \
                 patch('adaptive9_attest._numeric_in_new_process',
                       return_value=numeric):
                audit_one(
                    'isolet', 42, results_root=Path(scratch) / 'full',
                    data_root='/data', producer_root='/producer',
                    producer_commit=PRODUCER, ledger_path=path,
                    audit_commit=AUDIT)
            rows = ledger.read_ledger(path)
            self.assertEqual([(row['stage'], row['status']) for row in rows],
                             [('basic', 'passed'), ('numeric', 'failed'),
                              ('basic', 'passed'), ('numeric', 'passed')])

    def test_consistent_but_unapproved_producer_cannot_pass(self):
        other = 'f' * 40
        rows = []
        for dataset in ('cifar100', 'isolet', 'upmc_food101'):
            for seed in (42, 43, 44):
                job = f'{dataset}/adaptive/{seed}'
                for stage in (('basic', 'numeric') if seed == 42
                              else ('basic',)):
                    row = dict(passed(job, stage), producer_commit=other)
                    if stage == 'numeric':
                        row['result'] = dict(row['result'],
                                             producer_commit=other)
                    rows.append(row)
        with self.assertRaisesRegex(ValueError, 'approved producer'):
            finalize_records(
                rows, producer_commit=other, audit_commit=AUDIT,
                results_root=ROOT)

    def test_failed_verdict_is_preserved_and_followed_by_fresh_basic(self):
        failed_basic = dict(passed('isolet/adaptive/42', 'basic'),
                            status='failed')
        failed_numeric = dict(passed('isolet/adaptive/42', 'numeric'),
                              status='failed')
        self.assertEqual(next_audit_stage([failed_basic], 42), 'basic')
        self.assertEqual(next_audit_stage([
            passed('isolet/adaptive/42', 'basic'), failed_numeric], 42),
            'basic')
        self.assertEqual(next_audit_stage([
            passed('isolet/adaptive/42', 'basic')], 42), 'numeric')

    def test_numeric_worker_exposes_recorded_source_gpu_count(self):
        self.assertEqual(numeric_visibility('c5090.upcbrfed.xyz'), '0')
        self.assertEqual(numeric_visibility('c3080.upcbrfed.xyz'), '0,1')
        with self.assertRaisesRegex(ValueError, 'host'):
            numeric_visibility('unknown')

    def test_outside_matrix_job_cannot_append_to_ledger(self):
        with self.assertRaisesRegex(ValueError, 'nine-cell'):
            audit_one(
                'isolet', 45, results_root=ROOT, data_root='/data',
                producer_root='/producer', producer_commit=PRODUCER,
                ledger_path='/unused/ledger.jsonl', audit_commit=AUDIT)

    def test_requires_all_nine_basic_and_three_seed42_numeric(self):
        rows = []
        for dataset in ('cifar100', 'isolet', 'upmc_food101'):
            for seed in (42, 43, 44):
                job = f'{dataset}/adaptive/{seed}'
                rows.append(passed(job, 'basic'))
                if seed == 42:
                    rows.append(passed(job, 'numeric'))
        summary = finalize_records(
            rows, producer_commit=PRODUCER, audit_commit=AUDIT,
            results_root=ROOT)
        self.assertEqual(summary['basic_passed'], 9)
        self.assertEqual(summary['numeric_passed'], 3)
        missing_numeric = [row for row in rows
                           if not (row['job'] == 'upmc_food101/adaptive/42'
                                   and row['stage'] == 'numeric')]
        with self.assertRaisesRegex(ValueError, 'numeric'):
            finalize_records(
                missing_numeric, producer_commit=PRODUCER,
                audit_commit=AUDIT,
                results_root=ROOT)
        empty = [dict(row) for row in rows]
        numeric = next(row for row in empty
                       if row['job'] == 'isolet/adaptive/42'
                       and row['stage'] == 'numeric')
        numeric['result'] = {'basic_admission': 'formal'}
        with self.assertRaisesRegex(ValueError, 'numeric evidence'):
            finalize_records(
                empty, producer_commit=PRODUCER,
                audit_commit=AUDIT, results_root=ROOT)

    def test_rejects_numeric_metrics_that_differ_from_basic(self):
        rows = []
        for dataset in ('cifar100', 'isolet', 'upmc_food101'):
            for seed in (42, 43, 44):
                job = f'{dataset}/adaptive/{seed}'
                rows.append(passed(job, 'basic'))
                if seed == 42:
                    rows.append(passed(job, 'numeric'))
        numeric = next(row for row in rows
                       if row['job'] == 'cifar100/adaptive/42'
                       and row['stage'] == 'numeric')
        numeric['result'] = dict(numeric['result'])
        numeric['result']['metrics'] = dict(
            numeric['result']['metrics'], aa_final_cil=0.6)
        with self.assertRaisesRegex(ValueError, 'numeric evidence'):
            finalize_records(
                rows, producer_commit=PRODUCER,
                audit_commit=AUDIT, results_root=ROOT)


if __name__ == '__main__':
    unittest.main()
