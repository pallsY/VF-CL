"""Final gate over nine proposed-method originals and three replays."""

import unittest

from adaptive9_attest import finalize_records, audit_one, numeric_visibility


PRODUCER = 'a' * 40
AUDIT = 'b' * 40
ROOT = '/results/full'


def passed(job, stage):
    return {
        'job': job, 'stage': stage, 'status': 'passed',
        'source_kind': 'original', 'source_path': ROOT + '/' + job,
        'producer_commit': PRODUCER, 'audit_commit': AUDIT,
        'original_tree_sha256': 'c' * 64,
        'original_tree_after_sha256': 'c' * 64,
        'result': {'admission': 'formal', 'basic_admission': 'formal'},
        'event_sha256': 'd' * 64,
    }


class AdaptiveFinalGateTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
