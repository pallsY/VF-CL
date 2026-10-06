import json
import tempfile
import unittest
from pathlib import Path

from party_kd_lambda_validation import (
    audit_run,
    choose_lambda,
    expected_config,
    job_specs,
    metrics_for_result,
)


def make_history(final_value):
    history = []
    for stage in range(10):
        row = {
            f'task_{task}': (
                final_value if stage == 9 else 0.8 - 0.01 * task
            )
            for task in range(stage + 1)
        }
        history.append({
            'step': f'event_{stage}_CIL',
            'per_task_accs': row,
            'per_task_accs_taskil': {
                f'task_{task}': 0.7 for task in range(stage + 1)
            },
        })
    return history


def make_bic_history(final_value):
    events = []
    for entry in make_history(final_value):
        events.append({
            'step': entry['step'],
            'paired': {
                'calibrated': {
                    'per_task_accuracy': entry['per_task_accs'],
                    'task_il': entry['per_task_accs_taskil'],
                },
            },
            'calibration_audit': {
                'passed': True,
                'test_used_for_fit': False,
            },
            'privacy_audit': {
                'passed': True,
                'test_used_for_fit': False,
                'raw_images_saved': False,
                'party_embeddings_saved': False,
            },
            'task_il_max_abs_delta': 0.0,
        })
    return events


class PartyKDLambdaValidationTests(unittest.TestCase):
    def test_matrix_has_exactly_fifteen_unique_jobs(self):
        jobs = job_specs()
        self.assertEqual(len(jobs), 15)
        self.assertEqual(len(set(jobs)), 15)
        self.assertEqual(
            set(jobs),
            {
                *(f'lwf:{seed}' for seed in (42, 43, 44)),
                *(
                    f'lambda_{value}:{seed}'
                    for value in ('0.25', '0.50', '0.75', '1.00')
                    for seed in (42, 43, 44)
                ),
            },
        )

    def test_expected_config_freezes_validation_protocol(self):
        config = expected_config('lambda_0.50:43')
        self.assertEqual(config['party_kd_lambda'], 0.5)
        self.assertEqual(config['bic_fit_mode'], 'joint_each_stage')
        self.assertEqual(config['lambda_validation_per_class'], 25)
        self.assertEqual(config['lambda_validation_split_seed'], 20260729)
        self.assertEqual(config['save_task_checkpoints'], 3)
        self.assertEqual(config['unlearn_after_tasks'], [99])

    def test_metrics_use_raw_lwf_and_calibrated_lambda_history(self):
        result = {
            'task_acc_history': make_history(0.2),
            'bic_history': make_bic_history(0.4),
        }

        self.assertAlmostEqual(
            metrics_for_result('lwf:42', result)['aa_final_cil'], 0.2
        )
        self.assertAlmostEqual(
            metrics_for_result('lambda_0.25:42', result)['aa_final_cil'], 0.4
        )

    def test_selection_filters_by_lwf_bwt_then_maximizes_aa_final(self):
        rows = [
            {'method': 'lwf', 'bwt_cil_mean': -0.4},
            {'method': 'lambda_0.25', 'bwt_cil_mean': -0.39,
             'aa_final_cil_mean': 0.30, 'task_il_final_mean': 0.70},
            {'method': 'lambda_0.50', 'bwt_cil_mean': -0.41,
             'aa_final_cil_mean': 0.50, 'task_il_final_mean': 0.80},
            {'method': 'lambda_0.75', 'bwt_cil_mean': -0.38,
             'aa_final_cil_mean': 0.35, 'task_il_final_mean': 0.71},
            {'method': 'lambda_1.00', 'bwt_cil_mean': -0.37,
             'aa_final_cil_mean': 0.35, 'task_il_final_mean': 0.72},
        ]

        selected = choose_lambda(rows)

        self.assertEqual(selected['method'], 'lambda_1.00')
        self.assertEqual(selected['lambda'], 1.0)
        self.assertEqual(selected['lwf_bwt_constraint'], -0.4)

    def test_selection_fails_when_no_lambda_satisfies_bwt(self):
        rows = [
            {'method': 'lwf', 'bwt_cil_mean': -0.2},
            {'method': 'lambda_0.25', 'bwt_cil_mean': -0.3,
             'aa_final_cil_mean': 0.4, 'task_il_final_mean': 0.7},
        ]
        with self.assertRaisesRegex(ValueError, 'no lambda satisfies'):
            choose_lambda(rows)

    def test_audit_rejects_test_or_overlap_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            config = expected_config('lambda_0.25:42')
            (run_dir / 'config.json').write_text(json.dumps(config))
            (run_dir / 'results.json').write_text(json.dumps({
                'task_acc_history': make_history(0.2),
                'bic_history': make_bic_history(0.4),
                'calibration_audit': {
                    'passed': True,
                    'test_used_for_fit': False,
                },
                'selection_audit': {
                    'passed': False,
                    'test_used_for_selection': True,
                    'calibration_manifest_sha256': 'a' * 64,
                    'validation_manifest_sha256': 'b' * 64,
                },
            }))
            checkpoint = run_dir / 'checkpoints' / 'event_9_CIL.pt'
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b'checkpoint')

            with self.assertRaisesRegex(ValueError, 'selection audit failed'):
                audit_run('lambda_0.25:42', run_dir, 'deadbeef')


if __name__ == '__main__':
    unittest.main()
