import unittest
import tempfile
import hashlib
import json
from pathlib import Path

import party_drift_adjudicate as decision
import party_drift_manifest as manifest


class AdjudicationTests(unittest.TestCase):
    def test_complete_183_cell_matrix_uses_paired_scores_only_after_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            listing, runs, output = base / 'listing', base / 'runs', base / 'decision'
            entries = []
            for key in sorted(manifest.expected_job_keys()):
                dataset, method, alpha_text, seed = key.split('/')
                alpha = None if alpha_text == 'none' else float(alpha_text)
                name = key.replace('/', '_').replace('.', 'p')
                job_root = runs / name
                config = {
                    'development_job_key': key, 'seed': int(seed),
                    'pilot_dataset': dataset, 'party_replay_arm': method,
                    'party_replay_alpha': alpha, 'output_dir': str(job_root),
                }
                raw = manifest._canonical_bytes(config)
                row = {
                    'job_key': key, 'dataset': dataset, 'method': method,
                    'alpha': alpha, 'seed': int(seed),
                    'config_relpath': f'configs/{name}.json',
                    'config_sha256': hashlib.sha256(raw).hexdigest(),
                    'source_config_sha256': '1' * 64,
                    'source_data_sha256': '2' * 64,
                    'holdout_sha256': '3' * 64,
                    'validation_manifest_sha256': '4' * 64,
                    'production_commit': 'a' * 40,
                }
                entries.append((row, config))
            digest = manifest.write_manifest_files(listing, entries)
            runs.mkdir()
            events = []
            for row, config in entries:
                ds, method = row['dataset'], row['method']
                count = (100 if ds == 'cifar100' else
                         26 if ds == 'isolet' else 101) * 30
                score = 90 if method == 'drift' else 70 if method in (
                    'uniform', 'static') else 80
                correct = count * score // 100
                run = Path(config['output_dir']) / 'attempt-01'
                run.mkdir(parents=True)
                checkpoint = run / 'checkpoints' / 'formal_final.pt'
                checkpoint.parent.mkdir()
                checkpoint.write_bytes(b'checkpoint')
                checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                flow = run / 'data_flow_audit.jsonl'
                flow.write_text('{"split":"train"}\n')
                readout = {
                    'job_key': row['job_key'], 'test_accessed': False,
                    'checkpoint_sha256': checkpoint_sha,
                    'readout_rule': ('final_bic_joint_alpha_beta' if ds == 'cifar100'
                                     else 'icarl_nme' if method == 'icarl'
                                     else 'model_head'),
                    'metrics': {'cil_correct': correct, 'count': count,
                                'cil': correct / count},
                    'readout_details': ({'raw_metrics': {
                        'cil_correct': max(0, correct - 1), 'count': count,
                        'cil': max(0, correct - 1) / count},
                        'bic_fit_source': ('calibration' if method == 'er_ace'
                                           else 'validation'),
                        'bic_fit_count': 2500}
                        if ds == 'cifar100' else {}),
                }
                raw = json.dumps(readout).encode()
                (run / 'HOLDOUT_READOUT.json').write_bytes(raw)
                replay = method in manifest.REPLAY_ARMS
                if replay:
                    audit = run / 'party_drift_audit' / 'selection.jsonl'
                    audit.parent.mkdir()
                    audit.write_text('{"selected":[0,1]}\n')
                (run / 'JOB_SUCCESS.json').write_text(json.dumps({
                    'status': 'success', 'job_key': row['job_key'],
                    'manifest_sha256': digest,
                    'readout_sha256': hashlib.sha256(raw).hexdigest(),
                    'checkpoint_sha256': checkpoint_sha,
                    'data_flow_sha256': hashlib.sha256(flow.read_bytes()).hexdigest(),
                    'selection_audit_sha256': (
                        hashlib.sha256(audit.read_bytes()).hexdigest()
                        if replay else None),
                    'replay_steps': 1 if replay else 0,
                    'test_accessed': False,
                }), encoding='utf-8')
                events.append({'kind': 'finish', 'job_key': row['job_key'],
                               'attempt': 1, 'classification': 'success',
                               'job_success_sha256': hashlib.sha256(
                                   (run / 'JOB_SUCCESS.json').read_bytes()).hexdigest()})
            event_bytes = ''.join(json.dumps(item) + '\n' for item in events).encode()
            (runs / 'ATTEMPTS.jsonl').write_bytes(event_bytes)
            (runs / 'QUEUE_COMPLETE.json').write_text(json.dumps({
                'status': 'complete', 'accounted_for': 183,
                'manifest_sha256': digest,
                'attempts_sha256': hashlib.sha256(event_bytes).hexdigest(),
            }), encoding='utf-8')
            result = decision.adjudicate(listing / 'manifest.jsonl', digest,
                                         runs, output)
            self.assertEqual(result['decision'], 'advance_to_confirmation')
            self.assertEqual(result['selected_alpha']['isolet']['drift'], 0.25)
            self.assertEqual(len((output / 'all_183_cells.csv').read_text().splitlines()),
                             184)
            self.assertIn('raw_cil', (output / 'all_183_cells.csv').read_text().splitlines()[0])
            first = Path(entries[0][1]['output_dir']) / 'attempt-01' / 'JOB_SUCCESS.json'
            first.write_text(first.read_text() + ' ')
            with self.assertRaises(ValueError):
                decision.adjudicate(listing / 'manifest.jsonl', digest,
                                    runs, base / 'second-decision')

    def test_refuses_to_open_results_before_complete_queue_gate(self):
        self.assertTrue(hasattr(decision, 'validate_queue_complete'))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                decision.validate_queue_complete(Path(tmp), 'a' * 64)

    def test_historical_infrastructure_failure_remains_after_successful_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / 'ATTEMPTS.jsonl'
            events.write_text('', encoding='utf-8')
            (root / 'QUEUE_FAILURE.json').write_text('{}', encoding='utf-8')
            (root / 'QUEUE_COMPLETE.json').write_text(json.dumps({
                'status': 'complete', 'accounted_for': 183,
                'manifest_sha256': 'a' * 64,
                'attempts_sha256': hashlib.sha256(b'').hexdigest(),
            }), encoding='utf-8')
            self.assertEqual(decision.validate_queue_complete(
                root, 'a' * 64)['status'], 'complete')

    def test_alpha_uses_exact_three_seed_mean_then_smaller_tie(self):
        self.assertTrue(hasattr(decision, 'select_alpha'))
        grid = {
            0.25: [(8, 10), (9, 10), (7, 10)],
            0.5: [(9, 10), (8, 10), (7, 10)],
            1.0: [(7, 10), (7, 10), (7, 10)],
        }
        self.assertEqual(decision.select_alpha(grid), 0.25)
        grid[0.25][2] = (6, 10)
        self.assertEqual(decision.select_alpha(grid), 0.5)
        grid[0.5][2] = None
        self.assertEqual(decision.select_alpha(grid), 0.25)

    def test_four_predeclared_outcomes(self):
        self.assertTrue(hasattr(decision, 'outcome_case'))
        self.assertEqual(decision.outcome_case(True, True), 'advance_to_confirmation')
        self.assertEqual(decision.outcome_case(True, False), 'replay_gain_no_selector')
        self.assertEqual(decision.outcome_case(False, True), 'selector_gain_no_first')
        self.assertEqual(decision.outcome_case(False, False), 'direction_closed')


if __name__ == '__main__':
    unittest.main()
