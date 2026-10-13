"""Small, synthetic checks for the independent Replay-11 numeric gate."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import replay11_numeric as numeric


class NumericAuditTests(unittest.TestCase):
    def test_isolet_vector_cache_scores_fresh_trainer(self):
        class VectorTrainer:
            def evaluate(self, batches):
                labels = torch.cat([y for _, y in batches])
                # The first feature is a fixed class id encoded in this fixture.
                classes = torch.cat([x[:, 0].long() for x, _ in batches])
                probabilities = torch.nn.functional.one_hot(classes, 4).float()
                return 0.0, probabilities, labels

        batches = ((torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
                    torch.tensor([0, 1, 2, 3])),)
        self.assertEqual(numeric.score_state(
            VectorTrainer(), object(), batches, {0: [0, 1], 1: [2, 3]}),
            {'primary': {'task_0': 2, 'task_1': 2},
             'task_il': {'task_0': 2, 'task_1': 2}})

    def test_unique_count_and_ambiguous_rounding(self):
        self.assertEqual(numeric.unique_correct(0.123, 1000), 123)
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            numeric.unique_correct(0.5, 10001)

    def test_wrong_checkpoint_is_rejected_by_content_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            path = run / 'checkpoints' / 'formal_final.pt'
            path.parent.mkdir()
            path.write_bytes(b'original checkpoint')
            record = {'path': 'checkpoints/formal_final.pt',
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(numeric.verified_bytes(run, record), b'original checkpoint')
            path.write_bytes(b'changed checkpoint')
            with self.assertRaisesRegex(ValueError, 'sha256'):
                numeric.verified_bytes(run, record)

    def test_frozen_isolet_fixture_reads_snapshot_bytes_without_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory) / 'isolet' / 'mir' / '42'
            run = job / 'run'
            (run / 'formal_snapshots').mkdir(parents=True)
            (run / 'checkpoints').mkdir()
            for name in ('formal_snapshots/event_0_CIL.pt',
                         'checkpoints/event_0_CIL.pt',
                         'checkpoints/formal_final.pt'):
                (run / name).write_bytes(name.encode())
            def record(name):
                return {'path': name, 'sha256': hashlib.sha256(
                    (run / name).read_bytes()).hexdigest()}
            identity = {'source_commit': 'a' * 40,
                        'task_classes': {'0': [0, 1]},
                        'snapshots': [{'file': record('formal_snapshots/event_0_CIL.pt'),
                                       'source_file': record('checkpoints/event_0_CIL.pt')}],
                        'final_checkpoint': {'file': record('checkpoints/formal_final.pt')}}
            (run / 'FORMAL_STATE_FROZEN.json').write_text(json.dumps({'identity': identity}))
            (run / 'checkpoints/formal_final.pt').write_bytes(b'published final checkpoint')
            (run / 'FORMAL_EVALUATION_PUBLISHED.json').write_text(json.dumps({
                'checkpoint': record('checkpoints/formal_final.pt')}))
            (run / 'FORMAL_EVALUATION_COMPLETE.json').write_text(json.dumps({
                'cache_identity': {'batch_count': 1, 'sample_count': 2, 'batches': []}}))
            (run / 'results.json').write_text(json.dumps({'task_acc_history': []}))
            (run / 'config.json').write_text(json.dumps({'seed': 42}))
            original_read_bytes = Path.read_bytes
            def no_checkpoint_read_bytes(path):
                if path.suffix == '.pt':
                    raise AssertionError('checkpoint was buffered in full')
                return original_read_bytes(path)
            with patch.object(Path, 'read_bytes', no_checkpoint_read_bytes):
                result = numeric.read_job_evidence(job)
            self.assertEqual(result['snapshots'][0]['file']['path'],
                             'formal_snapshots/event_0_CIL.pt')
            self.assertEqual(result['final']['path'], 'checkpoints/formal_final.pt')
            self.assertNotEqual(result['final']['sha256'],
                                identity['final_checkpoint']['file']['sha256'])
            (run / 'formal_snapshots/event_0_CIL.pt').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'sha256'):
                numeric.read_job_evidence(job)

    def test_wrong_cache_batch_is_rejected(self):
        batches = ((torch.tensor([[1.0], [2.0]]), torch.tensor([0, 1])),)
        identity = numeric.cache_identity(batches)
        self.assertEqual(numeric.verify_cache(batches, identity), {0: 1, 1: 1})
        changed = ((torch.tensor([[1.0], [3.0]]), torch.tensor([0, 1])),)
        with self.assertRaisesRegex(ValueError, 'cache'):
            numeric.verify_cache(changed, identity)

    def test_streamed_checkpoint_loads_restricted_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'checkpoint.pt'
            torch.save({'weight': torch.tensor([2.0])}, path)
            record = {'path': path.name,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(numeric.load_checkpoint(root, record)['weight'].item(), 2.0)
            record['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'sha256'):
                numeric.load_checkpoint(root, record)

    def test_two_task_isolet_style_integer_and_metric_gate(self):
        # ISOLET uses a vector test cache and two-class tasks; totals can differ.
        stored = [
            {'step': 'event_0_CIL', 'per_task_accs': {'task_0': 0.75}},
            {'step': 'event_1_CIL',
             'per_task_accs': {'task_0': 0.5, 'task_1': 0.6},
             'per_task_accs_taskil': {'task_0': 0.75, 'task_1': 0.8}},
        ]
        totals = {'task_0': 4, 'task_1': 5}
        observed = {
            'diagonal': {'task_0': 3, 'task_1': 3},
            'final': {'task_0': 2, 'task_1': 3},
            'task_il': {'task_0': 3, 'task_1': 4},
        }
        self.assertEqual(numeric.compare_trajectory(stored, totals, observed), {
            'aa_final_cil': 0.55, 'bwt_cil': -0.25,
            'task_il_final': 0.775,
        })
        observed['final']['task_0'] = 1
        with self.assertRaisesRegex(ValueError, 'integer'):
            numeric.compare_trajectory(stored, totals, observed)

    def test_non_preselected_seed_cannot_launch_numeric_replay(self):
        with self.assertRaisesRegex(ValueError, 'seed 42'):
            numeric.audit_job('isolet/mir/43', 'a' * 40, '.', '.')

    def test_public_numeric_gate_requires_pinned_basic_admission(self):
        import replay11_evidence as evidence
        with patch.object(evidence, 'verify_v3_formal_job',
                          side_effect=ValueError('basic evidence rejected')) as basic:
            with patch.object(numeric.subprocess, 'run') as worker:
                with self.assertRaisesRegex(ValueError, 'basic evidence rejected'):
                    numeric.audit_job('isolet/mir/42', evidence.CORRECTED_COMMIT,
                                      '.', evidence.FROZEN_DATA_ROOT)
                basic.assert_called_once()
                worker.assert_not_called()

    def test_method_task_il_uses_direct_prediction_count(self):
        class Trainer:
            def evaluate(self, batches):
                labels = torch.cat([y for _, y in batches])
                probabilities = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
                return 0.0, probabilities, labels

        class Method:
            name = 'HPCR_VFL'

            def evaluate_class_il_readouts_cached(self, batches, task_classes):
                return {
                    'class_il_pred_task_counts': {
                        'task_0': {'correct': 2, 'total': 2}},
                    'class_il_pred_task': {'task_0': 1.0},
                    'task_il_oracle': {'task_0': 0.0},
                }

        batches = ((torch.tensor([[1.0], [2.0]]), torch.tensor([0, 1])),)
        with self.assertRaisesRegex(ValueError, 'Task-IL.*differs'):
            numeric.score_state(Trainer(), Method(), batches, {0: [0, 1]})

    def test_frozen_tasks_must_match_full_planned_class_partition(self):
        planned = [[0, 1], [2, 3]]
        self.assertEqual(numeric.validate_task_classes(
            {0: [0, 1], 1: [2, 3]}, planned, 4),
            {0: [0, 1], 1: [2, 3]})
        self.assertEqual(numeric.validate_task_classes(
            {0: [0, 1], 1: [2, 3]}, {0: [0, 1], 1: [2, 3]}, 4),
            {0: [0, 1], 1: [2, 3]})
        with self.assertRaisesRegex(ValueError, 'task classes'):
            numeric.validate_task_classes({0: [0], 1: [2, 3]}, planned, 4)


if __name__ == '__main__':
    unittest.main()
