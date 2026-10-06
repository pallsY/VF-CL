import hashlib
import json
import unittest
from unittest import mock
import tempfile
import subprocess
from types import SimpleNamespace
from pathlib import Path

import torch
from models import TopModel

from factorized_paired_evaluation import (
    _canonical_sha256, _sha256, _validated_output_classes,
    _verify_published_transaction,
    _test_source_path, _verified_evaluator_commit,
    evaluate_run, summarize_paired,
)


class PairedEvaluationTests(unittest.TestCase):
    def test_rejects_output_inside_published_source_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            run = source / 'runs' / 'job'
            run.mkdir(parents=True)
            output = source / 'new-output'
            with self.assertRaisesRegex(ValueError, 'overlap'):
                evaluate_run(run, output)
            self.assertFalse(output.exists())

    def test_rejects_self_consistent_marker_without_complete_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            run = source / 'runs' / 'job'
            checkpoint = run / 'checkpoints' / 'formal_final.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'not a checkpoint')
            (run / 'FORMAL_STATE_FROZEN.json').write_text('{}')
            (run / 'config.json').write_text('{}')
            result = run / 'results.json'
            result.write_text('{}')
            published = {
                'status': 'published',
                'checkpoint': {'sha256': _sha256(checkpoint)},
                'results': {'sha256': _sha256(result)},
                'results_sha256': _canonical_sha256({}),
            }
            (run / 'FORMAL_EVALUATION_PUBLISHED.json').write_text(
                json.dumps(published)
            )
            output = Path(directory) / 'paired-output'
            with self.assertRaisesRegex(ValueError, 'complete/seal'):
                evaluate_run(run, output)
            self.assertFalse(output.exists())

    def test_evaluator_commit_rejects_dirty_source(self):
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(['git', 'init', '-q', directory], check=True)
            source = Path(directory) / 'module.py'
            source.write_text('value = 1\n')
            subprocess.run(['git', '-C', directory, 'add', 'module.py'], check=True)
            subprocess.run([
                'git', '-C', directory, '-c', 'user.name=Test',
                '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'init',
            ], check=True)
            self.assertEqual(len(_verified_evaluator_commit(directory)), 40)
            source.write_text('value = 2\n')
            with self.assertRaisesRegex(ValueError, 'dirty'):
                _verified_evaluator_commit(directory)

    def test_accepts_interleaved_task_classes_with_valid_head_map(self):
        model = TopModel(2, 4)
        model.set_logit_calibration([0, 1, 2, 3], [1.0] * 4,
                                    [0.0] * 4, [0, 1, 0, 1], 1.0)
        model.set_adaptive_mixture(torch.zeros(4, 2), torch.zeros(4),
                                   0.5, [0, 1, 2, 3])
        self.assertEqual(_validated_output_classes(
            model, {0: [0, 2], 1: [1, 3]}
        ), [0, 1, 2, 3])
        with self.assertRaisesRegex(ValueError, 'task map'):
            _validated_output_classes(model, {0: [0, 1], 1: [2, 3]})

    def test_resolves_default_vector_test_source(self):
        args = SimpleNamespace(data='tabvfl', data_path='/dataset',
                               vector_npz=None)
        self.assertEqual(_test_source_path(args),
                         Path('/dataset/tabvfl/tabvfl.npz'))
        args.vector_npz = '/frozen/isolet_vfl.npz'
        self.assertEqual(_test_source_path(args), Path(args.vector_npz))
        args.data = 'cifar100'
        args.vector_npz = None
        self.assertEqual(_test_source_path(args),
                         Path('/dataset/cifar-100-python/test'))

    def test_requires_on_disk_frozen_marker_even_with_sealed_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                    'factorized_paired_evaluation._load_sealed_complete_artifact',
                    return_value=({'identity': {}, 'freeze': {}}, {})), \
                 mock.patch('factorized_paired_evaluation._validate_formal_published'):
                with self.assertRaises(ValueError):
                    _verify_published_transaction(Path(directory))

    def test_published_digest_uses_canonical_result_not_file_bytes(self):
        result = {'b': 2, 'a': 1}
        file_bytes = json.dumps(result).encode()
        published_digest = hashlib.sha256(
            b'{\n  "a": 1,\n  "b": 2\n}\n'
        ).hexdigest()
        self.assertNotEqual(hashlib.sha256(file_bytes).hexdigest(), published_digest)
        self.assertEqual(_canonical_sha256(result), published_digest)

    def test_bwt_excludes_final_numeric_task_even_with_lexical_key_order(self):
        diagonal = {'task_0': 0.9, 'task_1': 0.8,
                    'task_10': 0.7, 'task_2': 0.6}
        final = {'task_0': 0.6, 'task_1': 0.7,
                 'task_10': 0.9, 'task_2': 0.5}
        measured = {
            task: {'count': 10, 'mixed_correct': round(value * 10),
                   'mixed_taskil_correct': round(value * 10),
                   'factorized_correct': round(value * 10),
                   'factorized_taskil_correct': round(value * 10)}
            for task, value in final.items()
        }
        history = [{'deferred_diagonal': diagonal,
                    'deferred_final_task': 'task_10',
                    'per_task_accs': final,
                    'per_task_accs_taskil': final}]
        baseline = {'AA_final': 0.675, 'BWT': -0.1667,
                    'AA_final_taskil': 0.675}
        result = summarize_paired(measured, history, baseline)
        self.assertEqual(result['mixed']['BWT'], -0.1667)

    def test_one_task_has_zero_bwt(self):
        history = [{'deferred_diagonal': {'task_0': 0.7},
                    'deferred_final_task': 'task_0',
                    'per_task_accs': {'task_0': 0.8},
                    'per_task_accs_taskil': {'task_0': 0.9}}]
        measured = {'task_0': {
            'count': 10, 'mixed_correct': 8, 'mixed_taskil_correct': 9,
            'factorized_correct': 8, 'factorized_taskil_correct': 9,
        }}
        baseline = {'AA_final': 0.8, 'BWT': 0.0,
                    'AA_final_taskil': 0.9}
        self.assertEqual(summarize_paired(measured, history, baseline)
                         ['factorized']['BWT'], 0.0)

    def test_reuses_unchanged_diagonal_and_rejects_mixed_mismatch(self):
        history = [
            {'per_task_accs': {'task_0': 0.9}},
            {
                'deferred_diagonal': {'task_0': 0.9, 'task_1': 0.8},
                'deferred_final_task': 'task_1',
                'per_task_accs': {'task_0': 0.6, 'task_1': 0.8},
                'per_task_accs_taskil': {'task_0': 0.8, 'task_1': 0.9},
            },
        ]
        baseline = {'AA_final': 0.7, 'BWT': -0.3,
                    'AA_final_taskil': 0.85}
        measured = {
            'task_0': {'count': 10, 'mixed_correct': 6,
                       'mixed_taskil_correct': 8, 'factorized_correct': 7,
                       'factorized_taskil_correct': 9},
            'task_1': {'count': 10, 'mixed_correct': 8,
                       'mixed_taskil_correct': 9, 'factorized_correct': 8,
                       'factorized_taskil_correct': 9},
        }
        result = summarize_paired(measured, history, baseline)
        self.assertEqual(result['factorized']['AA_final'], 0.75)
        self.assertEqual(result['factorized']['BWT'], -0.2)
        self.assertEqual(result['factorized']['AA_final_taskil'], 0.9)
        self.assertEqual(result['mixed'], baseline)

        measured['task_0']['mixed_correct'] = 5
        with self.assertRaisesRegex(ValueError, 'Mixed baseline mismatch'):
            summarize_paired(measured, history, baseline)


if __name__ == '__main__':
    unittest.main()
