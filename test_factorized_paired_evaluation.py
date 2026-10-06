import hashlib
import json
import unittest

from factorized_paired_evaluation import _canonical_sha256, summarize_paired


class PairedEvaluationTests(unittest.TestCase):
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
