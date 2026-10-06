import unittest

from factorized_paired_evaluation import summarize_paired


class PairedEvaluationTests(unittest.TestCase):
    def test_reuses_unchanged_diagonal_and_rejects_mixed_mismatch(self):
        history = [
            {'per_task_accs': {'task_0': 0.9}},
            {
                'deferred_diagonal': {'task_0': 0.9, 'task_1': 0.8},
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
