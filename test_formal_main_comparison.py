import unittest

from formal_main_comparison import summarize


class FormalMainComparisonTests(unittest.TestCase):
    def test_three_paired_seeds_and_sample_sd(self):
        rows = []
        for seed, accuracy in ((42, 0.5), (43, 0.6), (44, 0.7)):
            rows.append({
                'dataset': 'isolet', 'method': 'er', 'seed': seed,
                'metrics': {
                    'aa_final_cil': accuracy,
                    'bwt_cil': -0.1,
                    'task_il_final': 0.9,
                },
                'max_raw_examples': 520,
                'max_learning_tensor_bytes': 1000,
                'validation_examples': 0,
                'calibration_examples': 0,
                'head_candidate_steps': 0,
            })
        result = summarize(rows)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]['aa_final_mean_pct'], 60.0)
        self.assertAlmostEqual(result[0]['aa_final_sd_pct'], 10.0)
        self.assertEqual(result[0]['seed_42_aa_final_pct'], 50.0)

    def test_missing_seed_is_rejected(self):
        rows = [{
            'dataset': 'isolet', 'method': 'er', 'seed': 42,
            'metrics': {'aa_final_cil': 0.5, 'bwt_cil': 0.0,
                        'task_il_final': 0.9},
            'max_raw_examples': 520, 'max_learning_tensor_bytes': 1000,
            'validation_examples': 0, 'calibration_examples': 0,
            'head_candidate_steps': 0,
        }]
        with self.assertRaisesRegex(ValueError, 'seeds'):
            summarize(rows)


if __name__ == '__main__':
    unittest.main()
