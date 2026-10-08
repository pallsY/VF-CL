import unittest

from analyze_cifar_selection_view_pilot import summarize_classes


class SelectionViewAnalyzerTests(unittest.TestCase):
    def test_summary_preserves_paired_sign_and_overlap(self):
        row = {
            'selection_time': {'stochastic': .3, 'deterministic': .2},
            'final_aggregate': {
                'stochastic': .2, 'deterministic': .1, 'offline': .05,
            },
            'final_parties': [
                {'stochastic': .2, 'deterministic': .1, 'offline': .05},
                {'stochastic': .1, 'deterministic': .2, 'offline': .05},
                {'stochastic': .2, 'deterministic': .1, 'offline': .05},
                {'stochastic': .2, 'deterministic': .1, 'offline': .05},
            ],
            'content_overlap_count': 7,
        }
        result = summarize_classes([row for _ in range(100)])
        self.assertAlmostEqual(result['selection_time']['s_minus_d_mean'], .1)
        self.assertAlmostEqual(result['final_aggregate']['s_minus_d_mean'], .1)
        self.assertAlmostEqual(result['final_aggregate']['d_minus_o_mean'], .05)
        self.assertEqual(result['final_aggregate']['fraction_s_better'], 0.0)
        self.assertAlmostEqual(result['final_parties'][1]['s_minus_d_mean'], -.1)
        self.assertEqual(result['content_overlap_total'], 700)

    def test_rejects_incomplete_class_panel(self):
        with self.assertRaisesRegex(ValueError, '100 class'):
            summarize_classes([])


if __name__ == '__main__':
    unittest.main()
