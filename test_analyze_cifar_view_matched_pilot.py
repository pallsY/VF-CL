import unittest

from analyze_cifar_view_matched_pilot import summarize_classes


class ViewMatchedAnalyzerTests(unittest.TestCase):
    def test_paired_summary_keeps_negative_view_effect(self):
        row = {
            'aggregate': {'augmented': .3, 'matched': .2, 'offline': .1},
            'parties': [
                {'augmented': .2, 'matched': .1, 'offline': .05},
                {'augmented': .1, 'matched': .2, 'offline': .05},
                {'augmented': .3, 'matched': .2, 'offline': .1},
                {'augmented': .3, 'matched': .2, 'offline': .1},
            ],
        }
        summary = summarize_classes([row for _ in range(100)])
        self.assertAlmostEqual(summary['aggregate']['view_gap_mean'], .1)
        self.assertAlmostEqual(summary['aggregate']['remaining_gap_mean'], .1)
        self.assertAlmostEqual(summary['aggregate']['augmented_median'], .3)
        self.assertEqual(summary['aggregate']['fraction_view_gap_positive'], 1.0)
        self.assertAlmostEqual(summary['parties'][1]['view_gap_mean'], -.1)
        self.assertEqual(summary['parties'][1]['fraction_view_gap_positive'], 0.0)

    def test_rejects_incomplete_class_panel(self):
        with self.assertRaisesRegex(ValueError, '100 class'):
            summarize_classes([])


if __name__ == '__main__':
    unittest.main()
