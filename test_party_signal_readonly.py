"""Small checks for the predeclared party-signal diagnostic."""

import unittest

import torch

from analyze_party_signal_readonly import (
    analyze_scores, compare_seeds, normalized_margin, validate_output_root,
)


class PartySignalReadonlyTests(unittest.TestCase):
    def test_output_cannot_be_inside_a_frozen_source(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source'
            source.mkdir()
            with self.assertRaisesRegex(ValueError, 'outside'):
                validate_output_root(source / 'diagnostic', [source])
            validate_output_root(Path(tmp) / 'separate', [source])

    def test_margin_selector_uses_scores_without_labels(self):
        scores = torch.tensor([
            [[3.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            [[2.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ])
        margin = normalized_margin(scores)
        self.assertEqual(margin.shape, (2, 2))
        self.assertEqual(margin.argmax(0).tolist(), [1, 0])

    def test_oracle_and_label_free_selector_are_separate(self):
        # Two parties, four samples, three one-class tasks. Fusion makes two
        # errors; a party can correct both, while the fixed margin selector
        # need not recover both and can also damage correct fused predictions.
        party = torch.tensor([
            [[3., 1., 0.], [3., 1., 0.], [0., 3., 1.], [3., 1., 0.]],
            [[0., 2., 0.], [0., 1., 3.], [3., 1., 0.], [0., 1., 4.]],
        ])
        fused = torch.tensor([
            [3., 1., 0.], [0., 3., 1.], [0., 3., 1.], [3., 1., 0.],
        ])
        result = analyze_scores(party, fused, torch.tensor([0, 0, 1, 2]),
                                {0: [0], 1: [1], 2: [2]})
        summary = result['summary']
        self.assertEqual(summary['count'], 4)
        self.assertEqual(summary['fused_error_count'], 2)
        self.assertEqual(summary['oracle_rescue_count'], 2)
        self.assertEqual(summary['selector_rescue_count'], 2)
        self.assertEqual(summary['selector_damage_count'], 1)
        self.assertEqual(summary['selector_task_accuracy'], 0.75)
        self.assertEqual(len(result['per_class']), 3)
        self.assertNotIn('per_sample_predictions', result)

    def test_pair_report_detects_stable_class_specific_pattern(self):
        first = {'per_class': [
            {'class_id': 0, 'party_task_accuracy': [0.9, 0.1]},
            {'class_id': 1, 'party_task_accuracy': [0.1, 0.9]},
        ]}
        second = {'per_class': [
            {'class_id': 0, 'party_task_accuracy': [0.8, 0.2]},
            {'class_id': 1, 'party_task_accuracy': [0.2, 0.8]},
        ]}
        result = compare_seeds(first, second)
        self.assertEqual(result['unique_winner_classes'], 2)
        self.assertEqual(result['winner_agreement_fraction'], 1.0)
        self.assertAlmostEqual(result['class_party_residual_correlation'], 1.0)


if __name__ == '__main__':
    unittest.main()
