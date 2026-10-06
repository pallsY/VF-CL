import unittest

import numpy as np

from offline_calibration_ablation import (
    CACHE_KEYS,
    evaluate_cache,
    nested_budget_indices,
    validate_cache_schema,
    validate_raw_replay,
)


class OfflineCalibrationAblationTests(unittest.TestCase):
    def test_nested_budgets_are_balanced_strict_subsets(self):
        manifest = {
            'per_class': 25,
            'by_class': {
                str(class_id): list(range(class_id * 25, (class_id + 1) * 25))
                for class_id in range(100)
            },
        }
        five = nested_budget_indices(manifest, 5)
        ten = nested_budget_indices(manifest, 10)
        twenty_five = nested_budget_indices(manifest, 25)

        self.assertEqual(len(five), 500)
        self.assertEqual(len(ten), 1000)
        self.assertEqual(len(twenty_five), 2500)
        self.assertLess(set(five), set(ten))
        self.assertLess(set(ten), set(twenty_five))

    def test_raw_replay_rejects_one_changed_prediction(self):
        with self.assertRaisesRegex(ValueError, 'raw replay mismatch'):
            validate_raw_replay(
                np.array([0, 1]), np.array([0, 2]), 0.5, 0.5
            )

    def test_cache_schema_allows_only_privacy_safe_arrays(self):
        cache = {
            'calibration_logits': np.zeros((2, 3)),
            'calibration_labels': np.zeros(2),
            'calibration_indices': np.arange(2),
            'test_logits': np.zeros((2, 3)),
            'test_labels': np.zeros(2),
        }
        self.assertEqual(set(CACHE_KEYS), set(cache))
        validate_cache_schema(cache)
        cache['images'] = np.zeros((2, 3, 32, 32))
        with self.assertRaisesRegex(ValueError, 'cache schema'):
            validate_cache_schema(cache)

    def test_evaluate_cache_emits_grid_and_sequential_reference(self):
        logits = np.array([
            [3., 1., 5., 4.], [1., 3., 4., 5.],
            [0., 1., 5., 2.], [1., 0., 2., 5.],
        ] * 3, dtype=np.float32)
        labels = np.array([0, 1, 2, 3] * 3)
        cache = {
            'calibration_logits': logits,
            'calibration_labels': labels,
            'calibration_indices': np.arange(12),
            'test_logits': logits,
            'test_labels': labels,
        }
        manifest = {
            'per_class': 3,
            'by_class': {'0': [0, 4, 8], '1': [1, 5, 9],
                         '2': [2, 6, 10], '3': [3, 7, 11]},
        }
        records = evaluate_cache(
            cache, {0: [0, 1], 1: [2, 3]}, manifest,
            sequential_state={'tasks': {}}, budgets=(1, 2, 3), steps=5,
        )
        self.assertEqual(len(records), 11)
        self.assertEqual(records[-1]['method'], 'sequential_alpha_beta')


if __name__ == '__main__':
    unittest.main()
