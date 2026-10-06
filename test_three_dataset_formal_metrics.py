"""Strict regression tests for common formal metric reconstruction."""
import copy
import math
import unittest

from three_dataset_formal_metrics import (
    FORMULA_VERSION, FormalMetrics, reconstruct_metrics,
)


def canonical_rows():
    return [
        {'step': 'event_0_CIL', 'values': {'task_0': .80}},
        {'step': 'event_1_CIL', 'values': {'task_0': .70, 'task_1': .75}},
        {'step': 'event_2_CIL',
         'values': {'task_0': .65, 'task_1': .78, 'task_2': .70}},
    ]


class FormalMetricReconstructionTests(unittest.TestCase):
    def test_final_minus_diagonal_formula_and_immutable_result(self):
        rows = canonical_rows()
        metrics = reconstruct_metrics(rows, rows, (0, 1, 2))
        self.assertEqual('final-minus-diagonal-v1', FORMULA_VERSION)
        self.assertIsInstance(metrics, FormalMetrics)
        self.assertAlmostEqual((.65 + .78 + .70) / 3, metrics.aa_final)
        self.assertAlmostEqual(-.06, metrics.bwt)
        self.assertAlmostEqual((.65 + .78 + .70) / 3, metrics.taskil_final)
        self.assertEqual((.80, (.70 + .75) / 2, (.65 + .78 + .70) / 3),
                         metrics.aa_trajectory)
        self.assertEqual((.65, .78, .70), metrics.class_final)
        self.assertEqual((.65, .78, .70), metrics.taskil_final_by_task)
        with self.assertRaises(AttributeError):
            metrics.bwt = 0.0

    def test_nonconsecutive_authoritative_task_ids_are_not_relabelled(self):
        rows = [
            {'step': 'event_0_CIL', 'values': {'task_10': .9}},
            {'step': 'event_1_CIL',
             'values': {'task_10': .7, 'task_40': .8}},
        ]
        metrics = reconstruct_metrics(rows, rows, (10, 40))
        self.assertEqual((.7, .8), metrics.class_final)
        self.assertAlmostEqual(-.2, metrics.bwt)

    def test_rejects_missing_final_or_diagonal_row(self):
        rows = canonical_rows()
        for broken in (rows[:-1], [rows[1], rows[2], rows[0]]):
            with self.subTest(rows=broken), self.assertRaises(ValueError):
                reconstruct_metrics(broken, broken, (0, 1, 2))

    def test_rejects_duplicate_or_reordered_steps(self):
        rows = canonical_rows()
        duplicate = copy.deepcopy(rows)
        duplicate[1]['step'] = 'event_0_CIL'
        for broken in (duplicate, [rows[0], rows[2], rows[1]]):
            with self.subTest(rows=broken), self.assertRaises(ValueError):
                reconstruct_metrics(broken, broken, (0, 1, 2))

    def test_rejects_unexpected_missing_or_extra_row_keys(self):
        rows = canonical_rows()
        unexpected = copy.deepcopy(rows)
        unexpected[1]['values']['task_99'] = .9
        missing = copy.deepcopy(rows)
        del missing[2]['values']['task_1']
        extra = copy.deepcopy(rows)
        extra[0]['source'] = 'untrusted'
        for broken in (unexpected, missing, extra):
            with self.subTest(rows=broken), self.assertRaises(ValueError):
                reconstruct_metrics(broken, broken, (0, 1, 2))

    def test_rejects_invalid_task_identities(self):
        rows = canonical_rows()
        for broken_rows, ids in ((rows, ()), ([rows[0]], (0,)),
                                 ([rows[0], rows[1]], (0, 0)),
                                 ([rows[0], rows[1]], (False, 1))):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                reconstruct_metrics(broken_rows, broken_rows, ids)

    def test_rejects_booleans_strings_and_nonfinite_values(self):
        for value in (True, '0.8', math.nan, math.inf, -math.inf):
            rows = canonical_rows()
            rows[0]['values']['task_0'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                reconstruct_metrics(rows, rows, (0, 1, 2))

    def test_rejects_classil_taskil_identity_progression_disagreement(self):
        class_rows = canonical_rows()
        taskil_rows = copy.deepcopy(class_rows)
        taskil_rows[1]['values'] = {'task_0': .7, 'task_2': .75}
        with self.assertRaises(ValueError):
            reconstruct_metrics(class_rows, taskil_rows, (0, 1, 2))

    def test_does_not_mutate_inputs_and_repeated_calls_are_equal(self):
        class_rows = canonical_rows()
        taskil_rows = copy.deepcopy(class_rows)
        before = copy.deepcopy((class_rows, taskil_rows))
        first = reconstruct_metrics(class_rows, taskil_rows, (0, 1, 2))
        second = reconstruct_metrics(class_rows, taskil_rows, (0, 1, 2))
        self.assertEqual(before, (class_rows, taskil_rows))
        self.assertEqual(first, second)
        self.assertTrue(all(isinstance(value, tuple) for value in (
            first.aa_trajectory, first.class_final, first.taskil_final_by_task,
        )))


if __name__ == '__main__':
    unittest.main()
