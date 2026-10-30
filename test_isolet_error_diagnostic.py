import unittest

from isolet_error_diagnostic import analyze, error_type, paired_excess, sign_pattern


class DiagnosticRulesTests(unittest.TestCase):
    def test_exclusive_error_types_follow_frozen_task_map(self):
        tasks = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
        self.assertEqual(error_type(0, 1, tasks), 'within_task')
        self.assertEqual(error_type(0, 4, tasks), 'old_to_new')
        self.assertEqual(error_type(0, 2, tasks), 'old_to_other_old_task')
        self.assertEqual(error_type(4, 0, tasks), 'new_to_old')
        with self.assertRaises(ValueError):
            error_type(0, 0, tasks)

    def test_paired_excess_uses_discordant_correctness(self):
        truth = [0, 1, 2, 3]
        f0 = [1, 1, 0, 3]
        icarl = [0, 0, 2, 3]
        self.assertEqual(paired_excess(truth, f0, icarl, [0, 1, 2, 3]),
                         (2, 1, 1))
        self.assertEqual(paired_excess(truth, f0, icarl, [0, 1]),
                         (1, 1, 0))
        self.assertEqual(sign_pattern([1, 2, 1, 3, 1]), 'f0_deficit')
        self.assertEqual(sign_pattern([1, 0, 1, 3, 1]), 'mixed_or_zero')

    def test_eight_groups_and_repeated_sample_identity(self):
        tasks = {task: [2 * task, 2 * task + 1] for task in range(13)}
        truth = list(range(26))
        f0 = truth.copy()
        f0[1] = 0
        paired = {seed: {
            'f0': {'ids': truth, 'truth': truth, 'pred': f0,
                   'tasks': tasks, 'holdout_sha': 'same',
                   'scores': [[0.0] * 26 for _ in truth]},
            'icarl': {'ids': truth, 'truth': truth, 'pred': truth,
                      'tasks': tasks, 'holdout_sha': 'same', 'scores': None},
        } for seed in (80, 81, 82, 83, 84)}
        tables, groups = analyze(paired)
        self.assertEqual(len(groups), 8)
        self.assertEqual(groups['e_set']['pattern'], 'f0_deficit')
        self.assertEqual(groups['e_set']['pooled_net'], 5)
        self.assertEqual(groups['e_set']['pooled_f0_only_wrong'], 5)
        self.assertEqual(groups['e_set']['pooled_icarl_only_wrong'], 0)
        self.assertEqual(groups['e_set']['pooled_f0_errors'], 5)
        self.assertEqual(groups['e_set']['pooled_icarl_errors'], 0)
        self.assertEqual(groups['non_e_set']['pattern'], 'mixed_or_zero')
        self.assertEqual(len(tables['sample']), 26)
        self.assertEqual(tables['sample'][1]['f0_only_wrong'], 5)


if __name__ == '__main__':
    unittest.main()
