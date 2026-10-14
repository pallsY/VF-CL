"""Adaptive final-task audit semantics across post-task consolidation."""

import unittest

from replay11_numeric import compare_trajectory, post_final_task


HISTORY = [
    {'step': 'event_0_CIL', 'per_task_accs': {'task_0': 0.8}},
    {'step': 'event_1_CIL',
     'per_task_accs': {'task_0': 0.7, 'task_1': 0.9},
     'per_task_accs_taskil': {'task_0': 0.8, 'task_1': 1.0}},
]
TOTALS = {'task_0': 10, 'task_1': 10}
OBSERVED = {
    'diagonal': {'task_0': 8, 'task_1': 4},
    'final': {'task_0': 7, 'task_1': 9},
    'task_il': {'task_0': 8, 'task_1': 10},
}


class AdaptiveFinalTaskTests(unittest.TestCase):
    def test_final_state_rule_requires_matching_frozen_protocol(self):
        config = {'cl_method': 'proto_evolve',
                  'head_consolidation_enabled': 1,
                  'head_consolidation_mode': 'adaptive_dual_branch',
                  'num_tasks': 10}
        self.assertTrue(post_final_task(config, dict(config)))
        changed = dict(config, head_consolidation_mode='full_classifier')
        with self.assertRaisesRegex(ValueError, 'frozen protocol differs'):
            post_final_task(config, changed)

    def test_final_task_uses_post_consolidation_checkpoint(self):
        metrics = compare_trajectory(
            HISTORY, TOTALS, OBSERVED, post_final=True)
        self.assertEqual(metrics['aa_final_cil'], 0.8)
        self.assertAlmostEqual(metrics['bwt_cil'], -0.1)
        self.assertEqual(metrics['task_il_final'], 0.9)

    def test_prior_task_mismatch_still_fails(self):
        changed = dict(OBSERVED)
        changed['diagonal'] = {'task_0': 7, 'task_1': 4}
        with self.assertRaisesRegex(ValueError, 'diagonal integer differs'):
            compare_trajectory(HISTORY, TOTALS, changed, post_final=True)


if __name__ == '__main__':
    unittest.main()
