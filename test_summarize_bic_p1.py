import copy
import unittest

from summarize_bic_p1 import evaluate_pilot, summarize_formal


def fixture(delta=0.005):
    return {
        'bic_final': {
            'paired': {
                'raw': {
                    'overall_accuracy': 0.140,
                    'task_il': {'task_0': 0.7, 'task_9': 0.8},
                    'task_prediction_fraction': {'task_9': 0.75},
                },
                'calibrated': {
                    'overall_accuracy': 0.140 + delta,
                    'task_il': {'task_0': 0.7, 'task_9': 0.8},
                    'task_prediction_fraction': {'task_9': 0.60},
                },
            },
            'calibration_audit': {'passed': True, 'test_used_for_fit': False},
            'disabled_identity_max_abs_diff': 0.0,
            'task_il_max_abs_delta': 0.0,
        }
    }


class PilotGateTests(unittest.TestCase):
    def test_exact_threshold_passes_but_just_below_fails(self):
        self.assertTrue(evaluate_pilot(fixture(0.005))['passed'])
        self.assertFalse(evaluate_pilot(fixture(0.004999))['passed'])

    def test_every_frozen_condition_is_required(self):
        mutations = [
            lambda value: value['bic_final']['paired']['calibrated']['task_il'].__setitem__('task_0', 0.69),
            lambda value: value['bic_final']['paired']['calibrated']['task_prediction_fraction'].__setitem__('task_9', 0.75),
            lambda value: value['bic_final']['calibration_audit'].__setitem__('passed', False),
            lambda value: value['bic_final']['calibration_audit'].__setitem__('test_used_for_fit', True),
            lambda value: value['bic_final'].__setitem__('disabled_identity_max_abs_diff', 1e-5),
        ]
        for mutate in mutations:
            value = copy.deepcopy(fixture())
            mutate(value)
            self.assertFalse(evaluate_pilot(value)['passed'])

    def test_formal_summary_reports_each_paired_delta(self):
        summary = summarize_formal({'static42': fixture(0.005), 'uniform43': fixture(0.01)})
        self.assertEqual(summary['runs']['static42']['delta_aa_final'], 0.005)
        self.assertAlmostEqual(summary['runs']['uniform43']['delta_aa_final'], 0.01)
        self.assertEqual(summary['positive_runs'], 2)


if __name__ == '__main__':
    unittest.main()
