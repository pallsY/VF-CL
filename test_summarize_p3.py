import unittest

from summarize_p3 import seed_gate


def fixture(delta=0.006, task_il_delta=0.0, last_fraction_drop=0.1):
    return {
        'bic_final': {
            'fit': {'mode': 'joint_alpha_beta'},
            'paired': {
                'raw': {
                    'overall_accuracy': 0.10,
                    'task_prediction_fraction': {'task_9': 0.7},
                },
                'calibrated': {
                    'overall_accuracy': 0.10 + delta,
                    'task_prediction_fraction': {
                        'task_9': 0.7 - last_fraction_drop
                    },
                },
            },
            'task_il_max_abs_delta': task_il_delta,
            'calibration_audit': {'passed': True, 'test_used_for_fit': False},
            'privacy_audit': {'passed': True, 'test_used_for_fit': False},
        }
    }


class SummarizeP3Tests(unittest.TestCase):
    def test_gate_accepts_frozen_threshold(self):
        self.assertTrue(seed_gate(fixture(delta=0.005))['passed'])

    def test_gate_rejects_accuracy_taskil_and_fraction_failures(self):
        self.assertFalse(seed_gate(fixture(delta=0.0049))['passed'])
        self.assertFalse(seed_gate(fixture(task_il_delta=2e-6))['passed'])
        self.assertFalse(seed_gate(fixture(last_fraction_drop=0.0))['passed'])


if __name__ == '__main__':
    unittest.main()
