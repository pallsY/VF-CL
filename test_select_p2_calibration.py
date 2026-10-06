import unittest

from select_p2_calibration import select_method


def selection_fixture(beta_mean=0.180, best_mean=0.1819, joint_gain=None,
                      joint_positive_runs=4):
    raw = 0.15
    sequential = 0.175
    if joint_gain is not None:
        best_mean = sequential + joint_gain
        beta_mean = 0.16
    rows = []
    for index in range(4):
        for method, accuracy in (
            ('beta_only', beta_mean),
            ('sequential_alpha_beta', sequential),
            ('joint_alpha_beta', best_mean),
            ('alpha_only', 0.16),
        ):
            value = accuracy
            if method == 'joint_alpha_beta' and index >= joint_positive_runs:
                value = sequential
            rows.append({
                'run_key': f'run_{index}', 'method': method, 'budget': 25,
                'raw': {'overall_accuracy': raw, 'task_il': {'task_0': 0.5},
                        'nll': 3.0, 'ece': 0.4},
                'calibrated': {
                    'overall_accuracy': value, 'task_il': {'task_0': 0.5},
                    'nll': 2.0 if method != 'sequential_alpha_beta' else 2.1,
                    'ece': 0.2 if method != 'sequential_alpha_beta' else 0.21,
                },
                'audit': {'passed': True},
            })
    return rows


class SelectP2CalibrationTests(unittest.TestCase):
    def test_beta_only_wins_when_within_two_tenths_of_a_point(self):
        result = select_method(selection_fixture(beta_mean=0.180, best_mean=0.1819))
        self.assertEqual(result['selected_method'], 'beta_only')

    def test_joint_requires_half_point_and_all_four_positive(self):
        self.assertEqual(
            select_method(selection_fixture(joint_gain=0.0049))['selected_method'],
            'sequential_alpha_beta',
        )
        self.assertEqual(
            select_method(selection_fixture(
                joint_gain=0.005, joint_positive_runs=3
            ))['selected_method'],
            'sequential_alpha_beta',
        )
        self.assertEqual(
            select_method(selection_fixture(
                joint_gain=0.005, joint_positive_runs=4
            ))['selected_method'],
            'joint_alpha_beta',
        )

    def test_nonpositive_run_makes_method_ineligible(self):
        rows = selection_fixture(beta_mean=0.180, best_mean=0.181)
        beta = [row for row in rows if row['method'] == 'beta_only'][0]
        beta['calibrated']['overall_accuracy'] = beta['raw']['overall_accuracy']
        result = select_method(rows)
        self.assertFalse(result['methods']['beta_only']['eligible'])


if __name__ == '__main__':
    unittest.main()
