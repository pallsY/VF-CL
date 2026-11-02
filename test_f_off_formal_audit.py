import unittest

from f_off_formal_audit import summarize, formal_score


class FormalAuditTests(unittest.TestCase):
    def test_cifar_uses_final_calibrated_bic_not_raw_head(self):
        result = {
            'cl_metrics': {'AA_final': 0.4},
            'bic_final': {
                'fit': {'mode': 'joint_alpha_beta'},
                'calibration_audit': {'passed': True, 'overlap_count': 0,
                                      'test_used_for_fit': False},
                'paired': {'calibrated': {'overall_accuracy': 0.5}},
            },
        }
        self.assertEqual(formal_score('cifar100', result), 0.5)
        result['bic_final']['calibration_audit']['test_used_for_fit'] = True
        with self.assertRaises(ValueError):
            formal_score('cifar100', result)
        self.assertEqual(formal_score('isolet', {'cl_metrics': {'AA_final': 0.9}}), 0.9)

    def test_three_seed_mean_and_sample_spread(self):
        rows = {
            'cifar100': {42: 0.3, 43: 0.4, 44: 0.5},
            'isolet': {42: 0.9, 43: 0.8, 44: 0.7},
            'upmc_food101': {42: 0.7, 43: 0.7, 44: 0.7},
        }
        result = summarize(rows)
        self.assertAlmostEqual(result['cifar100']['mean'], 0.4)
        self.assertAlmostEqual(result['cifar100']['sample_sd'], 0.1)
        self.assertEqual(result['upmc_food101']['sample_sd'], 0.0)
        with self.assertRaises(ValueError):
            summarize({'cifar100': rows['cifar100']})


if __name__ == '__main__':
    unittest.main()
