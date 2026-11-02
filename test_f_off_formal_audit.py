import unittest

from f_off_formal_audit import summarize


class FormalAuditTests(unittest.TestCase):
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
