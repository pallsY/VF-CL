import unittest
from fractions import Fraction

from frozen_ablation_audit import party_kd_contribution_label


class AblationInterpretationTests(unittest.TestCase):
    def test_party_kd_requires_two_strict_dataset_mean_drops(self):
        deltas = {'cifar100': Fraction(1, 100),
                  'isolet': Fraction(1, 1000),
                  'upmc_food101': Fraction(-1, 50)}
        self.assertEqual(party_kd_contribution_label(deltas), 'supported')
        deltas['isolet'] = Fraction(0)
        self.assertEqual(party_kd_contribution_label(deltas),
                         'inconsistent')
        with self.assertRaises(ValueError):
            party_kd_contribution_label({'isolet': Fraction(1, 10)})


if __name__ == '__main__':
    unittest.main()
