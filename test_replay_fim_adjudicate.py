from fractions import Fraction
import unittest

import replay_fim_adjudicate as a


class ReplayFIMAdjudicateTests(unittest.TestCase):
    def test_stage_a_requires_three_wins_and_uses_maximin(self):
        base = dict(cifar100=Fraction(1, 2), isolet=Fraction(1, 2),
                    upmc_food101=Fraction(1, 2))
        scores = {
            'f0': dict(cifar100=Fraction(51, 100), isolet=Fraction(51, 100),
                       upmc_food101=Fraction(51, 100)),
            'f1': dict(cifar100=Fraction(53, 100), isolet=Fraction(52, 100),
                       upmc_food101=Fraction(52, 100)),
            'f_off': dict(cifar100=Fraction(55, 100), isolet=Fraction(55, 100),
                          upmc_food101=Fraction(49, 100)),
        }
        self.assertEqual(a.choose_policy(scores, base), 'f1')
        scores['f1']['isolet'] = Fraction(1, 2)
        self.assertEqual(a.choose_policy(scores, base), 'f0')
        scores['f0']['isolet'] = Fraction(1, 2)
        self.assertIsNone(a.choose_policy(scores, base))

    def test_party_kd_gate_needs_half_point_on_two_and_no_large_harm(self):
        full = {name: Fraction(1, 2) for name in a.DATASETS}
        no = {'cifar100': Fraction(49, 100),
              'isolet': Fraction(99, 200),
              'upmc_food101': Fraction(101, 200)}
        self.assertTrue(a.party_kd_supported(full, no))
        no['upmc_food101'] = Fraction(51, 100)
        self.assertFalse(a.party_kd_supported(full, no))
        no['isolet'] = Fraction(499, 1000)
        no['upmc_food101'] = Fraction(1, 2)
        self.assertFalse(a.party_kd_supported(full, no))


if __name__ == '__main__':
    unittest.main()
