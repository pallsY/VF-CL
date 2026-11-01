import unittest
from fractions import Fraction

from frozen_ablation_audit import party_kd_contribution_label, verify_run_gate
import test_frozen_ablation_run as run_tests


class AblationInterpretationTests(unittest.TestCase):
    def test_gate_binds_file_and_content_hashes_from_real_publication(self):
        case = run_tests.AblationRunTests()
        self.addCleanup(case.doCleanups)
        fixture = case.producer_fixture()
        from frozen_ablation_seal import verify_formal_seal
        gate = verify_formal_seal(fixture.run)
        self.assertEqual(verify_run_gate(fixture.run, gate), gate)
        for field in ('results_file_sha256', 'results_content_sha256',
                      'checkpoint_sha256', 'formal_published_sha256'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_run_gate(fixture.run, {**gate, field: '0' * 64})
        # The historical ambiguous field cannot satisfy the new gate schema.
        old = dict(gate)
        old['results_sha256'] = old.pop('results_file_sha256')
        with self.assertRaises(ValueError):
            verify_run_gate(fixture.run, old)

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
