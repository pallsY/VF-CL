import unittest

import torch

from diagnose_party_task_evidence import summarize_task_evidence


class PartyEvidenceTests(unittest.TestCase):
    def test_oracle_coverage_is_diagnostic_only(self):
        parties = [
            torch.tensor([[3.0, 0.0], [4.0, 0.0]]),
            torch.tensor([[0.0, 4.0], [0.0, 3.0]]),
        ]
        fused = parties[0] + parties[1]
        result = summarize_task_evidence(
            parties, fused, torch.tensor([0, 1]), {0: [0], 1: [1]})
        self.assertEqual(result['party_task_accuracy'], [0.5, 0.5])
        self.assertEqual(result['fused_task_accuracy'], 0.0)
        self.assertEqual(result['fused_error_rescue_fraction'], 1.0)


if __name__ == '__main__':
    unittest.main()
