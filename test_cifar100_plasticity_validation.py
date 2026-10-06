import tempfile
import unittest
from pathlib import Path

from cifar100_plasticity_validation import (
    CANDIDATES,
    claim_next,
    command_for,
    expected_config,
    job_specs,
    parse_job,
)


class PlasticityValidationTests(unittest.TestCase):
    def test_jobs_are_unique_seed42_candidates(self):
        jobs = job_specs()
        self.assertEqual(len(jobs), 4)
        self.assertEqual(len(set(jobs)), 4)
        self.assertEqual({parse_job(job)[1] for job in jobs}, {42})
        self.assertEqual({parse_job(job)[0] for job in jobs}, set(CANDIDATES))

    def test_candidate_overrides_are_scoped(self):
        baseline = expected_config("fim_0.10:42")
        no_fim = expected_config("fim_0.00:42")
        low_feat = expected_config("feat_0.03_fim_0.10:42")
        low_kd = expected_config("distill_0.25_fim_0.10:42")
        self.assertEqual(baseline["feat_distill_weight"], 0.05)
        self.assertEqual(baseline["fim_freeze_frac"], 0.10)
        self.assertEqual(no_fim["fim_freeze_frac"], 0.0)
        self.assertEqual(low_feat["feat_distill_weight"], 0.03)
        self.assertEqual(low_kd["distill_weight"], 0.25)
        self.assertEqual(low_kd["party_kd_lambda"], 1.0)
        self.assertEqual(low_kd["lambda_validation_enabled"], 1)

    def test_command_has_unique_output_name_and_no_test_selection(self):
        command = command_for(
            "fim_0.10:42", "/study", "/repo", "/python"
        )
        pairs = dict(zip(command[2::2], command[3::2]))
        self.assertEqual(pairs["--lambda_validation_enabled"], "1")
        self.assertEqual(pairs["--seed"], "42")
        self.assertIn("fim_0p10", pairs["--exp_name"])

    def test_claims_are_atomic_and_exhaustive(self):
        with tempfile.TemporaryDirectory() as tmp:
            claimed = [claim_next(Path(tmp)) for _ in range(5)]
        self.assertEqual(claimed[:4], job_specs())
        self.assertIsNone(claimed[4])

    def test_stage_b_seeds_are_supported(self):
        self.assertEqual(parse_job("fim_0.10:43"), ("fim_0.10", 43))
        self.assertEqual(parse_job("fim_0.10:44"), ("fim_0.10", 44))

    def test_unknown_seed_or_candidate_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_job("fim_0.10:45")
        with self.assertRaises(ValueError):
            parse_job("unknown:42")


if __name__ == "__main__":
    unittest.main()
