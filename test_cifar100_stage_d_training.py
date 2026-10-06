import unittest

from cifar100_stage_d_training import (
    CANDIDATES,
    expected_config,
    job_specs,
    parse_job,
)


class StageDTrainingTests(unittest.TestCase):
    def test_stage_a_has_six_seed42_neighbors(self):
        jobs = job_specs()
        self.assertEqual(len(jobs), 6)
        self.assertEqual({parse_job(job)[1] for job in jobs}, {42})
        self.assertEqual(len(CANDIDATES), 6)

    def test_distillation_neighbor_is_explicit(self):
        config = expected_config("d0.15_f0.05_m0.10:42")
        self.assertEqual(config["distill_weight"], 0.15)
        self.assertEqual(config["feat_distill_weight"], 0.05)
        self.assertEqual(config["fim_freeze_frac"], 0.10)
        self.assertEqual(config["lambda_validation_enabled"], 1)

    def test_fim_neighbor_preserves_other_incumbent_values(self):
        config = expected_config("d0.25_f0.05_m0.00:44")
        self.assertEqual(config["distill_weight"], 0.25)
        self.assertEqual(config["feat_distill_weight"], 0.05)
        self.assertEqual(config["fim_freeze_frac"], 0.00)
        self.assertEqual(config["seed"], 44)


if __name__ == "__main__":
    unittest.main()
