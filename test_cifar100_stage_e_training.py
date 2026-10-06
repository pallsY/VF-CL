import unittest

from cifar100_stage_e_training import expected_config, job_specs


class StageETrainingTests(unittest.TestCase):
    def test_stage_a_has_four_seed42_candidates(self):
        jobs = job_specs()
        self.assertEqual(len(jobs), 4)
        self.assertEqual({job.rsplit(":", 1)[1] for job in jobs}, {"42"})

    def test_balanced_replay_is_explicit(self):
        config = expected_config("mean_l0.15_m0.00:42")
        self.assertEqual(config["proto_replay_loss_norm"], "sample_mean")
        self.assertEqual(config["proto_replay_ratio"], 1.0)
        self.assertEqual(config["proto_lambda_a"], 0.15)
        self.assertEqual(config["distill_weight"], 0.25)
        self.assertEqual(config["feat_distill_weight"], 0.05)
        self.assertEqual(config["fim_freeze_frac"], 0.0)

    def test_validation_only_protocol_is_preserved(self):
        config = expected_config("mean_l0.10_m0.05:44")
        self.assertEqual(config["lambda_validation_enabled"], 1)
        self.assertEqual(config["seed"], 44)


if __name__ == "__main__": unittest.main()
