import unittest

from cifar100_stage_f import (
    CANDIDATES,
    eligible,
    expected_config,
    select_stage_a,
    stage_a_jobs,
)
from cl_methods.proto_evolve import effective_distill_weight


class DistillationScheduleTests(unittest.TestCase):
    def test_constant_schedule_preserves_base_weight(self):
        self.assertEqual(
            [effective_distill_weight(0.25, "constant", task) for task in range(10)],
            [0.25] * 10,
        )

    def test_decay_010_schedule_matches_protocol(self):
        actual = [effective_distill_weight(0.25, "stage_f_decay_010", task) for task in range(10)]
        self.assertEqual(actual, [0.25] * 5 + [0.15] * 3 + [0.10] * 2)

    def test_decay_005_schedule_matches_protocol(self):
        actual = [effective_distill_weight(0.25, "stage_f_decay_005", task) for task in range(10)]
        self.assertEqual(actual, [0.25] * 5 + [0.15] * 3 + [0.05] * 2)

    def test_unknown_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            effective_distill_weight(0.25, "unknown", 1)


class StageFProtocolTests(unittest.TestCase):
    def test_stage_a_has_three_seed_42_jobs(self):
        self.assertEqual(stage_a_jobs(), [f"{candidate}:42" for candidate in CANDIDATES])

    def test_candidate_changes_only_named_schedule_from_stage_e_choice(self):
        control = expected_config("constant_025:42")
        decay = expected_config("decay_010:42")
        self.assertEqual(control["distill_weight"], 0.25)
        self.assertEqual(control["feat_distill_weight"], 0.05)
        self.assertEqual(control["proto_lambda_a"], 0.15)
        self.assertEqual(control["proto_replay_loss_norm"], "sample_mean")
        changed = {key for key in control if control[key] != decay[key]}
        self.assertEqual(changed, {"distill_weight_schedule"})

    def test_gate_requires_all_three_metrics(self):
        good = {"aa_final_cil": 0.371, "bwt_cil": -0.17, "task_il_final": 0.756}
        self.assertTrue(eligible(good))
        for key, value in (("aa_final_cil", 0.369), ("bwt_cil", -0.181), ("task_il_final", 0.754)):
            bad = dict(good)
            bad[key] = value
            self.assertFalse(eligible(bad))

    def test_stage_a_promotes_at_most_two_by_aa_final(self):
        records = []
        for index, candidate in enumerate(CANDIDATES):
            records.append({
                "candidate": candidate,
                "metrics": {
                    "aa_final_cil": 0.371 + index * 0.001,
                    "bwt_cil": -0.17,
                    "task_il_final": 0.756,
                },
            })
        promoted = select_stage_a(records)
        self.assertEqual(promoted, ["decay_005", "decay_010"])


if __name__ == "__main__":
    unittest.main()
