import unittest
from pathlib import Path

from isolet_stage_i import (
    CONFIGS,
    FIXED_TCB,
    ISOLET_TASK_IL_FLOOR,
    parse_candidate,
    should_run_s3,
    stage_i_command,
)


def option(command, name):
    return command[command.index(name) + 1]


class ISOLETStageITests(unittest.TestCase):
    def test_four_configs_are_atomic_and_frozen(self):
        self.assertEqual(set(CONFIGS), {
            "s0_incumbent",
            "s1_replay_ratio_1.5",
            "s2_proto_lambda_0.25",
            "s3_distill_0.35",
        })
        self.assertEqual(CONFIGS["s0_incumbent"], {})
        self.assertTrue(all(
            len(CONFIGS[name]) == 1 for name in CONFIGS if name != "s0_incumbent"
        ))
        self.assertEqual(FIXED_TCB["name"], "tcb_c0.001_t0.001_g1")
        self.assertEqual(ISOLET_TASK_IL_FLOOR, 0.9813)

    def test_incumbent_is_reference_only(self):
        with self.assertRaises(ValueError):
            parse_candidate("s0_incumbent")
        with self.assertRaises(ValueError):
            parse_candidate("unknown")

    def test_commands_change_only_the_registered_method_parameter(self):
        root = Path("/tmp/stage-i-test")
        commands = {
            name: stage_i_command(name, "cuda:0", root / name)
            for name in CONFIGS if name != "s0_incumbent"
        }
        self.assertEqual(
            option(commands["s1_replay_ratio_1.5"], "--proto_replay_ratio"),
            "1.5",
        )
        self.assertEqual(
            option(commands["s2_proto_lambda_0.25"], "--proto_lambda_a"),
            "0.25",
        )
        self.assertEqual(
            option(commands["s3_distill_0.35"], "--distill_weight"),
            "0.35",
        )
        for command in commands.values():
            self.assertEqual(option(command, "--task_ce_mode"), "current")
            self.assertEqual(option(command, "--optimizer"), "adamw")
            self.assertEqual(option(command, "--lr"), "0.001")
            self.assertEqual(option(command, "--lambda_validation_enabled"), "1")
            self.assertEqual(option(command, "--lambda_validation_per_class"), "40")
            self.assertEqual(option(command, "--seed"), "42")

    def test_s3_runs_only_after_both_initial_candidates_fail(self):
        failed = [{"passed": False}]
        completed = ["s1_replay_ratio_1.5", "s2_proto_lambda_0.25"]
        self.assertTrue(should_run_s3(completed, failed))
        self.assertFalse(should_run_s3(completed[:1], failed))
        self.assertFalse(should_run_s3(completed, [{"passed": True}]))
        self.assertFalse(should_run_s3(
            completed + ["s3_distill_0.35"], failed
        ))


if __name__ == "__main__":
    unittest.main()
