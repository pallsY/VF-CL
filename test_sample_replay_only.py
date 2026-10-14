import unittest
from types import SimpleNamespace

import config
import runner


class SampleReplayOnlyProtocolTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(
            seed=54, data="tabvfl", cl_method="proto_evolve",
            num_tasks=13, num_parties=4,
            head_consolidation_enabled=1,
            head_consolidation_mode="adaptive_dual_branch",
            head_consolidation_samples_per_class=20,
            proto_replay_gradient_enabled=0, proto_sdc=False,
            feat_distill_weight=0.0, party_drift_telemetry=0,
            sample_replay_only=1, formal_deferred_evaluation=False,
        )

    def test_mode_has_strict_configuration_and_checkpoint_identity(self):
        self.assertTrue(hasattr(config, "validate_sample_replay_only"))
        args = self.args()
        config.validate_sample_replay_only(args)
        self.assertIs(runner._checkpoint_protocol(args)["sample_replay_only"], True)
        args.sample_replay_only = 0
        self.assertNotIn("sample_replay_only", runner._checkpoint_protocol(args))

    def test_mode_rejects_incompatible_replay_configuration(self):
        self.assertTrue(hasattr(config, "validate_sample_replay_only"))
        invalid = {
            "cl_method": "er",
            "head_consolidation_enabled": 0,
            "head_consolidation_mode": "full_classifier",
            "head_consolidation_samples_per_class": 40,
            "proto_replay_gradient_enabled": 1,
            "proto_sdc": True,
            "feat_distill_weight": 0.02,
            "party_drift_telemetry": 1,
        }
        for field, value in invalid.items():
            with self.subTest(field=field):
                args = self.args()
                setattr(args, field, value)
                with self.assertRaisesRegex(ValueError, field):
                    config.validate_sample_replay_only(args)


if __name__ == "__main__":
    unittest.main()
