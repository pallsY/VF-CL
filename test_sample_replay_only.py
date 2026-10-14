import unittest
from types import SimpleNamespace
from unittest import mock

import config
import runner
import torch

from cl_methods.proto_evolve import ProtoEvolveCL
import run_replay_factorial_pilot as pilot


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


class SampleReplayOnlyMethodTests(unittest.TestCase):
    def method(self):
        method = ProtoEvolveCL.__new__(ProtoEvolveCL)
        method.args = SimpleNamespace(seed=54, num_tasks=10, device="cpu")
        method.trainer = SimpleNamespace(
            bottoms=[torch.nn.Linear(2, 2)],
            top_model=torch.nn.Linear(2, 2),
        )
        method.sample_replay_only = True
        method.head_consolidation_enabled = True
        method.head_consolidation_mode = "adaptive_dual_branch"
        method.head_consolidation_schedule = "final"
        method.head_consolidation_samples_per_class = 20
        method.current_task_classes = [7]
        method.global_protos = {}
        method.prev_protos = {}
        method.head_raw_replay = {}
        method.head_task_classes = {}
        method.dep_tracking_enabled = False
        method._old_bottoms = None
        method._old_top = None
        return method

    def test_after_task_keeps_raw_buffer_without_persistent_prototypes(self):
        method = self.method()
        method._compute_fim = mock.Mock()
        method._compute_protos = mock.Mock(return_value=(
            {7: {"mean": torch.ones(2), "std": torch.ones(2)}},
            {7: torch.randn(20, 2)},
        ))
        method._write_party_drift_record = mock.Mock()
        method._consolidate_head = mock.Mock()

        method.after_task(None, 0)

        self.assertEqual(method.global_protos, {})
        self.assertEqual(method.prev_protos, {})
        self.assertEqual(method.head_raw_replay[7].size(0), 20)
        method._consolidate_head.assert_called_once_with(0)

    def test_final_head_gets_class_ids_from_raw_buffer(self):
        method = self.method()
        method.head_raw_replay = {7: torch.randn(20, 2)}
        method._adaptive_ul_follows = lambda _task_id: False

        class ReachedRawEmbedding(Exception):
            pass

        method._embed_head_raw_replay = mock.Mock(
            side_effect=ReachedRawEmbedding
        )
        with self.assertRaises(ReachedRawEmbedding):
            method._consolidate_head(9)
        method._embed_head_raw_replay.assert_called_once_with((7,))

    def test_no_prototype_replay_uses_full_current_ce(self):
        method = self.method()
        method.lambda_A = 0.05
        method._old_top = None
        ce = torch.tensor(2.0, requires_grad=True)
        loss = method._proto_loss(
            method.trainer.bottoms, method.trainer.top_model,
            torch.zeros(2, 2), torch.tensor([7, 7]), loss_ce=ce,
        )
        loss.backward()
        self.assertEqual(ce.grad.item(), 1.0)

    def test_new_mode_rejects_checkpoint_with_prototype_cache(self):
        method = self.method()
        with self.assertRaisesRegex(ValueError, "contains prototypes"):
            method.load_state({
                "global_protos": {7: {"mean": torch.ones(2)}},
                "prev_protos": {},
            })


class SampleReplayOnlyPilotTests(unittest.TestCase):
    def test_readout_has_separate_variant_and_zero_prototype_memory(self):
        self.assertTrue(hasattr(pilot, "sample_only_result_fields"))
        args = SimpleNamespace(sample_replay_only=1)
        measured = {
            "prototype_bytes": 0,
            "raw_replay_per_class": 20,
            "raw_replay_bytes": 2000,
        }
        self.assertEqual(
            pilot.sample_only_result_fields(args, measured),
            {"variant": "sample_replay_only"},
        )
        self.assertEqual(
            pilot.sample_only_result_fields(
                SimpleNamespace(sample_replay_only=0), measured),
            {},
        )
        for key, value in (("prototype_bytes", 8),
                           ("raw_replay_per_class", 0),
                           ("raw_replay_bytes", 0)):
            with self.subTest(key=key):
                bad = {**measured, key: value}
                with self.assertRaisesRegex(ValueError, key):
                    pilot.sample_only_result_fields(args, bad)


if __name__ == "__main__":
    unittest.main()
