"""The opt-in resource-lite mode cannot retain prototypes or a teacher."""

import tempfile
import unittest
from unittest import mock

import torch

from config import validate_sample_replay_only
from cl_methods.proto_evolve import ProtoEvolveCL
from test_proto_evolve_adaptive_head import TinyTrainer
from test_sample_replay_only_formal import sample_args


def lite_args(output_dir):
    args = sample_args(output_dir)
    args.resource_lite = 1
    args.head_consolidation_mode = 'full_classifier'
    args.head_consolidation_samples_per_class = 18
    args.distill_weight = 0.0
    args.party_kd_enabled = 0
    args.dep_tracking_enabled = 0
    args.expected_party_kd_variant = 'base'
    return args


class AdaptiveResourceLiteTests(unittest.TestCase):
    def test_only_declared_teacher_free_train_only_head_is_accepted(self):
        with tempfile.TemporaryDirectory() as output_dir:
            args = lite_args(output_dir)
            validate_sample_replay_only(args)
            disabled = lite_args(output_dir)
            disabled.sample_replay_only = 0
            with self.assertRaisesRegex(ValueError, 'sample_replay_only'):
                validate_sample_replay_only(disabled)
            for field, value in (
                    ('head_consolidation_mode', 'adaptive_dual_branch'),
                    ('head_consolidation_samples_per_class', 20),
                    ('distill_weight', 0.25),
                    ('party_kd_enabled', 1),
                    ('proto_sdc', True),
                    ('feat_distill_weight', 0.05),
                    ('proto_replay_gradient_enabled', 1)):
                with self.subTest(field=field):
                    changed = lite_args(output_dir)
                    setattr(changed, field, value)
                    with self.assertRaisesRegex(ValueError, field):
                        validate_sample_replay_only(changed)

    def test_task_boundary_keeps_eighteen_raw_and_no_teacher(self):
        with tempfile.TemporaryDirectory() as output_dir:
            method = ProtoEvolveCL(TinyTrainer(), lite_args(output_dir))
            method.current_task_classes = [0]
            moments = {0: {'mean': torch.zeros(2), 'std': torch.ones(2)}}
            raw = {0: torch.zeros(18, 4)}
            with mock.patch.object(method, '_compute_fim'), \
                    mock.patch.object(method, '_compute_protos',
                                      return_value=(moments, raw)) as compute, \
                    mock.patch.object(method, '_consolidate_head'):
                method.after_task(None, 0)
            self.assertEqual(compute.call_args.kwargs['replay_capacity'], 18)
            self.assertEqual(method.head_raw_replay[0].shape[0], 18)
            self.assertEqual(method.global_protos, {})
            self.assertEqual(method.prev_protos, {})
            self.assertIsNone(method._old_top)
            self.assertIsNone(method._old_bottoms)
            self.assertFalse(method.get_state()['has_old_teacher'])

    def test_loaded_teacher_or_prototypes_are_rejected(self):
        with tempfile.TemporaryDirectory() as output_dir:
            method = ProtoEvolveCL(TinyTrainer(), lite_args(output_dir))
            with self.assertRaisesRegex(ValueError, 'teacher'):
                method.load_state({'has_old_teacher': True})
            with self.assertRaisesRegex(ValueError, 'prototypes'):
                method.load_state({'global_protos': {0: 1}})

    def test_final_full_head_uses_only_training_replay(self):
        with tempfile.TemporaryDirectory() as output_dir:
            method = ProtoEvolveCL(TinyTrainer(), lite_args(output_dir))
            method.head_raw_replay = {
                0: torch.randn(18, 4), 1: torch.randn(18, 4),
            }
            method.head_task_classes = {0: [0], 1: [1]}
            audit = method._consolidate_head(1)
            self.assertEqual(audit['mode'], 'full_classifier')
            self.assertEqual(len(method.head_consolidation_history), 1)
            self.assertEqual(method.head_validation_sha256, '')
            self.assertIsNone(method.adaptive_audit_bundle)
            self.assertEqual(method.trainer.top_model._adaptive_full_weight.numel(), 0)


if __name__ == '__main__':
    unittest.main()
