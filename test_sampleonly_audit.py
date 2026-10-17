"""Fail-closed checks for the independent sample-only auditor."""

import unittest

import torch

import adaptive9_evidence as evidence


class SampleOnlyAuditTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            'sample_replay_only': 1,
            'proto_replay_gradient_enabled': 0,
            'proto_sdc': False,
            'feat_distill_weight': 0.0,
            'head_consolidation_samples_per_class': 20,
            'head_consolidation_schedule': 'final',
            'distill_weight': 0.25,
            'distill_weight_schedule': 'constant',
            'proto_lambda_a': 0.15,
            'proto_replay_loss_norm': 'sample_mean',
            'dep_tracking_enabled': 1,
            'party_kd_enabled': 1,
            'party_kd_mode': 'uniform',
        }
        self.checkpoint = {
            'seen_task_classes': {0: [0, 1]},
            'cl_state': {
                'global_protos': {}, 'prev_protos': {},
                'head_raw_replay': {0: torch.zeros(20, 4),
                                    1: torch.zeros(20, 4)},
            },
        }

    def test_accepts_only_twenty_raw_examples_per_seen_class(self):
        self.assertEqual(evidence.verify_sample_only_state(
            self.checkpoint, self.config), 2)

    def test_rejects_prototypes_wrong_count_and_missing_class(self):
        variants = (
            {'global_protos': {0: {'mean': torch.zeros(4)}}},
            {'prev_protos': {0: {'mean': torch.zeros(4)}}},
            {'head_raw_replay': {0: torch.zeros(19, 4),
                                 1: torch.zeros(20, 4)}},
            {'head_raw_replay': {0: torch.zeros(20, 4)}},
        )
        for changed in variants:
            with self.subTest(changed=changed.keys()):
                checkpoint = dict(self.checkpoint,
                                  cl_state=dict(self.checkpoint['cl_state'],
                                                **changed))
                with self.assertRaisesRegex(ValueError, 'sample-only'):
                    evidence.verify_sample_only_state(checkpoint,
                                                      self.config)

    def test_rejects_feature_switches(self):
        for key, value in (
                ('sample_replay_only', 0),
                ('proto_replay_gradient_enabled', 1),
                ('proto_sdc', True),
                ('feat_distill_weight', 0.05)):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, 'sample-only'):
                    evidence.verify_sample_only_state(
                        self.checkpoint, dict(self.config, **{key: value}))

    def test_rejects_extra_training_advantages(self):
        for key, value in (
                ('distill_weight', 0.5),
                ('head_consolidation_schedule', 'every'),
                ('head_consolidation_samples_per_class', 40),
                ('party_kd_mode', 'static')):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, 'sample-only'):
                    evidence.verify_sample_only_state(
                        self.checkpoint, dict(self.config, **{key: value}))


if __name__ == '__main__':
    unittest.main()
