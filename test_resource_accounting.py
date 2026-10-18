"""Resource roles are counted from tensor payloads, not checkpoint size."""

import unittest

import numpy as np
import torch

from resource_accounting import account_method_state, payload_bytes


class ResourceAccountingTests(unittest.TestCase):
    def test_shared_torch_storage_is_counted_once(self):
        value = torch.arange(12, dtype=torch.float32)
        self.assertEqual(payload_bytes({'a': value, 'b': value.view(3, 4)}), 48)

    def test_numpy_payload_and_alias_are_counted_once(self):
        value = np.zeros((4, 3), dtype=np.float64)
        self.assertEqual(payload_bytes([value, value[:, :2]]), 96)

    def test_derpp_separates_examples_logits_and_labels(self):
        state = {
            'buffer_size': 2, 'num_seen': 9,
            'examples': torch.zeros(2, 3),
            'labels': torch.zeros(2, dtype=torch.long),
            'logits': torch.zeros(2, 5),
        }
        row = account_method_state('der_pp', state)
        self.assertEqual(row['raw_examples'], 2)
        self.assertEqual(row['raw_bytes'], 24)
        self.assertEqual(row['logits_bytes'], 40)
        self.assertEqual(row['other_bytes'], 16)
        self.assertEqual(row['method_learning_bytes'], 80)

    def test_sample_only_excludes_audit_bundle_and_has_no_prototypes(self):
        state = {
            'head_raw_replay': {0: torch.zeros(2, 3)},
            'global_protos': {}, 'prev_protos': {},
            'adaptive_audit_bundle': {'checkpoint_copy': torch.zeros(100)},
            'dep_tracker_contrib': torch.zeros(2),
            'fim_masks': [{}, {}],
            'has_old_teacher': True,
        }
        row = account_method_state('proto_evolve', state)
        self.assertEqual(row['raw_examples'], 2)
        self.assertEqual(row['raw_bytes'], 24)
        self.assertEqual(row['feature_bytes'], 0)
        self.assertEqual(row['other_bytes'], 8)
        self.assertEqual(row['audit_bytes'], 400)
        self.assertEqual(row['method_learning_bytes'], 32)
        self.assertTrue(row['unserialized_teacher'])

    def test_original_adaptive_prototypes_are_learning_state(self):
        state = {
            'head_raw_replay': {0: torch.zeros(2, 3)},
            'global_protos': {0: {'mean': torch.zeros(3),
                                  'std': torch.ones(3)}},
            'prev_protos': {},
            'adaptive_audit_bundle': None,
            'has_old_teacher': True,
        }
        row = account_method_state('proto_evolve', state)
        self.assertEqual(row['raw_bytes'], 24)
        self.assertEqual(row['feature_bytes'], 24)
        self.assertEqual(row['method_learning_bytes'], 48)

    def test_icarl_teacher_weights_are_learning_state(self):
        state = {
            'data': {0: torch.zeros(2, 3)},
            'teacher_bottoms': [{'w': torch.zeros(3, 3)}],
            'teacher_top': {'w': torch.zeros(2, 3)},
            'total_budget': 2,
        }
        row = account_method_state('icarl', state)
        self.assertEqual(row['raw_examples'], 2)
        self.assertEqual(row['teacher_bytes'], 60)
        self.assertEqual(row['method_learning_bytes'], 84)
        self.assertFalse(row['unserialized_teacher'])

    def test_hpcr_memory_contains_raw_logits_and_features(self):
        state = {'memory': {0: [(torch.zeros(3), 0,
                                 torch.zeros(5), torch.zeros(2))]}}
        row = account_method_state('hpcr', state)
        self.assertEqual(row['raw_examples'], 1)
        self.assertEqual(row['raw_bytes'], 12)
        self.assertEqual(row['logits_bytes'], 20)
        self.assertEqual(row['feature_bytes'], 8)

    def test_adagauss_gaussians_and_adapters_are_not_raw_examples(self):
        state = {
            'gaussians': {0: {'mean': torch.zeros(3),
                              'cov': torch.zeros(3, 3)}},
            'adapters': [{'weight': torch.zeros(2, 2)}],
        }
        row = account_method_state('adagauss', state)
        self.assertEqual(row['raw_examples'], 0)
        self.assertEqual(row['feature_bytes'], 48)
        self.assertEqual(row['other_bytes'], 16)

    def test_unknown_tensor_field_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'unclassified'):
            account_method_state('er', {
                'data': {0: torch.zeros(2, 3)},
                'surprise': torch.ones(4),
            })

    def test_same_storage_in_two_roles_is_rejected(self):
        shared = torch.zeros(2, 3)
        with self.assertRaisesRegex(ValueError, 'multiple roles'):
            account_method_state('der_pp', {
                'examples': shared, 'logits': shared,
            })


if __name__ == '__main__':
    unittest.main()
