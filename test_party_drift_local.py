import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from calibration_split import build_manifest
from party_drift_local import (
    attenuation_weights,
    build_tuning_holdout,
    decode_notification,
    encode_notification,
    message_ledger,
    replay_step_message_bytes,
    replay_step_audit,
    rows_from_audit,
    verify_audit_record,
    verify_replay_invariants,
)


class PartyDriftLocalTests(unittest.TestCase):
    def test_holdout_is_disjoint_and_refuses_wrong_source_hash(self):
        labels = np.repeat(np.arange(3), 60)
        old = build_manifest(labels, 10, 17)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'labels.npy'
            np.save(source, labels)
            import hashlib
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            target = Path(tmp) / 'holdout.json'
            result = build_tuning_holdout(
                labels, source, digest, old, None, target, seed=20261023,
                per_class=30,
            )
            self.assertEqual(len(result['ordered_indices']), 90)
            self.assertFalse(set(result['ordered_indices']) & set(old['ordered_indices']))
            self.assertEqual(json.loads(target.read_text())['sha256'], result['sha256'])
            with self.assertRaises(FileExistsError):
                build_tuning_holdout(labels, source, digest, old, None, target,
                                     seed=20261023, per_class=30)
            with self.assertRaises(ValueError):
                build_tuning_holdout(labels, source, '0' * 64, old, None,
                                     Path(tmp) / 'bad.json', seed=20261023,
                                     per_class=30)

    def test_notification_is_exactly_sixteen_bytes(self):
        payload = encode_notification(3, 4, 1025, 2)
        self.assertEqual(len(payload), 16)
        self.assertEqual(decode_notification(payload), (3, 4, 1025, 2))
        with self.assertRaises(ValueError):
            encode_notification(-1, 0, 0, 0)
        self.assertEqual(replay_step_message_bytes([0, 2], [2, 3, 4], 4),
                         3 * 4 * (2 + 4) * 4 + 2 * 4 * 8 + 2 * 16)

    def test_ledger_counts_both_run_configs_and_excludes_task_zero_replay(self):
        rows = [
            {'train': 20, 'classes': 2},
            {'train': 20, 'classes': 2},
        ]
        dev = message_ledger(rows, epochs=1, current_batch=8,
                             replay_batch=4, embedding_dims=[2, 2],
                             tuning_per_class=2, probe_per_class=1,
                             full_every=2)
        final = message_ledger(rows, epochs=1, current_batch=8,
                               replay_batch=4, embedding_dims=[2, 2],
                               tuning_per_class=0, probe_per_class=1,
                               full_every=2)
        self.assertEqual(dev['replay_steps'], 2)
        self.assertEqual(dev['full_replay_bytes'], 2 * 4 * 4 * 4 * 3)
        self.assertEqual(dev['selected_party_visits'], 3)
        self.assertEqual(dev['control_bytes'], 3 * 16 + 2 * 16)
        self.assertEqual(dev['probe_bytes'], 2 * (2 + 2) * 4 * 2)
        self.assertEqual(dev['tuning_bytes'], 8 * 2 * 2 * 4)
        self.assertEqual(final['tuning_bytes'], 0)
        self.assertNotEqual(dev['total_reference_bytes'], final['total_reference_bytes'])

    def test_audit_replays_selection_without_consuming_torch_rng(self):
        before = torch.get_rng_state().clone()
        a = replay_step_audit([0.7, 0.2, 0.1], 2, 42, 1, 0, 0, 4)
        b = replay_step_audit([0.7, 0.2, 0.1], 2, 42, 1, 0, 0, 4)
        self.assertEqual(a, b)
        self.assertEqual(len(a['selected']), 2)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        full = replay_step_audit([0.7, 0.2, 0.1], 2, 42, 1, 0, 3, 4)
        self.assertEqual(full['selected'], [0, 1, 2])
        counted = replay_step_audit([0.7, 0.2, 0.1], 2, 42, 1, 0, 0, 4,
                                    embedding_dims=[2, 2, 2], replay_batch=4)
        self.assertEqual(counted['message_bytes'],
                         replay_step_message_bytes(counted['selected'],
                                                   [2, 2, 2], 4))
        verify_audit_record(counted, 2)
        counted['message_bytes'] += 1
        with self.assertRaises(ValueError):
            verify_audit_record(counted, 2)
        extended = {**a, 'logical_replay_ids': [[0, 1]],
                    'duplicate_samples': 0,
                    'forward_calls': [int(i in a['selected']) for i in range(3)]}
        verify_audit_record(extended, 2)
        verify_audit_record(a, 2)
        changed = {**a, 'selected': [2]}
        with self.assertRaises(ValueError):
            verify_audit_record(changed, 2)

    def test_control_audit_keeps_arm_and_alpha_without_changing_selection(self):
        row = replay_step_audit([0.5, 0.5], 1, 70, 1, 0, 0, 4)
        row.update({'replay_arm': 'uniform', 'replay_alpha': 0.5})
        verify_audit_record(row, 1)
        row['replay_alpha'] = float('nan')
        with self.assertRaises(ValueError):
            verify_audit_record(row, 1)

    def test_cmcdr_audit_validates_intervention_count(self):
        row = replay_step_audit([0.5, 0.5], 2, 70, 1, 0, 0, 4,
                                embedding_dims=[2, 2], replay_batch=64)
        row.update({'replay_arm': 'cmcdr_budget', 'replay_alpha': 0.5,
                    'cmcdr_examples': 2, 'cmcdr_subsets': 4,
                    'cmcdr_profile_loss': 0.1})
        verify_audit_record(row, 2)
        row['cmcdr_examples'] = 65
        with self.assertRaises(ValueError):
            verify_audit_record(row, 2)

    def test_attenuation_weight_fallback_and_smoothing(self):
        self.assertEqual(attenuation_weights([0, 0, 0, 0]),
                         [0.25] * 4)
        np.testing.assert_allclose(attenuation_weights([3, 1, 0, 0]),
                                   [0.65, 0.25, 0.05, 0.05])
        with self.assertRaises(ValueError):
            attenuation_weights([1, -1])
        record = replay_step_audit(
            attenuation_weights([3, 1, 0, 0]), 2, 42, 1, 0, 0, 4,
            attenuation=[3, 1, 0, 0],
        )
        verify_audit_record(record, 2)
        record['attenuation'][0] = 0
        with self.assertRaises(ValueError):
            verify_audit_record(record, 2)

    def test_task_counts_recovered_from_first_audited_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'flow.jsonl'
            path.write_text('\n'.join(json.dumps(x) for x in [
                {'loader_key': "('train', (0, 1))", 'loader_iteration': 0,
                 'indices': [1, 2]},
                {'loader_key': "('train', (0, 1))", 'loader_iteration': 0,
                 'indices': [3]},
                {'loader_key': "('train', (0, 1))", 'loader_iteration': 1,
                 'indices': [1, 2]},
                {'loader_key': "('train', (2, 3))", 'loader_iteration': 0,
                 'indices': [4, 5]},
            ]), encoding='utf-8')
            rows = rows_from_audit(path, [
                {'new_classes': [0, 1]}, {'new_classes': [2, 3]},
            ])
            self.assertEqual(rows, [{'train': 3, 'classes': 2},
                                    {'train': 2, 'classes': 2}])

    def test_replay_invariants_check_only_replay_effects(self):
        selected = torch.nn.Linear(2, 2)
        unselected = torch.nn.Sequential(torch.nn.Linear(2, 2),
                                         torch.nn.BatchNorm1d(2))
        x = torch.ones(3, 2)
        def replay():
            return selected(x).square().mean()
        verify_replay_invariants(replay, [selected, unselected], [0])
        with self.assertRaises(AssertionError):
            verify_replay_invariants(lambda: unselected(x).square().mean(),
                                     [selected, unselected], [0])
        with self.assertRaises(AssertionError):
            verify_replay_invariants(lambda: selected(x).square().mean() +
                                     torch.rand(1).sum(),
                                     [selected, unselected], [0])
        with self.assertRaisesRegex(AssertionError, 'gradient'):
            verify_replay_invariants(
                lambda: (selected(x) +
                         torch.nn.functional.linear(
                             x, unselected[0].weight, unselected[0].bias)
                         ).square().mean(),
                [selected, unselected], [0],
            )
        def mutate_bn_without_graph():
            with torch.no_grad():
                unselected[1](x)
            return selected(x).square().mean()
        with self.assertRaisesRegex(AssertionError, 'BN state'):
            verify_replay_invariants(mutate_bn_without_graph,
                                     [selected, unselected], [0])
        checked = verify_replay_invariants(
            replay, [selected, unselected], [0], retain_graph=True)
        checked['loss'].backward()
        self.assertIsNotNone(selected.weight.grad)


if __name__ == '__main__':
    unittest.main()
