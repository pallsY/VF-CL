import unittest
from types import SimpleNamespace

import numpy as np
import torch

from cl_methods.proto_evolve import herding_indices
from data_utils import TensorViewDataset
from launch_isolet_replay_selection_pilot import (
    PilotDataset, build_holdout_manifest, derive_config, hybrid_indices,
    reject_test_access,
)


class ReplayPilotTests(unittest.TestCase):
    def test_hybrid_keeps_herding_prefix_and_adds_deterministic_coverage(self):
        features = torch.tensor([
            [1.0, 0.0], [0.98, 0.2], [-1.0, 0.0],
            [0.0, 1.0], [0.0, -1.0], [-0.98, 0.2],
        ])
        indices = hybrid_indices(features, 4)
        self.assertEqual(len(indices), 4)
        self.assertEqual(len(set(indices.tolist())), 4)
        self.assertEqual(indices[:2].tolist(), herding_indices(features, 2).tolist())
        self.assertEqual(indices.tolist(), hybrid_indices(features, 4).tolist())
        normalized = torch.nn.functional.normalize(features, dim=1)
        selected = indices[:2]
        distances = 1.0 - normalized @ normalized[selected].T
        nearest = distances.min(dim=1).values
        nearest[selected] = -float('inf')
        self.assertEqual(int(nearest.argmax()), int(indices[2]))

    def test_holdout_is_disjoint_and_removed_only_from_training(self):
        labels = np.array([0] * 12 + [1] * 12)
        gate = {0, 1, 12, 13}
        manifest, holdout = build_holdout_manifest(labels, gate, 2, 37)
        self.assertEqual(len(holdout), 4)
        self.assertFalse(gate & holdout)
        self.assertEqual(manifest['per_class'], 2)
        fresh, fresh_indices = build_holdout_manifest(labels, gate, 2, 20261011)
        self.assertNotEqual(manifest['sha256'], fresh['sha256'])
        self.assertFalse(gate & fresh_indices)
        newer, newer_indices = build_holdout_manifest(labels, gate, 2, 20261012)
        self.assertNotEqual(fresh['sha256'], newer['sha256'])
        self.assertFalse(gate & newer_indices)
        dataset = PilotDataset.__new__(PilotDataset)
        dataset.args = SimpleNamespace(
            deterministic=0, batch_size=4, num_workers=0,
            bic_enabled=0, lambda_validation_enabled=1,
            formal_deferred_evaluation=False, data_flow_audit=0,
        )
        dataset.trainset = TensorViewDataset(np.arange(24)[:, None], labels)
        dataset.validation_indices = gate
        dataset.holdout_indices = holdout
        train = dataset.get_train_loader([0, 1], shuffle=False)
        self.assertEqual(set(train.dataset.indices), set(range(24)) - gate - holdout)
        evaluation = dataset.holdout_loader([0, 1])
        self.assertEqual(set(evaluation.dataset.indices), holdout)

    def test_config_changes_only_approved_pilot_fields(self):
        source = {
            'data': 'tabvfl', 'num_classes': 26, 'num_tasks': 13,
            'classes_per_task': 2, 'num_parties': 4,
            'seed': 42, 'formal_deferred_evaluation': True,
            'head_consolidation_enabled': 1,
            'head_consolidation_mode': 'adaptive_dual_branch',
            'head_consolidation_samples_per_class': 20,
            'head_consolidation_schedule': 'final',
            'lambda_validation_enabled': 1,
            'lambda_validation_per_class': 40,
            'lambda_validation_split_seed': 20260809,
            'epochs_per_task': 50, 'batch_size': 128,
            'optimizer': 'adamw',
            'proto_lambda_a': 0.15, 'distill_weight': 0.25,
            'feat_distill_weight': 0.05,
            'data_path': '/home/chase/Yangxx/VF-CL/data',
            'vector_npz': '/home/chase/Yangxx/VF-CL/data/isolet/isolet_vfl.npz',
            'results_dir': '/old', 'output_dir': '/old/run',
            'exp_name': 'old',
        }
        config, changed = derive_config(source, '/tmp/pilot', 47, 'hybrid')
        self.assertEqual(config['proto_lambda_a'], 0.05)
        self.assertEqual(config['distill_weight'], 0.10)
        self.assertEqual(config['feat_distill_weight'], 0.02)
        self.assertEqual(config['seed'], 47)
        self.assertEqual(config['head_consolidation_samples_per_class'], 20)
        self.assertEqual(config['lambda_validation_split_seed'], 20260809)
        self.assertEqual(set(changed), {
            'seed', 'formal_deferred_evaluation', 'data_path', 'vector_npz',
            'results_dir', 'output_dir', 'exp_name', 'proto_lambda_a',
            'distill_weight', 'feat_distill_weight',
        })
        newer, _ = derive_config(source, '/tmp/pilot-new', 49, 'herding')
        self.assertEqual(newer['seed'], 49)
        expanded, expanded_changes = derive_config(source, '/tmp/pilot-capacity', 51, 'herding', 40)
        self.assertEqual(expanded['seed'], 51)
        self.assertEqual(expanded['head_consolidation_samples_per_class'], 40)
        self.assertIn('head_consolidation_samples_per_class', expanded_changes)
        with self.assertRaises(ValueError):
            derive_config(source, '/tmp/pilot-capacity', 51, 'herding', 30)
        with self.assertRaises(ValueError):
            derive_config(source, '/tmp/pilot-capacity', 51, 'hybrid', 40)
        with self.assertRaises(ValueError):
            derive_config(source, '/tmp/pilot-capacity', 53, 'herding', 40)
        with self.assertRaises(ValueError):
            derive_config(source, '/tmp/pilot', 42, 'hybrid')

        with self.assertRaises(ValueError):
            derive_config({**source, 'head_consolidation_samples_per_class': 40},
                          '/tmp/pilot', 47, 'hybrid')

    def test_rejects_test_loader_access_records(self):
        reject_test_access([{'split': 'train'}, {'split': 'validation'}])
        with self.assertRaises(ValueError):
            reject_test_access([{'split': 'test'}])
        with self.assertRaises(ValueError):
            reject_test_access([{'loader_key': "('test', (0, 1))"}])
if __name__ == '__main__':
    unittest.main()
