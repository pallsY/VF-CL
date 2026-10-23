import unittest
import copy
import tempfile
from types import SimpleNamespace

import torch

from config import validate_sample_replay_only
import party_drift_local
from cl_methods.proto_evolve import ProtoEvolveCL
from test_proto_evolve_adaptive_head import TinyTrainer


class PartyDriftControlsTests(unittest.TestCase):
    @staticmethod
    def prepared_method(directory, arm, alpha=0.5):
        args = SimpleNamespace(
            cl_method='proto_evolve', num_parties=2, num_classes=4,
            num_tasks=2, seed=74, device='cpu', batch_size=2, embed_dim=2,
            data='synthvfl', party_col_ranges=[(0, 2), (2, 4)],
            aggregation='sum', output_dir=directory,
            head_consolidation_enabled=1, head_consolidation_schedule='final',
            head_consolidation_mode='adaptive_dual_branch',
            head_consolidation_samples_per_class=20,
            unlearn_after_tasks=[], unlearn_classes=[], sample_replay_only=1,
            proto_replay_gradient_enabled=0, proto_sdc=False,
            feat_distill_weight=0.0, distill_weight=0.25,
            lwf_temperature=2.0, party_drift_enabled=1,
            party_drift_preflight=0, party_replay_arm=arm,
            party_replay_alpha=alpha,
        )
        trainer = TinyTrainer()
        method = ProtoEvolveCL(trainer, args)
        method.before_task(0, [0, 1], [0, 1])
        method.head_raw_replay = {
            0: torch.ones(20, 4), 1: -torch.ones(20, 4),
        }
        method._old_bottoms = copy.deepcopy(trainer.bottoms)
        method._old_top = copy.deepcopy(trainer.top_model)
        method.before_task(1, [2, 3], [0, 1, 2, 3])
        return method

    def test_static_uses_frozen_old_class_weights_and_gamma(self):
        self.assertTrue(hasattr(party_drift_local, 'static_importance_weights'))
        values = party_drift_local.static_importance_weights(
            {0: [0.8, 0.1, 0.05, 0.05],
             1: [0.2, 0.3, 0.4, 0.1],
             2: [0.01, 0.01, 0.01, 0.97]},
            old_classes=[0, 1], gamma=0.2,
        )
        self.assertEqual(len(values), 4)
        for got, expected in zip(values, [0.45, 0.21, 0.23, 0.11]):
            self.assertAlmostEqual(got, expected)

    def test_static_rejects_missing_or_nonfinite_source(self):
        self.assertTrue(hasattr(party_drift_local, 'static_importance_weights'))
        with self.assertRaises(ValueError):
            party_drift_local.static_importance_weights({0: [0.5, 0.5]}, [0, 1])
        with self.assertRaises(ValueError):
            party_drift_local.static_importance_weights({0: [float('nan'), 0.5]}, [0])

    def test_replay_arm_and_alpha_validate_before_training(self):
        args = SimpleNamespace(
            party_drift_enabled=1, sample_replay_only=1,
            party_replay_arm='uniform', party_replay_alpha=0.5,
            cl_method='proto_evolve', head_consolidation_enabled=1,
            head_consolidation_mode='adaptive_dual_branch',
            head_consolidation_schedule='final',
            head_consolidation_samples_per_class=20,
            proto_replay_gradient_enabled=0, proto_sdc=False,
            feat_distill_weight=0.0,
        )
        validate_sample_replay_only(args)
        args.party_replay_arm = 'unlisted'
        with self.assertRaises(ValueError):
            validate_sample_replay_only(args)
        args.party_replay_arm = 'uniform'
        args.party_replay_alpha = -0.5
        with self.assertRaises(ValueError):
            validate_sample_replay_only(args)

    def test_uniform_and_static_refresh_without_new_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            uniform = self.prepared_method(directory, 'uniform')
            self.assertTrue(hasattr(uniform, '_update_party_replay_weights'))
            uniform._update_party_replay_weights(0)
            self.assertEqual(uniform._party_drift_weights, [0.5, 0.5])
            static = self.prepared_method(directory, 'static')
            static.class_party_weights = {0: [0.9, 0.1], 1: [0.7, 0.3]}
            static._update_party_replay_weights(0)
            for got, expected in zip(static._party_drift_weights, [0.74, 0.26]):
                self.assertAlmostEqual(got, expected)

    def test_all_party_replay_selects_every_party_and_records_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            method = self.prepared_method(directory, 'all_party', alpha=0.5)
            self.assertTrue(hasattr(method, 'party_replay_arm'))
            self.assertEqual(method.party_replay_arm, 'all_party')
            method._proto_loss(
                method.trainer.bottoms, method.trainer.top_model,
                torch.ones(2, 4), torch.tensor([2, 3]),
                torch.tensor(0.25, requires_grad=True),
            )
            self.assertEqual(method.party_drift_step_records[-1]['selected'], [0, 1])
            self.assertEqual(method.party_drift_step_records[-1]['replay_alpha'], 0.5)

    def test_cmcdr_arm_writes_full_and_budgeted_intervention_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            direct = self.prepared_method(directory, 'cmcdr')
            direct._proto_loss(
                direct.trainer.bottoms, direct.trainer.top_model,
                torch.ones(2, 4), torch.tensor([2, 3]),
                torch.tensor(0.25, requires_grad=True),
            )
            self.assertEqual(direct.party_drift_step_records[-1]['cmcdr_examples'], 64)
            self.assertEqual(direct.party_drift_step_records[-1]['cmcdr_subsets'], 4)
            budget = self.prepared_method(directory, 'cmcdr_budget')
            budget._cmcdr_budget_positions = {(0, 0), (0, 3)}
            budget._proto_loss(
                budget.trainer.bottoms, budget.trainer.top_model,
                torch.ones(2, 4), torch.tensor([2, 3]),
                torch.tensor(0.25, requires_grad=True),
            )
            self.assertEqual(budget.party_drift_step_records[-1]['cmcdr_examples'], 2)


if __name__ == '__main__':
    unittest.main()
