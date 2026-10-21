import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from cl_methods.proto_evolve import ProtoEvolveCL
from test_proto_evolve_adaptive_head import TinyTrainer


def sample_args(output_dir):
    return SimpleNamespace(
        cl_method='proto_evolve', num_parties=2, num_classes=2,
        num_tasks=2, seed=42, device='cpu', batch_size=2,
        data='synthvfl', party_col_ranges=[(0, 2), (2, 4)],
        aggregation='sum', output_dir=output_dir,
        head_consolidation_enabled=1, head_consolidation_schedule='final',
        head_consolidation_mode='adaptive_dual_branch',
        head_consolidation_samples_per_class=20,
        unlearn_after_tasks=[], unlearn_classes=[],
        sample_replay_only=1, proto_replay_gradient_enabled=0,
        proto_sdc=False, feat_distill_weight=0.0, distill_weight=0.25,
    )


class TrainReplayMethodTests(unittest.TestCase):
    def test_task_zero_has_no_replay_effect(self):
        with tempfile.TemporaryDirectory() as root:
            methods = []
            for enabled in (0, 1):
                torch.manual_seed(15)
                args = sample_args(root)
                args.train_replay_enabled = enabled
                method = ProtoEvolveCL(TinyTrainer(), args)
                method.before_task(0, [0], [0])
                loss = method._proto_loss(
                    method.trainer.bottoms, method.trainer.top_model,
                    torch.zeros(2, 4), torch.zeros(2, dtype=torch.long),
                    torch.tensor(0.25))
                self.assertEqual(float(loss), 0.25)
                with patch.object(method, '_compute_fim'), \
                     patch.object(method, '_compute_protos', return_value=(
                         {0: {'mean': torch.zeros(2), 'std': torch.ones(2)}},
                         {0: torch.zeros(20, 4)})), \
                     patch.object(method, '_consolidate_head'):
                    method.after_task(None, 0)
                methods.append(method)
            self.assertTrue(torch.equal(methods[0].head_raw_replay[0],
                                        methods[1].head_raw_replay[0]))
            self.assertFalse(hasattr(methods[1], '_train_replay_sampler'))
            self.assertEqual(methods[1].train_replay_history[0]['steps'], 0)

    def test_default_off_keeps_frozen_state_shape(self):
        with tempfile.TemporaryDirectory() as root:
            method = ProtoEvolveCL(TinyTrainer(), sample_args(root))
            self.assertNotIn('train_replay_audit', method.get_state())
            self.assertFalse(hasattr(method, '_train_replay_sampler'))

    def test_enabled_replays_old_ce_on_full_head_only(self):
        with tempfile.TemporaryDirectory() as root:
            args = sample_args(root)
            args.train_replay_enabled = 1
            trainer = TinyTrainer()
            method = ProtoEvolveCL(trainer, args)
            method.head_raw_replay = {0: torch.zeros(20, 4)}
            method.before_task(1, [1], [0, 1])
            current = torch.ones(2, 4)
            labels = torch.ones(2, dtype=torch.long)
            base_ce = torch.tensor(0.25, requires_grad=True)
            replay_logits = trainer.top_model(torch.zeros(64, 2))
            expected = base_ce + F.cross_entropy(
                replay_logits, torch.zeros(64, dtype=torch.long))
            actual = method._proto_loss(
                trainer.bottoms, trainer.top_model, current, labels, base_ce)
            self.assertTrue(torch.allclose(actual, expected))
            self.assertEqual(method.train_replay_audit['steps'], 1)
            self.assertEqual(method.train_replay_audit['samples'], 64)
            self.assertEqual(method.train_replay_audit['upload_bytes'], 64 * 2 * 2 * 4)
            self.assertEqual(method.train_replay_audit['return_bytes'], 64 * 2 * 2 * 4)


if __name__ == '__main__':
    unittest.main()
