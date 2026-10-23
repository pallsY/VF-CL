import copy
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from cl_methods.proto_evolve import ProtoEvolveCL
from party_drift_replay import (
    BalancedOldReplay,
    contribution_attenuation,
    masked_replay_loss,
)
from test_proto_evolve_adaptive_head import TinyTrainer


class PartyDriftReplayTests(unittest.TestCase):
    def test_method_switch_keeps_frozen_path_and_adds_only_replay_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            losses = []
            for enabled in (0, 1):
                torch.manual_seed(41)
                args = SimpleNamespace(
                    cl_method='proto_evolve', num_parties=2, num_classes=4,
                    num_tasks=2, seed=73, device='cpu', batch_size=2,
                    embed_dim=2,
                    data='synthvfl', party_col_ranges=[(0, 2), (2, 4)],
                    aggregation='sum', output_dir=directory,
                    head_consolidation_enabled=1,
                    head_consolidation_schedule='final',
                    head_consolidation_mode='adaptive_dual_branch',
                    head_consolidation_samples_per_class=20,
                    unlearn_after_tasks=[], unlearn_classes=[],
                    sample_replay_only=1, proto_replay_gradient_enabled=0,
                    proto_sdc=False, feat_distill_weight=0.0,
                    distill_weight=0.25, lwf_temperature=2.0,
                    party_drift_enabled=enabled,
                    party_drift_preflight=enabled,
                )
                method = ProtoEvolveCL(TinyTrainer(), args)
                method.before_task(0, [0, 1], [0, 1])
                current = torch.ones(2, 4)
                labels = torch.tensor([0, 1])
                no_old = method._proto_loss(
                    method.trainer.bottoms, method.trainer.top_model,
                    current, labels, torch.tensor(0.25))
                self.assertEqual(float(no_old), 0.25)
                method.head_raw_replay = {
                    0: torch.ones(20, 4), 1: -torch.ones(20, 4),
                }
                method._old_bottoms = copy.deepcopy(method.trainer.bottoms)
                method._old_top = copy.deepcopy(method.trainer.top_model)
                method.before_task(1, [2, 3], [0, 1, 2, 3])
                if enabled:
                    before_rng = torch.get_rng_state().clone()
                    method._update_party_drift_probe(0)
                    self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
                    self.assertAlmostEqual(sum(method._party_drift_weights), 1.0)
                loss = method._proto_loss(
                    method.trainer.bottoms, method.trainer.top_model,
                    current, torch.tensor([2, 3]),
                    torch.tensor(0.25, requires_grad=True),
                )
                losses.append(float(loss))
                if not enabled:
                    self.assertNotIn('party_drift_audit', method.get_state())
                else:
                    self.assertEqual(method.party_drift_step, 1)
                    self.assertEqual(len(method.party_drift_step_records), 1)
            self.assertGreater(losses[1], losses[0])

    def test_old_class_sampler_is_balanced_and_uses_only_numpy_rng(self):
        data = {0: torch.arange(20).float().unsqueeze(1),
                1: torch.arange(20, 40).float().unsqueeze(1)}
        before = torch.get_rng_state().clone()
        cuda_before = (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else [])
        first = BalancedOldReplay(data, seed=73, batch_size=64)
        second = BalancedOldReplay(data, seed=73, batch_size=64)
        x, y, ids, duplicates = first.sample()
        x_again, y_again, ids_again, duplicates_again = second.sample()
        self.assertTrue(torch.equal(x, x_again))
        self.assertTrue(torch.equal(y, y_again))
        self.assertEqual(ids, ids_again)
        self.assertEqual(duplicates, duplicates_again)
        self.assertEqual(torch.bincount(y).tolist(), [32, 32])
        self.assertGreater(duplicates, 0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        if cuda_before:
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(
                cuda_before, torch.cuda.get_rng_state_all())))

    def test_masked_loss_skips_unselected_bottoms_and_preserves_rng(self):
        torch.manual_seed(9)
        bottoms = [torch.nn.Linear(2, 3) for _ in range(4)]
        old_bottoms = copy.deepcopy(bottoms)
        top = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(3, 4))
        old_top = copy.deepcopy(top).eval()
        for bottom in old_bottoms:
            bottom.eval()
        parts = [torch.randn(5, 2) for _ in range(4)]
        labels = torch.tensor([0, 1, 0, 1, 0])
        calls = [0, 0, 0, 0]
        hooks = [bottom.register_forward_hook(
            lambda _m, _i, _o, k=k: calls.__setitem__(k, calls[k] + 1)
        ) for k, bottom in enumerate(bottoms)]
        before = torch.get_rng_state().clone()
        cuda_before = (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else [])
        try:
            loss, components = masked_replay_loss(
                bottoms, top, old_bottoms, old_top, parts, labels,
                selected=[0, 2], old_classes=[0, 1], alpha=1.0, seed=124,
            )
            loss.backward()
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(calls, [1, 0, 1, 0])
        self.assertIsNone(bottoms[1].weight.grad)
        self.assertIsNone(bottoms[3].weight.grad)
        self.assertIsNotNone(bottoms[0].weight.grad)
        self.assertIsNotNone(bottoms[2].weight.grad)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        if cuda_before:
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(
                cuda_before, torch.cuda.get_rng_state_all())))
        self.assertEqual(set(components), {'ce', 'logit_kd'})

    def test_probe_detects_positive_party_contribution_decay(self):
        old_bottoms = [torch.nn.Linear(1, 1, bias=False) for _ in range(2)]
        for bottom in old_bottoms:
            bottom.weight.data.fill_(1)
        bottoms = copy.deepcopy(old_bottoms)
        bottoms[1].weight.data.zero_()
        old_top = torch.nn.Linear(1, 2, bias=False)
        old_top.weight.data[:] = torch.tensor([[1.0], [-1.0]])
        top = copy.deepcopy(old_top)
        parts = [torch.tensor([[1.0], [-1.0]]) for _ in range(2)]
        labels = torch.tensor([0, 1])
        decay = contribution_attenuation(
            bottoms, top, old_bottoms, old_top,
            parts, labels, old_classes=[0, 1],
        )
        np.testing.assert_allclose(decay[0], 0.0, atol=1e-6)
        self.assertGreater(decay[1], 0.0)

    def test_old_class_kd_does_not_update_new_class_rows(self):
        bottoms = [torch.nn.Linear(2, 2)]
        old_bottoms = copy.deepcopy(bottoms)
        top = torch.nn.Linear(2, 4)
        old_top = copy.deepcopy(top)
        loss, _ = masked_replay_loss(
            bottoms, top, old_bottoms, old_top,
            [torch.ones(3, 2)], torch.tensor([0, 1, 0]),
            selected=[0], old_classes=[0, 1], alpha=0.0, seed=12,
        )
        loss.backward()
        self.assertTrue(torch.count_nonzero(top.weight.grad[2:]) == 0)
        self.assertTrue(torch.count_nonzero(top.bias.grad[2:]) == 0)

    def test_concat_aggregation_keeps_zero_slots_for_unselected_parties(self):
        bottoms = [torch.nn.Linear(2, 3) for _ in range(4)]
        old_bottoms = copy.deepcopy(bottoms)
        top = torch.nn.Linear(12, 4)
        old_top = copy.deepcopy(top)
        parts = [torch.ones(5, 2) for _ in range(4)]
        loss, _ = masked_replay_loss(
            bottoms, top, old_bottoms, old_top, parts,
            torch.tensor([0, 1, 0, 1, 0]), selected=[0, 2],
            old_classes=[0, 1], alpha=1.0, seed=12,
            aggregation='concat',
        )
        loss.backward()
        self.assertIsNone(bottoms[1].weight.grad)
        self.assertIsNone(bottoms[3].weight.grad)
        self.assertIsNotNone(bottoms[0].weight.grad)

    def test_concat_probe_measures_same_party_contribution_decay(self):
        old_bottoms = [torch.nn.Linear(1, 1, bias=False) for _ in range(2)]
        for bottom in old_bottoms:
            bottom.weight.data.fill_(1)
        bottoms = copy.deepcopy(old_bottoms)
        bottoms[1].weight.data.zero_()
        old_top = torch.nn.Linear(2, 2, bias=False)
        old_top.weight.data[:] = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
        top = copy.deepcopy(old_top)
        parts = [torch.tensor([[1.0], [-1.0]]) for _ in range(2)]
        labels = torch.tensor([0, 1])
        decay = contribution_attenuation(
            bottoms, top, old_bottoms, old_top,
            parts, labels, old_classes=[0, 1], aggregation='concat',
        )
        np.testing.assert_allclose(decay[0], 0.0, atol=1e-6)
        self.assertGreater(decay[1], 0.0)


if __name__ == '__main__':
    unittest.main()
