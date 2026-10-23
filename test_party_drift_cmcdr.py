import copy
import unittest

import torch

import party_drift_replay


class CMCDRTests(unittest.TestCase):
    def test_mobius_two_party_coordinates_match_hand_calculation(self):
        self.assertTrue(hasattr(party_drift_replay, 'mobius_profile'))
        benefits = torch.tensor([[1.0, 4.0, 7.0, 13.0]])
        profile = party_drift_replay.mobius_profile(benefits)
        self.assertTrue(torch.equal(profile, torch.tensor([[3.0, 6.0, 3.0]])))

    def test_profile_loss_uses_all_subsets_and_freezes_teacher(self):
        self.assertTrue(hasattr(party_drift_replay, 'cmcdr_profile_loss'))
        torch.manual_seed(17)
        bottoms = [torch.nn.Linear(1, 1, bias=False) for _ in range(2)]
        old_bottoms = copy.deepcopy(bottoms)
        top = torch.nn.Linear(1, 2)
        old_top = copy.deepcopy(top)
        parts = [torch.tensor([[1.0], [-1.0]]) for _ in range(2)]
        current = [bottom(part) for bottom, part in zip(bottoms, parts)]
        teacher = [bottom(part) for bottom, part in zip(old_bottoms, parts)]
        current_calls = []
        old_calls = []
        h1 = top.register_forward_hook(lambda *_: current_calls.append(1))
        h2 = old_top.register_forward_hook(lambda *_: old_calls.append(1))
        try:
            zero = party_drift_replay.cmcdr_profile_loss(
                current, teacher, top, old_top,
                torch.tensor([0, 1]), [0, 1], aggregation='sum',
            )
        finally:
            h1.remove()
            h2.remove()
        self.assertEqual(len(current_calls), 4)
        self.assertEqual(len(old_calls), 4)
        self.assertEqual(float(zero.detach()), 0.0)
        current[0] = current[0] * 0.5
        loss = party_drift_replay.cmcdr_profile_loss(
            current, teacher, top, old_top,
            torch.tensor([0, 1]), [0, 1], aggregation='sum',
        )
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(bottoms[0].weight.grad)
        self.assertIsNone(old_bottoms[0].weight.grad)

    def test_budget_positions_match_intervention_count_and_are_repeatable(self):
        self.assertTrue(hasattr(party_drift_replay, 'cmcdr_positions'))
        first = party_drift_replay.cmcdr_positions(
            seed=70, task=1, replay_steps=10, batch_size=64,
            old_classes=10, probe_events=2, parties=4, probe_per_class=2,
        )
        second = party_drift_replay.cmcdr_positions(
            seed=70, task=1, replay_steps=10, batch_size=64,
            old_classes=10, probe_events=2, parties=4, probe_per_class=2,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertTrue(all(0 <= step < 10 and 0 <= row < 64
                            for step, row in first))

    def test_replay_loss_adds_profile_only_for_selected_budget_rows(self):
        torch.manual_seed(37)
        bottoms = [torch.nn.Linear(1, 2) for _ in range(2)]
        old_bottoms = copy.deepcopy(bottoms)
        top = torch.nn.Linear(2, 4)
        old_top = copy.deepcopy(top)
        parts = [torch.tensor([[1.0], [-1.0], [2.0]]) for _ in range(2)]
        labels = torch.tensor([0, 1, 0])
        kwargs = dict(selected=[0, 1], old_classes=[0, 1], alpha=0.5,
                      seed=123, temperature=2, kd_weight=0.25)
        base, _ = party_drift_replay.masked_replay_loss(
            bottoms, top, old_bottoms, old_top, parts, labels, **kwargs)
        self.assertGreater(float(base.detach()), 0)
        with_profile, components = party_drift_replay.masked_replay_loss(
            bottoms, top, old_bottoms, old_top, parts, labels,
            cmcdr_indices=[0, 2], cmcdr_weight=1.0, **kwargs)
        self.assertIn('cmcdr', components)
        self.assertEqual(components['cmcdr_examples'], 2)
        self.assertAlmostEqual(float((with_profile - base).detach()),
                               float(components['cmcdr']), places=6)
        no_examples, components = party_drift_replay.masked_replay_loss(
            bottoms, top, old_bottoms, old_top, parts, labels,
            cmcdr_indices=[], cmcdr_weight=1.0, **kwargs)
        self.assertEqual(components['cmcdr_examples'], 0)
        self.assertTrue(torch.equal(no_examples, base))


if __name__ == '__main__':
    unittest.main()
