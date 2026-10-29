"""FIM policy changes only the next-task mask, not the Fisher pass."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from cl_methods.proto_evolve import ProtoEvolveCL


class FIMPolicyTests(unittest.TestCase):
    def _run(self, policy, prior=None):
        torch.manual_seed(19)
        bottoms = [nn.Linear(2, 2), nn.Linear(2, 2)]
        top = nn.Linear(4, 2)
        trainer = SimpleNamespace(
            bottoms=bottoms, top_model=top,
            _aggregate=lambda parts: torch.cat(parts, dim=1),
        )
        method = ProtoEvolveCL.__new__(ProtoEvolveCL)
        method.trainer = trainer
        method.args = SimpleNamespace(
            num_parties=2, device='cpu', data='tabvfl',
            party_col_ranges=[(0, 2), (2, 4)],
            fim_freeze_frac=0.0, fim_mask_policy=policy,
        )
        method.fim_masks = deepcopy(prior or [{}, {}])
        batch = [(torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]]),
                  torch.tensor([0, 1]))]
        method._compute_fim(batch, 0)
        return method, [deepcopy(b.state_dict()) for b in bottoms], top.weight.grad

    def test_all_policies_execute_same_fisher_pass_and_only_change_mask(self):
        baseline = self._run('cumulative')
        fresh = self._run('fresh')
        off = self._run('off')
        self.assertEqual(baseline[0].fim_masks, fresh[0].fim_masks)
        self.assertTrue(any(any(m.values()) for m in baseline[0].fim_masks))
        self.assertFalse(any(any(m.values()) for m in off[0].fim_masks))
        for run in (fresh, off):
            for expected, actual in zip(baseline[1], run[1]):
                for name in expected:
                    torch.testing.assert_close(expected[name], actual[name])
            torch.testing.assert_close(baseline[2], run[2])

    def test_fresh_replaces_old_mask_while_cumulative_retains_it(self):
        prior = [{'weight': True, 'bias': True} for _ in range(2)]
        cumulative = self._run('cumulative', prior)[0].fim_masks
        fresh = self._run('fresh', prior)[0].fim_masks
        off = self._run('off', prior)[0].fim_masks
        self.assertTrue(all(all(mask.values()) for mask in cumulative))
        self.assertTrue(all(sum(mask.values()) == 1 for mask in fresh))
        self.assertFalse(any(any(mask.values()) for mask in off))


if __name__ == '__main__':
    unittest.main()
