"""Algebraic party-slice replay check; no distributed privacy claim."""

from copy import deepcopy
import unittest

import torch
from torch import nn


class PartySliceEquivalenceTests(unittest.TestCase):
    def test_aligned_party_slices_match_concat_forward_and_gradients(self):
        torch.manual_seed(29)
        raw = torch.randn(6, 4)
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        ids = [11, 23, 37, 41, 53, 67]
        bottoms = [nn.Linear(2, 3), nn.Linear(2, 3)]
        top = nn.Linear(6, 2)
        local_bottoms = deepcopy(bottoms)
        local_top = deepcopy(top)

        central_logits = top(torch.cat([
            bottoms[0](raw[:, :2]), bottoms[1](raw[:, 2:])], dim=1))
        nn.functional.cross_entropy(central_logits, labels).backward()

        party0 = {key: value for key, value in zip(ids, raw[:, :2])}
        party1 = {key: value for key, value in zip(ids, raw[:, 2:])}
        aligned0 = torch.stack([party0[key] for key in ids])
        aligned1 = torch.stack([party1[key] for key in ids])
        local_logits = local_top(torch.cat([
            local_bottoms[0](aligned0), local_bottoms[1](aligned1)], dim=1))
        nn.functional.cross_entropy(local_logits, labels).backward()

        torch.testing.assert_close(central_logits, local_logits)
        for central, local in zip([*bottoms, top], [*local_bottoms, local_top]):
            for left, right in zip(central.parameters(), local.parameters()):
                torch.testing.assert_close(left.grad, right.grad)


if __name__ == '__main__':
    unittest.main()
