import copy
import types
import unittest

import torch

from models import TopModel
from party_utility_screen import replay_scores


class ReplayUtilityTests(unittest.TestCase):
    def test_helpful_party_gets_weight_and_identical_models_have_zero_drift(self):
        args = types.SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 1), (1, 2)],
            num_parties=2, aggregation='concat', device='cpu',
        )
        old_top = TopModel(2, 2).eval()
        with torch.no_grad():
            old_top.classifier.weight.copy_(torch.tensor([[2., 0.], [0., 2.]]))
            old_top.classifier.bias.zero_()
        new_top = copy.deepcopy(old_top)
        with torch.no_grad():
            new_top.classifier.weight[0, 0] = 1.

        def trainer(top):
            return types.SimpleNamespace(
                bottoms=[torch.nn.Identity(), torch.nn.Identity()],
                top_model=top,
                _aggregate=lambda parts: torch.cat(parts, dim=1),
            )

        replay = {0: torch.ones(1, 2), 1: torch.ones(1, 2)}
        frozen = {0: [0.5, 0.5], 1: [0.5, 0.5]}
        changed = replay_scores(trainer(old_top), trainer(new_top),
                                replay, [0, 1], frozen, args)
        row = next(row for row in changed if row['class_id'] == 0)
        self.assertAlmostEqual(row['utility_weights'][0], 1.0)
        self.assertAlmostEqual(row['utility_weights'][1], 0.0)
        self.assertAlmostEqual(row['utility'], 1.0)
        self.assertAlmostEqual(row['uniform'], 0.5)

        identical = replay_scores(trainer(old_top), trainer(old_top),
                                  replay, [0, 1], frozen, args)
        self.assertTrue(all(row[key] == 0.0 for row in identical
                            for key in ('utility', 'uniform', 'frozen', 'shuffled')))


if __name__ == '__main__':
    unittest.main()
