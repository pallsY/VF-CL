import unittest

import torch
import torch.nn.functional as F

from party_task_router_dev import fit_router, route


class PartyTaskRouterTests(unittest.TestCase):
    def test_preserves_within_task_class_ratios(self):
        base = F.log_softmax(torch.tensor([
            [2.0, 1.0, 0.0, -1.0],
            [0.2, -0.5, 1.0, 0.0],
            [1.0, 0.0, 0.5, -0.5],
            [0.2, 0.1, 0.3, 0.4],
        ], dtype=torch.float64), dim=1)
        parties = [
            torch.tensor([[2., 1., 0., 0.], [0., 0., 1., 0.],
                          [1., 0., 0., 0.], [0., 0., 0., 1.]],
                         dtype=torch.float64),
            torch.tensor([[1., 0., 0., 0.], [0., 0., 1., 1.],
                          [0., 1., 0., 0.], [0., 0., 1., 0.]],
                         dtype=torch.float64),
        ]
        tasks = {0: [0, 1], 1: [2, 3]}
        fitted = fit_router(base, parties, torch.tensor([0, 2, 1, 3]),
                            tasks)
        output = route(base, parties, tasks, fitted)
        self.assertTrue(torch.allclose(output.logsumexp(1),
                                       torch.zeros(4, dtype=torch.float64)))
        self.assertTrue(torch.allclose(output[:, 0] - output[:, 1],
                                       base[:, 0] - base[:, 1]))
        self.assertTrue(torch.allclose(output[:, 2] - output[:, 3],
                                       base[:, 2] - base[:, 3]))

    def test_rejects_missing_party_scores(self):
        base = F.log_softmax(torch.zeros(2, 2), dim=1)
        with self.assertRaisesRegex(ValueError, 'party'):
            fit_router(base, [], torch.tensor([0, 1]),
                       {0: [0], 1: [1]})


if __name__ == '__main__':
    unittest.main()
