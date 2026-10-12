import copy
import random
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from cl_methods.mir import MIRCL
from models import TopModel


class MIRSelectionTest(unittest.TestCase):
    def test_replay_uses_highest_interference_subset(self):
        bottom = torch.nn.Linear(1, 1)
        top = TopModel(1, 4)
        trainer = SimpleNamespace(
            bottoms=[bottom], top_model=top,
            _aggregate=lambda values: values[0],
            bytes_transmitted=0, comm_rounds=0,
        )
        args = SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 1)],
            device='cpu', lr=0.1, batch_size=2,
            er_per_class=1, er_batch=2, mir_candidates=3,
        )
        method = MIRCL(trainer, args)
        method.buffer.data = {
            class_id: [torch.tensor([float(class_id + 1)])]
            for class_id in range(4)
        }
        entries = [(example, label)
                   for label, values in method.buffer.data.items()
                   for example in values]
        random.seed(13)
        expected_pool = random.sample(entries, 3)
        random.seed(13)
        method._interference_scores = lambda *_: torch.tensor([0.0, 1.0, 2.0])
        forwarded = []
        hook = bottom.register_forward_pre_hook(
            lambda _model, inputs: forwarded.append(inputs[0].detach().clone()))
        try:
            method._replay_loss(
                [bottom], top, torch.tensor([[0.0]]), torch.tensor([0]),
                torch.tensor(0.0, requires_grad=True))
        finally:
            hook.remove()
        expected = torch.stack([expected_pool[2][0], expected_pool[1][0]])
        torch.testing.assert_close(forwarded[-1], expected)

    def test_candidate_pool_must_exceed_replay_count(self):
        args = SimpleNamespace(batch_size=64, er_per_class=20,
                               er_batch=64, mir_candidates=50)
        with self.assertRaises(ValueError):
            MIRCL(SimpleNamespace(), args)
        args.mir_candidates = 128
        method = MIRCL(SimpleNamespace(), args)
        self.assertGreater(method.candidates, method.buffer_batch_size)

    def test_scores_equal_loss_increase_after_virtual_update(self):
        bottom = torch.nn.Linear(1, 1, bias=False)
        top = torch.nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            bottom.weight.fill_(1.0)
            top.weight.copy_(torch.tensor([[1.0], [-1.0]]))
        trainer = SimpleNamespace(bottoms=[bottom], top_model=top,
                                  _aggregate=lambda parts: parts[0],
                                  bytes_transmitted=0, comm_rounds=0)
        args = SimpleNamespace(data='tabvfl', party_col_ranges=[(0, 1)],
                               device='cpu', lr=0.2, batch_size=2,
                               bottom_lr_scale=0.25,
                               er_per_class=2, er_batch=1)
        method = MIRCL(trainer, args)
        incoming_x = torch.tensor([[1.0]])
        incoming_y = torch.tensor([0])
        candidate_x = torch.tensor([[1.0], [-1.0], [2.0]])
        candidate_y = torch.tensor([1, 0, 1])

        expected_bottom, expected_top = copy.deepcopy(bottom), copy.deepcopy(top)
        incoming_loss = F.cross_entropy(
            expected_top(expected_bottom(incoming_x)), incoming_y)
        parameters = list(expected_bottom.parameters()) + list(expected_top.parameters())
        gradients = torch.autograd.grad(incoming_loss, parameters)
        before = F.cross_entropy(top(bottom(candidate_x)), candidate_y,
                                 reduction='none').detach()
        with torch.no_grad():
            for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
                parameter.sub_((0.05 if index == 0 else 0.2) * gradient)
            after = F.cross_entropy(
                expected_top(expected_bottom(candidate_x)), candidate_y,
                reduction='none')

        scores = method._interference_scores(
            incoming_x, incoming_y, candidate_x, candidate_y)
        torch.testing.assert_close(scores, after - before)
        self.assertEqual(scores.argmax().item(), (after - before).argmax().item())


if __name__ == '__main__':
    unittest.main()
