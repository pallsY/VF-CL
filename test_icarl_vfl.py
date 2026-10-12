import unittest
import copy
from types import SimpleNamespace

import torch

from cl_methods.icarl import ICaRLCL, herding_indices
from models import TopModel


class ICaRLHerdingTest(unittest.TestCase):
    def test_nearest_mean_readout_does_not_update_batchnorm(self):
        batchnorm = torch.nn.BatchNorm1d(2)
        bottom = torch.nn.Sequential(torch.nn.Linear(2, 2), batchnorm)
        top = TopModel(2, 2)
        trainer = SimpleNamespace(
            bottoms=[bottom], top_model=top,
            _aggregate=lambda parts: parts[0],
            bytes_transmitted=0, comm_rounds=0,
        )
        args = SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 2)], device='cpu',
            er_per_class=2, er_batch=2, batch_size=2,
        )
        method = ICaRLCL(trainer, args)
        method.buffer.data = {
            0: [torch.tensor([1.0, 0.0]), torch.tensor([1.1, 0.1])],
            1: [torch.tensor([0.0, 1.0]), torch.tensor([0.1, 1.1])],
        }
        before = batchnorm.num_batches_tracked.clone()
        method._evaluate(
            {0: [0, 1]},
            lambda _task, _classes: [(
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                torch.tensor([0, 1]))],
        )
        torch.testing.assert_close(batchnorm.num_batches_tracked, before)
        self.assertTrue(bottom.training)
        self.assertTrue(top.training)

    def test_selects_unique_examples_toward_class_mean(self):
        features = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.6],
        ])
        selected = herding_indices(features, 2)
        self.assertEqual(selected, [2, 1])
        self.assertEqual(len(set(selected)), 2)

    def test_checkpoint_restores_exemplars_and_teacher(self):
        bottom = torch.nn.Linear(1, 2)
        top = TopModel(2, 2)
        trainer = SimpleNamespace(bottoms=[bottom], top_model=top)
        args = SimpleNamespace(device='cpu', er_per_class=2,
                               er_batch=2, batch_size=2)
        first = ICaRLCL(trainer, args)
        first.buffer.data = {0: [torch.tensor([1.0])]}
        first.buffer.seen_count = {0: 1}
        first.teacher_bottoms = copy.deepcopy(trainer.bottoms)
        first.teacher_top = copy.deepcopy(trainer.top_model)
        first.new_classes = [0]
        first.old_classes = []

        second = ICaRLCL(trainer, args)
        second.load_state(first.get_state())
        torch.testing.assert_close(second.buffer.data[0][0],
                                   first.buffer.data[0][0])
        for expected, actual in zip(first.teacher_top.parameters(),
                                    second.teacher_top.parameters()):
            torch.testing.assert_close(expected, actual)


if __name__ == '__main__':
    unittest.main()
