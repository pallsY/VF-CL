import unittest
from types import SimpleNamespace

import torch

from vfl_trainer import VFLTrainer


class DevelopmentNonfiniteGuardTests(unittest.TestCase):
    def test_exploding_finite_epoch_loss_is_recorded_as_training_anomaly(self):
        args = SimpleNamespace(
            device='cpu', data='tabvfl', num_parties=1,
            party_col_ranges=[(0, 2)], aggregation='sum',
            lr=0.01, momentum=0.0, weight_decay=0.0,
            development_nonfinite_guard=1, data_flow_audit=0,
        )
        trainer = VFLTrainer([torch.nn.Linear(2, 2)],
                             torch.nn.Linear(2, 2), args)
        loader = [(torch.ones(2, 2), torch.tensor([0, 1]))]
        with self.assertRaises(FloatingPointError):
            trainer.train_epoch(loader, trainer._create_optimizers(),
                                lambda _b, _t, _x, _y, ce: ce * 1e9)

    def test_nan_training_epoch_is_a_terminal_training_error(self):
        args = SimpleNamespace(
            device='cpu', data='tabvfl', num_parties=1,
            party_col_ranges=[(0, 2)], aggregation='sum',
            lr=0.01, momentum=0.0, weight_decay=0.0,
            development_nonfinite_guard=1, data_flow_audit=0,
        )
        trainer = VFLTrainer([torch.nn.Linear(2, 2)],
                             torch.nn.Linear(2, 2), args)
        loader = [(torch.ones(2, 2), torch.tensor([0, 1]))]
        def nan_loss(_bottoms, _top, _x, _y, ce):
            return ce * torch.tensor(float('nan'))
        with self.assertRaises(FloatingPointError):
            trainer.train_epoch(loader, trainer._create_optimizers(), nan_loss)


if __name__ == '__main__':
    unittest.main()
