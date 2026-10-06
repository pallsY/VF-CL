"""Regression tests for opt-in split-trainer gradient clipping."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from models import TopModel
from vfl_trainer import VFLTrainer


class GradientClippingTests(unittest.TestCase):
    @staticmethod
    def _trainer():
        args = SimpleNamespace(
            data='synthvfl', party_col_ranges=[(0, 1)], device='cpu',
            aggregation='sum', lr=1.0, bottom_lr_scale=1.0,
            optimizer='sgd', momentum=0.0, weight_decay=0.0,
            data_flow_audit=0, own_concentrate_weight=0.0,
        )
        bottom = nn.Linear(1, 1, bias=False)
        top = TopModel(1, 2)
        return VFLTrainer([bottom], top, args)

    @staticmethod
    def _loader():
        return [(
            torch.tensor([[1000.0], [-1000.0]]),
            torch.tensor([0, 1]),
        )]

    @staticmethod
    def _parameters(trainer):
        return tuple(trainer.bottoms[0].parameters()) \
            + tuple(trainer.top_model.parameters())

    def test_joint_clip_bounds_real_top_and_bottom_update(self):
        torch.manual_seed(0)
        trainer = self._trainer()
        before = [parameter.detach().clone()
                  for parameter in self._parameters(trainer)]

        def extra(bottoms, _top, _batch_x, _batch_y, loss_ce):
            return loss_ce + 1e6 * sum(
                parameter.square().sum()
                for parameter in bottoms[0].parameters()
            )

        trainer.train_task(
            self._loader(), 1, extra_loss_fn=extra, grad_clip_norm=1.0,
        )
        after = [parameter.detach()
                 for parameter in self._parameters(trainer)]
        delta = torch.cat([
            (right - left).flatten()
            for left, right in zip(before, after)
        ]).norm()
        self.assertLessEqual(float(delta), 1.00001)
        self.assertGreater(float(delta), 0.0)

    def test_nonfinite_clipped_loss_rejects_without_mutation(self):
        torch.manual_seed(1)
        trainer = self._trainer()
        before = [parameter.detach().clone()
                  for parameter in self._parameters(trainer)]

        def extra(_bottoms, _top, _batch_x, _batch_y, loss_ce):
            return loss_ce * torch.tensor(float('nan'))

        with self.assertRaisesRegex(FloatingPointError, 'non-finite'):
            trainer.train_task(
                self._loader(), 1, extra_loss_fn=extra,
                grad_clip_norm=1.0,
            )
        self.assertTrue(all(
            torch.equal(left, right)
            for left, right in zip(before, self._parameters(trainer))
        ))

    def test_invalid_gradient_clip_norms_reject(self):
        for value in (True, 0, -1.0, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, 'positive finite'):
                self._trainer().train_task(
                    self._loader(), 1, grad_clip_norm=value,
                )

    def test_omitted_and_explicit_none_paths_are_identical(self):
        torch.manual_seed(2)
        omitted = self._trainer()
        explicit = deepcopy(omitted)
        omitted.train_task(self._loader(), 1)
        explicit.train_task(self._loader(), 1, grad_clip_norm=None)
        self.assertTrue(all(
            torch.equal(left, right)
            for left, right in zip(
                self._parameters(omitted), self._parameters(explicit),
            )
        ))


if __name__ == '__main__':
    unittest.main()
