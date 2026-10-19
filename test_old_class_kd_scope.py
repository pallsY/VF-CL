import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cl_methods.proto_evolve import ProtoEvolveCL
from models import TopModel
from vfl_trainer import VFLTrainer


class OldClassKDScopeTests(unittest.TestCase):
    def _method(self):
        torch.manual_seed(7)
        method = ProtoEvolveCL.__new__(ProtoEvolveCL)
        top = TopModel(2, 5, cosine=False)
        with torch.no_grad():
            top.classifier.weight.copy_(torch.tensor([
                [-1., 0.], [1., 0.], [0., 1.], [0., 2.], [2., 1.],
            ]))
            top.classifier.bias.zero_()
        old_top = deepcopy(top)
        with torch.no_grad():
            old_top.classifier.weight.zero_()
        bottoms = [torch.nn.Identity(), torch.nn.Identity()]
        method.args = SimpleNamespace(
            device='cpu', num_parties=2, aggregation='concat', seed=7,
        )
        method.trainer = SimpleNamespace(
            bottoms=bottoms, top_model=top,
            _aggregate=lambda embs: torch.cat(embs, dim=1),
        )
        method._old_bottoms = [deepcopy(bottom) for bottom in bottoms]
        method._old_top = old_top
        method.global_protos = {}
        method.current_task_classes = []
        method.dep_tracking_enabled = False
        method.fim_masks = [{}, {}]
        method.temperature = 2.0
        method.effective_distill_weight = 1.0
        method.party_kd_lambda = 1.0
        method.party_kd_enabled = True
        method.party_kd_mode = 'uniform'
        method.feat_distill_weight = 0.0
        method.current_supcon_weight = 0.0
        method.current_supcon_start_task = 5
        method.current_supcon_temperature = 0.1
        return method

    def test_task_boundary_records_only_previously_seen_classes(self):
        method = self._method()
        method.before_task(1, [4], [1, 3, 4])
        self.assertEqual(method.kd_old_classes, [1, 3])

    def test_unlearned_class_is_not_in_old_scope(self):
        method = self._method()
        method.before_task(2, [4], [3, 4])
        self.assertEqual(method.kd_old_classes, [3])

    def test_missing_old_row_fails_instead_of_silently_truncating(self):
        method = self._method()
        method.before_task(1, [4], [1, 3, 4])
        method.kd_old_classes = [1, 5]
        batch = torch.tensor([[1., 2.], [2., 1.]])
        with patch('cl_methods.proto_evolve.split_features',
                   side_effect=lambda x, _: [x[:, :1], x[:, 1:]]):
            with self.assertRaisesRegex(ValueError, 'old KD class'):
                method._proto_loss(
                    method.trainer.bottoms, method.trainer.top_model,
                    batch, torch.tensor([4, 4]), loss_ce=torch.zeros(()),
                )

    def test_global_kl_does_not_update_new_or_future_rows(self):
        self._assert_gradient_scope(global_weight=1.0, party_weight=0.0)

    def test_party_kd_does_not_update_new_or_future_rows(self):
        self._assert_gradient_scope(global_weight=0.0, party_weight=1.0)

    def test_two_tasks_train_with_full_width_head(self):
        args = SimpleNamespace(
            device='cpu', data='synthvfl', party_col_ranges=[(0, 1), (1, 2)],
            num_parties=2, num_classes=5, num_tasks=2, aggregation='concat',
            seed=7, lr=0.01, momentum=0.0, weight_decay=0.0,
            epochs_per_task=1, fim_freeze_frac=0.0, proto_sdc=False,
            party_kd_enabled=1, party_kd_mode='uniform',
            party_kd_lambda=1.0,
        )
        trainer = VFLTrainer(
            [torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)],
            TopModel(2, 5, cosine=False), args,
        )
        method = ProtoEvolveCL(trainer, args)
        old_loader = [(torch.tensor([[1., 2.], [2., 1.], [1., 1.], [2., 2.]]),
                       torch.tensor([1, 3, 1, 3]))]
        new_loader = [(torch.tensor([[3., 1.], [1., 3.], [2., 2.], [3., 3.]]),
                       torch.tensor([4, 4, 4, 4]))]
        method.before_task(0, [1, 3], [1, 3])
        method.train_task(old_loader, 0)
        method.after_task(old_loader, 0)
        method.before_task(1, [4], [1, 3, 4])
        history, _ = method.train_task(new_loader, 1)
        self.assertEqual(method.kd_old_classes, [1, 3])
        self.assertTrue(torch.isfinite(torch.tensor(history[0]['loss'])))

    def _assert_gradient_scope(self, global_weight, party_weight):
        method = self._method()
        method.before_task(1, [4], [1, 3, 4])
        method.effective_distill_weight = global_weight
        method.party_kd_lambda = party_weight
        batch = torch.tensor([[1., 2.], [2., 1.]])
        with patch('cl_methods.proto_evolve.split_features',
                   side_effect=lambda x, _: [x[:, :1], x[:, 1:]]):
            loss = method._proto_loss(
                method.trainer.bottoms, method.trainer.top_model,
                batch, torch.tensor([4, 4]), loss_ce=torch.zeros(()),
            )
        loss.backward()
        gradient = method.trainer.top_model.classifier.weight.grad
        self.assertGreater(torch.count_nonzero(gradient[[1, 3]]).item(), 0)
        self.assertEqual(torch.count_nonzero(gradient[[0, 2, 4]]).item(), 0)


if __name__ == '__main__':
    unittest.main()
