import unittest
import copy
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cl_methods.icarl import ICaRLCL, herding_indices
from models import TopModel
from vfl_trainer import VFLTrainer


class ICaRLHerdingTest(unittest.TestCase):
    def test_stratified_replay_estimates_full_union_bce(self):
        bottom = torch.nn.Identity()
        top = TopModel(2, 3)
        teacher = TopModel(2, 2)
        trainer = SimpleNamespace(
            bottoms=[bottom], top_model=top,
            _aggregate=lambda parts: parts[0],
            bytes_transmitted=0, comm_rounds=0,
        )
        args = SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 2)], device='cpu',
            num_classes=3, er_per_class=20, er_batch=2, batch_size=1,
        )
        method = ICaRLCL(trainer, args)
        method.old_classes = [0, 1]
        method.teacher_bottoms = [bottom]
        method.teacher_top = teacher
        method.buffer.data = {
            0: [torch.tensor([1.0, 0.0])],
            1: [torch.tensor([0.0, 1.0])],
        }
        method._task_current_count = 100
        fresh_x = torch.tensor([[1.0, 1.0]])
        fresh_y = torch.tensor([2])
        loss = method._icarl_loss([bottom], top, fresh_x, fresh_y)

        all_x = torch.cat((fresh_x, torch.eye(2)))
        labels = torch.tensor([2, 0, 1])
        logits = top(all_x)
        targets = torch.zeros_like(logits)
        targets.scatter_(1, labels[:, None], 1.0)
        targets[:, :2] = teacher(all_x).sigmoid()
        per_sample = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none').mean(1)
        expected = (100 * per_sample[0] + per_sample[1:].sum()) / 102
        torch.testing.assert_close(loss, expected)
        got_grad = torch.autograd.grad(loss, top.classifier.weight)[0]
        want_grad = torch.autograd.grad(expected, top.classifier.weight)[0]
        torch.testing.assert_close(got_grad, want_grad)

    def test_four_party_two_task_checkpoint_and_nme(self):
        torch.manual_seed(7)
        args = SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 1), (1, 2), (2, 3), (3, 4)],
            aggregation='concat', device='cpu', num_classes=4,
            er_per_class=20, er_batch=2, batch_size=4,
            epochs_per_task=1, lr=0.01, optimizer='sgd', momentum=0.0,
            weight_decay=0.0, bottom_lr_scale=1.0,
            data_flow_audit=0, own_concentrate_weight=0.0,
        )
        bottoms = [torch.nn.Linear(1, 1) for _ in range(4)]
        trainer = VFLTrainer(bottoms, TopModel(4, 1), args)
        method = ICaRLCL(trainer, args)
        for task_id, classes in enumerate(([0, 1], [2, 3])):
            labels = torch.tensor(classes * 4)
            examples = torch.stack([
                torch.tensor([float(label == column) for column in range(4)])
                for label in labels
            ])
            loader = DataLoader(TensorDataset(examples, labels), batch_size=4)
            method.before_task(task_id, classes, list(range(2 * task_id + 2)))
            method.train_task(loader, task_id)
            method.after_task(loader, task_id)
        self.assertEqual(trainer.comm_rounds > 0, True)
        self.assertEqual(set(method.buffer.data), {0, 1, 2, 3})
        saved_model = trainer.get_state()
        saved_method = method.get_state()
        restored = VFLTrainer(
            [torch.nn.Linear(1, 1) for _ in range(4)], TopModel(4, 4), args)
        restored.load_state(saved_model)
        restored_method = ICaRLCL(restored, args)
        restored_method.load_state(saved_method)
        for class_id, exemplars in method.buffer.data.items():
            for actual, expected in zip(restored_method.buffer.data[class_id], exemplars):
                torch.testing.assert_close(actual, expected)
        batches = lambda _task, classes: [(
            torch.stack([
                torch.tensor([float(label == column) for column in range(4)])
                for label in classes
            ]),
            torch.tensor(classes))]
        original = method._evaluate({0: [0, 1], 1: [2, 3]}, batches)
        recovered = restored_method._evaluate({0: [0, 1], 1: [2, 3]}, batches)
        self.assertEqual(original['class_il_pred_task'],
                         recovered['class_il_pred_task'])

    def test_binary_targets_match_old_teacher_and_new_labels(self):
        bottom = torch.nn.Identity()
        top = TopModel(2, 3)
        teacher = TopModel(2, 2)
        trainer = SimpleNamespace(
            bottoms=[bottom], top_model=top,
            _aggregate=lambda parts: parts[0],
            bytes_transmitted=0, comm_rounds=0,
        )
        args = SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 2)], device='cpu',
            num_classes=3, er_per_class=20, er_batch=2, batch_size=2,
        )
        method = ICaRLCL(trainer, args)
        method.old_classes = [0, 1]
        method.teacher_bottoms = [bottom]
        method.teacher_top = teacher
        examples = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([2, 0])
        with torch.no_grad():
            logits = top(examples)
            targets = torch.zeros_like(logits)
            targets[:, :2] = teacher(examples).sigmoid()
            targets[0, 2] = 1.0
            expected = F.binary_cross_entropy_with_logits(logits, targets)
        actual = method._icarl_loss(
            [bottom], top, examples, labels,
            loss_ce=torch.tensor(1000.0),
        )
        torch.testing.assert_close(actual, expected)

    def test_nearest_exemplar_mean_is_primary_inference(self):
        bottom = torch.nn.Identity()
        top = TopModel(2, 2)
        with torch.no_grad():
            top.classifier.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
            top.classifier.bias.zero_()
        trainer = SimpleNamespace(
            bottoms=[bottom], top_model=top,
            _aggregate=lambda parts: parts[0],
            bytes_transmitted=0, comm_rounds=0,
        )
        args = SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 2)], device='cpu',
            num_classes=2, er_per_class=20, er_batch=2, batch_size=2,
        )
        method = ICaRLCL(trainer, args)
        method.buffer.data = {
            0: [torch.tensor([1.0, 0.0])],
            1: [torch.tensor([0.0, 1.0])],
        }
        readouts = method._evaluate(
            {0: [0, 1]},
            lambda _task, _classes: [(
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                torch.tensor([0, 1]))],
        )
        self.assertEqual(readouts['class_il_pred_task']['task_0'], 1.0)
        self.assertEqual(readouts['class_il_global']['task_0'], 0.0)

    def test_fixed_total_budget_reduces_old_prefixes(self):
        bottom = torch.nn.Linear(2, 2, bias=False)
        top = TopModel(2, 1)
        trainer = SimpleNamespace(
            bottoms=[bottom], top_model=top,
            _aggregate=lambda parts: parts[0],
            bytes_transmitted=0, comm_rounds=0,
        )
        args = SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 2)], device='cpu',
            num_classes=6, er_per_class=20, er_batch=2, batch_size=16,
        )
        method = ICaRLCL(trainer, args)
        for task_id, size in enumerate((130, 130, 130, 10)):
            labels = torch.full((size,), task_id, dtype=torch.long)
            values = torch.arange(size, dtype=torch.float32)
            examples = torch.stack((values + task_id * 200, values.square()), 1)
            loader = DataLoader(TensorDataset(examples, labels), batch_size=16)
            method.before_task(task_id, [task_id], list(range(task_id + 1)))
            old = {label: [item.clone() for item in stored]
                   for label, stored in method.buffer.data.items()}
            method.after_task(loader, task_id)
            per_class = 120 // (task_id + 1)
            self.assertEqual(method.total_budget, 120)
            self.assertLessEqual(method.buffer.total_size(), 120)
            for label, stored in method.buffer.data.items():
                self.assertEqual(len(stored), min(per_class, size if label == task_id else 130))
                if label in old:
                    for actual, expected in zip(stored, old[label][:per_class]):
                        torch.testing.assert_close(actual, expected)

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
            num_classes=2, er_per_class=2, er_batch=2, batch_size=2,
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
        args = SimpleNamespace(device='cpu', num_classes=2, er_per_class=2,
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
        different_budget = ICaRLCL(
            trainer, SimpleNamespace(device='cpu', num_classes=3,
                                     er_per_class=2, er_batch=2, batch_size=2))
        with self.assertRaisesRegex(ValueError, 'total exemplar budget'):
            different_budget.load_state(first.get_state())


if __name__ == '__main__':
    unittest.main()
