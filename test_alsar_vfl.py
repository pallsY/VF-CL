import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from models import TopModel
from vfl_trainer import VFLTrainer
from cl_methods.alsar import ALSARCL, choose_freeze_prefix, sar_probabilities


class AdaptiveFreezeHookTest(unittest.TestCase):
    def test_similarity_uses_pre_update_head_weights(self):
        args = SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 1)],
            aggregation='concat', device='cpu', er_per_class=2,
            er_batch=2, batch_size=2,
        )
        bottom = torch.nn.Linear(1, 2)
        trainer = SimpleNamespace(
            bottoms=[bottom], top_model=TopModel(2, 2),
            _aggregate=lambda values: values[0],
            bytes_transmitted=0, comm_rounds=0,
        )
        method = ALSARCL(trainer, args)
        labels = torch.tensor([0, 1])
        method._before_batch(torch.tensor([[1.0], [2.0]]), labels)
        old_weight = trainer.top_model.classifier.weight.detach().clone()
        expected = ((method.batch_logits.softmax(1)
                     - F.one_hot(labels, 2).float()) @ old_weight)
        with torch.no_grad():
            trainer.top_model.classifier.weight[0].add_(10.0)
        method._record_gradients([bottom])
        for class_id in (0, 1):
            torch.testing.assert_close(method.class_gradient[class_id],
                                       expected[class_id])
        method._after_batch()

    def test_memory_usage_and_fisher_survive_checkpoint(self):
        bottom = torch.nn.Linear(1, 1, bias=False)
        trainer = SimpleNamespace(bottoms=[bottom], top_model=TopModel(1, 2))
        args = SimpleNamespace(er_per_class=2, er_batch=2, batch_size=2)
        first = ALSARCL(trainer, args)
        first.after_task([(torch.tensor([[1.0], [2.0], [3.0]]),
                           torch.tensor([0, 0, 0]))], 0)
        self.assertEqual(len(first.buffer.data[0]), 2)
        first.fisher[0][0] = 2.5
        second = ALSARCL(trainer, args)
        second.load_state(first.get_state())
        self.assertEqual(second.fisher[0][0], 2.5)
        self.assertEqual(len(second.usage[0]), 2)
        self.assertEqual(len(second.last_used[0]), 2)

    def test_bfc_freezes_more_when_batch_information_is_low(self):
        fisher = torch.tensor([1.0, 0.1])
        backward = torch.tensor([4.0, 4.0])
        self.assertEqual(choose_freeze_prefix(fisher, 0.1, 1.0, backward), 2)
        self.assertEqual(choose_freeze_prefix(fisher, 10.0, 1.0, backward), 0)

    def test_sar_favors_less_used_memory_with_class_similarity(self):
        labels = torch.tensor([0, 1])
        discounted = torch.tensor([10.0, 1.0])
        similarities = torch.eye(2)
        probability = sar_probabilities(labels, discounted,
                                         similarities, temperature=1.0)
        self.assertGreater(float(probability[1]), float(probability[0]))
        self.assertAlmostEqual(float(probability.sum()), 1.0)

    def test_batch_hook_freezes_and_restores_party_layer(self):
        args = SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 1)], aggregation='concat',
            device='cpu', optimizer='sgd', lr=0.1, bottom_lr_scale=1.0,
            momentum=0.0, weight_decay=0.0, own_concentrate_weight=0.0,
            data_flow_audit=0,
        )
        bottom = torch.nn.Linear(1, 1, bias=False)
        trainer = VFLTrainer([bottom], TopModel(1, 2), args)
        original = bottom.weight.detach().clone()
        events = []

        def before(_x, _y):
            events.append('before')
            bottom.weight.requires_grad_(False)

        def after():
            events.append('after')
            bottom.weight.requires_grad_(True)

        trainer.before_batch_fn = before
        trainer.after_batch_fn = after
        trainer.train_epoch([(torch.tensor([[1.0], [-1.0]]),
                              torch.tensor([0, 1]))],
                            trainer._create_optimizers())
        self.assertEqual(events, ['before', 'after'])
        self.assertTrue(bottom.weight.requires_grad)
        torch.testing.assert_close(bottom.weight, original)


if __name__ == '__main__':
    unittest.main()
