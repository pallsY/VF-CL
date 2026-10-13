import unittest
import copy
import json
import random
import tempfile
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from models import TopModel
from vfl_trainer import VFLTrainer
from data_utils import make_deterministic_loader
from determinism import tensor_sha256
from torch.utils.data import TensorDataset
from cl_methods.alsar import (ALSARCL, choose_freeze_prefix,
                              choose_freeze_prefixes, sar_probabilities,
                              update_class_similarity,
                              classifier_sample_gradients)
from replay11_matrix import METHOD_OPTIONS


class AdaptiveFreezeHookTest(unittest.TestCase):
    def test_sar_equations_keep_negative_effective_usage(self):
        labels = torch.tensor([0, 1])
        usage = torch.tensor([0.0, 1.0])
        similarity = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        expected = F.softmax(-torch.tensor([-1.0, 2.0]), dim=0)
        torch.testing.assert_close(
            sar_probabilities(labels, usage, similarity, 1.0), expected)

    def test_paper_retrieval_defaults(self):
        trainer = SimpleNamespace(
            bottoms=[torch.nn.Linear(1, 1)], top_model=TopModel(1, 2))
        method = ALSARCL(trainer, SimpleNamespace(
            er_per_class=2, er_batch=1, batch_size=2))
        self.assertEqual(method.temperature, 0.125)
        self.assertEqual(method.frequency_scale, 4)
        self.assertEqual(METHOD_OPTIONS['alsar']['alsar_temperature'], 0.125)
        self.assertNotIn('alsar_decay', METHOD_OPTIONS['alsar'])

    def test_dynamic_frequency_discount_uses_current_memory_size(self):
        trainer = SimpleNamespace(
            bottoms=[torch.nn.Linear(1, 1)], top_model=TopModel(1, 2))
        method = ALSARCL(trainer, SimpleNamespace(
            er_per_class=4, er_batch=3, batch_size=2, device='cpu'))
        method.buffer.data = {0: [torch.tensor([0.0]), torch.tensor([1.0])]}
        method.usage = {0: [4.0, 0.0]}
        torch.manual_seed(4)
        selected = method._select_replay()
        self.assertEqual(len(selected[0]), 2)
        expected = [4.0 * (1 - 2 / 8) + 1.0, 1.0]
        self.assertEqual(method.usage[0], expected)
        method.buffer.data[0].extend([torch.tensor([2.0]), torch.tensor([3.0])])
        method.usage[0].extend([0.0, 0.0])
        selected = method._select_replay()
        chosen = {int(value) for value in selected[0][:, 0]}
        expected = [value * (1 - 3 / 16) + (index in chosen)
                    for index, value in enumerate(expected + [0.0, 0.0])]
        self.assertEqual(method.usage[0], expected)

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
        self.assertNotIn('last_used', second.get_state())

    def test_bfc_freezes_more_when_batch_information_is_low(self):
        fisher = torch.tensor([1.0, 0.1])
        backward = torch.tensor([4.0, 4.0])
        self.assertEqual(choose_freeze_prefix(fisher, 0.1, 1.0, backward), 2)
        self.assertEqual(choose_freeze_prefix(fisher, 10.0, 1.0, backward), 0)

    def test_full_model_bfc_matches_paper_equations(self):
        fisher = [[2.0, 0.5], [1.0, 3.0]]
        backward = [[3.0, 1.0], [2.0, 4.0]]
        forward, top_fisher, top_backward, ratio = 5.0, 2.0, 3.0, 0.3
        choices = list(product(range(3), repeat=2))
        rates = {}
        for prefixes in choices:
            remaining_fisher = top_fisher + sum(
                sum(values[n:]) for values, n in zip(fisher, prefixes))
            remaining_backward = top_backward + sum(
                sum(values[n:]) for values, n in zip(backward, prefixes))
            rates[prefixes] = remaining_fisher / (forward + remaining_backward)
        best_rate = max(rates.values())
        benefits = {
            prefixes: best_rate * sum(
                sum(values[:n]) for values, n in zip(backward, prefixes))
            - ratio * sum(sum(values[:n]) for values, n in zip(fisher, prefixes))
            for prefixes in choices
        }
        expected = max(choices, key=benefits.get)
        self.assertEqual(
            choose_freeze_prefixes(fisher, ratio, forward, backward,
                                   top_fisher, top_backward), expected)

    def test_every_fourth_step_refreshes_all_fisher_blocks(self):
        bottom = torch.nn.Linear(1, 1, bias=False)
        top = TopModel(1, 2)
        trainer = SimpleNamespace(bottoms=[bottom], top_model=top)
        method = ALSARCL(trainer, SimpleNamespace(
            er_per_class=2, er_batch=2, batch_size=2))
        method.costs = [torch.tensor([1.0], dtype=torch.float64)]
        method.top_cost = 2.0
        method._preliminary_forward = lambda _x: (
            torch.ones(2, 1), torch.zeros(2, 2))
        method.fisher = [[1.0]]
        method.top_fisher = 1.0
        x, y = torch.ones(2, 1), torch.tensor([0, 1])
        refreshed = []
        for _ in range(5):
            method._before_batch(x, y)
            refreshed.append(method.refresh_fisher)
            bottom.weight.grad = torch.ones_like(bottom.weight)
            top.classifier.weight.grad = torch.ones_like(top.classifier.weight)
            old = (method.fisher[0][0], method.top_fisher)
            method._record_top_gradients([top])
            method._record_gradients([bottom])
            if not method.refresh_fisher:
                self.assertEqual((method.fisher[0][0], method.top_fisher), old)
            method._after_batch()
        self.assertEqual(refreshed, [True, False, False, False, True])

    def test_fisher_trace_uses_raw_squared_gradients(self):
        bottom = torch.nn.Linear(1, 1, bias=False)
        top = TopModel(1, 2)
        method = ALSARCL(SimpleNamespace(bottoms=[bottom], top_model=top),
                         SimpleNamespace(er_per_class=2, er_batch=2,
                                         batch_size=2, alsar_fisher_rate=1.0))
        method.refresh_fisher = True
        method.frozen = [0]
        bottom.weight.grad = torch.full_like(bottom.weight, 2.0)
        top.classifier.weight.grad = torch.full_like(
            top.classifier.weight, 2.0)
        method._record_top_gradients([top])
        method._record_gradients([bottom])
        self.assertEqual(method.fisher[0][0], 4.0)
        self.assertEqual(method.top_fisher, 8.0)

    def test_last_feature_gradient_ema_starts_at_zero(self):
        bottom = torch.nn.Linear(1, 2)
        top = TopModel(2, 2)
        with torch.no_grad():
            top.classifier.weight.copy_(torch.eye(2))
        method = ALSARCL(SimpleNamespace(bottoms=[bottom], top_model=top),
                         SimpleNamespace(er_per_class=2, er_batch=2,
                                         batch_size=2))
        method.costs = [torch.tensor([1.0], dtype=torch.float64)]
        method.top_cost = 1.0
        method._preliminary_forward = lambda _x: (
            torch.zeros(2, 2), torch.zeros(2, 2))
        method._before_batch(torch.ones(2, 1), torch.tensor([0, 1]))
        self.assertAlmostEqual(method.last_gradient_mean, 0.005)
        method._after_batch()

    def test_sar_favors_less_used_memory_with_class_similarity(self):
        labels = torch.tensor([0, 1])
        discounted = torch.tensor([10.0, 1.0])
        similarities = torch.eye(2)
        probability = sar_probabilities(labels, discounted,
                                         similarities, temperature=1.0)
        self.assertGreater(float(probability[1]), float(probability[0]))
        self.assertAlmostEqual(float(probability.sum()), 1.0)

    def test_similarity_is_ema_of_per_example_cosine(self):
        gradients = torch.tensor([[100.0, 0.0], [0.0, 1.0],
                                  [1.0, 0.0]])
        labels = torch.tensor([0, 0, 1])
        updated = update_class_similarity(
            torch.zeros(2, 2), gradients, labels, 0.1)
        self.assertAlmostEqual(float(updated[0, 1]), 0.05)
        self.assertAlmostEqual(float(updated[1, 0]), 0.05)
        self.assertAlmostEqual(float(updated[0, 0]), 0.0)
        self.assertAlmostEqual(float(updated[1, 1]), 0.0)

    def test_per_example_classifier_gradient_matches_outer_product(self):
        features = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        logits = torch.zeros(2, 2)
        labels = torch.tensor([0, 1])
        expected = torch.tensor([[-1.0, -1.5, 1.0, 1.5],
                                 [2.0, 2.5, -2.0, -2.5]])
        torch.testing.assert_close(
            classifier_sample_gradients(features, logits, labels), expected)

    def test_four_party_two_task_training_updates_similarity_and_restores_freeze(self):
        torch.manual_seed(7)
        args = SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 1), (1, 2),
                                             (2, 3), (3, 4)],
            aggregation='concat', device='cpu', er_per_class=2,
            er_batch=2, batch_size=4, epochs_per_task=1,
            optimizer='sgd', lr=0.01, bottom_lr_scale=1.0,
            momentum=0.0, weight_decay=0.0,
            own_concentrate_weight=0.0, data_flow_audit=0)
        bottoms = [torch.nn.Linear(1, 1) for _ in range(4)]
        trainer = VFLTrainer(bottoms, TopModel(4, 2), args)
        method = ALSARCL(trainer, args)
        first = [(torch.tensor([[1., 2., 3., 4.], [2., 3., 4., 5.],
                                [3., 4., 5., 6.], [4., 5., 6., 7.]]),
                  torch.tensor([0, 1, 0, 1]))]
        method.before_task(0, [0, 1], [0, 1])
        method.train_task(first, 0)
        method.after_task(first, 0)
        saved_trainer = trainer.get_state()
        saved_method = copy.deepcopy(method.get_state())
        saved_torch_rng = torch.get_rng_state()
        saved_python_rng = random.getstate()
        second = [(first[0][0] + 1, torch.tensor([2, 3, 2, 3]))]
        method.before_task(1, [2, 3], [0, 1, 2, 3])
        before_second = trainer.get_state()
        rng_before_second = torch.get_rng_state()
        original_before_batch = method._before_batch
        original_trace = []
        def trace_original(x, y):
            original_before_batch(x, y)
            original_trace.append((tuple(method.frozen),
                                   method.replay_batch[0].clone(),
                                   method.top_cost,
                                   [cost.clone() for cost in method.costs]))
        method._before_batch = trace_original
        method.train_task(second, 1)
        self.assertEqual(method.class_similarity.shape, (4, 4))
        self.assertTrue(torch.isfinite(method.class_similarity).all())
        torch.testing.assert_close(method.class_similarity,
                                   method.class_similarity.T)
        self.assertTrue(all(p.requires_grad for bottom in bottoms
                            for p in bottom.parameters()))
        reloaded = ALSARCL(trainer, args)
        reloaded.load_state(method.get_state())
        torch.testing.assert_close(reloaded.class_similarity,
                                   method.class_similarity)
        self.assertEqual(reloaded.usage, method.usage)
        clone = VFLTrainer(
            [torch.nn.Linear(1, 1) for _ in range(4)], TopModel(4, 2), args)
        clone.load_state(saved_trainer)
        resumed = ALSARCL(clone, args)
        resumed.load_state(saved_method)
        torch.set_rng_state(saved_torch_rng)
        random.setstate(saved_python_rng)
        resumed.before_task(1, [2, 3], [0, 1, 2, 3])
        for expected, actual in zip(before_second['bottoms'], clone.get_state()['bottoms']):
            for name, parameter in expected.items():
                torch.testing.assert_close(parameter, actual[name])
        for name, parameter in before_second['top_model'].items():
            torch.testing.assert_close(parameter, clone.get_state()['top_model'][name])
        torch.testing.assert_close(rng_before_second, torch.get_rng_state())
        resumed_before_batch = resumed._before_batch
        resumed_trace = []
        def trace_resumed(x, y):
            resumed_before_batch(x, y)
            resumed_trace.append((tuple(resumed.frozen),
                                  resumed.replay_batch[0].clone(),
                                  resumed.top_cost,
                                  [cost.clone() for cost in resumed.costs]))
        resumed._before_batch = trace_resumed
        resumed.train_task(second, 1)
        self.assertEqual(original_trace[0][0], resumed_trace[0][0])
        torch.testing.assert_close(original_trace[0][1], resumed_trace[0][1])
        self.assertEqual(original_trace[0][2], resumed_trace[0][2])
        for expected, actual in zip(original_trace[0][3], resumed_trace[0][3]):
            torch.testing.assert_close(expected, actual)
        for expected, actual in zip(trainer.bottoms, clone.bottoms):
            for name, parameter in expected.state_dict().items():
                torch.testing.assert_close(parameter, actual.state_dict()[name])
        torch.testing.assert_close(method.class_similarity,
                                   resumed.class_similarity)
        self.assertEqual(method.usage, resumed.usage)

    def test_one_replay_draw_feeds_bfc_and_cross_entropy(self):
        args = SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 1)],
            aggregation='concat', device='cpu', er_per_class=2,
            er_batch=1, batch_size=2)
        bottom = torch.nn.Linear(1, 2)
        trainer = SimpleNamespace(
            bottoms=[bottom], top_model=TopModel(2, 2),
            _aggregate=lambda values: values[0],
            bytes_transmitted=0, comm_rounds=0)
        method = ALSARCL(trainer, args)
        method.task_id = 1
        method.buffer.data = {0: [torch.tensor([10.0])],
                              1: [torch.tensor([20.0])]}
        method.usage = {0: [0.0], 1: [1.0]}
        seen = []
        original = method._preliminary_forward
        def inspect(examples):
            seen.append(examples.detach().clone())
            return original(examples)
        method._preliminary_forward = inspect
        current_x = torch.tensor([[1.0], [2.0]])
        current_y = torch.tensor([0, 1])
        method._before_batch(current_x, current_y)
        replay_x, replay_y = method.replay_batch
        torch.testing.assert_close(seen[0][-len(replay_x):], replay_x)
        expected = F.cross_entropy(trainer.top_model(bottom(replay_x)), replay_y)
        actual = method._replay_loss(
            trainer.bottoms, trainer.top_model, current_x, current_y,
            torch.tensor(0.0))
        torch.testing.assert_close(actual, expected)
        method._after_batch()

    def test_audited_loader_records_original_current_batch(self):
        with tempfile.TemporaryDirectory() as output_dir:
            args = SimpleNamespace(
                data='tabvfl', party_col_ranges=[(0, 1), (1, 2),
                                                 (2, 3), (3, 4)],
                aggregation='concat', device='cpu', er_per_class=2,
                er_batch=2, batch_size=4, epochs_per_task=1,
                optimizer='sgd', lr=0.01, bottom_lr_scale=1.0,
                momentum=0.0, weight_decay=0.0,
                own_concentrate_weight=0.0, data_flow_audit=1,
                seed=7, num_workers=0, output_dir=output_dir)
            trainer = VFLTrainer(
                [torch.nn.Linear(1, 1) for _ in range(4)],
                TopModel(4, 2), args)
            method = ALSARCL(trainer, args)
            current = torch.arange(16, dtype=torch.float32).reshape(4, 4)
            first = make_deterministic_loader(
                TensorDataset(current, torch.tensor([0, 1, 0, 1])),
                range(4), args, ('first',), shuffle=False, audit=True)
            method.before_task(0, [0, 1], [0, 1])
            method.train_task(first, 0)
            method.after_task(first, 0)
            later = current + 1
            second = make_deterministic_loader(
                TensorDataset(later, torch.tensor([2, 3, 2, 3])),
                range(4), args, ('second',), shuffle=False, audit=True)
            method.before_task(1, [2, 3], [0, 1, 2, 3])
            method.train_task(second, 1)
            audit = [json.loads(line) for line in
                     (Path(output_dir) / 'data_flow_audit.jsonl').read_text(
                         encoding='utf-8').splitlines()]
            self.assertEqual(len(audit), 2)
            self.assertEqual(audit[0]['batch_sha256'], tensor_sha256(current))
            self.assertEqual(audit[1]['batch_sha256'], tensor_sha256(later))

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
