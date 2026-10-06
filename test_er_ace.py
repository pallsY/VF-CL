import random
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from cl_methods.er_ace import ERAccCL
from cl_methods.ewc import EWCCL
from cl_methods.lwf import LwFCL
from models import TopModel
from vfl_trainer import VFLTrainer


class ERACEReservoirTests(unittest.TestCase):
    def method(self, capacity=3, classes=5):
        return ERAccCL(None, SimpleNamespace(
            num_classes=classes, er_ace_buffer_size=capacity,
            er_ace_batch=64, device='cpu'))

    def populated(self):
        method = self.method()
        self.assertTrue(hasattr(method, 'buffer'), 'ER-ACE needs raw replay')
        method.buffer.add_batch(torch.arange(12.).reshape(3, 4),
                                torch.tensor([0, 1, 2]))
        return method

    def assert_state_equal(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            if torch.is_tensor(left[key]):
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            else:
                self.assertEqual(left[key], right[key])

    def test_auto_and_explicit_capacity(self):
        for classes in (5, 100, 200):
            with self.subTest(classes=classes):
                method = self.method(0, classes)
                self.assertTrue(hasattr(method, 'buffer'))
                self.assertEqual(method.buffer_size, 20 * classes)
                self.assertEqual(method.buffer.buffer_size, 20 * classes)
        self.assertEqual(self.populated().buffer_size, 3)

    def test_constructor_rejects_invalid_capacity_and_class_count(self):
        for size in (-1, True, 1.5, '3', None):
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.method(size)
        for classes in (0, -1, True, 2.5, '5', None):
            with self.subTest(classes=classes), self.assertRaises(ValueError):
                self.method(classes=classes)

    def test_empty_state_has_exact_fields_and_round_trips(self):
        source = self.method()
        state = source.get_state()
        self.assertEqual(set(state), {'buffer_size', 'num_seen', 'examples', 'labels'})
        self.assertEqual(state['buffer_size'], 3)
        self.assertEqual(state['num_seen'], 0)
        self.assertIsNone(state['examples'])
        self.assertEqual(state['labels'].shape, (0,))
        self.assertEqual(state['labels'].dtype, torch.long)
        restored = self.populated()
        restored.load_state(state)
        self.assert_state_equal(restored.get_state(), state)
        self.assertIsNone(restored.buffer.sample(64))

    def test_reservoir_uses_python_rng_and_counts_dropped_samples(self):
        method = self.populated()
        with patch('cl_methods.er_ace.random.randint', side_effect=[1, 4, 0]) as rng:
            method.buffer.add_batch(torch.arange(12., 24.).reshape(3, 4),
                                    torch.tensor([3, 4, 0]))
        self.assertEqual(rng.call_args_list, [((0, 3),), ((0, 4),), ((0, 5),)])
        state = method.get_state()
        self.assertEqual(method.buffer.size(), 3)
        self.assertEqual(state['num_seen'], 6)
        torch.testing.assert_close(state['labels'], torch.tensor([0, 3, 2]))
        torch.testing.assert_close(state['examples'], torch.tensor([
            [20., 21., 22., 23.], [12., 13., 14., 15.], [8., 9., 10., 11.]]))

    def test_insert_and_samples_are_independent_detached_cpu_clones(self):
        method = self.method()
        self.assertTrue(hasattr(method, 'buffer'))
        batch = torch.arange(24.).reshape(6, 4).requires_grad_()
        labels = torch.tensor([0, 1, 2, 3, 4, 0])
        method.buffer.add_batch(batch[:3], labels[:3])
        expected = deepcopy(method.get_state())
        with torch.no_grad():
            batch.zero_()
            labels.zero_()
        self.assert_state_equal(method.get_state(), expected)
        for example in method.buffer.ex:
            self.assertEqual(example.device.type, 'cpu')
            self.assertFalse(example.requires_grad)
            self.assertIsNone(example.grad_fn)
            self.assertEqual(example.untyped_storage().nbytes(), example.numel() * example.element_size())
        with patch('cl_methods.er_ace.random.sample', return_value=[2, 0]) as rng:
            x, y = method.buffer.sample(2)
        rng.assert_called_once_with(range(3), 2)
        torch.testing.assert_close(x, expected['examples'][[2, 0]])
        torch.testing.assert_close(y, torch.tensor([2, 0]))
        x.zero_()
        y.zero_()
        self.assert_state_equal(method.get_state(), expected)
        self.assertEqual(method.buffer.sample(64)[0].shape, (3, 4))
        self.assertIsNone(method.buffer.sample(0))
        for count in (-1, True, 1.5, '2', None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                method.buffer.sample(count)

    def test_nonempty_state_round_trip_and_bidirectional_alias_independence(self):
        source = self.populated()
        state = source.get_state()
        expected = deepcopy(state)
        restored = self.method()
        restored.load_state(state)
        self.assert_state_equal(restored.get_state(), expected)
        state['examples'].zero_()
        state['labels'].zero_()
        self.assert_state_equal(source.get_state(), expected)
        self.assert_state_equal(restored.get_state(), expected)
        restored.buffer.ex[0].zero_()
        self.assert_state_equal(source.get_state(), expected)

    def test_checkpoint_rejects_malformed_state_atomically(self):
        corruptions = {
            'not mapping': lambda s: None,
            'missing': lambda s: {k: v for k, v in s.items() if k != 'num_seen'},
            'extra logits': lambda s: dict(s, logits=torch.zeros(3, 5)),
            'capacity mismatch': lambda s: dict(s, buffer_size=4),
            'capacity bool': lambda s: dict(s, buffer_size=True),
            'capacity float': lambda s: dict(s, buffer_size=3.0),
            'counter bool': lambda s: dict(s, num_seen=True),
            'counter float': lambda s: dict(s, num_seen=3.0),
            'counter string': lambda s: dict(s, num_seen='3'),
            'counter negative': lambda s: dict(s, num_seen=-1),
            'counter under size': lambda s: dict(s, num_seen=2),
            'oversized': lambda s: dict(s, examples=torch.zeros(4, 4), labels=torch.zeros(4, dtype=torch.long), num_seen=4),
            'unequal': lambda s: dict(s, labels=torch.tensor([0, 1])),
            'underfull after overflow': lambda s: dict(s, examples=torch.zeros(2, 4), labels=torch.tensor([0, 1]), num_seen=4),
            'missing examples': lambda s: dict(s, examples=None),
            'examples list': lambda s: dict(s, examples=s['examples'].tolist()),
            'examples rank': lambda s: dict(s, examples=torch.zeros(3)),
            'empty features': lambda s: dict(s, examples=torch.zeros(3, 0)),
            'example integer': lambda s: dict(s, examples=s['examples'].long()),
            'example dtype changed': lambda s: dict(s, examples=s['examples'].double()),
            'example shape changed': lambda s: dict(s, examples=torch.zeros(3, 5)),
            'example complex': lambda s: dict(s, examples=s['examples'].cfloat()),
            'example nan': lambda s: dict(s, examples=torch.full((3, 4), float('nan'))),
            'example inf': lambda s: dict(s, examples=torch.full((3, 4), float('inf'))),
            'labels list': lambda s: dict(s, labels=[0, 1, 2]),
            'labels rank': lambda s: dict(s, labels=s['labels'].reshape(3, 1)),
            'labels float': lambda s: dict(s, labels=s['labels'].float()),
            'labels bool': lambda s: dict(s, labels=s['labels'].bool()),
            'labels int32': lambda s: dict(s, labels=s['labels'].int()),
            'labels negative': lambda s: dict(s, labels=torch.tensor([-1, 1, 2])),
            'labels out of range': lambda s: dict(s, labels=torch.tensor([0, 1, 5])),
        }
        for label, corrupt in corruptions.items():
            with self.subTest(label=label):
                receiver = self.populated()
                before = receiver.get_state()
                old_ex, old_lb = receiver.buffer.ex, receiver.buffer.lb
                with self.assertRaisesRegex(ValueError, 'ER-ACE'):
                    receiver.load_state(corrupt(deepcopy(before)))
                self.assertIs(receiver.buffer.ex, old_ex)
                self.assertIs(receiver.buffer.lb, old_lb)
                self.assert_state_equal(receiver.get_state(), before)

    def test_load_detaches_cpu_clones_and_supports_image_shape(self):
        source = self.method()
        source.buffer.add_batch(torch.arange(24., dtype=torch.float64).reshape(2, 3, 4),
                                torch.tensor([1, 2]))
        state = source.get_state()
        state['examples'].requires_grad_()
        restored = self.method()
        restored.load_state(state)
        self.assert_state_equal(restored.get_state(), source.get_state())
        for value in restored.buffer.ex:
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)
            self.assertEqual(value.device.type, 'cpu')
            self.assertEqual(value.untyped_storage().nbytes(), value.numel() * value.element_size())
        with torch.no_grad():
            state['examples'].zero_()
        self.assert_state_equal(restored.get_state(), source.get_state())

    def test_add_batch_rejects_inconsistent_shape_dtype_or_invalid_later_row_atomically(self):
        invalid = [
            (torch.zeros(2, 5), torch.tensor([0, 1])),
            (torch.zeros(2, 4, dtype=torch.float64), torch.tensor([0, 1])),
            (torch.zeros(2, 4), torch.tensor([0, 5])),
            (torch.tensor([[0., 0., 0., 0.], [0., float('nan'), 0., 0.]]), torch.tensor([0, 1])),
            (torch.zeros(2, 4), torch.tensor([0])),
        ]
        for examples, labels in invalid:
            with self.subTest(shape=examples.shape, dtype=examples.dtype):
                method = self.populated()
                before = method.get_state()
                rng = random.getstate()
                with self.assertRaisesRegex(ValueError, 'ER-ACE'):
                    method.buffer.add_batch(examples, labels)
                self.assert_state_equal(method.get_state(), before)
                self.assertEqual(random.getstate(), rng)

    def test_fixed_rng_resume_preserves_sampling_and_next_reservoir_update(self):
        self.addCleanup(random.setstate, random.getstate())
        source = self.populated()
        restored = self.method()
        restored.load_state(source.get_state())
        random.seed(312)
        boundary_rng = random.getstate()
        next_x, next_y = torch.arange(20.).reshape(5, 4), torch.arange(5)
        replay = source.buffer.sample(2)
        source.buffer.add_batch(next_x, next_y)
        random.setstate(boundary_rng)
        restored_replay = restored.buffer.sample(2)
        restored.buffer.add_batch(next_x, next_y)
        torch.testing.assert_close(replay, restored_replay, rtol=0, atol=0)
        self.assert_state_equal(source.get_state(), restored.get_state())


class ERACELossTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        random.seed(312)
        torch.manual_seed(312)
        self.x = torch.tensor([[0.1, 0.7, -0.4, 0.2],
                               [0.8, -0.3, 0.5, 1.2]])
        self.y = torch.tensor([3, 1])
        self.replay_x = torch.tensor([[-0.7, 0.4, 1.1, -0.8],
                                      [1.3, 0.2, -0.9, 0.6]])
        self.replay_y = torch.tensor([0, 2])

    def method(self, lr=0.0, capacity=3, batch=64, aggregation='sum'):
        args = SimpleNamespace(
            num_parties=2, num_classes=5, device='cpu', data='synthvfl',
            party_col_ranges=[(0, 2), (2, 4)], aggregation=aggregation,
            lr=lr, momentum=0.0, weight_decay=0.0, epochs_per_task=1,
            er_ace_buffer_size=capacity, er_ace_batch=batch,
        )
        trainer = VFLTrainer([nn.Linear(2, 2), nn.Linear(2, 2)],
                             TopModel(2 if aggregation == 'sum' else 4, 5), args)
        return ERAccCL(trainer, args)

    @staticmethod
    def logits(method, x):
        trainer = method.trainer
        embeddings = [trainer.bottoms[0](x[:, :2]),
                      trainer.bottoms[1](x[:, 2:])]
        return trainer.top_model(trainer._aggregate(embeddings))

    def direct_loss(self, method, x, y, replay=None):
        # Deliberately independent of the implementation's class mapping.
        local_y = torch.tensor([method.new_classes.index(int(c)) for c in y])
        incoming = F.cross_entropy(self.logits(method, x)[:, method.new_classes],
                                   local_y)
        if replay is None:
            return incoming
        return incoming + F.cross_entropy(self.logits(method, replay[0]), replay[1])

    def test_incoming_uses_exact_noncontiguous_current_classes_not_full_head(self):
        for task_id in (0, 1):
            for labels in (self.y, torch.tensor([3, 3])):
                with self.subTest(task_id=task_id, labels=labels.tolist()):
                    method = self.method()
                    method.before_task(task_id, [3, 1], [0, 1, 3])
                    expected = self.direct_loss(method, self.x, labels)
                    full_ce = F.cross_entropy(self.logits(method, self.x), labels)
                    self.assertGreater(abs(expected.item() - full_ce.item()), 0.1)
                    history, _ = method.train_task([(self.x, labels)], task_id)
                    self.assertAlmostEqual(history[0]['loss'], expected.item(), places=6)

    def test_before_task_rejects_invalid_classes_before_mutation(self):
        invalid = [[], [1, 1], [-1], [5], [True], [1.0], ['1'], None]
        for classes in invalid:
            with self.subTest(classes=classes):
                method = self.method()
                method.before_task(0, [3, 1], [1, 3])
                before = method.trainer.get_state()
                with self.assertRaisesRegex(ValueError, 'ER-ACE.*classes'):
                    method.before_task(1, classes, [0, 1, 2, 3])
                self.assertEqual(method.new_classes, [3, 1])
                torch.testing.assert_close(method.trainer.get_state(), before, rtol=0, atol=0)
        method = self.method()
        with self.assertRaisesRegex(ValueError, 'ER-ACE.*classes'):
            method.before_task(1, [1, 3], [0, 1])
        for classes in invalid:
            with self.subTest(seen_classes=classes), self.assertRaisesRegex(ValueError, 'ER-ACE.*classes'):
                method.before_task(1, [1, 3], classes)

    def test_constructor_rejects_invalid_replay_batch(self):
        for count in (0, -1, True, 1.5, '2', None):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'ER-ACE'):
                self.method(batch=count)

    def test_batchnorm_incoming_updates_once_and_replay_updates_once(self):
        for with_replay in (False, True):
            method = self.method(lr=0.03, capacity=8)
            method.trainer.bottoms = [nn.Sequential(bottom, nn.BatchNorm1d(2))
                                      for bottom in method.trainer.bottoms]
            method.before_task(1, [3, 1], [0, 1, 2, 3])
            reference = deepcopy(method)
            replay = (self.replay_x, self.replay_y) if with_replay else None
            if with_replay:
                method.buffer.add_batch(*replay)
            expected = self.direct_loss(reference, self.x, self.y, replay)
            reference_opts = reference.trainer._create_optimizers()
            expected.backward()
            for optimizer in [*reference_opts[0], reference_opts[1]]:
                optimizer.step()
            history, _ = method.train_task([(self.x, self.y)], 1)
            self.assertAlmostEqual(history[0]['loss'], expected.item(), places=6)
            for party in range(2):
                actual_bn = method.trainer.bottoms[party][1]
                expected_bn = reference.trainer.bottoms[party][1]
                for name in ('num_batches_tracked', 'running_mean', 'running_var'):
                    with self.subTest(replay=with_replay, party=party, buffer=name):
                        torch.testing.assert_close(getattr(actual_bn, name),
                                                   getattr(expected_bn, name), rtol=0, atol=1e-7)
            torch.testing.assert_close(method.trainer.get_state(), reference.trainer.get_state(),
                                       rtol=1e-6, atol=1e-7)

    def test_trainer_exact_ce_classes_uses_first_logits_and_preserves_slice_settings(self):
        method = self.method()
        trainer = method.trainer
        trainer.ce_lo, trainer.ce_hi = 1, 4
        trainer.ce_classes = [3, 1]
        expected = F.cross_entropy(self.logits(method, self.x)[:, [3, 1]],
                                   torch.tensor([0, 1]))
        history, _ = trainer.train_task([(self.x, self.y)], 1)
        self.assertAlmostEqual(history[0]['loss'], expected.item(), places=6)
        self.assertEqual((trainer.ce_lo, trainer.ce_hi), (1, 4))
        method.before_task(1, [4, 2], [0, 1, 2, 3, 4])
        self.assertEqual(trainer.ce_classes, [4, 2])
        self.assertIsNot(trainer.ce_classes, method.new_classes)

    def test_trainer_rejects_invalid_exact_ce_classes_and_membership(self):
        for classes in ([], [1, 1], [-1], [5], [True], [1.0], ['1'], True, '13', {1, 3}):
            with self.subTest(classes=classes):
                method = self.method()
                method.trainer.ce_classes = classes
                with self.assertRaisesRegex(ValueError, 'CE classes'):
                    method.trainer.train_task([(self.x, self.y)], 1)
        for labels in (torch.tensor([3, 0]), torch.tensor([3, 5]), self.y.float(),
                       self.y.int(), self.y.bool(), self.y[:, None], self.y[:1], self.y[:0]):
            with self.subTest(labels=labels.tolist(), dtype=labels.dtype):
                method = self.method()
                method.trainer.ce_classes = [3, 1]
                with self.assertRaisesRegex(ValueError, 'CE.*labels'):
                    method.trainer.train_task([(self.x, labels)], 1)

    def test_lwf_and_ewc_keep_their_original_contiguous_ce_scope(self):
        for method_type in (LwFCL, EWCCL):
            with self.subTest(method=method_type.__name__):
                fixture = self.method()
                trainer = fixture.trainer
                method = method_type(trainer, fixture.args)
                self.assertIsNone(getattr(trainer, 'ce_classes', None))
                method.before_task(0, [2, 3], [0, 1, 2, 3])
                self.assertEqual((trainer.ce_lo, trainer.ce_hi), (2, 4))
                labels = torch.tensor([2, 3])
                expected = F.cross_entropy(self.logits(fixture, self.x)[:, 2:4], labels - 2)
                history, _ = method.train_task([(self.x, labels)], 0)
                self.assertAlmostEqual(history[0]['loss'], expected.item(), places=6)
                self.assertIsNone(getattr(trainer, 'ce_classes', None))

    def test_hook_reuses_incoming_ace_without_forward_and_preserves_modes_and_gradients(self):
        method = self.method(capacity=8)
        method.before_task(1, [3, 1], [0, 1, 3])
        method.trainer.bottoms[0].eval()
        method.trainer.top_model.eval()
        # Preserve mixed parent/child flags, not just each model's root flag.
        method.trainer.top_model.classifier.train()
        modules = [module for model in [*method.trainer.bottoms, method.trainer.top_model]
                   for module in model.modules()]
        modes = [module.training for module in modules]
        expected = self.direct_loss(method, self.x, self.y)
        parameters = [p for model in [*method.trainer.bottoms, method.trainer.top_model]
                      for p in model.parameters()]
        expected_gradients = torch.autograd.grad(expected, parameters, retain_graph=True)
        forwards = []
        handle = method.trainer.top_model.register_forward_pre_hook(
            lambda _module, _inputs: forwards.append(True))
        try:
            loss = method._er_ace_loss(method.trainer.bottoms, method.trainer.top_model,
                                       self.x.requires_grad_(), self.y, expected)
        finally:
            handle.remove()
        self.assertIs(loss, expected)
        self.assertEqual(forwards, [])
        self.assertEqual(loss.ndim, 0)
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        loss.backward()
        torch.testing.assert_close([p.grad for p in parameters], expected_gradients, rtol=0, atol=0)
        torch.testing.assert_close(method.trainer.top_model.classifier.weight.grad[[0, 2, 4]],
                                   torch.zeros(3, 2), rtol=0, atol=0)
        self.assertEqual([module.training for module in modules], modes)
        self.assertTrue(all(not x.requires_grad and x.grad_fn is None for x in method.buffer.ex))
        # Replay adds full-head CE while reusing the incoming term's graph.
        expected = self.direct_loss(method, self.x, self.y, (self.x.detach(), self.y))
        incoming = self.direct_loss(method, self.x, self.y)
        loss = method._er_ace_loss(method.trainer.bottoms, method.trainer.top_model,
                                   self.x, self.y, incoming)
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        self.assertEqual([module.training for module in modules], modes)

    def test_hook_requires_trainer_incoming_ace_instead_of_recomputing(self):
        method = self.method()
        method.before_task(0, [3, 1], [1, 3])
        with self.assertRaisesRegex(ValueError, 'ER-ACE.*incoming ACE'):
            method._er_ace_loss(method.trainer.bottoms, method.trainer.top_model, self.x, self.y)

    def test_hook_rejects_invalid_current_labels_before_rng_forward_or_buffer_changes(self):
        cases = [(self.x, torch.tensor([3, 0])),
                 (self.x, self.y.float()), (self.x, self.y[:, None]),
                 (self.x, torch.tensor([3, 5])), (self.x, torch.tensor([-1, 1])),
                 (self.x[:0], self.y[:0]), (self.x, self.y[:1]),
                 (torch.full_like(self.x, float('nan')), self.y)]
        for x, y in cases:
            with self.subTest(shape=y.shape, dtype=y.dtype):
                method = self.method()
                method.before_task(1, [3, 1], [0, 1, 3])
                method.buffer.add_batch(self.replay_x, self.replay_y)
                state, rng = method.get_state(), random.getstate()
                forwards = []
                handle = method.trainer.top_model.register_forward_pre_hook(
                    lambda _module, _inputs: forwards.append(True))
                try:
                    with self.assertRaisesRegex(ValueError, 'ER-ACE'):
                        method._er_ace_loss(method.trainer.bottoms, method.trainer.top_model, x, y)
                finally:
                    handle.remove()
                self.assertEqual(forwards, [])
                self.assertEqual(random.getstate(), rng)
                torch.testing.assert_close(method.get_state(), state, rtol=0, atol=0)
        method = self.method()
        with self.assertRaisesRegex(ValueError, 'ER-ACE.*before_task'):
            method._er_ace_loss(method.trainer.bottoms, method.trainer.top_model, self.x, self.y)

    def test_single_current_class_has_zero_incoming_ce_but_still_populates_replay(self):
        method = self.method(lr=0.03)
        method.before_task(0, [3], [3])
        before = method.trainer.get_state()
        history, _ = method.train_task([(self.x, torch.tensor([3, 3]))], 0)
        self.assertEqual(history[0]['loss'], 0.0)
        torch.testing.assert_close(method.trainer.get_state(), before, rtol=0, atol=0)
        self.assertEqual(method.buffer.num_seen, 2)

    def test_total_is_incoming_ace_plus_full_classifier_replay_ce(self):
        method = self.method()
        method.before_task(1, [3, 1], [0, 1, 2, 3])
        method.buffer.add_batch(self.replay_x, self.replay_y)
        expected = self.direct_loss(method, self.x, self.y,
                                    (self.replay_x, self.replay_y))
        # Class 4 is unseen but must still contribute to the replay denominator.
        truncated = F.cross_entropy(self.logits(method, self.replay_x)[:, :4],
                                    self.replay_y)
        self.assertGreater(abs(truncated.item() - F.cross_entropy(
            self.logits(method, self.replay_x), self.replay_y).item()), 0.01)
        history, _ = method.train_task([(self.x, self.y)], 1)
        self.assertAlmostEqual(history[0]['loss'], expected.item(), places=6)

    def test_task_zero_samples_existing_examples_before_online_insert(self):
        method = self.method(capacity=8)
        method.before_task(0, [3, 1], [1, 3])
        second_x, second_y = self.x.flip(0) + 0.5, self.y.flip(0)
        first = self.direct_loss(method, self.x, self.y)
        second = self.direct_loss(method, second_x, second_y, (self.x, self.y))
        seen_at_forward = []
        handle = method.trainer.top_model.register_forward_pre_hook(
            lambda _module, _args: seen_at_forward.append(method.buffer.num_seen))
        try:
            history, _ = method.train_task([(self.x, self.y), (second_x, second_y)], 0)
        finally:
            handle.remove()
        self.assertEqual(method.buffer.num_seen, 4)
        self.assertEqual(seen_at_forward, [0, 2, 2])
        self.assertAlmostEqual(history[0]['loss'], ((first + second) / 2).item(), places=6)
        state = deepcopy(method.get_state())
        method.after_task([], 0)
        torch.testing.assert_close(method.get_state(), state, rtol=0, atol=0)
        self.x.zero_()
        self.assertFalse(torch.equal(method.get_state()['examples'][:2], self.x))

    def test_real_trainer_updates_all_models_like_direct_formula_without_stale_gradients(self):
        for aggregation in ('sum', 'concat'):
            with self.subTest(aggregation=aggregation):
                method = self.method(lr=0.03, capacity=8, aggregation=aggregation)
                method.before_task(1, [3, 1], [0, 1, 2, 3])
                method.buffer.add_batch(self.replay_x, self.replay_y)
                reference = self.method(lr=0.03, capacity=8, aggregation=aggregation)
                reference.trainer.load_state(method.trainer.get_state())
                reference.before_task(1, [3, 1], [0, 1, 2, 3])
                before = method.trainer.get_state()
                optimizers = reference.trainer._create_optimizers()
                all_opts = [*optimizers[0], optimizers[1]]
                replay_x, replay_y = self.replay_x, self.replay_y
                for x, y in [(self.x, self.y), (self.x.flip(0), self.y.flip(0))]:
                    for optimizer in all_opts:
                        optimizer.zero_grad()
                    self.direct_loss(reference, x, y, (replay_x, replay_y)).backward()
                    for optimizer in all_opts:
                        optimizer.step()
                    replay_x = torch.cat([replay_x, x])
                    replay_y = torch.cat([replay_y, y])
                method.train_task([(self.x, self.y),
                                   (self.x.flip(0), self.y.flip(0))], 1)
                after = method.trainer.get_state()
                torch.testing.assert_close(after, reference.trainer.get_state(),
                                           rtol=1e-6, atol=1e-7)
                for index in range(2):
                    self.assertFalse(torch.equal(before['bottoms'][index]['weight'],
                                                 after['bottoms'][index]['weight']))
                    self.assertTrue(all(p.grad is None or not torch.count_nonzero(p.grad)
                                        for p in method.trainer.bottoms[index].parameters()))

    def test_trainer_split_and_direct_only_gradient_updates_match_formulas(self):
        for gradient_path in ('split', 'split-plus-direct', 'direct-only'):
            with self.subTest(gradient_path=gradient_path):
                method, reference = self.method(lr=0.03), self.method(lr=0.03)
                reference.trainer.load_state(method.trainer.get_state())

                def penalty(bottoms):
                    return 0.3 * sum(p.square().sum() for bottom in bottoms
                                     for p in bottom.parameters())

                def loss_hook(bottoms, top, x, y, loss_ce):
                    if gradient_path == 'direct-only':
                        return penalty(bottoms)
                    return loss_ce + penalty(bottoms)

                reference_opts = reference.trainer._create_optimizers()
                for _ in range(2):
                    for optimizer in [*reference_opts[0], reference_opts[1]]:
                        optimizer.zero_grad()
                    loss = F.cross_entropy(self.logits(reference, self.x), self.y)
                    if gradient_path == 'direct-only':
                        loss = penalty(reference.trainer.bottoms)
                    elif gradient_path == 'split-plus-direct':
                        loss = loss + penalty(reference.trainer.bottoms)
                    loss.backward()
                    for optimizer in [*reference_opts[0], reference_opts[1]]:
                        optimizer.step()
                method.trainer.train_task([(self.x, self.y)] * 2, 1,
                                          extra_loss_fn=None if gradient_path == 'split' else loss_hook)
                torch.testing.assert_close(method.trainer.get_state(), reference.trainer.get_state(),
                                           rtol=1e-6, atol=1e-7)

    def test_fixed_rng_resume_matches_replay_loss_gradients_update_and_buffer(self):
        source = self.method(lr=0.03, batch=2)
        source.before_task(0, [0, 2], [0, 2])
        source.train_task([(self.replay_x, self.replay_y),
                           (self.replay_x.flip(0), self.replay_y.flip(0))], 0)
        self.assertGreater(source.buffer.size(), 0)
        trainer_state, method_state = source.trainer.get_state(), source.get_state()
        boundary_rng = random.getstate(), torch.get_rng_state()

        def next_step(method):
            method.before_task(1, [3, 1], [0, 1, 2, 3])
            # Predict the real sample without advancing the global Python RNG.
            predictor = random.Random()
            predictor.setstate(random.getstate())
            indices = predictor.sample(range(method.buffer.size()), 2)
            state = method.get_state()
            replay = state['examples'][indices], state['labels'][indices]
            expected = self.direct_loss(method, self.x, self.y, replay)
            gradients, forwards, handles = [], [], []
            for model_index, model in enumerate([*method.trainer.bottoms,
                                                method.trainer.top_model]):
                for name, parameter in model.named_parameters():
                    handles.append(parameter.register_hook(
                        lambda grad, key=(model_index, name): gradients.append((key, grad.clone()))))
            handles.append(method.trainer.bottoms[0].register_forward_pre_hook(
                lambda _module, inputs: forwards.append(inputs[0].detach().clone())))
            try:
                history, _ = method.train_task([(self.x, self.y)], 1)
            finally:
                for handle in handles:
                    handle.remove()
            self.assertAlmostEqual(history[0]['loss'], expected.item(), places=6)
            torch.testing.assert_close(forwards[-1], replay[0][:, :2], rtol=0, atol=0)
            return history, forwards, gradients, method.trainer.get_state(), method.get_state()

        uninterrupted = next_step(source)
        uninterrupted_rng = random.getstate(), torch.get_rng_state()
        restored = self.method(lr=0.03, batch=2)
        restored.trainer.load_state(trainer_state)
        restored.load_state(method_state)
        random.setstate(boundary_rng[0])
        torch.set_rng_state(boundary_rng[1])
        resumed = next_step(restored)
        self.assertEqual([key for key, _ in uninterrupted[2]],
                         [key for key, _ in resumed[2]])
        for index in range(len(uninterrupted)):
            left, right = uninterrupted[index], resumed[index]
            if index == 2:
                left = [grad for _, grad in left]
                right = [grad for _, grad in right]
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(random.getstate(), uninterrupted_rng[0])
        torch.testing.assert_close(torch.get_rng_state(), uninterrupted_rng[1], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
