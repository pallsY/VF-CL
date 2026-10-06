import hashlib
import json
import random
import unittest
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from adaptive_head_consolidation import (
    BIAS_BRANCH_CONFIG,
    FULL_BRANCH_CONFIG,
    SOLVER_MAX_ITERATIONS,
    SOLVER_TOLERANCE,
)
from cl_methods.der_pp import DERppCL
from cl_methods.er import ERCL
from cl_methods.ewc import EWCCL
from cl_methods.proto_evolve import ProtoEvolveCL
from cl_methods.proto_fedspace import ProtoFedSpaceCL
from cl_methods.target import ConditionalGenerator, TARGETCL
from models import TopModel
from vfl_trainer import VFLTrainer


class ContinuationFixture:
    """Tiny real task boundary shared by method and runner resume tests."""

    def __init__(self, method_type, output_dir=''):
        self.args = SimpleNamespace(
            num_parties=2, num_classes=5, num_tasks=2, device='cpu',
            data='synthvfl', party_col_ranges=[(0, 2), (2, 4)],
            aggregation='sum', lr=0.03, momentum=0.0, weight_decay=0.0,
            epochs_per_task=1, proto_aug_weight=0.7, repr_loss_weight=0.4,
            seed=123, cl_method=('target' if method_type is TARGETCL
                                 else 'proto_fedspace'),
            output_dir=output_dir, resume_run_dir=output_dir,
            save_task_checkpoints=3,
        )
        self.trainer = VFLTrainer(
            [nn.Linear(2, 2), nn.Linear(2, 2)],
            TopModel(2, 3, cosine=False), self.args,
        )
        self.method = method_type(self.trainer, self.args)
        self.x = torch.tensor([
            [0.1, 0.7, -0.4, 0.2], [0.8, -0.3, 0.5, 1.2],
            [-0.7, 0.4, 1.1, -0.8], [1.3, 0.2, -0.9, 0.6],
            [0.4, -1.1, 0.3, 0.9], [-0.2, 1.4, 0.7, -0.5],
        ])

    @staticmethod
    def rng_state():
        return random.getstate(), np.random.get_state(), torch.get_rng_state()

    @staticmethod
    def restore_rng(state):
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])

    def finish_first_task(self):
        loader = [(self.x, torch.tensor([0, 1, 2, 0, 1, 2]))]
        self.method.before_task(0, [0, 1, 2], [0, 1, 2])
        self.method.train_task(loader, 0)
        if isinstance(self.method, TARGETCL):
            # One real generator epoch is enough for a continuation contract test.
            self.method.generator_epochs = 1
            self.method.k0 = self.method.fim_alpha = 0
        self.method.after_task(loader, 0)
        if isinstance(self.method, TARGETCL):
            self.method.forgotten = {1}

    def next_task(self, test):
        method, trainer = self.method, self.trainer
        # Also verify Python RNG restoration; replay consumes NumPy and Torch.
        trace = {'rng_probe': (random.random(), np.random.rand(), torch.rand(2))}
        method.before_task(1, [3, 4], [0, 1, 2, 3, 4])
        before = trainer.get_state()
        trace['before'] = before
        trace['trainable'] = [
            {name: p.requires_grad for name, p in bottom.named_parameters()}
            for bottom in trainer.bottoms
        ]
        teachers = ([method.old_top] if isinstance(method, TARGETCL)
                    else method.old_bottoms)
        test.assertTrue(teachers)
        for teacher in teachers:
            test.assertFalse(teacher.training)
            test.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        trace['teachers'] = [deepcopy(model.state_dict()) for model in teachers]
        trace['losses'], trace['replay'], trace['gradients'] = [], [], []
        handles = []
        if isinstance(method, TARGETCL):
            trace['fim_masks'] = deepcopy(method.fim_masks)
            mask_values = [v for mask in method.fim_masks for v in mask.values()]
            for value in mask_values:
                test.assertIs(type(value), bool)
            test.assertIn(True, mask_values)
            test.assertIn(False, mask_values)
            for mask, trainable in zip(method.fim_masks, trace['trainable']):
                for name, active in trainable.items():
                    test.assertEqual(active, not mask.get(name, False))

            def capture_generator(tid):
                def capture(_module, inputs, output):
                    local, noise = inputs
                    labels = [method.task_classes[tid][c] for c in local.tolist()]
                    trace['replay'].append((labels, noise.detach().clone(),
                                            output.detach().clone()))
                return capture

            for tid, generator in method.generators.items():
                handles.append(generator.register_forward_hook(capture_generator(tid)))
            loss_fn = method._target_loss
        else:
            test.assertGreater(method.radius, 0.0)
            loss_fn = method._proto_repr_loss

        def capture_loss(bottoms, top, bx, by, loss_ce):
            replay = []
            if isinstance(method, ProtoFedSpaceCL):
                # Independent RNG copies predict the labels/noise without advancing
                # either global stream or replacing the real augmentation/loss.
                numpy_rng = np.random.RandomState()
                numpy_rng.set_state(np.random.get_state())
                torch_rng = torch.Generator().set_state(torch.get_rng_state())
                classes = list(method.protos)
                labels = [classes[numpy_rng.randint(len(classes))]
                          for _ in range(by.size(0))]
                centers = torch.stack([method.protos[c] for c in labels])
                expected = torch.stack([
                    method.protos[c] + torch.randn(method.protos[c].shape,
                                                   generator=torch_rng) * method.radius
                    for c in labels
                ])
                handle = top.register_forward_pre_hook(
                    lambda _module, inputs: replay.append(inputs[0].detach().clone()))
                try:
                    loss = loss_fn(bottoms, top, bx, by, loss_ce)
                finally:
                    handle.remove()
                test.assertEqual(len(replay), 1)
                torch.testing.assert_close(replay[0], expected, rtol=0, atol=0)
                test.assertFalse(torch.equal(replay[0], centers))
                trace['replay'].append((labels, replay[0]))
            else:
                loss = loss_fn(bottoms, top, bx, by, loss_ce)
            extra = (loss - loss_ce).detach().clone()
            test.assertGreater(extra.item(), 0.0)
            trace['losses'].append((loss.detach().clone(), extra))
            return loss

        for name, parameter in trainer.top_model.named_parameters():
            handles.append(parameter.register_hook(
                lambda grad, name=name: trace['gradients'].append((name, grad.clone()))))
        try:
            trainer.train_task(
                [(self.x, torch.tensor([3, 4, 3, 4, 3, 4])),
                 (self.x.flip(0), torch.tensor([4, 3, 4, 3, 4, 3]))],
                1, extra_loss_fn=capture_loss,
            )
        finally:
            for handle in handles:
                handle.remove()
        trace['after'] = trainer.get_state()
        test.assertFalse(torch.equal(
            before['top_model']['classifier.weight'],
            trace['after']['top_model']['classifier.weight']))
        for teacher, snapshot in zip(teachers, trace['teachers']):
            torch.testing.assert_close(teacher.state_dict(), snapshot, rtol=0, atol=0)
            test.assertTrue(all(p.grad is None for p in teacher.parameters()))
        if isinstance(method, TARGETCL):
            labels = [label for replay in trace['replay'] for label in replay[0]]
            test.assertEqual(len(labels), 2 * len(self.x))
            test.assertEqual(set(labels), {0, 2})
            for k, mask in enumerate(method.fim_masks):
                for name, frozen in mask.items():
                    if frozen:
                        torch.testing.assert_close(
                            trace['after']['bottoms'][k][name],
                            before['bottoms'][k][name], rtol=0, atol=0)
        return trace


def assert_continuation_equal(test, uninterrupted, restored):
    test.assertEqual(set(uninterrupted), set(restored))
    for key in uninterrupted:
        if key == 'gradients':
            test.assertEqual([name for name, _ in uninterrupted[key]],
                             [name for name, _ in restored[key]])
            left = [value for _, value in uninterrupted[key]]
            right = [value for _, value in restored[key]]
        else:
            left, right = uninterrupted[key], restored[key]
        torch.testing.assert_close(left, right, rtol=0, atol=0, msg=key)


class TinyTrainer:
    def __init__(self):
        self.bottoms = [nn.Linear(2, 2), nn.Linear(2, 2)]
        self.top_model = nn.Linear(2, 4)


class AdaptiveTinyTrainer(TinyTrainer):
    def __init__(self):
        super().__init__()
        self.top_model = TopModel(2, 4, cosine=False)


class ResumeMethodStateTests(unittest.TestCase):
    def _check_next_task_continuation(self, method_type):
        original_rng = ContinuationFixture.rng_state()
        self.addCleanup(ContinuationFixture.restore_rng, original_rng)
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        source = ContinuationFixture(method_type)
        source.finish_first_task()
        trainer_state = source.trainer.get_state()
        method_state = source.method.get_state()
        boundary_rng = ContinuationFixture.rng_state()
        uninterrupted = source.next_task(self)

        restored = ContinuationFixture(method_type)
        restored.trainer.load_state(trainer_state)
        restored.method.load_state(method_state)
        ContinuationFixture.restore_rng(boundary_rng)
        resumed = restored.next_task(self)

        assert_continuation_equal(self, uninterrupted, resumed)

    def test_target_next_task_matches_restored_state(self):
        self._check_next_task_continuation(TARGETCL)

    def test_proto_fedspace_next_task_matches_restored_state(self):
        self._check_next_task_continuation(ProtoFedSpaceCL)

    @staticmethod
    def _proto_fedspace_source_and_state(formal=False):
        args = SimpleNamespace(
            num_classes=4, device='cpu', proto_aug_weight=0.7,
            formal_deferred_evaluation=formal,
        )
        source = ProtoFedSpaceCL(TinyTrainer(), args)
        source.protos = {
            0: torch.tensor([0.25, -0.5]),
            3: torch.tensor([1.5, 0.75]),
        }
        source.radius = np.float64(0.625)
        return source, {'protos': deepcopy(source.protos), 'radius': 0.625}

    def test_proto_fedspace_round_trip_restores_nonzero_radius_loss(self):
        source, expected = self._proto_fedspace_source_and_state()
        state = source.get_state()
        self.assertIs(type(state['radius']), float)
        for class_id in source.protos:
            self.assertNotEqual(
                source.protos[class_id].data_ptr(),
                state['protos'][class_id].data_ptr(),
            )
        restored = ProtoFedSpaceCL(TinyTrainer(), source.args)
        restored.trainer.top_model.load_state_dict(source.trainer.top_model.state_dict())
        restored.load_state(state)

        batch_y = torch.tensor([0, 1, 2, 3])
        batch_x = torch.empty(4, 4)
        np.random.seed(123)
        torch.manual_seed(456)
        source_loss = source._proto_repr_loss(
            [], source.trainer.top_model, batch_x, batch_y,
        )
        np.random.seed(123)
        torch.manual_seed(456)
        restored_loss = restored._proto_repr_loss(
            [], restored.trainer.top_model, batch_x, batch_y,
        )

        torch.testing.assert_close(source_loss, restored_loss, rtol=0, atol=0)
        self.assertEqual(set(state), {'protos', 'radius'})
        self.assertEqual(restored.radius, expected['radius'])
        self.assertEqual(set(restored.protos), set(expected['protos']))
        for class_id, proto in expected['protos'].items():
            torch.testing.assert_close(restored.protos[class_id], proto, rtol=0, atol=0)
            self.assertNotEqual(
                restored.protos[class_id].data_ptr(), state['protos'][class_id].data_ptr(),
            )
        state['protos'][0].zero_()
        torch.testing.assert_close(source.protos[0], expected['protos'][0], rtol=0, atol=0)
        torch.testing.assert_close(restored.protos[0], expected['protos'][0], rtol=0, atol=0)

    def test_proto_fedspace_rejects_malformed_state_atomically(self):
        _, state = self._proto_fedspace_source_and_state()
        corruptions = {
            'not a mapping': None,
            'extra key': lambda s: s.update(unexpected=1),
            'missing protos': lambda s: s.pop('protos'),
            'missing radius with prototypes': lambda s: s.pop('radius'),
            'negative radius': lambda s: s.update(radius=-0.1),
            'nan radius': lambda s: s.update(radius=float('nan')),
            'infinite radius': lambda s: s.update(radius=float('inf')),
            'boolean radius': lambda s: s.update(radius=True),
            'huge integer radius': lambda s: s.update(radius=10 ** 10000),
            'lossy integer radius': lambda s: s.update(radius=2 ** 53 + 1),
            'longdouble overflow radius': lambda s: s.update(
                radius=np.longdouble(np.finfo(np.float64).max) * 2),
            'longdouble precision radius': lambda s: s.update(
                radius=np.longdouble('0.100000000000000000001')),
            'string class': lambda s: s['protos'].update({'1': torch.zeros(2)}),
            'boolean class': lambda s: (
                s['protos'].pop(0), s['protos'].update({True: torch.zeros(2)})),
            'negative class': lambda s: s['protos'].update({-1: torch.zeros(2)}),
            'class out of range': lambda s: s['protos'].update({4: torch.zeros(2)}),
            'prototype type': lambda s: s['protos'].update({0: [0.0, 1.0]}),
            'prototype rank': lambda s: s['protos'].update({0: torch.zeros(1, 2)}),
            'prototype width': lambda s: s['protos'].update({0: torch.zeros(3)}),
            'prototype dtype': lambda s: s['protos'].update(
                {0: torch.zeros(2, dtype=torch.int64)}),
            'nan prototype': lambda s: s['protos'][0].fill_(float('nan')),
            'infinite prototype': lambda s: s['protos'][3].fill_(float('inf')),
        }
        for label, corrupt in corruptions.items():
            with self.subTest(label=label):
                bad = deepcopy(state)
                if corrupt is None:
                    bad = None
                else:
                    corrupt(bad)
                receiver, _ = self._proto_fedspace_source_and_state()
                before_protos = receiver.protos
                before_proto_objects = dict(receiver.protos)
                before_radius = receiver.radius
                before_state = receiver.get_state()
                with self.assertRaisesRegex(ValueError, 'ProtoFedSpace'):
                    receiver.load_state(bad)
                self.assertIs(receiver.protos, before_protos)
                self.assertIs(receiver.radius, before_radius)
                self.assertEqual(set(receiver.protos), set(before_proto_objects))
                for class_id, proto in before_state['protos'].items():
                    self.assertIs(receiver.protos[class_id], before_proto_objects[class_id])
                    torch.testing.assert_close(
                        receiver.protos[class_id], proto, rtol=0, atol=0,
                    )

    def test_proto_fedspace_rejects_classifier_dtype_mismatch_atomically(self):
        source, _ = self._proto_fedspace_source_and_state()
        for trainer_type in (TinyTrainer, AdaptiveTinyTrainer):
            with self.subTest(top=trainer_type.__name__):
                receiver = ProtoFedSpaceCL(trainer_type(), source.args)
                receiver.protos = deepcopy(source.protos)
                receiver.radius = source.radius
                state = receiver.get_state()
                state['protos'][0] = state['protos'][0].to(torch.float64)
                before_protos = receiver.protos
                before_proto_objects = dict(receiver.protos)
                before_radius = receiver.radius

                with self.assertRaisesRegex(ValueError, 'ProtoFedSpace.*dtype'):
                    receiver.load_state(state)

                self.assertIs(receiver.protos, before_protos)
                self.assertIs(receiver.radius, before_radius)
                self.assertEqual(set(receiver.protos), set(before_proto_objects))
                for class_id, proto in before_proto_objects.items():
                    self.assertIs(receiver.protos[class_id], proto)
                    self.assertEqual(proto.dtype, receiver.trainer.top_model
                                     .classifier.weight.dtype
                                     if hasattr(receiver.trainer.top_model, 'classifier')
                                     else receiver.trainer.top_model.weight.dtype)
                np.random.seed(123)
                torch.manual_seed(456)
                loss = receiver._proto_repr_loss(
                    [], receiver.trainer.top_model, torch.empty(2, 4),
                    torch.tensor([0, 3]),
                )
                self.assertTrue(torch.isfinite(loss).item())

    def test_proto_fedspace_legacy_state_fails_closed_for_continuation(self):
        source, _ = self._proto_fedspace_source_and_state()
        for state in ({}, {'protos': {}}):
            with self.subTest(mode='nonformal pristine', state=state):
                receiver = ProtoFedSpaceCL(TinyTrainer(), source.args)
                receiver.load_state(state)
                self.assertEqual(receiver.protos, {})
                self.assertEqual(receiver.radius, 0.0)
            with self.subTest(mode='nonformal continuation', state=state):
                receiver, _ = self._proto_fedspace_source_and_state()
                before_protos = receiver.protos
                before_proto_objects = dict(receiver.protos)
                before_radius = receiver.radius
                with self.assertRaisesRegex(ValueError, 'ProtoFedSpace'):
                    receiver.load_state(state)
                self.assertIs(receiver.protos, before_protos)
                self.assertIs(receiver.radius, before_radius)
                self.assertEqual(set(receiver.protos), set(before_proto_objects))
                for class_id, proto in before_proto_objects.items():
                    self.assertIs(receiver.protos[class_id], proto)
            with self.subTest(mode='formal', state=state):
                receiver, _ = self._proto_fedspace_source_and_state(formal=True)
                with self.assertRaisesRegex(ValueError, 'ProtoFedSpace'):
                    receiver.load_state(state)

    def _target_source_and_state(self):
        args = SimpleNamespace(num_parties=2, num_classes=6, device='cpu')
        source = TARGETCL(TinyTrainer(), args)
        source.task_classes = {0: [2, 0], 1: [1, 3, 5]}
        source.forgotten = {0, 3}
        source.fim_masks = [
            {'weight': True, 'bias': False}, {'bias': True},
        ]
        for tid, classes in source.task_classes.items():
            gen = ConditionalGenerator(len(classes), 2, 3 + tid, 5 + tid)
            with torch.no_grad():
                for index, param in enumerate(gen.parameters()):
                    param.copy_(torch.arange(param.numel()).reshape_as(param)
                                * 0.001 + tid + index)
            source.generators[tid] = gen
        # Explicit fixture also exercises the loader before get_state is fixed.
        state = {
            'task_classes': deepcopy(source.task_classes),
            'forgotten': sorted(source.forgotten),
            'fim_masks': deepcopy(source.fim_masks),
            'generators': {
                tid: {
                    'num_classes': gen.num_classes, 'embed_dim': gen.embed_dim,
                    'noise_dim': gen.noise_dim, 'hidden': 5 + tid,
                    'state_dict': deepcopy(gen.state_dict()),
                } for tid, gen in source.generators.items()
            },
        }
        return source, state

    def test_target_state_includes_all_continuation_fields(self):
        source, expected = self._target_source_and_state()
        state = source.get_state()
        self.assertEqual(set(state), set(expected))
        for field in ('task_classes', 'forgotten', 'fim_masks'):
            self.assertEqual(state[field], expected[field])
        for tid, gen in source.generators.items():
            self.assertEqual(gen.hidden, expected['generators'][tid]['hidden'])
            record = state['generators'][tid]
            for key in ('num_classes', 'embed_dim', 'noise_dim', 'hidden'):
                self.assertEqual(record[key], expected['generators'][tid][key])
            for name, value in record['state_dict'].items():
                self.assertEqual(value.device.type, 'cpu')
                self.assertFalse(value.requires_grad)
                torch.testing.assert_close(value, gen.state_dict()[name], rtol=0, atol=0)
                self.assertNotEqual(value.data_ptr(), gen.state_dict()[name].data_ptr())
        state['task_classes'][0].append(4)
        state['fim_masks'][0]['weight'] = False
        state['forgotten'].append(5)
        self.assertEqual(source.task_classes, expected['task_classes'])
        self.assertEqual(source.fim_masks, expected['fim_masks'])
        self.assertEqual(source.forgotten, {0, 3})

    def test_target_round_trip_restores_exact_frozen_generators(self):
        source, _ = self._target_source_and_state()
        state = source.get_state()
        restored = TARGETCL(TinyTrainer(), source.args)
        restored.load_state(state)
        self.assertEqual(restored.task_classes, source.task_classes)
        self.assertEqual(restored.forgotten, source.forgotten)
        self.assertEqual(restored.fim_masks, source.fim_masks)
        self.assertEqual(set(restored.generators), set(source.generators))
        for tid, gen in restored.generators.items():
            self.assertFalse(gen.training)
            self.assertTrue(all(not p.requires_grad for p in gen.parameters()))
            self.assertTrue(all(p.device.type == 'cpu' for p in gen.parameters()))
            for name, value in gen.state_dict().items():
                torch.testing.assert_close(value, source.generators[tid].state_dict()[name],
                                           rtol=0, atol=0)
            labels = torch.arange(gen.num_classes)
            noise = torch.arange(gen.num_classes * gen.noise_dim, dtype=torch.float32)
            noise = noise.reshape(gen.num_classes, gen.noise_dim)
            torch.testing.assert_close(gen(labels, noise), source.generators[tid](labels, noise),
                                       rtol=0, atol=0)
        state['task_classes'][0].append(4)
        state['fim_masks'][0]['weight'] = False
        state['generators'][0]['state_dict']['class_emb.weight'].zero_()
        self.assertEqual(restored.task_classes, source.task_classes)
        self.assertEqual(restored.fim_masks, source.fim_masks)
        torch.testing.assert_close(restored.generators[0].class_emb.weight,
                                   source.generators[0].class_emb.weight, rtol=0, atol=0)

    def test_target_rejects_malformed_state_without_mutating_receiver(self):
        source, state = self._target_source_and_state()
        corruptions = {
            'not a mapping': lambda s: None,
            'extra key': lambda s: dict(s, unexpected=1),
            'task mismatch': lambda s: s['generators'].pop(1),
            'task id type': lambda s: s['task_classes'].update({'2': [4]}),
            'class id type': lambda s: s['task_classes'][0].__setitem__(0, True),
            'duplicate class': lambda s: s['task_classes'][0].append(2),
            'class in two tasks': lambda s: s['task_classes'][1].__setitem__(0, 2),
            'class out of range': lambda s: s['task_classes'][0].__setitem__(0, 6),
            'empty classes': lambda s: s['task_classes'].__setitem__(0, []),
            'class count': lambda s: s['generators'][0].update(num_classes=3),
            'metadata type': lambda s: s['generators'][0].update(hidden=True),
            'metadata zero': lambda s: s['generators'][0].update(noise_dim=0),
            'metadata extra': lambda s: s['generators'][0].update(extra=1),
            'metadata missing': lambda s: s['generators'][0].pop('hidden'),
            'state dict missing': lambda s: s['generators'][0]['state_dict'].pop('net.0.bias'),
            'wrong shape': lambda s: s['generators'][1]['state_dict'].update(
                {'net.0.bias': torch.zeros(1)}),
            'nonfinite': lambda s: s['generators'][1]['state_dict']['net.0.bias'].fill_(float('nan')),
            'wrong dtype': lambda s: s['generators'][0]['state_dict'].update(
                {'net.0.bias': torch.zeros(5, dtype=torch.int64)}),
            'mask count': lambda s: s.update(fim_masks=[{}]),
            'mask value': lambda s: s['fim_masks'][0].update(weight=1),
            'mask name': lambda s: s['fim_masks'][0].update(unknown=True),
            'forgotten type': lambda s: s.update(forgotten=[True]),
            'forgotten unknown': lambda s: s.update(forgotten=[4]),
            'forgotten duplicate': lambda s: s.update(forgotten=[0, 0]),
        }
        for missing in state:
            corruptions['missing ' + missing] = lambda s, key=missing: s.pop(key)
        for label, corrupt in corruptions.items():
            with self.subTest(label=label):
                bad = deepcopy(state)
                if label == 'not a mapping':
                    bad = None
                elif label == 'extra key':
                    bad = corrupt(bad)
                else:
                    corrupt(bad)
                receiver, _ = self._target_source_and_state()
                before = {name: getattr(receiver, name) for name in state}
                before_state = receiver.get_state()
                with self.assertRaisesRegex(ValueError, 'TARGET'):
                    receiver.load_state(bad)
                for name, value in before.items():
                    self.assertIs(getattr(receiver, name), value)
                after_state = receiver.get_state()
                for name in ('task_classes', 'forgotten', 'fim_masks'):
                    self.assertEqual(after_state[name], before_state[name])
                for tid, gen in before_state['generators'].items():
                    for name, value in gen['state_dict'].items():
                        torch.testing.assert_close(
                            after_state['generators'][tid]['state_dict'][name],
                            value, rtol=0, atol=0,
                        )

    def test_target_empty_initial_state_round_trip_and_legacy(self):
        args = SimpleNamespace(num_parties=2, num_classes=6, device='cpu')
        for state in ({}, {'task_classes': {}}, TARGETCL(TinyTrainer(), args).get_state()):
            receiver = TARGETCL(TinyTrainer(), args)
            receiver.load_state(state)
            self.assertEqual(receiver.generators, {})
            self.assertEqual(receiver.task_classes, {})
            self.assertEqual(receiver.forgotten, set())
            self.assertEqual(receiver.fim_masks, [{}, {}])

    def test_target_rejects_generator_width_incompatible_with_top_atomically(self):
        source, state = self._target_source_and_state()
        source.args.embed_dim = 3  # The actual top input, not this arg, is authoritative.
        bad = deepcopy(state)
        wrong = ConditionalGenerator(3, embed_dim=3, noise_dim=4, hidden=6)
        bad['generators'][1].update(embed_dim=3, state_dict=wrong.state_dict())
        for trainer_type in (TinyTrainer, AdaptiveTinyTrainer):
            with self.subTest(top=trainer_type.__name__):
                receiver = TARGETCL(trainer_type(), source.args)
                receiver.load_state(state)
                before = {name: getattr(receiver, name) for name in state}
                before_state = receiver.get_state()
                with self.assertRaisesRegex(ValueError, 'TARGET.*embed'):
                    receiver.load_state(bad)
                for name, value in before.items():
                    self.assertIs(getattr(receiver, name), value)
                after_state = receiver.get_state()
                for name in ('task_classes', 'forgotten', 'fim_masks'):
                    self.assertEqual(after_state[name], before_state[name])
                for tid, gen in before_state['generators'].items():
                    for name, value in gen['state_dict'].items():
                        torch.testing.assert_close(
                            after_state['generators'][tid]['state_dict'][name],
                            value, rtol=0, atol=0,
                        )

    def test_target_formal_rejects_even_empty_legacy_state(self):
        args = SimpleNamespace(num_parties=2, num_classes=6, device='cpu',
                               formal_deferred_evaluation=True)
        for state in ({}, {'task_classes': {}}, {'task_classes': {0: [0]}}):
            with self.subTest(state=state):
                with self.assertRaisesRegex(ValueError, 'TARGET'):
                    TARGETCL(TinyTrainer(), args).load_state(state)

    def test_target_nonformal_rejects_nonempty_legacy_state(self):
        args = SimpleNamespace(num_parties=2, num_classes=6, device='cpu')
        with self.assertRaisesRegex(ValueError, 'TARGET'):
            TARGETCL(TinyTrainer(), args).load_state({'task_classes': {0: [0]}})

    @staticmethod
    def _primary_gate(g=0.375):
        return {
            'gate_rule': 'class_balanced',
            'is_primary': True,
            'g': float(g),
            'boundary_derivatives': [-1.0, 1.0],
            'final_interval': [float(g), float(g)],
            'final_interval_width': 0.0,
            'iterations': 0,
            'converged': True,
            'tolerance': SOLVER_TOLERANCE,
            'max_iterations': SOLVER_MAX_ITERATIONS,
            'full_branch_nll': 1.0,
            'bias_branch_nll': 1.5,
            'mixture_nll': 0.75,
        }

    @staticmethod
    def _validation_manifest():
        ordered = [5, 9]
        return {
            'dataset': 'fixture-train',
            'seed': 20260814,
            'per_class': 1,
            'by_class': {'0': [5], '1': [9]},
            'ordered_indices': ordered,
            'sha256': hashlib.sha256(json.dumps(
                ordered, separators=(',', ':')
            ).encode('utf-8')).hexdigest(),
        }

    @staticmethod
    def _adaptive_args():
        return SimpleNamespace(
            num_parties=2, num_classes=4, num_tasks=2,
            dep_tracking_enabled=0,
            party_kd_enabled=0, party_kd_mode='uniform',
            party_proto_enabled=0, party_proto_mode='uniform',
            seed=42, device='cpu', batch_size=2,
            head_consolidation_enabled=1,
            head_consolidation_schedule='final',
            head_consolidation_mode='adaptive_dual_branch',
        )

    def _adaptive_source_and_state(self):
        source = ProtoEvolveCL(AdaptiveTinyTrainer(), self._adaptive_args())
        full_weight = source.trainer.top_model.classifier.weight.detach()[[0, 1]].clone()
        full_bias = source.trainer.top_model.classifier.bias.detach()[[0, 1]].clone()
        source.trainer.top_model.set_adaptive_mixture(
            full_weight, full_bias, 0.375, [0, 1],
        )
        manifest = self._validation_manifest()
        source.head_validation_sha256 = manifest['sha256']
        source.head_consolidation_history = [{
            'method_version': 1,
            'pre_head_sha256': 'a' * 64,
            'candidate_hashes': {
                'pre': 'a' * 64,
                'full': 'b' * 64,
                'bias': 'c' * 64,
            },
            'candidate_configs': {
                'full': deepcopy(FULL_BRANCH_CONFIG),
                'bias': deepcopy(BIAS_BRANCH_CONFIG),
            },
            'gate': self._primary_gate(),
            'validation_manifest': manifest,
            'ordered_classes': [0, 1],
            'task_id': 1,
            'task_boundary': 'event_1_CIL',
        }]
        return source, source.get_state()

    def test_er_round_trip_restores_samples_and_seen_counts(self):
        args = SimpleNamespace(er_per_class=2, er_batch=2, batch_size=2)
        source = ERCL(None, args)
        source.buffer.add_batch(
            torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
            torch.tensor([0, 1]),
        )
        source.buffer.seen_count[0] = 7

        restored = ERCL(None, args)
        restored.load_state(source.get_state())

        self.assertEqual(restored.buffer.seen_count, {0: 7, 1: 1})
        torch.testing.assert_close(restored.buffer.data[0][0], source.buffer.data[0][0])
        torch.testing.assert_close(restored.buffer.data[1][0], source.buffer.data[1][0])

    def test_der_round_trip_restores_examples_logits_and_counter(self):
        args = SimpleNamespace(
            der_alpha=0.5, der_beta=0.5, der_buffer_size=3,
            der_batch=2, batch_size=2, num_classes=10,
        )
        source = DERppCL(None, args)
        source.buffer.add(
            torch.arange(12, dtype=torch.float32).reshape(3, 4),
            torch.tensor(2),
            torch.arange(10, dtype=torch.float32),
        )
        source.buffer.num_seen = 9

        restored = DERppCL(None, args)
        restored.load_state(source.get_state())

        self.assertEqual(restored.buffer.num_seen, 9)
        self.assertEqual(restored.buffer.lb, [2])
        torch.testing.assert_close(restored.buffer.ex[0], source.buffer.ex[0])
        torch.testing.assert_close(restored.buffer.lg[0], source.buffer.lg[0])

    def test_ewc_round_trip_restores_fisher_and_anchor(self):
        args = SimpleNamespace(
            num_parties=1, ewc_lambda=1000.0, ewc_fisher_decay=0.9,
            ewc_fisher_samples=1024, lwf_ce_newonly=True,
            feat_distill_weight=0.0,
        )
        source = EWCCL(None, args)
        source.fisher = [{"weight": torch.tensor([0.25, 0.75])}]
        source.old_params = [{"weight": torch.tensor([1.0, 2.0])}]

        restored = EWCCL(None, args)
        restored.load_state(source.get_state())

        torch.testing.assert_close(
            restored.fisher[0]["weight"], source.fisher[0]["weight"]
        )
        torch.testing.assert_close(
            restored.old_params[0]["weight"], source.old_params[0]["weight"]
        )

    def test_proto_evolve_round_trip_restores_task_boundary_state(self):
        args = SimpleNamespace(
            num_parties=2, num_classes=4, dep_tracking_enabled=1,
            dep_tracking_momentum=0.9, party_kd_enabled=1,
            party_kd_mode='uniform', party_kd_lambda=0.5,
            party_proto_enabled=0, party_proto_mode='uniform',
            seed=42, device='cpu',
        )
        source = ProtoEvolveCL(TinyTrainer(), args)
        source.global_protos = {
            0: {'mean': torch.tensor([1.0, 2.0]),
                'std': torch.tensor([0.1, 0.2])}
        }
        source.prev_protos = {
            0: {'mean': torch.tensor([3.0, 4.0]),
                'std': torch.tensor([0.3, 0.4])}
        }
        source.fim_masks = [
            {'weight': True, 'bias': False},
            {'weight': False, 'bias': True},
        ]
        source.dep_tracker.contrib.copy_(torch.arange(8).reshape(4, 2))
        source.current_task_classes = [0, 1]
        source.class_party_contrib = {0: [0.2, 0.8]}
        source.class_party_weights = {0: [0.2, 0.8]}
        source.head_raw_replay = {
            0: torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        }
        source.head_consolidation_history = [
            {'task_id': 0, 'persistent_raw_example_count': 2}
        ]
        source._old_bottoms = [
            deepcopy(bottom) for bottom in source.trainer.bottoms
        ]
        source._old_top = deepcopy(source.trainer.top_model)

        restored = ProtoEvolveCL(TinyTrainer(), args)
        restored.load_state(source.get_state())

        self.assertEqual(restored.fim_masks, source.fim_masks)
        self.assertEqual(restored.current_task_classes, [0, 1])
        self.assertEqual(restored.prev_protos.keys(), source.prev_protos.keys())
        torch.testing.assert_close(
            restored.dep_tracker.contrib, source.dep_tracker.contrib
        )
        torch.testing.assert_close(
            restored.head_raw_replay[0], source.head_raw_replay[0]
        )
        self.assertEqual(
            restored.head_consolidation_history,
            source.head_consolidation_history,
        )
        self.assertIsNotNone(restored._old_bottoms)
        self.assertIsNotNone(restored._old_top)
        for current, teacher in zip(
                restored.trainer.bottoms, restored._old_bottoms):
            for current_param, teacher_param in zip(
                    current.parameters(), teacher.parameters()):
                torch.testing.assert_close(current_param, teacher_param)
                self.assertFalse(teacher_param.requires_grad)

    def test_adaptive_proto_evolve_round_trip_restores_exact_method_evidence(self):
        source, state = self._adaptive_source_and_state()
        restored = ProtoEvolveCL(AdaptiveTinyTrainer(), self._adaptive_args())
        restored.trainer.top_model.load_state_dict(
            source.trainer.top_model.state_dict()
        )

        restored.load_state(state)

        self.assertEqual(
            restored.head_consolidation_history,
            source.head_consolidation_history,
        )
        self.assertEqual(
            restored.head_validation_sha256,
            self._validation_manifest()['sha256'],
        )
        round_trip = restored.get_state()
        for key in (
                'adaptive_method_version', 'adaptive_top_version',
                'adaptive_class_order', 'adaptive_gate',
                'head_validation_sha256'):
            self.assertEqual(round_trip[key], state[key])

    def test_adaptive_resume_rejects_method_top_class_gate_and_hash_mismatch(self):
        source, state = self._adaptive_source_and_state()
        mismatches = {
            'method version': ('adaptive_method_version', 2),
            'top version': ('adaptive_top_version', 0),
            'class order': ('adaptive_class_order', [1, 2]),
            'gate': ('adaptive_gate', 0.5),
            'validation hash': ('head_validation_sha256', 'different'),
        }
        for label, (key, value) in mismatches.items():
            with self.subTest(label=label):
                restored = ProtoEvolveCL(
                    AdaptiveTinyTrainer(), self._adaptive_args()
                )
                restored.trainer.top_model.load_state_dict(
                    source.trainer.top_model.state_dict()
                )
                bad = deepcopy(state)
                bad[key] = value

                with self.assertRaisesRegex(ValueError, label):
                    restored.load_state(bad)

                self.assertEqual(restored.head_consolidation_history, [])

    def test_adaptive_resume_rejects_non_list_history_before_restoring_state(self):
        source = ProtoEvolveCL(AdaptiveTinyTrainer(), self._adaptive_args())
        state = source.get_state()
        for malformed in (None, tuple()):
            with self.subTest(history=malformed):
                restored = ProtoEvolveCL(
                    AdaptiveTinyTrainer(), self._adaptive_args()
                )
                bad = deepcopy(state)
                bad['head_consolidation_history'] = malformed

                with self.assertRaisesRegex(ValueError, 'history'):
                    restored.load_state(bad)

                self.assertEqual(restored.head_consolidation_history, [])

    def test_adaptive_resume_rejects_non_json_history_record(self):
        source, state = self._adaptive_source_and_state()
        restored = ProtoEvolveCL(AdaptiveTinyTrainer(), self._adaptive_args())
        restored.trainer.top_model.load_state_dict(
            source.trainer.top_model.state_dict()
        )
        bad = deepcopy(state)
        bad['head_consolidation_history'][0]['ordered_classes'] = torch.tensor(
            [0, 1]
        )

        with self.assertRaisesRegex(ValueError, 'history'):
            restored.load_state(bad)

        self.assertEqual(restored.head_consolidation_history, [])

    def test_adaptive_resume_rejects_coerced_identity_types(self):
        source, state = self._adaptive_source_and_state()
        record_mismatches = {
            'method_version': True,
            'task_id': 1.9,
            'ordered_classes': [0.0, 1.0],
        }
        for field, value in record_mismatches.items():
            with self.subTest(history_field=field):
                restored = ProtoEvolveCL(
                    AdaptiveTinyTrainer(), self._adaptive_args()
                )
                restored.trainer.top_model.load_state_dict(
                    source.trainer.top_model.state_dict()
                )
                bad = deepcopy(state)
                bad['head_consolidation_history'][0][field] = value
                with self.assertRaisesRegex(ValueError, 'history'):
                    restored.load_state(bad)

        metadata_mismatches = {
            'adaptive_method_version': (True, 'method version'),
            'adaptive_top_version': (True, 'top version'),
            'adaptive_class_order': ([0.0, 1.0], 'class order'),
        }
        for field, (value, message) in metadata_mismatches.items():
            with self.subTest(metadata_field=field):
                restored = ProtoEvolveCL(
                    AdaptiveTinyTrainer(), self._adaptive_args()
                )
                restored.trainer.top_model.load_state_dict(
                    source.trainer.top_model.state_dict()
                )
                bad = deepcopy(state)
                bad[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    restored.load_state(bad)

    def test_adaptive_resume_rejects_malformed_nested_evidence(self):
        source, state = self._adaptive_source_and_state()
        corruptions = {
            'hashes-none': lambda record: record.update(candidate_hashes=None),
            'hashes-incomplete': lambda record: record.update(
                candidate_hashes={'pre': 'a' * 64, 'full': 'b' * 64}
            ),
            'hashes-origin-mismatch': lambda record: record.update(
                candidate_hashes={
                    'pre': 'd' * 64,
                    'full': 'b' * 64,
                    'bias': 'c' * 64,
                }
            ),
            'hash-format': lambda record: record.update(
                pre_head_sha256='short',
                candidate_hashes={
                    'pre': 'short', 'full': 'full', 'bias': 'bias'
                },
            ),
            'configs-false': lambda record: record.update(candidate_configs=False),
            'configs-not-frozen': lambda record: record['candidate_configs'][
                'full'
            ].update(lr=True),
            'gate-none': lambda record: record.update(gate=None),
            'gate-incomplete': lambda record: record.update(
                gate={'gate_rule': 'class_balanced', 'g': 0.375}
            ),
            'gate-false': lambda record: record.update(gate=False),
            'manifest-none': lambda record: record.update(
                validation_manifest=None
            ),
            'manifest-incomplete': lambda record: record.update(
                validation_manifest={'sha256': 'd' * 64}
            ),
        }
        for label, corrupt in corruptions.items():
            with self.subTest(label=label):
                restored = ProtoEvolveCL(
                    AdaptiveTinyTrainer(), self._adaptive_args()
                )
                restored.trainer.top_model.load_state_dict(
                    source.trainer.top_model.state_dict()
                )
                bad = deepcopy(state)
                corrupt(bad['head_consolidation_history'][0])
                with self.assertRaisesRegex(ValueError, 'history'):
                    restored.load_state(bad)
                self.assertEqual(restored.head_consolidation_history, [])

        restored = ProtoEvolveCL(
            AdaptiveTinyTrainer(), self._adaptive_args()
        )
        restored.trainer.top_model.load_state_dict(
            source.trainer.top_model.state_dict()
        )
        bad = deepcopy(state)
        bad['head_consolidation_history'][0][
            'validation_manifest'
        ]['sha256'] = 'd' * 64
        bad['head_validation_sha256'] = 'd' * 64
        with self.assertRaisesRegex(ValueError, 'history'):
            restored.load_state(bad)

    def test_adaptive_resume_rejects_earlier_pending_with_its_own_ul(self):
        args = self._adaptive_args()
        args.unlearn_after_tasks = [0, 1]
        args.unlearn_classes = [[0], [1]]
        source = ProtoEvolveCL(AdaptiveTinyTrainer(), args)
        source._adaptive_pending_task_id = 1
        state = source.get_state()
        state['adaptive_pending_task_id'] = 0
        restored = ProtoEvolveCL(AdaptiveTinyTrainer(), args)
        before = deepcopy(restored.trainer.top_model.state_dict())

        with self.assertRaisesRegex(ValueError, 'pending task'):
            restored.load_state(state)

        self.assertIsNone(restored._adaptive_pending_task_id)
        self.assertEqual(restored.head_consolidation_history, [])
        for name, value in restored.trainer.top_model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
