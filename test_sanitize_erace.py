"""ER-ACE raw-replay sanitization and real checkpoint continuation contracts."""
import random
import tempfile
import unittest
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from bic_calibration import TaskAffineCalibrator
from cl_methods.der_pp import DERppCL
from cl_methods.er_ace import ERAccCL
from cl_methods.sanitize import sanitize_cl_state
from data_utils import TaskManager
from metrics import MetricsTracker
from models import TopModel
from runner import (_capture_rng_state, _load_resume_checkpoint,
                    _restore_rng_state, _save_cil_checkpoint)
from vfl_trainer import VFLTrainer


class ERACESanitizeTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(_restore_rng_state, _capture_rng_state())
        random.seed(312)
        np.random.seed(312)
        torch.manual_seed(312)
        self.x = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10

    def method(self, directory='', forgotten=(0,)):
        args = SimpleNamespace(
            num_parties=2, num_classes=5, num_tasks=2, device='cpu',
            data='synthvfl', party_col_ranges=[(0, 2), (2, 4)],
            aggregation='sum', lr=0.03, momentum=0.0, weight_decay=0.0,
            epochs_per_task=1, er_ace_buffer_size=6, er_ace_batch=2,
            seed=312, cl_method='er_ace', output_dir=directory,
            resume_run_dir=directory, save_task_checkpoints=3,
            custom_tasks='0,1,2|3,4', unlearn_after_tasks=[0],
            unlearn_classes=[list(forgotten)],
        )
        trainer = VFLTrainer([nn.Linear(2, 2), nn.Linear(2, 2)],
                             TopModel(2, 5), args)
        return ERAccCL(trainer, args)

    def finish_first_cil(self, method):
        loader = [(self.x, torch.tensor([0, 1, 2, 0, 1, 2]))] * 2
        method.before_task(0, [0, 1, 2], [0, 1, 2])
        method.train_task(loader, 0)
        method.after_task(loader, 0)
        self.assertEqual(method.buffer.num_seen, 12)
        self.assertEqual(set(method.buffer.lb), {0, 1, 2})

    def sanitize(self, method, forgotten):
        try:
            return sanitize_cl_state(method, method.trainer, forgotten)
        except AttributeError as error:
            self.fail(f'ER-ACE raw replay must sanitize without logits: {error}')

    def assert_rng_equal(self, expected, actual=None):
        actual = _capture_rng_state() if actual is None else actual
        self.assertEqual(actual['python'], expected['python'])
        self.assertEqual(actual['numpy']['bit_generator'], expected['numpy']['bit_generator'])
        for key in ('keys', 'position', 'has_gauss', 'cached_gaussian'):
            torch.testing.assert_close(actual['numpy'][key], expected['numpy'][key],
                                       rtol=0, atol=0)
        for key in ('torch', 'cuda'):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_partial_purge_keeps_exact_examples_resets_count_and_preserves_rng(self):
        method = self.method()
        self.finish_first_cil(method)
        before, rng = method.get_state(), _capture_rng_state()
        old_examples = method.buffer.ex[:]
        keep = [i for i, label in enumerate(method.buffer.lb) if label != 0]
        self.assertEqual(self.sanitize(method, [0]),
                         [f'buffer:-{6 - len(keep)}ex'])
        self.assertFalse(hasattr(method.buffer, 'lg'))
        self.assertEqual(method.buffer.num_seen, len(keep))
        self.assertEqual(method.buffer.size(), min(6, method.buffer.num_seen))
        torch.testing.assert_close(method.get_state()['examples'],
                                   before['examples'][keep], rtol=0, atol=0)
        torch.testing.assert_close(method.get_state()['labels'],
                                   before['labels'][keep], rtol=0, atol=0)
        for retained, index in zip(method.buffer.ex, keep):
            self.assertIs(retained, old_examples[index])
        self.assert_rng_equal(rng)
        restored = self.method()
        restored.load_state(method.get_state())
        torch.testing.assert_close(restored.get_state(), method.get_state(),
                                   rtol=0, atol=0)

    def test_noop_purge_preserves_lists_history_and_rng(self):
        method = self.method()
        self.finish_first_cil(method)
        before, rng = method.get_state(), _capture_rng_state()
        old_ex, old_lb = method.buffer.ex, method.buffer.lb
        for forgotten in ([], [4]):
            self.assertEqual(self.sanitize(method, forgotten), [])
            self.assertIs(method.buffer.ex, old_ex)
            self.assertIs(method.buffer.lb, old_lb)
            torch.testing.assert_close(method.get_state(), before, rtol=0, atol=0)
            self.assert_rng_equal(rng)

    def test_invalid_reservoir_purge_fails_before_any_mutation(self):
        for corruption in ('unequal fields', 'inconsistent count'):
            with self.subTest(corruption=corruption):
                method = self.method()
                method.buffer.add_batch(self.x, torch.tensor([0, 1, 2, 0, 1, 2]))
                if corruption == 'unequal fields':
                    method.buffer.ex.pop()
                else:
                    method.buffer.num_seen = 2
                before, rng = deepcopy(method.get_state()), _capture_rng_state()
                old_ex, old_lb = method.buffer.ex, method.buffer.lb
                with self.assertRaisesRegex(ValueError, 'ER-ACE'):
                    self.sanitize(method, [2])
                self.assertIs(method.buffer.ex, old_ex)
                self.assertIs(method.buffer.lb, old_lb)
                torch.testing.assert_close(method.get_state(), before, rtol=0, atol=0)
                self.assert_rng_equal(rng)

    def next_cil(self, method, forgotten):
        buffer = method.buffer
        predictor = random.Random()
        predictor.setstate(random.getstate())
        indices = predictor.sample(range(buffer.size()), min(2, buffer.size()))
        expected_replay = ([buffer.ex[i] for i in indices],
                           [buffer.lb[i] for i in indices])
        self.assertFalse(set(expected_replay[1]) & set(forgotten))
        method.before_task(1, [3, 4], [0, 1, 2, 3, 4])
        before = method.trainer.get_state()
        forwards, gradients, handles = [], [], []
        handles.append(method.trainer.bottoms[0].register_forward_pre_hook(
            lambda _module, inputs: forwards.append(inputs[0].detach().clone())))
        for index, model in enumerate([*method.trainer.bottoms, method.trainer.top_model]):
            for name, parameter in model.named_parameters():
                handles.append(parameter.register_hook(
                    lambda grad, key=(index, name): gradients.append((key, grad.clone()))))
        try:
            history, _ = method.train_task(
                [(self.x, torch.tensor([3, 4, 3, 4, 3, 4]))], 1)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(len(forwards), 2 if indices else 1)
        if indices:
            torch.testing.assert_close(forwards[1],
                                       torch.stack(expected_replay[0])[:, :2],
                                       rtol=0, atol=0)
        self.assertFalse(set(buffer.lb) & set(forgotten))
        self.assertEqual(buffer.size(), min(buffer.buffer_size, buffer.num_seen))
        self.assertFalse(torch.equal(before['top_model']['classifier.weight'],
                                     method.trainer.get_state()['top_model']['classifier.weight']))
        return {
            'loss': history[0]['loss'], 'forwards': forwards,
            'gradient_names': [name for name, _ in gradients],
            'gradients': [grad for _, grad in gradients],
            'trainer': method.trainer.get_state(), 'buffer': method.get_state(),
            'rng': _capture_rng_state(),
        }

    def check_checkpoint_continuation(self, forgotten):
        with tempfile.TemporaryDirectory() as directory:
            source = self.method(directory, forgotten)
            self.finish_first_cil(source)
            self.sanitize(source, forgotten)
            purged = source.get_state()
            if set(forgotten) == {0, 1, 2}:
                self.assertIsNone(purged['examples'])
                self.assertEqual(purged['num_seen'], 0)
                self.assertEqual(purged['labels'].shape, (0,))
            tracker = MetricsTracker()
            _save_cil_checkpoint(
                source.trainer, source, source.args, 'event_1_UL', 0,
                [0, 1, 2], {0: [0, 1, 2]}, tracker_state=tracker.to_dict(),
                force=True, forgotten_classes=forgotten,
            )
            uninterrupted = self.next_cil(source, forgotten)
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            restored = self.method(directory, forgotten)
            task_manager = TaskManager(restored.args)
            start, seen, _ = _load_resume_checkpoint(
                restored.args, restored.trainer, restored, task_manager,
                MetricsTracker(), TaskAffineCalibrator(),
            )
            self.assertEqual(start, 2)
            self.assertEqual(seen, {0: [0, 1, 2]})
            self.assertEqual(task_manager.get_forgotten_classes(), list(forgotten))
            torch.testing.assert_close(restored.get_state(), purged, rtol=0, atol=0)
            resumed = self.next_cil(restored, forgotten)
            self.assertEqual(uninterrupted.pop('gradient_names'),
                             resumed.pop('gradient_names'))
            self.assert_rng_equal(uninterrupted.pop('rng'), resumed.pop('rng'))
            torch.testing.assert_close(resumed, uninterrupted, rtol=0, atol=0)
            if set(forgotten) == {0, 1, 2}:
                self.assertEqual(restored.buffer.size(), 6)
                self.assertEqual(restored.buffer.num_seen, 6)
                torch.testing.assert_close(restored.get_state()['examples'], self.x,
                                           rtol=0, atol=0)

    def test_cil_partial_purge_checkpoint_reload_next_cil_matches_uninterrupted(self):
        self.check_checkpoint_continuation([0])

    def test_cil_purge_all_checkpoint_reload_next_cil_refills_identically(self):
        self.check_checkpoint_continuation([0, 1, 2])

    def test_derpp_keeps_existing_logits_and_stream_count_semantics(self):
        method = DERppCL(None, SimpleNamespace(num_classes=5, batch_size=2,
                                               der_buffer_size=3))
        for index in range(6):
            method.buffer.add(self.x[index], index % 3, torch.full((5,), float(index)))
        before, rng = method.get_state(), _capture_rng_state()
        forgotten = [method.buffer.lb[0]]
        keep = [i for i, label in enumerate(method.buffer.lb) if label not in forgotten]
        self.assertEqual(sanitize_cl_state(method, None, forgotten),
                         [f'buffer:-{3 - len(keep)}ex'])
        after = method.get_state()
        self.assertEqual(after['num_seen'], before['num_seen'])
        for key in ('examples', 'labels', 'logits'):
            torch.testing.assert_close(after[key], before[key][keep], rtol=0, atol=0)
        self.assert_rng_equal(rng)


if __name__ == '__main__':
    unittest.main()
