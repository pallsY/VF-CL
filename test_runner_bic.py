import unittest
import json
import inspect
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from vfl_trainer import VFLTrainer
from bic_calibration import TaskAffineCalibrator
from runner import (
    _fit_and_evaluate_bic,
    _fit_and_evaluate_final_bic,
    _should_fit_bic,
    _save_cil_checkpoint,
    run_experiment,
)


class FlattenBottom(torch.nn.Module):
    def forward(self, value):
        return value.flatten(1)


class RunnerBiCTests(unittest.TestCase):
    def test_joint_final_mode_only_fits_last_task(self):
        args = SimpleNamespace(bic_enabled=1, bic_fit_mode='joint_final', num_tasks=10)
        self.assertFalse(_should_fit_bic(args, 8))
        self.assertTrue(_should_fit_bic(args, 9))
        args.bic_fit_mode = 'sequential'
        self.assertTrue(_should_fit_bic(args, 0))

    def test_joint_each_stage_fits_every_task(self):
        args = SimpleNamespace(
            bic_enabled=1, bic_fit_mode='joint_each_stage', num_tasks=10
        )
        self.assertTrue(_should_fit_bic(args, 0))
        self.assertTrue(_should_fit_bic(args, 8))
        self.assertTrue(_should_fit_bic(args, 9))

    def test_run_experiment_initializes_bic_state_before_timeline(self):
        setup = inspect.getsource(run_experiment).split('for idx, event in enumerate(timeline):')[0]
        self.assertIn('bic_calibrator = TaskAffineCalibrator()', setup)
        self.assertIn('bic_history = []', setup)

    def test_final_only_checkpoint_mode_skips_intermediate_tasks(self):
        class Stateful:
            def get_state(self):
                return {'value': 1}

        with tempfile.TemporaryDirectory() as output_dir:
            args = SimpleNamespace(
                output_dir=output_dir, save_task_checkpoints=2, num_tasks=10
            )
            _save_cil_checkpoint(Stateful(), Stateful(), args, 'event_0_CIL', 0,
                                 list(range(10)), {0: list(range(10))})
            self.assertFalse((Path(output_dir) / 'checkpoints').exists())
            _save_cil_checkpoint(Stateful(), Stateful(), args, 'event_9_CIL', 9,
                                 list(range(90, 100)), {9: list(range(90, 100))})
            self.assertTrue((Path(output_dir) / 'checkpoints' / 'event_9_CIL.pt').exists())

    def test_collect_logits_returns_cpu_logits_and_labels_in_loader_order(self):
        args = SimpleNamespace(
            device='cpu', aggregation='sum', data='cifar100', num_parties=2,
            party_widths=None, lr=0.1, momentum=0.0, weight_decay=0.0,
        )
        top = torch.nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            top.weight.copy_(torch.tensor([[1.0], [-1.0]]))
        trainer = VFLTrainer([FlattenBottom(), FlattenBottom()], top, args)
        inputs = torch.tensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]]])
        labels = torch.tensor([1, 0])
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=1, shuffle=False)

        logits, actual_labels = trainer.collect_logits(loader)

        self.assertEqual(logits.device.type, 'cpu')
        self.assertTrue(torch.equal(actual_labels, labels))
        self.assertTrue(torch.equal(logits, torch.tensor([[3.0, -3.0], [7.0, -7.0]])))

    def test_fit_and_evaluate_uses_calibration_loader_and_writes_safe_artifact(self):
        class FakeDataset:
            def __init__(self):
                self.fit_requests = []

            def get_calibration_loader(self, classes):
                self.fit_requests.append(tuple(classes))
                return ('calibration', tuple(classes))

            def get_task_loaders(self, classes, shuffle_train=False):
                return None, ('test', tuple(classes))

            def calibration_audit(self):
                return {'passed': True, 'test_used_for_fit': False}

        class FakeTrainer:
            def collect_logits(self, loader):
                kind, classes = loader
                if kind == 'calibration':
                    return torch.tensor([[3., 1., 5., 4.], [1., 3., 4., 5.],
                                         [0., 1., 5., 2.], [1., 0., 2., 5.]]), torch.tensor([0, 1, 2, 3])
                if classes == (0, 1):
                    return torch.tensor([[3., 1., 5., 4.], [1., 3., 4., 5.]]), torch.tensor([0, 1])
                return torch.tensor([[0., 1., 5., 2.], [1., 0., 2., 5.]]), torch.tensor([2, 3])

        with tempfile.TemporaryDirectory() as output_dir:
            args = SimpleNamespace(output_dir=output_dir, bic_lr=0.05, bic_steps=20)
            dataset = FakeDataset()
            record = _fit_and_evaluate_bic(
                FakeTrainer(), dataset, TaskAffineCalibrator(), args,
                'event_1_CIL', 1, [2, 3], {0: [0, 1], 1: [2, 3]},
            )
            artifact = json.loads(
                (Path(output_dir) / 'bic' / 'event_1_CIL.json').read_text()
            )

        self.assertEqual(dataset.fit_requests, [(0, 1, 2, 3)])
        self.assertTrue(record['calibration_audit']['passed'])
        self.assertNotIn('embedding', json.dumps(artifact).lower())
        self.assertNotIn('image', json.dumps(artifact).lower())

    def test_final_joint_fit_uses_all_tasks_once(self):
        class FakeDataset:
            def get_calibration_loader(self, classes):
                return ('calibration', tuple(classes))

            def get_task_loaders(self, classes, shuffle_train=False):
                return None, ('test', tuple(classes))

            def calibration_audit(self):
                return {'passed': True, 'test_used_for_fit': False}

        class FakeTrainer:
            def collect_logits(self, loader):
                kind, classes = loader
                logits = torch.tensor([[3., 1., 5., 4.], [1., 3., 4., 5.],
                                       [0., 1., 5., 2.], [1., 0., 2., 5.]])
                labels = torch.tensor([0, 1, 2, 3])
                if kind == 'test':
                    mask = torch.tensor([int(label) in classes for label in labels])
                    return logits[mask], labels[mask]
                return logits, labels

        with tempfile.TemporaryDirectory() as output_dir:
            args = SimpleNamespace(output_dir=output_dir, bic_lr=0.05, bic_steps=20)
            record = _fit_and_evaluate_final_bic(
                FakeTrainer(), FakeDataset(), TaskAffineCalibrator(), args, 'event_1_CIL',
                {0: [0, 1], 1: [2, 3]},
            )

        self.assertEqual(record['fit']['mode'], 'joint_alpha_beta')
        self.assertEqual(set(record['parameters']), {'0', '1'})
        self.assertLessEqual(record['task_il_max_abs_delta'], 1e-6)


if __name__ == '__main__':
    unittest.main()
