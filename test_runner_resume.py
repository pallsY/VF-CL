import json
import copy
import inspect
import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import runner

from bic_calibration import TaskAffineCalibrator
from metrics import MetricsTracker
from models import TopModel
from cl_methods.proto_fedspace import ProtoFedSpaceCL
from cl_methods.target import TARGETCL
from ul_methods.retrain import RetrainUL
from test_resume_method_state import ContinuationFixture, assert_continuation_equal
from runner import (
    _atomic_json_dump,
    _atomic_torch_save,
    _load_resume_checkpoint,
    _save_cil_checkpoint,
    run_experiment,
)


class Stateful:
    def __init__(self, value=0):
        self.value = value

    def get_state(self):
        return {"value": self.value}

    def load_state(self, state):
        self.value = state["value"]


class TaskState:
    def __init__(self):
        self.seen = []
        self.forgotten = []

    def advance_task(self, task_id):
        self.seen.append(task_id)

    def apply_unlearn(self, classes):
        self.forgotten.extend(classes)


class AdaptiveTrainer:
    def __init__(self, top_model):
        self.bottoms = []
        self.top_model = top_model

    def get_state(self):
        return {
            'bottoms': [],
            'top_model': copy.deepcopy(self.top_model.state_dict()),
        }


class RunnerResumeTests(unittest.TestCase):
    def test_erace_cil_to_retrain_ul_resets_exact_ce_scope_and_matches_resume(self):
        self.addCleanup(ContinuationFixture.restore_rng, ContinuationFixture.rng_state())

        class Dataset:
            x = torch.arange(48, dtype=torch.float32).reshape(12, 4) / 10
            y = torch.arange(6).repeat_interleave(2)

            def get_task_loaders(self, classes, **kwargs):
                selected = torch.isin(self.y, torch.tensor(classes))
                batches = [(self.x[selected], self.y[selected])]
                return batches, batches

            def get_forget_retain_loaders(self, forgotten, seen):
                retain = [c for c in seen if c not in forgotten]
                retain_train, retain_test = self.get_task_loaders(retain)
                forget_train, forget_test = self.get_task_loaders(forgotten)
                return dict(retain_train=retain_train, retain_test=retain_test,
                            forget_train=forget_train, forget_test=forget_test)

        class StopAfterRetrain(Exception):
            pass

        traces, methods = [], []
        original_factory = runner.get_cl_method

        def method_factory(name, trainer, args):
            method = original_factory(name, trainer, args)
            methods.append(method)
            return method

        class ObservedRetrain(RetrainUL):
            def unlearn(self, forgotten, retain_loader, forget_loader, **kwargs):
                # Run the real oracle, including fresh build_models and train_task.
                result = super().unlearn(forgotten, retain_loader, forget_loader, **kwargs)
                traces.append({
                    'scope': (self.trainer.ce_classes, self.trainer.ce_lo, self.trainer.ce_hi),
                    'labels': torch.cat([y for _, y in retain_loader]),
                    'history': result['history'],
                    'trainer': self.trainer.get_state(),
                    'method': methods[-1].get_state(),
                    'rng': runner._capture_rng_state(),
                })
                # Bound this test before unrelated UL attacks/evaluation/publication.
                raise StopAfterRetrain

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                output_dir=directory, resume_run_dir='', save_task_checkpoints=3,
                num_tasks=3, classes_per_task=2, num_classes=6, num_parties=2,
                seed=123, data='synthvfl', cl_method='er_ace', ul_method='retrain',
                unlearn_after_tasks=[1], unlearn_classes=[[2]],
                device='cpu', deterministic=0, num_workers=0,
                party_col_ranges=[(0, 2), (2, 4)], model_type='mlp', embed_dim=2,
                aggregation='sum', lr=0.03, momentum=0.0, weight_decay=0.0,
                epochs_per_task=1, batch_size=4, er_ace_buffer_size=8, er_ace_batch=2,
                replay_mode='prototype', task_ce_mode='method', bic_enabled=0,
                head_consolidation_enabled=0, party_kd_enabled=0,
            )
            with patch.object(runner, 'VFLDataset', return_value=Dataset()), \
                    patch.object(runner, 'get_cl_method', side_effect=method_factory), \
                    patch.object(runner, 'get_ul_method',
                                 side_effect=lambda name, trainer, args: ObservedRetrain(trainer, args)):
                for resumed in (False, True):
                    with self.subTest(resumed=resumed):
                        args.resume_run_dir = directory if resumed else ''
                        try:
                            with self.assertRaises(StopAfterRetrain):
                                run_experiment(args)
                        except ValueError as error:
                            self.fail(f'ER-ACE CIL scope leaked into real retained-label retraining: {error}')
                        self.assertEqual(traces[-1]['scope'], (None, 0, None))
                        self.assertEqual(set(traces[-1]['labels'].tolist()), {0, 1, 3})
                        methods[-1].before_task(2, [4, 5], [0, 1, 3, 4, 5])
                        self.assertEqual(methods[-1].trainer.ce_classes, [4, 5])
                checkpoint = torch.load(Path(directory) / 'checkpoints' / 'resume_latest.pt',
                                        map_location='cpu', weights_only=True)
                self.assertEqual(checkpoint['step'], 'event_1_CIL')
        self.assertEqual(len(traces), 2)
        for field in ('labels', 'history', 'trainer', 'method'):
            torch.testing.assert_close(traces[0][field], traces[1][field], rtol=0, atol=0)
        self.assertTrue(runner._checkpoint_values_equal(traces[0]['rng'], traces[1]['rng']))

    def _check_public_baseline_checkpoint_continuation(self, method_type):
        original_rng = ContinuationFixture.rng_state()
        self.addCleanup(ContinuationFixture.restore_rng, original_rng)
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        with tempfile.TemporaryDirectory() as output_dir:
            source = ContinuationFixture(method_type, output_dir)
            source.finish_first_task()
            tracker = MetricsTracker()
            tracker.record_task_accuracies('event_0_CIL', {'task_0': 0.5}, 0.5)
            forgotten = sorted(getattr(source.method, 'forgotten', []))
            step = 'event_1_UL' if forgotten else 'event_0_CIL'
            _save_cil_checkpoint(
                source.trainer, source.method, source.args, step, 0,
                [0, 1, 2], {0: [0, 1, 2]}, tracker_state=tracker.to_dict(),
                forgotten_classes=forgotten,
            )
            checkpoint = torch.load(
                Path(output_dir) / 'checkpoints' / 'resume_latest.pt',
                map_location='cpu', weights_only=True,
            )
            self.assertEqual(checkpoint['schema_version'], 4)
            self.assertLess(checkpoint['task_id'], source.args.num_tasks - 1)
            uninterrupted = source.next_task(self)

            # Fresh construction and all RNG streams deliberately diverge before
            # the real loader's candidate validation and final RNG restoration.
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            restored = ContinuationFixture(method_type, output_dir)
            task_state, restored_tracker = TaskState(), MetricsTracker()
            start, seen, history = _load_resume_checkpoint(
                restored.args, restored.trainer, restored.method, task_state,
                restored_tracker, TaskAffineCalibrator(),
            )
            self.assertEqual(start, 2 if forgotten else 1)
            self.assertEqual(seen, {0: [0, 1, 2]})
            self.assertEqual(task_state.seen, [0])
            self.assertEqual(task_state.forgotten, forgotten)
            self.assertEqual(history, [])
            self.assertEqual(restored_tracker.to_dict(), tracker.to_dict())
            resumed = restored.next_task(self)

            assert_continuation_equal(self, uninterrupted, resumed)

    def test_target_schema_v4_checkpoint_continues_identically(self):
        self._check_public_baseline_checkpoint_continuation(TARGETCL)

    def test_proto_fedspace_schema_v4_checkpoint_continues_identically(self):
        self._check_public_baseline_checkpoint_continuation(ProtoFedSpaceCL)

    def args(self, output_dir):
        return SimpleNamespace(
            output_dir=output_dir,
            resume_run_dir=output_dir,
            save_task_checkpoints=3,
            num_tasks=10,
            seed=44,
            data="cifar100",
            cl_method="er",
        )

    def test_mode_three_keeps_one_rolling_checkpoint_then_final(self):
        with tempfile.TemporaryDirectory() as output_dir:
            args = self.args(output_dir)
            tracker = MetricsTracker()
            tracker.record_task_accuracies(
                "event_0_CIL", {"task_0": 0.5}, 0.5
            )
            _save_cil_checkpoint(
                Stateful(3), Stateful(4), args, "event_0_CIL", 0,
                list(range(10)), {0: list(range(10))},
                tracker_state=tracker.to_dict(), bic_history=[{"task": 0}],
            )
            checkpoint_dir = Path(output_dir) / "checkpoints"
            self.assertTrue((checkpoint_dir / "resume_latest.pt").is_file())
            self.assertFalse((checkpoint_dir / "event_0_CIL.pt").exists())

            _save_cil_checkpoint(
                Stateful(5), Stateful(6), args, "event_9_CIL", 9,
                list(range(90, 100)), {9: list(range(90, 100))},
                tracker_state=tracker.to_dict(), bic_history=[{"task": 9}],
            )
            self.assertTrue((checkpoint_dir / "event_9_CIL.pt").is_file())
            self.assertFalse((checkpoint_dir / "resume_latest.pt").exists())

    def test_atomic_save_preserves_previous_checkpoint_when_write_fails(self):
        with tempfile.TemporaryDirectory() as output_dir:
            target = Path(output_dir) / "checkpoint.pt"
            torch.save({"value": "old"}, target)
            real_torch_save = torch.save

            def fail_after_write(value, path):
                real_torch_save({"value": "partial"}, path)
                raise RuntimeError("power cut")

            with patch("runner.torch.save", side_effect=fail_after_write):
                with self.assertRaisesRegex(RuntimeError, "power cut"):
                    _atomic_torch_save({"value": "new"}, target)

            self.assertEqual(torch.load(target, weights_only=False)["value"], "old")
            self.assertFalse(target.with_suffix(".pt.tmp").exists())

    def test_atomic_writes_sync_and_preserve_previous_results_on_failure(self):
        with tempfile.TemporaryDirectory() as output_dir:
            checkpoint = Path(output_dir) / "checkpoint.pt"
            with patch("runner.os.fsync", wraps=os.fsync) as sync:
                _atomic_torch_save({"value": "durable"}, checkpoint)
            self.assertGreaterEqual(sync.call_count, 2)

            target = Path(output_dir) / "results.json"
            target.write_text(json.dumps({"value": "old"}))

            def fail_after_write(value, handle, **kwargs):
                handle.write('{"value":')
                raise RuntimeError("power cut")

            with patch("runner.json.dump", side_effect=fail_after_write):
                with self.assertRaisesRegex(RuntimeError, "power cut"):
                    _atomic_json_dump({"value": "new"}, target)

            self.assertEqual(json.loads(target.read_text()), {"value": "old"})
            self.assertFalse(target.with_suffix(".json.tmp").exists())

    def test_main_experiment_uses_atomic_results_write(self):
        source = inspect.getsource(run_experiment)
        self.assertIn('_atomic_json_dump(final', source)
        self.assertNotIn("with open(os.path.join(args.output_dir,'results.json')", source)

    def test_resume_restores_task_model_method_metrics_bic_and_rng(self):
        with tempfile.TemporaryDirectory() as output_dir:
            args = self.args(output_dir)
            trainer = Stateful(11)
            method = Stateful(12)
            tracker = MetricsTracker()
            tracker.record_task_accuracies(
                "event_0_CIL", {"task_0": 0.25}, 0.25
            )
            calibrator = TaskAffineCalibrator()

            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            _save_cil_checkpoint(
                trainer, method, args, "event_0_CIL", 0,
                list(range(10)), {0: list(range(10))},
                bic_state=calibrator.state_dict(),
                tracker_state=tracker.to_dict(),
                bic_history=[{"step": "event_0_CIL"}],
            )
            expected_random = random.random()
            expected_numpy = np.random.rand()
            expected_torch = torch.rand(1)

            trainer.value = -1
            method.value = -2
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            restored_tracker = MetricsTracker()
            restored_calibrator = TaskAffineCalibrator()
            task_state = TaskState()

            start_idx, seen, history = _load_resume_checkpoint(
                args, trainer, method, task_state, restored_tracker,
                restored_calibrator,
            )

            self.assertEqual(start_idx, 1)
            self.assertEqual(seen, {0: list(range(10))})
            self.assertEqual(history, [{"step": "event_0_CIL"}])
            self.assertEqual(trainer.value, 11)
            self.assertEqual(method.value, 12)
            self.assertEqual(task_state.seen, [0])
            self.assertEqual(
                restored_tracker.to_dict()["task_acc_history"],
                tracker.to_dict()["task_acc_history"],
            )
            self.assertEqual(random.random(), expected_random)
            self.assertEqual(np.random.rand(), expected_numpy)
            torch.testing.assert_close(torch.rand(1), expected_torch)

    def test_latest_adaptive_checkpoint_allows_only_dynamic_top_buffers(self):
        installed = TopModel(3, 4, cosine=False)
        installed.set_adaptive_endpoint(1.0, [0, 1, 2, 3])
        checkpoint = {
            'schema_version': 3,
            'trainer_state': AdaptiveTrainer(installed).get_state(),
            'protocol': {
                'head_consolidation_enabled': 1,
                'head_consolidation_mode': 'adaptive_dual_branch',
                'num_parties': 0,
            },
            'cl_state': {'value': 7},
            'bic_state': None,
            'tracker_state': MetricsTracker().to_dict(),
            'event_idx': 1,
            'seen_task_classes': {0: [0, 1], 1: [2, 3]},
            'forgotten_classes': [],
        }
        fresh = AdaptiveTrainer(TopModel(3, 4, cosine=False))
        with patch('runner._validate_resume_checkpoint_payload'):
            decoded = runner._validate_resume_candidate_loadability(
                checkpoint, object(), fresh, Stateful(), TaskState(),
                TaskAffineCalibrator(), MetricsTracker(),
            )
        self.assertIs(decoded, checkpoint)

        malformed = copy.deepcopy(checkpoint)
        malformed['trainer_state']['top_model']['classifier.weight'] = \
            torch.zeros(5, 3)
        with patch('runner._validate_resume_checkpoint_payload'):
            with self.assertRaisesRegex(ValueError, 'trainer structure'):
                runner._validate_resume_candidate_loadability(
                    malformed, object(), fresh, Stateful(), TaskState(),
                    TaskAffineCalibrator(), MetricsTracker(),
                )


if __name__ == "__main__":
    unittest.main()
