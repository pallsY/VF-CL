"""Completed-stage method state is stricter than pristine construction state."""
import copy
from contextlib import ExitStack
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import runner
from bic_calibration import TaskAffineCalibrator
from cl_methods.er_ace import ERAccCL
from cl_methods.proto_fedspace import ProtoFedSpaceCL
from cl_methods.sanitize import sanitize_cl_state
from cl_methods.target import ConditionalGenerator, TARGETCL
from data_utils import TaskManager
from metrics import MetricsTracker
from test_resume_method_state import ContinuationFixture


class CompletedMethodHistoryTests(unittest.TestCase):
    METHODS = (TARGETCL, ProtoFedSpaceCL, ERAccCL)

    def setUp(self):
        self.addCleanup(ContinuationFixture.restore_rng, ContinuationFixture.rng_state())
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        contexts.enter_context(patch('torch.cuda.is_available', return_value=False))
        # Source authority is tested separately; these unit tests exercise the
        # unchanged full payload schema, strict reload and real timeline checks.
        contexts.enter_context(patch('adaptive_consolidation_audit._formal_source_provenance',
                                     return_value={'unit_test_source': 'lifecycle'}))

    def fixture(self, method_type, output_dir, formal=False, forgotten=()):
        fixture = ContinuationFixture(method_type, output_dir)
        args = fixture.args
        args.cl_method = {TARGETCL: 'target', ProtoFedSpaceCL: 'proto_fedspace',
                          ERAccCL: 'er_ace'}[method_type]
        args.num_tasks = 3
        args.custom_tasks = '0,1,2|3|4'
        args.unlearn_after_tasks = [0] if forgotten else []
        args.unlearn_classes = [list(forgotten)] if forgotten else []
        args.formal_deferred_evaluation = formal
        args.head_consolidation_enabled = 0
        args.head_consolidation_mode = 'full_classifier' if formal else None
        fixture.trainer.top_model.expand_classes(5, 'cpu')
        fixture.method = method_type(fixture.trainer, args)
        fixture.task_manager = TaskManager(args)
        fixture.tracker = MetricsTracker()
        fixture.calibrator = TaskAffineCalibrator()
        return fixture

    def payload(self, fixture, method_type, event_idx=0):
        timeline = fixture.task_manager.get_timeline()[:event_idx + 1]
        seen, forgotten = {}, []
        tracker = MetricsTracker()
        for idx, event in enumerate(timeline):
            step = f"event_{idx}_{event['type']}"
            if event['type'] == 'CIL':
                seen[event['task_id']] = event['new_classes']
            else:
                forgotten.extend(event['forget_classes'])
            if fixture.args.formal_deferred_evaluation:
                comm = fixture.trainer.get_comm_stats()
                record = dict(event_idx=idx, type=event['type'],
                              evaluation_deferred=True, comm=comm)
                if event['type'] == 'CIL':
                    record.update(task_id=event['task_id'],
                                  new_classes=event['new_classes'], train_time=0.0)
                else:
                    record['forget_classes'] = event['forget_classes']
                tracker.record_step(record)
                tracker.record_timing(step, 0.0)
                tracker.record_comm(step, comm)
        method = method_type(fixture.trainer, fixture.args)
        cached_forgotten = set(forgotten) if getattr(fixture.args, 'sanitize_cl_state', 1) else set()
        retained = sorted({c for classes in seen.values() for c in classes} - cached_forgotten)
        if method_type is TARGETCL:
            method.task_classes = copy.deepcopy(seen)
            method.forgotten = cached_forgotten
            method.generators = {
                tid: ConditionalGenerator(len(classes), embed_dim=2, noise_dim=3, hidden=4)
                for tid, classes in seen.items()
            }
        elif method_type is ProtoFedSpaceCL:
            method.protos = {c: torch.tensor([float(c), 0.5]) for c in retained}
            method.radius = 0.25
        elif retained:
            method.buffer.add_batch(torch.ones(len(retained), 4), torch.tensor(retained))
        task_id = max(seen)
        runner._save_cil_checkpoint(
            fixture.trainer, method, fixture.args, step, task_id, seen[task_id], seen,
            tracker_state=tracker.to_dict(), forgotten_classes=forgotten,
        )
        return torch.load(Path(fixture.args.output_dir) / 'checkpoints/resume_latest.pt',
                          map_location='cpu', weights_only=True)

    @staticmethod
    def snapshot(fixture):
        return copy.deepcopy({
            'trainer': fixture.trainer.get_state(), 'method': fixture.method.get_state(),
            'tracker': fixture.tracker.to_dict(), 'bic': fixture.calibrator.state_dict(),
            'tasks': {key: value for key, value in vars(fixture.task_manager).items()
                      if key != 'args'}, 'rng': runner._capture_rng_state(),
        })

    @staticmethod
    def preflight(fixture, payload):
        return runner._validate_resume_candidate_loadability(
            payload, fixture.args, fixture.trainer, fixture.method,
            fixture.task_manager, fixture.calibrator, fixture.tracker,
        )

    def assert_unchanged(self, fixture, before):
        for key, value in self.snapshot(fixture).items():
            self.assertTrue(runner._checkpoint_values_equal(before[key], value), key)

    def test_empty_latest_rejected_atomically_and_older_checkpoint_resumes(self):
        for formal in (False, True):
            for method_type in self.METHODS:
                with self.subTest(formal=formal, method=method_type.__name__), \
                        tempfile.TemporaryDirectory() as directory:
                    fixture = self.fixture(method_type, directory, formal)
                    older = self.payload(fixture, method_type)
                    latest = self.payload(fixture, method_type, 1)
                    latest['cl_state'] = runner._encode_checkpoint_value(fixture.method.get_state())
                    # Pristine state still loads; the completed-stage context is
                    # what makes exactly this same state invalid.
                    fixture.method.load_state(runner._decode_checkpoint_value(latest['cl_state']))
                    runner._validate_resume_checkpoint_payload(latest, fixture.args)
                    before = self.snapshot(fixture)
                    with self.assertRaisesRegex(ValueError, 'TARGET|ProtoFedSpace|ER-ACE'):
                        self.preflight(fixture, latest)
                    self.assert_unchanged(fixture, before)
                    rolling = Path(directory) / 'checkpoints/resume_latest.pt'
                    torch.save(latest, rolling)
                    with self.assertRaisesRegex(RuntimeError, 'no valid resume checkpoint'):
                        runner._load_resume_checkpoint(
                            fixture.args, fixture.trainer, fixture.method,
                            fixture.task_manager, fixture.tracker, fixture.calibrator)
                    self.assert_unchanged(fixture, before)
                    torch.save(older, Path(directory) / 'checkpoints/event_0_CIL.pt')
                    # No external publication is part of this loader unit test.
                    with patch.object(runner, '_recover_adaptive_cil_snapshot'):
                        result = runner._load_resume_checkpoint(
                            fixture.args, fixture.trainer, fixture.method,
                            fixture.task_manager, fixture.tracker, fixture.calibrator)
                    self.assertEqual(result, (1, {0: [0, 1, 2]}, []))
                    decoded = runner._decode_checkpoint_value(older)
                    for key, actual in (
                            ('trainer_state', fixture.trainer.get_state()),
                            ('cl_state', fixture.method.get_state()),
                            ('tracker_state', fixture.tracker.to_dict()),
                            ('rng_state', runner._capture_rng_state())):
                        self.assertTrue(runner._checkpoint_values_equal(decoded[key], actual), key)
                    self.assertEqual(fixture.task_manager.get_all_seen_classes(), [0, 1, 2])
                    self.assertEqual(fixture.task_manager.get_forgotten_classes(), [])

    def test_populated_valid_preflight_is_rng_and_live_state_neutral(self):
        for formal in (False, True):
            for method_type in self.METHODS:
                with self.subTest(formal=formal, method=method_type.__name__), \
                        tempfile.TemporaryDirectory() as directory:
                    fixture = self.fixture(method_type, directory, formal)
                    payload = self.payload(fixture, method_type, 1)
                    before = self.snapshot(fixture)
                    decoded = self.preflight(fixture, payload)
                    self.assertTrue(runner._checkpoint_values_equal(
                        decoded, runner._decode_checkpoint_value(payload)))
                    self.assert_unchanged(fixture, before)

    def test_populated_method_must_match_completed_task_and_retained_class_history(self):
        for formal in (False, True):
            for method_type in self.METHODS:
                with self.subTest(formal=formal, method=method_type.__name__), \
                        tempfile.TemporaryDirectory() as directory:
                    fixture = self.fixture(method_type, directory, formal, forgotten=[1])
                    payload = self.payload(fixture, method_type, 1)
                    state = runner._decode_checkpoint_value(payload['cl_state'])
                    corruptions = []
                    if method_type is TARGETCL:
                        missing_task = copy.deepcopy(state)
                        missing_task['task_classes'] = {1: missing_task['task_classes'].pop(0)}
                        missing_task['generators'] = {1: missing_task['generators'].pop(0)}
                        corruptions.append(missing_task)
                        wrong_classes = copy.deepcopy(state)
                        wrong_classes['task_classes'][0] = [0, 1, 3]
                        corruptions.append(wrong_classes)
                        wrong_forget = copy.deepcopy(state)
                        wrong_forget['forgotten'] = []
                        corruptions.append(wrong_forget)
                    elif method_type is ProtoFedSpaceCL:
                        missing = copy.deepcopy(state)
                        del missing['protos'][0]
                        corruptions.append(missing)
                        stale = copy.deepcopy(state)
                        stale['protos'][1] = torch.ones(2)
                        corruptions.append(stale)
                    else:
                        state['labels'][0] = 1
                        corruptions.append(state)
                    for index, state in enumerate(corruptions):
                        with self.subTest(corruption=index):
                            invalid = {**payload, 'cl_state': runner._encode_checkpoint_value(state)}
                            # Each corruption is internally loadable; it fails only
                            # by comparison with the outer completed-stage history.
                            method_type(fixture.trainer, fixture.args).load_state(state)
                            before = self.snapshot(fixture)
                            with self.assertRaisesRegex(ValueError, 'TARGET|ProtoFedSpace|ER-ACE'):
                                self.preflight(fixture, invalid)
                            self.assert_unchanged(fixture, before)

    def test_malformed_later_target_generator_rejection_preserves_probe_rng(self):
        for formal in (False, True):
            with self.subTest(formal=formal), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(TARGETCL, directory, formal)
                payload = self.payload(fixture, TARGETCL, 1)
                payload['cl_state']['generators'][1]['state_dict']['class_emb.weight'][0, 0] = float('nan')
                before = self.snapshot(fixture)
                with self.assertRaisesRegex(ValueError, 'TARGET generator state dict'):
                    self.preflight(fixture, payload)
                self.assert_unchanged(fixture, before)

    def test_all_forgotten_ul_allows_empty_only_for_methods_that_purge_history(self):
        for formal in (False, True):
            for method_type in self.METHODS:
                with self.subTest(formal=formal, method=method_type.__name__), \
                        tempfile.TemporaryDirectory() as directory:
                    fixture = self.fixture(method_type, directory, formal, forgotten=[0, 1, 2])
                    payload = self.payload(fixture, method_type, 1)
                    before = self.snapshot(fixture)
                    self.preflight(fixture, payload)
                    self.assert_unchanged(fixture, before)
                    if method_type is TARGETCL:
                        payload['cl_state'] = runner._encode_checkpoint_value(fixture.method.get_state())
                        with self.assertRaisesRegex(ValueError, 'TARGET'):
                            self.preflight(fixture, payload)
                        self.assert_unchanged(fixture, before)

    def test_proto_all_forgotten_sanitize_resume_can_train_next_cil_identically(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.fixture(ProtoFedSpaceCL, directory, forgotten=[0, 1, 2])
            source.finish_first_task()
            sanitize_cl_state(source.method, source.trainer, [0, 1, 2])
            self.assertEqual(source.method.protos, {})
            payload = self.payload(source, ProtoFedSpaceCL, 1)
            payload['cl_state'] = runner._encode_checkpoint_value(source.method.get_state())
            torch.save(payload, Path(directory) / 'checkpoints/resume_latest.pt')

            def next_cil(fixture):
                fixture.method.before_task(1, [3], [3])
                before = fixture.trainer.get_state()
                try:
                    history, _ = fixture.method.train_task(
                        [(fixture.x, torch.full((len(fixture.x),), 3, dtype=torch.long))], 1)
                except RuntimeError as error:
                    self.fail(f'empty ProtoFedSpace must preserve trainable incoming CE: {error}')
                self.assertFalse(runner._checkpoint_values_equal(before, fixture.trainer.get_state()))
                return history, fixture.trainer.get_state()

            uninterrupted = next_cil(source)
            restored = self.fixture(ProtoFedSpaceCL, directory, forgotten=[0, 1, 2])
            result = runner._load_resume_checkpoint(
                restored.args, restored.trainer, restored.method,
                restored.task_manager, restored.tracker, restored.calibrator)
            self.assertEqual(result, (2, {0: [0, 1, 2]}, []))
            self.assertEqual(restored.task_manager.get_effective_classes(), [])
            resumed = next_cil(restored)
            torch.testing.assert_close(uninterrupted, resumed, rtol=0, atol=0)

    def test_unsanitized_ablation_preserves_complete_replay_history_after_ul(self):
        for formal in (False, True):
            for method_type in self.METHODS:
                for forgotten in ([1], [0, 1, 2]):
                    with self.subTest(formal=formal, method=method_type.__name__, forgotten=forgotten), \
                            tempfile.TemporaryDirectory() as directory:
                        fixture = self.fixture(method_type, directory, formal, forgotten)
                        fixture.args.sanitize_cl_state = 0
                        payload = self.payload(fixture, method_type, 1)
                        before = self.snapshot(fixture)
                        try:
                            self.preflight(fixture, payload)
                        except ValueError as error:
                            self.fail(f'documented unsanitized ablation must remain resumable: {error}')
                        self.assert_unchanged(fixture, before)
                        result = runner._load_resume_checkpoint(
                            fixture.args, fixture.trainer, fixture.method,
                            fixture.task_manager, fixture.tracker, fixture.calibrator)
                        self.assertEqual(result, (2, {0: [0, 1, 2]}, []))
                        self.assertEqual(fixture.task_manager.get_forgotten_classes(), forgotten)
                        self.assertTrue(runner._checkpoint_values_equal(
                            fixture.method.get_state(), runner._decode_checkpoint_value(payload['cl_state'])))

    def test_unsanitized_ablation_still_rejects_empty_completed_replay_atomically(self):
        for formal in (False, True):
            for method_type in self.METHODS:
                with self.subTest(formal=formal, method=method_type.__name__), \
                        tempfile.TemporaryDirectory() as directory:
                    fixture = self.fixture(method_type, directory, formal, [0, 1, 2])
                    fixture.args.sanitize_cl_state = 0
                    payload = self.payload(fixture, method_type, 1)
                    payload['cl_state'] = runner._encode_checkpoint_value(fixture.method.get_state())
                    before = self.snapshot(fixture)
                    with self.assertRaisesRegex(ValueError, 'TARGET|ProtoFedSpace|ER-ACE'):
                        self.preflight(fixture, payload)
                    self.assert_unchanged(fixture, before)


if __name__ == '__main__':
    unittest.main()
