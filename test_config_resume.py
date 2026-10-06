import unittest
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from bic_calibration import TaskAffineCalibrator
from adaptive_consolidation_audit import atomic_write_new_json
from config import get_config, validate_resume_config
from metrics import MetricsTracker
from runner import (
    _evaluate_formal_from_single_access,
    _load_resume_checkpoint,
    _publish_formal_deferred_result,
    _recover_atomic_checkpoint_temps as _runner_recover_checkpoint_temps,
    _save_cil_checkpoint,
)


class _State:
    def __init__(self, value=None):
        self.value = value

    def get_state(self):
        return {'value': self.value}

    def load_state(self, state):
        self.value = state['value']


class _NumpyMethod:
    def __init__(self):
        self.protos = {3: torch.tensor([1.25, -2.5])}
        self.radius = np.float64(0.75)

    def get_state(self):
        return {'protos': self.protos, 'radius': self.radius}

    def load_state(self, state):
        self.protos = state['protos']
        self.radius = state.get('radius', self.radius)


class _PrlMethod:
    def __init__(self):
        self.protos = {}

    def get_state(self):
        return {'protos': self.protos}

    def load_state(self, state):
        self.protos = state['protos']


class _Trainer(_State):
    def get_state(self):
        return {
            'bottoms': [],
            'top_model': {'value': torch.tensor([float(self.value)])},
        }

    def load_state(self, state):
        self.value = state['top_model']['value'].item()


class _Tasks:
    def advance_task(self, _task_id):
        pass

    def apply_unlearn(self, _classes):
        pass


class _Exploit:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.system, (f'touch {self.marker}',)


def _recover_atomic_checkpoint_temps(directory, args):
    method = _PrlMethod() if args.cl_method == 'prl' else _NumpyMethod()
    return _runner_recover_checkpoint_temps(
        directory, args, _Trainer(0), method, _Tasks(),
        MetricsTracker(), TaskAffineCalibrator(),
    )


class ConfigResumeTests(unittest.TestCase):
    def test_formal_runner_opens_one_all_class_test_loader_at_final_phase(self):
        class Dataset:
            def __init__(self):
                self.authorizations = []
                self.test_accesses = []
                self.iterations = 0

            def authorize_formal_access(self, **record):
                self.authorizations.append(record)

            def get_test_loader(self, classes):
                self.test_accesses.append(list(classes))

                def batches():
                    self.iterations += 1
                    yield (
                        torch.tensor([[1.0], [2.0]], requires_grad=True),
                        torch.tensor([0, 1]),
                    )

                return batches()

        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output_dir=tmp)
            dataset = Dataset()
            expected = {'task_acc_history': [{'step': 'event_1_CIL'}]}

            def evaluate(**kwargs):
                cache = kwargs['cached_test_batches']
                self.assertIsInstance(cache, tuple)
                self.assertEqual(len(cache), 1)
                self.assertEqual(cache[0][0].device.type, 'cpu')
                self.assertFalse(cache[0][0].requires_grad)
                return expected

            with mock.patch(
                    'runner.prepare_formal_deferred_evaluation',
                    return_value={'status': 'pending'}), mock.patch(
                        'runner.evaluate_formal_deferred_trajectory',
                        side_effect=evaluate):
                result, status = _evaluate_formal_from_single_access(
                    args=args, dataset=dataset, snapshot_paths=['stage'],
                    final_checkpoint='final', task_classes={0: [0], 1: [1]},
                    final_event=1, final_task=1,
                )
            self.assertEqual((result, status), (expected, 'complete'))
            self.assertEqual(dataset.test_accesses, [[0, 1]])
            self.assertEqual(dataset.iterations, 1)
            self.assertEqual(dataset.authorizations, [{
                'split': 'test', 'phase': 'final_test_post_install',
                'event_idx': 1, 'task_id': 1,
                'timeline_step': 'event_1_CIL', 'classes': [0, 1],
            }])

            blocked = Dataset()
            with mock.patch(
                    'runner.prepare_formal_deferred_evaluation',
                    return_value={
                        'status': 'complete', 'result': expected,
                    }):
                reused, reused_status = _evaluate_formal_from_single_access(
                    args=args, dataset=blocked, snapshot_paths=['stage'],
                    final_checkpoint='final', task_classes={0: [0], 1: [1]},
                    final_event=1, final_task=1,
                )
            self.assertEqual((reused, reused_status), (expected, 'complete'))
            self.assertEqual(blocked.authorizations, [])
            self.assertEqual(blocked.test_accesses, [])

    def test_formal_final_publication_failure_leaves_fail_closed_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(output_dir=tmp)
            checkpoint = root / 'formal_final.pt'
            torch.save({'schema_version': 4, 'tracker_state': {}}, checkpoint)
            complete = {
                'schema_version': 1, 'status': 'complete',
                'transaction_sha256': 'a' * 64,
                'evaluation': {'task_acc_history': []},
                'cache_identity': {'batch_count': 1},
            }

            def begin(**_kwargs):
                atomic_write_new_json(
                    root / 'FORMAL_EVALUATION_PUBLISHING.json', {
                        'schema_version': 1, 'status': 'publishing',
                        'transaction_sha256': complete['transaction_sha256'],
                    }
                )
                return complete

            with mock.patch(
                    'runner.begin_formal_deferred_publication',
                    side_effect=begin), mock.patch(
                        'runner._RESUME_CHECKPOINT_KEYS',
                        {'schema_version', 'tracker_state'}), mock.patch(
                        'runner._atomic_torch_save',
                        side_effect=RuntimeError(
                            'simulated publication failure'
                        )):
                with self.assertRaisesRegex(RuntimeError, 'publication failure'):
                    _publish_formal_deferred_result(
                        args=args, final_checkpoint=checkpoint,
                        tracker_state={'task_acc_history': []},
                        final_result={'task_acc_history': [], 'config': {}},
                    )
            self.assertTrue(
                (root / 'FORMAL_EVALUATION_PUBLISHING.json').is_file()
            )
            self.assertFalse(
                (root / 'FORMAL_EVALUATION_PUBLISHED.json').exists()
            )

    def _fixed_args(self, output_dir, cl_method='proto_aug'):
        return SimpleNamespace(
            output_dir=output_dir, resume_run_dir=output_dir,
            save_task_checkpoints=3, num_tasks=2, num_parties=0, seed=7,
            data='toy', cl_method=cl_method,
            head_consolidation_enabled=1,
            head_consolidation_mode='full_classifier',
        )

    def test_fixed_mode_checkpoint_round_trips_numpy_method_state_safely(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixed_args(tmp)
            trainer = _Trainer(11)
            method = _NumpyMethod()
            expected = method.protos[3].clone()
            _save_cil_checkpoint(
                trainer, method, args, 'event_0_CIL', 0, [0], {0: [0]},
            )
            self.assertTrue((Path(tmp) / 'checkpoints' / 'resume_latest.pt').is_file())
            method.protos = {}
            method.radius = np.float64(-1)
            _load_resume_checkpoint(
                args, trainer, method, _Tasks(), MetricsTracker(),
                TaskAffineCalibrator(),
            )
            torch.testing.assert_close(method.protos[3], expected)
            self.assertEqual(float(method.radius), 0.75)

    def test_new_checkpoint_canonicalizes_numpy_mapping_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixed_args(tmp)
            method = _NumpyMethod()
            method.protos = {
                np.int64(3): torch.tensor([2.0, 4.0]),
            }
            _save_cil_checkpoint(
                _Trainer(11), method, args,
                'event_0_CIL', 0, [0], {0: [0]},
            )
            method.protos = {}
            _load_resume_checkpoint(
                args, _Trainer(-1), method, _Tasks(), MetricsTracker(),
                TaskAffineCalibrator(),
            )
            torch.testing.assert_close(
                method.protos[3], torch.tensor([2.0, 4.0])
            )

    def test_legacy_prl_numpy_checkpoint_uses_scoped_safe_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixed_args(tmp, 'prl')
            _save_cil_checkpoint(
                _Trainer(11), _State(12), args,
                'event_0_CIL', 0, [0], {0: [0]},
            )
            path = Path(tmp) / 'checkpoints' / 'resume_latest.pt'
            legacy = torch.load(path, map_location='cpu', weights_only=True)
            expected = np.array([4.5, -3.25], dtype=np.float32)
            legacy['schema_version'] = 3
            legacy['cl_state'] = {'protos': {3: expected.copy()}}
            torch.save(legacy, path)
            method = _PrlMethod()
            _load_resume_checkpoint(
                args, _Trainer(-1), method, _Tasks(), MetricsTracker(),
                TaskAffineCalibrator(),
            )
            np.testing.assert_array_equal(method.protos[3], expected)

    def test_legacy_proto_aug_restores_tensor_prototypes_and_numpy_radius(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixed_args(tmp)
            _save_cil_checkpoint(
                _Trainer(11), _State(12), args,
                'event_0_CIL', 0, [0], {0: [0]},
            )
            path = Path(tmp) / 'checkpoints' / 'resume_latest.pt'
            legacy = torch.load(path, map_location='cpu', weights_only=True)
            expected = torch.tensor([4.5, -3.25])
            legacy['schema_version'] = 3
            legacy['cl_state'] = {
                'protos': {3: expected.clone()}, 'radius': np.float64(1.25),
            }
            torch.save(legacy, path)
            method = _NumpyMethod()
            method.protos = {}
            _load_resume_checkpoint(
                args, _Trainer(-1), method, _Tasks(), MetricsTracker(),
                TaskAffineCalibrator(),
            )
            torch.testing.assert_close(method.protos[3], expected)
            self.assertIsInstance(method.radius, np.float64)
            self.assertEqual(float(method.radius), 1.25)

    def test_legacy_fixed_compatibility_rejects_executable_reducer(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixed_args(tmp)
            _save_cil_checkpoint(
                _Trainer(11), _NumpyMethod(), args,
                'event_0_CIL', 0, [0], {0: [0]},
            )
            path = Path(tmp) / 'checkpoints' / 'resume_latest.pt'
            legacy = torch.load(path, map_location='cpu', weights_only=True)
            marker = Path(tmp) / 'executed'
            legacy['schema_version'] = 3
            legacy['cl_state'] = {
                'protos': {3: torch.tensor([1.0])},
                'radius': _Exploit(marker),
            }
            torch.save(legacy, path)
            with self.assertRaisesRegex(RuntimeError, 'no valid resume checkpoint'):
                _load_resume_checkpoint(
                    args, _Trainer(-1), _NumpyMethod(), _Tasks(),
                    MetricsTracker(), TaskAffineCalibrator(),
                )
            self.assertFalse(marker.exists())

    def test_recovery_promotes_newer_legacy_numpy_temp_with_scoped_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixed_args(tmp)
            _save_cil_checkpoint(
                _Trainer(11), _State(12), args,
                'event_0_CIL', 0, [0], {0: [0]},
            )
            target = Path(tmp) / 'checkpoints' / 'resume_latest.pt'
            old = torch.load(target, map_location='cpu', weights_only=True)
            old['schema_version'] = 3
            old['cl_state'] = {
                'protos': {3: torch.tensor([1.0])},
                'radius': np.float64(0.5),
            }
            torch.save(old, target)
            newer = dict(old)
            newer['event_idx'] = 1
            newer['step'] = 'event_1_CIL'
            temporary = Path(str(target) + '.tmp')
            with temporary.open('wb') as handle:
                torch.save(newer, handle)
                handle.flush()
                os.fsync(handle.fileno())
            _recover_atomic_checkpoint_temps(target.parent, args)
            recovered = torch.load(target, map_location='cpu', weights_only=False)
            self.assertEqual(recovered['event_idx'], 1)
            self.assertFalse(temporary.exists())

    def test_recovery_compares_legacy_target_with_new_schema_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixed_args(tmp)
            _save_cil_checkpoint(
                _Trainer(11), _NumpyMethod(), args,
                'event_0_CIL', 0, [0], {0: [0]},
            )
            target = Path(tmp) / 'checkpoints' / 'resume_latest.pt'
            newer = torch.load(target, map_location='cpu', weights_only=True)
            newer['event_idx'] = 1
            newer['step'] = 'event_1_CIL'
            old = copy.deepcopy(newer)
            old['schema_version'] = 3
            old['event_idx'] = 0
            old['step'] = 'event_0_CIL'
            old['cl_state'] = {
                'protos': {3: torch.tensor([1.0])},
                'radius': np.float64(0.5),
            }
            torch.save(old, target)
            temporary = Path(str(target) + '.tmp')
            torch.save(newer, temporary)
            _recover_atomic_checkpoint_temps(target.parent, args)
            recovered = torch.load(target, map_location='cpu', weights_only=True)
            self.assertEqual(recovered['schema_version'], 4)
            self.assertEqual(recovered['event_idx'], 1)

    def test_recovery_rejects_newer_proto_aug_with_empty_method_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixed_args(tmp)
            _save_cil_checkpoint(
                _Trainer(11), _NumpyMethod(), args,
                'event_0_CIL', 0, [0], {0: [0]},
            )
            target = Path(tmp) / 'checkpoints' / 'resume_latest.pt'
            newer = torch.load(target, map_location='cpu', weights_only=True)
            newer['event_idx'] = 1
            newer['step'] = 'event_1_CIL'
            newer['cl_state'] = {}
            temporary = Path(str(target) + '.tmp')
            torch.save(newer, temporary)
            with self.assertRaisesRegex(ValueError, 'method state'):
                _recover_atomic_checkpoint_temps(target.parent, args)
            recovered = torch.load(target, map_location='cpu', weights_only=True)
            self.assertEqual(recovered['event_idx'], 0)
            self.assertTrue(temporary.exists())

    def test_resume_rejects_changed_training_protocol(self):
        recorded = {
            'seed': 44,
            'classes_per_task': 10,
            'head_consolidation_mode': 'adaptive_dual_branch',
            'adaptive_source_commit': 'a' * 40,
            'output_dir': '/old/run',
            'resume_run_dir': '',
        }
        current = SimpleNamespace(
            seed=44,
            classes_per_task=20,
            head_consolidation_mode='adaptive_dual_branch',
            adaptive_source_commit='a' * 40,
            output_dir='/old/run',
            resume_run_dir='/old/run',
        )

        with self.assertRaisesRegex(ValueError, 'classes_per_task'):
            validate_resume_config(current, recorded)

    def test_resume_allows_only_runtime_directory_fields_to_change(self):
        recorded = {
            'seed': 44,
            'classes_per_task': 10,
            'head_consolidation_mode': 'adaptive_dual_branch',
            'adaptive_source_commit': 'a' * 40,
            'output_dir': '/old/run',
            'resume_run_dir': '',
        }
        current = SimpleNamespace(
            seed=44,
            classes_per_task=10,
            head_consolidation_mode='adaptive_dual_branch',
            adaptive_source_commit='a' * 40,
            output_dir='/old/run',
            resume_run_dir='/old/run',
        )

        validate_resume_config(current, recorded)

    def test_resume_rejects_a_config_missing_current_protocol_fields(self):
        recorded = {
            'seed': 44,
            'head_consolidation_mode': 'adaptive_dual_branch',
            'adaptive_source_commit': 'a' * 40,
            'output_dir': '/old/run',
            'resume_run_dir': '',
        }
        current = SimpleNamespace(
            seed=44,
            classes_per_task=10,
            head_consolidation_mode='adaptive_dual_branch',
            adaptive_source_commit='a' * 40,
            output_dir='/old/run',
            resume_run_dir='/old/run',
        )

        with self.assertRaisesRegex(ValueError, 'classes_per_task'):
            validate_resume_config(current, recorded)

    def test_resume_rejects_changed_adaptive_source_commit(self):
        recorded = {
            'seed': 44,
            'head_consolidation_mode': 'adaptive_dual_branch',
            'adaptive_source_commit': 'a' * 40,
            'output_dir': '/old/run',
            'resume_run_dir': '',
        }
        current = SimpleNamespace(
            seed=44,
            head_consolidation_mode='adaptive_dual_branch',
            adaptive_source_commit='b' * 40,
            output_dir='/old/run',
            resume_run_dir='/old/run',
        )

        with self.assertRaisesRegex(ValueError, 'adaptive_source_commit'):
            validate_resume_config(current, recorded)

    def test_legacy_config_defaults_and_formal_flag_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch('sys.argv', [
                    'main.py', '--results_dir', tmp,
                    '--exp_name', 'legacy-defaults']):
                legacy = get_config()
            self.assertIs(legacy.formal_deferred_evaluation, False)
            self.assertEqual(legacy.fedprotip_tip_threshold, 0.775)
            self.assertEqual(legacy.fedprotip_max_batches, 20)

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch('sys.argv', [
                    'main.py', '--results_dir', tmp,
                    '--exp_name', 'formal-explicit',
                    '--formal_deferred_evaluation', '1']):
                formal = get_config()
            self.assertIs(formal.formal_deferred_evaluation, True)

    def test_formal_new_run_uses_exact_name_while_legacy_keeps_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch('sys.argv', [
                    'main.py', '--results_dir', tmp,
                    '--exp_name', 'legacy-timestamped']):
                legacy = get_config()
            self.assertEqual(Path(tmp), Path(legacy.output_dir).parent)
            self.assertRegex(
                Path(legacy.output_dir).name,
                r'^legacy-timestamped_[0-9]{8}_[0-9]{6}$')

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch('sys.argv', [
                    'main.py', '--results_dir', tmp,
                    '--exp_name', 'formal-exact',
                    '--formal_deferred_evaluation', '1']):
                formal = get_config()
            self.assertEqual(
                os.path.abspath(os.path.join(tmp, 'formal-exact')),
                formal.output_dir)
            self.assertTrue(Path(formal.output_dir, 'config.json').is_file())

    def test_two_registered_formal_commands_create_distinct_exact_directories(self):
        from three_dataset_formal_registry import command_for, formal_specs

        with tempfile.TemporaryDirectory() as tmp:
            outputs = []
            for spec in formal_specs()[:2]:
                command = command_for(spec, 'cpu', tmp, smoke=False)
                start = 2 if spec.method in {
                    'finetune', 'lwf', 'gpm', 'fedprotip_vfl', 'er'
                } else 3
                with mock.patch('sys.argv', ['main.py', *command[start:]]):
                    args = get_config()
                expected = os.path.abspath(os.path.join(
                    tmp, command[command.index('--exp_name') + 1]))
                self.assertEqual(expected, args.output_dir)
                outputs.append(args.output_dir)
            self.assertEqual(2, len(set(outputs)))

    def test_resume_keeps_exact_recorded_directory_without_new_sibling(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_argv = [
                'main.py', '--results_dir', tmp, '--exp_name', 'resume-source']
            with mock.patch('sys.argv', original_argv):
                original = get_config()
            before = set(Path(tmp).iterdir())
            with mock.patch('sys.argv', [
                    *original_argv, '--resume_run_dir', original.output_dir]):
                resumed = get_config()
            self.assertEqual(original.output_dir, resumed.output_dir)
            self.assertEqual(before, set(Path(tmp).iterdir()))

    def test_parser_rejects_conflicting_formal_deferred_value(self):
        with mock.patch('sys.argv', [
                'main.py', '--formal_deferred_evaluation', '2']):
            with self.assertRaises(SystemExit):
                get_config()

    def test_resume_binds_all_formal_deferred_and_fedprotip_fields(self):
        fields = {
            'formal_deferred_evaluation': True,
            'fedprotip_tip_threshold': 0.775,
            'fedprotip_max_batches': 20,
        }
        recorded = {
            **fields,
            'seed': 44,
            'output_dir': '/old/run',
            'resume_run_dir': '',
        }
        current = SimpleNamespace(
            **fields,
            seed=44,
            output_dir='/old/run',
            resume_run_dir='/old/run',
        )
        validate_resume_config(current, recorded)

        changed_values = {
            'formal_deferred_evaluation': False,
            'fedprotip_tip_threshold': 0.8,
            'fedprotip_max_batches': 21,
        }
        for field, changed in changed_values.items():
            with self.subTest(field=field, case='changed'):
                changed_current = SimpleNamespace(**vars(current))
                setattr(changed_current, field, changed)
                with self.assertRaisesRegex(ValueError, field):
                    validate_resume_config(changed_current, recorded)
            with self.subTest(field=field, case='missing'):
                missing_recorded = dict(recorded)
                del missing_recorded[field]
                with self.assertRaisesRegex(ValueError, field):
                    validate_resume_config(current, missing_recorded)


if __name__ == '__main__':
    unittest.main()
