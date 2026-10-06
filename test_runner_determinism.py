import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import runner
from runner import initialize_experiment_rng


class _CheckpointState:
    def __init__(self, value):
        self.value = value

    def get_state(self):
        return {'value': self.value}

    def load_state(self, state):
        self.value = state['value']


class _ResumeTaskState:
    def __init__(self):
        self.seen = []
        self.forgotten = []

    def advance_task(self, task_id):
        self.seen.append(task_id)

    def apply_unlearn(self, classes):
        self.forgotten.extend(classes)


class RunnerDeterminismTests(unittest.TestCase):
    def setUp(self):
        self.old_env = dict(os.environ)
        os.environ.update({
            'PYTHONHASHSEED': '42',
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
        })

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_env)

    def test_deterministic_run_requires_two_workers(self):
        args = SimpleNamespace(seed=42, deterministic=1, num_workers=0)
        with self.assertRaisesRegex(ValueError, 'num_workers=2'):
            initialize_experiment_rng(args)

    def test_deterministic_run_enables_strict_algorithms(self):
        args = SimpleNamespace(seed=42, deterministic=1, num_workers=2)
        initialize_experiment_rng(args)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())


class AdaptiveRunnerDataFlowTest(unittest.TestCase):
    def test_intermediate_ul_checkpoint_resumes_forgotten_state(self):
        with tempfile.TemporaryDirectory() as output_dir:
            args = SimpleNamespace(
                output_dir=output_dir,
                resume_run_dir=output_dir,
                save_task_checkpoints=3,
                num_tasks=2,
                seed=42,
                data='cifar100',
                cl_method='proto_evolve',
                unlearn_after_tasks=[0],
            )
            trainer = _CheckpointState(11)
            method = _CheckpointState(12)
            try:
                runner._save_cil_checkpoint(
                    trainer, method, args, 'event_0_CIL', 0, [0, 1], {0: [0, 1]},
                    force=True, forgotten_classes=[],
                )
                trainer.value = 21
                method.value = 22
                runner._save_cil_checkpoint(
                    trainer, method, args, 'event_1_UL', 0, [0, 1], {0: [0, 1]},
                    force=True, forgotten_classes=[0],
                )
            except TypeError as exc:
                self.fail(f'UL checkpoint state is not accepted: {exc}')

            restored_trainer = _CheckpointState(-1)
            restored_method = _CheckpointState(-2)
            task_state = _ResumeTaskState()
            start_idx, seen, _ = runner._load_resume_checkpoint(
                args,
                restored_trainer,
                restored_method,
                task_state,
                runner.MetricsTracker(),
                runner.TaskAffineCalibrator(),
            )

            self.assertEqual(start_idx, 2)
            self.assertEqual(seen, {0: [0, 1]})
            self.assertEqual(restored_trainer.value, 21)
            self.assertEqual(restored_method.value, 22)
            self.assertEqual(task_state.seen, [0])
            self.assertEqual(task_state.forgotten, [0])

    def test_final_ul_checkpoint_is_rolling_and_newest_event_resume_source(self):
        with tempfile.TemporaryDirectory() as output_dir:
            args = SimpleNamespace(
                output_dir=output_dir,
                resume_run_dir=output_dir,
                save_task_checkpoints=3,
                num_tasks=1,
                seed=42,
                data='cifar100',
                cl_method='proto_evolve',
                unlearn_after_tasks=[0],
            )
            trainer = _CheckpointState(11)
            method = _CheckpointState(12)
            runner._save_cil_checkpoint(
                trainer, method, args, 'event_0_CIL', 0, [0, 1], {0: [0, 1]},
                force=True, forgotten_classes=[],
            )
            trainer.value = 21
            method.value = 22
            runner._save_cil_checkpoint(
                trainer, method, args, 'event_1_UL', 0, [0, 1], {0: [0, 1]},
                force=True, forgotten_classes=[0],
            )

            checkpoint_dir = os.path.join(output_dir, 'checkpoints')
            rolling = os.path.join(checkpoint_dir, 'resume_latest.pt')
            self.assertTrue(os.path.isfile(rolling), 'final UL must remain directly resumable')
            os.remove(rolling)

            restored_trainer = _CheckpointState(-1)
            restored_method = _CheckpointState(-2)
            task_state = _ResumeTaskState()
            start_idx, _, _ = runner._load_resume_checkpoint(
                args,
                restored_trainer,
                restored_method,
                task_state,
                runner.MetricsTracker(),
                runner.TaskAffineCalibrator(),
            )

            self.assertEqual(start_idx, 2)
            self.assertEqual(restored_trainer.value, 21)
            self.assertEqual(restored_method.value, 22)
            self.assertEqual(task_state.forgotten, [0])

    def test_adaptive_detection_requires_the_explicit_protocol(self):
        detect = getattr(runner, '_is_adaptive_mode', None)
        self.assertIsNotNone(detect)
        method = SimpleNamespace(set_head_validation_provider=lambda *args: None)
        self.assertFalse(detect(method, SimpleNamespace(
            head_consolidation_enabled=1,
            head_consolidation_mode='full_classifier',
        )))
        self.assertFalse(detect(method, SimpleNamespace(
            head_consolidation_enabled=0,
            head_consolidation_mode='adaptive_dual_branch',
        )))
        self.assertTrue(detect(method, SimpleNamespace(
            head_consolidation_enabled=1,
            head_consolidation_mode='adaptive_dual_branch',
        )))

    def test_adaptive_stream_uses_only_train_loaders_and_defers_all_evaluation(self):
        events = []
        trainers = []

        class FakeDataset:
            validation_manifest = {'sha256': 'v' * 64, 'ordered_indices': [1]}

            def get_train_loader(self, classes, shuffle=True):
                events.append(('train_loader', tuple(classes), shuffle))
                return object()

            def get_validation_loader(self, classes):
                events.append(('validation_loader', tuple(classes)))
                return iter(())

            def get_test_loader(self, classes):
                events.append(('test_loader', tuple(classes)))
                return iter(())

            def get_task_loaders(self, *args, **kwargs):
                raise AssertionError('adaptive runner called legacy coupled loaders')

            def get_forget_retain_loaders(self, *args, **kwargs):
                raise AssertionError('adaptive runner constructed coupled UL loaders')

            def selection_audit(self):
                return {'passed': True}

        class FakeTaskManager:
            def __init__(self, args):
                self.seen = []
                self.forgotten = []

            def get_timeline(self):
                return [
                    {'type': 'CIL', 'task_id': 0, 'new_classes': [0, 1]},
                    {'type': 'UL', 'forget_classes': [0]},
                ]

            def advance_task(self, task_id):
                self.seen = [0, 1]

            def get_effective_classes(self):
                return [value for value in self.seen if value not in self.forgotten]

            def get_all_seen_classes(self):
                return list(self.seen)

            def get_forgotten_classes(self):
                return list(self.forgotten)

            def apply_unlearn(self, classes):
                self.forgotten.extend(classes)

        class FakeTrainer:
            def __init__(self, bottoms, top, args):
                trainers.append(self)
                self.ce_lo = 0
                self.ce_hi = None
                self.bottoms = []
                self.top_model = SimpleNamespace(
                    classifier=SimpleNamespace(weight=torch.zeros(2, 1))
                )

            def reset_comm_stats(self):
                pass

            def get_comm_stats(self):
                return {}

            def get_state(self):
                return {}

        class FakeAdaptiveMethod:
            def set_head_validation_provider(self, loader_provider, manifest_provider):
                events.append(('provider_installed',))
                self.loader_provider = loader_provider
                self.manifest_provider = manifest_provider

            def before_task(self, task_id, new_classes, effective):
                events.append(('before_task',))

            def train_task(self, loader, task_id):
                events.append(('train_task',))
                return [], 0.25

            def after_task(self, loader, task_id):
                events.append(('after_task',))

            def get_state(self):
                return {}

        class FakeUlMethod:
            def __init__(self, trainer):
                self.trainer = trainer

            def unlearn(self, forget, retain_loader, forget_loader, effective_classes):
                if self.trainer.dataset_ref is not None:
                    raise AssertionError('adaptive UL retained validation-capable dataset_ref')
                events.append(('unlearn', tuple(forget), tuple(effective_classes)))
                return {}

        dataset = FakeDataset()
        method = FakeAdaptiveMethod()
        args = SimpleNamespace(
            cl_method='proto_evolve',
            ul_method='none',
            data='cifar100',
            seed=42,
            deterministic=0,
            num_workers=0,
            device='cpu',
            task_ce_mode='method',
            replay_mode='prototype',
            save_task_checkpoints=1,
            num_tasks=1,
            bic_enabled=0,
            lambda_validation_enabled=1,
            data_flow_audit=0,
            resume_run_dir='',
            party_kd_enabled=0,
            sanitize_cl_state=0,
            head_consolidation_enabled=1,
            head_consolidation_mode='adaptive_dual_branch',
        )

        with tempfile.TemporaryDirectory() as output_dir:
            args.output_dir = output_dir
            with mock.patch.object(runner, 'VFLDataset', return_value=dataset), mock.patch.object(
                    runner, 'TaskManager', FakeTaskManager), mock.patch.object(
                    runner, 'build_models', return_value=([], object())), mock.patch.object(
                    runner, 'VFLTrainer', FakeTrainer), mock.patch.object(
                    runner, 'get_cl_method', return_value=method), mock.patch.object(
                    runner, 'get_ul_method', side_effect=lambda name, trainer, args: FakeUlMethod(trainer)), mock.patch.object(
                    runner, '_save_cil_checkpoint',
                    side_effect=lambda *args, **kwargs: events.append(('checkpoint',))) as checkpoint, mock.patch.object(
                    runner, '_save_party_kd_audit'), mock.patch.object(
                    runner, '_save_final_probs') as save_final_probs:
                runner.run_experiment(args)

        self.assertEqual(events, [
            ('provider_installed',),
            ('train_loader', (0, 1), True),
            ('before_task',),
            ('train_task',),
            ('after_task',),
            ('checkpoint',),
            ('train_loader', (0,), True),
            ('train_loader', (1,), True),
            ('unlearn', (0,), (1,)),
            ('checkpoint',),
        ])
        snapshot = method.manifest_provider()
        snapshot['ordered_indices'].append(2)
        self.assertEqual(dataset.validation_manifest['ordered_indices'], [1])
        self.assertIs(trainers[0].dataset_ref, dataset)
        save_final_probs.assert_not_called()
        self.assertIs(checkpoint.call_args.kwargs.get('force'), True)
        self.assertEqual(checkpoint.call_count, 2)

    def test_forced_adaptive_checkpoint_keeps_every_stage_file(self):
        stateful = SimpleNamespace(
            get_state=lambda: {},
        )
        with tempfile.TemporaryDirectory() as output_dir:
            args = SimpleNamespace(
                output_dir=output_dir,
                save_task_checkpoints=3,
                num_tasks=2,
                seed=42,
                data='cifar100',
                cl_method='proto_evolve',
            )
            runner._save_cil_checkpoint(
                stateful,
                stateful,
                args,
                'event_0_CIL',
                0,
                [0, 1],
                {0: [0, 1]},
                force=True,
            )
            checkpoint_dir = os.path.join(output_dir, 'checkpoints')
            self.assertTrue(os.path.isfile(os.path.join(checkpoint_dir, 'event_0_CIL.pt')))
            self.assertTrue(os.path.isfile(os.path.join(checkpoint_dir, 'resume_latest.pt')))

    def test_formal_external_and_no_consolidation_stages_are_train_only_and_persisted(self):
        class FakeTrainer:
            def __init__(self, _bottoms, _top, args):
                self.args = args
                self.bottoms = []
                self.stage = -1
                self.dataset_ref = object()
                self.ce_lo = 0
                self.ce_hi = None
                self.top_model = SimpleNamespace(
                    classifier=SimpleNamespace(weight=torch.zeros(2, 1))
                )

            def reset_comm_stats(self):
                pass

            def get_comm_stats(self):
                return {
                    'comm_rounds': 0,
                    'megabytes_transmitted': 0.0,
                }

            def get_state(self):
                return {
                    'bottoms': [
                        {'weight': torch.tensor([float(self.stage), float(party)])}
                        for party in range(self.args.num_parties)
                    ],
                    'top_model': {'weight': torch.tensor([float(self.stage)])},
                }

        class FakeMethod:
            def __init__(self, trainer, expects_provider,
                         invokes_adaptive_consolidation):
                self.trainer = trainer
                self.expects_provider = expects_provider
                self.invokes_adaptive_consolidation = \
                    invokes_adaptive_consolidation
                self.stage = -1
                self.installations = []
                self.validation_provider = None
                self.manifest_provider = None

            def _assert_dataset_boundary(self):
                provider = self.trainer.dataset_ref
                if self.expects_provider:
                    if (provider is None
                            or not callable(getattr(provider, 'get_train_loader', None))
                            or hasattr(provider, 'get_validation_loader')
                            or hasattr(provider, 'get_test_loader')):
                        raise AssertionError('formal method did not receive a train-only provider')
                elif provider is not None:
                    raise AssertionError('formal method retained a dataset reference')

            def before_task(self, task_id, _new_classes, _effective):
                self._assert_dataset_boundary()
                self.stage = task_id

            def train_task(self, loader, task_id):
                self._assert_dataset_boundary()
                batches = list(loader)
                if not batches:
                    raise AssertionError('synthetic formal train loader was empty')
                self.trainer.stage = task_id
                return [], 0.01

            def after_task(self, _loader, _task_id):
                self._assert_dataset_boundary()
                if self.invokes_adaptive_consolidation:
                    self._consolidate_head(_task_id)

            def _consolidate_head(self, _task_id):
                if not self.invokes_adaptive_consolidation:
                    raise AssertionError('unexpected formal consolidation')
                formal_order.append('install')
                cached = tuple(self.validation_provider([0, 1]))
                self.installations.append((int(_task_id), cached))
                self.trainer.stage += 100
                return {'installed': True}

            def set_head_validation_provider(self, loader, manifest):
                self.validation_provider = loader
                self.manifest_provider = manifest

            def get_state(self):
                return {'stage': self.stage}

        def make_args(root, npz_path, cl_method):
            return SimpleNamespace(
                cl_method=cl_method, ul_method='none', data='synthvfl',
                vector_npz=str(npz_path), data_path=str(root), output_dir=str(root / 'run'),
                seed=42, deterministic=0, num_workers=0, batch_size=2,
                device='cpu', num_parties=2, num_tasks=2, classes_per_task=1,
                custom_tasks='', unlearn_after_tasks=[], unlearn_classes=[],
                task_ce_mode='method', replay_mode='prototype',
                save_task_checkpoints=0, resume_run_dir='', party_kd_enabled=0,
                sanitize_cl_state=0, bic_enabled=0, lambda_validation_enabled=0,
                data_flow_audit=0, head_consolidation_enabled=0,
                head_consolidation_mode='full_classifier',
                formal_deferred_evaluation=True,
                lambda_validation_per_class=1,
                lambda_validation_split_seed=7,
            )

        variants = (
            ('external', 'er', True, False),
            ('no_consolidation', 'proto_evolve', False, False),
            ('adaptive_final_install', 'proto_evolve', False, True),
        )
        for variant, cl_method, expects_provider, invokes_consolidation in variants:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                npz_path = root / 'synthetic.npz'
                np.savez(
                    npz_path,
                    X=np.arange(32, dtype=np.float32).reshape(8, 4),
                    y=np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
                    train_idx=np.arange(8), test_idx=np.arange(8),
                    view_names=np.array(['left', 'right']),
                    range_lo=np.array([0, 2]), range_hi=np.array([2, 4]),
                )
                args = make_args(root, npz_path, cl_method)
                if invokes_consolidation:
                    args.head_consolidation_enabled = 1
                    args.head_consolidation_mode = 'adaptive_dual_branch'
                    args.lambda_validation_enabled = 1
                methods = []

                def method_factory(_name, trainer, _args):
                    method = FakeMethod(
                        trainer, expects_provider, invokes_consolidation
                    )
                    methods.append(method)
                    return method

                formal_order = []
                test_accesses = []
                publications = []
                original_get_test_loader = runner.VFLDataset.get_test_loader
                original_get_validation_loader = \
                    runner.VFLDataset.get_validation_loader

                def prepare_formal(**_kwargs):
                    formal_order.append('freeze')
                    if invokes_consolidation:
                        payload = runner._safe_torch_load(
                            _kwargs['final_checkpoint']
                        )
                        self.assertEqual(
                            payload['trainer_state']['top_model']['weight'],
                            torch.tensor([101.0]),
                        )
                    return {'status': 'pending'}

                def tracked_get_validation_loader(dataset, classes):
                    formal_order.append('validation')
                    return original_get_validation_loader(dataset, classes)

                def tracked_get_test_loader(dataset, classes):
                    formal_order.append('test')
                    test_accesses.append(tuple(classes))
                    return original_get_test_loader(dataset, classes)

                def evaluate_formal(**kwargs):
                    formal_order.append('evaluate')
                    cache = kwargs['cached_test_batches']
                    self.assertIsInstance(cache, tuple)
                    self.assertTrue(cache)
                    self.assertTrue(all(
                        batch_x.device.type == 'cpu'
                        and batch_y.device.type == 'cpu'
                        and not batch_x.requires_grad
                        and not batch_y.requires_grad
                        for batch_x, batch_y in cache
                    ))
                    evaluated = runner.MetricsTracker()
                    comm = {'comm_rounds': 0, 'megabytes_transmitted': 0.0}
                    for task_id in range(2):
                        step = f'event_{task_id}_CIL'
                        evaluated.record_timing(step, 0.0)
                        evaluated.record_comm(step, comm)
                        evaluated.record_step({
                            'event_idx': task_id, 'type': 'CIL',
                            'task_id': task_id, 'new_classes': [task_id],
                            'evaluation_deferred': True, 'train_time': 0.01,
                            'comm': comm,
                        })
                    evaluated.record_task_accuracies(
                        'event_0_CIL', {'task_0': 1.0}, 1.0
                    )
                    evaluated.task_acc_matrix[-1]['deferred_diagonal'] = {
                        'task_0': 1.0,
                    }
                    evaluated.record_task_accuracies(
                        'event_1_CIL', {'task_0': 1.0, 'task_1': 1.0}, 1.0,
                        per_task_taskil={'task_0': 1.0, 'task_1': 1.0},
                    )
                    evaluated.task_acc_matrix[-1].update({
                        'deferred_diagonal': {
                            'task_0': 1.0, 'task_1': 1.0,
                        },
                        'deferred_final': True,
                        'deferred_final_task': 'task_1',
                    })
                    return evaluated.to_dict()

                def publish_formal(**kwargs):
                    formal_order.append('publish')
                    publications.append(kwargs)
                    payload = runner._safe_torch_load(
                        kwargs['final_checkpoint']
                    )
                    payload['tracker_state'] = kwargs['tracker_state']
                    runner._atomic_torch_save(
                        payload, kwargs['final_checkpoint']
                    )
                    runner._atomic_json_dump(
                        kwargs['final_result'],
                        os.path.join(args.output_dir, 'results.json'),
                    )

                with mock.patch.object(runner, 'build_models', return_value=([], object())), \
                        mock.patch.object(runner, 'VFLTrainer', FakeTrainer), \
                        mock.patch.object(runner, 'get_cl_method', side_effect=method_factory), \
                        mock.patch.object(runner, 'get_ul_method', return_value=object()), \
                        mock.patch.object(
                            runner.VFLDataset, 'get_validation_loader',
                            new=tracked_get_validation_loader,
                        ) as validation_loader, mock.patch.object(
                            runner.VFLDataset, 'get_test_loader',
                            new=tracked_get_test_loader,
                        ), mock.patch.object(
                            runner, '_evaluate_cil_readouts',
                            side_effect=AssertionError('formal stage evaluated CIL readouts'),
                        ) as readouts, mock.patch.object(
                            runner, '_save_final_probs',
                            side_effect=AssertionError('formal stage saved final probabilities'),
                        ) as final_probs, mock.patch.object(
                            runner, 'prepare_formal_deferred_evaluation',
                            side_effect=prepare_formal,
                        ), mock.patch.object(
                            runner, 'evaluate_formal_deferred_trajectory',
                            side_effect=evaluate_formal,
                        ), mock.patch.object(
                            runner, '_publish_formal_deferred_result',
                            side_effect=publish_formal,
                        ):
                    result = runner.run_experiment(args)

                if invokes_consolidation:
                    self.assertEqual(len(methods[0].installations), 1)
                    self.assertEqual(methods[0].installations[0][0], 1)
                else:
                    self.assertFalse(methods[0].installations)
                self.assertEqual(test_accesses, [(0, 1)])
                self.assertEqual(
                    formal_order,
                    (['validation', 'install', 'freeze', 'test',
                      'evaluate', 'publish']
                     if invokes_consolidation else
                     ['freeze', 'test', 'evaluate', 'publish'])
                )
                readouts.assert_not_called()
                final_probs.assert_not_called()
                self.assertEqual(len(publications), 1)
                self.assertIsNone(methods[0].trainer.dataset_ref)
                self.assertEqual(result['task_acc_history'], [
                    {
                        'step': 'event_0_CIL',
                        'per_task_accs': {'task_0': 1.0},
                        'overall_acc': 1.0,
                        'deferred_diagonal': {'task_0': 1.0},
                    },
                    {
                        'step': 'event_1_CIL',
                        'per_task_accs': {'task_0': 1.0, 'task_1': 1.0},
                        'overall_acc': 1.0,
                        'per_task_accs_taskil': {
                            'task_0': 1.0, 'task_1': 1.0,
                        },
                        'deferred_diagonal': {
                            'task_0': 1.0, 'task_1': 1.0,
                        },
                        'deferred_final': True,
                        'deferred_final_task': 'task_1',
                    },
                ])
                self.assertEqual(result['cl_metrics']['AA_final'], 1.0)
                self.assertEqual(result['cl_metrics']['BWT'], 0.0)
                comm = {'comm_rounds': 0, 'megabytes_transmitted': 0.0}
                self.assertEqual(
                    result['step_results'],
                    [
                        {
                            'event_idx': task_id, 'type': 'CIL', 'task_id': task_id,
                            'new_classes': [task_id], 'evaluation_deferred': True,
                            'train_time': 0.01, 'comm': comm,
                        }
                        for task_id in range(2)
                    ],
                )
                for event_idx in range(2):
                    checkpoint = root / 'run' / 'checkpoints' / f'event_{event_idx}_CIL.pt'
                    snapshot = root / 'run' / 'formal_snapshots' / f'event_{event_idx}_CIL.pt'
                    self.assertTrue(checkpoint.is_file())
                    self.assertTrue(snapshot.is_file())
                    checkpoint_payload = torch.load(
                        checkpoint, map_location='cpu', weights_only=True
                    )
                    self.assertEqual(
                        checkpoint_payload['schema_version'],
                        4,
                    )
                    self.assertTrue(runner._valid_tracker_checkpoint_state(
                        checkpoint_payload['tracker_state'],
                        checkpoint_payload['protocol'],
                    ))
                self.assertFalse((root / 'run' / 'checkpoints' / 'resume_latest.pt').exists())
                persisted = json.loads((root / 'run' / 'results.json').read_text())
                self.assertEqual(
                    persisted['task_acc_history'], result['task_acc_history']
                )
                self.assertEqual(persisted['step_results'], result['step_results'])

    def test_formal_internal_final_install_reuses_one_authorized_validation_cache(self):
        finalize = getattr(runner, '_finalize_formal_internal_state', None)
        self.assertTrue(callable(finalize), 'missing formal final install adapter')
        events = []
        batches = (
            (torch.tensor([[0.0], [1.0]]), torch.tensor([0, 1])),
            (torch.tensor([[2.0], [3.0]]), torch.tensor([2, 3])),
        )
        manifest = {'sha256': 'a' * 64, 'by_class': {
            str(class_id): [class_id] for class_id in range(4)
        }}

        class Dataset:
            validation_manifest = manifest

            def authorize_formal_access(self, **record):
                events.append(('authorize', record))

            def get_validation_loader(self, classes):
                events.append(('validation_loader', tuple(classes)))
                return iter(batches)

        class Method:
            head_consolidation_mode = 'adaptive_dual_branch'

            def set_head_validation_provider(self, loader, get_manifest):
                self.loader = loader
                self.get_manifest = get_manifest

            def _consolidate_head(self, task_id):
                events.append(('install', task_id))
                first = tuple(self.loader([0, 1, 2, 3]))
                second = tuple(self.loader([0, 1, 2, 3]))
                self.cached_twice = (first, second)
                self.manifest = self.get_manifest()
                return {'installed': True}

        args = SimpleNamespace(
            formal_deferred_evaluation=True,
            cl_method='proto_evolve', head_consolidation_enabled=1,
            head_consolidation_mode='adaptive_dual_branch',
        )
        method = Method()
        cache = finalize(
            args=args, dataset=Dataset(), cl_method=method,
            task_classes={0: [0, 1], 1: [2, 3]},
            final_event=1, final_task=1,
        )
        self.assertIsInstance(cache, tuple)
        self.assertEqual([event[0] for event in events], [
            'authorize', 'validation_loader', 'install',
        ])
        self.assertEqual(events[0][1], {
            'split': 'validation', 'phase': 'final_validation_pre_install',
            'event_idx': 1, 'task_id': 1,
            'timeline_step': 'event_1_CIL', 'classes': [0, 1, 2, 3],
        })
        self.assertEqual(method.manifest, manifest)
        self.assertEqual(len(method.cached_twice[0]), 2)
        self.assertTrue(all(
            torch.equal(left_x, right_x) and torch.equal(left_y, right_y)
            for (left_x, left_y), (right_x, right_y)
            in zip(*method.cached_twice)
        ))

    def test_formal_ul_timeline_is_rejected_before_dataset_construction(self):
        class FormalUlTasks:
            def __init__(self, _args):
                pass

            def get_timeline(self):
                return [
                    {'type': 'CIL', 'task_id': 0, 'new_classes': [0]},
                    {'type': 'UL', 'forget_classes': [0]},
                ]

        args = SimpleNamespace(
            cl_method='er', ul_method='none', data='synthvfl', seed=42,
            deterministic=0, num_workers=0, formal_deferred_evaluation=True,
        )
        with mock.patch.object(runner, 'TaskManager', FormalUlTasks), \
                mock.patch.object(
                    runner, 'VFLDataset',
                    side_effect=AssertionError('dataset constructed before UL rejection'),
                ) as dataset:
            with self.assertRaisesRegex(ValueError, 'UL'):
                runner.run_experiment(args)
        dataset.assert_not_called()

    def test_formal_task4_methods_pass_the_deferred_capability_guard(self):
        for cl_method, bic_enabled in (
                ('fedprotip_vfl', 0), ('er', 1)):
            with self.subTest(cl_method=cl_method, bic_enabled=bic_enabled):
                args = SimpleNamespace(
                    cl_method=cl_method, ul_method='none', data='cifar100',
                    seed=42, deterministic=0,
                    formal_deferred_evaluation=True,
                    head_consolidation_enabled=0,
                    head_consolidation_mode='full_classifier',
                    bic_enabled=bic_enabled,
                    num_tasks=10, bic_fit_mode='joint_each_stage',
                    lambda_validation_enabled=1,
                )
                with mock.patch.object(
                        runner, 'TaskManager',
                        side_effect=AssertionError('passed Task 4 guard')) \
                        as tasks, mock.patch.object(
                            runner, 'VFLDataset',
                            side_effect=AssertionError(
                                'unsupported formal method accessed data'
                            )) as dataset:
                    with self.assertRaisesRegex(AssertionError, 'passed Task 4 guard'):
                        runner.run_experiment(args)
                tasks.assert_called_once_with(args)
                dataset.assert_not_called()


if __name__ == '__main__':
    unittest.main()
