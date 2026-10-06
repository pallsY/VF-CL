"""Behavioral and security tests for formal historical admission."""
import copy
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import weakref
from unittest import mock

import numpy as np
import torch

import adaptive_consolidation_audit as consolidation_audit
import three_dataset_formal_driver as formal_driver
import three_dataset_formal_audit as formal_audit
import three_dataset_formal_registry as formal_registry
import adaptive_head_consolidation as adaptive_head
from adaptive_consolidation_audit import (
    _formal_file_record, save_deferred_cil_snapshot,
)
from adaptive_dual_branch_validation import _ablation_runtime
from adaptive_tinyimagenet_heldout import _fixed_branch_runtime
from bic_calibration import TaskAffineCalibrator
from calibration_split import build_manifest
from cl_methods import get_cl_method
from config import get_config
from data_utils import TaskManager, VFLDataset
from metrics import MetricsTracker
from models import TopModel, build_models
from head_consolidation import freeze_state, hash_top_state
from runner import (
    _atomic_torch_save, _capture_rng_state, _encode_checkpoint_value,
    _evaluate_formal_from_single_access, _publish_formal_deferred_result,
    _save_cil_checkpoint,
)
from three_dataset_formal_registry import (
    FormalSpec, command_for, protocol_for, registry_sha256,
    validation_access_for,
)
from vfl_trainer import VFLTrainer

from three_dataset_formal_audit import (
    AdmissionRecord, _EvidenceError, _task_classes,
    _validate_bic_checkpoint_state, _validate_results, audit_candidate,
    write_admission,
)


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree_sha256(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob('*')):
        if path.is_file() and not path.is_symlink():
            relative = path.relative_to(root).as_posix().encode()
            content = path.read_bytes()
            digest.update(len(relative).to_bytes(8, 'big'))
            digest.update(relative)
            digest.update(len(content).to_bytes(8, 'big'))
            digest.update(content)
    return digest.hexdigest()


def _bic_summary(task_id):
    keys = [f'task_{value}' for value in range(task_id + 1)]
    return {
        'overall_accuracy': .5,
        'per_task_accuracy': {key: .5 for key in keys},
        'task_il': {key: .75 for key in keys},
        'task_prediction_fraction': {
            key: 1.0 / len(keys) for key in keys
        },
        'ece': .1,
    }


def _valid_bic_history(protocol, calibration, fit_corpus):
    options = protocol['base_options']
    history = []
    for task_id in range(options['num_tasks']):
        raw = _bic_summary(task_id)
        history.append({
            'step': f'event_{task_id}_CIL',
            'task_id': task_id,
            'fit': {
                'mode': 'joint_alpha_beta',
                'lr': options['bic_lr'],
                'steps': options['bic_steps'],
                'loss_before': 1.0,
                'loss_after': .5,
            },
            'paired': {
                'raw': raw,
                'calibrated': copy.deepcopy(raw),
            },
            'parameters': {
                str(value): {'alpha': 1.0, 'beta': 0.0}
                for value in range(task_id + 1)
            },
            'calibration_audit': copy.deepcopy(calibration),
            'fit_corpus': copy.deepcopy(fit_corpus),
            'disabled_identity_max_abs_diff': 0.0,
            'task_il_max_abs_delta': 0.0,
            'privacy_audit': {
                'passed': True,
                'test_used_for_fit': False,
                'raw_images_saved': False,
                'party_embeddings_saved': False,
            },
        })
    return history


def _cifar_results(protocol, calibration_sha='a' * 64,
                   validation_sha='b' * 64):
    tracker = MetricsTracker()
    tasks = _task_classes(protocol)
    diagonals = {}
    for index, classes in enumerate(tasks):
        step = f'event_{index}_CIL'
        final = index == len(tasks) - 1
        values = ({f'task_{task}': .5 for task in range(index + 1)}
                  if final else {f'task_{index}': .5})
        comm = {'comm_rounds': 0, 'megabytes_transmitted': 0.0}
        tracker.record_task_accuracies(
            step, values, .5, per_task_taskil=(values if final else None))
        diagonals[f'task_{index}'] = .5
        tracker.task_acc_matrix[-1]['deferred_diagonal'] = (
            dict(diagonals) if final else dict(values)
        )
        if final:
            tracker.task_acc_matrix[-1].update({
                'deferred_final': True,
                'deferred_final_task': f'task_{index}',
            })
        tracker.record_comm(step, comm)
        tracker.record_timing(step, 0.0)
        tracker.record_step({
            'event_idx': index, 'type': 'CIL', 'task_id': index,
            'new_classes': list(classes), 'evaluation_deferred': True,
            'train_time': 0.0, 'comm': comm,
        })
    calibration = {
        'passed': True, 'manifest_sha256': calibration_sha,
        'per_class': 25, 'calibration_count': 2500,
        'training_count': 45000, 'overlap_count': 0,
        'test_used_for_fit': False,
    }
    fit_corpus = {
        'schema_version': 1,
        'split': 'calibration',
        'phase': 'final_bic_calibration_post_freeze',
        'cache_identity': {
            'batch_count': 1, 'sample_count': 1,
            'batches': [{}],
        },
        'manifest_sha256': calibration_sha,
        'selection_manifest_key': 'calibration_manifest_sha256',
    }
    history = _valid_bic_history(protocol, calibration, fit_corpus)
    results = tracker.to_dict()
    results.update({
        'config': {
            'cl_method': protocol['base_options']['cl_method'],
            'ul_method': protocol['base_options']['ul_method'],
            'data': 'cifar100', 'num_tasks': 10, 'seed': protocol['seed'],
        },
        'bic_history': history, 'bic_final': history[-1],
        'calibration_audit': calibration,
        'bic_fit_corpus': fit_corpus,
        'selection_audit': {
            'passed': True,
            'calibration_manifest_sha256': calibration_sha,
            'validation_manifest_sha256': validation_sha,
            'calibration_per_class': 25, 'validation_per_class': 25,
            'training_count': 45000, 'calibration_count': 2500,
            'validation_count': 2500,
            'training_calibration_overlap_count': 0,
            'training_validation_overlap_count': 0,
            'calibration_validation_overlap_count': 0,
            'evaluation_source': 'cifar100-train-validation',
            'test_used_for_selection': False,
        },
    })
    results['source_provenance'] = {
        'schema_version': 1, 'source_commit': '0' * 40,
        'source_sha256': {},
    }
    return results


class _Exploit:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return os.system, (f'touch {self.marker}',)


PRODUCER_SOURCES = (
    'adaptive_consolidation_audit.py', 'adaptive_dual_branch_validation.py',
    'adaptive_head_consolidation.py', 'adaptive_tinyimagenet_heldout.py',
    'bic_calibration.py', 'calibration_split.py', 'cl_methods/__init__.py',
    'cl_methods/adagauss.py', 'cl_methods/afc.py', 'cl_methods/der_pp.py',
    'cl_methods/er.py', 'cl_methods/er_ace.py', 'cl_methods/ewc.py',
    'cl_methods/fedprotip_vfl.py', 'cl_methods/finetune.py',
    'cl_methods/gpm.py', 'cl_methods/lwf.py', 'cl_methods/lwf_wa.py',
    'cl_methods/proto_evolve.py', 'cl_methods/proto_fedspace.py',
    'cl_methods/target.py', 'config.py', 'data_utils.py', 'determinism.py',
    'head_consolidation.py', 'main.py', 'metrics.py', 'models.py', 'runner.py',
    'three_dataset_formal_runtime.py', 'ul_methods/__init__.py',
    'ul_methods/retrain.py', 'vfl_trainer.py',
)


class RealProducerFixture:
    """Bounded command/config/runner producer over temporary vector data."""
    def __init__(self, method='finetune'):
        self.temporary = tempfile.TemporaryDirectory(prefix='formal_real_audit_')
        self.root = Path(self.temporary.name)
        self.data = self.root / 'data'
        self.source = Path(__file__).resolve().parent
        self.formal_root = self.root / 'formal'
        self.runs = self.formal_root / 'runs'
        (self.data / 'isolet').mkdir(parents=True)
        self.runs.mkdir(parents=True)
        self.spec = FormalSpec(
            'isolet', method, 42,
            method in {'fixed_half', 'sample_mean_nll'},
        )
        self.normalized_spec = asdict(self.spec)
        self.protocol = json.loads(json.dumps(protocol_for(self.spec)))
        self.code_commit = subprocess.run(
            ['git', '-C', str(self.source), 'rev-parse', 'HEAD'],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        self.npz = self.data / 'isolet' / 'isolet_vfl.npz'
        self.metadata = self.data / 'isolet' / 'isolet_vfl.metadata.json'
        self._write_vector_data()
        self.command = self._synthetic_command()
        self.args = self._parse_command()
        self.run = Path(self.args.output_dir)
        (self.run / 'validation').mkdir()
        self.config = self.run / 'config.json'
        self.results = self.run / 'results.json'
        self.data_flow = self.run / 'data_flow_audit.jsonl'
        self.manifest = self.run / 'validation' / 'validation_manifest.json'
        labels = [label for label in range(26) for _ in range(4)]
        self.manifest_value = build_manifest(
            labels, per_class=1, seed=self.protocol['validation_split_seed'],
            dataset='isolet_vfl.npz-train',
        )
        self._write_json(self.manifest, self.manifest_value)
        self._build_real_evidence()
        self.declaration = self._declaration()
        self._authoritative_data = {
            'isolet/isolet_vfl.npz': _sha256(self.npz),
            'isolet/isolet_vfl.metadata.json': _sha256(self.metadata),
        }
        self._authoritative_manifest = {
            'logical_path': 'validation/validation_manifest.json',
            'dataset': 'isolet_vfl.npz-train',
            'sha256': self.manifest_value['sha256'],
            'seed': self.protocol['validation_split_seed'],
            'per_class': self.manifest_value['per_class'],
        }

    def close(self):
        self.temporary.cleanup()

    def _write_json(self, path, value):
        Path(path).write_bytes(_json_bytes(value))

    def _write_vector_data(self):
        rows = 26 * 5
        X = np.arange(rows * 8, dtype=np.float32).reshape(rows, 8) / 1000
        y = np.repeat(np.arange(26, dtype=np.int64), 5)
        train_idx = np.array([
            label * 5 + offset for label in range(26) for offset in range(4)
        ], dtype=np.int64)
        test_idx = np.array([label * 5 + 4 for label in range(26)],
                            dtype=np.int64)
        np.savez(
            self.npz, X=X, y=y, train_idx=train_idx, test_idx=test_idx,
            view_names=np.array(['v0', 'v1', 'v2', 'v3']),
            range_lo=np.array([0, 2, 4, 6], dtype=np.int64),
            range_hi=np.array([2, 4, 6, 8], dtype=np.int64),
        )
        self._write_json(self.metadata, {
            'dataset': 'isolet', 'num_classes': 26,
            'views': ['v0', 'v1', 'v2', 'v3'],
        })

    def _synthetic_command(self):
        with mock.patch.object(
                formal_registry, '_deployment_paths', return_value=(
                    self.root, self.source, Path(sys.executable))):
            return command_for(
                self.spec, 'cpu', str(self.runs), smoke=False)

    def _parse_command(self):
        option_start = self.command.index('--data')
        argv = ['main.py', *self.command[option_start:]]
        with mock.patch.object(sys, 'argv', argv):
            args = get_config()
        # Bounded synthetic producer: loader workers do not change the frozen
        # scientific protocol or deterministic batch identities.
        args.num_workers = 0
        args.lambda_validation_per_class = 1
        args.resume_run_dir = args.output_dir
        return args

    def _task_classes(self):
        return tuple(tuple(classes)
                     for classes in self.protocol['base_options']['custom_tasks'])

    def _timeline(self):
        return [
            {'step': f'event_{index}_CIL', 'task_id': index,
             'classes': list(classes)}
            for index, classes in enumerate(self._task_classes())
        ]

    def _build_real_evidence(self):
        args = self.args
        dataset = VFLDataset(args)
        task_manager = TaskManager(args)
        bottoms, top = build_models(args)
        trainer = VFLTrainer(bottoms, top, args)
        trainer.dataset_ref = dataset
        method = get_cl_method(args.cl_method, trainer, args)
        tracker = MetricsTracker()
        seen = {}
        snapshot_paths = []
        for index, classes in enumerate(self._task_classes()):
            step = f'event_{index}_CIL'
            seen[index] = list(classes)
            comm = {'comm_rounds': 0, 'megabytes_transmitted': 0.0}
            tracker.record_comm(step, comm)
            tracker.record_timing(step, 0.0)
            tracker.record_step({
                'event_idx': index, 'type': 'CIL', 'task_id': index,
                'new_classes': list(classes),
                'evaluation_deferred': True,
                'train_time': 0.0, 'comm': comm,
            })
            loader = dataset.get_train_loader(classes)
            for epoch in range(args.epochs_per_task):
                if (args.cl_method == 'proto_evolve' and index > 0
                        and method.use_sdc and epoch % method.sdc_interval == 0):
                    for _batch in loader:
                        pass
                for batch_idx, (batch_x, _batch_y) in enumerate(loader):
                    trainer._record_data_flow(loader, batch_idx, batch_x)
            if args.cl_method == 'fedprotip_vfl':
                method.before_task(index, classes, [
                    class_id for task in self._task_classes()[:index + 1]
                    for class_id in task
                ])
                method.after_task(loader, index)
            elif args.cl_method == 'proto_evolve':
                method.before_task(index, classes, [
                    class_id for task in self._task_classes()[:index + 1]
                    for class_id in task
                ])
            _save_cil_checkpoint(
                trainer, method, args, step, index, classes, seen,
                tracker_state=tracker.to_dict(), force=True,
            )
            snapshot_paths.append(Path(save_deferred_cil_snapshot(
                trainer, method, args, index, index, seen,
                protocol_kind='formal',
            )))
        all_classes = tuple(
            class_id for classes in self._task_classes()
            for class_id in classes)
        final_task = len(self._task_classes()) - 1
        boundary = {
            'event_idx': final_task,
            'task_id': final_task,
            'timeline_step': f'event_{final_task}_CIL',
            'classes': list(all_classes),
        }
        if validation_access_for(self.spec):
            dataset.authorize_formal_access(
                split='validation', phase='final_validation_pre_install',
                **boundary,
            )
            next(iter(dataset.get_validation_loader(all_classes)))
            self._install_internal_evidence(trainer, method)
        source_checkpoint = self.run / 'checkpoints' / 'event_12_CIL.pt'
        self.checkpoint = self.run / 'checkpoints' / 'formal_final.pt'
        final_payload = torch.load(
            source_checkpoint, map_location='cpu', weights_only=True
        )
        if validation_access_for(self.spec):
            final_payload['trainer_state'] = _encode_checkpoint_value(
                trainer.get_state())
            final_payload['cl_state'] = _encode_checkpoint_value(
                method.get_state())
            final_payload['rng_state'] = _encode_checkpoint_value(
                _capture_rng_state())
        _atomic_torch_save(final_payload, self.checkpoint)
        with self._internal_runtime():
            deferred, status = _evaluate_formal_from_single_access(
                args=args, dataset=dataset, snapshot_paths=snapshot_paths,
                final_checkpoint=self.checkpoint,
                task_classes=seen, final_event=final_task,
                final_task=final_task,
            )
            if status != 'complete':
                raise RuntimeError(
                    f'unexpected formal producer status: {status}')
            tracker.load_dict(deferred)
            results = tracker.to_dict()
            results['selection_audit'] = dataset.selection_audit()
            results['config'] = {
                'cl_method': args.cl_method, 'ul_method': args.ul_method,
                'data': args.data, 'num_tasks': args.num_tasks,
                'seed': args.seed,
            }
            _publish_formal_deferred_result(
                args=args, final_checkpoint=self.checkpoint,
                tracker_state=tracker.to_dict(), final_result=results,
            )

    def _internal_runtime(self):
        fixed = {'fixed_full': 'full', 'fixed_bias': 'bias'}
        if self.spec.method in fixed:
            return _fixed_branch_runtime(fixed[self.spec.method])
        ablation = {
            'fixed_half': 'fixed_half_ablation',
            'sample_mean_nll': 'sample_mean_nll',
        }.get(self.spec.method)
        return (_ablation_runtime(ablation)
                if ablation is not None else nullcontext())

    def _install_internal_evidence(self, trainer, method):
        if (self.protocol['base_options']['cl_method'] != 'proto_evolve'
                or self.spec.method == 'no_consolidation'):
            return
        classes = tuple(
            class_id for task in self._task_classes() for class_id in task)
        pre = copy.deepcopy(trainer.top_model).eval()
        pre_state = dict(freeze_state(pre.state_dict()))
        pre_hash = hash_top_state(pre_state)
        fixed = self.spec.method in {'fixed_full', 'fixed_bias'}
        branch = 'full' if self.spec.method == 'fixed_full' else 'bias'
        full_state = (pre_state if not fixed or branch == 'full'
                      else {})
        bias_state = (pre_state if not fixed or branch == 'bias'
                      else {})
        candidates = adaptive_head.FrozenAdaptiveCandidates(
            pre_head_sha256=pre_hash,
            full_state=full_state, bias_state=bias_state,
            full_head_sha256=hash_top_state(full_state),
            bias_head_sha256=hash_top_state(bias_state),
            full_audit={}, bias_audit={}, ordered_classes=classes,
        )
        logits = torch.eye(len(classes), dtype=torch.float64)
        full_log_p = torch.log_softmax(logits * 2.0, dim=1)
        bias_log_p = torch.log_softmax(logits, dim=1)
        labels = torch.tensor(classes, dtype=torch.long)
        with self._internal_runtime():
            configs = {
                'full': adaptive_head.FULL_BRANCH_CONFIG,
                'bias': adaptive_head.BIAS_BRANCH_CONFIG,
            }
            import cl_methods.proto_evolve as proto_module
            gate = proto_module.solve_global_mixture_weight(
                full_log_p, bias_log_p, labels, classes)
            install_gate = gate
            if self.spec.method in {'fixed_half', 'sample_mean_nll'}:
                install_gate = {
                    **gate, 'gate_rule': 'class_balanced',
                    'is_primary': True,
                }
            installed = proto_module.install_and_reload_verify(
                pre, candidates, install_gate)
            result = adaptive_head.AdaptiveConsolidationResult(
                pre_head_sha256=pre_hash,
                candidate_hashes={
                    'pre': pre_hash,
                    'full': hash_top_state(full_state),
                    'bias': hash_top_state(bias_state),
                },
                candidate_configs={
                    'full': configs['full'],
                    'bias': configs['bias'],
                },
                gate=gate, validation_manifest=self.manifest_value,
                ordered_classes=classes, task_id=12,
                task_boundary='event_12_CIL',
            ).to_dict()
        trainer.top_model.load_state_dict(installed.state_dict(), strict=True)
        method.head_consolidation_history = [result]
        method.head_validation_sha256 = self.manifest_value['sha256']
        method.adaptive_audit_bundle = {
            'method_version': 1, 'result': result,
            'pre_state': pre_state, 'full_state': full_state,
            'bias_state': bias_state,
            'installed_state': dict(freeze_state(installed.state_dict())),
            'event_order': [
                'candidates_frozen', 'validation_iterated', 'gate_solved',
                'state_installed', 'diagnostics_computed',
            ],
        }

    @staticmethod
    def _entry(path, root, logical_path=None):
        entry = {'path': str(path), 'root': str(root), 'sha256': _sha256(path)}
        if logical_path is not None:
            entry['logical_path'] = logical_path
        return entry

    def _declaration(self):
        results = self._entry(self.results, self.run)
        results['data_flow'] = self._entry(self.data_flow, self.run)
        return {
            'spec': self.normalized_spec, 'run_dir': str(self.run),
            'code_commit': self.code_commit,
            'source_files': [
                self._entry(self.source / name, self.source, name)
                for name in PRODUCER_SOURCES
            ],
            'data_files': [
                self._entry(self.npz, self.data, 'isolet/isolet_vfl.npz'),
                self._entry(self.metadata, self.data,
                            'isolet/isolet_vfl.metadata.json'),
            ],
            'validation_manifest': self._entry(
                self.manifest, self.run, 'validation/validation_manifest.json'
            ),
            'config': self._entry(self.config, self.run),
            'results': results,
            'checkpoint': self._entry(self.checkpoint, self.run),
            'formal_artifacts': [
                self._entry(path, self.run, path.relative_to(self.run).as_posix())
                for path in self._formal_artifact_paths()
            ],
        }

    def _formal_artifact_paths(self):
        paths = []
        for index in range(len(self._task_classes())):
            paths.extend((
                self.run / 'checkpoints' / f'event_{index}_CIL.pt',
                self.run / 'formal_snapshots' / f'event_{index}_CIL.pt',
            ))
        splits = (['validation'] if validation_access_for(self.spec) else [])
        splits.append('test')
        paths.extend(
            self.run / 'formal_access' / f'{split}.consumed.json'
            for split in splits
        )
        paths.extend(self.run / name for name in (
            'FORMAL_STATE_FROZEN.json',
            'FORMAL_EVALUATION_PENDING.json',
            'FORMAL_EVALUATION_CONSUMING.json',
            'FORMAL_EVALUATION_COMPLETE.json',
            'FORMAL_EVALUATION_SEALED.json',
            'FORMAL_EVALUATION_PUBLISHING.json',
            'FORMAL_EVALUATION_PUBLISHED.json',
        ))
        return paths

    def authoritative_data(self):
        return copy.deepcopy(self._authoritative_data)

    def authoritative_manifest(self):
        return copy.deepcopy(self._authoritative_manifest)

    def _valid_results(self):
        return json.loads(self.results.read_text())

    def _write_results(self, value):
        self._write_json(self.results, value)

    def _valid_checkpoint(self):
        return torch.load(self.checkpoint, map_location='cpu', weights_only=True)

    def _write_checkpoint(self, value):
        torch.save(value, self.checkpoint)

    def refresh(self, name):
        target = self.declaration[name]
        target['sha256'] = _sha256(target['path'])

    def refresh_formal(self, logical):
        entry = next(
            item for item in self.declaration['formal_artifacts']
            if item['logical_path'] == logical
        )
        entry['sha256'] = _sha256(entry['path'])

    @staticmethod
    def _control_write(path, value):
        path.write_bytes(_json_bytes(value) + b'\n')
        path.chmod(0o444)

    @staticmethod
    def install_formal_bundle(root, census, plan):
        root.mkdir(parents=True, exist_ok=True)
        (root / 'runs').mkdir(exist_ok=True)
        formal_driver.install_json_exclusive(
            root / 'FORMAL_REGISTRY.json', formal_driver._registry_payload())
        formal_driver.install_json_exclusive(
            root / 'COMPATIBILITY_CENSUS.json', census)
        formal_driver.install_json_exclusive(root / 'FORMAL_PLAN.json', plan)
        formal_driver.install_json_exclusive(
            root / 'MISSING_JOBS.json',
            formal_driver._missing_jobs_payload(plan, census))
        (root / 'claims').mkdir(mode=0o700)
        formal_driver.install_formal_root_identity(root, plan)

    def install_completed_controls(self):
        if formal_registry.experiment_profile() == formal_registry.FULL_MATRIX_PROFILE:
            from test_three_dataset_full_matrix_report import FullMatrixReportTests
            from three_dataset_full_matrix_report import reuse_census
            reuse = FullMatrixReportTests.make_bundle(self, 0)
            formal_driver.install_json_exclusive(
                self.formal_root / 'FULL_MATRIX_REUSE.json', reuse)
            census = reuse_census(reuse)
        else:
            census = formal_driver.build_census({})
        plan = formal_driver.build_plan(census)
        self.install_formal_bundle(self.formal_root, census, plan)
        plan_sha256 = formal_driver._digest(plan)
        key = formal_driver.spec_key(self.spec)
        source_sha256 = {
            entry['logical_path']: entry['sha256']
            for entry in self.declaration['source_files']
        }
        root_identity = formal_driver._root_identity(self.formal_root)
        command = list(self.command)
        command_sha256 = hashlib.sha256(_json_bytes(command)).hexdigest()
        job = {
            'kind': 'formal_job_spec',
            'spec_key': key,
            'spec': self.normalized_spec,
            'registry_sha256': registry_sha256(),
            'metric_formula_version': formal_audit.FORMULA_VERSION,
            'plan_sha256': plan_sha256,
            'run_dir': str(self.run),
            'source_commit': self.code_commit,
            'source_sha256': source_sha256,
            'command': command,
            'command_sha256': command_sha256,
            'root_identity': root_identity,
        }
        owner = {
            'kind': 'formal_job_claim',
            'job': key,
            'launcher_token': 'temporary-producer-token',
            'worker_role': 'formal-worker',
            'phase': ('explanation' if self.spec.explanation else 'formal'),
            'pid': os.getpid(),
            'pgid': os.getpgid(os.getpid()),
            'process_start_time': formal_driver._process_start_time(os.getpid()),
            'source_commit': self.code_commit,
            'root_identity': root_identity,
        }
        if formal_driver.claim_next(
                plan, self.formal_root / 'claims', owner['phase'], owner) != key:
            raise RuntimeError('temporary formal claim was not installed')
        installed_claim = (
            self.formal_root / 'claims' / formal_driver._claim_name(key))
        started = formal_driver.mark_claim_started(
            installed_claim, owner, self.run, plan, command_sha256)
        job_path = self.run / 'FORMAL_JOB_SPEC.json'
        claim_path = self.run / 'CLAIM_OWNER.json'
        launch_path = self.run / 'LAUNCH_STARTED.json'
        log_path = self.run / 'job.log'
        self._control_write(job_path, job)
        self._control_write(claim_path, owner)
        job_sha256 = _sha256(job_path)
        claim_sha256 = _sha256(claim_path)
        launch = {
            'kind': 'formal_launch_started',
            'spec_key': key,
            'plan_sha256': plan_sha256,
            'job_spec_sha256': job_sha256,
            'claim_sha256': claim_sha256,
            'command_sha256': command_sha256,
            'source_commit': self.code_commit,
            'root_identity': root_identity,
            'worker_role': owner['worker_role'],
            'phase': owner['phase'],
            'pid': owner['pid'],
            'pgid': owner['pgid'],
            'process_start_time': owner['process_start_time'],
        }
        self._control_write(launch_path, launch)
        log_path.write_bytes(b'formal temporary producer completed\n')
        log_path.chmod(0o444)
        artifact_paths = [
            self.config, self.results, self.data_flow, self.manifest,
            self.checkpoint, log_path, *self._formal_artifact_paths(),
        ]
        artifact_sha256 = {
            path.relative_to(self.run).as_posix(): _sha256(path)
            for path in artifact_paths
        }
        resource = {
            'hardware_identity': {
                'gpu_name': 'temporary-test-gpu',
                'gpu_count': 1,
                'cuda': 'temporary-cuda',
                'torch': str(torch.__version__),
                'driver': 'temporary-driver',
            },
            'instrumentation': 'formal-resource-v1',
            'runtime_seconds': 1.25,
            'peak_gpu_memory_bytes': 1024,
            'checkpoint_size_bytes': self.checkpoint.stat().st_size,
            'added_parameters': 0,
            'communication_bytes': 0,
            'replay_type': 'none',
            'raw_examples_per_class': 0,
            'persistent_embeddings': 0,
            'privacy_label': 'no-persistent-raw-or-embedding-replay',
        }
        launch_sha256 = _sha256(launch_path)
        evidence = {
            'kind': 'formal_resource_evidence',
            'spec_key': key,
            'plan_sha256': plan_sha256,
            'job_spec_sha256': job_sha256,
            'claim_sha256': claim_sha256,
            'launch_sha256': launch_sha256,
            'command_sha256': command_sha256,
            'artifact_sha256': artifact_sha256,
            'resource': resource,
        }
        resource_path = self.run / 'RESOURCE_EVIDENCE.json'
        self._control_write(resource_path, evidence)

        artifact_ns = max(path.stat().st_mtime_ns for path in artifact_paths)
        control_ns = min(path.stat().st_mtime_ns for path in artifact_paths) - 3
        for offset, path in enumerate((job_path, claim_path, launch_path)):
            os.utime(path, ns=(control_ns + offset, control_ns + offset))
        os.utime(resource_path, ns=(artifact_ns + 1, artifact_ns + 1))
        return {
            'census': census, 'plan': plan, 'job': job, 'owner': owner,
            'launch': launch, 'started': started,
            'evidence': evidence, 'job_path': job_path,
            'claim_path': claim_path, 'launch_path': launch_path,
            'resource_path': resource_path, 'artifact_paths': artifact_paths,
            'formal_root': self.formal_root,
            'installed_claim': installed_claim,
        }


class ProducerConfigSchemaTests(unittest.TestCase):
    def emitted_config(self, spec):
        from three_dataset_formal_runtime import variant_runtime

        with tempfile.TemporaryDirectory(prefix='formal_config_schema_') as root:
            command = command_for(spec, 'cpu', root)
            argv = ['main.py', *command[command.index('--data'):]]
            runtime = (nullcontext() if spec.method in formal_registry.EXTERNAL_METHODS
                       else variant_runtime(spec.method))
            # Explanation commands require the parser's explicit authorization;
            # match the existing formal registry command/config contract test.
            environment = ({'VFCL_REVIEWED_ADAPTIVE_ABLATION': '1'}
                           if spec.explanation else {})
            with mock.patch.object(sys, 'argv', argv), mock.patch.dict(os.environ, environment), \
                    redirect_stdout(io.StringIO()), runtime:
                args = get_config()
            return json.loads((Path(args.output_dir) / 'config.json').read_text())

    def test_all_registered_commands_emit_auditable_config_types(self):
        specs = (*formal_registry.formal_specs(), *formal_registry.explanation_specs())
        self.assertEqual(87, len(specs))
        for spec in specs:
            with self.subTest(spec=spec):
                config = self.emitted_config(spec)
                if spec.dataset == 'cifar100':
                    self.assertNotIn('party_col_ranges', config)
                else:
                    self.assertIn('party_col_ranges', config)
                    self.assertIsNone(config['party_col_ranges'])
                formal_audit._validate_producer_config_types(config, protocol_for(spec))

    def test_dataset_specific_field_presence_and_types_remain_strict(self):
        for dataset in formal_registry.DATASETS:
            spec = FormalSpec(dataset, 'finetune', 43)
            original = self.emitted_config(spec)
            if dataset == 'cifar100':
                mutations = [('unexpected_null', {**original, 'party_col_ranges': None}),
                             ('unexpected_ranges', {**original, 'party_col_ranges': [[0, 2]]})]
            else:
                missing = dict(original)
                del missing['party_col_ranges']
                mutations = [('missing_ranges', missing)]
                mutations.extend((f'wrong_ranges_{index}', {**original, 'party_col_ranges': value})
                                 for index, value in enumerate(([], [[0, 2]], 0, False, '')))
            for name, config in mutations:
                with self.subTest(dataset=dataset, mutation=name):
                    with self.assertRaises(_EvidenceError):
                        formal_audit._validate_producer_config_types(config, protocol_for(spec))

    def test_other_fields_and_trusted_dataset_stay_strict(self):
        for dataset in formal_registry.DATASETS:
            spec = FormalSpec(dataset, 'finetune', 44)
            original = self.emitted_config(spec)
            missing = dict(original)
            del missing['seed']
            spoofed = dict(original)
            if dataset == 'cifar100':
                spoofed.update(data='tabvfl', party_col_ranges=None)
            else:
                spoofed['data'] = 'cifar100'
                del spoofed['party_col_ranges']
            for config in (missing, {**original, 'unknown': None},
                           {**original, 'seed': True},
                           {**original, 'oracle_lr': 1}, spoofed):
                with self.subTest(dataset=dataset, config_change=set(config) ^ set(original)):
                    with self.assertRaises(_EvidenceError):
                        formal_audit._validate_producer_config_types(config, protocol_for(spec))


class ImageProbeLoaderTests(unittest.TestCase):
    def test_probe_replays_image_augmentation_with_producer_workers(self):
        from types import SimpleNamespace
        from torchvision import datasets, transforms
        from data_utils import make_deterministic_loader
        from determinism import tensor_sha256

        spec = FormalSpec('cifar100', 'finetune', 43)
        protocol = protocol_for(spec)
        manifest = {'per_class': 25, 'seed': protocol['validation_split_seed']}
        probe = formal_audit._probe_args(protocol, manifest, Path('/unused'), {})
        producer = SimpleNamespace(seed=43, batch_size=4,
                                   num_workers=protocol['base_options']['num_workers'])
        probe.batch_size = producer.batch_size
        dataset = datasets.FakeData(size=8, image_size=(3, 32, 32), num_classes=2,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(32, padding=4),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                   ]))
        def batches(args):
            loader = make_deterministic_loader(
                dataset, list(range(8)), args, ('train', (0, 1)), True, audit=True)
            hashes = [tensor_sha256(x) for _epoch in range(2) for x, _y in loader]
            return hashes, loader.audit_sampler.epoch_orders

        expected, expected_orders = batches(producer)
        actual, actual_orders = batches(probe)
        self.assertEqual(expected_orders, actual_orders)
        self.assertEqual(expected, actual)

    def test_vector_probe_preserves_zero_worker_behavior(self):
        for dataset in ('isolet', 'upmc_food101'):
            protocol = protocol_for(FormalSpec(dataset, 'finetune', 42))
            manifest = {'per_class': 1, 'seed': protocol['validation_split_seed']}
            with self.subTest(dataset=dataset):
                args = formal_audit._probe_args(protocol, manifest, Path('/unused'), {})
                self.assertEqual(0, args.num_workers)


class FormalAuditTests(unittest.TestCase):
    def test_full_matrix_factory_sources_are_in_formal_inventory(self):
        expected = {
            'finetune': 'cl_methods/finetune.py',
            'lwf': 'cl_methods/lwf.py',
            'ewc': 'cl_methods/ewc.py',
            'er': 'cl_methods/er.py',
            'der_pp': 'cl_methods/der_pp.py',
            'er_ace': 'cl_methods/er_ace.py',
            'gpm': 'cl_methods/gpm.py',
            'fedprotip_vfl': 'cl_methods/fedprotip_vfl.py',
            'target': 'cl_methods/target.py',
            'afc': 'cl_methods/afc.py',
            'lwf_wa': 'cl_methods/lwf_wa.py',
            'adagauss': 'cl_methods/adagauss.py',
            'proto_fedspace': 'cl_methods/proto_fedspace.py',
            'adaptive': 'cl_methods/proto_evolve.py',
        }
        self.assertEqual(tuple(expected), formal_registry.FULL_MATRIX_METHODS)
        source = Path(__file__).resolve().parent
        actual = {}
        with mock.patch.dict(os.environ, {
                'VFCL_EXPERIMENT_PROFILE': formal_registry.FULL_MATRIX_PROFILE}):
            for method in formal_registry.FULL_MATRIX_METHODS:
                options = protocol_for(FormalSpec('isolet', method, 42))['base_options']
                args = SimpleNamespace(**{**options, 'device': 'cpu'})
                trainer = SimpleNamespace(
                    top_model=TopModel(2, args.num_classes), evaluate=None)
                instance = get_cl_method(args.cl_method, trainer, args)
                actual[method] = Path(inspect.getfile(type(instance))).resolve().relative_to(
                    source).as_posix()
        self.assertEqual(expected, actual)
        inventory = consolidation_audit.FORMAL_SOURCE_FILES
        missing = sorted(set(actual.values()) - set(inventory))
        self.assertFalse(missing, f'full-matrix source inventory is missing: {missing}')
        self.assertIsInstance(inventory, tuple)
        self.assertEqual(inventory, tuple(sorted(set(inventory))))
        self.assertEqual(inventory, formal_audit._SOURCE_INVENTORY)
        tracked = subprocess.run(
            ['git', '-C', str(source), 'ls-files', '--stage', '-z'],
            check=True, capture_output=True, text=True,
        ).stdout.split('\0')
        regular = {
            entry.split('\t', 1)[1]
            for entry in tracked if entry.startswith(('100644 ', '100755 '))
        }
        for logical in inventory:
            with self.subTest(logical=logical):
                path = source / logical
                self.assertFalse(Path(logical).is_absolute())
                self.assertNotIn('..', Path(logical).parts)
                self.assertIn(logical, regular)
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())

    def test_continuation_completed_plan_binding_is_dataset_scoped(self):
        with mock.patch.dict(os.environ, {
                'VFCL_EXPERIMENT_PROFILE':
                    'single-dataset-verified-continuation-v1',
                'VFCL_FORMAL_DATASET': 'cifar100'}):
            self.assertEqual(formal_audit._completed_profile_binding(), {
                'experiment_profile': 'single-dataset-verified-continuation-v1',
                'formal_dataset': 'cifar100',
            })
        with mock.patch.dict(os.environ, {
                'VFCL_EXPERIMENT_PROFILE': 'single-dataset-full-matrix',
                'VFCL_FORMAL_DATASET': 'cifar100'}):
            self.assertEqual(formal_audit._completed_profile_binding(), {
                'experiment_profile': 'single-dataset-full-matrix',
                'formal_dataset': 'cifar100',
            })

    def test_completed_plan_schema_is_exact_for_active_profile(self):
        profile = formal_registry.experiment_profile()
        census = formal_driver.build_census({})
        plan = formal_driver.build_plan(census)
        spec = formal_registry.formal_specs()[0]
        self.assertEqual(
            formal_audit._completed_digest(plan),
            formal_audit._validate_completed_plan(plan, spec),
        )
        if profile == formal_driver.RECOVERY_PROFILE:
            for mutation in ('missing', 'wrong', 'extra'):
                forged = copy.deepcopy(plan)
                if mutation == 'missing':
                    del forged['experiment_profile']
                elif mutation == 'wrong':
                    forged['experiment_profile'] = formal_driver.PILOT_PROFILE
                else:
                    forged['unexpected_profile_field'] = profile
                with self.subTest(mutation=mutation), self.assertRaises(_EvidenceError):
                    formal_audit._validate_completed_plan(forged, spec)
        else:
            forged = {**plan, 'experiment_profile': formal_driver.RECOVERY_PROFILE}
            with self.assertRaises(_EvidenceError):
                formal_audit._validate_completed_plan(forged, spec)

    def fixture(self, method='finetune'):
        fixture = RealProducerFixture(method)
        self.addCleanup(fixture.close)
        return fixture

    def audit(self, fixture, declaration=None):
        with mock.patch(
                'three_dataset_formal_audit._SOURCE_INVENTORY',
                PRODUCER_SOURCES, create=True), mock.patch(
                'three_dataset_formal_audit._AUTHORITATIVE_DATA',
                {fixture.spec.dataset: fixture.authoritative_data()},
                create=True), mock.patch(
                'three_dataset_formal_audit._AUTHORITATIVE_MANIFEST',
                {fixture.spec.dataset: fixture.authoritative_manifest()},
                create=True):
            return audit_candidate(
                fixture.spec,
                fixture.declaration if declaration is None else declaration,
            )

    def completed_audit(self, fixture, controls):
        with mock.patch(
                'three_dataset_formal_audit._SOURCE_INVENTORY',
                PRODUCER_SOURCES, create=True), mock.patch(
                'three_dataset_formal_audit._AUTHORITATIVE_DATA',
                {fixture.spec.dataset: fixture.authoritative_data()},
                create=True), mock.patch(
                'three_dataset_formal_audit._AUTHORITATIVE_MANIFEST',
                {fixture.spec.dataset: fixture.authoritative_manifest()},
                create=True), mock.patch.object(
                formal_registry, '_deployment_paths', return_value=(
                    fixture.root, fixture.source, Path(sys.executable))):
            return formal_audit.audit_completed_run(
                fixture.spec, fixture.run, controls['plan'])

    @staticmethod
    def rewrite_control(path, value, mtime_ns=None):
        path.chmod(0o644)
        path.write_bytes(_json_bytes(value) + b'\n')
        path.chmod(0o444)
        if mtime_ns is not None:
            os.utime(path, ns=(mtime_ns, mtime_ns))

    def rewrite_completed_chain(self, controls, plan=None, command=None,
                                root_identity=None, owner_value=None):
        plan = copy.deepcopy(controls['plan'] if plan is None else plan)
        job = copy.deepcopy(controls['job'])
        owner = copy.deepcopy(
            controls['owner'] if owner_value is None else owner_value)
        launch = copy.deepcopy(controls['launch'])
        evidence = copy.deepcopy(controls['evidence'])
        if command is not None:
            job['command'] = list(command)
            job['command_sha256'] = hashlib.sha256(
                _json_bytes(job['command'])).hexdigest()
        if root_identity is not None:
            job['root_identity'] = copy.deepcopy(root_identity)
            owner['root_identity'] = copy.deepcopy(root_identity)
            launch['root_identity'] = copy.deepcopy(root_identity)
        for name in (
                'source_commit', 'root_identity', 'worker_role', 'phase',
                'pid', 'pgid', 'process_start_time'):
            launch[name] = copy.deepcopy(owner[name])
        plan_sha256 = formal_driver._digest(plan)
        job['plan_sha256'] = plan_sha256
        launch['plan_sha256'] = plan_sha256
        evidence['plan_sha256'] = plan_sha256
        launch['command_sha256'] = job['command_sha256']
        evidence['command_sha256'] = job['command_sha256']
        mtimes = {
            name: controls[f'{name}_path'].stat().st_mtime_ns
            for name in ('job', 'claim', 'launch', 'resource')
        }
        self.rewrite_control(controls['job_path'], job, mtimes['job'])
        self.rewrite_control(controls['claim_path'], owner, mtimes['claim'])
        launch['job_spec_sha256'] = _sha256(controls['job_path'])
        launch['claim_sha256'] = _sha256(controls['claim_path'])
        self.rewrite_control(
            controls['launch_path'], launch, mtimes['launch'])
        evidence['job_spec_sha256'] = launch['job_spec_sha256']
        evidence['claim_sha256'] = launch['claim_sha256']
        evidence['launch_sha256'] = _sha256(controls['launch_path'])
        self.rewrite_control(
            controls['resource_path'], evidence, mtimes['resource'])
        return {
            **controls, 'plan': plan, 'job': job, 'owner': owner,
            'launch': launch, 'evidence': evidence,
        }

    def test_full_profile_completed_run_preserves_installed_disk_reservation(self):
        with mock.patch.dict(os.environ, {
                formal_registry.PROFILE_ENV: formal_registry.FULL_MATRIX_PROFILE,
        }), mock.patch.object(os, 'statvfs', return_value=SimpleNamespace(
                f_bavail=1000 * 1024 ** 3, f_frsize=1)):
            fixture = self.fixture()
            controls = fixture.install_completed_controls()
            receipt = controls['installed_claim'] / 'disk-reservation.json'
            installed_owner, installed_started = formal_driver._read_started_claim(
                controls['installed_claim'], formal_driver.spec_key(fixture.spec),
                controls['owner']['root_identity'], fixture.code_commit)
            self.assertEqual(controls['owner'], installed_owner)
            self.assertEqual(_sha256(receipt),
                             installed_started['disk_reservation_sha256'])
            self.assertEqual(0o444, stat.S_IMODE(receipt.stat().st_mode))
            self.assertNotIn('disk_reservation_sha256', controls['launch'])
            with mock.patch.object(formal_audit, '_SOURCE_INVENTORY',
                                   PRODUCER_SOURCES), mock.patch.object(
                    formal_audit, '_AUTHORITATIVE_DATA',
                    {fixture.spec.dataset: fixture.authoritative_data()}), \
                    mock.patch.object(
                    formal_registry, '_deployment_paths', return_value=(
                        fixture.root, fixture.source, Path(sys.executable))):
                declaration = formal_audit.declaration_from_new_run(
                    fixture.spec, fixture.run, controls['plan'])
            self.assertEqual(fixture.declaration, declaration)
            admitted = self.completed_audit(fixture, controls)
            self.assertEqual('REUSABLE', admitted.status, admitted.reason)
            self.assertEqual('admitted', admitted.reason)

    def test_completed_run_uses_exact_control_path_and_real_auditor(self):
        fixture = self.fixture()
        controls = fixture.install_completed_controls()
        other_root = fixture.root / 'other-formal'
        fixture.install_formal_bundle(
            other_root, controls['census'], controls['plan'])
        with mock.patch(
                'three_dataset_formal_audit._SOURCE_INVENTORY',
                PRODUCER_SOURCES, create=True), mock.patch(
                'three_dataset_formal_audit._AUTHORITATIVE_DATA',
                {fixture.spec.dataset: fixture.authoritative_data()},
                create=True), mock.patch(
                'three_dataset_formal_audit._AUTHORITATIVE_MANIFEST',
                {fixture.spec.dataset: fixture.authoritative_manifest()},
                create=True), mock.patch.object(
                formal_registry, '_deployment_paths', return_value=(
                    fixture.root, fixture.source, Path(sys.executable))), \
                mock.patch.object(
                Path, 'glob', side_effect=AssertionError('no discovery')), \
                mock.patch.object(
                    Path, 'rglob', side_effect=AssertionError('no discovery')), \
                mock.patch.object(
                    os, 'walk', side_effect=AssertionError('no discovery')):
            declaration = formal_audit.declaration_from_new_run(
                fixture.spec, fixture.run, controls['plan'])
            admitted = formal_audit.audit_completed_run(
                fixture.spec, fixture.run, controls['plan'])
            completed = formal_driver.completed_run_record(
                fixture.spec, fixture.run, controls['plan'])
            with mock.patch('sys.stdout', io.StringIO()), \
                    self.assertRaises(ValueError):
                formal_driver.main([
                    'audit-run', '--root', str(other_root),
                    '--spec-key', formal_driver.spec_key(fixture.spec),
                    '--run-dir', str(fixture.run),
                ])
        self.assertEqual(fixture.declaration, declaration)
        self.assertEqual('REUSABLE', admitted.status, admitted.reason)
        self.assertEqual('admitted', admitted.reason)
        self.assertEqual(
            str(fixture.run / 'checkpoints' / 'formal_final.pt'),
            declaration['checkpoint']['path'],
        )
        self.assertEqual({
            'kind', 'spec_key', 'dataset', 'method', 'seed', 'explanation',
            'registry_sha256', 'metric_formula_version', 'plan_sha256',
            'source_commit', 'protocol_sha256', 'trajectory_sha256',
            'admission_record_sha256', 'artifact_sha256', 'metrics',
            'command_sha256', 'log_sha256', 'claim_sha256', 'launch_sha256',
            'resource', 'record_sha256',
        }, set(completed))
        self.assertEqual(
            completed['record_sha256'], formal_driver._digest({
                key: value for key, value in completed.items()
                if key != 'record_sha256'
            }),
        )
        self.assertEqual(
            controls['evidence']['resource'], completed['resource'])

    def test_published_checkpoint_bound_resource_preserves_completed_audit_evidence(self):
        fixture = self.fixture('finetune')
        controls = fixture.install_completed_controls()
        original = controls['evidence']
        measurements = {k: v for k, v in original['resource'].items()
                        if k not in {'instrumentation', 'checkpoint_size_bytes'}}
        path = controls['resource_path']
        path.unlink()  # Replace only this test-generated resource through the real publisher.
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            formal_driver.resource_record(fixture.formal_root,
                formal_driver.spec_key(fixture.spec), fixture.run,
                measurements={**measurements, 'added_parameters': 1})
        self.assertFalse(path.exists())
        evidence = formal_driver.resource_record(fixture.formal_root,
            formal_driver.spec_key(fixture.spec), fixture.run, measurements=measurements)
        self.assertEqual(original, evidence)
        self.assertEqual(original, json.loads(path.read_text()))
        admitted = self.completed_audit(fixture, controls)
        self.assertEqual('REUSABLE', admitted.status, admitted.reason)

    def test_completed_controls_precede_oldest_artifact_without_duration_limit(self):
        fixture = self.fixture('finetune')
        existing = [fixture.config, fixture.results, fixture.data_flow,
                    fixture.manifest, fixture.checkpoint,
                    *fixture._formal_artifact_paths()]
        forced_ns = min(path.stat().st_mtime_ns for path in existing) - 60_000_000_000
        os.utime(fixture.config, ns=(forced_ns, forced_ns))
        before = {path: path.stat().st_mtime_ns for path in existing}
        controls = fixture.install_completed_controls()
        self.assertEqual({path: path.stat().st_mtime_ns for path in existing}, before)
        artifacts = [path.stat().st_mtime_ns for path in controls['artifact_paths']]
        job_ns = controls['job_path'].stat().st_mtime_ns
        claim_ns = controls['claim_path'].stat().st_mtime_ns
        launch_ns = controls['launch_path'].stat().st_mtime_ns
        resource_ns = controls['resource_path'].stat().st_mtime_ns
        self.assertLess(job_ns, claim_ns)
        self.assertLess(claim_ns, launch_ns)
        self.assertLessEqual(launch_ns, min(artifacts))
        self.assertGreaterEqual(resource_ns, max(artifacts))

    def test_completed_run_control_and_tamper_boundaries_fail_closed(self):
        fixture = self.fixture()
        controls = fixture.install_completed_controls()
        cases = []

        job_extra = copy.deepcopy(controls['job'])
        job_extra['extra'] = True
        cases.append(('extra-job-member', controls['job_path'], job_extra, None))
        wrong_registry = copy.deepcopy(controls['job'])
        wrong_registry['registry_sha256'] = '0' * 64
        cases.append(('wrong-registry', controls['job_path'], wrong_registry, None))
        wrong_formula = copy.deepcopy(controls['job'])
        wrong_formula['metric_formula_version'] = 'tampered'
        cases.append(('wrong-formula', controls['job_path'], wrong_formula, None))
        wrong_spec = copy.deepcopy(controls['job'])
        wrong_spec['spec']['seed'] = 43
        cases.append(('wrong-spec', controls['job_path'], wrong_spec, None))
        wrong_source = copy.deepcopy(controls['job'])
        wrong_source['source_sha256'][PRODUCER_SOURCES[0]] = '0' * 64
        cases.append(('wrong-source-map', controls['job_path'], wrong_source, None))
        wrong_command = copy.deepcopy(controls['job'])
        wrong_command['command'].append('--tampered')
        cases.append(('wrong-command', controls['job_path'], wrong_command, None))
        wrong_owner = copy.deepcopy(controls['owner'])
        wrong_owner['worker_role'] = 'wrong-role'
        cases.append(('wrong-owner', controls['claim_path'], wrong_owner, None))
        wrong_launch = copy.deepcopy(controls['launch'])
        wrong_launch['process_start_time'] += 'x'
        cases.append(('wrong-launch', controls['launch_path'], wrong_launch, None))
        wrong_phase = copy.deepcopy(controls['launch'])
        wrong_phase['phase'] = (
            'formal' if controls['owner']['phase'] == 'explanation'
            else 'explanation')
        cases.append(('wrong-phase', controls['launch_path'], wrong_phase, None))
        wrong_pgid = copy.deepcopy(controls['launch'])
        wrong_pgid['pgid'] += 1
        cases.append(('wrong-pgid', controls['launch_path'], wrong_pgid, None))
        wrong_hardware = copy.deepcopy(controls['evidence'])
        wrong_hardware['resource']['hardware_identity']['gpu_count'] = True
        cases.append(('wrong-hardware', controls['resource_path'], wrong_hardware, None))
        wrong_artifact = copy.deepcopy(controls['evidence'])
        wrong_artifact['artifact_sha256']['results.json'] = '0' * 64
        cases.append(('wrong-artifact', controls['resource_path'], wrong_artifact, None))
        alternate_checkpoint = copy.deepcopy(controls['evidence'])
        digest = alternate_checkpoint['artifact_sha256'].pop(
            'checkpoints/formal_final.pt')
        alternate_checkpoint['artifact_sha256'][
            'checkpoints/alternate.pt'] = digest
        cases.append((
            'alternate-checkpoint', controls['resource_path'],
            alternate_checkpoint, None,
        ))
        for name, path, malformed, _ in cases:
            original = json.loads(path.read_text())
            original_mtime = path.stat().st_mtime_ns
            with self.subTest(name=name):
                self.rewrite_control(path, malformed, original_mtime)
                self.assertNotEqual(
                    'REUSABLE', self.completed_audit(fixture, controls).status)
                self.rewrite_control(path, original, original_mtime)

        launch_mtime = controls['launch_path'].stat().st_mtime_ns
        artifact_mtime = max(
            path.stat().st_mtime_ns for path in controls['artifact_paths'])
        os.utime(controls['launch_path'], ns=(artifact_mtime + 1,
                                                  artifact_mtime + 1))
        self.assertNotEqual(
            'REUSABLE', self.completed_audit(fixture, controls).status)
        os.utime(controls['launch_path'], ns=(launch_mtime, launch_mtime))

        job_mtime = controls['job_path'].stat().st_mtime_ns
        os.utime(controls['job_path'], ns=(launch_mtime + 1,
                                               launch_mtime + 1))
        self.assertNotEqual(
            'REUSABLE', self.completed_audit(fixture, controls).status)
        os.utime(controls['job_path'], ns=(job_mtime, job_mtime))

        controls['job_path'].chmod(0o644)
        self.assertNotEqual(
            'REUSABLE', self.completed_audit(fixture, controls).status)
        controls['job_path'].chmod(0o444)

        saved = fixture.run / '.resource.saved'
        controls['resource_path'].rename(saved)
        controls['resource_path'].symlink_to(saved)
        self.assertNotEqual(
            'REUSABLE', self.completed_audit(fixture, controls).status)
        controls['resource_path'].unlink()
        saved.rename(controls['resource_path'])

        missing = fixture.run / '.launch.saved'
        controls['launch_path'].rename(missing)
        self.assertNotEqual(
            'REUSABLE', self.completed_audit(fixture, controls).status)
        missing.rename(controls['launch_path'])

        forged_plan = copy.deepcopy(controls['plan'])
        forged_plan['census_sha256'] = '0' * 64
        reordered = list(controls['job']['command'])
        reordered[-4:] = reordered[-2:] + reordered[-4:-2]
        zero_root = {
            'dev': 0, 'inode': 0, 'ctime_ns': 0, 'size': 0,
            'hash': '0' * 64,
        }
        coherent = (
            ('installed-plan', {'plan': forged_plan}),
            ('exact-command', {'command': reordered}),
            ('actual-root', {'root_identity': zero_root}),
        )
        for name, change in coherent:
            with self.subTest(name=name):
                forged = self.rewrite_completed_chain(controls, **change)
                try:
                    self.assertNotEqual(
                        'REUSABLE', self.completed_audit(
                            fixture, forged).status)
                finally:
                    self.rewrite_completed_chain(controls)

        forged_owner = copy.deepcopy(controls['owner'])
        forged_owner['worker_role'] = 'forged-worker'
        forged = self.rewrite_completed_chain(
            controls, owner_value=forged_owner)
        try:
            self.assertNotEqual(
                'REUSABLE', self.completed_audit(fixture, forged).status)
        finally:
            self.rewrite_completed_chain(controls)

        installed_owner = controls['installed_claim'] / 'owner.json'
        installed_started = controls['installed_claim'] / 'started.json'
        for name, path in (
                ('missing-owner', installed_owner),
                ('missing-started', installed_started)):
            saved = controls['installed_claim'] / f'.{name}.saved'
            path.rename(saved)
            try:
                with self.subTest(name=name):
                    self.assertNotEqual(
                        'REUSABLE', self.completed_audit(
                            fixture, controls).status)
            finally:
                saved.rename(path)

        owner_mtime = installed_owner.stat().st_mtime_ns
        for name, value in (
                ('empty-owner', {}),
                ('stale-owner', {
                    **controls['owner'],
                    'process_start_time':
                        controls['owner']['process_start_time'] + 'x',
                })):
            with self.subTest(name=name):
                self.rewrite_control(installed_owner, value, owner_mtime)
                try:
                    self.assertNotEqual(
                        'REUSABLE', self.completed_audit(
                            fixture, controls).status)
                finally:
                    self.rewrite_control(
                        installed_owner, controls['owner'], owner_mtime)

        started_mtime = installed_started.stat().st_mtime_ns
        changed_started = {
            **controls['started'], 'command_sha256': '0' * 64,
        }
        self.rewrite_control(
            installed_started, changed_started, started_mtime)
        try:
            self.assertNotEqual(
                'REUSABLE', self.completed_audit(fixture, controls).status)
        finally:
            self.rewrite_control(
                installed_started, controls['started'], started_mtime)

        installed_claim = controls['installed_claim']
        saved_claim = installed_claim.parent / '.claim.saved'
        installed_claim.rename(saved_claim)
        installed_claim.symlink_to(saved_claim, target_is_directory=True)
        try:
            self.assertNotEqual(
                'REUSABLE', self.completed_audit(fixture, controls).status)
        finally:
            installed_claim.unlink()
            saved_claim.rename(installed_claim)

    def test_parent_releases_formal_contents_before_strict_probe(self):
        fixture = self.fixture()
        refs = []
        original = formal_audit._verify_formal_artifacts

        class Tracked(dict):
            pass

        def verify(*args, **kwargs):
            value = Tracked(original(*args, **kwargs))
            refs.append(weakref.ref(value))
            return value

        def probe(*args, **kwargs):
            self.assertEqual(len(refs), 1)
            self.assertTrue(
                refs[0]() is None, 'verified formal contents still retained'
            )

        with mock.patch('three_dataset_formal_audit._verify_formal_artifacts',
                        side_effect=verify), mock.patch(
                'three_dataset_formal_audit._probe_checkpoint',
                side_effect=probe):
            self.assertEqual('REUSABLE', self.audit(fixture).status)

    def test_real_producer_results_and_schema_v4_checkpoint_are_reusable(self):
        fixture = self.fixture()
        results = json.loads(fixture.results.read_text())
        checkpoint = fixture._valid_checkpoint()
        self.assertEqual(4, checkpoint['schema_version'])
        self.assertEqual(
            {'cl_metrics', 'task_acc_history', 'ul_metrics', 'comm_stats',
             'timing', 'step_results', 'config', 'selection_audit',
             'source_provenance'},
            set(results),
        )
        self.assertEqual('REUSABLE', self.audit(fixture).status)

    def test_cifar_producer_results_union_uses_integer_split_contract(self):
        protocol = protocol_for(FormalSpec('cifar100', 'finetune', 42))
        manifest = {
            'sha256': 'b' * 64, 'per_class': 25,
            'by_class': {str(value): [value] for value in range(100)},
        }
        results = _cifar_results(protocol)
        _validate_results(results, protocol, manifest)
        malformed = copy.deepcopy(results)
        malformed['selection_audit']['calibration_per_class'] = {'all': 25}
        with self.assertRaises(_EvidenceError):
            _validate_results(malformed, protocol, manifest)

    def test_bic_history_rejects_nonproducer_records_through_audit_candidate(self):
        protocol = protocol_for(FormalSpec('cifar100', 'finetune', 42))
        manifest = {
            'sha256': 'b' * 64, 'per_class': 25,
            'by_class': {str(value): [value] for value in range(100)},
        }
        results = _cifar_results(protocol)
        garbage = {'garbage': 'accepted-by-the-old-shape-check'}
        results['bic_history'] = [garbage]
        results['bic_final'] = garbage
        with self.assertRaises(_EvidenceError):
            _validate_results(results, protocol, manifest)

    def test_bic_cross_manifest_hash_mismatch_is_not_reusable(self):
        protocol = protocol_for(FormalSpec('cifar100', 'finetune', 42))
        manifest = {
            'sha256': 'b' * 64, 'per_class': 25,
            'by_class': {str(value): [value] for value in range(100)},
        }
        results = _cifar_results(protocol)
        results['selection_audit']['calibration_manifest_sha256'] = 'c' * 64
        with self.assertRaises(_EvidenceError):
            _validate_results(results, protocol, manifest)

    def test_bic_overall_accuracy_is_bound_to_producer_sample_counts(self):
        protocol = protocol_for(FormalSpec('cifar100', 'finetune', 42))
        manifest = {
            'sha256': 'b' * 64, 'per_class': 25,
            'by_class': {str(value): [value] for value in range(100)},
        }
        results = _cifar_results(protocol)
        results['bic_history'][0]['paired']['raw'][
            'overall_accuracy'] = .9
        with self.assertRaises(_EvidenceError):
            _validate_results(results, protocol, manifest)

    def test_bic_checkpoint_state_is_exact_and_bound_to_final_parameters(self):
        protocol = protocol_for(FormalSpec('cifar100', 'finetune', 42))
        calibrator = TaskAffineCalibrator()
        identity_raw = math.log(math.expm1(1.0 - 1e-6))
        for task_id, classes in enumerate(_task_classes(protocol)):
            calibrator.set_task(task_id, classes, identity_raw, 0.0)
        state = calibrator.state_dict()
        final = _cifar_results(protocol)['bic_final']
        final['parameters'] = {
            str(task_id): calibrator.parameters_for(task_id)
            for task_id in range(protocol['base_options']['num_tasks'])
        }
        _validate_bic_checkpoint_state(state, protocol, final)
        mismatched = copy.deepcopy(state)
        mismatched['tasks']['9']['beta'] = .25
        with self.assertRaises(_EvidenceError):
            _validate_bic_checkpoint_state(mismatched, protocol, final)

    def test_formal_bic_bundle_preserves_adaptive_internal_fit_corpus_identity(
            self):
        registered = protocol_for(FormalSpec('cifar100', 'adaptive', 42))
        protocol = registered['base_options']
        tasks = _task_classes(registered)
        identity = {
            'protocol': copy.deepcopy(protocol),
            'task_classes': {
                str(task_id): list(classes)
                for task_id, classes in enumerate(tasks)
            },
        }
        cached_batches = ((torch.zeros((1, 1)), torch.tensor([0])),)
        cache_identity = consolidation_audit._formal_cache_identity(
            cached_batches)
        calibration = {
            'passed': True, 'manifest_sha256': 'b' * 64,
            'per_class': 25, 'calibration_count': 2500,
            'training_count': 45000, 'overlap_count': 0,
            'test_used_for_fit': False,
        }
        fit_corpus = consolidation_audit._formal_bic_fit_corpus(
            identity, cached_batches, calibration, {'sha256': 'a' * 64},
            {'validation_manifest_sha256': 'a' * 64},
        )
        self.assertEqual({
            'schema_version': 1,
            'split': 'validation',
            'phase': 'final_validation_pre_install',
            'cache_identity': cache_identity,
            'manifest_sha256': 'a' * 64,
            'selection_manifest_key': 'validation_manifest_sha256',
        }, fit_corpus)
        validation_manifest = build_manifest(
            [class_id for class_id in range(100) for _ in range(25)],
            per_class=25, seed=registered['validation_split_seed'],
            dataset='cifar100-train',
        )
        selection = {
            'passed': True,
            'calibration_manifest_sha256': calibration['manifest_sha256'],
            'validation_manifest_sha256': validation_manifest['sha256'],
            'calibration_per_class': 25, 'validation_per_class': 25,
            'training_count': 45000, 'calibration_count': 2500,
            'validation_count': 2500,
            'training_calibration_overlap_count': 0,
            'training_validation_overlap_count': 0,
            'calibration_validation_overlap_count': 0,
            'evaluation_source': 'cifar100-train-validation',
            'test_used_for_selection': False,
        }
        history = _valid_bic_history(registered, calibration, fit_corpus)
        calibrator = TaskAffineCalibrator()
        identity_raw = math.log(math.expm1(1.0 - 1e-6))
        for task_id, classes in enumerate(tasks):
            calibrator.set_task(task_id, classes, identity_raw, 0.0)
        for task_id, record in enumerate(history):
            record['parameters'] = {
                str(value): calibrator.parameters_for(value)
                for value in range(task_id + 1)
            }
        bundle = {
            'state': calibrator.state_dict(),
            'history': history,
            'calibration_audit': calibration,
            'validation_manifest': validation_manifest,
            'selection_audit': selection,
            'fit_corpus': fit_corpus,
        }
        authoritative = {
            'logical_path': 'validation/validation_manifest.json',
            'dataset': validation_manifest['dataset'],
            'sha256': validation_manifest['sha256'],
            'seed': validation_manifest['seed'],
            'per_class': validation_manifest['per_class'],
        }
        with mock.patch.dict(
                formal_audit._AUTHORITATIVE_MANIFEST,
                {'cifar100': authoritative}):
            self.assertIs(
                bundle,
                consolidation_audit._validate_formal_bic_bundle(
                    bundle, identity, cache_identity),
            )

    def test_formal_bic_producer_bundle_requires_exact_corpus_discriminators(
            self):
        registered = protocol_for(FormalSpec('cifar100', 'adaptive', 42))
        protocol = registered['base_options']
        tasks = _task_classes(registered)
        identity = {
            'protocol': copy.deepcopy(protocol),
            'task_classes': {
                str(task_id): list(classes)
                for task_id, classes in enumerate(tasks)
            },
        }
        cached_batches = ((torch.zeros((1, 1)), torch.tensor([0])),)
        calibration = {
            'passed': True, 'manifest_sha256': 'b' * 64,
            'per_class': 25, 'calibration_count': 2500,
            'training_count': 45000, 'overlap_count': 0,
            'test_used_for_fit': False,
        }
        fit_corpus = consolidation_audit._formal_bic_fit_corpus(
            identity, cached_batches, calibration, {'sha256': 'a' * 64},
            {'validation_manifest_sha256': 'a' * 64},
        )
        validation_manifest = build_manifest(
            [class_id for class_id in range(100) for _ in range(25)],
            per_class=25, seed=registered['validation_split_seed'],
            dataset='cifar100-train',
        )
        selection = {
            'passed': True,
            'calibration_manifest_sha256': calibration['manifest_sha256'],
            'validation_manifest_sha256': validation_manifest['sha256'],
            'calibration_per_class': 25, 'validation_per_class': 25,
            'training_count': 45000, 'calibration_count': 2500,
            'validation_count': 2500,
            'training_calibration_overlap_count': 0,
            'training_validation_overlap_count': 0,
            'calibration_validation_overlap_count': 0,
            'evaluation_source': 'cifar100-train-validation',
            'test_used_for_selection': False,
        }
        calibrator = TaskAffineCalibrator()
        identity_raw = math.log(math.expm1(1.0 - 1e-6))
        for task_id, classes in enumerate(tasks):
            calibrator.set_task(task_id, classes, identity_raw, 0.0)
        history = _valid_bic_history(registered, calibration, fit_corpus)
        for task_id, record in enumerate(history):
            record['parameters'] = {
                str(value): calibrator.parameters_for(value)
                for value in range(task_id + 1)
            }
        arguments = {
            'history': history,
            'state': calibrator.state_dict(),
            'calibration_audit': calibration,
            'task_classes': {
                task_id: list(classes)
                for task_id, classes in enumerate(tasks)
            },
            'validation_manifest': validation_manifest,
            'selection_audit': selection,
            'fit_corpus': fit_corpus,
        }
        authoritative = {
            'logical_path': 'validation/validation_manifest.json',
            'dataset': validation_manifest['dataset'],
            'sha256': validation_manifest['sha256'],
            'seed': validation_manifest['seed'],
            'per_class': validation_manifest['per_class'],
        }
        with mock.patch.dict(
                formal_audit._AUTHORITATIVE_MANIFEST,
                {'cifar100': authoritative}):
            for key in (
                    'cl_method', 'head_consolidation_enabled',
                    'head_consolidation_mode'):
                with self.subTest(missing=key):
                    options = copy.deepcopy(protocol)
                    del options[key]
                    with self.assertRaises(KeyError):
                        formal_audit._validate_formal_bic_producer_bundle(
                            options=options, **arguments)

            for key, value in (
                    ('cl_method', 'finetune'),
                    ('head_consolidation_enabled', False),
                    ('head_consolidation_mode', 'full_classifier')):
                with self.subTest(forged=key):
                    options = copy.deepcopy(protocol)
                    options[key] = value
                    with self.assertRaises(_EvidenceError) as raised:
                        formal_audit._validate_formal_bic_producer_bundle(
                            options=options, **arguments)
                    self.assertEqual(
                        'bic-fit-corpus-invalid', raised.exception.reason)

            external_registered = protocol_for(
                FormalSpec('cifar100', 'finetune', 42))
            external_protocol = external_registered['base_options']
            external_tasks = _task_classes(external_registered)
            external_identity = {'protocol': copy.deepcopy(external_protocol)}
            external_corpus = consolidation_audit._formal_bic_fit_corpus(
                external_identity, cached_batches, calibration,
                validation_manifest, selection,
            )
            external_calibrator = TaskAffineCalibrator()
            for task_id, classes in enumerate(external_tasks):
                external_calibrator.set_task(
                    task_id, classes, identity_raw, 0.0)
            external_history = _valid_bic_history(
                external_registered, calibration, external_corpus)
            for task_id, record in enumerate(external_history):
                record['parameters'] = {
                    str(value): external_calibrator.parameters_for(value)
                    for value in range(task_id + 1)
                }
            external_arguments = {
                **arguments,
                'history': external_history,
                'state': external_calibrator.state_dict(),
                'task_classes': {
                    task_id: list(classes)
                    for task_id, classes in enumerate(external_tasks)
                },
                'fit_corpus': external_corpus,
            }
            self.assertTrue(
                formal_audit._validate_formal_bic_producer_bundle(
                    options=external_protocol, **external_arguments))
            for key, value in (
                    ('split', 'validation'),
                    ('selection_manifest_key',
                     'validation_manifest_sha256')):
                with self.subTest(external_forged=key):
                    malformed = copy.deepcopy(external_corpus)
                    malformed[key] = value
                    malformed_arguments = {
                        **external_arguments,
                        'history': _valid_bic_history(
                            external_registered, calibration, malformed),
                        'fit_corpus': malformed,
                    }
                    with self.assertRaises(_EvidenceError) as raised:
                        formal_audit._validate_formal_bic_producer_bundle(
                            options=external_protocol,
                            **malformed_arguments)
                    self.assertEqual(
                        'bic-fit-corpus-invalid', raised.exception.reason)

    def test_representative_external_and_internal_contracts_are_reusable(self):
        for method in (
                'finetune', 'no_consolidation', 'fedprotip_vfl', 'fixed_full'):
            with self.subTest(method=method):
                fixture = self.fixture(method)
                self.assertEqual(
                    protocol_for(fixture.spec)['method_contract']['cl_method'],
                    fixture._valid_checkpoint()['protocol']['cl_method'],
                )
                self.assertEqual('REUSABLE', self.audit(fixture).status)

    def test_real_formal_artifact_mutation_matrix_is_rejected(self):
        fixture = self.fixture()
        mutations = {
            'stage_checkpoint': 'checkpoints/event_6_CIL.pt',
            'stage_snapshot': 'formal_snapshots/event_6_CIL.pt',
            'freeze': 'FORMAL_STATE_FROZEN.json',
            'complete_tracker': 'FORMAL_EVALUATION_COMPLETE.json',
            'complete_cache': 'FORMAL_EVALUATION_COMPLETE.json',
            'seal': 'FORMAL_EVALUATION_SEALED.json',
            'published': 'FORMAL_EVALUATION_PUBLISHED.json',
        }
        for name, logical in mutations.items():
            with self.subTest(name=name):
                entry = next(
                    item for item in fixture.declaration['formal_artifacts']
                    if item['logical_path'] == logical
                )
                path = Path(entry['path'])
                original = path.read_bytes()
                details = path.stat()
                try:
                    path.chmod(0o600)
                    if name in {'stage_checkpoint', 'stage_snapshot'}:
                        payload = torch.load(
                            path, map_location='cpu', weights_only=True
                        )
                        if name == 'stage_checkpoint':
                            payload['task_id'] = 99
                        else:
                            payload['strict_reload_sha256'] = '0' * 64
                        torch.save(payload, path)
                    else:
                        payload = json.loads(path.read_text())
                        if name == 'freeze':
                            payload['identity']['protocol']['seed'] += 1
                        elif name == 'complete_tracker':
                            payload['evaluation']['step_results'][0][
                                'task_id'] = 99
                        elif name == 'complete_cache':
                            payload['cache_identity']['sample_count'] += 1
                        elif name == 'seal':
                            payload['protocol_sha256'] = '0' * 64
                        else:
                            payload['results_sha256'] = '0' * 64
                        fixture._write_json(path, payload)
                    fixture.refresh_formal(logical)
                    self.assertNotEqual(
                        'REUSABLE', self.audit(fixture).status
                    )
                finally:
                    path.write_bytes(original)
                    os.chmod(path, stat.S_IMODE(details.st_mode))
                    os.utime(path, ns=(details.st_atime_ns, details.st_mtime_ns))
                    fixture.refresh_formal(logical)

        missing = copy.deepcopy(fixture.declaration)
        missing['formal_artifacts'] = [
            entry for entry in missing['formal_artifacts']
            if entry['logical_path'] != 'formal_snapshots/event_0_CIL.pt'
        ]
        self.assertNotEqual(
            'REUSABLE', self.audit(fixture, missing).status
        )

    def test_consumed_formal_access_marker_mutations_are_rejected(self):
        fixture = self.fixture('fixed_full')
        cases = {
            'wrong_phase': (
                'validation', lambda marker: marker['access'].__setitem__(
                    'phase', 'final_test_post_install')),
            'wrong_classes': (
                'test', lambda marker: marker['access']['classes'].reverse()),
            'wrong_event': (
                'validation', lambda marker: marker['access'].__setitem__(
                    'event_idx', 11)),
            'wrong_task': (
                'test', lambda marker: marker['access'].__setitem__(
                    'task_id', 11)),
            'wrong_timeline': (
                'validation', lambda marker: marker['access'].__setitem__(
                    'timeline_step', 'event_11_CIL')),
            'wrong_status': (
                'test', lambda marker: marker.__setitem__(
                    'status', 'authorized')),
            'wrong_split': (
                'test', lambda marker: marker.__setitem__(
                    'access', json.loads(
                        (fixture.run / 'formal_access'
                         / 'validation.consumed.json').read_text()
                    )['access'])),
        }
        for name, (split, mutate) in cases.items():
            with self.subTest(name=name):
                logical = f'formal_access/{split}.consumed.json'
                path = fixture.run / logical
                original = path.read_bytes()
                mode = stat.S_IMODE(path.stat().st_mode)
                try:
                    marker = json.loads(original)
                    mutate(marker)
                    path.chmod(0o600)
                    fixture._write_json(path, marker)
                    fixture.refresh_formal(logical)
                    self.assertNotEqual(
                        'REUSABLE', self.audit(fixture).status
                    )
                finally:
                    path.chmod(0o600)
                    path.write_bytes(original)
                    path.chmod(mode)
                    fixture.refresh_formal(logical)

        for name, mutate in {
                'missing': lambda entries: entries.pop(next(
                    index for index, entry in enumerate(entries)
                    if entry['logical_path']
                    == 'formal_access/validation.consumed.json')),
                'duplicate': lambda entries: entries.append(copy.deepcopy(
                    next(entry for entry in entries if entry['logical_path']
                         == 'formal_access/test.consumed.json'))),
                'extra': lambda entries: entries.append({
                    **copy.deepcopy(next(
                        entry for entry in entries if entry['logical_path']
                        == 'formal_access/test.consumed.json')),
                    'logical_path': 'formal_access/calibration.consumed.json',
                }),
        }.items():
            with self.subTest(name=name):
                declaration = copy.deepcopy(fixture.declaration)
                mutate(declaration['formal_artifacts'])
                self.assertNotEqual(
                    'REUSABLE', self.audit(fixture, declaration).status
                )

        records = [
            json.loads(line) for line in fixture.data_flow.read_text().splitlines()
        ]
        original = fixture.data_flow.read_bytes()
        try:
            records.remove(next(
                record for record in records if record.get('split') == 'test'
            ))
            fixture.data_flow.write_text(''.join(
                json.dumps(record, sort_keys=True) + '\n'
                for record in records
            ))
            fixture.declaration['results']['data_flow']['sha256'] = _sha256(
                fixture.data_flow)
            self.assertNotEqual('REUSABLE', self.audit(fixture).status)
        finally:
            fixture.data_flow.write_bytes(original)
            fixture.declaration['results']['data_flow']['sha256'] = _sha256(
                fixture.data_flow)

    def test_scalar_integer_config_fields_reject_list_coercion(self):
        fixture = self.fixture()
        original = fixture.config.read_bytes()
        for field in ('seed', 'num_tasks', 'num_parties', 'batch_size'):
            with self.subTest(field=field):
                config = json.loads(original)
                config[field] = [config[field]]
                fixture._write_json(fixture.config, config)
                fixture.refresh('config')
                self.assertNotEqual('REUSABLE', self.audit(fixture).status)
        fixture.config.write_bytes(original)
        fixture.refresh('config')

    def test_parser_produced_config_lists_remain_reusable(self):
        fixture = self.fixture()
        config = json.loads(fixture.config.read_text())
        self.assertEqual([999], config['unlearn_after_tasks'])
        self.assertEqual([[0]], config['unlearn_classes'])
        self.assertEqual('REUSABLE', self.audit(fixture).status)

    def test_inactive_config_fields_require_exact_parser_types(self):
        fixture = self.fixture()
        original = fixture.config.read_bytes()
        mutations = {
            'inactive_int_list': ('bic_steps', [200]),
            'inactive_int_bool': ('adagauss_adapter_epochs', True),
            'inactive_int_nested': ('fedosd_local_epochs', [[1]]),
            'inactive_bool_int': ('proto_sdc', 1),
            'inactive_float_int': ('oracle_lr', 1),
            'inactive_str_list': ('seeds', []),
        }
        for name, (field, value) in mutations.items():
            with self.subTest(name=name):
                config = json.loads(original)
                config[field] = value
                fixture._write_json(fixture.config, config)
                fixture.refresh('config')
                self.assertNotEqual('REUSABLE', self.audit(fixture).status)
        fixture.config.write_bytes(original)
        fixture.refresh('config')

    def test_coordinated_formal_cache_chain_forgery_is_rejected(self):
        fixture = self.fixture()

        def rewrite(logical, value):
            path = fixture.run / logical
            mode = stat.S_IMODE(path.stat().st_mode)
            path.chmod(0o600)
            fixture._write_json(path, value)
            path.chmod(mode)
            fixture.refresh_formal(logical)

        consuming_logical = 'FORMAL_EVALUATION_CONSUMING.json'
        complete_logical = 'FORMAL_EVALUATION_COMPLETE.json'
        seal_logical = 'FORMAL_EVALUATION_SEALED.json'
        publishing_logical = 'FORMAL_EVALUATION_PUBLISHING.json'
        published_logical = 'FORMAL_EVALUATION_PUBLISHED.json'
        consuming = json.loads((fixture.run / consuming_logical).read_text())
        complete = json.loads((fixture.run / complete_logical).read_text())
        genuine = copy.deepcopy(complete['cache_identity']['batches'][0])
        forged_first = copy.deepcopy(genuine)
        forged_first['input_sha256'] = '0' * 64
        forged_first['label_sha256'] = '1' * 64
        forged_second = copy.deepcopy(genuine)
        forged_second['input_sha256'] = '2' * 64
        forged_second['label_sha256'] = '3' * 64
        forged = {
            'batch_count': 2,
            'sample_count': 2 * genuine['label_shape'][0],
            'batches': [forged_second, forged_first],
        }
        consuming['cache_identity'] = copy.deepcopy(forged)
        complete['cache_identity'] = copy.deepcopy(forged)
        rewrite(consuming_logical, consuming)
        rewrite(complete_logical, complete)

        seal = json.loads((fixture.run / seal_logical).read_text())
        seal['cache_identity'] = copy.deepcopy(forged)
        seal['consuming'] = _formal_file_record(
            fixture.run / consuming_logical, fixture.run)
        seal['complete'] = _formal_file_record(
            fixture.run / complete_logical, fixture.run)
        rewrite(seal_logical, seal)
        seal_record = _formal_file_record(
            fixture.run / seal_logical, fixture.run)

        publishing = json.loads(
            (fixture.run / publishing_logical).read_text())
        publishing['seal'] = copy.deepcopy(seal_record)
        rewrite(publishing_logical, publishing)
        published = json.loads((fixture.run / published_logical).read_text())
        published['seal'] = copy.deepcopy(seal_record)
        rewrite(published_logical, published)
        self.assertNotEqual('REUSABLE', self.audit(fixture).status)

    def test_coherent_zero_metric_chain_is_recomputed_from_frozen_models(self):
        fixture = self.fixture()

        results = fixture._valid_results()
        for row in results['task_acc_history']:
            for key in (
                    'per_task_accs', 'per_task_accs_debiased',
                    'per_task_accs_taskil', 'deferred_diagonal'):
                if key in row:
                    row[key] = {name: 0.0 for name in row[key]}
            for companion in row.get('companion_readouts', {}).values():
                companion.update({name: 0.0 for name in companion})
            row['overall_acc'] = 0.0
        tracker = MetricsTracker()
        tracker.load_dict({key: results[key] for key in (
            'cl_metrics', 'task_acc_history', 'ul_metrics', 'comm_stats',
            'timing', 'step_results',
        )})
        zero_tracker = tracker.to_dict()
        results.update(zero_tracker)
        fixture._write_results(results)
        fixture.refresh('results')

        checkpoint = fixture._valid_checkpoint()
        checkpoint['tracker_state'] = copy.deepcopy(zero_tracker)
        fixture.checkpoint.chmod(0o600)
        fixture._write_checkpoint(checkpoint)
        fixture.refresh('checkpoint')

        complete_path = fixture.run / 'FORMAL_EVALUATION_COMPLETE.json'
        complete = json.loads(complete_path.read_text())
        complete['evaluation'] = copy.deepcopy(zero_tracker)
        complete['evaluation_sha256'] = hashlib.sha256(
            consolidation_audit._strict_json(zero_tracker).encode('utf-8')
        ).hexdigest()
        complete_path.chmod(0o600)
        fixture._write_json(complete_path, complete)
        fixture.refresh_formal(complete_path.name)

        seal_path = fixture.run / 'FORMAL_EVALUATION_SEALED.json'
        seal = json.loads(seal_path.read_text())
        seal['complete'] = _formal_file_record(complete_path, fixture.run)
        seal_path.chmod(0o600)
        fixture._write_json(seal_path, seal)
        fixture.refresh_formal(seal_path.name)
        seal_record = _formal_file_record(seal_path, fixture.run)

        results_sha256 = hashlib.sha256(
            consolidation_audit._strict_json(results).encode('utf-8')
        ).hexdigest()
        publishing_path = fixture.run / 'FORMAL_EVALUATION_PUBLISHING.json'
        publishing = json.loads(publishing_path.read_text())
        publishing.update({
            'evaluation_sha256': complete['evaluation_sha256'],
            'results_sha256': results_sha256,
            'seal': copy.deepcopy(seal_record),
        })
        publishing_path.chmod(0o600)
        fixture._write_json(publishing_path, publishing)
        fixture.refresh_formal(publishing_path.name)

        published_path = fixture.run / 'FORMAL_EVALUATION_PUBLISHED.json'
        published = json.loads(published_path.read_text())
        published.update({
            'evaluation_sha256': complete['evaluation_sha256'],
            'results_sha256': results_sha256,
            'seal': copy.deepcopy(seal_record),
            'checkpoint': _formal_file_record(
                fixture.checkpoint, fixture.run),
            'results': _formal_file_record(fixture.results, fixture.run),
        })
        published_path.chmod(0o600)
        fixture._write_json(published_path, published)
        fixture.refresh_formal(published_path.name)

        record = self.audit(fixture)
        self.assertNotEqual('REUSABLE', record.status)

    def test_fedprotip_formal_instance_state_mutation_is_rejected(self):
        fixture = self.fixture('fedprotip_vfl')
        logical = 'formal_snapshots/event_12_CIL.pt'
        entry = next(
            item for item in fixture.declaration['formal_artifacts']
            if item['logical_path'] == logical
        )
        path = Path(entry['path'])
        payload = torch.load(path, map_location='cpu', weights_only=True)
        payload['cl_state']['tip_threshold'] = 0.5
        path.chmod(0o600)
        torch.save(payload, path)
        fixture.refresh_formal(logical)
        self.assertNotEqual('REUSABLE', self.audit(fixture).status)

    def test_nonaccess_methods_hold_out_the_exact_complement_without_access(self):
        for method in ('finetune', 'no_consolidation'):
            with self.subTest(method=method):
                fixture = self.fixture(method)
                results = fixture._valid_results()
                selection = results['selection_audit']
                self.assertEqual((78, 26, 0), (
                    selection['training_count'],
                    selection['validation_count'],
                    selection['training_validation_overlap_count'],
                ))
                records = [
                    json.loads(line)
                    for line in fixture.data_flow.read_text().splitlines()
                ]
                train_indices = {
                    index for record in records if 'indices' in record
                    for index in record['indices']
                }
                heldout = set(fixture.manifest_value['ordered_indices'])
                self.assertEqual(set(range(104)) - heldout, train_indices)
                self.assertFalse(any(
                    record.get('split') == 'validation'
                    for record in records
                ))
                self.assertEqual(
                    1, sum(record.get('split') == 'test'
                           for record in records))
                self.assertEqual('REUSABLE', self.audit(fixture).status)

    def test_nonaccess_methods_reject_injected_validation_access(self):
        for method in ('finetune', 'no_consolidation'):
            with self.subTest(method=method):
                fixture = self.fixture(method)
                records = [
                    json.loads(line)
                    for line in fixture.data_flow.read_text().splitlines()
                ]
                test_position = next(
                    index for index, record in enumerate(records)
                    if record.get('split') == 'test'
                )
                test = records[test_position]
                records.insert(test_position, {
                    **test,
                    'split': 'validation',
                    'loader_key': repr((
                        'lambda_validation', tuple(
                            class_id for task in fixture._task_classes()
                            for class_id in task))),
                    'phase': 'final_validation_pre_install',
                })
                fixture.data_flow.write_text(''.join(
                    json.dumps(record, sort_keys=True) + '\n'
                    for record in records
                ))
                fixture.declaration['results']['data_flow']['sha256'] = (
                    _sha256(fixture.data_flow))
                self.assertNotEqual('REUSABLE', self.audit(fixture).status)

    def test_validation_access_method_has_one_final_access_and_one_test(self):
        fixture = self.fixture('fixed_full')
        records = [
            json.loads(line)
            for line in fixture.data_flow.read_text().splitlines()
        ]
        accesses = [
            record for record in records
            if record.get('split') in {'validation', 'test'}
        ]
        self.assertEqual(
            [('validation', 'final_validation_pre_install'),
             ('test', 'final_test_post_install')],
            [(record['split'], record['phase']) for record in accesses],
        )
        self.assertEqual('REUSABLE', self.audit(fixture).status)

    def test_internal_manifest_must_match_authoritative_outer_manifest(self):
        fixture = self.fixture('adaptive')
        checkpoint = fixture._valid_checkpoint()
        state = checkpoint['cl_state']
        history = state['head_consolidation_history']
        forged = copy.deepcopy(history[0]['validation_manifest'])
        first, second = sorted(forged['by_class'], key=int)[:2]
        forged['by_class'][first], forged['by_class'][second] = (
            forged['by_class'][second], forged['by_class'][first]
        )
        identity_key = (
            'ordered_indices' if 'ordered_indices' in forged
            else 'ordered_sample_ids'
        )
        ordered = [
            identity
            for class_id in sorted(forged['by_class'], key=int)
            for identity in forged['by_class'][class_id]
        ]
        forged[identity_key] = ordered
        forged['sha256'] = hashlib.sha256(_json_bytes(ordered)).hexdigest()
        self.assertNotEqual(
            fixture.manifest_value['sha256'], forged['sha256'])
        history[0]['validation_manifest'] = forged
        state['head_validation_sha256'] = forged['sha256']
        state['adaptive_audit_bundle']['result'] = copy.deepcopy(history[0])
        fixture._write_checkpoint(checkpoint)
        fixture.refresh('checkpoint')
        self.assertNotEqual('REUSABLE', self.audit(fixture).status)

    def test_relabelled_finetune_state_cannot_satisfy_other_methods(self):
        for method in ('lwf', 'er', 'fixed_full'):
            with self.subTest(method=method):
                fixture = self.fixture()
                fixture.spec = FormalSpec('isolet', method, 42)
                fixture.normalized_spec = asdict(fixture.spec)
                fixture.protocol = json.loads(json.dumps(
                    protocol_for(fixture.spec)))
                fixture._write_json(fixture.config, fixture.protocol)
                results = fixture._valid_results()
                results['config']['cl_method'] = (
                    fixture.protocol['method_contract']['cl_method'])
                fixture._write_results(results)
                fixture.declaration['spec'] = fixture.normalized_spec
                fixture.refresh('config')
                fixture.refresh('results')
                self.assertNotEqual('REUSABLE', self.audit(fixture).status)

    def test_no_consolidation_state_cannot_be_relabelled_internal(self):
        for method in (
                'fixed_full', 'fixed_bias', 'adaptive',
                'fixed_half', 'sample_mean_nll'):
            with self.subTest(method=method):
                fixture = self.fixture('no_consolidation')
                fixture.spec = FormalSpec(
                    'isolet', method, 42,
                    method in {'fixed_half', 'sample_mean_nll'},
                )
                fixture.normalized_spec = asdict(fixture.spec)
                fixture.protocol = json.loads(json.dumps(
                    protocol_for(fixture.spec)))
                fixture._write_json(fixture.config, fixture.protocol)
                results = fixture._valid_results()
                results['config']['cl_method'] = 'proto_evolve'
                fixture._write_results(results)
                checkpoint = fixture._valid_checkpoint()
                options = fixture.protocol['base_options']
                checkpoint['protocol'].update({
                    'cl_method': options['cl_method'],
                    'head_consolidation_enabled': int(
                        options['head_consolidation_enabled']),
                    'head_consolidation_mode':
                        options['head_consolidation_mode'],
                })
                fixture._write_checkpoint(checkpoint)
                fixture.declaration['spec'] = fixture.normalized_spec
                fixture.refresh('config')
                fixture.refresh('results')
                fixture.refresh('checkpoint')
                self.assertNotEqual('REUSABLE', self.audit(fixture).status)

    def test_internal_states_reject_every_cross_relabel(self):
        methods = ('no_consolidation', 'fixed_full')
        for source in methods[1:]:
            fixture = self.fixture(source)
            original = {
                'spec': fixture.spec,
                'normalized_spec': copy.deepcopy(fixture.normalized_spec),
                'protocol': copy.deepcopy(fixture.protocol),
                'declaration': copy.deepcopy(fixture.declaration),
                'config': fixture.config.read_bytes(),
                'results': fixture.results.read_bytes(),
                'checkpoint': fixture.checkpoint.read_bytes(),
            }
            for target in methods:
                if target == source:
                    continue
                with self.subTest(source=source, target=target):
                    target_spec = FormalSpec(
                        'isolet', target, 42,
                        target in {'fixed_half', 'sample_mean_nll'},
                    )
                    target_protocol = json.loads(json.dumps(
                        protocol_for(target_spec)))
                    fixture.spec = target_spec
                    fixture.normalized_spec = asdict(target_spec)
                    fixture.protocol = target_protocol
                    fixture._write_json(fixture.config, target_protocol)
                    results = json.loads(original['results'])
                    results['config']['cl_method'] = 'proto_evolve'
                    fixture._write_results(results)
                    checkpoint = torch.load(
                        io.BytesIO(original['checkpoint']), map_location='cpu',
                        weights_only=True)
                    options = target_protocol['base_options']
                    checkpoint['protocol'].update({
                        'cl_method': 'proto_evolve',
                        'head_consolidation_enabled': int(
                            options['head_consolidation_enabled']),
                        'head_consolidation_mode':
                            options['head_consolidation_mode'],
                    })
                    fixture._write_checkpoint(checkpoint)
                    fixture.declaration = copy.deepcopy(original['declaration'])
                    fixture.declaration['spec'] = fixture.normalized_spec
                    fixture.refresh('config')
                    fixture.refresh('results')
                    fixture.refresh('checkpoint')
                    self.assertNotEqual(
                        'REUSABLE', self.audit(fixture).status)
            fixture.spec = original['spec']
            fixture.normalized_spec = original['normalized_spec']
            fixture.protocol = original['protocol']
            fixture.declaration = original['declaration']
            fixture.config.write_bytes(original['config'])
            fixture.results.write_bytes(original['results'])
            fixture.checkpoint.write_bytes(original['checkpoint'])

    def test_test_access_identity_is_bound_to_authoritative_split(self):
        fixture = self.fixture()
        records = [json.loads(line)
                   for line in fixture.data_flow.read_text().splitlines()]
        access = next(record for record in records
                      if record.get('split') == 'test')
        access['loader_key'] = repr(('test', (999,)))
        fixture.data_flow.write_text(
            ''.join(json.dumps(record, sort_keys=True) + '\n'
                    for record in records))
        fixture.declaration['results']['data_flow']['sha256'] = _sha256(
            fixture.data_flow)
        self.assertNotEqual('REUSABLE', self.audit(fixture).status)

    def test_validation_after_an_earlier_task_is_rejected(self):
        fixture = self.fixture('fixed_full')
        records = [json.loads(line)
                   for line in fixture.data_flow.read_text().splitlines()]
        validation = next(record for record in records
                          if record.get('split') == 'validation')
        records.remove(validation)
        validation['loader_key'] = repr(
            ('lambda_validation', fixture._task_classes()[0]))
        validation.update({
            'event_idx': 0, 'task_id': 0,
            'timeline_step': 'event_0_CIL',
        })
        second_task = repr(('train', fixture._task_classes()[1]))
        position = next(index for index, record in enumerate(records)
                        if record.get('loader_key') == second_task)
        records.insert(position, validation)
        fixture.data_flow.write_text(
            ''.join(json.dumps(record, sort_keys=True) + '\n'
                    for record in records))
        fixture.declaration['results']['data_flow']['sha256'] = _sha256(
            fixture.data_flow)
        self.assertNotEqual('REUSABLE', self.audit(fixture).status)

    def test_test_access_after_an_earlier_task_is_rejected(self):
        fixture = self.fixture()
        records = [json.loads(line)
                   for line in fixture.data_flow.read_text().splitlines()]
        test = next(record for record in records
                    if record.get('split') == 'test')
        records.remove(test)
        test['loader_key'] = repr(('test', fixture._task_classes()[0]))
        test.update({
            'event_idx': 0, 'task_id': 0,
            'timeline_step': 'event_0_CIL',
        })
        second_task = repr(('train', fixture._task_classes()[1]))
        position = next(index for index, record in enumerate(records)
                        if record.get('loader_key') == second_task)
        records.insert(position, test)
        fixture.data_flow.write_text(
            ''.join(json.dumps(record, sort_keys=True) + '\n'
                    for record in records))
        fixture.declaration['results']['data_flow']['sha256'] = _sha256(
            fixture.data_flow)
        self.assertNotEqual('REUSABLE', self.audit(fixture).status)

    def test_canonical_final_split_boundaries_are_reusable(self):
        internal = self.fixture('fixed_full')
        internal_records = [json.loads(line)
                            for line in internal.data_flow.read_text().splitlines()]
        task_one_key = repr(('train', internal._task_classes()[1]))
        first_task_one = next(record for record in internal_records
                              if record['loader_key'] == task_one_key)
        self.assertEqual(1, first_task_one['loader_iteration'])
        self.assertEqual('REUSABLE', self.audit(internal).status)

        external = self.fixture()
        self.assertEqual('REUSABLE', self.audit(external).status)
        records = [json.loads(line)
                   for line in external.data_flow.read_text().splitlines()]
        test_position = next(index for index, record in enumerate(records)
                             if record.get('split') == 'test')
        test = records[test_position]
        records.insert(test_position, {
            **test,
            'split': 'validation',
            'loader_key': repr((
                'lambda_validation', tuple(
                    class_id for task in external._task_classes()
                    for class_id in task))),
            'phase': 'final_validation_pre_install',
        })
        external.data_flow.write_text(
            ''.join(json.dumps(record, sort_keys=True) + '\n'
                    for record in records))
        external.declaration['results']['data_flow']['sha256'] = _sha256(
            external.data_flow)
        self.assertNotEqual('REUSABLE', self.audit(external).status)

    def test_split_access_requires_explicit_authoritative_boundary(self):
        fixture = self.fixture('fixed_full')
        records = [json.loads(line)
                   for line in fixture.data_flow.read_text().splitlines()]
        validation = next(record for record in records
                          if record.get('split') == 'validation')
        del validation['phase']
        fixture.data_flow.write_text(
            ''.join(json.dumps(record, sort_keys=True) + '\n'
                    for record in records))
        fixture.declaration['results']['data_flow']['sha256'] = _sha256(
            fixture.data_flow)
        self.assertNotEqual('REUSABLE', self.audit(fixture).status)

    def test_candidate_git_repository_cannot_define_trusted_commit(self):
        fixture = self.fixture()
        attacker = fixture.root / 'attacker'
        for name in PRODUCER_SOURCES:
            target = attacker / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fixture.source / name, target)
        (attacker / 'main.py').write_text(
            (attacker / 'main.py').read_text() + '\nATTACKER = True\n',
            encoding='utf-8',
        )
        subprocess.run(['git', 'init', '-q', str(attacker)], check=True)
        subprocess.run(
            ['git', '-C', str(attacker), 'config', 'user.email', 'a@b.c'],
            check=True,
        )
        subprocess.run(
            ['git', '-C', str(attacker), 'config', 'user.name', 'attacker'],
            check=True,
        )
        subprocess.run(['git', '-C', str(attacker), 'add', '.'], check=True)
        subprocess.run(
            ['git', '-C', str(attacker), 'commit', '-qm', 'forged'], check=True,
        )
        fixture.declaration['code_commit'] = subprocess.run(
            ['git', '-C', str(attacker), 'rev-parse', 'HEAD'], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        fixture.declaration['source_files'] = [
            fixture._entry(attacker / name, attacker, name)
            for name in PRODUCER_SOURCES
        ]
        record = self.audit(fixture)
        self.assertEqual('REJECTED', record.status)
        self.assertEqual('source-commit-invalid', record.reason)

    def test_results_tracker_is_exactly_bound_to_final_checkpoint(self):
        fixture = self.fixture()
        results = fixture._valid_results()
        results['cl_metrics'] = {'forged': 1.0}
        fixture._write_results(results)
        fixture.refresh('results')
        record = self.audit(fixture)
        self.assertNotEqual('REUSABLE', record.status)

    def test_declared_spec_uses_exact_primitive_types(self):
        fixture = self.fixture()
        fixture.declaration['spec']['seed'] = 42.0
        record = self.audit(fixture)
        self.assertEqual('REJECTED', record.status)
        self.assertEqual('declaration-spec-mismatch', record.reason)

    def test_nonsense_or_nonloadable_real_checkpoint_state_is_not_reusable(self):
        fixture = self.fixture()
        checkpoint = fixture._valid_checkpoint()
        checkpoint['trainer_state'] = {'nonsense': {'weight': torch.ones(1)}}
        fixture._write_checkpoint(checkpoint)
        fixture.refresh('checkpoint')
        record = self.audit(fixture)
        self.assertEqual('RERUN_REQUIRED', record.status)
        self.assertEqual('strict-resume-failed', record.reason)

    def test_source_inventory_and_git_commit_are_authoritative(self):
        fixture = self.fixture()
        fixture.declaration['source_files'].pop()
        record = self.audit(fixture)
        self.assertEqual('REJECTED', record.status)
        self.assertEqual('source-inventory-mismatch', record.reason)

    def test_rewritten_data_and_manifest_cannot_self_attest(self):
        fixture = self.fixture()
        fixture.metadata.write_bytes(b'{"dataset":"replacement"}')
        fixture.declaration['data_files'][1]['sha256'] = _sha256(fixture.metadata)
        manifest = json.loads(fixture.manifest.read_text())
        manifest['dataset'] = 'replacement-train'
        manifest['sha256'] = 'f' * 64
        fixture._write_json(fixture.manifest, manifest)
        fixture.declaration['validation_manifest']['sha256'] = _sha256(
            fixture.manifest
        )
        record = self.audit(fixture)
        self.assertEqual('REJECTED', record.status)
        self.assertEqual('data-inventory-mismatch', record.reason)

    def test_missing_or_incomplete_producer_data_flow_is_not_reusable(self):
        fixture = self.fixture()
        del fixture.declaration['results']['data_flow']
        record = self.audit(fixture)
        self.assertEqual('REJECTED', record.status)
        self.assertEqual('results-declaration-schema-invalid', record.reason)

        fabricated = self.fixture()
        records = [
            {'loader_key': repr(('train', tuple(classes))),
             'loader_iteration': 0, 'batch': 0, 'indices': [index],
             'dtype': 'torch.float32', 'shape': [1, 8],
             'batch_sha256': f'{index + 1:064x}'}
            for index, classes in enumerate(fabricated._task_classes())
        ]
        records.insert(1, {
            'event': 'first_iteration',
            'loader_key': repr(('test', fabricated._task_classes()[0])),
            'split': 'test',
        })
        fabricated.data_flow.write_text(
            ''.join(json.dumps(row, sort_keys=True) + '\n' for row in records),
            encoding='utf-8',
        )
        fabricated.declaration['results']['data_flow']['sha256'] = _sha256(
            fabricated.data_flow)
        self.assertNotEqual('REUSABLE', self.audit(fabricated).status)

        overlap = self.fixture()
        records = [json.loads(line) for line in overlap.data_flow.read_text().splitlines()]
        train = next(row for row in records if 'indices' in row)
        train['indices'][0] = 8
        overlap.data_flow.write_text(
            ''.join(json.dumps(row, sort_keys=True) + '\n' for row in records),
            encoding='utf-8',
        )
        overlap.declaration['results']['data_flow']['sha256'] = _sha256(
            overlap.data_flow)
        self.assertNotEqual('REUSABLE', self.audit(overlap).status)

        mismatched = self.fixture()
        results = mismatched._valid_results()
        results['selection_audit'] = {'passed': True}
        mismatched._write_results(results)
        mismatched.refresh('results')
        self.assertEqual('REJECTED', self.audit(mismatched).status)

    def test_json_duplicate_members_and_primitive_type_confusion_are_rejected(self):
        duplicate = self.fixture()
        content = duplicate.config.read_text()
        content = content.replace(
            '"deterministic": 1',
            '"deterministic": 1,"deterministic": 1',
            1,
        )
        duplicate.config.write_text(content, encoding='utf-8')
        duplicate.refresh('config')
        record = self.audit(duplicate)
        self.assertEqual('REJECTED', record.status)
        self.assertEqual('config-json-invalid', record.reason)

        confused = self.fixture()
        config = json.loads(confused.config.read_text())
        config['deterministic'] = '1'
        confused._write_json(confused.config, config)
        confused.refresh('config')
        record = self.audit(confused)
        self.assertEqual('REJECTED', record.status)
        self.assertEqual('config-schema-invalid', record.reason)

    def test_run_root_exchange_during_read_is_rejected(self):
        fixture = self.fixture()
        clone = fixture.root / 'clone'
        shutil.copytree(fixture.run, clone)
        original_open = os.open
        swapped = False

        def exchange_open(path, flags, *args, **kwargs):
            nonlocal swapped
            name = os.fspath(path)
            if not swapped and (name == os.fspath(fixture.config)
                                or name == fixture.config.name):
                swapped = True
                hold = fixture.root / 'hold'
                os.rename(fixture.run, hold)
                os.rename(clone, fixture.run)
                os.rename(hold, clone)
            return original_open(path, flags, *args, **kwargs)

        with mock.patch('three_dataset_formal_audit.os.open', exchange_open):
            record = self.audit(fixture)
        self.assertTrue(swapped)
        self.assertEqual('REJECTED', record.status)
        self.assertEqual('root-identity-changed', record.reason)

    def test_failed_install_leaves_no_final_or_temp_and_can_retry(self):
        fixture = self.fixture()
        record = self.audit(fixture)
        parent = fixture.root / 'admission_failure'
        parent.mkdir()
        destination = parent / 'record.json'
        with mock.patch('three_dataset_formal_audit.os.write', return_value=0):
            with self.assertRaises(OSError):
                write_admission(record, destination)
        self.assertFalse(destination.exists())
        self.assertEqual([], list(parent.iterdir()))
        write_admission(record, destination)
        self.assertTrue(destination.is_file())

    def test_nonfinite_trajectory_is_damaged_and_rejected(self):
        fixture = self.fixture()
        results = json.loads(fixture.results.read_text())
        results['task_acc_history'][0]['per_task_accs']['task_0'] = float('nan')
        fixture._write_results(results)
        fixture.refresh('results')
        record = self.audit(fixture)
        self.assertEqual('REJECTED', record.status)

    def test_reusable_record_recomputes_metrics_and_preserves_source_tree(self):
        fixture = self.fixture()
        before = _tree_sha256(fixture.root)
        record = self.audit(fixture)
        self.assertIsInstance(record, AdmissionRecord)
        self.assertEqual('REUSABLE', record.status)
        self.assertEqual('admitted', record.reason)
        self.assertEqual('final-minus-diagonal-v1',
                         record.metric_formula_version)
        self.assertNotEqual(987.654321, record.metrics.bwt)
        self.assertEqual(fixture.normalized_spec, record.spec)
        self.assertRegex(record.protocol_sha256, r'^[0-9a-f]{64}$')
        self.assertRegex(record.source_sha256, r'^[0-9a-f]{64}$')
        self.assertIn('trajectory', record.artifact_sha256)
        self.assertIn('data_flow', record.artifact_sha256)
        self.assertEqual(before, _tree_sha256(fixture.root))

    def test_declared_hash_tamper_table_is_rejected(self):
        for name in ('config', 'results', 'checkpoint',
                     'validation_manifest', 'source', 'data'):
            with self.subTest(name=name):
                fixture = self.fixture()
                if name in ('source', 'data'):
                    fixture.declaration[f'{name}_files'][0]['sha256'] = '0' * 64
                else:
                    fixture.declaration[name]['sha256'] = '0' * 64
                record = self.audit(fixture)
                self.assertEqual('REJECTED', record.status)
                self.assertIsNone(record.metrics)
                self.assertEqual({}, record.artifact_sha256)

    def test_declared_commit_is_validated_and_bound_into_source_digest(self):
        first = self.fixture()
        first_record = self.audit(first)
        self.assertEqual('REUSABLE', first_record.status)
        invalid = self.fixture()
        invalid.declaration['code_commit'] = 'b' * 40
        record = self.audit(invalid)
        self.assertEqual('REJECTED', record.status)
        self.assertEqual('source-commit-invalid', record.reason)

    def test_path_boundary_table_is_rejected(self):
        cases = ('symlink_ancestor', 'final_symlink', 'path_escape')
        for name in cases:
            with self.subTest(name=name):
                fixture = self.fixture()
                declaration = copy.deepcopy(fixture.declaration)
                if name == 'symlink_ancestor':
                    alias = fixture.root / 'run_alias'
                    alias.symlink_to(fixture.run, target_is_directory=True)
                    declaration['config']['path'] = str(alias / 'config.json')
                elif name == 'final_symlink':
                    alias = fixture.run / 'checkpoint_alias.pt'
                    alias.symlink_to(fixture.checkpoint)
                    declaration['checkpoint']['path'] = str(alias)
                else:
                    outside = fixture.root / 'outside.json'
                    outside.write_bytes(fixture.config.read_bytes())
                    declaration['config']['path'] = str(outside)
                    declaration['config']['sha256'] = _sha256(outside)
                record = self.audit(fixture, declaration)
                self.assertEqual('REJECTED', record.status)

    def test_manifest_has_a_tree_preserving_logical_identity(self):
        fixture = self.fixture()
        self.assertEqual(
            'REUSABLE',
            self.audit(fixture).status,
        )
        fixture.declaration['validation_manifest']['logical_path'] = '../manifest.json'
        self.assertEqual(
            'REJECTED',
            self.audit(fixture).status,
        )

    def test_inode_swap_during_pinned_read_is_rejected(self):
        fixture = self.fixture()
        original_read = os.read
        config_inode = fixture.config.stat().st_ino
        replacement = fixture.run / 'replacement.json'
        replacement.write_bytes(fixture.config.read_bytes())
        swapped = False

        def swapping_read(fd, size):
            nonlocal swapped
            content = original_read(fd, size)
            if not swapped and os.fstat(fd).st_ino == config_inode:
                swapped = True
                os.replace(replacement, fixture.config)
            return content

        with mock.patch('three_dataset_formal_audit.os.read', swapping_read):
            record = self.audit(fixture)
        self.assertTrue(swapped)
        self.assertEqual('REJECTED', record.status)

    def test_final_declared_name_swap_after_read_is_rejected(self):
        fixture = self.fixture()
        replacement = fixture.run / 'replacement.json'
        replacement.write_bytes(fixture.config.read_bytes())
        original = __import__('three_dataset_formal_audit')._read_descriptor
        config_inode = fixture.config.stat().st_ino
        swapped = False

        def swap_after_read(descriptor):
            nonlocal swapped
            result = original(descriptor)
            if not swapped and os.fstat(descriptor).st_ino == config_inode:
                swapped = True
                os.replace(replacement, fixture.config)
            return result

        with mock.patch(
                'three_dataset_formal_audit._read_descriptor', swap_after_read):
            record = self.audit(fixture)
        self.assertTrue(swapped)
        self.assertEqual('REJECTED', record.status)

    def test_protocol_mismatch_table_requires_rerun(self):
        changes = {
            'wrong_seed': ('seed', 43),
            'wrong_tasks': ('num_tasks', 9),
            'wrong_parties': ('num_parties', 99),
            'wrong_replay': ('replay_mode', 'full'),
        }
        for name, (field, value) in changes.items():
            with self.subTest(name=name):
                fixture = RealProducerFixture()
                self.addCleanup(fixture.close)
                config = json.loads(fixture.config.read_text())
                config[field] = value
                fixture._write_json(fixture.config, config)
                fixture.refresh('config')
                record = self.audit(fixture)
                self.assertEqual('RERUN_REQUIRED', record.status)
                self.assertEqual('protocol-mismatch', record.reason)

    def test_damaged_trajectory_mutation_table_is_rejected(self):
        for name in ('missing_diagonal', 'missing_final', 'relabelled_task'):
            with self.subTest(name=name):
                fixture = self.fixture()
                results = json.loads(fixture.results.read_text())
                if name == 'missing_diagonal':
                    del results['task_acc_history'][4]['per_task_accs']['task_4']
                elif name == 'missing_final':
                    results['task_acc_history'].pop()
                else:
                    row = results['task_acc_history'][3]
                    row['per_task_accs']['task_99'] = row['per_task_accs'].pop('task_3')
                fixture._write_results(results)
                fixture.refresh('results')
                record = self.audit(fixture)
                self.assertEqual('REJECTED', record.status)
                self.assertIsNone(record.metrics)

    def test_declaration_results_and_duplicate_schema_extras_are_rejected(self):
        mutations = ('declaration_extra', 'config_extra', 'results_extra',
                     'undeclared_path', 'duplicate_logical')
        for name in mutations:
            with self.subTest(name=name):
                fixture = self.fixture()
                if name == 'declaration_extra':
                    fixture.declaration['fallback'] = '/tmp/not-allowed'
                elif name == 'config_extra':
                    config = json.loads(fixture.config.read_text())
                    config['unexpected'] = True
                    fixture._write_json(fixture.config, config)
                    fixture.refresh('config')
                elif name in ('results_extra', 'undeclared_path'):
                    results = json.loads(fixture.results.read_text())
                    results['extra' if name == 'results_extra'
                            else 'artifact_path'] = (
                                True if name == 'results_extra' else '/tmp/undeclared'
                            )
                    fixture._write_results(results)
                    fixture.refresh('results')
                else:
                    fixture.declaration['source_files'].append(
                        copy.deepcopy(fixture.declaration['source_files'][0])
                    )
                record = self.audit(fixture)
                self.assertEqual('REJECTED', record.status)

    def test_strict_resume_failure_is_not_reusable(self):
        fixture = self.fixture()
        checkpoint = fixture._valid_checkpoint()
        checkpoint['event_idx'] -= 1
        fixture._write_checkpoint(checkpoint)
        fixture.refresh('checkpoint')
        record = self.audit(fixture)
        self.assertEqual('RERUN_REQUIRED', record.status)
        self.assertEqual('strict-resume-failed', record.reason)

    def test_non_bic_checkpoint_rejects_garbage_calibrator_state(self):
        fixture = self.fixture()
        checkpoint = fixture._valid_checkpoint()
        checkpoint['bic_state'] = {'garbage': {}}
        fixture._write_checkpoint(checkpoint)
        fixture.refresh('checkpoint')
        self.assertNotEqual('REUSABLE', self.audit(fixture).status)

    def test_forbidden_reducer_is_rejected_without_execution(self):
        fixture = self.fixture()
        marker = fixture.root / 'executed'
        torch.save({'payload': _Exploit(marker)}, fixture.checkpoint)
        fixture.refresh('checkpoint')
        record = self.audit(fixture)
        self.assertEqual('REJECTED', record.status)
        self.assertEqual('unsafe-checkpoint', record.reason)
        self.assertFalse(marker.exists())

    def test_invalid_hash_and_nonabsolute_paths_fail_closed(self):
        for name in ('invalid_hash', 'relative_path'):
            with self.subTest(name=name):
                fixture = self.fixture()
                if name == 'invalid_hash':
                    fixture.declaration['config']['sha256'] = 'A' * 64
                else:
                    fixture.declaration['config']['path'] = 'config.json'
                record = self.audit(fixture)
                self.assertEqual('REJECTED', record.status)

    def test_write_admission_is_exclusive_canonical_and_identity_verified(self):
        fixture = self.fixture()
        record = self.audit(fixture)
        destination = fixture.root / 'admissions' / 'cell.json'
        destination.parent.mkdir()
        write_admission(record, destination)
        payload = destination.read_bytes()
        expected = json.dumps(
            asdict(record), sort_keys=True, separators=(',', ':'),
            allow_nan=False,
        ).encode() + b'\n'
        details = destination.lstat()
        self.assertEqual(expected, payload)
        self.assertTrue(stat.S_ISREG(details.st_mode))
        self.assertEqual(0o444, stat.S_IMODE(details.st_mode))
        self.assertEqual(hashlib.sha256(expected).hexdigest(), _sha256(destination))
        with self.assertRaises(FileExistsError):
            write_admission(record, destination)
        self.assertEqual(expected, destination.read_bytes())

    def test_write_admission_rejects_final_swap_after_descriptor_read(self):
        from three_dataset_formal_audit import _read_descriptor

        fixture = self.fixture()
        record = self.audit(fixture)
        destination = fixture.root / 'admission_swap' / 'record.json'
        destination.parent.mkdir()
        replacement = b'adversarial replacement\n'

        def swap_after_read(descriptor):
            verified = _read_descriptor(descriptor)
            destination.unlink()
            destination.write_bytes(replacement)
            destination.chmod(0o444)
            return verified

        with mock.patch(
                'three_dataset_formal_audit._read_descriptor',
                side_effect=swap_after_read):
            with self.assertRaises(RuntimeError):
                write_admission(record, destination)
        self.assertEqual(replacement, destination.read_bytes())

    def test_write_admission_rejects_symlink_ancestor_and_final_symlink(self):
        fixture = self.fixture()
        record = self.audit(fixture)
        real = fixture.root / 'real_admissions'
        real.mkdir()
        alias = fixture.root / 'admission_alias'
        alias.symlink_to(real, target_is_directory=True)
        with self.assertRaises(ValueError):
            write_admission(record, alias / 'record.json')
        target = real / 'target.json'
        target.write_bytes(b'existing')
        final_alias = real / 'record.json'
        final_alias.symlink_to(target)
        with self.assertRaises(FileExistsError):
            write_admission(record, final_alias)
        self.assertEqual(b'existing', target.read_bytes())


class StrictDataFlowTest(unittest.TestCase):
    class Loader:
        batch_size = 2

        def __init__(self, classes):
            self.classes = classes
            self.audit_key = repr(('train', classes))
            self.audit_sampler = SimpleNamespace(epoch_orders=[])
            self.completed = 0
            self.last_batch = None

        def __iter__(self):
            iteration = len(self.audit_sampler.epoch_orders)
            indices = list(range(6))
            offset = iteration % len(indices)
            order = indices[offset:] + indices[:offset]
            self.audit_sampler.epoch_orders.append(order)
            for batch in range(3):
                self.last_batch = (iteration, batch)
                batch_indices = order[2 * batch:2 * (batch + 1)]
                # Frozen image contracts augment in workers; zero-worker
                # vector loaders have no stochastic augmentation. Full replay
                # advances loader state without emulating SDC's sample limit.
                inputs = torch.tensor(batch_indices, dtype=torch.float32)
                inputs = inputs[:, None] + 100 * self.completed
                yield inputs, torch.zeros(2, dtype=torch.long)
            self.completed += 1

    def fixture(self, *, epochs=1, interval=1, use_sdc=True,
                method_name='adaptive'):
        from cl_methods.proto_evolve import ProtoEvolveCL
        from determinism import tensor_sha256

        spec = FormalSpec('isolet', method_name, 42)
        protocol = protocol_for(spec)
        method = object.__new__(ProtoEvolveCL)
        method.use_sdc = use_sdc
        method.sdc_interval = interval
        formal_audit._strict_method_contract(method, protocol)
        task_classes = [(0,), (1,)]
        records = []
        for task_id, classes in enumerate(task_classes):
            loader = self.Loader(classes)
            for epoch in range(epochs):
                if (method_name == 'adaptive' and task_id > 0 and use_sdc
                        and epoch % interval == 0):
                    for _batch in loader:
                        pass
                for batch, (inputs, _) in enumerate(loader):
                    records.append({
                        'loader_key': loader.audit_key,
                        'loader_iteration': len(loader.audit_sampler.epoch_orders) - 1,
                        'batch': batch,
                        'indices': loader.audit_sampler.epoch_orders[-1][
                            2 * batch:2 * (batch + 1)],
                        'dtype': str(inputs.dtype), 'shape': list(inputs.shape),
                        'batch_sha256': tensor_sha256(inputs),
                    })
        boundary = {'event_idx': 1, 'task_id': 1, 'timeline_step': 'event_1_CIL'}
        accesses = ([('validation', 'lambda_validation',
                      'final_validation_pre_install')]
                    if validation_access_for(spec) else [])
        accesses += [('test', 'test', 'final_test_post_install')]
        for split, prefix, phase in accesses:
            records.append({
                'event': 'first_iteration', 'split': split, 'phase': phase,
                **boundary, 'classes': [0, 1],
                'loader_key': repr((prefix, (0, 1))),
            })
        loaders = {classes: self.Loader(classes) for classes in task_classes}
        dataset = SimpleNamespace(
            get_train_loader=loaders.__getitem__, loaders=loaders)
        args = SimpleNamespace(epochs_per_task=epochs, model_type='resnet18')
        return dataset, args, records, task_classes, boundary, protocol, method

    def test_adaptive_sdc_first_task_one_batch(self):
        fixture = self.fixture()
        dataset, _, records, _, _, _, _ = fixture
        self.assertEqual(0, records[0]['loader_iteration'])
        self.assertEqual(1, records[3]['loader_iteration'])
        no_prepass_records = self.fixture(use_sdc=False)[2]
        first_mismatch = next(
            (task, epoch, batch)
            for task in range(2) for epoch in range(1) for batch in range(3)
            if records[3 * task + batch] != no_prepass_records[3 * task + batch])
        self.assertEqual((1, 0, 0), first_mismatch)
        formal_audit._strict_data_flow(*fixture)
        self.assertEqual(1, dataset.loaders[(0,)].completed)
        self.assertEqual(2, dataset.loaders[(1,)].completed)
        self.assertEqual((1, 2), dataset.loaders[(1,)].last_batch)

    def test_sdc_interval_two_only_prepasses_divisible_epochs(self):
        fixture = self.fixture(epochs=3, interval=2)
        dataset, _, records, _, _, _, _ = fixture
        self.assertEqual([0, 1, 2], [records[i]['loader_iteration']
                                     for i in (0, 3, 6)])
        self.assertEqual([1, 2, 4], [records[i]['loader_iteration']
                                     for i in (9, 12, 15)])
        formal_audit._strict_data_flow(*fixture)
        self.assertEqual(3, dataset.loaders[(0,)].completed)
        self.assertEqual(5, dataset.loaders[(1,)].completed)

    def test_disabled_sdc_preserves_epoch_iteration(self):
        fixture = self.fixture(epochs=3, use_sdc=False)
        self.assertEqual([0, 1, 2], [fixture[2][i]['loader_iteration']
                                     for i in (9, 12, 15)])
        formal_audit._strict_data_flow(*fixture)
        self.assertEqual(3, fixture[0].loaders[(1,)].completed)

    def test_non_adaptive_protocol_does_not_inspect_sdc_fields(self):
        fixture = self.fixture(epochs=3, method_name='finetune')
        formal_audit._strict_data_flow(*fixture[:-1], object())
        self.assertEqual(3, fixture[0].loaders[(1,)].completed)

    def test_adaptive_sdc_flag_requires_boolean(self):
        for value in (0, 1, 'true', None):
            with self.subTest(value=value):
                fixture = self.fixture()
                fixture[-1].use_sdc = value
                with self.assertRaisesRegex(ValueError, 'invalid adaptive SDC enable flag'):
                    formal_audit._strict_data_flow(*fixture)

    def test_adaptive_sdc_interval_requires_positive_nonboolean_integer(self):
        for value in (True, False, 1.0, '1', None, 0, -1):
            with self.subTest(value=value):
                fixture = self.fixture(use_sdc=False)
                fixture[-1].sdc_interval = value
                with self.assertRaisesRegex(ValueError, 'invalid adaptive SDC interval'):
                    formal_audit._strict_data_flow(*fixture)

    def test_adaptive_protocol_rejects_unexpected_method(self):
        fixture = self.fixture()
        impostor = SimpleNamespace(use_sdc=True, sdc_interval=1)
        with self.assertRaisesRegex(ValueError, 'invalid adaptive SDC method'):
            formal_audit._strict_data_flow(*fixture[:-1], impostor)

    def test_training_evidence_tampering_still_rejected(self):
        for field, value in (
                ('loader_iteration', 0), ('loader_iteration', 2),
                ('indices', [5, 4]), ('batch_sha256', '0' * 64),
                ('loader_key', repr(('train', (0,)))), ('batch', 1)):
            with self.subTest(field=field, value=value):
                fixture = self.fixture()
                fixture[2][3][field] = value
                with self.assertRaisesRegex(
                        ValueError, 'producer data flow does not match loader'):
                    formal_audit._strict_data_flow(*fixture)

    def test_missing_or_extra_producer_batches_still_rejected(self):
        for extra in (False, True):
            with self.subTest(extra=extra):
                fixture = self.fixture()
                if extra:
                    fixture[2].insert(3, copy.deepcopy(fixture[2][3]))
                else:
                    del fixture[2][3]
                with self.assertRaisesRegex(
                        ValueError, 'producer data flow does not match loader'):
                    formal_audit._strict_data_flow(*fixture)

    def test_checkpoint_boundary_still_validated(self):
        fixture = self.fixture()
        fixture[4]['task_id'] = 0
        with self.assertRaisesRegex(ValueError, 'checkpoint is not the final CIL boundary'):
            formal_audit._strict_data_flow(*fixture)


if __name__ == '__main__':
    unittest.main()
