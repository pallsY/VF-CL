"""Experiment runner: full CL+UL timeline."""
import copy
import codecs
import collections
import io
import pickle
import json, math, os, random, stat, time, torch, numpy as np
from collections.abc import Mapping
from data_utils import TaskManager, VFLDataset
from bic_calibration import (
    TaskAffineCalibrator,
    fit_final_calibrator,
    summarize_paired_logits,
)
from determinism import configure_determinism
from models import build_models, TopModel
from vfl_trainer import VFLTrainer
from cl_methods import get_cl_method
from ul_methods import get_ul_method
from adaptive_head_consolidation import (
    ADAPTIVE_METHOD_VERSION,
    AdaptiveConsolidationResult,
)
from metrics import (
    MetricsTracker,
    cache_formal_batches,
    collect_global_probs,
    evaluate_per_task,
    evaluate_per_task_full,
    evaluate_per_task_full_cached,
    evaluate_unlearning,
    select_formal_cached_batches,
)
from adaptive_consolidation_audit import (
    atomic_write_new_json,
    audit_adaptive_checkpoint,
    begin_formal_deferred_publication,
    complete_formal_deferred_publication,
    evaluate_deferred_cil_trajectory,
    evaluate_formal_deferred_trajectory,
    load_frozen_adaptive_evidence,
    prepare_adaptive_run_provenance,
    prepare_formal_deferred_evaluation,
    save_deferred_cil_snapshot,
    save_final_adaptive_checkpoint,
    trainer_state_sha256,
    _read_file,
    _restricted_torch_load,
    _same_file_identity,
    _same_inode_content,
    _safe_torch_load,
    _strict_load_trainer_state,
    _trusted_dir,
)


class _FormalTrainOnlyDataset:
    def __init__(self, dataset):
        self._dataset = dataset

    def get_train_loader(self, classes, shuffle=True):
        return self._dataset.get_train_loader(classes, shuffle=shuffle)


def _call_method_hook(hook, args, trainer, train_only_dataset, *hook_args):
    if not bool(getattr(args, 'formal_deferred_evaluation', False)):
        return hook(*hook_args)
    trainer.dataset_ref = train_only_dataset
    try:
        return hook(*hook_args)
    finally:
        trainer.dataset_ref = None


def _call_after_task(cl_method, args, trainer, train_only_dataset,
                     loader, task_id):
    adaptive = (
        getattr(args, 'formal_deferred_evaluation', False) is True
        and getattr(
            cl_method, 'head_consolidation_mode',
            getattr(args, 'head_consolidation_mode', ''),
        ) == 'adaptive_dual_branch'
        and callable(getattr(cl_method, '_consolidate_head', None))
    )
    if not adaptive:
        return _call_method_hook(
            cl_method.after_task, args, trainer, train_only_dataset,
            loader, task_id,
        )
    had_override = '_consolidate_head' in vars(cl_method)
    override = vars(cl_method).get('_consolidate_head')
    cl_method._consolidate_head = lambda *_args, **_kwargs: None
    try:
        return _call_method_hook(
            cl_method.after_task, args, trainer, train_only_dataset,
            loader, task_id,
        )
    finally:
        if had_override:
            cl_method._consolidate_head = override
        else:
            del cl_method._consolidate_head


def configure_task_ce(trainer, mode, new_classes, seen_classes):
    """Apply a method-independent CE scope after method.before_task()."""
    if mode == 'method':
        return
    if mode == 'full':
        trainer.ce_lo, trainer.ce_hi = 0, None
        trainer.ce_classes = None
        return
    if mode == 'current':
        classes = new_classes
    elif mode == 'seen':
        classes = seen_classes
    else:
        raise ValueError(f'unsupported task_ce_mode: {mode}')
    ordered = sorted(int(label) for label in classes)
    if not ordered:
        raise ValueError(f'task_ce_mode={mode} requires at least one class')
    if ordered != list(range(ordered[0], ordered[-1] + 1)):
        raise ValueError(
            f'task_ce_mode={mode} requires contiguous class ids, got {ordered}'
        )
    trainer.ce_classes = None
    trainer.ce_lo = ordered[0]
    trainer.ce_hi = ordered[-1] + 1


def initialize_experiment_rng(args):
    if getattr(args, 'deterministic', 0):
        if args.num_workers != 2:
            raise ValueError('deterministic protocol requires num_workers=2')
        configure_determinism(args.seed)
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def _save_final_probs(trainer, dataset, task_mgr, args):
    """Persist final-state output distribution on the retained test set, for
    post-hoc KL-to-Oracle. Sample order is set-determined (shuffle=False), so
    Oracle and every method align row-for-row when retained sets match."""
    try:
        retained = sorted(task_mgr.get_effective_classes())
        if not retained:
            return
        _, kl_test = dataset.get_task_loaders(retained, shuffle_train=False)
        probs, labels = collect_global_probs(trainer, kl_test, args.num_classes, args)
        np.savez(os.path.join(args.output_dir, 'final_probs.npz'),
                 probs=probs, labels=labels, retained=np.array(retained))
    except Exception as e:
        print(f"  [warn] failed to save final_probs for KL: {e}")


def _fsync_parent(target):
    parent = os.path.dirname(os.path.abspath(os.fspath(target)))
    descriptor = os.open(parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_torch_save(value, target):
    target = os.fspath(target)
    parent = os.path.dirname(os.path.abspath(target))
    target_name = os.path.basename(target)
    temporary = target_name + '.tmp'
    descriptor = None
    owns_temporary = False
    try:
        with _trusted_dir(parent) as directory:
            try:
                existing = os.stat(
                    target_name, dir_fd=directory, follow_symlinks=False
                )
            except FileNotFoundError:
                existing = None
            if existing is not None and not stat.S_ISREG(existing.st_mode):
                raise ValueError('unsafe checkpoint target is not replaceable')
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, 'O_NOFOLLOW', 0),
                0o600,
                dir_fd=directory,
            )
            owns_temporary = True
            with os.fdopen(descriptor, 'wb', closefd=False) as handle:
                torch.save(value, handle)
                handle.flush()
                os.fsync(handle.fileno())
            written = os.fstat(descriptor)
            current = os.stat(
                temporary, dir_fd=directory, follow_symlinks=False
            )
            if (not stat.S_ISREG(current.st_mode)
                    or not _same_file_identity(written, current)):
                raise RuntimeError('checkpoint temp changed before install')
            os.replace(
                temporary, target_name,
                src_dir_fd=directory, dst_dir_fd=directory,
            )
            installed = os.stat(
                target_name, dir_fd=directory, follow_symlinks=False
            )
            if not _same_file_identity(os.fstat(descriptor), installed):
                raise RuntimeError('checkpoint install changed inode')
            owns_temporary = False
            os.fsync(directory)
    finally:
        try:
            if owns_temporary:
                try:
                    with _trusted_dir(parent) as directory:
                        current = os.stat(
                            temporary, dir_fd=directory,
                            follow_symlinks=False,
                        )
                        if (descriptor is None
                                or not _same_file_identity(
                                    os.fstat(descriptor), current
                                )):
                            raise RuntimeError(
                                'checkpoint temp changed during cleanup'
                            )
                        os.unlink(temporary, dir_fd=directory)
                        os.fsync(directory)
                except FileNotFoundError:
                    pass
        finally:
            if descriptor is not None:
                os.close(descriptor)


def _atomic_json_dump(value, target):
    target = os.fspath(target)
    temporary = target + '.tmp'
    try:
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_parent(target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _publish_formal_deferred_result(
        *, args, final_checkpoint, tracker_state, final_result,
        bic_state=None, bic_history=None):
    """Republish the evaluated schema-v4 checkpoint/results fail closed."""
    from adaptive_consolidation_audit import _formal_source_provenance
    provenance = _formal_source_provenance()
    recorded_args = getattr(args, 'formal_source_provenance', None)
    if recorded_args is not None and recorded_args != provenance:
        raise ValueError('formal source provenance changed')
    recorded = final_result.get('source_provenance')
    if recorded is not None and recorded != provenance:
        raise ValueError('formal result source provenance mismatch')
    final_result['source_provenance'] = copy.deepcopy(provenance)
    if bool(getattr(args, 'bic_enabled', 0)) and (
            bic_state is None or type(bic_history) is not list
            or final_result.get('bic_history') != bic_history
            or final_result.get('bic_final')
            != (bic_history[-1] if bic_history else None)
            or type(final_result.get('bic_fit_corpus')) is not dict):
        raise ValueError('formal BiC publication state/results mismatch')
    begin_formal_deferred_publication(
        output_dir=args.output_dir,
        tracker_state=tracker_state,
        final_result=final_result,
    )
    payload = _safe_torch_load(final_checkpoint)
    if (type(payload) is not dict or payload.get('schema_version') != 4
            or set(payload) != _RESUME_CHECKPOINT_KEYS):
        raise ValueError('formal final checkpoint schema is invalid')
    payload['tracker_state'] = copy.deepcopy(tracker_state)
    if bool(getattr(args, 'bic_enabled', 0)):
        payload['bic_state'] = copy.deepcopy(bic_state)
        payload['bic_history'] = copy.deepcopy(bic_history)
    _atomic_torch_save(payload, final_checkpoint)
    results_path = os.path.join(args.output_dir, 'results.json')
    _atomic_json_dump(final_result, results_path)
    complete_formal_deferred_publication(
        args=args,
        output_dir=args.output_dir,
        final_checkpoint=final_checkpoint,
        results_path=results_path,
    )


def _evaluate_formal_from_single_access(
        *, args, dataset, snapshot_paths, final_checkpoint, task_classes,
        final_event, final_task, cached_calibration_batches=None,
        calibration_audit=None, validation_manifest=None,
        selection_audit=None):
    transaction = prepare_formal_deferred_evaluation(
        args=args, snapshot_paths=snapshot_paths,
        final_checkpoint=final_checkpoint, task_classes=task_classes,
        output_dir=args.output_dir,
    )
    if transaction['status'] != 'pending':
        return transaction['result'], transaction['status']
    ordered_tasks = sorted(task_classes)
    all_classes = [
        class_id for task_id in ordered_tasks
        for class_id in task_classes[task_id]
    ]
    if bool(getattr(args, 'bic_enabled', 0)):
        split, phase = _formal_bic_cache_plan(args)
        if split == 'validation':
            if cached_calibration_batches is None:
                raise RuntimeError(
                    'formal internal validation cache was not installed'
                )
        else:
            if cached_calibration_batches is not None:
                raise RuntimeError('formal external calibration cache is early')
            dataset.authorize_formal_calibration_access(
                phase=phase, event_idx=final_event, task_id=final_task,
                timeline_step=f'event_{final_event}_CIL',
                classes=all_classes,
            )
            calibration_loader = dataset.get_calibration_loader(all_classes)
            cached_calibration_batches = cache_formal_batches(
                calibration_loader
            )
        calibration_audit = (
            copy.deepcopy(dataset.calibration_audit())
            if calibration_audit is None else copy.deepcopy(calibration_audit)
        )
        validation_manifest = (
            copy.deepcopy(dataset.validation_manifest)
            if validation_manifest is None else copy.deepcopy(validation_manifest)
        )
        selection_audit = (
            copy.deepcopy(dataset.selection_audit())
            if selection_audit is None
            and getattr(args, 'lambda_validation_enabled', 0)
            else copy.deepcopy(selection_audit)
        )
    dataset.authorize_formal_access(
        split='test', phase='final_test_post_install',
        event_idx=final_event, task_id=final_task,
        timeline_step=f'event_{final_event}_CIL', classes=all_classes,
    )
    cached_test_batches = cache_formal_batches(
        dataset.get_test_loader(all_classes)
    )
    return evaluate_formal_deferred_trajectory(
        args=args, snapshot_paths=snapshot_paths,
        final_checkpoint=final_checkpoint, task_classes=task_classes,
        cached_test_batches=cached_test_batches, output_dir=args.output_dir,
        cached_calibration_batches=cached_calibration_batches,
        calibration_audit=calibration_audit,
        validation_manifest=validation_manifest,
        selection_audit=selection_audit,
    ), 'complete'


def _capture_rng_state():
    numpy_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': {
            'bit_generator': numpy_state[0],
            'keys': torch.from_numpy(numpy_state[1].astype(np.int64)),
            'position': int(numpy_state[2]),
            'has_gauss': int(numpy_state[3]),
            'cached_gaussian': float(numpy_state[4]),
        },
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state):
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((
        numpy_state['bit_generator'],
        numpy_state['keys'].detach().cpu().numpy().astype(np.uint32),
        numpy_state['position'], numpy_state['has_gauss'],
        numpy_state['cached_gaussian'],
    ))
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state.get('cuda'):
        torch.cuda.set_rng_state_all(state['cuda'])


def _checkpoint_protocol(args):
    protocol = {
        key: getattr(args, key, None)
        for key in (
            'seed', 'data', 'cl_method', 'num_tasks', 'num_parties',
            'head_consolidation_enabled', 'head_consolidation_mode',
        )
    }
    if getattr(args, 'formal_deferred_evaluation', False) is True:
        from adaptive_consolidation_audit import _formal_source_provenance
        provenance = _formal_source_provenance()
        recorded = getattr(args, 'formal_source_provenance', None)
        if recorded is not None and recorded != provenance:
            raise ValueError('formal source provenance changed')
        args.formal_source_provenance = copy.deepcopy(provenance)
        protocol['formal_deferred_evaluation'] = True
        protocol['source_provenance'] = copy.deepcopy(provenance)
        if getattr(args, 'cl_method', None) == 'er_ace':
            protocol.update({
                'num_classes': args.num_classes,
                'er_ace_buffer_size': getattr(args, 'er_ace_buffer_size', 0),
                'er_ace_batch': getattr(args, 'er_ace_batch', 64),
            })
        if getattr(args, 'cl_method', None) == 'fedprotip_vfl':
            protocol.update({
                'fedprotip_tip_threshold': float(
                    getattr(args, 'fedprotip_tip_threshold', 0.775)
                ),
                'fedprotip_max_batches': int(
                    getattr(args, 'fedprotip_max_batches', 20)
                ),
            })
        if bool(getattr(args, 'bic_enabled', 0)):
            protocol.update({
                'bic_enabled': True,
                'bic_fit_mode': getattr(args, 'bic_fit_mode', None),
                'bic_lr': float(getattr(args, 'bic_lr', 0.0)),
                'bic_steps': int(getattr(args, 'bic_steps', 0)),
                'bic_per_class': int(getattr(args, 'bic_per_class', 0)),
                'lambda_validation_enabled': bool(getattr(
                    args, 'lambda_validation_enabled', 0
                )),
                'lambda_validation_per_class': int(getattr(
                    args, 'lambda_validation_per_class', 0
                )),
                'lambda_validation_split_seed': int(getattr(
                    args, 'lambda_validation_split_seed', 0
                )),
            })
    return protocol


_RESUME_CHECKPOINT_KEYS = {
    'schema_version', 'step', 'event_idx', 'task_id', 'new_classes',
    'seen_task_classes', 'forgotten_classes', 'trainer_state', 'cl_state',
    'bic_state', 'bic_history', 'tracker_state', 'rng_state', 'protocol',
}


def _valid_method_checkpoint_state(method, state, adaptive, num_parties):
    if type(state) is not dict:
        return False
    if method == 'prl':
        protos = state.get('protos')
        return (set(state) == {'protos'} and type(protos) is dict
                and all(type(class_id) is int
                        and isinstance(value, np.ndarray)
                        and value.ndim == 1 and value.size
                        and np.issubdtype(value.dtype, np.floating)
                        for class_id, value in protos.items()))
    if method == 'proto_aug':
        protos = state.get('protos')
        return (set(state) == {'protos', 'radius'} and type(protos) is dict
                and all(type(class_id) is int
                        and isinstance(value, torch.Tensor)
                        and value.ndim == 1 and value.numel()
                        and torch.is_floating_point(value)
                        for class_id, value in protos.items())
                and type(state['radius']) in (float, np.float64))
    if adaptive:
        required = {
            'global_protos', 'prev_protos', 'fim_masks',
            'dep_tracker_contrib', 'current_task_classes', 'has_old_teacher',
            'class_party_contrib', 'class_party_weights',
            'effective_class_party_weights', 'party_weight_manifest_hash',
            'party_shuffle_seed', 'party_shuffle_mapping',
            'distill_weight_schedule', 'effective_distill_weight',
            'current_task_id', 'head_raw_replay', 'head_task_classes',
            'head_consolidation_history', 'adaptive_method_version',
            'adaptive_top_version', 'adaptive_class_order', 'adaptive_gate',
            'head_validation_sha256', 'adaptive_pending_task_id',
            'adaptive_audit_bundle',
        }
        return (method == 'proto_evolve' and set(state) == required
                and all(type(state[key]) is dict for key in (
                    'global_protos', 'prev_protos', 'class_party_contrib',
                    'class_party_weights', 'effective_class_party_weights',
                    'party_shuffle_mapping', 'head_raw_replay',
                    'head_task_classes',
                ))
                and type(state['fim_masks']) is list
                and len(state['fim_masks']) == num_parties
                and all(type(value) is dict for value in state['fim_masks'])
                and isinstance(state['dep_tracker_contrib'], torch.Tensor)
                and type(state['current_task_classes']) is list
                and all(type(value) is int
                        for value in state['current_task_classes'])
                and type(state['has_old_teacher']) is bool
                and type(state['party_weight_manifest_hash']) is str
                and type(state['party_shuffle_seed']) is int
                and type(state['distill_weight_schedule']) is str
                and type(state['effective_distill_weight']) is float
                and type(state['current_task_id']) is int
                and type(state['head_consolidation_history']) is list
                and type(state['adaptive_method_version']) is int
                and type(state['adaptive_top_version']) is int
                and type(state['adaptive_class_order']) is list
                and all(type(value) is int
                        for value in state['adaptive_class_order'])
                and (state['adaptive_gate'] is None
                     or type(state['adaptive_gate']) is float)
                and type(state['head_validation_sha256']) is str
                and (state['adaptive_pending_task_id'] is None
                     or type(state['adaptive_pending_task_id']) is int)
                and (state['adaptive_audit_bundle'] is None
                     or type(state['adaptive_audit_bundle']) is dict))
    return True


def _checkpoint_state_structure(value):
    if isinstance(value, Mapping):
        return ('mapping', tuple(
            (key, _checkpoint_state_structure(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        ))
    if type(value) in (list, tuple):
        return (type(value).__name__, tuple(
            _checkpoint_state_structure(item) for item in value
        ))
    if isinstance(value, torch.Tensor):
        return ('tensor', str(value.dtype), tuple(value.shape))
    if isinstance(value, np.ndarray):
        return ('ndarray', value.dtype.str, tuple(value.shape))
    return ('scalar', type(value).__name__)


_ADAPTIVE_DYNAMIC_TOP_BUFFERS = frozenset({
    '_adaptive_full_weight',
    '_adaptive_full_bias',
    '_adaptive_class_order',
})


def _resume_trainer_structure_matches(fresh, saved, adaptive):
    if not adaptive:
        return (_checkpoint_state_structure(fresh)
                == _checkpoint_state_structure(saved))
    if (type(fresh) is not dict or type(saved) is not dict
            or set(fresh) != {'bottoms', 'top_model'}
            or set(saved) != {'bottoms', 'top_model'}
            or not isinstance(fresh['top_model'], Mapping)
            or not isinstance(saved['top_model'], Mapping)
            or set(fresh['top_model']) != set(saved['top_model'])
            or not _ADAPTIVE_DYNAMIC_TOP_BUFFERS.issubset(
                fresh['top_model'])):
        return False
    for key in _ADAPTIVE_DYNAMIC_TOP_BUFFERS:
        left, right = fresh['top_model'][key], saved['top_model'][key]
        if (not isinstance(left, torch.Tensor)
                or not isinstance(right, torch.Tensor)
                or left.dtype != right.dtype):
            return False
    fresh_static = {
        'bottoms': fresh['bottoms'],
        'top_model': {
            key: value for key, value in fresh['top_model'].items()
            if key not in _ADAPTIVE_DYNAMIC_TOP_BUFFERS
        },
    }
    saved_static = {
        'bottoms': saved['bottoms'],
        'top_model': {
            key: value for key, value in saved['top_model'].items()
            if key not in _ADAPTIVE_DYNAMIC_TOP_BUFFERS
        },
    }
    return (_checkpoint_state_structure(fresh_static)
            == _checkpoint_state_structure(saved_static))


def _checkpoint_values_equal(left, right):
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (set(left) == set(right)
                and all(_checkpoint_values_equal(left[key], right[key])
                        for key in left))
    if type(left) in (list, tuple) and type(right) is type(left):
        return (len(left) == len(right)
                and all(_checkpoint_values_equal(a, b)
                        for a, b in zip(left, right)))
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape)
                and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return (left.dtype == right.dtype and left.shape == right.shape
                and np.array_equal(left, right))
    if (type(left) is float and type(right) is float
            and math.isnan(left) and math.isnan(right)):
        return True
    return type(left) is type(right) and left == right


def _is_metric_number(value):
    return type(value) in (int, float)


def _is_accuracy_map(value):
    return (type(value) is dict
            and all(type(key) is str
                    and (item is None or _is_metric_number(item))
                    for key, item in value.items()))


def _same_accuracy_keys(primary, *companions):
    keys = set(primary)
    return all(value is None or set(value) == keys for value in companions)


def _is_comm_payload(value, include_step=False):
    required = {'comm_rounds', 'megabytes_transmitted'}
    if include_step:
        required.add('step')
    return (type(value) is dict and set(value) == required
            and (not include_step or type(value['step']) is str)
            and type(value['comm_rounds']) is int
            and _is_metric_number(value['megabytes_transmitted']))


def _is_ul_payload(value, include_step=False):
    required = {'forget_acc', 'retain_acc', 'mia_score'}
    if include_step:
        required.add('step')
    if (type(value) is not dict or not required.issubset(value)
            or not set(value).issubset(required | {
                'parties_touched', 'parties_total',
                'relearn_auc', 'reconnect_auc',
            })
            or (include_step and type(value['step']) is not str)
            or any(value[key] is not None and not _is_metric_number(value[key])
                   for key in ('forget_acc', 'retain_acc', 'mia_score'))):
        return False
    has_touched = 'parties_touched' in value
    has_total = 'parties_total' in value
    if (has_touched != has_total
            or (has_touched
                and (type(value['parties_touched']) is not int
                     or type(value['parties_total']) is not int))):
        return False
    if 'reconnect_auc' in value and 'relearn_auc' not in value:
        return False
    for key in ('relearn_auc', 'reconnect_auc'):
        if key in value and (
                type(value[key]) is not dict
                or any(type(class_id) is not int
                       or not _is_metric_number(score)
                       for class_id, score in value[key].items())):
            return False
    return True


def _valid_task_accuracy_record(
        record, oracle_steps, oracle_protocol, deferred_protocol,
        deferred_final_task):
    base = {'step', 'per_task_accs', 'overall_acc'}
    normal = {
        frozenset(base),
        frozenset(base | {'per_task_accs_taskil'}),
        frozenset(base | {
            'per_task_accs_debiased', 'per_task_accs_taskil',
        }),
        frozenset(base | {
            'per_task_accs_taskil', 'companion_readouts',
        }),
        frozenset(base | {
            'per_task_accs_debiased', 'per_task_accs_taskil',
            'companion_readouts',
        }),
    }
    deferred_diagonal = frozenset(base | {'deferred_diagonal'})
    deferred_final = frozenset(base | {
        'per_task_accs_taskil', 'deferred_diagonal', 'deferred_final',
        'deferred_final_task',
    })
    deferred_final_companions = frozenset(
        deferred_final | {'companion_readouts'}
    )
    keys = frozenset(record) if type(record) is dict else frozenset()
    if (type(record) is not dict or not base.issubset(record)
            or type(record['step']) is not str
            or not _is_accuracy_map(record['per_task_accs'])
            or not _is_metric_number(record['overall_acc'])):
        return False
    oracle_row = record['step'] in oracle_steps
    if oracle_row:
        if (not oracle_protocol
                or keys != frozenset(base | {
                    'per_task_accs_debiased', 'per_task_accs_taskil',
                })):
            return False
    elif oracle_protocol or keys not in normal | {
            deferred_diagonal, deferred_final, deferred_final_companions}:
        return False
    elif not record['step'].endswith(('_CIL', '_UL')):
        return False
    elif ((keys in {
            deferred_diagonal, deferred_final, deferred_final_companions})
          != deferred_protocol):
        return False
    for key in ('per_task_accs_debiased', 'per_task_accs_taskil',
                'deferred_diagonal'):
        if key in record and not _is_accuracy_map(record[key]):
            return False
    if 'companion_readouts' in record and (
            type(record['companion_readouts']) is not dict
            or any(type(name) is not str or not _is_accuracy_map(values)
                   for name, values in record['companion_readouts'].items())):
        return False
    if keys == deferred_diagonal:
        return (record['step'].endswith('_CIL')
                and bool(record['per_task_accs'])
                and _checkpoint_values_equal(
                    record['per_task_accs'], record['deferred_diagonal']))
    if keys in {deferred_final, deferred_final_companions} and (
            not record['step'].endswith('_CIL')
            or record['deferred_final'] is not True
            or record['deferred_final_task'] != deferred_final_task
            or not record['per_task_accs']
            or set(record['per_task_accs'])
            != set(record['per_task_accs_taskil'])
            or set(record['per_task_accs'])
            != set(record['deferred_diagonal'])):
        return False
    return True


def _is_oracle_protocol(protocol):
    return (type(protocol) is dict
            and protocol.get('cl_method') == 'oracle'
            and not protocol.get('head_consolidation_enabled')
            and protocol.get('head_consolidation_mode') == 'full_classifier')


def _is_deferred_protocol(protocol):
    if (type(protocol) is not dict
            or type(protocol.get('num_tasks')) is not int
            or protocol['num_tasks'] <= 0):
        return False
    if _is_formal_deferred_protocol(protocol):
        return True
    return (protocol.get('cl_method') == 'proto_evolve'
            and bool(protocol.get('head_consolidation_enabled'))
            and protocol.get('head_consolidation_mode')
            == 'adaptive_dual_branch')


def _is_formal_deferred_protocol(protocol):
    return (type(protocol) is dict
            and protocol.get('formal_deferred_evaluation') is True
            and type(protocol.get('num_tasks')) is int
            and protocol['num_tasks'] > 0)


def _oracle_step_name(record):
    return f"event_{record['event_idx']}_{record['type']}"


def _valid_step_record(record, oracle_protocol, formal_protocol):
    if (type(record) is not dict or type(record.get('event_idx')) is not int
            or record.get('type') not in {'CIL', 'UL'}):
        return False
    keys = set(record)
    oracle = {
        'event_idx', 'type', 'overall_acc', 'per_task',
        'per_task_debiased', 'per_task_taskil',
    }
    cil_deferred = {
        'event_idx', 'type', 'task_id', 'evaluation_deferred',
        'train_time', 'comm',
    }
    cil_formal_deferred = cil_deferred | {'new_classes'}
    cil_evaluated = {
        'event_idx', 'type', 'task_id', 'per_task_acc',
        'per_task_acc_debiased', 'per_task_acc_taskil', 'overall_acc',
        'companion_readouts', 'train_time', 'comm',
    }
    ul_deferred = {
        'event_idx', 'type', 'forget_classes', 'evaluation_deferred', 'comm',
    }
    ul_evaluated = {
        'event_idx', 'type', 'forget_classes', 'ul_eval', 'per_task_acc',
        'per_task_acc_debiased', 'per_task_acc_taskil', 'overall_acc',
    }
    if keys == oracle:
        return (oracle_protocol
                and all(_is_accuracy_map(record[key]) for key in (
                    'per_task', 'per_task_debiased', 'per_task_taskil'))
                and _same_accuracy_keys(
                    record['per_task'], record['per_task_debiased'],
                    record['per_task_taskil'])
                and _is_metric_number(record['overall_acc']))
    if oracle_protocol:
        return False
    if keys == cil_deferred or keys == cil_formal_deferred:
        return (record['type'] == 'CIL'
                and type(record['task_id']) is int
                and ((keys == cil_formal_deferred) == formal_protocol)
                and (keys != cil_formal_deferred
                     or (type(record['new_classes']) is list
                         and bool(record['new_classes'])
                         and all(type(value) is int
                                 for value in record['new_classes'])))
                and record['evaluation_deferred'] is True
                and _is_metric_number(record['train_time'])
                and _is_comm_payload(record['comm']))
    if keys == cil_evaluated:
        return (record['type'] == 'CIL'
                and type(record['task_id']) is int
                and _is_accuracy_map(record['per_task_acc'])
                and (record['per_task_acc_debiased'] is None
                     or _is_accuracy_map(record['per_task_acc_debiased']))
                and _is_accuracy_map(record['per_task_acc_taskil'])
                and _is_metric_number(record['overall_acc'])
                and type(record['companion_readouts']) is dict
                and all(type(name) is str and _is_accuracy_map(values)
                        for name, values
                        in record['companion_readouts'].items())
                and _same_accuracy_keys(
                    record['per_task_acc'],
                    record['per_task_acc_debiased'],
                    record['per_task_acc_taskil'],
                    *record['companion_readouts'].values())
                and _is_metric_number(record['train_time'])
                and _is_comm_payload(record['comm']))
    if keys == ul_deferred:
        return (record['type'] == 'UL'
                and type(record['forget_classes']) is list
                and all(type(value) is int
                        for value in record['forget_classes'])
                and record['evaluation_deferred'] is True
                and _is_comm_payload(record['comm']))
    if keys == ul_evaluated:
        return (record['type'] == 'UL'
                and type(record['forget_classes']) is list
                and all(type(value) is int
                        for value in record['forget_classes'])
                and _is_ul_payload(record['ul_eval'])
                and all(_is_accuracy_map(record[key]) for key in (
                    'per_task_acc', 'per_task_acc_debiased',
                    'per_task_acc_taskil'))
                and _same_accuracy_keys(
                    record['per_task_acc'],
                    record['per_task_acc_debiased'],
                    record['per_task_acc_taskil'])
                and _is_metric_number(record['overall_acc']))
    return False


def _step_producer(record, oracle_protocol, formal_protocol):
    if not _valid_step_record(record, oracle_protocol, formal_protocol):
        return None
    if oracle_protocol:
        kind = 'oracle'
    elif record.get('evaluation_deferred') is True:
        kind = f"{record['type'].lower()}_deferred"
    else:
        kind = f"{record['type'].lower()}_evaluated"
    return {
        'step': _oracle_step_name(record),
        'type': record['type'],
        'kind': kind,
        'record': record,
    }


def _task_row_matches_producer(row, producer):
    record = producer['record']
    expected = {'step': producer['step']}
    if producer['kind'] == 'oracle':
        expected.update({
            'per_task_accs': record['per_task'],
            'overall_acc': record['overall_acc'],
            'per_task_accs_debiased': record['per_task_debiased'],
            'per_task_accs_taskil': record['per_task_taskil'],
        })
    elif producer['kind'] == 'cil_evaluated':
        expected.update({
            'per_task_accs': record['per_task_acc'],
            'overall_acc': record['overall_acc'],
            'per_task_accs_taskil': record['per_task_acc_taskil'],
        })
        if record['per_task_acc_debiased'] is not None:
            expected['per_task_accs_debiased'] = \
                record['per_task_acc_debiased']
        if record['companion_readouts']:
            expected['companion_readouts'] = record['companion_readouts']
    elif producer['kind'] == 'ul_evaluated':
        expected.update({
            'per_task_accs': record['per_task_acc'],
            'overall_acc': record['overall_acc'],
            'per_task_accs_debiased': record['per_task_acc_debiased'],
            'per_task_accs_taskil': record['per_task_acc_taskil'],
        })
    else:
        return False
    return _checkpoint_values_equal(row, expected)


def _valid_tracker_checkpoint_state(state, protocol):
    if (type(state) is not dict or set(state) != {
                'cl_metrics', 'task_acc_history', 'ul_metrics', 'comm_stats',
                'timing', 'step_results',
            }):
        return False
    oracle_protocol = _is_oracle_protocol(protocol)
    formal_protocol = _is_formal_deferred_protocol(protocol)
    deferred_protocol = _is_deferred_protocol(protocol)
    deferred_final_task = f"task_{protocol['num_tasks'] - 1}" \
        if deferred_protocol else None
    if type(state['step_results']) is not list:
        return False
    producers = [
        _step_producer(record, oracle_protocol, formal_protocol)
        for record in state['step_results']
    ]
    if (any(producer is None for producer in producers)
            or len({producer['step'] for producer in producers})
            != len(producers)):
        return False
    if (deferred_protocol and any(producer['kind'] not in {
            'cil_deferred', 'ul_deferred'} for producer in producers)):
        return False
    oracle_steps = {
        producer['step'] for producer in producers
        if producer['kind'] == 'oracle'
    }
    records_valid = (type(state['cl_metrics']) is dict
            and type(state['task_acc_history']) is list
            and all(_valid_task_accuracy_record(
                        record, oracle_steps, oracle_protocol,
                        deferred_protocol, deferred_final_task)
                    for record in state['task_acc_history'])
            and type(state['ul_metrics']) is list
            and all(_is_ul_payload(record, include_step=True)
                    for record in state['ul_metrics'])
            and type(state['comm_stats']) is list
            and all(_is_comm_payload(record, include_step=True)
                    for record in state['comm_stats'])
            and type(state['timing']) is list
            and all(type(record) is dict
                    and set(record) == {'step', 'time_seconds'}
                    and type(record['step']) is str
                    and _is_metric_number(record['time_seconds'])
                    for record in state['timing']))
    if not records_valid:
        return False
    deferred_rows = [
        row for row in state['task_acc_history']
        if 'deferred_diagonal' in row or 'deferred_final' in row
    ]
    if deferred_rows:
        if (not deferred_protocol
                or len(deferred_rows) != len(state['task_acc_history'])
                or state['ul_metrics']):
            return False
        if not producers:
            return not state['comm_stats'] and not state['timing']
        if any(producer['kind'] not in {
                'cil_deferred', 'ul_deferred'} for producer in producers):
            return False
        cil_steps = [producer['step'] for producer in producers
                     if producer['kind'] == 'cil_deferred']
        all_steps = [producer['step'] for producer in producers]
        if ([row['step'] for row in state['task_acc_history']] != cil_steps
                or [record['step'] for record in state['comm_stats']]
                != all_steps
                or [record['step'] for record in state['timing']]
                != all_steps):
            return False
        return all(_checkpoint_values_equal(
            {key: value for key, value in comm.items() if key != 'step'},
            producer['record']['comm'],
        ) for comm, producer in zip(state['comm_stats'], producers))
    if (not producers and type(protocol) is dict
            and protocol.get('head_consolidation_mode') is None):
        return (not state['ul_metrics'] and not state['comm_stats']
                and not state['timing'])
    task_producers = [
        producer for producer in producers
        if producer['kind'] in {
            'oracle', 'cil_evaluated', 'ul_evaluated',
        }
    ]
    if (len(state['task_acc_history']) != len(task_producers)
            or any(not _task_row_matches_producer(row, producer)
                   for row, producer in zip(
                       state['task_acc_history'], task_producers))):
        return False
    all_steps = [producer['step'] for producer in producers]
    if ([record['step'] for record in state['comm_stats']] != all_steps
            or [record['step'] for record in state['timing']] != all_steps):
        return False
    ul_producers = [
        producer for producer in producers
        if producer['type'] == 'UL'
        and producer['kind'] in {'oracle', 'ul_evaluated'}
    ]
    if ([record['step'] for record in state['ul_metrics']]
            != [producer['step'] for producer in ul_producers]):
        return False
    for metric, producer in zip(state['ul_metrics'], ul_producers):
        if (producer['kind'] == 'ul_evaluated'
                and not _checkpoint_values_equal(
                    {key: value for key, value in metric.items()
                     if key != 'step'}, producer['record']['ul_eval'])):
            return False
    for comm, producer in zip(state['comm_stats'], producers):
        if ('comm' in producer['record']
                and not _checkpoint_values_equal(
                    {key: value for key, value in comm.items()
                     if key != 'step'}, producer['record']['comm'])):
            return False
    return True


def _validate_tracker_producer_timeline(state, protocol, timeline, event_idx):
    if protocol.get('head_consolidation_mode') not in {
            'full_classifier', 'task_class_bias', 'adaptive_dual_branch'}:
        return
    oracle_protocol = _is_oracle_protocol(protocol)
    formal_protocol = _is_formal_deferred_protocol(protocol)
    producers = [
        _step_producer(record, oracle_protocol, formal_protocol)
        for record in state['step_results']
    ]
    expected = timeline[:event_idx + 1]
    if len(producers) != len(expected):
        raise ValueError('resume checkpoint tracker timeline mismatch')
    active_task_classes = {}
    forgotten_classes = set()
    final_active_keys = set()
    for expected_idx, (producer, event) in enumerate(zip(producers, expected)):
        record = producer['record']
        if (record['event_idx'] != expected_idx
                or record['type'] != event['type']
                or producer['step']
                != f"event_{expected_idx}_{event['type']}"):
            raise ValueError('resume checkpoint tracker timeline mismatch')
        if (event['type'] == 'CIL'
                and (producer['kind'] != (
                    'oracle' if oracle_protocol else (
                        'cil_deferred' if _is_deferred_protocol(protocol)
                        else 'cil_evaluated'))
                     or record.get('task_id', event['task_id'])
                     != event['task_id']
                     or (protocol.get('formal_deferred_evaluation') is True
                         and record.get('new_classes')
                         != event['new_classes']))):
            raise ValueError('resume checkpoint tracker timeline mismatch')
        if (event['type'] == 'UL'
                and (producer['kind'] != (
                    'oracle' if oracle_protocol else (
                        'ul_deferred' if _is_deferred_protocol(protocol)
                        else 'ul_evaluated'))
                     or record.get('forget_classes', event['forget_classes'])
                     != event['forget_classes'])):
            raise ValueError('resume checkpoint tracker timeline mismatch')
        if event['type'] == 'CIL':
            active_task_classes[int(event['task_id'])] = [
                int(class_id) for class_id in event['new_classes']
                if int(class_id) not in forgotten_classes
            ]
        else:
            forgotten_classes.update(
                int(class_id) for class_id in event['forget_classes']
            )
            for task_id, classes in active_task_classes.items():
                active_task_classes[task_id] = [
                    class_id for class_id in classes
                    if class_id not in forgotten_classes
                ]
        final_active_keys = {
            f'task_{task_id}' for task_id, classes
            in active_task_classes.items() if classes
        }
        maps = []
        if producer['kind'] == 'oracle':
            maps = [
                record['per_task'], record['per_task_debiased'],
                record['per_task_taskil'],
            ]
        elif producer['kind'] in {'cil_evaluated', 'ul_evaluated'}:
            maps = [record['per_task_acc'], record['per_task_acc_taskil']]
            if record['per_task_acc_debiased'] is not None:
                maps.append(record['per_task_acc_debiased'])
            maps.extend(record.get('companion_readouts', {}).values())
        if any(set(values) != final_active_keys for values in maps):
            raise ValueError('resume checkpoint tracker task identity mismatch')
        if event['type'] == 'UL':
            ul_metric = next(
                (metric for metric in state['ul_metrics']
                 if metric['step'] == producer['step']), None
            )
            if ul_metric is not None:
                expected_classes = {
                    int(class_id) for class_id in event['forget_classes']
                }
                for key in ('relearn_auc', 'reconnect_auc'):
                    if (key in ul_metric
                            and set(ul_metric[key]) != expected_classes):
                        raise ValueError(
                            'resume checkpoint UL class identity mismatch'
                        )
    deferred_rows = [
        row for row in state['task_acc_history']
        if 'deferred_diagonal' in row or 'deferred_final' in row
    ]
    if deferred_rows:
        cil_events = [event for event in expected if event['type'] == 'CIL']
        for row, event in zip(deferred_rows, cil_events):
            expected_keys = final_active_keys \
                if row.get('deferred_final') is True \
                else {f"task_{int(event['task_id'])}"}
            if set(row['per_task_accs']) != expected_keys:
                raise ValueError(
                    'resume checkpoint deferred task identity mismatch'
                )


def _validate_rng_checkpoint_state(state):
    try:
        python_probe = random.Random()
        python_probe.setstate(state['python'])
        numpy_state = state['numpy']
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state((
            numpy_state['bit_generator'],
            numpy_state['keys'].detach().cpu().numpy().astype(np.uint32),
            numpy_state['position'], numpy_state['has_gauss'],
            numpy_state['cached_gaussian'],
        ))
        torch_probe = torch.Generator(device='cpu')
        torch_probe.set_state(state['torch'].detach().cpu())
        cuda_states = state['cuda']
        if cuda_states:
            if not torch.cuda.is_available() \
                    or len(cuda_states) != torch.cuda.device_count():
                raise ValueError('CUDA RNG device count mismatch')
            for index, cuda_state in enumerate(cuda_states):
                cuda_probe = torch.Generator(device=f'cuda:{index}')
                cuda_probe.set_state(cuda_state.detach().cpu())
    except Exception as error:
        raise ValueError('resume checkpoint RNG state is invalid') from error


def _validate_adaptive_checkpoint_semantics(
        state, trainer_state, protocol, args):
    if state['adaptive_method_version'] != ADAPTIVE_METHOD_VERSION:
        raise ValueError('adaptive method version mismatch')
    top = trainer_state['top_model']
    required_top = {
        '_adaptive_version', '_adaptive_enabled', '_adaptive_class_order',
        '_adaptive_gate',
    }
    if not required_top.issubset(top):
        raise ValueError('adaptive top metadata is missing')
    top_version = int(top['_adaptive_version'].item())
    top_enabled = bool(top['_adaptive_enabled'].item())
    top_classes = [
        int(class_id) for class_id in top['_adaptive_class_order'].tolist()
    ]
    top_gate = float(top['_adaptive_gate'].item()) if top_enabled else None
    if (state['adaptive_top_version'] != top_version
            or state['adaptive_class_order'] != top_classes
            or state['adaptive_gate'] != top_gate):
        raise ValueError('adaptive method/top metadata mismatch')
    history = state['head_consolidation_history']
    try:
        normalized = [
            AdaptiveConsolidationResult.from_dict(record).to_dict()
            for record in history
        ]
    except (TypeError, ValueError) as error:
        raise ValueError('adaptive method history is invalid') from error
    validation_hash = state['head_validation_sha256']
    pending = state['adaptive_pending_task_id']
    if normalized:
        if len(normalized) != 1:
            raise ValueError('adaptive method history mismatch')
        last = normalized[0]
        if (last.get('method_version') != ADAPTIVE_METHOD_VERSION
                or last.get('task_id') != protocol['num_tasks'] - 1
                or last.get('ordered_classes') != top_classes
                or last.get('gate', {}).get('g') != top_gate
                or last.get('validation_manifest', {}).get('sha256')
                != validation_hash
                or top_version != ADAPTIVE_METHOD_VERSION
                or pending is not None):
            raise ValueError('adaptive method history mismatch')
    elif (validation_hash or top_version != 0 or top_classes
          or top_gate is not None):
        raise ValueError('adaptive method history mismatch')
    if pending is not None:
        after_tasks = {
            int(value) for value in getattr(args, 'unlearn_after_tasks', [])
        }
        available = len(getattr(args, 'unlearn_classes', []))
        scheduled = 0
        final_ul_follows = False
        for task_id in range(protocol['num_tasks']):
            if task_id in after_tasks and scheduled < available:
                if task_id == pending:
                    final_ul_follows = True
                    break
                scheduled += 1
        if (pending != protocol['num_tasks'] - 1
                or not final_ul_follows):
            raise ValueError('adaptive pending task mismatch')


def _validate_checkpoint_replacement(candidate, current):
    candidate_decoded = _decode_checkpoint_value(candidate) \
        if candidate['schema_version'] == 4 else candidate
    current_decoded = _decode_checkpoint_value(current) \
        if current['schema_version'] == 4 else current
    if (_checkpoint_state_structure(candidate['trainer_state'])
            != _checkpoint_state_structure(current['trainer_state'])):
        raise ValueError('new checkpoint trainer structure is incompatible')
    if (_checkpoint_state_structure(candidate['rng_state'])
            != _checkpoint_state_structure(current['rng_state'])):
        raise ValueError('new checkpoint RNG structure is incompatible')
    candidate_method = candidate_decoded['cl_state']
    current_method = current_decoded['cl_state']
    if (set(candidate_method) != set(current_method)
            or any(type(candidate_method[key]) is not type(current_method[key])
                   for key in candidate_method)):
        raise ValueError('new checkpoint method structure is incompatible')


def _validate_resume_checkpoint_payload(checkpoint, args):
    protocol = checkpoint.get('protocol') if type(checkpoint) is dict else None
    trainer_state = checkpoint.get('trainer_state') \
        if type(checkpoint) is dict else None
    rng_state = checkpoint.get('rng_state') if type(checkpoint) is dict else None
    version = checkpoint.get('schema_version') if type(checkpoint) is dict else None

    def valid_state_dict(value):
        return (isinstance(value, Mapping) and bool(value)
                and all(type(key) is str and isinstance(item, torch.Tensor)
                        for key, item in value.items()))

    adaptive = bool(
        getattr(args, 'head_consolidation_enabled', 0)
        and getattr(args, 'head_consolidation_mode', '')
        == 'adaptive_dual_branch'
    )
    num_parties = protocol.get('num_parties') \
        if type(protocol) is dict else None
    valid_parties = (
        type(num_parties) is int and num_parties >= 0
    ) or (not adaptive and num_parties is None)
    valid_adaptive_trainer = (
        type(trainer_state) is dict
        and set(trainer_state) == {'bottoms', 'top_model'}
        and type(trainer_state['bottoms']) is list
        and len(trainer_state['bottoms']) == num_parties
        and all(valid_state_dict(value) for value in trainer_state['bottoms'])
        and valid_state_dict(trainer_state['top_model'])
    ) if valid_parties and adaptive else not adaptive
    tracker_state = checkpoint.get('tracker_state') \
        if type(checkpoint) is dict else None
    valid_tracker = _valid_tracker_checkpoint_state(tracker_state, protocol)
    if (type(checkpoint) is not dict
            or set(checkpoint) != _RESUME_CHECKPOINT_KEYS
            or version not in {3, 4}
            or (version == 3 and adaptive)
            or protocol != _checkpoint_protocol(args)
            or type(protocol.get('num_tasks')) is not int
            or protocol['num_tasks'] <= 0
            or not valid_parties
            or type(checkpoint.get('task_id')) is not int
            or not 0 <= checkpoint['task_id'] < protocol['num_tasks']
            or type(checkpoint.get('new_classes')) is not list
            or any(type(value) is not int for value in checkpoint['new_classes'])
            or type(checkpoint.get('seen_task_classes')) is not dict
            or any(type(task_id) is not int or type(classes) is not list
                   or not 0 <= task_id < protocol['num_tasks']
                   or any(type(value) is not int for value in classes)
                   for task_id, classes in checkpoint['seen_task_classes'].items())
            or type(checkpoint.get('forgotten_classes')) is not list
            or any(type(value) is not int
                   for value in checkpoint['forgotten_classes'])
            or type(trainer_state) is not dict
            or not valid_adaptive_trainer
            or type(checkpoint.get('cl_state')) is not dict
            or (checkpoint.get('bic_state') is not None
                and not isinstance(checkpoint['bic_state'], Mapping))
            or type(checkpoint.get('bic_history')) is not list
            or not valid_tracker
            or type(rng_state) is not dict
            or set(rng_state) != {'python', 'numpy', 'torch', 'cuda'}
            or type(rng_state['python']) is not tuple
            or len(rng_state['python']) != 3
            or type(rng_state['python'][0]) is not int
            or type(rng_state['python'][1]) is not tuple
            or not rng_state['python'][1]
            or any(type(value) is not int for value in rng_state['python'][1])
            or (rng_state['python'][2] is not None
                and type(rng_state['python'][2]) is not float)
            or type(rng_state['numpy']) is not dict
            or set(rng_state['numpy']) != {
                'bit_generator', 'keys', 'position', 'has_gauss',
                'cached_gaussian',
            }
            or rng_state['numpy']['bit_generator'] != 'MT19937'
            or not isinstance(rng_state['numpy']['keys'], torch.Tensor)
            or rng_state['numpy']['keys'].dtype != torch.int64
            or rng_state['numpy']['keys'].ndim != 1
            or not rng_state['numpy']['keys'].numel()
            or type(rng_state['numpy']['position']) is not int
            or not 0 <= rng_state['numpy']['position'] \
                <= rng_state['numpy']['keys'].numel()
            or type(rng_state['numpy']['has_gauss']) is not int
            or rng_state['numpy']['has_gauss'] not in (0, 1)
            or type(rng_state['numpy']['cached_gaussian']) is not float
            or not isinstance(rng_state['torch'], torch.Tensor)
            or rng_state['torch'].dtype != torch.uint8
            or rng_state['torch'].ndim != 1
            or not rng_state['torch'].numel()
            or type(rng_state['cuda']) is not list
            or any(not isinstance(value, torch.Tensor)
                   or value.dtype != torch.uint8
                   or value.ndim != 1 or not value.numel()
                   for value in rng_state['cuda'])):
        raise ValueError('resume checkpoint schema/state/protocol mismatch')
    try:
        tracker_probe = MetricsTracker()
        tracker_probe.load_dict(tracker_state)
        tracker_reloaded = tracker_probe.to_dict()
    except Exception as error:
        raise ValueError('resume checkpoint tracker state mismatch') from error
    if not _checkpoint_values_equal(tracker_reloaded, tracker_state):
        raise ValueError('resume checkpoint tracker state mismatch')
    try:
        method_state = _decode_checkpoint_value(checkpoint['cl_state']) \
            if version == 4 else checkpoint['cl_state']
    except Exception as error:
        raise ValueError('resume checkpoint method state mismatch') from error
    if not _valid_method_checkpoint_state(
            protocol['cl_method'], method_state, adaptive,
            protocol['num_parties']):
        raise ValueError('resume checkpoint method state mismatch')
    _validate_rng_checkpoint_state(rng_state)
    if adaptive:
        _validate_adaptive_checkpoint_semantics(
            method_state, trainer_state, protocol, args
        )
    return checkpoint


def _validate_completed_method_history(checkpoint, method):
    """Bind internally valid continuation state to its completed CIL/UL stage."""
    name = checkpoint['protocol'].get('cl_method')
    if name not in {'target', 'proto_fedspace', 'er_ace'}:
        return
    seen = checkpoint['seen_task_classes']
    forgotten = set(checkpoint['forgotten_classes'])
    all_seen = {c for classes in seen.values() for c in classes}
    if not seen or not all_seen or not forgotten.issubset(all_seen):
        raise ValueError('resume checkpoint completed class history is invalid')
    # The explicit relapse ablation keeps caches despite the outer UL history.
    if not getattr(method.args, 'sanitize_cl_state', 1):
        forgotten = set()
    retained = all_seen - forgotten
    if name == 'target':
        # TARGET keeps generator conditioning tables, including forgotten rows.
        # Their internal order need not equal a custom task's input class order.
        if (set(method.task_classes) != set(seen)
                or any(set(method.task_classes[tid]) != set(classes)
                       for tid, classes in seen.items())
                or method.forgotten != forgotten):
            raise ValueError('TARGET completed task/class history mismatch')
    elif name == 'proto_fedspace':
        if set(method.protos) != retained:
            raise ValueError('ProtoFedSpace retained class history mismatch')
    else:
        if retained and not method.buffer.size():
            prefix = 'formal ' if _is_formal_deferred_protocol(checkpoint['protocol']) else ''
            raise ValueError(prefix + 'ER-ACE requires nonempty replay state')
        if not set(method.buffer.lb).issubset(retained):
            raise ValueError('ER-ACE retained class history mismatch')


def _validate_resume_candidate_loadability(
        checkpoint, args, trainer, cl_method, task_mgr,
        bic_calibrator, tracker):
    _validate_resume_checkpoint_payload(checkpoint, args)
    decoded = _decode_checkpoint_value(checkpoint) \
        if checkpoint['schema_version'] == 4 else checkpoint
    trainer_probe = copy.deepcopy(trainer)
    adaptive = bool(
        decoded['protocol']['head_consolidation_enabled']
        and decoded['protocol']['head_consolidation_mode']
        == 'adaptive_dual_branch'
    )
    formal = _is_formal_deferred_protocol(decoded['protocol'])
    if not _resume_trainer_structure_matches(
            trainer_probe.get_state(), decoded['trainer_state'], adaptive):
        raise ValueError('resume checkpoint trainer structure mismatch')
    if adaptive or formal:
        _strict_load_trainer_state(trainer_probe, decoded['trainer_state'], {
            'num_parties': decoded['protocol']['num_parties'],
        })
    else:
        trainer_probe.load_state(decoded['trainer_state'])
    if not _checkpoint_values_equal(
            trainer_probe.get_state(), decoded['trainer_state']):
        raise ValueError('resume checkpoint trainer reload mismatch')
    method_probe = copy.deepcopy(cl_method)
    if hasattr(method_probe, 'trainer'):
        method_probe.trainer = trainer_probe
    if hasattr(method_probe, 'load_state'):
        # Reconstructing TARGET generators consumes RNG even on a rejected
        # candidate. Probes must not perturb any stream before final selection.
        probe_rng = _capture_rng_state()
        try:
            method_probe.load_state(decoded['cl_state'])
        finally:
            _restore_rng_state(probe_rng)
    if (hasattr(method_probe, 'get_state')
            and not _checkpoint_values_equal(
                method_probe.get_state(), decoded['cl_state'])):
        raise ValueError('resume checkpoint method reload mismatch')
    _validate_completed_method_history(decoded, method_probe)
    if decoded['bic_state'] is not None:
        copy.deepcopy(bic_calibrator).load_state_dict(decoded['bic_state'])
    tracker_probe = copy.deepcopy(tracker)
    tracker_probe.load_dict(decoded['tracker_state'])
    if not _checkpoint_values_equal(
            tracker_probe.to_dict(), decoded['tracker_state']):
        raise ValueError('resume checkpoint tracker reload mismatch')
    task_probe = copy.deepcopy(task_mgr)
    if hasattr(task_probe, 'get_timeline'):
        timeline = task_probe.get_timeline()
        event_idx = decoded['event_idx']
        if not 0 <= event_idx < len(timeline):
            raise ValueError('resume checkpoint timeline event is invalid')
        _validate_tracker_producer_timeline(
            decoded['tracker_state'], decoded['protocol'], timeline, event_idx
        )
        expected_seen = {}
        expected_forgotten = []
        latest_task_id = None
        for event in timeline[:event_idx + 1]:
            if event['type'] == 'CIL':
                latest_task_id = int(event['task_id'])
                expected_seen[latest_task_id] = [
                    int(class_id) for class_id in event['new_classes']
                ]
            else:
                expected_forgotten.extend(
                    int(class_id) for class_id in event['forget_classes']
                )
        current = timeline[event_idx]
        if (decoded['step'] != f"event_{event_idx}_{current['type']}"
                or decoded['task_id'] != latest_task_id
                or decoded['seen_task_classes'] != expected_seen
                or decoded['new_classes'] != expected_seen[latest_task_id]
                or decoded['forgotten_classes'] != expected_forgotten):
            raise ValueError('resume checkpoint task/class timeline mismatch')
    for task_id in sorted(decoded['seen_task_classes']):
        task_probe.advance_task(task_id)
    if decoded['forgotten_classes']:
        task_probe.apply_unlearn(decoded['forgotten_classes'])
    return decoded


def _encode_checkpoint_key(value):
    if isinstance(value, np.generic):
        return value.item()
    if type(value) is tuple:
        return tuple(_encode_checkpoint_key(item) for item in value)
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError('checkpoint mapping key is not canonical')


def _encode_checkpoint_value(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError('checkpoint NumPy object arrays are not supported')
        return {
            '__vfcl_numpy_array_v1__': True,
            'dtype': value.dtype.str,
            'tensor': torch.from_numpy(np.ascontiguousarray(value)),
        }
    if isinstance(value, np.generic):
        return {
            '__vfcl_numpy_scalar_v1__': True,
            'dtype': value.dtype.str,
            'value': value.item(),
        }
    if isinstance(value, dict):
        encoded = {}
        for key, item in value.items():
            encoded[_encode_checkpoint_key(key)] = _encode_checkpoint_value(item)
        return encoded
    if isinstance(value, list):
        return [_encode_checkpoint_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_encode_checkpoint_value(item) for item in value)
    return value


def _decode_checkpoint_value(value):
    if type(value) is dict and set(value) == {
            '__vfcl_numpy_array_v1__', 'dtype', 'tensor'}:
        if value['__vfcl_numpy_array_v1__'] is not True \
                or not isinstance(value['tensor'], torch.Tensor):
            raise ValueError('checkpoint NumPy array encoding is invalid')
        array = value['tensor'].detach().cpu().numpy().copy()
        dtype = np.dtype(value['dtype'])
        if dtype.hasobject:
            raise ValueError('checkpoint NumPy object arrays are not supported')
        return array.astype(dtype, copy=False)
    if type(value) is dict and set(value) == {
            '__vfcl_numpy_scalar_v1__', 'dtype', 'value'}:
        if value['__vfcl_numpy_scalar_v1__'] is not True:
            raise ValueError('checkpoint NumPy scalar encoding is invalid')
        dtype = np.dtype(value['dtype'])
        if dtype.hasobject:
            raise ValueError('checkpoint NumPy object scalars are not supported')
        return dtype.type(value['value'])
    if type(value) is dict:
        return {key: _decode_checkpoint_value(item) for key, item in value.items()}
    if type(value) is list:
        return [_decode_checkpoint_value(item) for item in value]
    if type(value) is tuple:
        return tuple(_decode_checkpoint_value(item) for item in value)
    return value


def _safe_legacy_fixed_numpy_load(path, args, content=None):
    if (getattr(args, 'cl_method', None) not in {'prl', 'proto_aug'}
            or (getattr(args, 'head_consolidation_enabled', 0)
                and getattr(args, 'head_consolidation_mode', '')
                == 'adaptive_dual_branch')):
        raise ValueError('legacy NumPy resume is not enabled for this mode')
    if content is None:
        content, _ = _read_file(path)
    numpy_reconstruct = np._core.multiarray._reconstruct
    numpy_scalar = np._core.multiarray.scalar
    allowed = {
        ('_codecs', 'encode'): codecs.encode,
        ('collections', 'OrderedDict'): collections.OrderedDict,
        ('numpy', 'dtype'): np.dtype,
        ('numpy', 'ndarray'): np.ndarray,
        ('numpy.core.multiarray', '_reconstruct'): numpy_reconstruct,
        ('numpy._core.multiarray', '_reconstruct'): numpy_reconstruct,
        ('numpy.core.multiarray', 'scalar'): numpy_scalar,
        ('numpy._core.multiarray', 'scalar'): numpy_scalar,
        ('torch._utils', '_rebuild_tensor_v2'): torch._utils._rebuild_tensor_v2,
    }
    for name in (
            'ByteStorage', 'CharStorage', 'ShortStorage', 'IntStorage',
            'LongStorage', 'HalfStorage', 'FloatStorage', 'DoubleStorage',
            'BoolStorage', 'BFloat16Storage'):
        storage = getattr(torch, name, None)
        if storage is not None:
            allowed[('torch', name)] = storage

    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return allowed[(module, name)]
            except KeyError as error:
                raise pickle.UnpicklingError(
                    f'legacy checkpoint global is forbidden: {module}.{name}'
                ) from error

    class RestrictedPickleModule:
        __name__ = 'vfcl_legacy_fixed_numpy_pickle'
        Unpickler = RestrictedUnpickler

        @staticmethod
        def load(handle, **kwargs):
            return RestrictedUnpickler(handle, **kwargs).load()

        @staticmethod
        def loads(data, **kwargs):
            return RestrictedUnpickler(io.BytesIO(data), **kwargs).load()

    try:
        checkpoint = torch.load(
            io.BytesIO(content), map_location='cpu', weights_only=False,
            pickle_module=RestrictedPickleModule,
        )
    except Exception as error:
        raise ValueError('legacy fixed NumPy checkpoint load failed') from error
    protocol = checkpoint.get('protocol') if type(checkpoint) is dict else None
    state = checkpoint.get('cl_state') if type(checkpoint) is dict else None
    method = getattr(args, 'cl_method', None)
    numpy_elsewhere = {
        key: value for key, value in checkpoint.items() if key != 'cl_state'
    } if type(checkpoint) is dict else None

    def contains_numpy(value):
        if isinstance(value, (np.ndarray, np.generic)):
            return True
        if type(value) is dict:
            return any(contains_numpy(key) or contains_numpy(item)
                       for key, item in value.items())
        if type(value) in (list, tuple):
            return any(contains_numpy(item) for item in value)
        return False

    if (type(checkpoint) is not dict
            or set(checkpoint) != _RESUME_CHECKPOINT_KEYS
            or checkpoint.get('schema_version') != 3
            or type(protocol) is not dict
            or protocol != _checkpoint_protocol(args)
            or (protocol.get('head_consolidation_enabled')
                and protocol.get('head_consolidation_mode')
                == 'adaptive_dual_branch')
            or not _valid_method_checkpoint_state(
                method, state, False, protocol.get('num_parties', -1)
            )
            or contains_numpy(numpy_elsewhere)):
        raise ValueError('legacy fixed NumPy checkpoint schema mismatch')
    return checkpoint


def _save_cil_checkpoint(trainer, cl_method, args, step, task_id, new_classes, seen_task_classes,
                         bic_state=None, tracker_state=None, bic_history=None, force=False,
                         forgotten_classes=None):
    if not force and not getattr(args, 'save_task_checkpoints', 0):
        return
    if not force and args.save_task_checkpoints == 2 and task_id != args.num_tasks - 1:
        return
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    if force:
        filename = f'{step}.pt'
    elif args.save_task_checkpoints == 3:
        filename = (
            f'{step}.pt' if task_id == args.num_tasks - 1
            else 'resume_latest.pt'
        )
    else:
        filename = f'{step}.pt'
    ckpt = _encode_checkpoint_value({
        'schema_version': 4,
        'step': step,
        'event_idx': int(step.split('_')[1]),
        'task_id': task_id,
        'new_classes': list(new_classes),
        'seen_task_classes': {int(k): list(v) for k, v in seen_task_classes.items()},
        'forgotten_classes': [int(class_id) for class_id in (forgotten_classes or [])],
        'trainer_state': trainer.get_state(),
        'cl_state': cl_method.get_state() if hasattr(cl_method, 'get_state') else {},
        'bic_state': bic_state,
        'bic_history': list(bic_history or []),
        'tracker_state': (
            tracker_state if tracker_state is not None
            else MetricsTracker().to_dict()
        ),
        'rng_state': _capture_rng_state(),
        'protocol': _checkpoint_protocol(args),
    })
    _atomic_torch_save(ckpt, os.path.join(ckpt_dir, filename))
    if args.save_task_checkpoints == 3:
        rolling = os.path.join(ckpt_dir, 'resume_latest.pt')
        if force:
            _atomic_torch_save(ckpt, rolling)
        elif task_id == args.num_tasks - 1:
            if os.path.exists(rolling):
                os.remove(rolling)
                _fsync_parent(rolling)


def _load_resume_checkpoint(args, trainer, cl_method, task_mgr, tracker, bic_calibrator):
    if not getattr(args, 'resume_run_dir', ''):
        return 0, {}, []
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    if not os.path.isdir(checkpoint_dir):
        return 0, {}, []
    _recover_atomic_checkpoint_temps(
        checkpoint_dir, args, trainer, cl_method, task_mgr,
        tracker, bic_calibrator,
    )
    event_files = []
    for name in os.listdir(checkpoint_dir):
        parts = name[:-3].split('_') if name.endswith('.pt') else []
        if (len(parts) == 3 and parts[0] == 'event' and parts[1].isdigit()
                and parts[2] in {'CIL', 'UL'}):
            event_files.append(os.path.join(checkpoint_dir, name))
    candidates = [os.path.join(checkpoint_dir, 'resume_latest.pt')] + sorted(event_files)
    checkpoints = []
    errors = []
    for path in candidates:
        try:
            try:
                candidate = _safe_torch_load(path)
            except ValueError:
                candidate = _safe_legacy_fixed_numpy_load(path, args)
            candidate = _validate_resume_candidate_loadability(
                candidate, args, trainer, cl_method,
                task_mgr, bic_calibrator, tracker,
            )
            event_idx = int(candidate['event_idx'])
            step = str(candidate['step'])
            step_parts = step.split('_')
            if (len(step_parts) != 3 or step_parts[0] != 'event'
                    or not step_parts[1].isdigit() or int(step_parts[1]) != event_idx
                    or step_parts[2] not in {'CIL', 'UL'}):
                raise ValueError(f'invalid checkpoint step {step!r}')
            if os.path.basename(path) != 'resume_latest.pt' and os.path.basename(path) != f'{step}.pt':
                raise ValueError('checkpoint filename does not match recorded step')
            checkpoints.append((
                event_idx,
                os.path.basename(path) != 'resume_latest.pt',
                os.path.basename(path),
                candidate,
            ))
        except FileNotFoundError:
            continue
        except Exception as exc:
            errors.append(f'{path}: {exc}')
    if not checkpoints:
        if errors:
            raise RuntimeError('no valid resume checkpoint: ' + '; '.join(errors))
        return 0, {}, []
    unique_events = {}
    for item in checkpoints:
        event_idx = item[0]
        previous = unique_events.get(event_idx)
        if previous is not None and not _checkpoint_values_equal(
                previous[3], item[3]):
            raise ValueError(
                'conflicting resume checkpoints have the same event identity'
            )
        if previous is None or item[:3] > previous[:3]:
            unique_events[event_idx] = item
    checkpoints = list(unique_events.values())
    checkpoint = max(checkpoints, key=lambda item: item[:3])[3]
    expected = _checkpoint_protocol(args)
    recorded = checkpoint.get('protocol', {})
    mismatch = {
        key: (recorded.get(key), value)
        for key, value in expected.items()
        if recorded.get(key) != value
    }
    if mismatch:
        raise ValueError(f'resume checkpoint protocol mismatch: {mismatch}')
    adaptive_resume = bool(
        recorded.get('head_consolidation_enabled')
        and recorded.get('head_consolidation_mode') == 'adaptive_dual_branch'
    )
    if adaptive_resume and (
            type(checkpoint) is not dict
            or set(checkpoint) != _RESUME_CHECKPOINT_KEYS
            or checkpoint.get('schema_version') != 4):
        raise ValueError('adaptive resume checkpoint schema mismatch')
    if adaptive_resume or recorded.get('formal_deferred_evaluation') is True:
        _strict_load_trainer_state(trainer, checkpoint['trainer_state'], {
            'num_parties': recorded['num_parties'],
        })
    else:
        trainer.load_state(checkpoint['trainer_state'])
    if hasattr(cl_method, 'load_state'):
        cl_method.load_state(checkpoint.get('cl_state', {}))
    if checkpoint.get('bic_state') is not None:
        bic_calibrator.load_state_dict(checkpoint['bic_state'])
    tracker.load_dict(checkpoint.get('tracker_state', {}))
    seen_task_classes = {
        int(task_id): [int(class_id) for class_id in classes]
        for task_id, classes in checkpoint['seen_task_classes'].items()
    }
    for task_id in sorted(seen_task_classes):
        task_mgr.advance_task(task_id)
    forgotten_classes = [int(class_id) for class_id in checkpoint.get('forgotten_classes', [])]
    if forgotten_classes:
        task_mgr.apply_unlearn(forgotten_classes)
    _restore_rng_state(checkpoint['rng_state'])
    _recover_adaptive_cil_snapshot(checkpoint, trainer, cl_method, args)
    start_event_idx = int(checkpoint['event_idx']) + 1
    print(f"[resume] loaded {checkpoint['step']}; continuing at event {start_event_idx}")
    return start_event_idx, seen_task_classes, list(checkpoint.get('bic_history', []))


def _recover_atomic_checkpoint_temps(
        checkpoint_dir, args, trainer=None, cl_method=None, task_mgr=None,
        tracker=None, bic_calibrator=None):
    """Recover a durable checkpoint temp or discard a torn regular temp."""
    probes = (trainer, cl_method, task_mgr, tracker, bic_calibrator)
    has_live_probes = all(probe is not None for probe in probes)
    if any(probe is not None for probe in probes) and not has_live_probes:
        raise ValueError('checkpoint recovery requires every live probe')

    def validate_payload(payload):
        _validate_resume_checkpoint_payload(payload, args)
        if has_live_probes:
            _validate_resume_candidate_loadability(
                payload, args, trainer, cl_method, task_mgr,
                bic_calibrator, tracker,
            )

    def load_payload(path, content):
        try:
            return _restricted_torch_load(content)
        except ValueError:
            if args is None:
                raise
            return _safe_legacy_fixed_numpy_load(path, args, content)

    def identity(payload, target_name):
        if type(payload) is not dict:
            raise ValueError('checkpoint temp payload is invalid')
        step = payload.get('step')
        event_idx = payload.get('event_idx')
        parts = step.split('_') if type(step) is str else []
        if (type(event_idx) is not int or len(parts) != 3
                or parts[0] != 'event' or not parts[1].isdigit()
                or int(parts[1]) != event_idx
                or parts[2] not in {'CIL', 'UL'}
                or (target_name != 'resume_latest.pt'
                    and target_name != f'{step}.pt')):
            raise ValueError('checkpoint temp identity is invalid')
        return event_idx, step

    for name in sorted(os.listdir(checkpoint_dir)):
        if not name.endswith('.pt.tmp'):
            continue
        path = os.path.join(checkpoint_dir, name)
        content, pinned = _read_file(path)
        try:
            payload = load_payload(path, content)
        except ValueError:
            with _trusted_dir(checkpoint_dir) as directory:
                details = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if (not stat.S_ISREG(details.st_mode)
                        or not _same_file_identity(pinned, details)):
                    raise ValueError('unsafe checkpoint temp is not recoverable')
                os.unlink(name, dir_fd=directory)
                os.fsync(directory)
            continue
        validate_payload(payload)
        target_name = name[:-4]
        temp_event, _ = identity(payload, target_name)
        if not has_live_probes:
            continue
        with _trusted_dir(checkpoint_dir) as directory:
            current = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if (not stat.S_ISREG(current.st_mode)
                    or not _same_file_identity(pinned, current)):
                raise RuntimeError('checkpoint temp changed during recovery')
            try:
                target = os.stat(
                    target_name, dir_fd=directory, follow_symlinks=False
                )
            except FileNotFoundError:
                os.link(
                    name, target_name,
                    src_dir_fd=directory, dst_dir_fd=directory,
                    follow_symlinks=False,
                )
                installed = os.stat(
                    target_name, dir_fd=directory, follow_symlinks=False
                )
                if not _same_inode_content(pinned, installed):
                    raise RuntimeError(
                        'checkpoint recovery installed wrong inode'
                    )
            else:
                if not stat.S_ISREG(target.st_mode):
                    raise ValueError('unsafe recovered checkpoint target')
                target_path = os.path.join(checkpoint_dir, target_name)
                target_content, target_pinned = _read_file(target_path)
                target_payload = load_payload(target_path, target_content)
                validate_payload(target_payload)
                target_event, _ = identity(target_payload, target_name)
                target_now = os.stat(
                    target_name, dir_fd=directory, follow_symlinks=False
                )
                current = os.stat(
                    name, dir_fd=directory, follow_symlinks=False
                )
                if (not _same_file_identity(target_pinned, target_now)
                        or not _same_file_identity(pinned, current)):
                    raise RuntimeError(
                        'checkpoint changed during recovery decision'
                    )
                if temp_event > target_event:
                    _validate_checkpoint_replacement(payload, target_payload)
                    os.replace(
                        name, target_name,
                        src_dir_fd=directory, dst_dir_fd=directory,
                    )
                    installed = os.stat(
                        target_name, dir_fd=directory, follow_symlinks=False
                    )
                    if not _same_inode_content(pinned, installed):
                        raise RuntimeError(
                            'checkpoint recovery installed wrong inode'
                        )
                    os.fsync(directory)
                    continue
                if temp_event == target_event and content != target_content:
                    raise ValueError(
                        'conflicting checkpoint temp has the same event identity'
                    )
                os.unlink(name, dir_fd=directory)
                os.fsync(directory)
                continue
            current = os.stat(
                name, dir_fd=directory, follow_symlinks=False
            )
            if not _same_inode_content(pinned, current):
                raise RuntimeError(
                    'checkpoint temp changed during recovery cleanup'
                )
            os.unlink(name, dir_fd=directory)
            os.fsync(directory)


def _recover_adaptive_cil_snapshot(checkpoint, trainer, cl_method, args):
    """Materialize a missing stage snapshot from its durable CIL checkpoint."""
    adaptive = bool(
        getattr(args, 'head_consolidation_enabled', 0)
        and getattr(args, 'head_consolidation_mode', '')
        == 'adaptive_dual_branch'
    )
    step = checkpoint.get('step') if type(checkpoint) is dict else None
    formal = getattr(args, 'formal_deferred_evaluation', False) is True
    if ((not adaptive and not formal)
            or type(step) is not str or not step.endswith('_CIL')):
        return None
    return save_deferred_cil_snapshot(
        trainer, cl_method, args,
        checkpoint['event_idx'], checkpoint['task_id'],
        checkpoint['seen_task_classes'],
        protocol_kind='formal' if formal else 'adaptive',
    )


def _prepare_adaptive_provenance(
        args, trainer=None, cl_method=None, task_mgr=None,
        tracker=None, bic_calibrator=None):
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    if (getattr(args, 'resume_run_dir', '')
            and os.path.lexists(checkpoint_dir)):
        _recover_atomic_checkpoint_temps(
            checkpoint_dir, args, trainer, cl_method, task_mgr,
            tracker, bic_calibrator,
        )
    return prepare_adaptive_run_provenance(args.output_dir)


def _save_party_kd_audit(cl_method, args, step):
    if not getattr(args, 'party_kd_enabled', 0) or not hasattr(cl_method, 'get_state'):
        return
    state = cl_method.get_state()
    audit_dir = os.path.join(args.output_dir, 'party_kd_audit')
    os.makedirs(audit_dir, exist_ok=True)
    torch.save({
        'step': step,
        'party_kd_mode': args.party_kd_mode,
        'class_party_weights': state.get('class_party_weights', {}),
        'effective_class_party_weights': state.get('effective_class_party_weights', {}),
        'party_weight_manifest_hash': state.get('party_weight_manifest_hash', ''),
        'party_shuffle_seed': state.get('party_shuffle_seed', -1),
        'party_shuffle_mapping': state.get('party_shuffle_mapping', {}),
    }, os.path.join(audit_dir, f'{step}.pt'))


def _fit_and_evaluate_bic(trainer, dataset, calibrator, args, step, task_id,
                          new_classes, seen_task_classes):
    seen_classes = [class_id for tid in sorted(seen_task_classes)
                    for class_id in seen_task_classes[tid]]
    calibration_loader = dataset.get_calibration_loader(seen_classes)
    calibration_logits, calibration_labels = trainer.collect_logits(calibration_loader)
    fit = calibrator.fit_task(
        calibration_logits, calibration_labels, task_id, new_classes, seen_classes,
        args.bic_lr, args.bic_steps,
    )
    rows, labels = [], []
    for tid in sorted(seen_task_classes):
        _, test_loader = dataset.get_task_loaders(
            seen_task_classes[tid], shuffle_train=False
        )
        task_logits, task_labels = trainer.collect_logits(test_loader)
        rows.append(task_logits)
        labels.append(task_labels)
    logits, labels = torch.cat(rows), torch.cat(labels)
    paired = summarize_paired_logits(logits, labels, seen_task_classes, calibrator)
    task_il_delta = max(
        abs(paired['calibrated']['task_il'][key] - paired['raw']['task_il'][key])
        for key in paired['raw']['task_il']
    )
    record = {
        'step': step,
        'task_id': int(task_id),
        'fit': fit,
        'paired': paired,
        'calibration_audit': dataset.calibration_audit(),
        'disabled_identity_max_abs_diff': float(
            (TaskAffineCalibrator().apply(logits) - logits).abs().max()
        ),
        'task_il_max_abs_delta': float(task_il_delta),
    }
    output_dir = os.path.join(args.output_dir, 'bic')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{step}.json'), 'w', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    torch.save(calibrator.state_dict(), os.path.join(output_dir, 'calibrator.pt'))
    return record


def _should_fit_bic(args, task_id):
    if not getattr(args, 'bic_enabled', 0):
        return False
    mode = getattr(args, 'bic_fit_mode', 'sequential')
    return (mode in ('sequential', 'joint_each_stage') or
            int(task_id) == int(args.num_tasks) - 1)


def _fit_and_evaluate_final_bic(trainer, dataset, calibrator, args, step,
                                seen_task_classes):
    seen_classes = [class_id for tid in sorted(seen_task_classes)
                    for class_id in seen_task_classes[tid]]
    calibration_loader = dataset.get_calibration_loader(seen_classes)
    calibration_logits, calibration_labels = trainer.collect_logits(calibration_loader)
    fitted, fit = fit_final_calibrator(
        calibration_logits, calibration_labels, seen_task_classes,
        'joint_alpha_beta', args.bic_lr, args.bic_steps,
    )
    calibrator.load_state_dict(fitted.state_dict())
    rows, labels = [], []
    for tid in sorted(seen_task_classes):
        _, test_loader = dataset.get_task_loaders(
            seen_task_classes[tid], shuffle_train=False
        )
        task_logits, task_labels = trainer.collect_logits(test_loader)
        rows.append(task_logits)
        labels.append(task_labels)
    logits, labels = torch.cat(rows), torch.cat(labels)
    paired = summarize_paired_logits(logits, labels, seen_task_classes, calibrator)
    task_il_delta = max(
        abs(paired['calibrated']['task_il'][key] - paired['raw']['task_il'][key])
        for key in paired['raw']['task_il']
    )
    calibration_audit = dataset.calibration_audit()
    record = {
        'step': step,
        'task_id': max(seen_task_classes),
        'fit': fit,
        'paired': paired,
        'parameters': {
            str(task_id): calibrator.parameters_for(task_id)
            for task_id in sorted(seen_task_classes)
        },
        'calibration_audit': calibration_audit,
        'disabled_identity_max_abs_diff': float(
            (TaskAffineCalibrator().apply(logits) - logits).abs().max()
        ),
        'task_il_max_abs_delta': float(task_il_delta),
        'privacy_audit': {
            'passed': bool(calibration_audit['passed']),
            'test_used_for_fit': False,
            'raw_images_saved': False,
            'party_embeddings_saved': False,
        },
    }
    output_dir = os.path.join(args.output_dir, 'bic')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{step}.json'), 'w', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    torch.save(calibrator.state_dict(), os.path.join(output_dir, 'calibrator.pt'))
    return record


def _validate_formal_union_logits(bundle, expected_labels, seen_classes):
    if (type(bundle) is not dict
            or set(bundle) != {'logits', 'labels', 'seen_classes'}):
        raise ValueError('formal union logits schema is invalid')
    logits, labels = bundle['logits'], bundle['labels']
    seen = tuple(int(value) for value in seen_classes)
    if (not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor)
            or logits.device.type != 'cpu' or labels.device.type != 'cpu'
            or logits.dtype != torch.float32 or labels.dtype != torch.long
            or logits.requires_grad or labels.requires_grad
            or logits.grad_fn is not None or labels.grad_fn is not None
            or logits.ndim != 2 or labels.ndim != 1
            or logits.shape[0] != labels.numel() or labels.numel() == 0
            or type(bundle['seen_classes']) is not tuple
            or any(type(value) is not int for value in bundle['seen_classes'])
            or bundle['seen_classes'] != seen
            or not seen or len(seen) != len(set(seen))
            or min(seen) < 0 or max(seen) >= logits.shape[1]
            or not torch.isfinite(logits).all()
            or set(labels.tolist()) != set(seen)
            or not torch.equal(labels, expected_labels)):
        raise ValueError('formal union logits identity is invalid')
    return logits, labels


def _formal_union_logits(trainer, selected, seen_classes):
    logits, labels = trainer.collect_logits(selected)
    bundle = {
        'logits': logits.detach().to(device='cpu', dtype=torch.float32),
        'labels': labels.detach().to(device='cpu', dtype=torch.long),
        'seen_classes': tuple(int(value) for value in seen_classes),
    }
    _validate_formal_union_logits(
        bundle, torch.cat([labels for _, labels in selected]), seen_classes
    )
    return bundle


def _fit_and_evaluate_final_bic_cached(
        trainer, cached_calibration_batches, cached_test_batches, calibrator,
        args, step, seen_task_classes, calibration_audit, fit_corpus=None,
        precomputed_test=None):
    """Rebuild one producer-equivalent BiC record from immutable caches."""
    seen_task_classes = {
        int(task_id): [int(class_id) for class_id in classes]
        for task_id, classes in seen_task_classes.items()
    }
    seen_classes = [
        class_id for task_id in sorted(seen_task_classes)
        for class_id in seen_task_classes[task_id]
    ]
    if precomputed_test is not None:
        expected_labels = torch.cat([
            labels for _, labels in select_formal_cached_batches(
                cached_test_batches, seen_classes
            )
        ])
        logits, labels = _validate_formal_union_logits(
            precomputed_test, expected_labels, seen_classes
        )
        test_before = (logits.clone(), labels.clone())

        def check_unchanged_test():
            current = _validate_formal_union_logits(
                precomputed_test, expected_labels, seen_classes
            )
            if any(not torch.equal(before, after)
                   for before, after in zip(test_before, current)):
                raise ValueError('formal union logits mutated after handoff')

    calibration_logits, calibration_labels = trainer.collect_logits(
        select_formal_cached_batches(cached_calibration_batches, seen_classes)
    )
    calibration_logits = calibration_logits.detach().to(
        device='cpu', dtype=torch.float32
    )
    calibration_labels = calibration_labels.detach().to(
        device='cpu', dtype=torch.long
    )
    fitted, fit = fit_final_calibrator(
        calibration_logits, calibration_labels, seen_task_classes,
        'joint_alpha_beta', args.bic_lr, args.bic_steps,
    )
    calibrator.load_state_dict(fitted.state_dict())
    if precomputed_test is None:
        logits, labels = trainer.collect_logits(
            select_formal_cached_batches(cached_test_batches, seen_classes)
        )
        logits = logits.detach().to(device='cpu', dtype=torch.float32)
        labels = labels.detach().to(device='cpu', dtype=torch.long)
    else:
        check_unchanged_test()
        if logits.shape[1] != calibration_logits.shape[1]:
            raise ValueError('formal union logits full-head width mismatch')
    paired = summarize_paired_logits(
        logits, labels, seen_task_classes, calibrator
    )
    task_il_delta = max(
        abs(paired['calibrated']['task_il'][key]
            - paired['raw']['task_il'][key])
        for key in paired['raw']['task_il']
    )
    audit = copy.deepcopy(calibration_audit)
    record = {
        'step': step,
        'task_id': max(seen_task_classes),
        'fit': fit,
        'paired': paired,
        'parameters': {
            str(task_id): calibrator.parameters_for(task_id)
            for task_id in sorted(seen_task_classes)
        },
        'calibration_audit': audit,
        'disabled_identity_max_abs_diff': float(
            (TaskAffineCalibrator().apply(logits) - logits).abs().max()
        ),
        'task_il_max_abs_delta': float(task_il_delta),
        'privacy_audit': {
            'passed': bool(audit.get('passed')),
            'test_used_for_fit': False,
            'raw_images_saved': False,
            'party_embeddings_saved': False,
        },
    }
    if fit_corpus is not None:
        record['fit_corpus'] = copy.deepcopy(fit_corpus)
    if precomputed_test is not None:
        check_unchanged_test()
    return record


def _formal_bic_cache_plan(args):
    if (getattr(args, 'data', None) != 'cifar100'
            or type(getattr(args, 'num_tasks', None)) is not int
            or args.num_tasks != 10
            or not bool(getattr(args, 'bic_enabled', 0))
            or getattr(args, 'bic_fit_mode', None) != 'joint_each_stage'
            or not bool(getattr(args, 'lambda_validation_enabled', 0))):
        raise ValueError(
            'formal BiC requires CIFAR, 10 tasks, joint_each_stage, '
            'and the authoritative validation split'
        )
    internal = _is_formal_internal_method(args)
    return (
        ('validation', 'final_validation_pre_install')
        if internal else
        ('calibration', 'final_bic_calibration_post_freeze')
    )


def _is_formal_internal_method(args):
    return (
        getattr(args, 'cl_method', None) == 'proto_evolve'
        and bool(getattr(args, 'head_consolidation_enabled', 0))
        and getattr(args, 'head_consolidation_mode', 'full_classifier')
        != 'full_classifier'
    )


def _finalize_formal_internal_state(
        *, args, dataset, cl_method, task_classes, final_event, final_task):
    """Install the final internal head from one authorized validation cache."""
    if not _is_formal_internal_method(args):
        return None
    all_classes = [
        class_id for task_id in sorted(task_classes)
        for class_id in task_classes[task_id]
    ]
    dataset.authorize_formal_access(
        split='validation', phase='final_validation_pre_install',
        event_idx=final_event, task_id=final_task,
        timeline_step=f'event_{final_event}_CIL', classes=all_classes,
    )
    cached = cache_formal_batches(dataset.get_validation_loader(all_classes))
    cl_method.set_head_validation_provider(
        lambda classes: select_formal_cached_batches(cached, classes),
        lambda: copy.deepcopy(dataset.validation_manifest),
    )
    if cl_method._consolidate_head(final_task) is None:
        raise RuntimeError('formal final consolidation did not install state')
    return cached


def run_oracle(args):
    """True Oracle: retrain from scratch at every event with proper training budget.

    Tuned to actually reach the upper bound:
    - epochs scaled to at least 200 (was 100, often underfit on CIFAR)
    - SGD with initial LR 0.1 (was using args.lr=1e-3, severely undertrained)
    - Cosine annealing LR schedule
    """
    args.cl_method = 'oracle'
    args.ul_method = 'oracle'
    args.head_consolidation_enabled = 0
    args.head_consolidation_mode = 'full_classifier'
    print(f"\n{'='*70}\n  Oracle: Joint training\n{'='*70}")
    initialize_experiment_rng(args)
    dataset = VFLDataset(args)
    task_mgr = TaskManager(args)
    timeline = task_mgr.get_timeline()
    tracker = MetricsTracker()
    seen_task_classes = {}

    # Override LR for Oracle (the args.lr used by CL methods is too small for from-scratch joint training)
    oracle_lr = getattr(args, 'oracle_lr', 0.1)
    oracle_epochs = max(200, args.epochs_per_task * 5)

    for idx, event in enumerate(timeline):
        step = f"event_{idx}_{event['type']}"
        t0 = time.time()
        if event['type'] == 'CIL':
            task_mgr.advance_task(event['task_id'])
            seen_task_classes[event['task_id']] = event['new_classes']
        elif event['type'] == 'UL':
            task_mgr.apply_unlearn(event['forget_classes'])

        eff = task_mgr.get_effective_classes()
        if not eff: continue
        print(f"[{step}] Oracle retrain on {len(eff)} classes (epochs={oracle_epochs}, lr={oracle_lr})")

        bottoms, top = build_models(args)
        num_c = max(eff)+1
        if args.aggregation == 'sum': td = args.embed_dim
        else: td = args.embed_dim * args.num_parties
        top = TopModel(td, num_c, cosine=getattr(args, 'cosine_head', False)).to(args.device)
        trainer = VFLTrainer(bottoms, top, args)

        # Use a higher LR with cosine annealing for Oracle
        # We do this by temporarily overriding args.lr inside trainer
        original_lr = args.lr
        args.lr = oracle_lr
        loader, _ = dataset.get_task_loaders(eff)

        # Manual cosine annealing across epochs
        opts_b, opt_t = trainer._create_optimizers(lr=oracle_lr)
        scheds_b = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=oracle_epochs) for o in opts_b]
        sched_t = torch.optim.lr_scheduler.CosineAnnealingLR(opt_t, T_max=oracle_epochs)
        history = []
        for ep in range(oracle_epochs):
            loss, acc = trainer.train_epoch(loader, (opts_b, opt_t))
            for s in scheds_b: s.step()
            sched_t.step()
            history.append({'epoch': ep, 'loss': loss, 'acc': acc})
            if (ep + 1) % 50 == 0:
                print(f"    Oracle epoch {ep+1}/{oracle_epochs}: train_acc={acc:.4f}")
        args.lr = original_lr
        elapsed = time.time() - t0

        eff_tasks = {t:c for t,c in seen_task_classes.items()
                     if any(cc not in task_mgr.get_forgotten_classes() for cc in c)}
        per_task, per_task_deb, per_task_til = evaluate_per_task_full(trainer, dataset, eff_tasks, args.device)
        _, test_l = dataset.get_task_loaders(eff)
        oa, _, _ = trainer.evaluate(test_l)

        tracker.record_task_accuracies(step, per_task, round(oa,4),
                                       per_task_debiased=per_task_deb, per_task_taskil=per_task_til)
        tracker.record_timing(step, time.time()-t0)
        tracker.record_comm(step, trainer.get_comm_stats())
        if event['type'] == 'UL':
            ul_eval = evaluate_unlearning(trainer, dataset, task_mgr.get_forgotten_classes(), eff, args)
            tracker.record_ul_result(step, ul_eval)
        tracker.record_step({'event_idx':idx,'type':event['type'],'overall_acc':round(oa,4),
                             'per_task':per_task,'per_task_debiased':per_task_deb,'per_task_taskil':per_task_til})
        print(f"  -> overall_acc={oa:.4f}, per_task={per_task}")

    final = tracker.to_dict()
    final['config'] = {'cl_method':'oracle','ul_method':'oracle','data':args.data,
                       'seed':args.seed,'oracle_epochs':oracle_epochs,'oracle_lr':oracle_lr}
    os.makedirs(args.output_dir, exist_ok=True)
    _atomic_json_dump(final, os.path.join(args.output_dir, 'results.json'))
    _save_final_probs(trainer, dataset, task_mgr, args)
    cl = tracker.compute_cl_metrics()
    print(f"\n  Oracle - AA_cil:{cl['AA_cil']}, AA_final:{cl['AA_final']}, BWT:{cl['BWT']}")
    return final


def _fedprotip_readout_result(readouts, sample_counts=None):
    required = {
        'class_il_pred_task', 'class_il_global',
        'task_il_oracle', 'task_prediction', 'class_il_pred_task_counts',
    }
    if set(readouts) != required:
        raise ValueError(f'FedProTIP readouts must be {sorted(required)}')
    primary = readouts['class_il_pred_task']
    task_keys = set(primary) if type(primary) is dict else set()
    public = (
        primary, readouts['class_il_global'], readouts['task_il_oracle'],
        readouts['task_prediction'],
    )
    if (not task_keys
            or any(type(mapping) is not dict or set(mapping) != task_keys
                   for mapping in public)
            or any(type(key) is not str or not key.startswith('task_')
                   or not key[5:].isdigit() for key in task_keys)
            or any(type(value) is not float or not math.isfinite(value)
                   or not 0.0 <= value <= 1.0
                   for mapping in public for value in mapping.values())):
        raise ValueError('FedProTIP readout task keys/values are invalid')
    evidence = readouts['class_il_pred_task_counts']
    if type(evidence) is not dict or set(evidence) != task_keys:
        raise ValueError('FedProTIP exact evidence task keys are invalid')
    if sample_counts is not None and (
            type(sample_counts) is not dict
            or set(sample_counts) != task_keys
            or any(type(count) is not int or count <= 0
                   for count in sample_counts.values())):
        raise ValueError('FedProTIP sample counts are invalid')
    correct_total = 0
    sample_total = 0
    for key in task_keys:
        counts = evidence[key]
        if (type(counts) is not dict
                or set(counts) != {'correct', 'total'}
                or type(counts.get('correct')) is not int
                or type(counts.get('total')) is not int
                or counts['total'] <= 0
                or counts['correct'] < 0
                or counts['correct'] > counts['total']
                or primary[key] != round(
                    counts['correct'] / counts['total'], 4
                )
                or (sample_counts is not None
                    and counts['total'] != sample_counts[key])):
            raise ValueError('FedProTIP exact evidence is invalid')
        correct_total += counts['correct']
        sample_total += counts['total']
    overall = correct_total / sample_total
    return {
        'primary': primary,
        'debiased': None,
        'task_il': readouts['task_il_oracle'],
        'companions': {
            'class_il_global': readouts['class_il_global'],
            'task_prediction': readouts['task_prediction'],
        },
        'overall': round(float(overall), 4),
    }


def _evaluate_cil_readouts(cl_method, trainer, dataset, task_classes,
                           effective_classes, device):
    if hasattr(cl_method, 'evaluate_class_il_readouts'):
        result = _fedprotip_readout_result(
            cl_method.evaluate_class_il_readouts(dataset, task_classes)
        )
        result['overall'] = round(
            sum(result['primary'].values()) / len(result['primary']), 4
        )
        return result

    raw, debiased, task_il = evaluate_per_task_full(
        trainer, dataset, task_classes, device
    )
    if effective_classes:
        _, test_loader = dataset.get_task_loaders(
            effective_classes, shuffle_train=False
        )
        overall, _, _ = trainer.evaluate(test_loader)
    else:
        overall = 0.0
    return {
        'primary': raw,
        'debiased': debiased,
        'task_il': task_il,
        'companions': {},
        'overall': round(float(overall), 4),
    }


def _evaluate_cil_readouts_cached(
        cl_method, trainer, cached_test_batches, task_classes, device, *,
        collect_union_logits=False):
    effective_classes = [
        int(class_id)
        for classes in task_classes.values()
        for class_id in classes
    ]
    selected = select_formal_cached_batches(
        cached_test_batches, effective_classes
    )
    observed = [
        label
        for _batch_x, batch_y in selected
        for label in batch_y.tolist()
    ]
    if (any(type(label) is not int for label in observed)
            or set(observed) != set(effective_classes)):
        raise ValueError(
            'formal cache labels must exactly match the evaluated classes'
        )
    if hasattr(cl_method, 'evaluate_class_il_readouts_cached'):
        sample_counts = {
            f'task_{task_id}': sum(
                label in {int(class_id) for class_id in classes}
                for label in observed
            )
            for task_id, classes in task_classes.items()
        }
        return _fedprotip_readout_result(
            cl_method.evaluate_class_il_readouts_cached(
                cached_test_batches, task_classes
            ),
            sample_counts=sample_counts,
        )
    raw, debiased, task_il = evaluate_per_task_full_cached(
        trainer, cached_test_batches, task_classes, device
    )
    if collect_union_logits:
        union = _formal_union_logits(trainer, selected, [
            class_id for task_id in sorted(task_classes)
            for class_id in task_classes[task_id]
        ])
        overall = int(union['logits'].argmax(1).eq(union['labels']).sum()) / union['labels'].numel()
    else:
        overall, _, _ = trainer.evaluate(selected)
    result = {
        'primary': raw,
        'debiased': debiased,
        'task_il': task_il,
        'companions': {},
        'overall': round(float(overall), 4),
    }
    if collect_union_logits:
        result['formal_union_logits'] = union
    return result


def _is_adaptive_mode(cl_method, args):
    return bool(
        getattr(args, 'head_consolidation_enabled', 0)
        and getattr(args, 'head_consolidation_mode', '') == 'adaptive_dual_branch'
        and hasattr(cl_method, 'set_head_validation_provider')
    )


def _sanitize_and_finalize_adaptive(cl_method, trainer, forget_classes,
                                    adaptive_mode):
    """Finalize adaptive state only after sanitization returns successfully."""
    from cl_methods.sanitize import sanitize_cl_state
    report = sanitize_cl_state(cl_method, trainer, forget_classes)
    if adaptive_mode:
        forgotten = {int(class_id) for class_id in forget_classes}
        replay = getattr(cl_method, 'head_raw_replay', None)
        if isinstance(replay, dict):
            cl_method.head_raw_replay = {
                int(class_id): values for class_id, values in replay.items()
                if int(class_id) not in forgotten
            }
        task_classes = getattr(cl_method, 'head_task_classes', None)
        if isinstance(task_classes, dict):
            cl_method.head_task_classes = {
                int(task_id): kept
                for task_id, classes in task_classes.items()
                if (kept := [
                    int(class_id) for class_id in classes
                    if int(class_id) not in forgotten
                ])
            }
        cl_method.finalize_adaptive_head_after_sanitize()
    return report


def run_experiment(args):
    """Run one CL x UL combination."""
    print(f"\n{'='*70}\n  {args.cl_method} x {args.ul_method} on {args.data}\n{'='*70}")
    if (getattr(args, 'head_consolidation_enabled', 0)
            and getattr(args, 'head_consolidation_mode', '')
            == 'adaptive_dual_branch'
            and args.cl_method == 'proto_evolve_radapt'):
        raise ValueError(
            'adaptive_dual_branch forbids proto_evolve_radapt because its '
            'training-time redundancy probe accesses final-test data'
        )
    initialize_experiment_rng(args)

    adaptive_provenance = None
    formal_mode = getattr(args, 'formal_deferred_evaluation', False) is True
    if formal_mode:
        _checkpoint_protocol(args)
        if getattr(args, 'bic_enabled', 0):
            _formal_bic_cache_plan(args)
        task_mgr = TaskManager(args)
        timeline = task_mgr.get_timeline()
        if any(event.get('type') == 'UL' for event in timeline):
            raise ValueError('formal deferred evaluation forbids UL timelines')
        dataset = VFLDataset(args)
    else:
        dataset = VFLDataset(args)
        task_mgr = TaskManager(args)
        timeline = task_mgr.get_timeline()

    bottoms, top = build_models(args)
    trainer = VFLTrainer(bottoms, top, args)
    # radapt CL/UL need dataset access (for the per-task per-party probe).
    # Attaching here keeps the change minimal and is a no-op for other methods.
    trainer.dataset_ref = None if formal_mode else dataset
    cl_method = get_cl_method(args.cl_method, trainer, args)
    ul_method = get_ul_method(args.ul_method, trainer, args)
    tracker = MetricsTracker()
    bic_calibrator = TaskAffineCalibrator()
    adaptive_mode = not formal_mode and _is_adaptive_mode(cl_method, args)
    train_only_mode = formal_mode or adaptive_mode
    train_only_dataset = (
        _FormalTrainOnlyDataset(dataset)
        if formal_mode and args.cl_method in {'gpm', 'er', 'fedprotip_vfl'}
        else None
    )
    if adaptive_mode:
        if adaptive_provenance is None:
            adaptive_provenance = _prepare_adaptive_provenance(
                args, trainer, cl_method, task_mgr,
                tracker, bic_calibrator,
            )
        cl_method.set_head_validation_provider(
            lambda classes: dataset.get_validation_loader(classes),
            lambda: copy.deepcopy(dataset.validation_manifest),
        )
    seen_task_classes = {}
    bic_history = []
    start_event_idx, seen_task_classes, bic_history = _load_resume_checkpoint(
        args, trainer, cl_method, task_mgr, tracker, bic_calibrator
    )

    for idx, event in enumerate(timeline):
        if idx < start_event_idx:
            continue
        step = f"event_{idx}_{event['type']}"
        trainer.reset_comm_stats()
        t0 = time.time()

        if event['type'] == 'CIL':
            tid = event['task_id']
            new_cls = event['new_classes']
            if adaptive_mode and hasattr(
                    cl_method, 'set_adaptive_event_boundary'):
                cl_method.set_adaptive_event_boundary(idx, 'CIL')
            task_mgr.advance_task(tid)
            seen_task_classes[tid] = new_cls
            eff = task_mgr.get_effective_classes()
            if train_only_mode:
                task_loader = dataset.get_train_loader(new_cls)
            else:
                task_loader, _ = dataset.get_task_loaders(new_cls)

            print(f"[{step}] CIL: Task {tid}, classes {new_cls}")

            _call_method_hook(
                cl_method.before_task, args, trainer, train_only_dataset,
                tid, new_cls, eff,
            )

            configure_task_ce(
                trainer, getattr(args, 'task_ce_mode', 'method'), new_cls, eff
            )

            # V-LETO needs pre-training prototypes
            if hasattr(cl_method, 'before_train_compute_pre_protos'):
                _call_method_hook(
                    cl_method.before_train_compute_pre_protos,
                    args, trainer, train_only_dataset, task_loader,
                )

            # Choose training data
            if args.replay_mode == 'full':
                if train_only_mode:
                    train_loader = dataset.get_train_loader(eff)
                else:
                    train_loader, _ = dataset.get_task_loaders(eff)
            else:
                train_loader = task_loader

            history, elapsed = _call_method_hook(
                cl_method.train_task, args, trainer, train_only_dataset,
                train_loader, tid,
            )
            _call_after_task(
                cl_method, args, trainer, train_only_dataset, task_loader, tid
            )

            if train_only_mode:
                tracker.record_timing(step, time.time()-t0)
                tracker.record_comm(step, trainer.get_comm_stats())
                boundary = {
                    'event_idx': idx,
                    'type': 'CIL',
                    'task_id': tid,
                    'evaluation_deferred': True,
                    'train_time': round(elapsed, 2),
                    'comm': trainer.get_comm_stats(),
                }
                if formal_mode:
                    boundary['new_classes'] = list(new_cls)
                tracker.record_step(boundary)
                _save_cil_checkpoint(
                    trainer, cl_method, args, step, tid, new_cls, seen_task_classes,
                    None, tracker.to_dict(), bic_history,
                    force=True,
                    forgotten_classes=task_mgr.get_forgotten_classes(),
                )
                save_deferred_cil_snapshot(
                    trainer, cl_method, args, idx, tid, seen_task_classes,
                    protocol_kind='formal' if formal_mode else 'adaptive',
                )
                _save_party_kd_audit(cl_method, args, step)
                print(f"  -> evaluation deferred, Time:{elapsed:.1f}s")
                continue

            # Evaluate
            eff_tasks = {t:c for t,c in seen_task_classes.items()
                         if any(cc not in task_mgr.get_forgotten_classes() for cc in c)}
            eff = task_mgr.get_effective_classes()
            readouts = _evaluate_cil_readouts(
                cl_method, trainer, dataset, eff_tasks, eff, args.device
            )
            per_task = readouts['primary']
            per_task_deb = readouts['debiased']
            per_task_til = readouts['task_il']
            companion_readouts = readouts['companions']
            oa = readouts['overall']

            if _should_fit_bic(args, tid):
                if getattr(args, 'bic_fit_mode', 'sequential') in (
                    'joint_final', 'joint_each_stage'
                ):
                    bic_history.append(_fit_and_evaluate_final_bic(
                        trainer, dataset, bic_calibrator, args, step, eff_tasks,
                    ))
                else:
                    bic_history.append(_fit_and_evaluate_bic(
                        trainer, dataset, bic_calibrator, args, step, tid, new_cls,
                        eff_tasks,
                    ))

            tracker.record_task_accuracies(step, per_task, round(oa,4),
                                           per_task_debiased=per_task_deb,
                                           per_task_taskil=per_task_til,
                                           companion_readouts=companion_readouts)
            tracker.record_timing(step, time.time()-t0)
            tracker.record_comm(step, trainer.get_comm_stats())
            tracker.record_step({'event_idx':idx,'type':'CIL','task_id':tid,
                                 'per_task_acc':per_task,'per_task_acc_debiased':per_task_deb,
                                 'per_task_acc_taskil':per_task_til,'overall_acc':round(oa,4),
                                 'companion_readouts':companion_readouts,
                                 'train_time':round(elapsed,2),'comm':trainer.get_comm_stats()})
            _save_cil_checkpoint(
                trainer, cl_method, args, step, tid, new_cls, seen_task_classes,
                bic_calibrator.state_dict() if getattr(args, 'bic_enabled', 0) else None,
                tracker.to_dict(), bic_history,
                forgotten_classes=task_mgr.get_forgotten_classes(),
            )
            _save_party_kd_audit(cl_method, args, step)
            print(f"  -> primary={oa:.4f}, {per_task}, Time:{elapsed:.1f}s")
            print(f"     debiased={per_task_deb}")
            print(f"     task_il ={per_task_til}")

            if os.environ.get('DUMP_HEAD_NORMS'):
                norms = trainer.top_model.classifier.weight.data.norm(dim=1)
                print(f"  [head ||W_c|| after {step}]")
                for t in sorted(seen_task_classes):
                    vals = ", ".join(f"c{c}:{norms[c].item():.2f}" for c in seen_task_classes[t])
                    print(f"     task{t}: {vals}")

        elif event['type'] == 'UL':
            fc = event['forget_classes']
            if adaptive_mode and hasattr(
                    cl_method, 'set_adaptive_event_boundary'):
                cl_method.set_adaptive_event_boundary(idx, 'UL')
            print(f"[{step}] UL: Forget {fc}")
            # PROTOCOL: UL operators train over the FULL retained head.
            # CL methods scope trainer CE via ce_lo/ce_hi or ce_classes for
            # the task they just trained; a UL method that reuses trainer.train_task
            # (retrain/luv/...) with that stale slice feeds negative relabeled
            # targets -> device-side gather fault (DCU VMFault, process hangs).
            trainer.ce_lo, trainer.ce_hi = 0, None
            trainer.ce_classes = None
            all_seen = task_mgr.get_all_seen_classes()
            eff = [c for c in all_seen if c not in fc and c not in task_mgr.get_forgotten_classes()]
            if adaptive_mode:
                loaders = {
                    'forget_train': dataset.get_train_loader(sorted(int(c) for c in fc)),
                    'retain_train': dataset.get_train_loader(sorted(int(c) for c in eff)),
                }
            else:
                loaders = dataset.get_forget_retain_loaders(fc, all_seen)
            # FIX (cumulative retain): get_forget_retain_loaders excludes only the
            # CURRENT event's forget class, so retain_train still contains classes
            # forgotten in EARLIER events -> fine-tuning UL methods (retrain/luv/
            # mode/fudp/fucrt/fedup/fedau/GA) re-learn them, e.g. class 0 reappears
            # at the 2nd event and inflates forget_acc to ~0.45. Rebuild the retain
            # loaders over eff (= all-seen minus ALL forgotten) so the whole UL
            # benchmark uses an honest cumulative retain set and is comparable.
            if eff and not adaptive_mode:
                _rt, _re = dataset.get_task_loaders(sorted(int(c) for c in eff))
                loaders['retain_train'] = _rt
                loaders['retain_test'] = _re
            # snapshot the PRE-unlearning head rows for the reconnection threat
            _W_snap = trainer.top_model.classifier.weight.detach().clone()
            # fair, method-agnostic communication proxy: per-party encoder snapshot.
            # Some baselines (fedosd/luv/GA) train all-party bottoms via their own
            # loops and bypass the byte counter (reporting 0 comm); count how many
            # parties' encoders were ACTUALLY modified -> the honest |parties touched|
            # axis (roar touches |S*|, fedosd/retrain touch all P).
            _pre_b = [torch.cat([p.detach().reshape(-1) for p in b.parameters()])
                      for b in trainer.bottoms]
            adaptive_dataset_ref = trainer.dataset_ref
            if adaptive_mode:
                trainer.dataset_ref = None
            try:
                ul_res = ul_method.unlearn(
                    fc, loaders['retain_train'], loaders['forget_train'],
                    effective_classes=eff,
                )
            finally:
                if adaptive_mode:
                    trainer.dataset_ref = adaptive_dataset_ref
            _post_b = [torch.cat([p.detach().reshape(-1) for p in b.parameters()])
                       for b in trainer.bottoms]
            _touched = sum(1 for a, c in zip(_pre_b, _post_b)
                           if a.numel() != c.numel() or not torch.allclose(a, c, atol=1e-6))
            task_mgr.apply_unlearn(fc)

            # PROTOCOL: sanitize the CL method's cached state (prototype stores,
            # frozen teachers, EWC anchors, replay bans, buffers). Without this,
            # the next CIL event distills/replays the forgotten class back in
            # ("relapse"). Runs AFTER the UL operator so re-snapshotted teachers
            # capture the post-unlearning weights. --sanitize_cl_state 0 ablates.
            if getattr(args, 'sanitize_cl_state', 1):
                _san = _sanitize_and_finalize_adaptive(
                    cl_method, trainer, fc, adaptive_mode
                )
                if _san:
                    print(f"  [sanitize] {', '.join(_san)}")

            if adaptive_mode:
                tracker.record_timing(step, time.time()-t0)
                tracker.record_comm(step, trainer.get_comm_stats())
                tracker.record_step({
                    'event_idx': idx,
                    'type': 'UL',
                    'forget_classes': fc,
                    'evaluation_deferred': True,
                    'comm': trainer.get_comm_stats(),
                })
                latest_task_id = max(seen_task_classes)
                _save_cil_checkpoint(
                    trainer, cl_method, args, step, latest_task_id,
                    seen_task_classes[latest_task_id], seen_task_classes,
                    None, tracker.to_dict(), bic_history,
                    force=True,
                    forgotten_classes=task_mgr.get_forgotten_classes(),
                )
                _save_party_kd_audit(cl_method, args, step)
                print(f"  -> evaluation deferred, UL:{ul_res}")
                continue

            curr_forgotten = task_mgr.get_forgotten_classes()
            curr_eff = task_mgr.get_effective_classes()
            ul_eval = evaluate_unlearning(trainer, dataset, curr_forgotten, curr_eff, args)
            ul_eval['parties_touched'] = _touched
            ul_eval['parties_total'] = len(trainer.bottoms)

            # Universal re-learned-head attack (apples-to-apples across ALL ul_methods):
            # fit the best linear forget-vs-rest probe on the post-unlearning party
            # embeddings -> ROC-AUC of how recoverable class f still is. Head-only
            # baselines get forget_acc=0 with ~0 communication but leave the evidence
            # in the bottoms (AUC high); methods that pay to scrub the bottoms (roar,
            # retrain) drive AUC to the retrain floor. This is the honest comparison
            # axis: communication paid vs evidence actually removed.
            try:
                from ul_methods.roar import RoarUL
                _atk = RoarUL(trainer, args)
                _ac = sorted(set(int(c) for c in curr_eff) | set(int(c) for c in fc))
                _al = dataset.get_task_loaders(_ac, shuffle_train=False)[1]
                ul_eval['relearn_auc'] = {int(f): _atk._relearn_attack([int(f)], _al)['auc']
                                          for f in fc}
                # ROAR's defended threat: reconnect the FROZEN pre-unlearning head row
                ul_eval['reconnect_auc'] = {int(f): _atk._reconnect_attack(_W_snap[int(f)], int(f), _al)
                                            for f in fc}
                print(f"  [attack] re-learned-head AUC {{ {', '.join(f'{k}:{round(v,3)}' for k,v in ul_eval['relearn_auc'].items())} }} | "
                      f"reconnect AUC {{ {', '.join(f'{k}:{round(v,3)}' for k,v in ul_eval['reconnect_auc'].items())} }}")
            except Exception as e:
                print(f"  [attack] warn: re-learned-head attack failed ({e})")

            eff_tasks = {t:[cc for cc in c if cc not in curr_forgotten]
                         for t,c in seen_task_classes.items()}
            eff_tasks = {t:c for t,c in eff_tasks.items() if c}
            per_task, per_task_deb, per_task_til = evaluate_per_task_full(trainer, dataset, eff_tasks, args.device)
            if curr_eff:
                _, tl = dataset.get_task_loaders(curr_eff)
                oa, _, _ = trainer.evaluate(tl)
            else: oa = 0.0

            tracker.record_task_accuracies(step, per_task, round(oa,4),
                                           per_task_debiased=per_task_deb, per_task_taskil=per_task_til)
            tracker.record_ul_result(step, ul_eval)
            tracker.record_timing(step, time.time()-t0)
            tracker.record_comm(step, trainer.get_comm_stats())
            tracker.record_step({'event_idx':idx,'type':'UL','forget_classes':fc,
                                 'ul_eval':ul_eval,'per_task_acc':per_task,'per_task_acc_debiased':per_task_deb,
                                 'per_task_acc_taskil':per_task_til,'overall_acc':round(oa,4)})
            print(f"  -> UL:{ul_eval}, Acc:{oa:.4f}, {per_task}")

    # END-OF-STREAM unlearning audit (the relapse axis): per-event ul_eval above
    # measures forgetting right after each UL event, but subsequent CL training
    # (distillation/replay/anchors) can resurrect the forgotten class. UA in the
    # benchmark tables is THIS number — forget-class accuracy at stream end —
    # plus the re-learned-head AUC on the final embeddings.
    final_ul = None
    forgotten_final = sorted(int(c) for c in task_mgr.get_forgotten_classes())
    if forgotten_final and not adaptive_mode:
        eff_final = sorted(int(c) for c in task_mgr.get_effective_classes())
        final_ul = evaluate_unlearning(trainer, dataset, forgotten_final, eff_final, args)
        per_fc = {}
        for f in forgotten_final:
            _, fl = dataset.get_task_loaders([f], shuffle_train=False)
            a, _, _ = trainer.evaluate(fl)
            per_fc[f] = round(a, 4)
        final_ul['forget_acc_per_class_final'] = per_fc
        try:
            from ul_methods.roar import RoarUL
            _atk = RoarUL(trainer, args)
            _al = dataset.get_task_loaders(sorted(set(eff_final) | set(forgotten_final)),
                                           shuffle_train=False)[1]
            final_ul['relearn_auc_final'] = {f: _atk._relearn_attack([f], _al)['auc']
                                             for f in forgotten_final}
        except Exception as e:
            print(f"  [final-audit] warn: re-learned-head attack failed ({e})")
        print(f"[final-audit] UA(end)={final_ul['forget_acc']} per-class={per_fc} "
              f"retain={final_ul['retain_acc']} auc={final_ul.get('relearn_auc_final')}")

    formal_final_checkpoint = None
    formal_already_published = False
    formal_calibration_audit = None
    formal_validation_cache = None
    formal_validation_manifest = None
    formal_selection_audit = None
    formal_bic_fit_corpus = None
    if formal_mode:
        ordered_tasks = sorted(seen_task_classes)
        if ordered_tasks != list(range(args.num_tasks)):
            raise RuntimeError('formal training did not produce every task stage')
        final_task = ordered_tasks[-1]
        final_event = next(
            event_idx for event_idx in range(len(timeline) - 1, -1, -1)
            if timeline[event_idx]['type'] == 'CIL'
        )
        source_checkpoint = os.path.join(
            args.output_dir, 'checkpoints', f'event_{final_event}_CIL.pt'
        )
        formal_final_checkpoint = os.path.join(
            args.output_dir, 'checkpoints', 'formal_final.pt'
        )
        if not os.path.lexists(formal_final_checkpoint):
            formal_validation_cache = _finalize_formal_internal_state(
                args=args, dataset=dataset, cl_method=cl_method,
                task_classes=seen_task_classes,
                final_event=final_event, final_task=final_task,
            )
            if formal_validation_cache is not None:
                formal_validation_manifest = copy.deepcopy(
                    dataset.validation_manifest
                )
                if getattr(args, 'bic_enabled', 0):
                    formal_calibration_audit = copy.deepcopy(
                        dataset.calibration_audit()
                    )
                    formal_selection_audit = copy.deepcopy(
                        dataset.selection_audit()
                    )
            final_payload = _safe_torch_load(source_checkpoint)
            if formal_validation_cache is not None:
                final_payload['trainer_state'] = _encode_checkpoint_value(
                    trainer.get_state()
                )
                final_payload['cl_state'] = _encode_checkpoint_value(
                    cl_method.get_state()
                    if hasattr(cl_method, 'get_state') else {}
                )
                final_payload['rng_state'] = _encode_checkpoint_value(
                    _capture_rng_state()
                )
            _atomic_torch_save(final_payload, formal_final_checkpoint)
        snapshot_paths = [
            os.path.join(
                args.output_dir, 'formal_snapshots', f'event_{event_idx}_CIL.pt'
            )
            for event_idx, event in enumerate(timeline)
            if event['type'] == 'CIL'
        ]
        live_trainer = trainer_state_sha256(trainer.get_state())
        live_method = copy.deepcopy(
            cl_method.get_state() if hasattr(cl_method, 'get_state') else {}
        )
        live_tracker = copy.deepcopy(tracker.to_dict())
        live_rng = _capture_rng_state()
        try:
            deferred, formal_status = _evaluate_formal_from_single_access(
                args=args, dataset=dataset, snapshot_paths=snapshot_paths,
                final_checkpoint=formal_final_checkpoint,
                task_classes=seen_task_classes, final_event=final_event,
                final_task=final_task,
                cached_calibration_batches=formal_validation_cache,
                calibration_audit=formal_calibration_audit,
                validation_manifest=formal_validation_manifest,
                selection_audit=formal_selection_audit,
            )
            formal_already_published = formal_status == 'published'
        finally:
            _restore_rng_state(live_rng)
        if (trainer_state_sha256(trainer.get_state()) != live_trainer
                or (hasattr(cl_method, 'get_state')
                    and not _checkpoint_values_equal(
                        cl_method.get_state(), live_method))
                or not _checkpoint_values_equal(tracker.to_dict(), live_tracker)
                or not _checkpoint_values_equal(_capture_rng_state(), live_rng)):
            raise RuntimeError('formal evaluation mutated live final state')
        if bool(getattr(args, 'bic_enabled', 0)):
            tracker.load_dict(deferred['tracker_state'])
            bic_history = copy.deepcopy(deferred['bic_history'])
            bic_calibrator.load_state_dict(copy.deepcopy(deferred['bic_state']))
            formal_calibration_audit = copy.deepcopy(
                deferred['calibration_audit']
            )
            formal_bic_fit_corpus = copy.deepcopy(
                deferred['bic_fit_corpus']
            )
        else:
            tracker.load_dict(deferred)

    if adaptive_mode and hasattr(cl_method, 'head_consolidation_history'):
        if not cl_method.head_consolidation_history:
            raise RuntimeError('adaptive state was not installed before freeze')
        final_checkpoint = save_final_adaptive_checkpoint(
            trainer, cl_method, args, adaptive_provenance
        )
        audit_spec = {
            **adaptive_provenance,
            'checkpoint': final_checkpoint.name,
        }
        evidence = audit_adaptive_checkpoint(args.output_dir, audit_spec)
        freeze_path = os.path.join(
            args.output_dir, 'ADAPTIVE_STATE_FROZEN.json'
        )
        if os.path.exists(freeze_path):
            if load_frozen_adaptive_evidence(args.output_dir) != evidence:
                raise RuntimeError('existing adaptive freeze evidence mismatches audit')
        else:
            atomic_write_new_json(freeze_path, evidence)
        stream_state = tracker.to_dict()
        live_model_sha256 = trainer_state_sha256(trainer.get_state())
        snapshot_dir = os.path.join(args.output_dir, 'adaptive_snapshots')
        snapshot_paths = [
            os.path.join(snapshot_dir, f'event_{event_idx}_CIL.pt')
            for event_idx, event in enumerate(timeline)
            if event['type'] == 'CIL'
        ]
        deferred = evaluate_deferred_cil_trajectory(
            snapshot_paths, final_checkpoint, dataset,
            seen_task_classes, args,
        )
        if trainer_state_sha256(trainer.get_state()) != live_model_sha256:
            raise RuntimeError('deferred evaluation mutated the live final model')
        deferred['comm_stats'] = stream_state['comm_stats']
        deferred['timing'] = stream_state['timing']
        deferred['step_results'] = stream_state['step_results']
        tracker.load_dict(deferred)

    final = tracker.to_dict()
    if getattr(args, 'bic_enabled', 0):
        final['bic_history'] = bic_history
        final['bic_final'] = bic_history[-1] if bic_history else None
        final['calibration_audit'] = (
            formal_calibration_audit if formal_mode
            else dataset.calibration_audit()
        )
        if formal_mode:
            final['bic_fit_corpus'] = formal_bic_fit_corpus
    if getattr(args, 'lambda_validation_enabled', 0):
        final['selection_audit'] = dataset.selection_audit()
    if final_ul is not None:
        final['final_ul_eval'] = final_ul
    final['config'] = {'cl_method':args.cl_method,'ul_method':args.ul_method,
                        'data':args.data,'num_tasks':args.num_tasks,'seed':args.seed}
    if formal_mode:
        final['source_provenance'] = copy.deepcopy(
            args.formal_source_provenance
        )
    os.makedirs(args.output_dir, exist_ok=True)
    if formal_mode:
        if not formal_already_published:
            _publish_formal_deferred_result(
                args=args, final_checkpoint=formal_final_checkpoint,
                tracker_state=tracker.to_dict(), final_result=final,
                bic_state=(bic_calibrator.state_dict()
                           if getattr(args, 'bic_enabled', 0) else None),
                bic_history=(bic_history
                             if getattr(args, 'bic_enabled', 0) else None),
            )
    else:
        _atomic_json_dump(final, os.path.join(args.output_dir, 'results.json'))

    if not adaptive_mode and not formal_mode:
        _save_final_probs(trainer, dataset, task_mgr, args)
    cl = tracker.compute_cl_metrics()
    print(f"\n  {args.cl_method} x {args.ul_method} - AA:{cl.get('AA', cl['AA_cil'])}, BWT:{cl['BWT']}")
    if tracker.ul_metrics:
        last_ul = tracker.ul_metrics[-1]
        print(f"  UL: F-Acc:{last_ul.get('forget_acc')}, MIA:{last_ul.get('mia_score')}")
    return final
