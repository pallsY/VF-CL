#!/usr/bin/env python3
"""Freeze and execute the one-shot adaptive TinyImageNet held-out protocol."""
import argparse
import contextlib
import hashlib
import json
import os
import re
import runpy
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import adaptive_dual_branch_validation as development
from adaptive_consolidation_audit import evaluate_deferred_cil_trajectory
from data_utils import VFLDataset
from prepare_tinyimagenet import build_heldout_manifest, frozen_protocol


REPO = development.REPO
WORKTREE = development.WORKTREE
PYTHON = development.PYTHON
DATA_ROOT = REPO / 'data' / 'tiny-imagenet-200'
BASE_COMMIT = 'cf08c992dd000139454c091c55d479880e2581eb'
PRIMARY_SEED = 42
FORMAL_SEEDS = (43, 44)
PRIMARY_VARIANTS = ('full', 'bias', 'adaptive')
ABLATIONS = ('no_consolidation', 'fixed_half', 'sample_mean_nll')
METRICS = ('AA_final', 'BWT', 'AA_final_taskil')
TOLERANCE = Decimal('0.01')
STRICT_EPSILON = Decimal('1e-12')
PROTOCOL = {
    **frozen_protocol(), 'solver_tolerance': 1e-12,
    'solver_max_iterations': 80, 'gate_rule': 'class_balanced',
}
SOURCE_FILES = (
    'config.py', 'prepare_tinyimagenet.py',
    'adaptive_tinyimagenet_heldout.py',
    'run_adaptive_tinyimagenet_heldout.sh',
    'test_adaptive_protocol_config.py', 'test_prepare_tinyimagenet.py',
    'test_adaptive_tinyimagenet_heldout.py',
)
EVIDENCE_NAMES = frozenset({
    'HELDOUT_PRIMARY_PLAN.json', 'FOLLOWUP_AUTHORIZATION.json',
    'planned_protocol.json', 'launch_started.json', 'job.log',
    'record.json', 'SUCCESS', 'FAILED.json',
    'HELDOUT_PRIMARY_GATE.json', 'EXECUTION_SUCCESS',
    'HELDOUT_GATE_SUCCESS', 'HELDOUT_GATE_FAILED',
})
ROOT_PATTERN = re.compile(
    r'adaptive_tinyimagenet_heldout_seed42_[0-9]{8}_[0-9]{6}'
)


def _canonical_bytes(payload):
    return json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()


def payload_sha256(payload):
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def freeze_identity(path):
    path = _reject_symlink_components(path)
    before = path.stat()
    digest = file_sha256(path)
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_ctime_ns', 'st_size')
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise ValueError('held-out freeze changed while reading its identity')
    return {
        'device': before.st_dev, 'inode': before.st_ino,
        'ctime_ns': before.st_ctime_ns, 'size': before.st_size,
        'content_sha256': digest,
    }


def _validate_freeze_precedes(freeze_path, artifacts):
    frozen = _reject_symlink_components(freeze_path).stat()
    for artifact in artifacts:
        details = _reject_symlink_components(artifact).stat()
        if (frozen.st_mtime_ns > details.st_mtime_ns
                or frozen.st_ctime_ns > details.st_ctime_ns):
            raise ValueError('held-out freeze must predate all later evidence')


def _has_protocol_evidence(root):
    root = Path(root)
    if not root.exists():
        return False
    for path in root.rglob('*'):
        relative = path.relative_to(root)
        if (path.name in EVIDENCE_NAMES
                or ('outputs' in relative.parts
                    and relative.parts[-1] != 'outputs')):
            return True
    return False


def _read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _reject_symlink_components(path):
    path = Path(path).absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f'symlinked held-out path is forbidden: {current}')
    return path


def _write_exclusive(path, payload):
    path = _reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(_canonical_bytes(payload) + b'\n')
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _git(*arguments):
    return subprocess.check_output(
        ['git', '-C', str(WORKTREE), *arguments], text=True,
    ).strip()


def source_identity():
    commit = _git('rev-parse', 'HEAD')
    subprocess.run(
        ['git', '-C', str(WORKTREE), 'merge-base', '--is-ancestor',
         BASE_COMMIT, commit], check=True, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    clean = not _git('status', '--porcelain')
    if not clean:
        raise ValueError('held-out freeze requires a clean worktree')
    hashes = {name: file_sha256(WORKTREE / name) for name in SOURCE_FILES}
    return {'commit': commit, 'clean': True, 'source_sha256': hashes}


def validate_development_gate(root):
    root = _reject_symlink_components(root).resolve()
    status, report = development._terminal_evidence(root)
    if status != 'GATE_SUCCESS' or report.get('status') != 'GATE_SUCCESS':
        raise ValueError('held-out freeze requires a real Task 8 GATE_SUCCESS')
    return {
        'status': status,
        'root': str(root),
        'marker_sha256': file_sha256(root / 'GATE_SUCCESS'),
        'report_sha256': file_sha256(root / 'PRIMARY_GATE.json'),
        'plan_sha256': file_sha256(root / 'PRIMARY_PLAN.json'),
    }


def freeze_payload(dataset_manifest, gate_evidence, source):
    required_hashes = (
        'archive_sha256', 'paths_sha256', 'class_order_sha256',
        'task_manifest_sha256', 'transforms_sha256', 'split_sha256',
        'manifest_sha256',
    )
    if not all(re.fullmatch(r'[0-9a-f]{64}', str(dataset_manifest.get(key, '')))
               for key in required_hashes):
        raise ValueError('held-out dataset manifest hashes are incomplete')
    if (len(dataset_manifest.get('class_order', [])) != 200
            or len(dataset_manifest.get('task_manifest', [])) != 10
            or any(len(task) != 20 for task in dataset_manifest['task_manifest'])
            or dataset_manifest.get('protocol') != frozen_protocol()):
        raise ValueError('held-out dataset manifest differs from frozen protocol')
    if gate_evidence.get('status') != 'GATE_SUCCESS':
        raise ValueError('development gate is not successful')
    if source.get('clean') is not True or not re.fullmatch(
            r'[0-9a-f]{40}', str(source.get('commit', ''))):
        raise ValueError('source identity is not a clean commit')
    return {
        'schema_version': 1, 'status': 'HELDOUT_PROTOCOL_FROZEN',
        'source_commit': source['commit'],
        'source_sha256': source['source_sha256'],
        'clean_worktree': True, 'protocol': PROTOCOL,
        'dataset_manifest': dataset_manifest,
        'development_gate': gate_evidence,
        'primary_seed': PRIMARY_SEED,
        'primary_variants': list(PRIMARY_VARIANTS),
        'formal_seeds_if_success': list(FORMAL_SEEDS),
        'ablations_if_success': list(ABLATIONS),
        'test_used_for_fit': False, 'test_used_for_selection': False,
    }


def _validate_dataset_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError('held-out dataset manifest is malformed')
    payload = dict(manifest)
    stored = payload.pop('manifest_sha256', None)
    if stored != payload_sha256(payload):
        raise ValueError('held-out dataset manifest hash changed')
    protocol = manifest.get('protocol')
    source_hashes = manifest.get('source_file_sha256')
    selected = manifest.get('training_validation_paths')
    class_order = manifest.get('class_order')
    tasks = manifest.get('task_manifest')
    training = manifest.get('training_paths')
    official = manifest.get('official_val_test_paths')
    if not (protocol == frozen_protocol()
            and isinstance(source_hashes, dict) and len(source_hashes) == 110000
            and all(re.fullmatch(r'[0-9a-f]{64}', str(value))
                    for value in source_hashes.values())
            and re.fullmatch(r'[0-9a-f]{64}', str(manifest.get('archive_sha256', '')))
            and isinstance(class_order, list) and len(class_order) == 200
            and len(set(class_order)) == 200
            and isinstance(tasks, list) and len(tasks) == 10
            and all(len(task) == 20 for task in tasks)
            and [item for task in tasks for item in task] == class_order
            and isinstance(training, list) and len(training) == 90000
            and isinstance(selected, list) and len(selected) == 10000
            and isinstance(official, list) and len(official) == 10000
            and not (set(training) & set(selected))
            and manifest.get('paths_sha256')
            == payload_sha256(sorted(source_hashes))
            and manifest.get('class_order_sha256')
            == payload_sha256(manifest.get('class_order'))
            and manifest.get('task_manifest_sha256')
            == payload_sha256(manifest.get('task_manifest'))
            and manifest.get('transforms_sha256') == payload_sha256({
                'train': protocol['train_transform'],
                'validation_test': protocol['validation_test_transform'],
            })
            and manifest.get('split_sha256') == hashlib.sha256(
                json.dumps(sorted(selected), separators=(',', ':')).encode()
            ).hexdigest()):
        raise ValueError('held-out dataset manifest internals changed')
    return manifest


def install_freeze(root, payload):
    root = Path(root)
    path = root / 'HELDOUT_PROTOCOL_FROZEN.json'
    if path.exists() or path.is_symlink():
        if _read_json(path) != payload:
            raise ValueError('held-out protocol freeze is immutable')
        return path
    if _has_protocol_evidence(root):
        raise ValueError('held-out freeze cannot replace existing evidence')
    return _write_exclusive(path, payload)


def freeze(root, dataset_root, development_root):
    dataset_manifest = build_heldout_manifest(dataset_root)
    _validate_dataset_manifest(dataset_manifest)
    payload = freeze_payload(
        dataset_manifest,
        validate_development_gate(development_root), source_identity(),
    )
    return install_freeze(root, payload)


def _set(command, option, value):
    value = str(value)
    if option in command:
        command[command.index(option) + 1] = value
    else:
        command.extend((option, value))


def fixed_branch_gate(variant):
    if variant not in ('full', 'bias', 'no_consolidation'):
        raise ValueError('unknown fixed held-out branch')
    return {'full': 1.0, 'bias': 0.0, 'no_consolidation': 0.5}[variant]


@contextlib.contextmanager
def _fixed_branch_runtime(variant):
    import adaptive_consolidation_audit as audit_module
    import adaptive_head_consolidation as head
    import cl_methods.proto_evolve as proto

    gate = fixed_branch_gate(variant)

    gate_rule = ('no_consolidation' if variant == 'no_consolidation'
                 else f'fixed_{variant}')

    def solve(log_p_full, log_p_bias, labels, classes):
        target = head._validated_problem(
            log_p_full, log_p_bias, labels, classes
        )
        rows = head.torch.arange(target.numel(), device=log_p_full.device)
        selected_full = log_p_full[rows, target]
        selected_bias = log_p_bias[rows, target]
        weights = head._weights(target, True)
        derivatives = (
            head._derivative(selected_full, selected_bias, weights, 0.0),
            head._derivative(selected_full, selected_bias, weights, 1.0),
        )
        return head._record(
            gate_rule, False, gate, derivatives, (gate, gate), 0,
            True, selected_full, selected_bias, weights,
        )

    targets = (head, proto, audit_module)
    originals = [module.solve_global_mixture_weight for module in targets]
    original_installers = [module.install_and_reload_verify for module in targets]
    original_validator = head._validate_primary_gate
    original_fit = proto.fit_adaptive_candidates
    original_configs = [
        (module.FULL_BRANCH_CONFIG, module.BIAS_BRANCH_CONFIG)
        for module in targets
    ]

    def fit(pre_top, *args, **kwargs):
        if variant != 'no_consolidation':
            return head.fit_fixed_endpoint_candidate(
                pre_top, variant, *args, **kwargs
            )
        state = head.freeze_state(pre_top.state_dict())
        state_hash = head.hash_top_state(state)
        return head.FrozenAdaptiveCandidates(
            pre_head_sha256=state_hash,
            full_state=state, bias_state=state,
            full_head_sha256=state_hash, bias_head_sha256=state_hash,
            full_audit={'mode': 'no_consolidation', 'skipped': True},
            bias_audit={'mode': 'no_consolidation', 'skipped': True},
            ordered_classes=tuple(sorted(args[0])),
        )

    def install(pre_top, candidates, gate_record):
        if variant != 'no_consolidation':
            return head.install_fixed_endpoint_and_reload_verify(
                pre_top, candidates, gate_record
            )
        primary = dict(gate_record, gate_rule='class_balanced', is_primary=True)
        return original_installers[0](pre_top, candidates, primary)

    def validate(gate_record):
        if (gate_record.get('gate_rule') == gate_rule
                and gate_record.get('is_primary') is False
                and gate_record.get('g') == gate):
            return original_validator(dict(
                gate_record, gate_rule='class_balanced', is_primary=True,
            ))
        return original_validator(gate_record)

    for module in targets:
        module.solve_global_mixture_weight = solve
        module.install_and_reload_verify = install
        if variant == 'full':
            module.BIAS_BRANCH_CONFIG = head.INACTIVE_BRANCH_CONFIG
        elif variant == 'bias':
            module.FULL_BRANCH_CONFIG = head.INACTIVE_BRANCH_CONFIG
    head._validate_primary_gate = validate
    proto.fit_adaptive_candidates = fit
    try:
        yield
    finally:
        for module, original, original_install, configs in zip(
                targets, originals, original_installers, original_configs):
            module.solve_global_mixture_weight = original
            module.install_and_reload_verify = original_install
            module.FULL_BRANCH_CONFIG, module.BIAS_BRANCH_CONFIG = configs
        head._validate_primary_gate = original_validator
        proto.fit_adaptive_candidates = original_fit


def _variant_runtime(variant):
    if variant in ('full', 'bias', 'no_consolidation'):
        return _fixed_branch_runtime(variant)
    ablation = {'fixed_half': 'fixed_half_ablation',
                'sample_mean_nll': 'sample_mean_nll'}.get(variant)
    return (development._ablation_runtime(ablation)
            if ablation is not None else contextlib.nullcontext())


def _validate_runtime_freeze(root, variant, seed):
    root = Path(root)
    job_root, _ = _job_context(root, variant, seed)
    primary = _read_json(root / 'HELDOUT_PRIMARY_PLAN.json')
    plan_path = job_root / 'planned_protocol.json'
    launch_path = job_root / 'launch_started.json'
    planned = _read_json(plan_path)
    launched = _read_json(launch_path)
    if (planned is None
            or planned.get('freeze_identity') != primary['freeze_identity']
            or launched != {
                'kind': 'heldout_launch_started',
                'variant': variant, 'seed': seed,
                'command': planned.get('command'),
                'planned_protocol_sha256': file_sha256(plan_path),
            }):
        raise ValueError('held-out runtime freeze evidence is incomplete')
    _validate_live_freeze(
        root, primary['freeze_identity'],
        [root / 'HELDOUT_PRIMARY_PLAN.json', plan_path, launch_path],
    )
    return primary['freeze_identity']


def _runtime_job_root(root, variant, seed):
    root = Path(root)
    if seed == PRIMARY_SEED and variant in PRIMARY_VARIANTS:
        return root / 'primary' / variant
    if seed == PRIMARY_SEED and variant in ABLATIONS:
        return root / 'ablations' / variant
    return root / 'formal' / f'seed{seed}' / variant


def _validate_live_freeze(root, expected_identity, artifacts):
    freeze_path = Path(root) / 'HELDOUT_PROTOCOL_FROZEN.json'
    if freeze_identity(freeze_path) != expected_identity:
        raise ValueError('held-out freeze identity changed before test access')
    _validate_freeze_precedes(freeze_path, artifacts)


@contextlib.contextmanager
def _runtime_test_guard(root, variant, seed, expected_identity=None):
    import adaptive_consolidation_audit as audit_module

    delegate = audit_module.evaluate_deferred_cil_trajectory

    def guarded(snapshot_paths, final_checkpoint, dataset, task_classes, args):
        live_identity = _validate_runtime_freeze(root, variant, seed)
        if (expected_identity is not None
                and live_identity != expected_identity):
            raise ValueError('held-out freeze identity changed before evaluation')
        job_root = _runtime_job_root(root, variant, seed)
        evidence = (
            Path(root) / 'HELDOUT_PRIMARY_PLAN.json',
            job_root / 'planned_protocol.json',
            job_root / 'launch_started.json',
        )
        get_test_loader = dataset.get_test_loader

        def guarded_loader(*arguments, **keywords):
            _validate_live_freeze(root, live_identity, evidence)
            return get_test_loader(*arguments, **keywords)

        dataset.get_test_loader = guarded_loader
        try:
            return delegate(
                snapshot_paths, final_checkpoint, dataset, task_classes, args,
            )
        finally:
            dataset.get_test_loader = get_test_loader

    audit_module.evaluate_deferred_cil_trajectory = guarded
    runner_module = sys.modules.get('runner')
    runner_delegate = None
    if runner_module is not None:
        runner_delegate = runner_module.evaluate_deferred_cil_trajectory
        runner_module.evaluate_deferred_cil_trajectory = guarded
    try:
        yield
    finally:
        audit_module.evaluate_deferred_cil_trajectory = delegate
        if runner_module is not None:
            runner_module.evaluate_deferred_cil_trajectory = runner_delegate


def _execute_heldout_main(variant):
    root = os.environ.get('VFCL_HELDOUT_ROOT')
    seed = os.environ.get('VFCL_HELDOUT_SEED')
    if not root or seed is None:
        raise ValueError('held-out runtime identity environment is missing')
    seed = int(seed)
    identity = _validate_runtime_freeze(root, variant, seed)
    with _runtime_test_guard(root, variant, seed, identity):
        with _variant_runtime(variant):
            runpy.run_path(str(WORKTREE / 'main.py'), run_name='__main__')


def _execute_fixed_branch_main(variant):
    _execute_heldout_main(variant)


def build_command(variant, seed, device, results_dir, commit):
    if variant not in (*PRIMARY_VARIANTS, *ABLATIONS):
        raise ValueError(f'unknown held-out variant: {variant}')
    if seed not in (PRIMARY_SEED, *FORMAL_SEEDS):
        raise ValueError(f'illegal held-out seed: {seed}')
    mode = {
        'full': 'adaptive_dual_branch', 'bias': 'adaptive_dual_branch',
        'adaptive': 'adaptive_dual_branch',
        'no_consolidation': 'adaptive_dual_branch',
        'fixed_half': 'adaptive_dual_branch',
        'sample_mean_nll': 'adaptive_dual_branch',
    }[variant]
    rule = {
        'fixed_half': 'fixed_half_ablation',
        'sample_mean_nll': 'sample_mean_ablation',
    }.get(variant, 'class_balanced')
    enabled = 1
    command = [str(PYTHON), str(WORKTREE / 'main.py')]
    frozen = {
        '--data': 'tinyimagenet', '--data_path': str(DATA_ROOT.parent),
        '--num_classes': 200, '--num_tasks': 10, '--classes_per_task': 20,
        '--custom_tasks': '|'.join(
            ','.join(str(class_id) for class_id in range(start, start + 20))
            for start in range(0, 200, 20)
        ),
        '--num_parties': 4, '--party_widths': '16,16,16,16',
        '--model_type': 'resnet18', '--aggregation': 'sum',
        '--cl_method': 'proto_evolve', '--epochs_per_task': 50,
        '--batch_size': 64, '--deterministic': 1, '--data_flow_audit': 1,
        '--seed': seed, '--device': device, '--results_dir': results_dir,
        '--exp_name': f'heldout_tinyimagenet_{variant}_seed{seed}',
        '--lambda_validation_enabled': 1,
        '--lambda_validation_per_class': 50,
        '--lambda_validation_split_seed': 20260813,
        '--head_consolidation_enabled': enabled,
        '--head_consolidation_mode': mode,
        '--head_consolidation_schedule': 'final',
        '--head_consolidation_samples_per_class': 20,
        '--head_consolidation_regularization': 0.01,
        '--head_consolidation_class_regularization': 0.01,
        '--head_consolidation_task_regularization': 0.01,
        '--head_consolidation_task_weight': 1.3,
        '--head_full_lr': 0.01, '--head_full_steps': 500,
        '--head_bias_lr': 0.03, '--head_bias_steps': 600,
        '--head_gate_rule': rule,
        '--head_gate_solver_tolerance': '1e-12',
        '--head_gate_solver_max_iterations': 80,
        '--save_task_checkpoints': 3,
    }
    for option, value in frozen.items():
        _set(command, option, value)
    wrapper = (
        'from adaptive_tinyimagenet_heldout import '
        f'_execute_heldout_main; _execute_heldout_main({variant!r})'
    )
    command = [command[0], '-c', wrapper, *command[2:]]
    return command


def _load_freeze(root):
    freeze_path = Path(root) / 'HELDOUT_PROTOCOL_FROZEN.json'
    payload = _read_json(freeze_path)
    if payload is None or payload.get('status') != 'HELDOUT_PROTOCOL_FROZEN':
        raise ValueError('held-out protocol freeze is missing')
    _validate_dataset_manifest(payload.get('dataset_manifest'))
    dataset_manifest = payload['dataset_manifest']
    dataset_root = dataset_manifest.get('dataset_root')
    if (not isinstance(dataset_root, str) or not dataset_root):
        raise ValueError('held-out dataset root is invalid')
    if (_reject_symlink_components(dataset_root).resolve()
            != _reject_symlink_components(DATA_ROOT).resolve()):
        raise ValueError('held-out freeze differs from the runtime data root')
    if (build_heldout_manifest(
                _reject_symlink_components(dataset_root)
            ) != dataset_manifest):
        raise ValueError('held-out dataset changed after protocol freeze')
    if (payload.get('protocol') != PROTOCOL
            or payload.get('primary_variants') != list(PRIMARY_VARIANTS)
            or payload.get('formal_seeds_if_success') != list(FORMAL_SEEDS)
            or payload.get('ablations_if_success') != list(ABLATIONS)
            or payload.get('source_commit') != source_identity()['commit']
            or payload.get('source_sha256') != source_identity()['source_sha256']
            or validate_development_gate(
                payload.get('development_gate', {}).get('root', ''))
            != payload.get('development_gate')):
        raise ValueError('held-out protocol freeze evidence changed')
    return payload


def plan(root):
    root = Path(root)
    frozen = _load_freeze(root)
    identity = freeze_identity(root / 'HELDOUT_PROTOCOL_FROZEN.json')
    source = source_identity()
    if (frozen.get('source_commit') != source['commit']
            or frozen.get('source_sha256') != source['source_sha256']):
        raise ValueError('source changed after held-out freeze')
    path = root / 'HELDOUT_PRIMARY_PLAN.json'
    jobs = {}
    for variant in PRIMARY_VARIANTS:
        results = root / 'primary' / variant / 'outputs'
        command = build_command(
            variant, PRIMARY_SEED, '__DEVICE__', results, source['commit'],
        )
        jobs[variant] = {
            'variant': variant, 'seed': PRIMARY_SEED, 'command': command,
            'command_sha256': payload_sha256(command),
        }
    payload = {
        'schema_version': 1, 'kind': 'heldout_primary_plan',
        'freeze_sha256': file_sha256(root / 'HELDOUT_PROTOCOL_FROZEN.json'),
        'freeze_identity': identity,
        'source_commit': source['commit'], 'primary_jobs': jobs,
        'metrics': list(METRICS), 'tolerance': str(TOLERANCE),
        'strict_epsilon': str(STRICT_EPSILON),
        'formal_seeds': list(FORMAL_SEEDS), 'ablations': list(ABLATIONS),
        'followup_jobs': _followup_jobs(root, source['commit']),
    }
    if path.exists() or path.is_symlink():
        if _read_json(path) != payload:
            raise ValueError('held-out primary plan is immutable')
        _validate_freeze_precedes(
            root / 'HELDOUT_PROTOCOL_FROZEN.json', [path],
        )
        return payload
    _write_exclusive(path, payload)
    _validate_freeze_precedes(root / 'HELDOUT_PROTOCOL_FROZEN.json', [path])
    return payload


def _metrics(record):
    metrics = record.get('metrics') if isinstance(record, dict) else None
    strict = development._gate_metrics(metrics)
    return {name: Decimal(str(strict[name])) for name in METRICS}


def primary_gate_report(records):
    if set(records) != set(PRIMARY_VARIANTS):
        raise ValueError('held-out gate requires exactly Full, Bias, and Adaptive')
    values = {variant: _metrics(record) for variant, record in records.items()}
    constraints = {}
    strict = False
    for metric in METRICS:
        reference = max(values['full'][metric], values['bias'][metric])
        adaptive = values['adaptive'][metric]
        constraints[metric] = {
            'metric': metric, 'adaptive': str(adaptive),
            'better_fixed': str(reference),
            'floor': str(reference - TOLERANCE),
            'passed': adaptive >= reference - TOLERANCE,
        }
        strict = strict or adaptive > reference + STRICT_EPSILON
    passed = all(item['passed'] for item in constraints.values()) and strict
    return passed, {
        'schema_version': 1,
        'status': 'HELDOUT_GATE_SUCCESS' if passed else 'HELDOUT_GATE_FAILED',
        'constraints': constraints, 'strict_improvement': strict,
        'test_used_for_fit': False, 'test_used_for_selection': False,
    }


def validate_terminal_gate(root):
    root = Path(root)
    report = _read_json(root / 'HELDOUT_PRIMARY_GATE.json')
    if report is None:
        raise ValueError('held-out success report is missing')
    records = {variant: _strict_record(root, variant)
               for variant in PRIMARY_VARIANTS}
    passed, expected = primary_gate_report(records)
    expected = {
        **expected,
        'primary_plan_sha256': file_sha256(
            root / 'HELDOUT_PRIMARY_PLAN.json'),
        'records': {variant: {
            'path': str(root / 'primary' / variant / 'record.json'),
            'sha256': file_sha256(root / 'primary' / variant / 'record.json'),
        } for variant in PRIMARY_VARIANTS},
    }
    marker = _read_json(root / 'HELDOUT_GATE_SUCCESS')
    _validate_freeze_precedes(
        root / 'HELDOUT_PROTOCOL_FROZEN.json',
        [root / 'HELDOUT_PRIMARY_GATE.json', root / 'HELDOUT_GATE_SUCCESS'],
    )
    if marker != {
            'status': 'HELDOUT_GATE_SUCCESS',
            'report_sha256': file_sha256(root / 'HELDOUT_PRIMARY_GATE.json'),
            'primary_plan_sha256': expected['primary_plan_sha256']}:
        raise ValueError('held-out success marker is not intact')
    if (not passed or report != expected
            or report.get('status') != 'HELDOUT_GATE_SUCCESS'
            or not report.get('strict_improvement')):
        raise ValueError('held-out seed 42 did not succeed')
    return report


def _followup_jobs(root, commit):
    root = Path(root)
    jobs = {}
    specs = [
        (variant, PRIMARY_SEED) for variant in ABLATIONS
    ] + [
        (variant, seed) for seed in FORMAL_SEEDS
        for variant in (*PRIMARY_VARIANTS, *ABLATIONS)
    ]
    for variant, seed in specs:
        job_root = (root / 'ablations' / variant
                    if seed == PRIMARY_SEED else
                    root / 'formal' / f'seed{seed}' / variant)
        command = build_command(
            variant, seed, '__DEVICE__', job_root / 'outputs', commit,
        )
        jobs[f'{variant}:seed{seed}'] = {
            'variant': variant, 'seed': seed, 'command': command,
            'command_sha256': payload_sha256(command),
        }
    return jobs


def authorize_followups(root):
    root = Path(root)
    report = validate_terminal_gate(root)
    payload = {
        'schema_version': 1, 'kind': 'heldout_followup_authorization',
        'primary_seed': PRIMARY_SEED, 'formal_seeds': list(FORMAL_SEEDS),
        'ablations': list(ABLATIONS),
        'gate_sha256': file_sha256(root / 'HELDOUT_PRIMARY_GATE.json'),
        'strict_improvement': report['strict_improvement'],
        'jobs': _followup_jobs(root, _validate_plan(root)['source_commit']),
    }
    path = root / 'FOLLOWUP_AUTHORIZATION.json'
    if path.exists() or path.is_symlink():
        if _read_json(path) != payload:
            raise ValueError('held-out follow-up authorization is immutable')
        return payload
    _write_exclusive(path, payload)
    return payload


def _command_flags(command):
    if not isinstance(command, list) or len(command) < 4:
        raise ValueError('held-out command is malformed')
    flags = {}
    index = 3 if command[1:2] == ['-c'] else 2
    while index < len(command):
        option = command[index]
        if not isinstance(option, str) or not option.startswith('--'):
            raise ValueError('held-out command contains a positional argument')
        if index + 1 >= len(command) or option in flags:
            raise ValueError('held-out command flags are incomplete or duplicated')
        flags[option] = str(command[index + 1])
        index += 2
    return flags


def _validate_plan(root):
    root = Path(root)
    payload = _read_json(root / 'HELDOUT_PRIMARY_PLAN.json')
    frozen = _load_freeze(root)
    current = source_identity()
    current_identity = freeze_identity(
        root / 'HELDOUT_PROTOCOL_FROZEN.json'
    )
    if (not isinstance(payload, dict)
            or payload.get('freeze_identity') != current_identity):
        raise ValueError('held-out freeze identity changed')
    _validate_freeze_precedes(
        root / 'HELDOUT_PROTOCOL_FROZEN.json',
        [root / 'HELDOUT_PRIMARY_PLAN.json'],
    )
    if not (payload.get('kind') == 'heldout_primary_plan'
            and set(payload.get('primary_jobs', {})) == set(PRIMARY_VARIANTS)
            and payload.get('formal_seeds') == list(FORMAL_SEEDS)
            and payload.get('ablations') == list(ABLATIONS)):
        raise ValueError('held-out primary plan is missing or changed')
    if (payload.get('followup_jobs')
            != _followup_jobs(root, payload.get('source_commit'))):
        raise ValueError('held-out primary follow-up plan changed')
    if (payload.get('freeze_sha256')
            != file_sha256(root / 'HELDOUT_PROTOCOL_FROZEN.json')):
        raise ValueError('held-out primary freeze content changed')
    if (payload.get('source_commit') != current['commit']
            or frozen.get('source_commit') != current['commit']
            or frozen.get('source_sha256') != current['source_sha256']):
        raise ValueError('held-out primary source identity changed')
    for variant in PRIMARY_VARIANTS:
        expected = build_command(
            variant, PRIMARY_SEED, '__DEVICE__',
            root / 'primary' / variant / 'outputs', payload['source_commit'],
        )
        job = payload['primary_jobs'][variant]
        if job != {
                'variant': variant, 'seed': PRIMARY_SEED,
                'command': expected, 'command_sha256': payload_sha256(expected)}:
            raise ValueError(f'held-out primary job changed: {variant}')
    return payload


def _validate_authorization(root):
    root = Path(root)
    payload = _read_json(root / 'FOLLOWUP_AUTHORIZATION.json')
    report = validate_terminal_gate(root)
    expected = {
        'schema_version': 1, 'kind': 'heldout_followup_authorization',
        'primary_seed': PRIMARY_SEED, 'formal_seeds': list(FORMAL_SEEDS),
        'ablations': list(ABLATIONS),
        'gate_sha256': file_sha256(root / 'HELDOUT_PRIMARY_GATE.json'),
        'strict_improvement': report['strict_improvement'],
        'jobs': _followup_jobs(root, _validate_plan(root)['source_commit']),
    }
    if payload != expected:
        raise ValueError('held-out follow-up authorization is missing or changed')
    return payload


def _job_context(root, variant, seed):
    root = Path(root)
    primary = _validate_plan(root)
    if seed == PRIMARY_SEED and variant in PRIMARY_VARIANTS:
        return (root / 'primary' / variant,
                primary['primary_jobs'][variant]['command'])
    try:
        authorization = _validate_authorization(root)
    except (OSError, ValueError) as error:
        raise ValueError('held-out follow-up authorization is required') from error
    key = f'{variant}:seed{seed}'
    if key not in authorization['jobs']:
        raise ValueError('job is outside the exact held-out authorization')
    job_root = (root / 'ablations' / variant if seed == PRIMARY_SEED
                else root / 'formal' / f'seed{seed}' / variant)
    return job_root, authorization['jobs'][key]['command']


def bound_job_plan(root, variant, seed, device):
    root = Path(root)
    job_root, template = _job_context(root, variant, seed)
    plan_path = job_root / 'planned_protocol.json'
    if not plan_path.exists() and (job_root / 'outputs').exists() and any(
            (job_root / 'outputs').iterdir()):
        raise ValueError('held-out output exists without a prelaunch plan')
    command = [str(device) if value == '__DEVICE__' else value for value in template]
    payload = {
        'schema_version': 1, 'kind': 'heldout_bound_job_plan',
        'variant': variant, 'seed': seed, 'device': str(device),
        'command': command, 'command_sha256': payload_sha256(command),
        'freeze_identity': freeze_identity(
            root / 'HELDOUT_PROTOCOL_FROZEN.json'
        ),
        'primary_plan_sha256': file_sha256(
            root / 'HELDOUT_PRIMARY_PLAN.json'
        ),
    }
    if plan_path.exists() or plan_path.is_symlink():
        if _read_json(plan_path) != payload:
            raise ValueError('held-out bound job plan is immutable')
        _validate_freeze_precedes(
            root / 'HELDOUT_PROTOCOL_FROZEN.json', [plan_path],
        )
        return payload
    _write_exclusive(plan_path, payload)
    _validate_freeze_precedes(
        root / 'HELDOUT_PROTOCOL_FROZEN.json', [plan_path],
    )
    return payload


def _discover_run(job_root, variant, seed):
    output = Path(job_root) / 'outputs'
    if output.is_symlink() or not output.is_dir():
        return None
    prefix = f'heldout_tinyimagenet_{variant}_seed{seed}_'
    pattern = re.compile(re.escape(prefix) + r'[0-9]{8}_[0-9]{6}')
    runs = [path for path in output.iterdir()
            if path.is_dir() and not path.is_symlink()
            and pattern.fullmatch(path.name)
            and (path / 'config.json').is_file()
            and (path / 'results.json').is_file()]
    if len(runs) > 1:
        raise ValueError('multiple held-out runs exist for one job')
    return runs[0] if runs else None


def _validate_run_identity(run, output, variant, seed, config):
    exp_name = f'heldout_tinyimagenet_{variant}_seed{seed}'
    pattern = re.compile(
        re.escape(exp_name) + r'_[0-9]{8}_[0-9]{6}'
    )
    if (not isinstance(config, dict) or not pattern.fullmatch(run.name)
            or config.get('exp_name') != exp_name
            or config.get('results_dir') != str(output)
            or config.get('output_dir') != str(run)):
        raise ValueError('held-out run identity changed')


def _deferred_metrics(run, config, adaptive):
    snapshots = adaptive.get('snapshots')
    if not isinstance(snapshots, list) or len(snapshots) != 10:
        raise ValueError('held-out adaptive snapshots are incomplete')
    paths = []
    for record in snapshots:
        relative = Path(record.get('path', '')) if isinstance(record, dict) else Path()
        if relative.is_absolute() or '..' in relative.parts or not relative.name:
            raise ValueError('held-out adaptive snapshot path is invalid')
        paths.append(_reject_symlink_components(run / relative))
    seen = snapshots[-1].get('seen_task_classes')
    if not isinstance(seen, dict) or len(seen) != 10:
        raise ValueError('held-out task identity is incomplete')
    task_classes = {
        int(task): [int(class_id) for class_id in classes]
        for task, classes in seen.items()
    }
    args = SimpleNamespace(**config)
    args.lambda_validation_enabled = 0
    args.bic_enabled = 0
    args.data_flow_audit = 0
    args.head_consolidation_enabled = 0
    evaluation = VFLDataset(args)
    deferred = evaluate_deferred_cil_trajectory(
        paths, run / 'adaptive_final.pt', evaluation, task_classes, args,
    )
    return _metrics({'metrics': deferred.get('cl_metrics')})


def _reject_retroactive_artifacts(plan_path, launch_path, artifacts):
    planned = _reject_symlink_components(plan_path).stat()
    launched = _reject_symlink_components(launch_path).stat()
    for path in artifacts:
        details = _reject_symlink_components(path).stat()
        if (planned.st_mtime_ns > details.st_mtime_ns
                or launched.st_mtime_ns > details.st_mtime_ns
                or planned.st_ctime_ns > details.st_ctime_ns
                or launched.st_ctime_ns > details.st_ctime_ns):
            raise ValueError('retroactive held-out provenance is forbidden')


def audit_completed_run(root, variant, seed, run_dir):
    root = Path(root)
    primary = _validate_plan(root)
    job_root, template = _job_context(root, variant, seed)
    plan_path = _reject_symlink_components(job_root / 'planned_protocol.json')
    launch_path = _reject_symlink_components(job_root / 'launch_started.json')
    planned = _read_json(plan_path)
    launched = _read_json(launch_path)
    if (planned is None
            or planned.get('freeze_identity') != primary['freeze_identity']
            or launched != {
            'kind': 'heldout_launch_started',
            'variant': variant, 'seed': seed,
            'command': planned.get('command'),
            'planned_protocol_sha256': file_sha256(
                plan_path)}):
        raise ValueError('held-out prelaunch evidence is incomplete')
    _validate_freeze_precedes(
        root / 'HELDOUT_PROTOCOL_FROZEN.json',
        [root / 'HELDOUT_PRIMARY_PLAN.json', plan_path, launch_path],
    )
    expected = [planned['device'] if value == '__DEVICE__' else value
                for value in template]
    if planned.get('command') != expected:
        raise ValueError('held-out command differs from its frozen template')
    run = _reject_symlink_components(run_dir).resolve()
    output = (job_root / 'outputs').resolve()
    if run.parent != output or run.is_symlink():
        raise ValueError('held-out run is outside its exact output root')
    config_path = run / 'config.json'
    results_path = run / 'results.json'
    validation_path = run / 'validation' / 'validation_manifest.json'
    for path in (config_path, results_path, validation_path):
        _reject_symlink_components(path)
    if any(path.is_symlink() or not path.is_file()
           for path in (config_path, results_path, validation_path)):
        raise ValueError('held-out run evidence is incomplete or symlinked')
    config = _read_json(config_path)
    results = _read_json(results_path)
    validation = _read_json(validation_path)
    _validate_run_identity(run, output, variant, seed, config)
    flags = _command_flags(planned['command'])
    expected_config = {
        option[2:]: (
            [int(item) for item in value.split(',')]
            if option == '--party_widths' else
            development.config_value(option, value)
        ) for option, value in flags.items()
    }
    if (config is None or results is None or validation is None
            or any(config.get(key) != value
                   for key, value in expected_config.items())
            or validation.get('seed') != 20260813
            or validation.get('per_class') != 50
            or validation.get('sha256')
            != _load_freeze(root)['dataset_manifest']['split_sha256']):
        raise ValueError('held-out run protocol evidence changed')
    metrics_decimal = _metrics({'metrics': results.get('cl_metrics')})
    metrics = {name: float(value) for name, value in metrics_decimal.items()}
    artifacts = {
        'config': config_path, 'results': results_path,
        'validation_manifest': validation_path,
    }
    if variant in (*PRIMARY_VARIANTS, *ABLATIONS):
        freeze_path = run / 'ADAPTIVE_STATE_FROZEN.json'
        checkpoint = run / 'adaptive_final.pt'
        _reject_symlink_components(freeze_path)
        _reject_symlink_components(checkpoint)
        if any(path.is_symlink() or not path.is_file()
               for path in (freeze_path, checkpoint)):
            raise ValueError('adaptive held-out state is not frozen')
        adaptive = _read_json(freeze_path)
        with _variant_runtime(variant):
            audited = (development.audit_adaptive_checkpoint(
                run, adaptive.get('audit_spec')) if adaptive is not None else None)
        if adaptive is None or audited != adaptive:
            raise ValueError('adaptive held-out checkpoint audit failed')
        data_flow = adaptive.get('data_flow', {})
        diagnostics = adaptive.get('diagnostics', {})
        gate = adaptive.get('result', {}).get('gate', {})
        source_splits = diagnostics.get('source_splits', {})
        expected_gate = {
            'full': ('fixed_full', False, 1.0),
            'bias': ('fixed_bias', False, 0.0),
            'adaptive': ('class_balanced', True, None),
            'fixed_half': ('fixed_half_ablation', False, 0.5),
            'sample_mean_nll': ('sample_mean_ablation', False, None),
            'no_consolidation': ('no_consolidation', False, 0.5),
        }[variant]
        if (adaptive.get('status') != 'ADAPTIVE_STATE_FROZEN'
                or adaptive.get('source', {}).get('source_commit')
                != _validate_plan(root)['source_commit']
                or adaptive.get('checkpoint', {}).get('path')
                != 'adaptive_final.pt'
                or gate.get('gate_rule') != expected_gate[0]
                or gate.get('is_primary') is not expected_gate[1]
                or (expected_gate[2] is not None
                    and gate.get('g') != expected_gate[2])
                or gate.get('tolerance') != 1e-12
                or gate.get('max_iterations') != 80
                or source_splits != {
                    'replay': 'persistent_training_replay',
                    'validation': 'frozen_training_validation',
                    'test_used': False,
                }
                or not isinstance(adaptive.get('replay'), dict)
                or adaptive['replay'].get('count', 0) <= 0
                or set(data_flow) != {
                    'candidates_frozen_before_validation',
                    'validation_before_freeze', 'test_before_install',
                    'test_used_for_diagnostics', 'solver_input',
                    'audit_prefix_record_count', 'audit_prefix_sha256',
                }
                or data_flow.get('candidates_frozen_before_validation') is not True
                or data_flow.get('validation_before_freeze') is not False
                or data_flow.get('test_before_install') is not False
                or data_flow.get('test_used_for_diagnostics') is not False
                or data_flow.get('solver_input')
                != 'class_balanced_validation_nll'
                or not isinstance(data_flow.get('audit_prefix_record_count'), int)
                or not re.fullmatch(
                    r'[0-9a-f]{64}',
                    str(data_flow.get('audit_prefix_sha256', ''))
                )):
            raise ValueError('held-out test was accessed before adaptive freeze')
        artifacts.update(adaptive_freeze=freeze_path,
                         adaptive_checkpoint=checkpoint)
        for index, snapshot in enumerate(adaptive.get('snapshots', [])):
            relative = Path(snapshot.get('path', ''))
            artifacts[f'adaptive_snapshot_{index}'] = run / relative
        data_flow_path = run / 'data_flow_audit.jsonl'
        _reject_symlink_components(data_flow_path)
        if data_flow_path.is_symlink() or not data_flow_path.is_file():
            raise ValueError('held-out data-flow audit is missing')
        artifacts['data_flow_audit'] = data_flow_path
        _validate_freeze_precedes(
            root / 'HELDOUT_PROTOCOL_FROZEN.json', artifacts.values(),
        )
        _reject_retroactive_artifacts(
            job_root / 'planned_protocol.json',
            job_root / 'launch_started.json', artifacts.values(),
        )
        with _variant_runtime(variant):
            deferred_metrics = _deferred_metrics(run, config, adaptive)
        if metrics_decimal != deferred_metrics:
            raise ValueError('held-out metrics differ from checkpoint evaluation')
    current_manifest = build_heldout_manifest(
        Path(config['data_path']) / 'tiny-imagenet-200'
    )
    if current_manifest != _load_freeze(root)['dataset_manifest']:
        raise ValueError('held-out dataset changed after protocol freeze')
    _reject_retroactive_artifacts(
        job_root / 'planned_protocol.json', job_root / 'launch_started.json',
        artifacts.values(),
    )
    return {
        'schema_version': 1, 'variant': variant, 'seed': seed,
        'freeze_identity': primary['freeze_identity'],
        'command': planned['command'], 'run_dir': str(run),
        'metrics': metrics,
        'artifacts': {name: str(path) for name, path in artifacts.items()},
        'sha256': {name: file_sha256(path) for name, path in artifacts.items()},
        'passed': True,
    }


def _strict_record(root, variant):
    root = Path(root)
    path = root / 'primary' / variant / 'record.json'
    success = path.parent / 'SUCCESS'
    record = _read_json(path)
    if (record is None or success.is_symlink() or not success.is_file()
            or record.get('variant') != variant
            or record.get('seed') != PRIMARY_SEED
            or record.get('passed') is not True):
        raise ValueError(f'missing held-out primary record: {variant}')
    current_identity = freeze_identity(
        root / 'HELDOUT_PROTOCOL_FROZEN.json'
    )
    if record.get('freeze_identity') != current_identity:
        raise ValueError('held-out record freeze identity changed')
    _validate_freeze_precedes(
        root / 'HELDOUT_PROTOCOL_FROZEN.json', [path, success],
    )
    audited = audit_completed_run(
        root, variant, PRIMARY_SEED, record.get('run_dir', ''),
    )
    if audited != record:
        raise ValueError(f'held-out primary record changed: {variant}')
    return record


def _run_one(root, variant, seed, device):
    root = Path(root)
    job_root, _ = _job_context(root, variant, seed)
    if (job_root / 'SUCCESS').is_file():
        record = _read_json(job_root / 'record.json')
        if record != audit_completed_run(
                root, variant, seed, record.get('run_dir', '')):
            raise ValueError('completed held-out record failed re-audit')
        return 0
    planned = bound_job_plan(root, variant, seed, device)
    launch = {
        'kind': 'heldout_launch_started', 'variant': variant, 'seed': seed,
        'command': planned['command'],
        'planned_protocol_sha256': file_sha256(
            job_root / 'planned_protocol.json'),
    }
    _write_exclusive(job_root / 'launch_started.json', launch)
    job_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['VFCL_HELDOUT_ROOT'] = str(root)
    environment['VFCL_HELDOUT_SEED'] = str(seed)
    if variant in ('fixed_half', 'sample_mean_nll'):
        environment['VFCL_REVIEWED_ADAPTIVE_ABLATION'] = '1'
    with (job_root / 'job.log').open('ab') as log:
        completed = subprocess.run(
            planned['command'], cwd=WORKTREE, env=environment,
            stdout=log, stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        _write_exclusive(job_root / 'FAILED.json', {
            'variant': variant, 'seed': seed,
            'returncode': completed.returncode,
        })
        return completed.returncode
    run = _discover_run(job_root, variant, seed)
    if run is None:
        raise ValueError('zero exit produced no auditable held-out run')
    record = audit_completed_run(root, variant, seed, run)
    _write_exclusive(job_root / 'record.json', record)
    _write_exclusive(job_root / 'SUCCESS', {'status': 'SUCCESS'})
    return 0


def run_job(root, variant, seed, device):
    root = Path(root)
    if any((root / name).exists() or (root / name).is_symlink()
           for name in ('HELDOUT_GATE_FAILED',)):
        raise ValueError('held-out seed 42 failed; further jobs are forbidden')
    if seed != PRIMARY_SEED or variant not in PRIMARY_VARIANTS:
        try:
            _validate_authorization(root)
        except (OSError, ValueError) as error:
            raise ValueError('held-out follow-up authorization is required') from error
    return _run_one(root, variant, seed, device)


def summarize(root):
    root = Path(root)
    records = {variant: _strict_record(root, variant)
               for variant in PRIMARY_VARIANTS}
    passed, report = primary_gate_report(records)
    report = {
        **report,
        'primary_plan_sha256': file_sha256(
            root / 'HELDOUT_PRIMARY_PLAN.json'),
        'records': {variant: {
            'path': str(root / 'primary' / variant / 'record.json'),
            'sha256': file_sha256(root / 'primary' / variant / 'record.json')
            if (root / 'primary' / variant / 'record.json').is_file()
            else payload_sha256(records[variant]),
        } for variant in PRIMARY_VARIANTS},
    }
    report_path = root / 'HELDOUT_PRIMARY_GATE.json'
    if report_path.exists() or report_path.is_symlink():
        if _read_json(report_path) != report:
            raise ValueError('held-out terminal report is immutable')
    else:
        _write_exclusive(report_path, report)
    common = {
        'report_sha256': file_sha256(report_path),
        'primary_plan_sha256': report['primary_plan_sha256'],
    }
    execution = {'status': 'EXECUTION_SUCCESS', **common}
    if not (root / 'EXECUTION_SUCCESS').exists():
        _write_exclusive(root / 'EXECUTION_SUCCESS', execution)
    status = 'HELDOUT_GATE_SUCCESS' if passed else 'HELDOUT_GATE_FAILED'
    marker = {'status': status, **common}
    if not (root / status).exists():
        _write_exclusive(root / status, marker)
    return (0 if passed else 3), report


def static_check():
    if not (PROTOCOL['classes'] == 200 and PROTOCOL['tasks'] == 10
            and PROTOCOL['validation_split_seed'] == 20260813
            and PRIMARY_VARIANTS == ('full', 'bias', 'adaptive')
            and FORMAL_SEEDS == (43, 44)):
        raise RuntimeError('held-out static contract failed')
    print(json.dumps({
        'status': 'OK', 'primary_seed': PRIMARY_SEED,
        'primary_variants': list(PRIMARY_VARIANTS),
        'formal_seeds_if_success': list(FORMAL_SEEDS),
        'ablations_if_success': list(ABLATIONS),
    }, sort_keys=True))
    return 0


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('check')
    freeze_cmd = sub.add_parser('freeze')
    freeze_cmd.add_argument('--root', type=Path, required=True)
    freeze_cmd.add_argument('--dataset-root', type=Path, required=True)
    freeze_cmd.add_argument('--development-root', type=Path, required=True)
    plan_cmd = sub.add_parser('plan')
    plan_cmd.add_argument('--root', type=Path, required=True)
    authorize = sub.add_parser('authorize-followups')
    authorize.add_argument('--root', type=Path, required=True)
    run = sub.add_parser('run-job')
    run.add_argument('--root', type=Path, required=True)
    run.add_argument('--variant', choices=(*PRIMARY_VARIANTS, *ABLATIONS),
                     required=True)
    run.add_argument('--seed', type=int, choices=(PRIMARY_SEED, *FORMAL_SEEDS),
                     required=True)
    run.add_argument('--device', required=True)
    summary = sub.add_parser('summarize')
    summary.add_argument('--root', type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.action == 'check':
            return static_check()
        if args.action == 'freeze':
            freeze(args.root, args.dataset_root, args.development_root)
            return 0
        if args.action == 'plan':
            plan(args.root)
            return 0
        if args.action == 'authorize-followups':
            authorize_followups(args.root)
            return 0
        if args.action == 'run-job':
            return run_job(args.root, args.variant, args.seed, args.device)
        if args.action == 'summarize':
            return summarize(args.root)[0]
    except (OSError, ValueError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
