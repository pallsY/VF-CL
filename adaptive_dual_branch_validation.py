"""Frozen three-dataset development gate for adaptive dual-branch heads."""
import argparse
import contextlib
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from PIL import Image
from adaptive_consolidation_audit import (
    SOURCE_FILES as ADAPTIVE_SOURCE_FILES,
    audit_adaptive_checkpoint,
    evaluate_deferred_cil_trajectory,
)
from data_utils import VFLDataset
from dataset_protocol_smoke import replace_arg
import unified_head_consolidation_factorial as factorial
from torchvision import datasets as vision_datasets, transforms as vision_transforms


REPO = factorial.ROOT
WORKTREE = Path(__file__).resolve().parent
PYTHON = Path(sys.executable).resolve()
SOURCE_COMMIT = '4651243cd8818df4f5790c9e58f3fe4236aad012'
SEED = 42
DATASETS = ('cifar100', 'isolet', 'upmc_food101')
SMOKE_DATASETS = (*DATASETS, 'synthetic_tinyimagenet')
METRICS = ('AA_final', 'BWT', 'AA_final_taskil')
TOLERANCE = Decimal('0.01')
INCUMBENT_CELLS = {'cifar100': 'B', 'isolet': 'A', 'upmc_food101': 'A'}
EXPECTED_VALIDATION_HASHES = dict(factorial.EXPECTED_VALIDATION_HASHES)
ABLATIONS = {
    'fixed_half_ablation': 'fixed_half_ablation',
    'sample_mean_nll': 'sample_mean_ablation',
}
TASK8_SOURCE_FILES = (
    'adaptive_dual_branch_validation.py',
    'run_adaptive_dual_branch_validation.sh',
)
ROOT_PATTERN = re.compile(
    r'adaptive_dual_branch_validation_seed42_[0-9]{8}_[0-9]{6}'
)
AUDIT_INCOMPLETE = 2
GATE_FAILED = 3


def _canonical_bytes(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
        allow_nan=False,
    ).encode('ascii')


def _payload_sha256(payload):
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_scalar(value):
    if not isinstance(value, str):
        return value
    if value in {'True', 'False'}:
        return value == 'True'
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


_STRING_CONFIG_OPTIONS = {
    '--device', '--data', '--data_path', '--mfeat_path', '--vector_npz',
    '--model_type', '--aggregation', '--optimizer', '--task_ce_mode',
    '--ul_method', '--cl_method', '--replay_mode', '--party_kd_mode',
    '--expected_party_kd_variant', '--proto_replay_loss_norm',
    '--head_consolidation_mode', '--head_consolidation_schedule',
    '--head_gate_rule', '--results_dir', '--exp_name',
}


def config_value(option, value):
    if option == '--party_widths':
        if isinstance(value, list):
            return value
        return [int(item) for item in str(value).split(',')]
    if option == '--unlearn_after_tasks':
        if isinstance(value, list):
            return value
        return [int(item) for item in str(value).split(',')]
    if option == '--unlearn_classes':
        if isinstance(value, list):
            return value
        return [
            [int(item) for item in group.split(',')]
            for group in str(value).split(';')
        ]
    return value if option in _STRING_CONFIG_OPTIONS else json_scalar(value)


def _reject_symlink_components(path):
    path = Path(path).absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            details = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(details.st_mode):
            raise ValueError(f'symlinked path component is not allowed: {current}')
    return path


def _validated_root(root, create=False):
    root = _reject_symlink_components(root)
    if not ROOT_PATTERN.fullmatch(root.name):
        raise ValueError('root basename does not match the frozen seed-42 protocol')
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ValueError('validation root is not a trusted directory')
    return root


def _within(path, root, label):
    path = _reject_symlink_components(path)
    root = _validated_root(root)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f'{label} is outside validation root') from error
    return path


def _read_json(path):
    path = _reject_symlink_components(path)
    try:
        raw = path.read_text(encoding='utf-8')
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_new_json(path, payload):
    path = _reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _canonical_bytes(payload) + b'\n'
    try:
        with path.open('xb') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError(f'immutable evidence already differs: {path}')
    return path


def _replace_json(path, payload):
    path = _reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(_canonical_bytes(payload) + b'\n')
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_symlink():
            raise ValueError(f'symlinked target is not allowed: {path}')
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return path


def _safe_unlink(path):
    path = _reject_symlink_components(path)
    try:
        details = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ValueError(f'unsafe marker target: {path}')
    path.unlink()


def _touch_new(path):
    path = _reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'unsafe marker target: {path}')
    else:
        os.close(descriptor)


def _git(*arguments):
    return subprocess.check_output(
        ['git', '-C', str(WORKTREE), *arguments], text=True,
    ).strip()


def adaptive_source_identity():
    commit = _git('rev-parse', 'HEAD')
    subprocess.run(
        ['git', '-C', str(WORKTREE), 'merge-base', '--is-ancestor',
         SOURCE_COMMIT, commit], check=True, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    hashes = {
        name: file_sha256(WORKTREE / name)
        for name in (*ADAPTIVE_SOURCE_FILES, *TASK8_SOURCE_FILES)
    }
    return {'commit': commit, 'sha256': hashes}


def incumbent_evidence(dataset):
    validate_job(dataset, SEED)
    cell = INCUMBENT_CELLS[dataset]
    with tempfile.TemporaryDirectory(prefix='adaptive-incumbent-audit-') as root:
        audited = factorial.audit_reference(dataset, cell, root)
    if audited.get('passed') is not True or not all(
            value is True for value in audited.get('checks', {}).values()):
        raise ValueError(f'factorial incumbent audit failed for {dataset}')
    if (audited.get('spec') != f'{dataset}:{cell}:{SEED}'
            or audited.get('source_paths', {}).get('run')
            != str(factorial.REFERENCE_RUNS[(dataset, cell)])):
        raise ValueError(f'factorial incumbent identity mismatch for {dataset}')
    hashes = audited.get('source_sha256')
    if not isinstance(hashes, dict) or not all(
            value is None or re.fullmatch(r'[0-9a-f]{64}', value)
            for value in hashes.values()):
        raise ValueError(f'factorial incumbent hashes are incomplete for {dataset}')
    metrics = _gate_metrics(audited.get('metrics'))
    compact = {
        'spec': audited['spec'], 'dataset': dataset, 'cell': cell, 'seed': SEED,
        'source_paths': audited['source_paths'],
        'source_sha256': hashes,
        'validation_hash': audited['validation_logical_hash']['actual'],
        'metrics': metrics,
        'checks': audited['checks'], 'passed': True,
    }
    return {**compact, 'record_sha256': _payload_sha256(compact)}


def primary_specs():
    return [f'{dataset}:adaptive:{SEED}' for dataset in DATASETS]


def ablation_specs():
    return [
        f'{dataset}:{name}:{SEED}'
        for name in ABLATIONS for dataset in DATASETS
    ]


def validate_job(dataset, seed):
    if dataset not in DATASETS or seed != SEED:
        raise ValueError(f'unsupported adaptive job: {dataset}:{seed}')
    return dataset, seed


def _set_arg(command, option, value):
    if option in command:
        replace_arg(command, option, value)
    else:
        command.extend([option, str(value)])


def _command_flags(command):
    try:
        start = next(
            index for index, value in enumerate(command)
            if isinstance(value, str) and value.startswith('--')
        )
    except StopIteration as error:
        raise ValueError('command has no option pairs') from error
    options = command[start:]
    if len(options) % 2:
        raise ValueError('command options are not exact pairs')
    return dict(zip(options[::2], options[1::2]))


def _ablation_solver_input(ablation):
    return {
        'fixed_half_ablation': 'fixed_g_0.5_class_balanced_validation_nll',
        'sample_mean_nll': 'sample_mean_validation_nll',
    }[ablation]


@contextlib.contextmanager
def _ablation_runtime(ablation):
    """Apply one command-bound ablation to real training and Task 7 re-audit."""
    if ablation not in ABLATIONS:
        raise ValueError(f'unknown adaptive ablation: {ablation}')
    import adaptive_consolidation_audit as audit_module
    import adaptive_head_consolidation as head_module
    import cl_methods.proto_evolve as proto_module

    solver = (
        head_module.fixed_half_gate_record
        if ablation == 'fixed_half_ablation'
        else head_module.solve_sample_mean_ablation
    )
    original_validator = head_module._validate_primary_gate
    original_diagnostics = head_module.build_adaptive_diagnostics

    def validate_ablation_gate(gate):
        if (not isinstance(gate, dict)
                or gate.get('gate_rule') != ABLATIONS[ablation]
                or gate.get('is_primary') is not False):
            raise ValueError('adaptive ablation gate identity is invalid')
        original_validator({
            **gate, 'gate_rule': 'class_balanced', 'is_primary': True,
        })

    def ablation_diagnostics(*arguments, **keywords):
        values = list(arguments)
        gate = values[7] if len(values) > 7 else keywords['gate']
        diagnostic_gate = dict(gate)
        if ablation == 'sample_mean_nll':
            candidates = values[2]
            validation_x, validation_y = values[4], values[5]
            full, bias = head_module.adaptive_candidate_log_probabilities(
                values[0], candidates, validation_x,
            )
            classes = candidates.ordered_classes
            diagnostic_gate.update({
                'full_branch_nll': float(
                    head_module.class_balanced_mixture_nll(
                        full, bias, validation_y, classes, 1.0
                    )
                ),
                'bias_branch_nll': float(
                    head_module.class_balanced_mixture_nll(
                        full, bias, validation_y, classes, 0.0
                    )
                ),
                'mixture_nll': float(
                    head_module.class_balanced_mixture_nll(
                        full, bias, validation_y, classes, gate['g']
                    )
                ),
            })
            values[7] = diagnostic_gate
        result = original_diagnostics(*values, **keywords)
        result['solver_input'] = _ablation_solver_input(ablation)
        result['ablation_solver_nlls'] = {
            'full': gate['full_branch_nll'],
            'bias': gate['bias_branch_nll'],
            'mixture': gate['mixture_nll'],
        }
        return result

    targets = (
        (head_module, '_validate_primary_gate', validate_ablation_gate),
        (proto_module, 'solve_global_mixture_weight', solver),
        (audit_module, 'solve_global_mixture_weight', solver),
        (proto_module, 'build_adaptive_diagnostics', ablation_diagnostics),
        (audit_module, 'build_adaptive_diagnostics', ablation_diagnostics),
    )
    originals = [(module, name, getattr(module, name))
                 for module, name, _ in targets]
    try:
        for module, name, replacement in targets:
            setattr(module, name, replacement)
        yield
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)


def _execute_ablation_main(ablation):
    """Execute the ordinary training entry under an auditable ablation rule."""
    with _ablation_runtime(ablation):
        runpy.run_path(str(WORKTREE / 'main.py'), run_name='__main__')


def build_command(dataset, device, results_dir, smoke=False, ablation=None):
    validate_job(dataset, SEED)
    if ablation is not None and ablation not in ABLATIONS:
        raise ValueError(f'unknown adaptive ablation: {ablation}')
    command = factorial.build_command(
        f'{dataset}:A:{SEED}', device, results_dir, smoke=smoke,
    )
    command[0] = str(PYTHON)
    command[1] = str(WORKTREE / 'main.py')
    frozen = {
        '--head_consolidation_enabled': 1,
        '--head_consolidation_mode': 'adaptive_dual_branch',
        '--head_consolidation_regularization': 0.01,
        '--head_consolidation_class_regularization': 0.01,
        '--head_consolidation_task_regularization': 0.01,
        '--head_consolidation_task_weight': 1.3,
        '--head_consolidation_samples_per_class': 20,
        '--head_consolidation_schedule': 'final',
        '--head_consolidation_lr': 0.01,
        '--head_consolidation_steps': 500,
        '--head_full_lr': 0.01,
        '--head_full_steps': 500,
        '--head_bias_lr': 0.03,
        '--head_bias_steps': 600,
        '--head_gate_rule': ABLATIONS.get(ablation, 'class_balanced'),
        '--head_gate_solver_tolerance': '1e-12',
        '--head_gate_solver_max_iterations': 80,
        '--lambda_validation_enabled': 1,
        '--lambda_validation_split_seed': (
            20260729 if dataset == 'cifar100'
            else factorial.DATASETS[dataset]['validation_split_seed']
        ),
        '--data_flow_audit': 1,
        '--save_task_checkpoints': 3,
        '--seed': SEED,
        '--results_dir': results_dir,
        '--exp_name': (
            f'adaptive_{dataset}_seed42' if ablation is None
            else f'adaptive_{dataset}_{ablation}_seed42'
        ),
    }
    for option, value in frozen.items():
        _set_arg(command, option, value)
    command = [str(value) for value in command]
    if ablation is not None:
        wrapper = (
            'from adaptive_dual_branch_validation import '
            f'_execute_ablation_main; _execute_ablation_main({ablation!r})'
        )
        command = [command[0], '-c', wrapper, *command[2:]]
    return command


def build_tiny_synthetic_smoke_command(root, device):
    root = _validated_root(root)
    fixture = root / 'synthetic_tiny_fixture'
    results = root / 'smoke' / 'synthetic_tinyimagenet' / 'outputs'
    command = [str(PYTHON), str(WORKTREE / 'main.py')]
    frozen = {
        '--data': 'tinyimagenet', '--data_path': fixture,
        '--num_classes': 4, '--num_tasks': 2,
        '--custom_tasks': '0,1|2,3', '--classes_per_task': 2,
        '--num_parties': 4, '--party_widths': '16,16,16,16',
        '--model_type': 'resnet18', '--aggregation': 'sum',
        '--epochs_per_task': 1, '--batch_size': 4, '--num_workers': 2,
        '--optimizer': 'sgd', '--lr': 0.001, '--bottom_lr_scale': 1.0,
        '--task_ce_mode': 'method', '--momentum': 0.9,
        '--weight_decay': 0.0005, '--device': device,
        '--unlearn_after_tasks': 99, '--unlearn_classes': 0,
        '--ul_method': 'retrain', '--replay_mode': 'prototype',
        '--deterministic': 1, '--data_flow_audit': 1,
        '--bic_enabled': 0, '--lambda_validation_enabled': 1,
        '--lambda_validation_per_class': 50,
        '--lambda_validation_split_seed': 20260813,
        '--save_task_checkpoints': 3, '--seed': SEED,
        '--cl_method': 'proto_evolve', '--dep_tracking_enabled': 1,
        '--party_kd_enabled': 1, '--party_kd_mode': 'uniform',
        '--expected_party_kd_variant': 'uniform', '--party_kd_lambda': 1.0,
        '--proto_replay_loss_norm': 'sample_mean',
        '--proto_replay_ratio': 1.0, '--proto_lambda_a': 0.15,
        '--fim_freeze_frac': 0,
        '--head_consolidation_enabled': 1,
        '--head_consolidation_mode': 'adaptive_dual_branch',
        '--head_consolidation_schedule': 'final',
        '--head_consolidation_samples_per_class': 20,
        '--head_consolidation_regularization': 0.01,
        '--head_consolidation_class_regularization': 0.01,
        '--head_consolidation_task_regularization': 0.01,
        '--head_consolidation_task_weight': 1.3,
        '--head_full_lr': 0.01, '--head_full_steps': 500,
        '--head_bias_lr': 0.03, '--head_bias_steps': 600,
        '--head_gate_rule': 'class_balanced',
        '--head_gate_solver_tolerance': '1e-12',
        '--head_gate_solver_max_iterations': 80,
        '--distill_weight': 0.25, '--feat_distill_weight': 0.05,
        '--current_supcon_weight': 0,
        '--results_dir': results,
        '--exp_name': 'adaptive_synthetic_tinyimagenet_smoke_seed42',
    }
    for option, value in frozen.items():
        _set_arg(command, option, value)
    return [str(value) for value in command]


def _smoke_plan_payload(root):
    root = _validated_root(root)
    source = adaptive_source_identity()
    jobs = {}
    for dataset in DATASETS:
        command = build_command(
            dataset, '__DEVICE__', root / 'smoke' / dataset / 'outputs',
            smoke=True,
        )
        jobs[dataset] = {
            'spec': f'{dataset}:synthetic_smoke:{SEED}',
            'dataset': dataset, 'seed': SEED, 'smoke': True,
            'command': command, 'command_sha256': _payload_sha256(command),
        }
    tiny = build_tiny_synthetic_smoke_command(root, '__DEVICE__')
    jobs['synthetic_tinyimagenet'] = {
        'spec': f'synthetic_tinyimagenet:synthetic_smoke:{SEED}',
        'dataset': 'synthetic_tinyimagenet', 'seed': SEED, 'smoke': True,
        'command': tiny, 'command_sha256': _payload_sha256(tiny),
    }
    return {
        'schema_version': 1, 'kind': 'adaptive_synthetic_smoke_plan',
        'source_commit': source['commit'], 'source_sha256': source['sha256'],
        'jobs': jobs,
    }


def _validate_smoke_plan(root, payload=None):
    root = _validated_root(root)
    payload = _read_json(root / 'SMOKE_PLAN.json') if payload is None else payload
    expected = _smoke_plan_payload(root)
    if payload != expected:
        raise ValueError('synthetic smoke plan changed')
    official = '/home/chase/Yangxx/VF-CL/data/tiny-imagenet-200'
    if official in '\n'.join(payload['jobs']['synthetic_tinyimagenet']['command']):
        raise ValueError('synthetic smoke command references official held-out data')
    return payload


def plan_smoke(root):
    root = _validated_root(root)
    forbidden = [
        root / name for name in (
            'SMOKE_AUDIT.json', 'SMOKE_EXECUTION_SUCCESS',
            'EXECUTION_SUCCESS', 'GATE_SUCCESS', 'GATE_FAILED',
            'PRIMARY_GATE.json',
        )
    ]
    if any(path.exists() or path.is_symlink() for path in forbidden):
        raise ValueError('cannot create a retroactive synthetic smoke plan')
    payload = _smoke_plan_payload(root)
    _write_new_json(root / 'SMOKE_PLAN.json', payload)
    return _validate_smoke_plan(root, payload)


def _tiny_fixture_expected_paths():
    paths = []
    for class_id in range(4):
        name = f'n{class_id:08d}'
        count = 52
        paths.extend(
            f'tiny-imagenet-200/train/{name}/images/{name}_{index:03d}.JPEG'
            for index in range(count)
        )
        paths.append(f'tiny-imagenet-200/val/{name}/{name}_val.JPEG')
    return paths


def _validate_tiny_smoke_fixture(root):
    root = _validated_root(root)
    fixture = _within(root / 'synthetic_tiny_fixture', root, 'Tiny smoke fixture')
    manifest = _read_json(root / 'SYNTHETIC_TINY_FIXTURE.json')
    expected_paths = _tiny_fixture_expected_paths()
    if not isinstance(manifest, dict) or not (
            manifest.get('kind') == 'synthetic_tinyimagenet_smoke_fixture'
            and manifest.get('classes') == 4
            and manifest.get('active_classes') == [0, 1, 2, 3]
            and manifest.get('tasks') == [[0, 1], [2, 3]]
            and manifest.get('image_size') == [64, 64]
            and manifest.get('relative_paths') == expected_paths):
        raise ValueError('synthetic Tiny smoke fixture manifest changed')
    hashes = manifest.get('sha256')
    if not isinstance(hashes, dict) or set(hashes) != set(expected_paths):
        raise ValueError('synthetic Tiny smoke fixture hashes changed')
    if len(set(hashes.values())) != len(expected_paths):
        raise ValueError('synthetic Tiny smoke fixture contains duplicate images')
    discovered = []
    for path in fixture.rglob('*'):
        if path.is_symlink():
            raise ValueError('synthetic Tiny smoke fixture contains a symlink')
        if path.is_file():
            relative = path.relative_to(fixture).as_posix()
            discovered.append(relative)
    if sorted(discovered) != sorted(expected_paths):
        raise ValueError('synthetic Tiny smoke fixture paths changed')
    for relative in expected_paths:
        path = fixture / relative
        if not path.is_file() or file_sha256(path) != hashes[relative]:
            raise ValueError('synthetic Tiny smoke fixture content changed')
    return manifest


def prepare_tiny_smoke_fixture(root):
    root = _validated_root(root)
    fixture = _within(root / 'synthetic_tiny_fixture', root, 'Tiny smoke fixture')
    manifest_path = root / 'SYNTHETIC_TINY_FIXTURE.json'
    if manifest_path.exists() or manifest_path.is_symlink():
        return _validate_tiny_smoke_fixture(root)
    if fixture.exists() or fixture.is_symlink():
        raise ValueError('partial synthetic Tiny smoke fixture exists')
    for class_id in range(4):
        name = f'n{class_id:08d}'
        train = fixture / 'tiny-imagenet-200' / 'train' / name / 'images'
        validation = fixture / 'tiny-imagenet-200' / 'val' / name
        train.mkdir(parents=True)
        validation.mkdir(parents=True)
        count = 52
        for index in range(count + 1):
            code = class_id * (count + 1) + index
            image = Image.new('RGB', (64, 64))
            for bit in range(8):
                shade = 255 if code & (1 << bit) else 0
                image.paste((shade, shade, shade), (bit * 8, 0, (bit + 1) * 8, 64))
            destination = (
                train / f'{name}_{index:03d}.JPEG'
                if index < count else validation / f'{name}_val.JPEG'
            )
            image.save(destination, format='JPEG')
    expected_paths = _tiny_fixture_expected_paths()
    payload = {
        'schema_version': 1,
        'kind': 'synthetic_tinyimagenet_smoke_fixture',
        'classes': 4, 'active_classes': [0, 1, 2, 3],
        'tasks': [[0, 1], [2, 3]], 'image_size': [64, 64],
        'relative_paths': expected_paths,
        'sha256': {
            relative: file_sha256(fixture / relative)
            for relative in expected_paths
        },
    }
    _write_new_json(manifest_path, payload)
    return _validate_tiny_smoke_fixture(root)


def _retroactive_artifacts(root):
    if not root.exists():
        return []
    allowed = {
        'logs', 'launcher_claim', 'launcher_token', 'IMPLEMENTATION_COMMIT',
    }
    artifacts = []
    for path in root.iterdir():
        if path.name in allowed:
            continue
        if path.name == 'PRIMARY_PLAN.json':
            continue
        artifacts.append(path)
    return artifacts


def plan(root):
    root = _validated_root(root)
    plan_path = root / 'PRIMARY_PLAN.json'
    existing = _read_json(plan_path) if plan_path.exists() else None
    if existing is not None:
        if _retroactive_artifacts(root):
            # Existing immutable plan may coexist only with artifacts it authorized.
            return _validate_primary_plan(root, existing)
        return _validate_primary_plan(root, existing)
    if root.exists() and _retroactive_artifacts(root):
        raise ValueError('cannot create a retroactive primary plan')
    root = _validated_root(root, create=True)
    source = adaptive_source_identity()
    incumbents = {dataset: incumbent_evidence(dataset) for dataset in DATASETS}
    jobs = {}
    for dataset in DATASETS:
        results = root / 'primary' / dataset / 'outputs'
        command = build_command(dataset, '__DEVICE__', results)
        smoke_command = build_command(
            dataset, '__DEVICE__', root / 'smoke' / dataset / 'outputs',
            smoke=True,
        )
        jobs[dataset] = {
            'spec': f'{dataset}:adaptive:{SEED}',
            'dataset': dataset, 'seed': SEED,
            'command': command, 'command_sha256': _payload_sha256(command),
            'smoke_command': smoke_command,
            'smoke_command_sha256': _payload_sha256(smoke_command),
            'validation_hash': EXPECTED_VALIDATION_HASHES[dataset],
            'incumbent_record_sha256': incumbents[dataset]['record_sha256'],
        }
    payload = {
        'schema_version': 1, 'kind': 'adaptive_primary_plan',
        'seed': SEED, 'datasets': list(DATASETS),
        'source_commit': source['commit'], 'source_sha256': source['sha256'],
        'jobs': jobs, 'incumbents': incumbents,
        'metrics': list(METRICS), 'tolerance': str(TOLERANCE),
        'ablation_names': list(ABLATIONS),
    }
    _write_new_json(plan_path, payload)
    return payload


def _validate_primary_plan(root, payload=None):
    root = _validated_root(root)
    payload = _read_json(root / 'PRIMARY_PLAN.json') if payload is None else payload
    if not isinstance(payload, dict) or not (
        payload.get('schema_version') == 1
        and payload.get('kind') == 'adaptive_primary_plan'
        and payload.get('seed') == SEED
        and payload.get('datasets') == list(DATASETS)
        and payload.get('metrics') == list(METRICS)
        and payload.get('tolerance') == str(TOLERANCE)
        and payload.get('ablation_names') == list(ABLATIONS)
        and list(payload.get('jobs', {})) == list(DATASETS)
        and set(payload.get('incumbents', {})) == set(DATASETS)
    ):
        raise ValueError('primary plan does not match the frozen contract')
    if {'commit': payload.get('source_commit'),
            'sha256': payload.get('source_sha256')} != adaptive_source_identity():
        raise ValueError('primary plan source identity is not the reviewed tree')
    for dataset in DATASETS:
        job = payload['jobs'][dataset]
        command = build_command(
            dataset, '__DEVICE__', root / 'primary' / dataset / 'outputs',
        )
        smoke = build_command(
            dataset, '__DEVICE__', root / 'smoke' / dataset / 'outputs',
            smoke=True,
        )
        incumbent = payload['incumbents'][dataset]
        compact = {key: value for key, value in incumbent.items()
                   if key != 'record_sha256'}
        if not (
            job == {
                'spec': f'{dataset}:adaptive:{SEED}',
                'dataset': dataset, 'seed': SEED,
                'command': command, 'command_sha256': _payload_sha256(command),
                'smoke_command': smoke,
                'smoke_command_sha256': _payload_sha256(smoke),
                'validation_hash': EXPECTED_VALIDATION_HASHES[dataset],
                'incumbent_record_sha256': incumbent.get('record_sha256'),
            }
            and incumbent.get('dataset') == dataset
            and incumbent.get('cell') == INCUMBENT_CELLS[dataset]
            and incumbent.get('seed') == SEED
            and incumbent.get('validation_hash')
            == EXPECTED_VALIDATION_HASHES[dataset]
            and incumbent.get('passed') is True
            and incumbent.get('record_sha256') == _payload_sha256(compact)
        ):
            raise ValueError(f'primary plan mismatch for {dataset}')
        if incumbent != incumbent_evidence(dataset):
            raise ValueError(f'factorial incumbent changed for {dataset}')
    return payload


def _terminal_evidence(root):
    root = _validated_root(root)
    primary = _validate_primary_plan(root)
    marker_names = [
        name for name in ('GATE_SUCCESS', 'GATE_FAILED')
        if (root / name).exists() or (root / name).is_symlink()
    ]
    if len(marker_names) != 1:
        raise ValueError('terminal evidence must contain exactly one gate marker')
    marker_name = marker_names[0]
    marker = _read_json(root / marker_name)
    execution = _read_json(root / 'EXECUTION_SUCCESS')
    report_path = root / 'PRIMARY_GATE.json'
    report = _read_json(report_path)
    if None in (marker, execution, report):
        raise ValueError('terminal evidence is incomplete or malformed')
    common = {
        'primary_plan_sha256': file_sha256(root / 'PRIMARY_PLAN.json'),
        'gate_report_sha256': file_sha256(report_path),
    }
    if (marker != {'status': marker_name, **common}
            or execution != {'status': 'EXECUTION_SUCCESS', **common}
            or report.get('status') != marker_name):
        raise ValueError('terminal marker/report identities disagree')
    records = {dataset: _strict_record(root, dataset) for dataset in DATASETS}
    passed, expected_report = _primary_gate_report(root, primary, records)
    expected_status = 'GATE_SUCCESS' if passed else 'GATE_FAILED'
    if marker_name != expected_status or report != expected_report:
        raise ValueError('terminal report differs from current audited gate')
    return marker_name, report


def _valid_success_marker(root):
    try:
        status, _ = _terminal_evidence(root)
        return status == 'GATE_SUCCESS'
    except (OSError, ValueError):
        return False


def _validate_ablation_plan(root, payload=None):
    root = _validated_root(root)
    if not _valid_success_marker(root):
        raise ValueError('ablation plan lacks intact primary success')
    path = root / 'ablations' / 'ABLATION_PLAN.json'
    payload = _read_json(path) if payload is None else payload
    if not isinstance(payload, dict) or not (
        payload.get('schema_version') == 1
        and payload.get('kind') == 'adaptive_ablation_plan'
        and payload.get('primary_plan_sha256')
        == file_sha256(root / 'PRIMARY_PLAN.json')
        and payload.get('primary_gate_sha256')
        == file_sha256(root / 'GATE_SUCCESS')
    ):
        raise ValueError('ablation plan identity mismatch')
    expected = {}
    for ablation in ABLATIONS:
        for dataset in DATASETS:
            key = f'{dataset}:{ablation}:{SEED}'
            command = build_command(
                dataset, '__DEVICE__',
                root / 'ablations' / ablation / dataset / 'outputs',
                ablation=ablation,
            )
            expected[key] = {
                'spec': key, 'dataset': dataset, 'seed': SEED,
                'ablation': ablation, 'command': command,
                'command_sha256': _payload_sha256(command),
                'primary_gate_sha256': file_sha256(root / 'GATE_SUCCESS'),
            }
    if payload.get('jobs') != expected:
        raise ValueError('ablation jobs differ from the pre-registered matrix')
    return payload


def plan_ablations(root):
    root = _validated_root(root)
    primary = _validate_primary_plan(root)
    if not _valid_success_marker(root) or (root / 'GATE_FAILED').exists():
        raise ValueError('ablations are illegal before an intact primary GATE_SUCCESS')
    ablation_root = root / 'ablations'
    path = ablation_root / 'ABLATION_PLAN.json'
    if path.exists():
        payload = _read_json(path)
        if payload is None:
            raise ValueError('malformed immutable ablation plan')
        return _validate_ablation_plan(root, payload)
    jobs = {}
    for ablation in ABLATIONS:
        for dataset in DATASETS:
            key = f'{dataset}:{ablation}:{SEED}'
            results = ablation_root / ablation / dataset / 'outputs'
            command = build_command(
                dataset, '__DEVICE__', results, ablation=ablation,
            )
            jobs[key] = {
                'spec': key, 'dataset': dataset, 'seed': SEED,
                'ablation': ablation, 'command': command,
                'command_sha256': _payload_sha256(command),
                'primary_gate_sha256': file_sha256(root / 'GATE_SUCCESS'),
            }
    payload = {
        'schema_version': 1, 'kind': 'adaptive_ablation_plan',
        'primary_plan_sha256': file_sha256(root / 'PRIMARY_PLAN.json'),
        'primary_gate_sha256': file_sha256(root / 'GATE_SUCCESS'),
        'jobs': jobs,
    }
    _write_new_json(path, payload)
    return _validate_ablation_plan(root, payload)


def _job_context(root, dataset, smoke=False, ablation=None):
    root = _validated_root(root)
    if smoke:
        smoke_plan = _validate_smoke_plan(root)
        return root / 'smoke' / dataset, smoke_plan['jobs'][dataset]['command']
    primary = _validate_primary_plan(root)
    if ablation is None:
        return root / 'primary' / dataset, primary['jobs'][dataset]['command']
    payload = _validate_ablation_plan(root)
    key = f'{dataset}:{ablation}:{SEED}'
    if key not in payload['jobs']:
        raise ValueError('ablation job is not pre-authorized')
    return root / 'ablations' / ablation / dataset, payload['jobs'][key]['command']


def bound_job_plan(root, dataset, device, smoke=False, ablation=None):
    validate_job(dataset, SEED)
    job_root, template = _job_context(root, dataset, smoke, ablation)
    plan_path = job_root / 'planned_protocol.json'
    if not plan_path.exists() and (job_root / 'outputs').exists() and any(
            (job_root / 'outputs').iterdir()):
        raise ValueError('completed or partial output exists without a prelaunch plan')
    output = job_root / 'outputs'
    command = build_command(
        dataset, device, output, smoke=smoke, ablation=ablation,
    )
    expected = [str(device) if value == '__DEVICE__' else value for value in template]
    if command != expected:
        raise ValueError('bound command differs from immutable command template')
    source = adaptive_source_identity()
    active_plan = (_validate_smoke_plan(root) if smoke
                   else _validate_primary_plan(root))
    if source != {
            'commit': active_plan['source_commit'],
            'sha256': active_plan['source_sha256']}:
        raise ValueError('implementation changed after the active plan')
    payload = {
        'schema_version': 1, 'kind': 'adaptive_bound_job_plan',
        'spec': (
            f'{dataset}:smoke:{SEED}' if smoke
            else f'{dataset}:{ablation or "adaptive"}:{SEED}'
        ),
        'dataset': dataset, 'seed': SEED, 'smoke': bool(smoke),
        'ablation': ablation, 'device': str(device), 'command': command,
        'command_sha256': _payload_sha256(command),
    }
    plan_name = 'SMOKE_PLAN.json' if smoke else 'PRIMARY_PLAN.json'
    payload['smoke_plan_sha256' if smoke else 'primary_plan_sha256'] = \
        file_sha256(_validated_root(root) / plan_name)
    _write_new_json(plan_path, payload)
    return payload


def _gate_metrics(metrics):
    if not isinstance(metrics, dict) or not set(METRICS).issubset(metrics):
        raise ValueError('record is missing a required gate metric')
    if not all(
        isinstance(metrics[name], (int, float))
        and not isinstance(metrics[name], bool)
        and math.isfinite(metrics[name]) and -1 <= metrics[name] <= 1
        for name in METRICS
    ):
        raise ValueError('gate metrics must be finite values in [-1, 1]')
    return {name: metrics[name] for name in METRICS}


def _evidence_paths(run):
    return {
        'config': run / 'config.json',
        'results': run / 'results.json',
        'adaptive_freeze': run / 'ADAPTIVE_STATE_FROZEN.json',
        'adaptive_checkpoint': run / 'adaptive_final.pt',
        'data_flow_audit': run / 'data_flow_audit.jsonl',
    }


def _read_only_deferred_dataset(dataset, args):
    args.lambda_validation_enabled = 0
    args.bic_enabled = 0
    args.data_flow_audit = 0
    args.head_consolidation_enabled = 0
    if dataset != 'cifar100':
        if args.data != 'tabvfl':
            raise ValueError('vector deferred dataset identity mismatch')
        return VFLDataset(args)
    if args.data != 'cifar100':
        raise ValueError('CIFAR-100 deferred dataset identity mismatch')
    evaluation = object.__new__(VFLDataset)
    evaluation.args = args
    transform = vision_transforms.Compose([
        vision_transforms.ToTensor(),
        vision_transforms.Normalize([.507, .487, .441], [.267, .256, .276]),
    ])
    evaluation.testset = vision_datasets.CIFAR100(
        args.data_path, train=False, download=False, transform=transform,
    )
    evaluation.calibration_indices = set()
    evaluation.validation_indices = set()
    return evaluation


def _deferred_gate_metrics(dataset, run, config, freeze, ablation=None):
    snapshots = freeze.get('snapshots')
    if type(snapshots) is not list or not snapshots:
        raise ValueError('adaptive freeze lacks deferred CIL snapshots')
    snapshot_paths = []
    for record in snapshots:
        relative = Path(record.get('path', '')) if type(record) is dict else Path()
        if (relative.is_absolute() or not relative.name
                or '..' in relative.parts):
            raise ValueError('deferred snapshot path is invalid')
        snapshot_paths.append(run / relative)
    seen = snapshots[-1].get('seen_task_classes')
    if type(seen) is not dict or not seen:
        raise ValueError('deferred task-class evidence is missing')
    try:
        task_classes = {
            int(task_id): [int(class_id) for class_id in classes]
            for task_id, classes in seen.items()
        }
    except (TypeError, ValueError) as error:
        raise ValueError('deferred task-class evidence is invalid') from error
    if (any(type(classes) is not list or not classes for classes in seen.values())
            or any(type(class_id) is not int
                   for classes in seen.values() for class_id in classes)):
        raise ValueError('deferred task-class evidence is invalid')
    args = SimpleNamespace(**config)
    evaluation_dataset = _read_only_deferred_dataset(dataset, args)
    with (_ablation_runtime(ablation)
          if ablation is not None else contextlib.nullcontext()):
        deferred = evaluate_deferred_cil_trajectory(
            snapshot_paths, run / 'adaptive_final.pt', evaluation_dataset,
            task_classes, args,
        )
    if type(deferred) is not dict:
        raise ValueError('deferred evaluation returned malformed evidence')
    return _gate_metrics(deferred.get('cl_metrics'))


def _copy_resume_checkpoints(source, target):
    source = _reject_symlink_components(source)
    target = Path(target)
    if source.is_symlink() or not source.is_dir():
        raise ValueError('resume checkpoint source is not a regular directory')
    entries = list(source.rglob('*'))
    if not entries or any(
            entry.is_symlink()
            or not (entry.is_dir() or entry.is_file()) for entry in entries):
        raise ValueError('resume checkpoint tree contains a symlink or non-regular entry')
    if target.exists() or target.is_symlink():
        raise ValueError('resume checkpoint scratch target already exists')
    shutil.copytree(source, target)


def _strict_resume_probe_main(run_dir):
    """Exercise runner checkpoint admission and strict state loading in isolation."""
    from bic_calibration import TaskAffineCalibrator
    from cl_methods import get_cl_method
    from data_utils import TaskManager, VFLDataset
    from metrics import MetricsTracker
    from models import build_models
    from runner import _load_resume_checkpoint
    from vfl_trainer import VFLTrainer

    run = _reject_symlink_components(run_dir)
    config = _read_json(run / 'config.json')
    if not isinstance(config, dict):
        raise ValueError('resume probe config is malformed')
    args = SimpleNamespace(**config)
    with tempfile.TemporaryDirectory(prefix='adaptive_resume_probe_') as scratch:
        args.output_dir = scratch
        args.resume_run_dir = str(run)
        dataset = VFLDataset(args)
        _copy_resume_checkpoints(
            run / 'checkpoints', Path(scratch) / 'checkpoints',
        )
        task_manager = TaskManager(args)
        bottoms, top = build_models(args)
        trainer = VFLTrainer(bottoms, top, args)
        trainer.dataset_ref = dataset
        method = get_cl_method(args.cl_method, trainer, args)
        start, seen, _ = _load_resume_checkpoint(
            args, trainer, method, task_manager, MetricsTracker(),
            TaskAffineCalibrator(),
        )
        timeline = task_manager.get_timeline()
        expected_tasks = sorted(
            int(event['task_id']) for event in timeline if event['type'] == 'CIL'
        )
        if start != len(timeline) or sorted(seen) != expected_tasks:
            raise ValueError('resume probe did not restore the completed timeline')
        print('RESUME_PROBE_SUCCESS')


def _run_resume_probe(run_dir):
    wrapper = (
        'import sys; from adaptive_dual_branch_validation import '
        '_strict_resume_probe_main; _strict_resume_probe_main(sys.argv[1])'
    )
    completed = subprocess.run(
        [str(PYTHON), '-c', wrapper, str(run_dir)], cwd=WORKTREE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if (completed.returncode != 0
            or 'RESUME_PROBE_SUCCESS' not in completed.stdout.splitlines()):
        raise ValueError('checkpoint resume probe failed: ' + completed.stdout[-1000:])
    return True


def _standard_smoke_checks(freeze, audited, resume_verified):
    data_flow = freeze.get('data_flow', {}) if isinstance(freeze, dict) else {}
    hashes = freeze.get('result', {}).get('candidate_hashes', {}) \
        if isinstance(freeze, dict) else {}
    return {
        'branch_isolation': (
            set(hashes) == {'pre', 'full', 'bias'}
            and all(re.fullmatch(r'[0-9a-f]{64}', str(value))
                    for value in hashes.values())
        ),
        'validation_after_freeze': (
            data_flow.get('candidates_frozen_before_validation') is True
            and data_flow.get('validation_before_freeze') is False
        ),
        'exact_reload': audited == freeze,
        'test_after_install': data_flow.get('test_before_install') is False,
        'adaptive_audit': (
            freeze.get('status') == 'ADAPTIVE_STATE_FROZEN'
            and data_flow.get('test_used_for_diagnostics') is False
        ),
        'resume': resume_verified is True,
    }


def audit_completed_run(dataset, command, run_dir, root, ablation=None,
                        smoke=False):
    validate_job(dataset, SEED)
    root = _validated_root(root)
    job_root, _ = _job_context(root, dataset, smoke, ablation)
    run = _reject_symlink_components(run_dir)
    output = job_root / 'outputs'
    flags = _command_flags(command)
    expected_run_name = re.compile(
        re.escape(flags.get('--exp_name', ''))
        + r'_[0-9]{8}_[0-9]{6}'
    )
    if (run.parent != output.absolute()
            or expected_run_name.fullmatch(run.name) is None):
        raise ValueError('adaptive run is outside its exact job output root')
    plan_path = job_root / 'planned_protocol.json'
    launch_path = job_root / 'launch_started.json'
    planned = _read_json(plan_path)
    launched = _read_json(launch_path)
    if planned is None or launched is None or not (
        planned.get('command') == command
        and planned.get('command_sha256') == _payload_sha256(command)
        and launched == {
            'kind': 'adaptive_launch_started',
            'spec': planned.get('spec'), 'command': command,
            'planned_protocol_sha256': file_sha256(plan_path),
        }
    ):
        raise ValueError('missing or mismatched prelaunch command evidence')
    expected = build_command(
        dataset, planned.get('device'), output, smoke=smoke, ablation=ablation,
    )
    if command != expected:
        raise ValueError('completed command differs from frozen protocol')
    paths = _evidence_paths(run)
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise ValueError('completed adaptive evidence is missing or symlinked')
    planned_details = plan_path.stat()
    launch_details = launch_path.stat()
    for path in paths.values():
        details = path.stat()
        if (planned_details.st_mtime_ns > details.st_mtime_ns
                or launch_details.st_mtime_ns > details.st_mtime_ns
                or planned_details.st_ctime_ns > details.st_ctime_ns
                or launch_details.st_ctime_ns > details.st_ctime_ns):
            raise ValueError('retroactive plan or launch evidence is rejected')
    config = _read_json(paths['config'])
    results = _read_json(paths['results'])
    freeze = _read_json(paths['adaptive_freeze'])
    if None in (config, results, freeze):
        raise ValueError('malformed completed adaptive JSON evidence')
    config_checks = {
        option[2:]: config.get(option[2:]) == config_value(option, value)
        for option, value in flags.items()
    }
    with (_ablation_runtime(ablation)
          if ablation is not None else contextlib.nullcontext()):
        audited = audit_adaptive_checkpoint(run, freeze.get('audit_spec'))
    gate = freeze.get('result', {}).get('gate', {})
    expected_rule = ABLATIONS.get(ablation, 'class_balanced')
    data_flow = freeze.get('data_flow')
    diagnostics = freeze.get('diagnostics', {})
    source_splits = diagnostics.get('source_splits', {})
    primary_gate = ablation is None
    active_plan = (_validate_smoke_plan(root) if smoke
                   else _validate_primary_plan(root))
    validation_hash = freeze.get('validation', {}).get(
        'manifest', {}).get('sha256')
    prefix_count = data_flow.get('audit_prefix_record_count') \
        if isinstance(data_flow, dict) else None
    prefix_sha256 = data_flow.get('audit_prefix_sha256') \
        if isinstance(data_flow, dict) else None
    base_data_flow = {
        key: data_flow.get(key) for key in (
            'candidates_frozen_before_validation', 'validation_before_freeze',
            'test_before_install', 'test_used_for_diagnostics', 'solver_input',
        )
    } if isinstance(data_flow, dict) else None
    checks = {
        'exact_command_config': (
            bool(config_checks) and all(config_checks.values())
            and config.get('results_dir') == str(output)
            and config.get('exp_name') == flags.get('--exp_name')
            and config.get('output_dir') == str(run)
        ),
        'task7_adaptive_reaudit': audited == freeze,
        'source_commit': freeze.get('source', {}).get('source_commit')
        == active_plan['source_commit'],
        'checkpoint_audit': (
            freeze.get('status') == 'ADAPTIVE_STATE_FROZEN'
            and freeze.get('checkpoint', {}).get('path') == 'adaptive_final.pt'
        ),
        'adaptive_gate': (
            gate.get('gate_rule') == expected_rule
            and gate.get('is_primary') is primary_gate
            and gate.get('tolerance') == 1e-12
            and gate.get('max_iterations') == 80
        ),
        'solver_audit': (
            diagnostics.get('solver_input') == (
                'class_balanced_validation_nll' if ablation is None
                else _ablation_solver_input(ablation)
            )
            and (
                'ablation_solver_nlls' not in diagnostics
                if ablation is None else
                diagnostics.get('ablation_solver_nlls') == {
                    'full': gate.get('full_branch_nll'),
                    'bias': gate.get('bias_branch_nll'),
                    'mixture': gate.get('mixture_nll'),
                }
            )
        ),
        'privacy_audit': (
            isinstance(freeze.get('replay'), dict)
            and freeze['replay'].get('count', 0) > 0
            and source_splits == {
                'replay': 'persistent_training_replay',
                'validation': 'frozen_training_validation',
                'test_used': False,
            }
        ),
        'data_flow_audit': (
            set(data_flow or {}) == {
                'candidates_frozen_before_validation',
                'validation_before_freeze', 'test_before_install',
                'test_used_for_diagnostics', 'solver_input',
                'audit_prefix_record_count', 'audit_prefix_sha256',
            }
            and base_data_flow == {
            'candidates_frozen_before_validation': True,
            'validation_before_freeze': False,
            'test_before_install': False,
            'test_used_for_diagnostics': False,
            'solver_input': 'class_balanced_validation_nll',
            }
            and isinstance(prefix_count, int) and not isinstance(prefix_count, bool)
            and prefix_count >= 0
            and isinstance(prefix_sha256, str)
            and re.fullmatch(r'[0-9a-f]{64}', prefix_sha256) is not None
        ),
        'validation_hash': (
            isinstance(validation_hash, str)
            and re.fullmatch(r'[0-9a-f]{64}', validation_hash) is not None
            and (smoke or validation_hash == EXPECTED_VALIDATION_HASHES[dataset])
        ),
    }
    if not all(checks.values()):
        raise ValueError(f'adaptive completed-run audit failed: {checks}')
    metrics = _gate_metrics(results.get('cl_metrics'))
    if metrics != _deferred_gate_metrics(
            dataset, run, config, freeze, ablation=ablation):
        raise ValueError(
            'results gate metrics differ from post-freeze checkpoint evaluation'
        )
    if smoke:
        checks.update(_standard_smoke_checks(
            freeze, audited, _run_resume_probe(run)
        ))
        if not all(checks.values()):
            raise ValueError(f'adaptive smoke audit failed: {checks}')
    return {
        'schema_version': 1,
        'spec': planned['spec'], 'dataset': dataset, 'seed': SEED,
        'ablation': ablation, 'smoke': bool(smoke),
        'command': command,
        'implementation_commit': active_plan['source_commit'],
        'validation_hash': (
            validation_hash if smoke else EXPECTED_VALIDATION_HASHES[dataset]
        ),
        'run_dir': str(run),
        'paths': {name: str(path) for name, path in paths.items()},
        'sha256': {name: file_sha256(path) for name, path in paths.items()},
        'metrics': metrics, 'checks': checks, 'passed': True,
    }


def _discover_run(job_root, dataset):
    output = job_root / 'outputs'
    if output.is_symlink() or not output.is_dir():
        return None
    candidates = [
        path for path in output.iterdir()
        if path.is_dir() and not path.is_symlink()
        and (path / 'results.json').is_file()
        and (path / 'ADAPTIVE_STATE_FROZEN.json').is_file()
    ]
    if len(candidates) != 1:
        if candidates:
            raise ValueError(f'multiple completed adaptive runs for {dataset}')
        return None
    return candidates[0]


def _install_record(root, dataset, ablation=None, smoke=False):
    job_root, _ = _job_context(root, dataset, smoke, ablation)
    planned = _read_json(job_root / 'planned_protocol.json')
    if planned is None:
        return False
    run = _discover_run(job_root, dataset)
    if run is None:
        return False
    record = audit_completed_run(
        dataset, planned['command'], run, root, ablation=ablation, smoke=smoke,
    )
    _replace_json(job_root / 'record.json', record)
    _safe_unlink(job_root / 'FAILED.json')
    _touch_new(job_root / 'SUCCESS')
    return True


def _tiny_smoke_job_root(root):
    return _validated_root(root) / 'smoke' / 'synthetic_tinyimagenet'


def _bound_tiny_smoke_plan(root, device):
    root = _validated_root(root)
    smoke = _validate_smoke_plan(root)
    job_root = _tiny_smoke_job_root(root)
    command = build_tiny_synthetic_smoke_command(root, device)
    template = smoke['jobs']['synthetic_tinyimagenet']['command']
    expected = [str(device) if value == '__DEVICE__' else value
                for value in template]
    if command != expected:
        raise ValueError('bound synthetic Tiny command changed')
    fixture = prepare_tiny_smoke_fixture(root)
    payload = {
        'schema_version': 1, 'kind': 'adaptive_tiny_synthetic_smoke_job',
        'dataset': 'synthetic_tinyimagenet', 'seed': SEED, 'smoke': True,
        'device': str(device), 'command': command,
        'command_sha256': _payload_sha256(command),
        'smoke_plan_sha256': file_sha256(root / 'SMOKE_PLAN.json'),
        'fixture_manifest_sha256': _payload_sha256(fixture),
    }
    _write_new_json(job_root / 'planned_protocol.json', payload)
    return payload


def _discover_tiny_smoke_run(root):
    output = _tiny_smoke_job_root(root) / 'outputs'
    prefix = 'adaptive_synthetic_tinyimagenet_smoke_seed42_'
    if output.is_symlink() or not output.is_dir():
        return None
    candidates = [
        path for path in output.iterdir()
        if path.is_dir() and not path.is_symlink()
        and path.name.startswith(prefix)
        and (path / 'results.json').is_file()
        and (path / 'ADAPTIVE_STATE_FROZEN.json').is_file()
    ]
    if len(candidates) > 1:
        raise ValueError('multiple synthetic Tiny smoke runs exist')
    return candidates[0] if candidates else None


def _audit_tiny_smoke_run(root, run):
    root = _validated_root(root)
    smoke = _validate_smoke_plan(root)
    fixture = _validate_tiny_smoke_fixture(root)
    job_root = _tiny_smoke_job_root(root)
    planned = _read_json(job_root / 'planned_protocol.json')
    launched = _read_json(job_root / 'launch_started.json')
    if planned is None or launched != {
            'kind': 'adaptive_tiny_synthetic_smoke_launch',
            'command': planned.get('command'),
            'planned_protocol_sha256': file_sha256(
                job_root / 'planned_protocol.json')}:
        raise ValueError('synthetic Tiny smoke prelaunch evidence changed')
    expected = build_tiny_synthetic_smoke_command(root, planned.get('device'))
    if (planned.get('command') != expected
            or planned.get('command_sha256') != _payload_sha256(expected)
            or planned.get('fixture_manifest_sha256') != _payload_sha256(fixture)
            or planned.get('smoke_plan_sha256')
            != file_sha256(root / 'SMOKE_PLAN.json')):
        raise ValueError('synthetic Tiny smoke plan changed')
    run = _reject_symlink_components(run)
    output = job_root / 'outputs'
    if run.parent != output.absolute() or run.is_symlink():
        raise ValueError('synthetic Tiny smoke run escaped its output root')
    paths = {
        'config': run / 'config.json', 'results': run / 'results.json',
        'adaptive_freeze': run / 'ADAPTIVE_STATE_FROZEN.json',
        'adaptive_checkpoint': run / 'adaptive_final.pt',
        'data_flow_audit': run / 'data_flow_audit.jsonl',
    }
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise ValueError('synthetic Tiny smoke evidence is incomplete')
    planned_details = (job_root / 'planned_protocol.json').stat()
    launch_details = (job_root / 'launch_started.json').stat()
    if any(
            planned_details.st_mtime_ns > path.stat().st_mtime_ns
            or launch_details.st_mtime_ns > path.stat().st_mtime_ns
            or planned_details.st_ctime_ns > path.stat().st_ctime_ns
            or launch_details.st_ctime_ns > path.stat().st_ctime_ns
            for path in paths.values()):
        raise ValueError('synthetic Tiny smoke provenance is retroactive')
    config = _read_json(paths['config'])
    results = _read_json(paths['results'])
    freeze = _read_json(paths['adaptive_freeze'])
    if None in (config, results, freeze):
        raise ValueError('synthetic Tiny smoke JSON evidence is malformed')
    flags = _command_flags(expected)
    config_checks = {
        option[2:]: config.get(option[2:]) == config_value(option, value)
        for option, value in flags.items()
    }
    audited = audit_adaptive_checkpoint(run, freeze.get('audit_spec'))
    data_flow = freeze.get('data_flow', {})
    candidate_hashes = freeze.get('result', {}).get('candidate_hashes', {})
    checks = {
        'branch_isolation': (
            set(candidate_hashes) == {'pre', 'full', 'bias'}
            and all(re.fullmatch(r'[0-9a-f]{64}', str(value))
                    for value in candidate_hashes.values())
        ),
        'validation_after_freeze': (
            data_flow.get('candidates_frozen_before_validation') is True
            and data_flow.get('validation_before_freeze') is False
        ),
        'exact_reload': audited == freeze,
        'test_after_install': data_flow.get('test_before_install') is False,
        'adaptive_audit': (
            freeze.get('status') == 'ADAPTIVE_STATE_FROZEN'
            and freeze.get('checkpoint', {}).get('path') == 'adaptive_final.pt'
            and data_flow.get('test_used_for_diagnostics') is False
        ),
        'resume': _run_resume_probe(run),
    }
    if (not config_checks or not all(config_checks.values())
            or config.get('output_dir') != str(run)
            or config.get('results_dir') != str(output)
            or not isinstance(results.get('cl_metrics'), dict)
            or not all(checks.values())
            or smoke['source_commit'] != freeze.get('source', {}).get(
                'source_commit')):
        raise ValueError(f'synthetic Tiny smoke audit failed: {checks}')
    return {
        'schema_version': 1, 'dataset': 'synthetic_tinyimagenet',
        'seed': SEED, 'smoke': True, 'command': expected,
        'implementation_commit': smoke['source_commit'],
        'run_dir': str(run), 'fixture_manifest_sha256': _payload_sha256(fixture),
        'paths': {name: str(path) for name, path in paths.items()},
        'sha256': {name: file_sha256(path) for name, path in paths.items()},
        'checks': checks, 'passed': True,
    }


def _install_tiny_smoke_record(root):
    job_root = _tiny_smoke_job_root(root)
    run = _discover_tiny_smoke_run(root)
    if run is None:
        return False
    record = _audit_tiny_smoke_run(root, run)
    _replace_json(job_root / 'record.json', record)
    _safe_unlink(job_root / 'FAILED.json')
    _touch_new(job_root / 'SUCCESS')
    return True


def run_tiny_smoke(root, device):
    root = _validated_root(root)
    job_root = _tiny_smoke_job_root(root)
    if (job_root / 'SUCCESS').is_file():
        return 0 if _install_tiny_smoke_record(root) else AUDIT_INCOMPLETE
    planned = _bound_tiny_smoke_plan(root, device)
    launch = {
        'kind': 'adaptive_tiny_synthetic_smoke_launch',
        'command': planned['command'],
        'planned_protocol_sha256': file_sha256(
            job_root / 'planned_protocol.json'),
    }
    _write_new_json(job_root / 'launch_started.json', launch)
    job_root.mkdir(parents=True, exist_ok=True)
    with (job_root / 'job.log').open('ab') as log:
        completed = subprocess.run(
            planned['command'], cwd=WORKTREE, env=os.environ.copy(),
            stdout=log, stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        _replace_json(job_root / 'FAILED.json', {
            'dataset': 'synthetic_tinyimagenet', 'seed': SEED,
            'smoke': True, 'returncode': completed.returncode,
            'command': planned['command'],
        })
        return completed.returncode
    try:
        if not _install_tiny_smoke_record(root):
            raise ValueError('zero exit produced no synthetic Tiny smoke run')
    except (OSError, ValueError) as error:
        _replace_json(job_root / 'FAILED.json', {
            'dataset': 'synthetic_tinyimagenet', 'seed': SEED,
            'smoke': True, 'returncode': 91, 'reason': str(error),
        })
        return 91
    return 0


def _smoke_records(root):
    root = _validated_root(root)
    records = {}
    for dataset in DATASETS:
        if not _install_record(root, dataset, smoke=True):
            raise ValueError(f'incomplete synthetic smoke: {dataset}')
        record = _read_json(root / 'smoke' / dataset / 'record.json')
        if record is None or record.get('smoke') is not True:
            raise ValueError(f'invalid synthetic smoke record: {dataset}')
        records[dataset] = record
    if not _install_tiny_smoke_record(root):
        raise ValueError('incomplete synthetic smoke: synthetic_tinyimagenet')
    record = _read_json(
        root / 'smoke' / 'synthetic_tinyimagenet' / 'record.json'
    )
    if record is None or record.get('smoke') is not True:
        raise ValueError('invalid synthetic Tiny smoke record')
    records['synthetic_tinyimagenet'] = record
    return records


def audit_smoke(root):
    root = _validated_root(root)
    plan_payload = _validate_smoke_plan(root)
    scientific = (
        'EXECUTION_SUCCESS', 'GATE_SUCCESS', 'GATE_FAILED',
        'PRIMARY_GATE.json',
    )
    if any((root / name).exists() or (root / name).is_symlink()
           for name in scientific):
        raise ValueError('scientific gate evidence is forbidden in smoke mode')
    records = _smoke_records(root)
    required_checks = {
        'branch_isolation', 'validation_after_freeze', 'exact_reload',
        'test_after_install', 'adaptive_audit', 'resume',
    }
    if set(records) != set(SMOKE_DATASETS) or any(
            record.get('smoke') is not True
            or record.get('passed') is not True
            or not required_checks.issubset(record.get('checks', {}))
            or not all(record['checks'][name] is True for name in required_checks)
            for record in records.values()):
        raise ValueError('synthetic smoke record failed its audit contract')
    report = {
        'schema_version': 1, 'status': 'SMOKE_EXECUTION_SUCCESS',
        'scientific_gate': False,
        'smoke_plan_sha256': file_sha256(root / 'SMOKE_PLAN.json'),
        'implementation_commit': plan_payload['source_commit'],
        'records': {
            name: {'sha256': _payload_sha256(record), 'passed': True}
            for name, record in records.items()
        },
    }
    _write_new_json(root / 'SMOKE_AUDIT.json', report)
    marker = {
        'status': 'SMOKE_EXECUTION_SUCCESS', 'scientific_gate': False,
        'report_sha256': file_sha256(root / 'SMOKE_AUDIT.json'),
        'smoke_plan_sha256': report['smoke_plan_sha256'],
    }
    _write_new_json(root / 'SMOKE_EXECUTION_SUCCESS', marker)
    return report


def audit(root):
    root = _validated_root(root)
    _validate_primary_plan(root)
    complete = True
    for dataset in DATASETS:
        try:
            complete = _install_record(root, dataset) and complete
        except (OSError, ValueError):
            complete = False
    ablation_plan = _read_json(root / 'ablations' / 'ABLATION_PLAN.json')
    if ablation_plan is not None:
        _validate_ablation_plan(root, ablation_plan)
        for ablation in ABLATIONS:
            for dataset in DATASETS:
                try:
                    _install_record(root, dataset, ablation=ablation)
                except (OSError, ValueError):
                    pass
    return 0 if complete else AUDIT_INCOMPLETE


def _run_one(root, dataset, device, smoke=False, ablation=None):
    job_root, _ = _job_context(root, dataset, smoke, ablation)
    if (job_root / 'SUCCESS').is_file():
        return 0 if _install_record(
            root, dataset, ablation=ablation, smoke=smoke,
        ) else AUDIT_INCOMPLETE
    planned = bound_job_plan(
        root, dataset, device, smoke=smoke, ablation=ablation,
    )
    launch_path = job_root / 'launch_started.json'
    launch = {
        'kind': 'adaptive_launch_started', 'spec': planned['spec'],
        'command': planned['command'],
        'planned_protocol_sha256': file_sha256(
            job_root / 'planned_protocol.json'
        ),
    }
    _write_new_json(launch_path, launch)
    environment = os.environ.copy()
    if ablation is not None:
        environment['VFCL_REVIEWED_ADAPTIVE_ABLATION'] = '1'
    job_root.mkdir(parents=True, exist_ok=True)
    log_path = job_root / 'job.log'
    if log_path.is_symlink():
        raise ValueError('symlinked job log is not allowed')
    with log_path.open('ab') as log:
        completed = subprocess.run(
            planned['command'], cwd=WORKTREE, env=environment,
            stdout=log, stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        _replace_json(job_root / 'FAILED.json', {
            'spec': planned['spec'], 'returncode': completed.returncode,
            'command': planned['command'],
        })
        return completed.returncode
    try:
        if not _install_record(
                root, dataset, ablation=ablation, smoke=smoke):
            raise ValueError('zero exit produced no auditable completed run')
    except (OSError, ValueError) as error:
        _replace_json(job_root / 'FAILED.json', {
            'spec': planned['spec'], 'returncode': 91,
            'command': planned['command'], 'reason': str(error),
        })
        return 91
    return 0


def run_job(root, dataset, seed, device, smoke=False):
    validate_job(dataset, seed)
    root = _validated_root(root)
    if smoke:
        return _run_one(root, dataset, device, smoke=True)
    ablation_plan = root / 'ablations' / 'ABLATION_PLAN.json'
    if ablation_plan.exists():
        if not _valid_success_marker(root):
            raise ValueError('ablation plan exists without intact primary success')
        for ablation in ABLATIONS:
            returncode = _run_one(
                root, dataset, device, ablation=ablation,
            )
            if returncode:
                return returncode
        return 0
    if (root / 'EXECUTION_SUCCESS').exists() or (root / 'GATE_FAILED').exists():
        raise ValueError('primary execution is terminal and cannot restart')
    return _run_one(root, dataset, device)


def _strict_record(root, dataset):
    path = root / 'primary' / dataset / 'record.json'
    success = path.parent / 'SUCCESS'
    record = _read_json(path)
    if record is None or success.is_symlink() or not success.is_file():
        raise ValueError(f'missing primary record for {dataset}')
    checks = record.get('checks')
    if not (
        record.get('spec') == f'{dataset}:adaptive:{SEED}'
        and record.get('dataset') == dataset and record.get('seed') == SEED
        and record.get('ablation') is None and record.get('smoke') is False
        and record.get('validation_hash') == EXPECTED_VALIDATION_HASHES[dataset]
        and record.get('passed') is True
        and isinstance(checks, dict) and checks and all(checks.values())
    ):
        raise ValueError(f'invalid primary record for {dataset}')
    _gate_metrics(record.get('metrics'))
    try:
        audited = audit_completed_run(
            dataset, record['command'], Path(record['run_dir']), root,
        )
    except (KeyError, OSError, ValueError) as error:
        raise ValueError(f'primary record re-audit failed for {dataset}') from error
    if record != audited:
        raise ValueError(f'primary record changed after audit for {dataset}')
    return record


def _primary_gate_report(root, primary, records):
    if set(records) != set(DATASETS):
        raise ValueError('gate requires exactly three current primary records')
    constraints = {}
    for dataset in DATASETS:
        incumbent = primary['incumbents'][dataset]
        if incumbent.get('record_sha256') != _payload_sha256({
                key: value for key, value in incumbent.items()
                if key != 'record_sha256'}):
            raise ValueError(f'incumbent evidence changed for {dataset}')
        metrics = _gate_metrics(records[dataset].get('metrics'))
        for metric in METRICS:
            actual = Decimal(str(metrics[metric]))
            reference = Decimal(str(incumbent['metrics'][metric]))
            floor = reference - TOLERANCE
            constraints[f'{dataset}:{metric}'] = {
                'dataset': dataset, 'metric': metric,
                'actual': metrics[metric],
                'incumbent': incumbent['metrics'][metric],
                'floor': str(floor), 'passed': actual >= floor,
            }
    passed = all(item['passed'] for item in constraints.values())
    status = 'GATE_SUCCESS' if passed else 'GATE_FAILED'
    report = {
        'schema_version': 1, 'status': status,
        'primary_plan_sha256': file_sha256(root / 'PRIMARY_PLAN.json'),
        'records': {
            dataset: {
                'path': str(root / 'primary' / dataset / 'record.json'),
                'sha256': file_sha256(
                    root / 'primary' / dataset / 'record.json'
                ),
            } for dataset in DATASETS
        },
        'constraints': constraints,
        'ablation_records_excluded': True,
    }
    return passed, report


def summarize(root):
    root = _validated_root(root)
    primary = _validate_primary_plan(root)
    if any(
        (root / name).exists() or (root / name).is_symlink()
        for name in ('EXECUTION_SUCCESS', 'GATE_SUCCESS', 'GATE_FAILED')
    ):
        status, report = _terminal_evidence(root)
        return (0 if status == 'GATE_SUCCESS' else GATE_FAILED), report
    try:
        records = {dataset: _strict_record(root, dataset) for dataset in DATASETS}
    except ValueError as error:
        return AUDIT_INCOMPLETE, {
            'schema_version': 1, 'status': 'INCOMPLETE', 'reason': str(error),
        }
    passed, report = _primary_gate_report(root, primary, records)
    status = report['status']
    _replace_json(root / 'PRIMARY_GATE.json', report)
    execution = {
        'status': 'EXECUTION_SUCCESS',
        'primary_plan_sha256': report['primary_plan_sha256'],
        'gate_report_sha256': file_sha256(root / 'PRIMARY_GATE.json'),
    }
    _write_new_json(root / 'EXECUTION_SUCCESS', execution)
    marker = {
        'status': status,
        'primary_plan_sha256': report['primary_plan_sha256'],
        'gate_report_sha256': execution['gate_report_sha256'],
    }
    _write_new_json(root / status, marker)
    opposite = root / ('GATE_FAILED' if passed else 'GATE_SUCCESS')
    _safe_unlink(opposite)
    return (0 if passed else GATE_FAILED), report


def static_check():
    if not (
        SEED == 42 and DATASETS == ('cifar100', 'isolet', 'upmc_food101')
        and set(EXPECTED_VALIDATION_HASHES) == set(DATASETS)
        and list(ABLATIONS) == ['fixed_half_ablation', 'sample_mean_nll']
        and all((WORKTREE / name).is_file() for name in ADAPTIVE_SOURCE_FILES)
    ):
        raise RuntimeError('adaptive development static contract failed')
    print(json.dumps({
        'status': 'OK', 'seed': SEED, 'datasets': list(DATASETS),
        'primary_jobs': primary_specs(), 'ablations': ablation_specs(),
    }, sort_keys=True))
    return 0


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='action', required=True)
    subparsers.add_parser('check')
    for action in ('plan', 'plan-ablations', 'audit', 'summarize'):
        command = subparsers.add_parser(action)
        command.add_argument('--root', type=Path, required=True)
    for action in ('plan-smoke', 'audit-smoke'):
        command = subparsers.add_parser(action)
        command.add_argument('--root', type=Path, required=True)
    run = subparsers.add_parser('run-job')
    run.add_argument('--root', type=Path, required=True)
    run.add_argument('--dataset', choices=DATASETS, required=True)
    run.add_argument('--seed', type=int, choices=[SEED], required=True)
    run.add_argument('--device', required=True)
    run.add_argument('--smoke', action='store_true')
    tiny = subparsers.add_parser('run-tiny-smoke')
    tiny.add_argument('--root', type=Path, required=True)
    tiny.add_argument('--device', required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.action == 'check':
            return static_check()
        if args.action == 'plan':
            plan(args.root)
            return 0
        if args.action == 'plan-ablations':
            plan_ablations(args.root)
            return 0
        if args.action == 'plan-smoke':
            plan_smoke(args.root)
            return 0
        if args.action == 'run-job':
            return run_job(
                args.root, args.dataset, args.seed, args.device, args.smoke,
            )
        if args.action == 'run-tiny-smoke':
            return run_tiny_smoke(args.root, args.device)
        if args.action == 'audit':
            return audit(args.root)
        if args.action == 'audit-smoke':
            audit_smoke(args.root)
            return 0
        if args.action == 'summarize':
            return summarize(args.root)[0]
    except (OSError, ValueError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    raise AssertionError('unreachable action')


if __name__ == '__main__':
    raise SystemExit(main())
