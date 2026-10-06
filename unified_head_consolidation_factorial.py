"""Frozen matrix contract for unified head-consolidation experiments."""
import argparse
import contextlib
import csv
from decimal import Decimal
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import tempfile

from dataset_protocol_smoke import replace_arg
from fair_main_table_3datasets import DATASETS, base_command
from selected_protocol_seed42 import validation_command


def deployment_paths(module_file=__file__, executable=sys.executable,
                     common_dir=None):
    worktree = Path(module_file).resolve().parent
    if common_dir is None:
        common_dir = subprocess.check_output(
            ['git', '-C', str(worktree), 'rev-parse', '--git-common-dir'],
            text=True,
        ).strip()
    common = Path(common_dir)
    if not common.is_absolute():
        common = (worktree / common).resolve()
    else:
        common = common.resolve()
    if common.name != '.git' or not common.is_dir():
        raise RuntimeError(f'invalid git common directory: {common}')
    return common.parent, worktree, Path(executable).resolve()


ROOT, WORKTREE, PYTHON = deployment_paths()
SOURCE_COMMIT = 'a60d429c91966c5d42d057c8bd5a380777cc69cb'
SEED = 42
DATASET_NAMES = ('cifar100', 'isolet', 'upmc_food101')
CELLS = {
    'A': ('full_classifier', 0.01, 500),
    'B': ('task_class_bias', 0.03, 600),
    'C': ('full_classifier', 0.03, 600),
    'D': ('task_class_bias', 0.01, 500),
}
INCUMBENTS = {'cifar100': 'B', 'isolet': 'A', 'upmc_food101': 'A'}
METRICS = ('AA_final', 'BWT', 'AA_final_taskil')
TOLERANCE = 0.01
CANONICAL_SOURCE_ROOT = Path('/home/chase/Yangxx/VF-CL')
REFERENCE_RELATIVE_RUNS = {
    ('cifar100', 'A'): Path('results/unified_head_consolidation_final_herding_seed42_20260809/cifar100/cifar100_unified_hc_final_herding_seed42_20260809_235858'),
    ('cifar100', 'B'): Path('results/unified_head_consolidation_factorial_seed42_20260813_151251/runs/cifar100_B_42/factorial_cifar100_B_seed42_20260813_171901'),
    ('isolet', 'A'): Path('results/unified_tcb_cross_dataset_validation_seed42_20260810/isolet/v1_full_classifier/isolet_v1_full_classifier_seed42_validation_20260810_110147'),
    ('isolet', 'B'): Path('results/unified_tcb_cross_dataset_validation_seed42_20260810/isolet/v2_task_class_bias/isolet_v2_task_class_bias_seed42_validation_20260810_110439'),
    ('upmc_food101', 'A'): Path('results/unified_tcb_cross_dataset_validation_seed42_20260810/upmc_food101/v1_full_classifier/upmc_food101_v1_full_classifier_seed42_validation_20260810_110147'),
    ('upmc_food101', 'B'): Path('results/unified_tcb_cross_dataset_validation_seed42_20260810/upmc_food101/v2_task_class_bias/upmc_food101_v2_task_class_bias_seed42_validation_20260810_110652'),
}
REFERENCE_RUNS = {
    key: ROOT / value for key, value in REFERENCE_RELATIVE_RUNS.items()
}
REFERENCE_DECLARED_RUNS = {
    key: CANONICAL_SOURCE_ROOT / value
    for key, value in REFERENCE_RELATIVE_RUNS.items()
}
REFERENCE_PROVENANCE = {
    ('cifar100', 'B'): ROOT / 'results/unified_head_consolidation_factorial_seed42_20260813_151251/runs/cifar100_B_42/planned_protocol.json',
    'isolet': ROOT / 'results/unified_tcb_cross_dataset_validation_seed42_20260810/isolet_MATCHED_PROTOCOL.json',
    'upmc_food101': ROOT / 'results/unified_tcb_cross_dataset_validation_seed42_20260810/upmc_food101_MATCHED_PROTOCOL.json',
}
_CIFAR_B_MATRIX = ROOT / 'results/unified_head_consolidation_factorial_seed42_20260813_151251'
_CIFAR_B_JOB = _CIFAR_B_MATRIX / 'runs/cifar100_B_42'
_CIFAR_B_IMPLEMENTATION_COMMIT = '5669fc897f7ce5b88decb3ec6dba5e614afc110e'
REFERENCE_AUXILIARY_PATHS = {
    ('cifar100', 'B'): {
        'data_flow_audit': REFERENCE_RUNS[('cifar100', 'B')] / 'data_flow_audit.jsonl',
        'launch_started': _CIFAR_B_JOB / 'launch_started.json',
        'record': _CIFAR_B_JOB / 'record.json',
        'code_commit': _CIFAR_B_MATRIX / 'CODE_COMMIT.txt',
        'matrix_identity': _CIFAR_B_MATRIX / 'MATRIX_IDENTITY.txt',
    },
}
REFERENCE_ARTIFACT_SHA256 = {
    ('cifar100', 'B'): {
        'config': '8da359a7fa3188e16a64feaf71ecf63bd5234d5d56bbf6b0e3be000cab44279f',
        'results': '2349f80862b11cd48a0af76ba09d850be61390c670fbb20351f398c2d12b3019',
        'validation_manifest': '7934fcbc5385883cd4312d7416d70dbae8a73044802431f766ee6de2249e5418',
        'final_checkpoint': 'ab7b84e7ba96766ead1838aa0b52e5ea877d69d2a4476484afe09478b649bdd8',
        'head_audit': '71840fa517a9c5d614526656e349a3c14d0466e5c77a573f147c38ffc47dc9f0',
        'data_flow_audit': '65f47b841c1253623df684f70793dec42c8c207100ccb8013fc976c065f310f9',
        'provenance': '594c1842570a731dc1385c6dcbfe62eb2a865722edfaaa8b1d8c20364cda58cc',
        'launch_started': '8d8ee990c89e9ad5dbf129fdff128a96a57f0609d0ca41be230557823c6d3589',
        'record': '0303fc8c890bb7c828e87b72c13035b1ebd862c55d16e6dab820ebec598f07be',
        'code_commit': '078da3cfb3af22383003cdbe763dfd7c3389efdc688fc5a72070a906622caf87',
        'matrix_identity': '4d612317ecf0129396be1dc18e267123a3e9629dc542ade2a28099d2a6111500',
    },
}
EXPECTED_DATA_ARTIFACTS = {
    'cifar100': {
        ROOT / 'data/cifar-100-python/meta': 'a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9',
        ROOT / 'data/cifar-100-python/test': '4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864',
        ROOT / 'data/cifar-100-python/train': '735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b',
    },
    'isolet': {
        ROOT / 'data/isolet/isolet_vfl.npz': 'd34312670de93198afcd2b126c95b79bae2b4cffeb30d480f097faf046b69514',
        ROOT / 'data/isolet/isolet_vfl.metadata.json': '79396dea1751b6094a5769f2dd789ad58b3ea9582a12c8c98d3ec07d8d1eb3cd',
    },
    'upmc_food101': {
        ROOT / 'data/upmc_food101/upmc_food101_vfl.npz': '9268de2e698e25138d533bac47ae1f9a763d73518cfe6c785eda0a8232d801cf',
        ROOT / 'data/upmc_food101/upmc_food101_vfl.metadata.json': '14e996a39a7aacb7f04e3053bef2586a2b3c1690f88b3f72fd01e911b5e4a821',
    },
}
EXPECTED_VALIDATION_HASHES = {
    'cifar100': '0aa4729ade65021ce774c757516584c4d2a2ce70d2879b045db47dbceae917fa',
    'isolet': '487e81663a12d4a663a1f421407aa3f88d788cc3c83f7323166cf7aff82902d3',
    'upmc_food101': 'b812e99da856d94ee5b6eea28eadaf9924f52e355d44941c42da4028c0895e06',
}
EXPECTED_VALIDATION_LABELS = {
    'cifar100': 'cifar100-train',
    'isolet': 'isolet_vfl.npz-train',
    'upmc_food101': 'upmc_food101_vfl.npz-train',
}
_IGNORED_CONFIG = {
    'device', 'output_dir', 'results_dir', 'exp_name', 'resume_run_dir',
    'head_consolidation_mode', 'head_consolidation_lr',
    'head_consolidation_steps',
}
_POST_SOURCE_REFERENCE_DEFAULTS = {
    'formal_deferred_evaluation': False,
    'fedprotip_tip_threshold': 0.775,
    'fedprotip_max_batches': 20,
    'head_full_lr': 0.01,
    'head_full_steps': 500,
    'head_bias_lr': 0.03,
    'head_bias_steps': 600,
    'head_gate_rule': 'class_balanced',
    'head_gate_solver_tolerance': 1e-12,
    'head_gate_solver_max_iterations': 80,
}
EXPECTED_TASKS = {'cifar100': 10, 'isolet': 13, 'upmc_food101': 10}
AUDIT_NO_COMPLETE = 90
AUDIT_MISMATCH = 91
_WRITE_NAMES = (
    'planned_protocol.json', 'launch_started.json', 'job.log', 'FAILED.json',
    'record.json', 'SUCCESS',
)


def all_specs():
    return [
        f'{dataset}:{cell}:{SEED}'
        for dataset in DATASET_NAMES
        for cell in CELLS
    ]


def jobs(worker=None, workers=2, specs=None):
    if workers < 1:
        raise ValueError('workers must be at least 1')
    if worker is not None and not 0 <= worker < workers:
        raise ValueError('worker is outside the worker range')
    specs = all_specs() if specs is None else list(specs)
    if worker is None:
        return specs
    return [spec for index, spec in enumerate(specs) if index % workers == worker]


def parse_spec(spec):
    try:
        dataset, cell, seed_text = spec.split(':')
        seed = int(seed_text)
    except (AttributeError, ValueError):
        raise ValueError(f'unknown spec: {spec}') from None
    if dataset not in DATASET_NAMES or cell not in CELLS or seed != SEED:
        raise ValueError(f'unknown spec: {spec}')
    return dataset, cell, seed


def cifar_base_command(device, results_dir, smoke):
    return [
        str(PYTHON), str(WORKTREE / 'main.py'),
        '--data', 'cifar100',
        '--data_path', str(ROOT / 'data'),
        '--num_classes', '100',
        '--num_tasks', '2' if smoke else '10',
        '--classes_per_task', '10',
        '--num_parties', '4',
        '--model_type', 'resnet18',
        '--aggregation', 'sum',
        '--epochs_per_task', '1' if smoke else '50',
        '--batch_size', '64',
        '--num_workers', '2',
        '--optimizer', 'sgd',
        '--lr', '0.001',
        '--bottom_lr_scale', '1.0',
        '--task_ce_mode', 'method',
        '--momentum', '0.9',
        '--weight_decay', '0.0005',
        '--device', str(device),
        '--unlearn_after_tasks', '99',
        '--unlearn_classes', '0',
        '--ul_method', 'retrain',
        '--replay_mode', 'prototype',
        '--deterministic', '1',
        '--data_flow_audit', '1',
        '--bic_enabled', '1',
        '--bic_fit_mode', 'joint_each_stage',
        '--bic_lr', '0.05',
        '--bic_per_class', '25',
        '--bic_split_seed', '20260722',
        '--bic_steps', '1000',
        '--lambda_validation_enabled', '1',
        '--lambda_validation_per_class', '25',
        '--lambda_validation_split_seed', '20260729',
        '--save_task_checkpoints', '3',
        '--seed', str(SEED),
        '--cl_method', 'proto_evolve',
        '--dep_tracking_enabled', '1',
        '--party_kd_enabled', '1',
        '--party_kd_mode', 'uniform',
        '--expected_party_kd_variant', 'uniform',
        '--party_kd_lambda', '1.0',
        '--proto_replay_loss_norm', 'sample_mean',
        '--proto_replay_ratio', '1.0',
        '--proto_lambda_a', '0.15',
        '--fim_freeze_frac', '0',
        '--head_consolidation_enabled', '1',
        '--head_consolidation_regularization', '0.01',
        '--head_consolidation_class_regularization', '0.01',
        '--head_consolidation_task_regularization', '0.01',
        '--head_consolidation_task_weight', '1.3',
        '--head_consolidation_samples_per_class', '20',
        '--head_consolidation_schedule', 'final',
        '--distill_weight', '0.25',
        '--feat_distill_weight', '0.05',
        '--current_supcon_weight', '0',
        '--results_dir', str(results_dir),
        '--exp_name', 'run',
    ]


def vector_command(dataset, device, results_dir, smoke, resume_run_dir):
    spec = f'{dataset}:ours:{SEED}'
    if smoke:
        command = base_command(spec, device, results_dir, resume_run_dir=resume_run_dir, smoke=True)
        cfg = DATASETS[dataset]
        replace_arg(command, '--lambda_validation_enabled', 1)
        replace_arg(command, '--lambda_validation_per_class', cfg['validation_per_class'])
        replace_arg(command, '--lambda_validation_split_seed', cfg['validation_split_seed'])
    else:
        command = validation_command(
            spec, device, results_dir, resume_run_dir=resume_run_dir,
        )
    command[1] = str(WORKTREE / 'main.py')
    return command


def build_command(spec, device, results_dir, smoke=False, resume_run_dir=None):
    dataset, cell, _ = parse_spec(spec)
    results_dir = Path(results_dir)
    if dataset == 'cifar100':
        command = cifar_base_command(device, results_dir, smoke)
    else:
        command = vector_command(
            dataset, device, results_dir, smoke, resume_run_dir,
        )
    mode, lr, steps = CELLS[cell]
    replace_arg(command, '--head_consolidation_mode', mode)
    replace_arg(command, '--head_consolidation_lr', lr)
    replace_arg(command, '--head_consolidation_steps', steps)
    replace_arg(command, '--results_dir', results_dir)
    replace_arg(command, '--exp_name', f'factorial_{dataset}_{cell}_seed42')
    if dataset == 'cifar100' and resume_run_dir:
        command.extend(['--resume_run_dir', str(resume_run_dir)])
    return [str(value) for value in command]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def _trusted_dir(path, create=False):
    path = Path(os.path.abspath(path))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open('/', flags)
    try:
        try:
            for component in path.parts[1:]:
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create:
                        raise
                    os.mkdir(component, dir_fd=descriptor)
                    child = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
        except OSError as error:
            raise ValueError(f'unsafe directory component in {path}') from error
        yield descriptor
    finally:
        os.close(descriptor)


def _ensure_directory(path):
    with _trusted_dir(path, create=True):
        return Path(path)


def _safe_stat(path):
    path = Path(path)
    try:
        with _trusted_dir(path.parent) as directory:
            details = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(details.st_mode):
        raise ValueError(f'symlinked target is not allowed: {path}')
    return details


def _write_text(path, content, exclusive=False):
    path = Path(path)
    _ensure_directory(path.parent)
    temporary = f'.{path.name}.{secrets.token_hex(8)}.tmp'
    with _trusted_dir(path.parent) as directory:
        try:
            existing = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and stat.S_ISLNK(existing.st_mode):
            raise ValueError(f'symlinked target is not allowed: {path}')
        if exclusive and existing is not None:
            raise FileExistsError(path)
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600, dir_fd=directory,
        )
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8', closefd=False) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if exclusive:
                try:
                    os.link(
                        temporary, path.name,
                        src_dir_fd=directory, dst_dir_fd=directory,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    raise FileExistsError(path) from None
                os.unlink(temporary, dir_fd=directory)
            else:
                os.replace(
                    temporary, path.name,
                    src_dir_fd=directory, dst_dir_fd=directory,
                )
        finally:
            os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass


def atomic_json(path, payload):
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + '\n')


def _exclusive_json(path, payload):
    _write_text(
        path, json.dumps(payload, indent=2, sort_keys=True) + '\n', exclusive=True,
    )


@contextlib.contextmanager
def _append_text(path):
    path = Path(path)
    _ensure_directory(path.parent)
    with _trusted_dir(path.parent) as directory:
        descriptor = os.open(
            path.name, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
            0o600, dir_fd=directory,
        )
    with os.fdopen(descriptor, 'a', encoding='utf-8') as handle:
        yield handle


def _safe_unlink(path, missing_ok=True):
    path = Path(path)
    with _trusted_dir(path.parent) as directory:
        try:
            details = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            if missing_ok:
                return
            raise FileNotFoundError(path)
        if stat.S_ISLNK(details.st_mode):
            raise ValueError(f'symlinked target is not allowed: {path}')
        os.unlink(path.name, dir_fd=directory)


def _safe_touch(path):
    path = Path(path)
    _ensure_directory(path.parent)
    with _trusted_dir(path.parent) as directory:
        descriptor = os.open(
            path.name, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW,
            0o600, dir_fd=directory,
        )
        os.close(descriptor)


def _safe_sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    with _trusted_dir(path.parent) as directory:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    with os.fdopen(descriptor, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_json(path):
    path = Path(path)
    with _trusted_dir(path.parent) as directory:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    with os.fdopen(descriptor, 'r', encoding='utf-8') as handle:
        try:
            payload = json.load(handle)
        except (OSError, ValueError, TypeError):
            return None
    return payload if isinstance(payload, dict) else None


def deterministic_env():
    env = os.environ.copy()
    env.update({
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'PYTHONHASHSEED': str(SEED),
    })
    return env


def _implementation_commit():
    return subprocess.check_output(
        ['git', '-C', str(WORKTREE), 'rev-parse', 'HEAD'], text=True,
    ).strip()


def _reject_symlink_components(path):
    path = Path(os.path.abspath(path))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open('/', flags)
    try:
        parts = path.parts[1:]
        for index, component in enumerate(parts):
            try:
                details = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                return
            if stat.S_ISLNK(details.st_mode):
                raise ValueError(f'symlinked path component is not allowed: {path}')
            if index < len(parts) - 1:
                if not stat.S_ISDIR(details.st_mode):
                    raise ValueError(f'non-directory path component: {path}')
                child = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
    finally:
        os.close(descriptor)


def _require_within(path, root, label):
    path = Path(os.path.abspath(path))
    root = Path(os.path.abspath(root))
    try:
        path.relative_to(root)
        _reject_symlink_components(root)
        _reject_symlink_components(path)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f'{label} is outside its root: {path}') from error
    return path


def job_root(matrix_root, spec, smoke=False):
    dataset, cell, seed = parse_spec(spec)
    matrix_root = Path(matrix_root)
    section = matrix_root / ('smoke' if smoke else 'runs')
    root = section / f'{dataset}_{cell}_{seed}'
    return _require_within(root, matrix_root, 'job root')


def _run_name_matches_spec(name, spec):
    if name.startswith('run_'):
        return True
    if spec is None:
        return False
    dataset, cell, seed = parse_spec(spec)
    return name.startswith(f'factorial_{dataset}_{cell}_seed{seed}_')


def discover_runs(root, spec=None):
    root = Path(root)
    if root.is_symlink():
        raise ValueError(f'symlinked job root is not allowed: {root}')
    if not root.is_dir():
        return []
    runs = []
    for path in sorted(root.iterdir()):
        if not _run_name_matches_spec(path.name, spec):
            continue
        if path.is_symlink() or path.resolve().parent != root.resolve():
            raise ValueError(f'run directory is outside its job root: {path}')
        if path.is_dir():
            runs.append(path)
    return runs


def find_complete(root, expected_tasks, spec=None):
    final = expected_tasks - 1
    complete = []
    for run in discover_runs(root, spec):
        results = run / 'results.json'
        checkpoint = run / 'checkpoints' / f'event_{final}_CIL.pt'
        if results.is_file():
            _require_within(results, run, 'results evidence')
        if checkpoint.is_file():
            _require_within(checkpoint, run, 'checkpoint evidence')
        if results.is_file() and checkpoint.is_file():
            complete.append(run)
    return complete


def find_resume(root, spec=None):
    identity = parse_spec(spec) if spec is not None else None
    for run in reversed(discover_runs(root, spec)):
        config_path = run / 'config.json'
        resume_path = run / 'checkpoints' / 'resume_latest.pt'
        if config_path.is_file():
            _require_within(config_path, run, 'resume config')
        if resume_path.is_file():
            _require_within(resume_path, run, 'resume checkpoint')
        config = _json(config_path) if config_path.is_file() else None
        if config_path.is_file() and resume_path.is_file() and (
            spec is None or (
                isinstance(config, dict)
                and config.get('exp_name') == (
                    f'factorial_{identity[0]}_{identity[1]}_seed{identity[2]}'
                )
                and config.get('seed') == identity[2]
                and (
                    config.get('head_consolidation_mode'),
                    config.get('head_consolidation_lr'),
                    config.get('head_consolidation_steps'),
                ) == CELLS[identity[1]]
            )
        ):
            return run
    return None


def _json(path):
    try:
        with Path(path).open(encoding='utf-8') as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _file_hash(path):
    try:
        return sha256(path)
    except OSError:
        return None


def _final_event(directory, suffix):
    try:
        candidates = list(Path(directory).glob(f'event_*_CIL.{suffix}'))
        return max(
            candidates,
            key=lambda path: int(re.fullmatch(r'event_(\d+)_CIL', path.stem).group(1)),
        )
    except (OSError, ValueError, AttributeError):
        return None


def _resolved_commit(commit):
    if not isinstance(commit, str) or not commit.strip():
        return None
    try:
        return subprocess.run(
            ['git', '-C', str(ROOT), 'rev-parse', f'{commit}^{{commit}}'],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _effective_config(spec, audit_dir, smoke=False):
    from config import get_config

    with tempfile.TemporaryDirectory() as results_dir:
        command = build_command(spec, 'cuda:0', results_dir, smoke=smoke)
        previous = sys.argv
        try:
            sys.argv = command[1:]
            with contextlib.redirect_stdout(io.StringIO()):
                return vars(get_config())
        finally:
            sys.argv = previous


def _validation_manifest_matches(manifest, dataset, config_payload):
    return isinstance(manifest, dict) and isinstance(config_payload, dict) and (
        manifest.get('dataset') == EXPECTED_VALIDATION_LABELS.get(dataset)
        and manifest.get('sha256') == EXPECTED_VALIDATION_HASHES.get(dataset)
        and manifest.get('seed') == config_payload.get(
            'lambda_validation_split_seed'
        )
        and manifest.get('per_class') == config_payload.get(
            'lambda_validation_per_class'
        )
    )


def _mismatches(expected, actual):
    missing = '<missing>'
    return {
        key: {'expected': expected.get(key, missing), 'actual': actual.get(key, missing)}
        for key in sorted((set(expected) | set(actual)) - _IGNORED_CONFIG)
        if not _config_values_match(
            expected.get(key, missing), actual.get(key, missing),
        )
    }


def _config_values_match(expected, actual):
    return type(expected) is type(actual) and expected == actual


def _reference_config_mismatches(expected, actual):
    compatible = dict(actual)
    for key, value in _POST_SOURCE_REFERENCE_DEFAULTS.items():
        if key not in compatible and _config_values_match(expected.get(key), value):
            compatible[key] = value
    return _mismatches(expected, compatible)


def _fit_flags(payload):
    found = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == 'test_used_for_fit':
                found.append(value)
            else:
                found.extend(_fit_flags(value))
    elif isinstance(payload, list):
        for value in payload:
            found.extend(_fit_flags(value))
    return found


def _data_artifacts(dataset):
    records = []
    for path, expected in EXPECTED_DATA_ARTIFACTS.get(dataset, {}).items():
        actual = _file_hash(path)
        records.append({
            'path': str(path), 'expected_sha256': expected,
            'actual_sha256': actual, 'passed': actual == expected,
        })
    return records


def _jsonl(path):
    records = []
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f'invalid JSONL record: {path}')
                records.append(record)
    if not records:
        raise ValueError(f'empty JSONL evidence: {path}')
    return records


def _peer_validation_hash(record_path, matrix_root, dataset):
    try:
        record_path = Path(record_path)
        record = _json(record_path)
        if not record or record.get('passed') is not True:
            return None
        spec = record.get('spec')
        peer_dataset, _, seed = parse_spec(spec)
        smoke = record.get('smoke')
        if peer_dataset != dataset or seed != SEED or not isinstance(smoke, bool):
            return None
        root = job_root(matrix_root, spec, smoke=smoke)
        if record_path.resolve() != (root / 'record.json').resolve():
            return None
        _require_within(record_path, root, 'peer record')
        success = _require_within(root / 'SUCCESS', root, 'peer success')
        if not success.is_file():
            return None
        if not (
            record.get('dataset') == dataset
            and record.get('seed') == SEED
            and record.get('source_commit') == SOURCE_COMMIT
            and record.get('implementation_commit') == _implementation_commit()
            and record.get('validation_hash') == EXPECTED_VALIDATION_HASHES.get(dataset)
        ):
            return None
        checks = record.get('checks')
        paths = record.get('paths')
        hashes = record.get('sha256')
        required = {
            'config', 'results', 'validation_manifest', 'head_audit',
            'checkpoint', 'data_flow_audit',
        }
        if not (
            isinstance(checks, dict) and checks and all(value is True for value in checks.values())
            and isinstance(paths, dict) and isinstance(hashes, dict)
            and set(paths) == set(hashes) == required
        ):
            return None
        run_dir = Path(record.get('run_dir'))
        if (
            run_dir.is_symlink() or run_dir.resolve().parent != root.resolve()
            or not _run_name_matches_spec(run_dir.name, spec)
        ):
            return None
        for name, value in paths.items():
            path = _require_within(Path(value), run_dir, f'peer {name}')
            if not path.is_file() or _file_hash(path) != hashes[name]:
                return None
        manifest = _json(paths['validation_manifest'])
        expected_config = _effective_config(spec, matrix_root, smoke=smoke)
        if (
            not _validation_manifest_matches(manifest, dataset, expected_config)
            or manifest.get('sha256') != record.get('validation_hash')
        ):
            return None
        return record['validation_hash']
    except (Exception, SystemExit):
        return None


def _matching_validation_records(matrix_root, dataset, excluded=None):
    hashes = []
    for section in ('runs', 'smoke'):
        parent = Path(matrix_root) / section
        if not parent.is_dir():
            continue
        for path in parent.glob('*/record.json'):
            if excluded is not None and path.resolve() == Path(excluded).resolve():
                continue
            validation_hash = _peer_validation_hash(path, matrix_root, dataset)
            if validation_hash is not None:
                hashes.append(validation_hash)
    return hashes


def _expected_seen_classes(dataset, smoke, config_payload):
    if not smoke:
        return config_payload.get('num_classes')
    if dataset == 'cifar100':
        classes_per_task = config_payload.get('classes_per_task')
        return classes_per_task * 2 if isinstance(classes_per_task, int) else None
    return sum(len(task) for task in DATASETS[dataset]['tasks'][:2])


def audit_completed_run(spec, command, run_dir, matrix_root, smoke=False):
    dataset, cell, seed = parse_spec(spec)
    expected_tasks = 2 if smoke else EXPECTED_TASKS[dataset]
    final = expected_tasks - 1
    run_dir = Path(run_dir)
    root = job_root(matrix_root, spec, smoke=smoke)
    try:
        if (
            run_dir.is_symlink()
            or run_dir.resolve().parent != root.resolve()
            or not _run_name_matches_spec(run_dir.name, spec)
        ):
            raise ValueError(f'run directory is outside its job root: {run_dir}')
    except OSError as error:
        raise ValueError(f'invalid run directory: {run_dir}') from error
    paths = {
        'config': run_dir / 'config.json',
        'results': run_dir / 'results.json',
        'validation_manifest': run_dir / 'validation' / 'validation_manifest.json',
        'head_audit': run_dir / 'head_consolidation' / f'event_{final}_CIL.json',
        'checkpoint': run_dir / 'checkpoints' / f'event_{final}_CIL.pt',
        'data_flow_audit': run_dir / 'data_flow_audit.jsonl',
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f'missing completed-run evidence: {missing}')
    for name, path in paths.items():
        try:
            path.resolve().relative_to(run_dir.resolve())
        except (OSError, ValueError) as error:
            raise ValueError(f'{name} evidence is outside the run directory: {path}') from error

    if not isinstance(command, list) or len(command) % 2:
        raise ValueError('invalid completed-run command')
    flags = dict(zip(command[2::2], command[3::2]))
    resume = flags.get('--resume_run_dir')
    if resume is not None and Path(resume).resolve() != run_dir.resolve():
        raise ValueError('resume directory does not match the audited run directory')
    expected_command = build_command(
        spec, flags.get('--device'), root, smoke=smoke,
        resume_run_dir=resume,
    )
    if command != expected_command:
        raise ValueError('completed-run command does not match the frozen build command')

    config_payload = _json(paths['config'])
    results = _json(paths['results'])
    manifest = _json(paths['validation_manifest'])
    head = _json(paths['head_audit'])
    if None in (config_payload, results, manifest, head):
        raise ValueError('malformed completed-run JSON evidence')
    data_flow = _jsonl(paths['data_flow_audit'])
    expected_config = _effective_config(spec, Path(matrix_root), smoke=smoke)
    factors = CELLS[cell]
    actual_factors = (
        config_payload.get('head_consolidation_mode'),
        config_payload.get('head_consolidation_lr'),
        config_payload.get('head_consolidation_steps'),
    )
    selection = results.get('selection_audit')
    expected_source = (
        'cifar100-train-validation'
        if dataset == 'cifar100' else 'vector-train-validation'
    )
    metrics_source = results.get('cl_metrics')
    metrics = {
        key: metrics_source.get(key) if isinstance(metrics_source, dict) else None
        for key in ('AA_final', 'BWT', 'AA_final_taskil')
    }
    numeric_metrics = all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value)
        for value in metrics.values()
    )
    trajectory = (
        metrics_source.get('AA_trajectory_taskil')
        if isinstance(metrics_source, dict) else None
    )
    raw_count = head.get('persistent_raw_example_count')
    class_count = head.get('class_count')
    embedding_count = head.get('persistent_embedding_count')
    expected_class_count = _expected_seen_classes(dataset, smoke, expected_config)
    data_artifacts = _data_artifacts(dataset)
    validation_hash = manifest.get('sha256')
    peer_hashes = _matching_validation_records(
        matrix_root, dataset, excluded=root / 'record.json',
    )
    fit_flags = _fit_flags([results, head, data_flow])
    checks = {
        'effective_config': not _mismatches(expected_config, config_payload),
        'device': config_payload.get('device') == flags.get('--device'),
        'factors': actual_factors == factors and (
            head.get('mode'), head.get('lr'), head.get('steps')
        ) == factors,
        'dataset': config_payload.get('data') == expected_config.get('data'),
        'seed': config_payload.get('seed') == seed == SEED,
        'replay': config_payload.get('replay_mode') == 'prototype' and (
            head.get('replay_selection') == 'normalized_feature_herding'
            and head.get('samples_per_class') == 20
            and isinstance(raw_count, int) and isinstance(class_count, int)
            and isinstance(embedding_count, int)
            and not isinstance(raw_count, bool) and not isinstance(class_count, bool)
            and not isinstance(embedding_count, bool)
            and class_count == expected_class_count
            and 0 <= raw_count <= 20 * expected_class_count
            and embedding_count == 0
        ),
        'schedule_source': (
            head.get('schedule') == 'final'
            and head.get('source') == 'balanced_current_encoder_raw_replay'
        ),
        'selection_audit': isinstance(selection, dict) and (
            selection.get('passed') is True
            and selection.get('test_used_for_selection') is False
            and selection.get('evaluation_source') == expected_source
        ),
        'test_not_used': (
            head.get('test_used') is False
            and head.get('validation_used') is False
            and all(value is False for value in fit_flags)
        ),
        'validation': (
            _validation_manifest_matches(manifest, dataset, expected_config)
            and all(value == validation_hash for value in peer_hashes)
        ),
        'data_artifacts': bool(data_artifacts) and all(
            item['passed'] for item in data_artifacts
        ),
        'metrics': numeric_metrics,
        'trajectory': (
            isinstance(trajectory, list) and len(trajectory) == expected_tasks
            and isinstance(trajectory[-1], dict)
            and trajectory[-1].get('step') == f'event_{final}_CIL'
        ),
        'final_task': (
            head.get('task_id') == final
            and head.get('task_boundary') == f'event_{final}_CIL'
        ),
        'data_flow_audit': bool(data_flow),
    }
    if not all(checks.values()):
        raise ValueError(f'{spec} completed-run audit failed: {checks}')

    return {
        'spec': spec, 'dataset': dataset, 'cell': cell, 'seed': seed,
        'factors': list(factors), 'command': list(command),
        'implementation_commit': _implementation_commit(),
        'source_commit': SOURCE_COMMIT,
        'result_dir': str(run_dir.parent), 'run_dir': str(run_dir),
        'paths': {name: str(path) for name, path in paths.items()},
        'sha256': {name: sha256(path) for name, path in paths.items()},
        'data_artifacts': data_artifacts,
        'validation_hash': validation_hash,
        'expected_tasks': expected_tasks,
        'checks': checks, 'metrics': metrics,
        'smoke': smoke, 'passed': True,
    }


def _valid_record_checked(root, spec, matrix_root, smoke):
    record = _json(Path(root) / 'record.json')
    if not record or not (
        record.get('passed') is True and record.get('spec') == spec
        and record.get('smoke') is smoke
    ):
        return False
    run_dir = Path(record.get('run_dir', ''))
    try:
        run_dir.resolve().relative_to(Path(root).resolve())
    except (OSError, ValueError):
        return False
    paths = record.get('paths')
    hashes = record.get('sha256')
    required_paths = {
        'config', 'results', 'validation_manifest', 'head_audit',
        'checkpoint', 'data_flow_audit',
    }
    checks = record.get('checks')
    if not (
        isinstance(paths, dict) and isinstance(hashes, dict)
        and set(paths) == set(hashes) == required_paths
        and isinstance(checks, dict) and checks and all(checks.values())
        and record.get('source_commit') == SOURCE_COMMIT
        and isinstance(record.get('implementation_commit'), str)
    ):
        return False
    for name, path in paths.items():
        path = Path(path)
        try:
            path.resolve().relative_to(run_dir.resolve())
        except (OSError, ValueError):
            return False
        if not path.is_file() or _file_hash(path) != hashes[name]:
            return False
    try:
        command = record.get('command')
        flags = dict(zip(command[2::2], command[3::2]))
        planned = _validated_plan(
            spec, flags.get('--device'), root, smoke,
        )
        if not _command_uses_plan(command, planned, root):
            return False
        audited = audit_completed_run(
            spec, command, run_dir, matrix_root, smoke=smoke,
        )
        _validated_launch_started(
            spec, planned, root, Path(root) / 'planned_protocol.json',
            audited['paths'].values(),
        )
    except Exception:
        return False
    return record == audited


def _valid_record(root, spec, matrix_root, smoke):
    try:
        return _valid_record_checked(root, spec, matrix_root, smoke)
    except (Exception, SystemExit):
        return False


def _write_targets(root):
    targets = {}
    for name in _WRITE_NAMES:
        path = Path(root) / name
        if path.is_symlink():
            raise ValueError(f'symlinked write target is not allowed: {path}')
        targets[name] = _require_within(path, root, f'{name} write target')
    return targets


def _plan_payload(spec, command, smoke):
    dataset, cell, seed = parse_spec(spec)
    if not isinstance(command, list) or len(command) % 2:
        raise ValueError('planned command is invalid')
    flags = dict(zip(command[2::2], command[3::2]))
    expected_tasks = 2 if smoke else EXPECTED_TASKS[dataset]
    return {
        'spec': spec, 'dataset': dataset, 'cell': cell, 'seed': seed,
        'factors': list(CELLS[cell]), 'command': list(command),
        'device': flags.get('--device'),
        'implementation_commit': _implementation_commit(),
        'source_commit': SOURCE_COMMIT,
        'data_artifacts': _data_artifacts(dataset),
        'validation': {
            'dataset': EXPECTED_VALIDATION_LABELS[dataset],
            'logical_sha256': EXPECTED_VALIDATION_HASHES[dataset],
            'split_seed': int(flags['--lambda_validation_split_seed']),
            'per_class': int(flags['--lambda_validation_per_class']),
            'evaluation_source': (
                'cifar100-train-validation'
                if dataset == 'cifar100' else 'vector-train-validation'
            ),
        },
        'expected_tasks': expected_tasks,
        'selection_source': 'training-validation',
        'selection_test_used': False, 'test_used_for_fit': False,
        'smoke': smoke,
    }


def _launch_payload(spec, command, plan_path):
    flags = dict(zip(command[2::2], command[3::2]))
    return {
        'spec': spec,
        'command': list(command),
        'device': flags.get('--device'),
        'implementation_commit': _implementation_commit(),
        'plan_sha256': _safe_sha256(plan_path),
    }


def _write_launch_started(spec, command, root, plan_path):
    path = Path(root) / 'launch_started.json'
    _exclusive_json(path, _launch_payload(spec, command, plan_path))
    return path


def _validated_launch_started(
        spec, command, root, plan_path, evidence_paths=()):
    path = Path(root) / 'launch_started.json'
    marker = _safe_json(path)
    if marker != _launch_payload(spec, command, plan_path):
        raise ValueError('launch-start evidence does not match planned protocol')
    plan_stat = _safe_stat(plan_path)
    marker_stat = _safe_stat(path)
    if plan_stat is None or marker_stat is None or (
            plan_stat.st_ctime_ns > marker_stat.st_ctime_ns):
        raise ValueError('planned protocol was not installed before launch evidence')
    for evidence in evidence_paths:
        evidence_stat = _safe_stat(evidence)
        if evidence_stat is None or marker_stat.st_ctime_ns > evidence_stat.st_ctime_ns:
            raise ValueError('launch evidence was manufactured after completion')
    return path


def _validated_plan(spec, device, root, smoke):
    path = Path(root) / 'planned_protocol.json'
    plan = _safe_json(path)
    if not plan:
        raise ValueError('complete or resumed run has no planned protocol')
    command = plan.get('command')
    if not isinstance(command, list) or len(command) % 2:
        raise ValueError('planned protocol command is invalid')
    flags = dict(zip(command[2::2], command[3::2]))
    resume = flags.get('--resume_run_dir')
    expected = build_command(
        spec, flags.get('--device'), root, smoke=smoke,
        resume_run_dir=resume,
    )
    if command != expected or flags.get('--device') != device:
        raise ValueError('planned protocol command or device is inconsistent')
    if plan != _plan_payload(spec, command, smoke):
        raise ValueError('planned protocol does not match frozen provenance')
    return command


def _command_uses_plan(command, planned, root):
    if command == planned:
        return True
    if not isinstance(command, list):
        return False
    if '--resume_run_dir' in planned or '--resume_run_dir' not in command:
        return False
    index = command.index('--resume_run_dir')
    resume = Path(command[index + 1])
    return command[:index] + command[index + 2:] == planned and (
        not resume.is_symlink() and resume.resolve().parent == Path(root).resolve()
    )


def _failure(root, spec, returncode, command, reason):
    targets = _write_targets(root)
    payload = {
        'spec': spec, 'returncode': returncode,
        'command': command, 'log': str(targets['job.log']),
        'reason': reason,
    }
    atomic_json(targets['FAILED.json'], payload)
    return returncode


def run_job(spec, device, matrix_root, smoke=False):
    dataset, cell, seed = parse_spec(spec)
    root = job_root(matrix_root, spec, smoke=smoke)
    _ensure_directory(root)
    targets = _write_targets(root)
    success = targets['SUCCESS']
    if success.is_file() and _valid_record(root, spec, matrix_root, smoke):
        print(f'SKIP complete {spec}')
        return 0
    _safe_unlink(success)

    expected_tasks = 2 if smoke else EXPECTED_TASKS[dataset]
    complete = find_complete(root, expected_tasks, spec)
    resume = None if complete else find_resume(root, spec)
    plan_path = targets['planned_protocol.json']
    plan_stat = _safe_stat(plan_path)
    plan_exists = plan_stat is not None and stat.S_ISREG(plan_stat.st_mode)
    try:
        planned = (
            _validated_plan(spec, device, root, smoke)
            if plan_exists else None
        )
    except ValueError as error:
        return _failure(root, spec, AUDIT_MISMATCH, [], str(error))

    if complete and planned is None:
        return _failure(
            root, spec, AUDIT_MISMATCH, [],
            'complete run has no valid pre-launch planned protocol',
        )
    if planned is None:
        command = build_command(
            spec, device, root, smoke=smoke, resume_run_dir=resume,
        )
    elif resume is not None and '--resume_run_dir' not in planned:
        command = [*planned, '--resume_run_dir', str(resume)]
    else:
        command = list(planned)

    if not complete:
        if planned is None:
            try:
                _exclusive_json(plan_path, _plan_payload(spec, command, smoke))
            except FileExistsError:
                return _failure(
                    root, spec, AUDIT_MISMATCH, command,
                    'planned protocol already exists and was not adopted',
                )
            planned = list(command)
        launch_path = targets['launch_started.json']
        if _safe_stat(launch_path) is None:
            try:
                _write_launch_started(spec, planned, root, plan_path)
            except FileExistsError:
                return _failure(
                    root, spec, AUDIT_MISMATCH, command,
                    'launch-start evidence appeared concurrently',
                )
        try:
            _validated_launch_started(spec, planned, root, plan_path)
        except ValueError as error:
            return _failure(root, spec, AUDIT_MISMATCH, command, str(error))
        with _append_text(targets['job.log']) as log:
            log.write('\n' + ' '.join(command) + '\n')
            completed = subprocess.run(
                command, cwd=WORKTREE, env=deterministic_env(),
                stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        if completed.returncode:
            return _failure(
                root, spec, completed.returncode, command, 'subprocess failed',
            )
        complete = find_complete(root, expected_tasks, spec)
    if not complete:
        return _failure(
            root, spec, AUDIT_NO_COMPLETE, command, 'no complete run found',
        )
    record = None
    audited_run = None
    error = ValueError('no legitimate complete run found')
    for candidate in reversed(complete):
        try:
            record = audit_completed_run(
                spec, command, candidate, matrix_root, smoke=smoke,
            )
            _validated_launch_started(
                spec, planned, root, plan_path, record['paths'].values(),
            )
        except Exception as candidate_error:
            error = candidate_error
            record = None
            continue
        audited_run = candidate
        break
    if record is None:
        return _failure(
            root, spec, AUDIT_MISMATCH, command, str(error),
        )
    atomic_json(targets['record.json'], record)
    _safe_unlink(targets['FAILED.json'])
    _safe_touch(success)
    print(f'COMPLETE {spec}: {audited_run}')
    return 0


def _reuse_write_targets(matrix_root):
    matrix_root = Path(matrix_root)
    audit_dir = matrix_root / 'reuse_audit'
    required = matrix_root / 'required_jobs.json'
    if matrix_root.is_symlink() or audit_dir.is_symlink() or required.is_symlink():
        raise ValueError('symlinked reuse write target is not allowed')
    return (
        _require_within(audit_dir, matrix_root, 'reuse audit root'),
        _require_within(required, matrix_root, 'required jobs target'),
    )


def audit_reference(dataset, cell, matrix_root):
    spec = f'{dataset}:{cell}:{SEED}'
    source = Path(REFERENCE_RUNS[(dataset, cell)])
    audit_dir, _ = _reuse_write_targets(matrix_root)
    _ensure_directory(audit_dir)
    config_path = source / 'config.json'
    results_path = source / 'results.json'
    manifest_path = source / 'validation/validation_manifest.json'
    checkpoint_path = _final_event(source / 'checkpoints', 'pt')
    head_path = _final_event(source / 'head_consolidation', 'json')
    provenance_path = REFERENCE_PROVENANCE.get(
        (dataset, cell), REFERENCE_PROVENANCE.get(dataset),
    )
    source_paths = {
        'run': str(source), 'config': str(config_path),
        'results': str(results_path), 'validation_manifest': str(manifest_path),
        'final_checkpoint': str(checkpoint_path) if checkpoint_path else None,
        'head_audit': str(head_path) if head_path else None,
        'provenance': str(provenance_path) if provenance_path else None,
    }
    source_paths.update({
        key: str(path)
        for key, path in REFERENCE_AUXILIARY_PATHS.get(
            (dataset, cell), {}
        ).items()
    })
    hashed_paths = {
        key: value for key, value in source_paths.items()
        if key != 'run'
    }
    source_hashes = {
        key: _file_hash(path) if path else None
        for key, path in hashed_paths.items()
    }
    config_payload = _json(config_path)
    results = _json(results_path)
    manifest = _json(manifest_path)
    head = _json(head_path) if head_path else None
    provenance = _json(provenance_path) if provenance_path else None

    claimed_commit = (
        provenance.get('code_commit', provenance.get('source_commit'))
        if provenance else None
    )
    actual_commit = _resolved_commit(claimed_commit)
    launch = _json(source_paths.get('launch_started'))
    frozen_record = _json(source_paths.get('record'))
    expected_reference_hashes = REFERENCE_ARTIFACT_SHA256.get((dataset, cell))
    artifacts = []
    for path, expected_hash in EXPECTED_DATA_ARTIFACTS.get(dataset, {}).items():
        actual_hash = _file_hash(path)
        artifacts.append({
            'path': str(path), 'expected_sha256': expected_hash,
            'actual_sha256': actual_hash, 'passed': actual_hash == expected_hash,
        })

    expected_config = None
    config_mismatches = {'config': {'expected': 'valid JSON object', 'actual': 'missing or malformed'}}
    if config_payload is not None:
        try:
            expected_config = _effective_config(spec, audit_dir)
            if expected_reference_hashes is not None:
                expected_config['data_path'] = str(CANONICAL_SOURCE_ROOT / 'data')
            config_mismatches = _reference_config_mismatches(
                expected_config, config_payload,
            )
        except (OSError, SystemExit, ValueError):
            pass

    expected_factors = CELLS[cell]
    actual_factors = None
    if config_payload is not None:
        actual_factors = (
            config_payload.get('head_consolidation_mode', 'full_classifier'),
            config_payload.get('head_consolidation_lr'),
            config_payload.get('head_consolidation_steps'),
        )
    selection = results.get('selection_audit') if results else None
    expected_evaluation_source = (
        'cifar100-train-validation'
        if dataset == 'cifar100' else 'vector-train-validation'
    )
    metrics_source = results.get('cl_metrics') if results else None
    metrics = {
        key: metrics_source.get(key) if isinstance(metrics_source, dict) else None
        for key in ('AA_final', 'BWT', 'AA_final_taskil')
    }
    fit_flags = _fit_flags(results)
    raw_count = head.get('persistent_raw_example_count') if head else None
    class_count = head.get('class_count') if head else None
    expected_class_count = (
        _expected_seen_classes(dataset, False, expected_config)
        if expected_config is not None else None
    )
    raw_limit_ok = (
        isinstance(raw_count, int) and isinstance(class_count, int)
        and not isinstance(raw_count, bool) and not isinstance(class_count, bool)
        and class_count == expected_class_count
        and 0 <= raw_count <= 20 * expected_class_count
    )
    replay = {
        'replay_selection': head.get('replay_selection') if head else None,
        'samples_per_class': head.get('samples_per_class') if head else None,
        'schedule': head.get('schedule') if head else None,
        'source': head.get('source') if head else None,
        'persistent_embedding_count': head.get('persistent_embedding_count') if head else None,
        'persistent_raw_example_count': raw_count,
        'raw_count_at_most_20_per_class': raw_limit_ok,
        'test_used': head.get('test_used') if head else None,
        'validation_used': head.get('validation_used') if head else None,
    }
    provenance_contract = bool(provenance) and (
        provenance.get('dataset') == dataset
        and provenance.get('seed') == SEED
        and provenance.get('selection_source') == 'training-validation'
        and provenance.get('selection_test_used') is False
    )
    if expected_reference_hashes is not None:
        provenance_contract = provenance_contract and (
            provenance.get('spec') == spec
            and provenance.get('cell') == cell
            and provenance.get('source_commit') == SOURCE_COMMIT
            and provenance.get('implementation_commit')
            == _CIFAR_B_IMPLEMENTATION_COMMIT
            and isinstance(launch, dict)
            and launch.get('spec') == spec
            and launch.get('implementation_commit')
            == provenance.get('implementation_commit')
            and launch.get('plan_sha256')
            == expected_reference_hashes['provenance']
            and isinstance(frozen_record, dict)
            and frozen_record.get('spec') == spec
            and frozen_record.get('source_commit') == SOURCE_COMMIT
            and frozen_record.get('implementation_commit')
            == provenance.get('implementation_commit')
            and frozen_record.get('run_dir')
            == str(REFERENCE_DECLARED_RUNS[(dataset, cell)])
            and frozen_record.get('factors') == list(CELLS[cell])
            and frozen_record.get('validation_hash')
            == EXPECTED_VALIDATION_HASHES[dataset]
            and frozen_record.get('passed') is True
            and isinstance(frozen_record.get('checks'), dict)
            and all(frozen_record['checks'].values())
            and frozen_record.get('sha256') == {
                'checkpoint': expected_reference_hashes['final_checkpoint'],
                'config': expected_reference_hashes['config'],
                'data_flow_audit': expected_reference_hashes['data_flow_audit'],
                'head_audit': expected_reference_hashes['head_audit'],
                'results': expected_reference_hashes['results'],
                'validation_manifest':
                    expected_reference_hashes['validation_manifest'],
            }
        )
    checks = {
        'reference_artifacts': (
            source_hashes == expected_reference_hashes
            if expected_reference_hashes is not None
            else all(source_hashes.values())
        ),
        'source_commit': actual_commit == SOURCE_COMMIT,
        'provenance_contract': provenance_contract,
        'dataset_artifacts': bool(artifacts) and all(item['passed'] for item in artifacts),
        'validation_logical_hash': bool(manifest) and (
            manifest.get('sha256') == EXPECTED_VALIDATION_HASHES.get(dataset)
        ),
        'validation_manifest_label': bool(manifest) and (
            manifest.get('dataset') == EXPECTED_VALIDATION_LABELS.get(dataset)
        ),
        'validation_manifest_contract': _validation_manifest_matches(
            manifest, dataset, expected_config,
        ),
        'effective_config': not config_mismatches,
        'factors': actual_factors == expected_factors,
        'replay_selection': replay == {
            'replay_selection': 'normalized_feature_herding',
            'samples_per_class': 20,
            'schedule': 'final',
            'source': 'balanced_current_encoder_raw_replay',
            'persistent_embedding_count': 0,
            'persistent_raw_example_count': raw_count,
            'raw_count_at_most_20_per_class': True,
            'test_used': False,
            'validation_used': False,
        },
        'selection_audit': bool(selection) and (
            selection.get('passed') is True
            and selection.get('test_used_for_selection') is False
            and selection.get('evaluation_source') == expected_evaluation_source
        ),
        'test_not_used_for_fit': all(value is False for value in fit_flags),
        'metrics': all(value is not None for value in metrics.values()),
    }
    record = {
        'spec': spec, 'dataset': dataset, 'cell': cell, 'seed': SEED,
        'source_paths': source_paths, 'source_sha256': source_hashes,
        'expected_source_commit': SOURCE_COMMIT,
        'claimed_source_commit': claimed_commit,
        'actual_source_commit': actual_commit,
        'dataset_artifacts': artifacts,
        'validation_logical_hash': {
            'expected': EXPECTED_VALIDATION_HASHES.get(dataset),
            'actual': manifest.get('sha256') if manifest else None,
        },
        'validation_manifest_label': {
            'expected': EXPECTED_VALIDATION_LABELS.get(dataset),
            'actual': manifest.get('dataset') if manifest else None,
        },
        'effective_config_mismatches': config_mismatches,
        'factor_values': {
            'expected': list(expected_factors),
            'actual': list(actual_factors) if actual_factors else None,
        },
        'replay_selection': replay,
        'selection_audit': {
            'passed': selection.get('passed') if selection else None,
            'test_used_for_selection': selection.get('test_used_for_selection') if selection else None,
            'expected_evaluation_source': expected_evaluation_source,
            'actual_evaluation_source': selection.get('evaluation_source') if selection else None,
        },
        'test_used_for_fit': fit_flags,
        'metrics': metrics, 'checks': checks,
        'passed': all(checks.values()),
    }
    atomic_json(audit_dir / f'{dataset}_{cell}.json', record)
    return record


def audit_reuse(matrix_root):
    _, required_path = _reuse_write_targets(matrix_root)
    records = [
        audit_reference(dataset, cell, matrix_root)
        for dataset in DATASET_NAMES for cell in ('A', 'B')
    ]
    required = [
        f'{dataset}:{cell}:{SEED}'
        for cell in ('C', 'D') for dataset in DATASET_NAMES
    ]
    required.extend(record['spec'] for record in records if not record['passed'])
    payload = {
        'required_jobs': list(dict.fromkeys(required)),
        'reused_jobs': [record['spec'] for record in records if record['passed']],
    }
    atomic_json(required_path, payload)
    return payload


def _gate_metrics(record):
    metrics = record.get('metrics') if isinstance(record, dict) else None
    if not isinstance(metrics, dict) or set(metrics) != set(METRICS):
        raise ValueError('record does not contain exactly the gate metrics')
    if not all(
        isinstance(metrics[name], (int, float))
        and not isinstance(metrics[name], bool)
        and math.isfinite(metrics[name])
        and -1 <= metrics[name] <= 1
        for name in METRICS
    ):
        raise ValueError('gate metrics must be finite fractions')
    return {name: metrics[name] for name in METRICS}


def _normalize_run_record(path, matrix_root):
    path = Path(path)
    root = path.parent
    if root.is_symlink() or path.is_symlink():
        raise ValueError('symlinked run record is not allowed')
    _require_within(path, matrix_root, 'run record')
    match = re.fullmatch(r'(cifar100|isolet|upmc_food101)_([A-D])_42', root.name)
    if not match or root.parent.resolve() != (Path(matrix_root) / 'runs').resolve():
        raise ValueError('run record is outside its exact job root')
    dataset, cell = match.groups()
    spec = f'{dataset}:{cell}:{SEED}'
    record = _json(path)
    success = root / 'SUCCESS'
    if success.is_symlink() or not success.is_file() or (root / 'FAILED.json').exists():
        raise ValueError('run record has no unambiguous success evidence')
    _require_within(success, root, 'run success')
    checks = record.get('checks') if record else None
    if not record or not (
        record.get('passed') is True
        and record.get('smoke') is False
        and record.get('spec') == spec
        and record.get('dataset') == dataset
        and record.get('cell') == cell
        and record.get('seed') == SEED
        and record.get('factors') == list(CELLS[cell])
        and record.get('implementation_commit') == _implementation_commit()
        and record.get('source_commit') == SOURCE_COMMIT
        and record.get('validation_hash') == EXPECTED_VALIDATION_HASHES[dataset]
        and isinstance(checks, dict) and checks
        and all(value is True for value in checks.values())
        and _valid_record(root, spec, matrix_root, False)
    ):
        raise ValueError('run record failed its strict Task 4 contract')
    return {
        'spec': spec, 'dataset': dataset, 'cell': cell, 'seed': SEED,
        'factors': list(CELLS[cell]), 'metrics': _gate_metrics(record),
        'validation_hash': record['validation_hash'],
        'source_type': 'run', 'source_path': str(path),
    }


def _normalize_reuse_record(path, matrix_root):
    path = Path(path)
    reuse_root = Path(matrix_root) / 'reuse_audit'
    if reuse_root.is_symlink() or path.is_symlink():
        raise ValueError('symlinked reuse record is not allowed')
    _require_within(path, reuse_root, 'reuse record')
    match = re.fullmatch(r'(cifar100|isolet|upmc_food101)_([AB])\.json', path.name)
    if not match or path.parent.resolve() != reuse_root.resolve():
        raise ValueError('reuse record has an invalid key')
    dataset, cell = match.groups()
    spec = f'{dataset}:{cell}:{SEED}'
    record = _json(path)
    checks = record.get('checks') if record else None
    validation = record.get('validation_logical_hash') if record else None
    factors = record.get('factor_values') if record else None
    identity_matches = record and (
        record.get('spec') == spec
        and record.get('dataset') == dataset
        and record.get('cell') == cell
        and record.get('seed') == SEED
    )
    if record and record.get('passed') is False:
        if not (
            identity_matches and isinstance(checks, dict) and checks
            and all(isinstance(value, bool) for value in checks.values())
            and any(value is False for value in checks.values())
        ):
            raise ValueError('rejected reuse record has invalid audit evidence')
        return None
    if not record or not (
        record.get('passed') is True
        and identity_matches
        and record.get('expected_source_commit') == SOURCE_COMMIT
        and record.get('actual_source_commit') == SOURCE_COMMIT
        and isinstance(checks, dict) and checks
        and all(value is True for value in checks.values())
        and validation == {
            'expected': EXPECTED_VALIDATION_HASHES[dataset],
            'actual': EXPECTED_VALIDATION_HASHES[dataset],
        }
        and factors == {
            'expected': list(CELLS[cell]), 'actual': list(CELLS[cell]),
        }
    ):
        raise ValueError('reuse record failed its strict Task 3 contract')
    with tempfile.TemporaryDirectory(prefix='.gate-reuse-', dir=matrix_root) as audit_root:
        if audit_reference(dataset, cell, audit_root) != record:
            raise ValueError('reuse record does not match a fresh strict audit')
    return {
        'spec': spec, 'dataset': dataset, 'cell': cell, 'seed': SEED,
        'factors': list(CELLS[cell]), 'metrics': _gate_metrics(record),
        'validation_hash': validation['actual'],
        'source_type': 'reuse', 'source_path': str(path),
    }


def load_matrix_records(matrix_root):
    matrix_root = Path(matrix_root)
    _ensure_directory(matrix_root)
    expected = {
        (dataset, cell) for dataset in DATASET_NAMES for cell in CELLS
    }
    accepted = {}
    conflicted = set()
    errors = []
    candidates = []
    reuse_root = matrix_root / 'reuse_audit'
    runs_root = matrix_root / 'runs'
    for root, pattern, normalizer in (
        (reuse_root, '*.json', _normalize_reuse_record),
        (runs_root, '*/record.json', _normalize_run_record),
    ):
        if root.is_symlink():
            errors.append({'source_path': str(root), 'reason': 'symlinked source root'})
            continue
        if root.is_dir():
            candidates.extend(
                (path, normalizer) for path in sorted(root.glob(pattern))
            )
    for path, normalizer in candidates:
        try:
            normalized = normalizer(path, matrix_root)
            if normalized is None:
                continue
            key = (normalized['dataset'], normalized['cell'])
            if key in accepted or key in conflicted:
                accepted.pop(key, None)
                conflicted.add(key)
                errors.append({
                    'source_path': str(path),
                    'reason': f'duplicate accepted source for {key[0]}:{key[1]}:42',
                })
            else:
                accepted[key] = normalized
        except (Exception, SystemExit) as error:
            errors.append({'source_path': str(path), 'reason': str(error)})
    for dataset, cell in sorted(expected - set(accepted)):
        errors.append({
            'source_path': None,
            'reason': f'missing accepted source for {dataset}:{cell}:42',
        })
    for dataset in DATASET_NAMES:
        hashes = {
            accepted[(dataset, cell)]['validation_hash']
            for cell in CELLS if (dataset, cell) in accepted
        }
        expected_hash = EXPECTED_VALIDATION_HASHES[dataset]
        if hashes and hashes != {expected_hash}:
            errors.append({
                'source_path': None,
                'reason': f'validation hashes are not frozen and identical for {dataset}',
            })
    errors.sort(key=lambda item: (item['source_path'] or '', item['reason']))
    sources = [
        {
            key: record[key]
            for key in ('spec', 'source_type', 'source_path', 'validation_hash')
        }
        for _, record in sorted(accepted.items())
    ]
    return {
        'complete': not errors and set(accepted) == expected,
        'by_key': accepted, 'accepted_sources': sources, 'errors': errors,
    }


def eligibility(cell, by_key):
    datasets = {}
    failures = []
    for dataset in DATASET_NAMES:
        incumbent_cell = INCUMBENTS[dataset]
        incumbent_record = by_key.get((dataset, incumbent_cell))
        candidate_record = by_key.get((dataset, cell))
        incumbent = incumbent_record.get('metrics') if incumbent_record else None
        candidate = candidate_record.get('metrics') if candidate_record else None
        exact_floor = {
            metric: Decimal(str(incumbent[metric])) - Decimal(str(TOLERANCE))
            if incumbent and metric in incumbent else None
            for metric in METRICS
        }
        floor = {
            metric: float(exact_floor[metric])
            if exact_floor[metric] is not None else None
            for metric in METRICS
        }
        checks = {}
        for metric in METRICS:
            passed = bool(
                candidate and metric in candidate and floor[metric] is not None
                and Decimal(str(candidate[metric])) >= exact_floor[metric]
            )
            checks[metric] = passed
            if not passed:
                failures.append({
                    'dataset': dataset, 'metric': metric,
                    'candidate': candidate.get(metric) if candidate else None,
                    'floor': floor[metric],
                })
        datasets[dataset] = {
            'incumbent_cell': incumbent_cell,
            'incumbent': {
                metric: incumbent.get(metric) if incumbent else None
                for metric in METRICS
            },
            'floor': floor,
            'candidate': {
                metric: candidate.get(metric) if candidate else None
                for metric in METRICS
            },
            'checks': checks,
        }
    return {
        'eligible': not failures, 'datasets': datasets, 'failures': failures,
    }


def macro_metrics(cell, by_key):
    if any((dataset, cell) not in by_key for dataset in DATASET_NAMES):
        return {metric: None for metric in METRICS}
    return {
        metric: sum(
            by_key[(dataset, cell)]['metrics'][metric]
            for dataset in DATASET_NAMES
        ) / len(DATASET_NAMES)
        for metric in METRICS
    }


def _atomic_text(path, content):
    _write_text(path, content)


def _gate_paths(matrix_root):
    matrix_root = Path(matrix_root)
    _ensure_directory(matrix_root)
    report = matrix_root / 'formal_report'
    if report.is_symlink():
        raise ValueError('symlinked formal report root is not allowed')
    paths = {
        name: report / name
        for name in ('PER_CELL.csv', 'FACTORIAL_SUMMARY.json', 'GATE.json')
    }
    paths.update({
        name: matrix_root / name
        for name in ('EXECUTION_SUCCESS', 'GATE_SUCCESS', 'GATE_FAILED')
    })
    if any(path.is_symlink() for path in paths.values()):
        raise ValueError('symlinked gate target is not allowed')
    _ensure_directory(report)
    return {
        name: _require_within(path, matrix_root, name)
        for name, path in paths.items()
    }


def _per_cell_csv(cells):
    fields = ['cell', 'mode', 'lr', 'steps', 'eligible', *(
        f'macro_{metric}' for metric in METRICS
    ), 'failures']
    for dataset in DATASET_NAMES:
        for metric in METRICS:
            fields.extend((
                f'{dataset}_{metric}_incumbent', f'{dataset}_{metric}_floor',
                f'{dataset}_{metric}_candidate', f'{dataset}_{metric}_passed',
            ))
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for cell in CELLS:
        item = cells[cell]
        row = {
            'cell': cell, **item['factors'], 'eligible': item['eligible'],
            **{
                f'macro_{metric}': item['macro_metrics'][metric]
                for metric in METRICS
            },
            'failures': json.dumps(
                item['failures'], sort_keys=True, separators=(',', ':'),
            ),
        }
        for dataset in DATASET_NAMES:
            values = item['datasets'][dataset]
            for metric in METRICS:
                row[f'{dataset}_{metric}_incumbent'] = values['incumbent'][metric]
                row[f'{dataset}_{metric}_floor'] = values['floor'][metric]
                row[f'{dataset}_{metric}_candidate'] = values['candidate'][metric]
                row[f'{dataset}_{metric}_passed'] = values['checks'][metric]
        writer.writerow(row)
    return output.getvalue()


def summarize(matrix_root):
    paths = _gate_paths(matrix_root)
    for name in ('EXECUTION_SUCCESS', 'GATE_SUCCESS', 'GATE_FAILED'):
        _safe_unlink(paths[name])
    loaded = load_matrix_records(matrix_root)
    by_key = loaded['by_key']
    cells = {}
    for cell, (mode, lr, steps) in CELLS.items():
        result = eligibility(cell, by_key)
        cells[cell] = {
            'factors': {'mode': mode, 'lr': lr, 'steps': steps},
            'datasets': result['datasets'], 'failures': result['failures'],
            'eligible': loaded['complete'] and result['eligible'],
            'macro_metrics': macro_metrics(cell, by_key),
        }
    eligible = [cell for cell in CELLS if cells[cell]['eligible']]
    ranked = sorted(eligible, key=lambda cell: (
        -cells[cell]['macro_metrics']['AA_final'],
        -cells[cell]['macro_metrics']['BWT'],
        -cells[cell]['macro_metrics']['AA_final_taskil'],
        cell,
    ))
    selected = ranked[0] if ranked else None
    payload = {
        'matrix_complete': loaded['complete'],
        'errors': loaded['errors'],
        'accepted_sources': loaded['accepted_sources'],
        'seed': SEED, 'tolerance': TOLERANCE,
        'metrics': list(METRICS), 'incumbents': dict(INCUMBENTS),
        'cells': cells, 'eligible_cells': eligible,
        'selected_cell': selected,
        'selected_config': cells[selected]['factors'] if selected else None,
        'selection_source': 'training-validation',
        'test_used_for_selection': False,
    }
    summary = dict(payload, report='factorial_summary')
    gate = dict(payload, report='gate')
    _atomic_text(paths['PER_CELL.csv'], _per_cell_csv(cells))
    atomic_json(paths['FACTORIAL_SUMMARY.json'], summary)
    atomic_json(paths['GATE.json'], gate)
    if not loaded['complete']:
        return 2
    _safe_touch(paths['EXECUTION_SUCCESS'])
    if selected is None:
        _safe_touch(paths['GATE_FAILED'])
        return 3
    _safe_touch(paths['GATE_SUCCESS'])
    return 0


def static_check():
    assert SEED == 42
    assert tuple(DATASET_NAMES) == ('cifar100', 'isolet', 'upmc_food101')
    assert set(EXPECTED_TASKS) == set(DATASET_NAMES)
    assert len(all_specs()) == 12 and len(set(all_specs())) == 12
    for spec in all_specs():
        dataset, cell, seed = parse_spec(spec)
        command = build_command(spec, 'cuda:0', Path('/tmp/factorial-static-check'))
        flags = dict(zip(command[2::2], command[3::2]))
        assert seed == SEED and flags['--seed'] == str(SEED)
        assert flags['--head_consolidation_schedule'] == 'final'
        assert flags['--head_consolidation_mode'] == CELLS[cell][0]
        assert dataset in EXPECTED_TASKS
    print('UNIFIED_HEAD_FACTORIAL_STATIC_CHECK_OK')


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('check')
    audit_parser = subparsers.add_parser('audit-reuse')
    audit_parser.add_argument('--matrix-root', required=True, type=Path)
    jobs_parser = subparsers.add_parser('jobs')
    jobs_parser.add_argument('--matrix-root', required=True, type=Path)
    jobs_parser.add_argument('--worker', type=int)
    jobs_parser.add_argument('--workers', type=int, default=2)
    run_parser = subparsers.add_parser('run-job')
    run_parser.add_argument('spec')
    run_parser.add_argument('--device', required=True)
    run_parser.add_argument('--matrix-root', required=True, type=Path)
    run_parser.add_argument('--smoke', action='store_true')
    summarize_parser = subparsers.add_parser('summarize')
    summarize_parser.add_argument('--matrix-root', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == 'check':
        static_check()
        return 0
    if args.command == 'audit-reuse':
        audit_reuse(args.matrix_root)
        return 0
    if args.command == 'jobs':
        payload = _json(args.matrix_root / 'required_jobs.json')
        required = payload.get('required_jobs') if payload else None
        if not isinstance(required, list):
            raise ValueError('required_jobs.json has no required_jobs list')
        for spec in required:
            parse_spec(spec)
        selected = jobs(args.worker, args.workers, required)
        for spec in selected:
            print(spec)
        return 0
    if args.command == 'summarize':
        return summarize(args.matrix_root)
    return run_job(args.spec, args.device, args.matrix_root, smoke=args.smoke)


if __name__ == '__main__':
    raise SystemExit(main())
