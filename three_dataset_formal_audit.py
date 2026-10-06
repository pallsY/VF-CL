"""Fail-closed, read-only admission of explicitly declared formal evidence."""
import ast
from contextlib import ExitStack, nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import struct
import subprocess
import sys
import tempfile

from three_dataset_formal_metrics import FORMULA_VERSION, reconstruct_metrics
from three_dataset_formal_registry import (
    FormalSpec, OPTION_SCHEMA, command_for, explanation_specs, formal_specs,
    experiment_profile, RECOVERY_PROFILE, SINGLE_DATASET_PROFILE,
    CONTINUATION_PROFILE, METHOD_SHARD_PROFILE,
    selected_formal_dataset, selected_formal_method,
    parsed_protocol, protocol_for, registry_sha256, safe_spec_name,
    validation_access_for,
)
from adaptive_consolidation_audit import FORMAL_SOURCE_FILES


_DECLARATION_KEYS = {
    'spec', 'run_dir', 'code_commit', 'source_files', 'data_files',
    'validation_manifest', 'config', 'results', 'checkpoint',
    'formal_artifacts',
}
_ENTRY_KEYS = {'path', 'root', 'sha256'}
_TREE_ENTRY_KEYS = _ENTRY_KEYS | {'logical_path'}
_RESULT_DECLARATION_KEYS = _ENTRY_KEYS | {'data_flow'}
_SHA256 = re.compile(r'[0-9a-f]{64}')
_COMMIT = re.compile(r'[0-9a-f]{40}')
_SOURCE_INVENTORY = FORMAL_SOURCE_FILES
_AUTHORITATIVE_DATA = {
    'cifar100': {
        'cifar-100-python/meta':
            'a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9',
        'cifar-100-python/test':
            '4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864',
        'cifar-100-python/train':
            '735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b',
    },
    'isolet': {
        'isolet/isolet_vfl.npz':
            'd34312670de93198afcd2b126c95b79bae2b4cffeb30d480f097faf046b69514',
        'isolet/isolet_vfl.metadata.json':
            '79396dea1751b6094a5769f2dd789ad58b3ea9582a12c8c98d3ec07d8d1eb3cd',
    },
    'upmc_food101': {
        'upmc_food101/upmc_food101_vfl.npz':
            '9268de2e698e25138d533bac47ae1f9a763d73518cfe6c785eda0a8232d801cf',
        'upmc_food101/upmc_food101_vfl.metadata.json':
            '14e996a39a7aacb7f04e3053bef2586a2b3c1690f88b3f72fd01e911b5e4a821',
    },
}
_AUTHORITATIVE_MANIFEST = {
    'cifar100': {
        'logical_path': 'validation/validation_manifest.json',
        'dataset': 'cifar100-train',
        'sha256':
            '0aa4729ade65021ce774c757516584c4d2a2ce70d2879b045db47dbceae917fa',
        'seed': 20260729, 'per_class': 25,
    },
    'isolet': {
        'logical_path': 'validation/validation_manifest.json',
        'dataset': 'isolet_vfl.npz-train',
        'sha256':
            '487e81663a12d4a663a1f421407aa3f88d788cc3c83f7323166cf7aff82902d3',
        'seed': 20260809, 'per_class': 40,
    },
    'upmc_food101': {
        'logical_path': 'validation/validation_manifest.json',
        'dataset': 'upmc_food101_vfl.npz-train',
        'sha256':
            'b812e99da856d94ee5b6eea28eadaf9924f52e355d44941c42da4028c0895e06',
        'seed': 20260809, 'per_class': 64,
    },
}
_TRACKER_KEYS = {
    'cl_metrics', 'task_acc_history', 'ul_metrics', 'comm_stats', 'timing',
    'step_results',
}
_BASE_RESULT_KEYS = _TRACKER_KEYS | {'config', 'source_provenance'}
_BIC_RESULT_KEYS = {
    'bic_history', 'bic_final', 'calibration_audit', 'bic_fit_corpus',
}
_RESULT_CONFIG_KEYS = {'cl_method', 'ul_method', 'data', 'num_tasks', 'seed'}
_SELECTION_KEYS = {
    'passed', 'calibration_manifest_sha256', 'validation_manifest_sha256',
    'calibration_per_class', 'validation_per_class', 'training_count',
    'calibration_count', 'validation_count',
    'training_calibration_overlap_count',
    'training_validation_overlap_count',
    'calibration_validation_overlap_count', 'evaluation_source',
    'test_used_for_selection',
}
_COMPLETED_PLAN_KEYS = {
    'kind', 'registry_sha256', 'metric_formula_version', 'formal_cells',
    'missing_jobs', 'explanation_cells', 'census_sha256',
}
_JOB_SPEC_KEYS = {
    'kind', 'spec_key', 'spec', 'registry_sha256',
    'metric_formula_version', 'plan_sha256', 'run_dir', 'source_commit',
    'source_sha256', 'command', 'command_sha256', 'root_identity',
}
_CLAIM_OWNER_KEYS = {
    'kind', 'job', 'launcher_token', 'worker_role', 'pid',
    'pgid', 'phase', 'process_start_time', 'source_commit', 'root_identity',
}
_LAUNCH_STARTED_KEYS = {
    'kind', 'spec_key', 'plan_sha256', 'job_spec_sha256', 'claim_sha256',
    'command_sha256', 'source_commit', 'root_identity', 'worker_role', 'pid',
    'process_start_time', 'pgid', 'phase',
}
_RESOURCE_EVIDENCE_KEYS = {
    'kind', 'spec_key', 'plan_sha256', 'job_spec_sha256', 'claim_sha256',
    'launch_sha256', 'command_sha256', 'artifact_sha256', 'resource',
}
_RESOURCE_KEYS = {
    'hardware_identity', 'instrumentation', 'runtime_seconds',
    'peak_gpu_memory_bytes', 'checkpoint_size_bytes', 'added_parameters',
    'communication_bytes', 'replay_type', 'raw_examples_per_class',
    'persistent_embeddings', 'privacy_label',
}
_HARDWARE_KEYS = {'gpu_name', 'gpu_count', 'cuda', 'torch', 'driver'}
_ROOT_IDENTITY_KEYS = {'dev', 'inode', 'ctime_ns', 'size', 'hash'}


@dataclass(frozen=True)
class AdmissionRecord:
    status: str
    reason: str
    spec: dict
    protocol_sha256: str
    source_sha256: str
    artifact_sha256: dict
    metrics: object
    metric_formula_version: str
    trajectory_sha256: str


class _EvidenceError(ValueError):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _reject(reason):
    raise _EvidenceError('REJECTED', reason)


def _rerun(reason):
    raise _EvidenceError('RERUN_REQUIRED', reason)


def _canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode()


def _normalized_spec(spec):
    if not isinstance(spec, FormalSpec):
        raise TypeError('spec must be a FormalSpec')
    return asdict(spec)


def _protocol(spec):
    value = json.loads(json.dumps(protocol_for(spec)))
    return value, hashlib.sha256(_canonical_json(value)).hexdigest()


def _failure(status, reason, spec, protocol_sha256):
    return AdmissionRecord(
        status, reason, spec, protocol_sha256, '', {}, None,
        FORMULA_VERSION, '',
    )


def _absolute(value, reason):
    if type(value) is not str or not value or not Path(value).is_absolute():
        _reject(reason)
    return Path(os.path.abspath(value))


def _logical(value):
    if type(value) is not str or not value:
        _reject('logical-path-invalid')
    path = PurePosixPath(value)
    if (path.is_absolute() or path.as_posix() != value
            or any(part in ('', '.', '..') for part in path.parts)):
        _reject('logical-path-invalid')
    return value


def _identity(value):
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
        value.st_uid, value.st_gid, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )


def _directory_identity(value):
    return value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid


def _open_directory(path):
    path = _absolute(os.fspath(path), 'root-not-absolute')
    flags = (os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
             | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0))
    try:
        descriptor = os.open(os.path.sep, flags)
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except OSError:
        try:
            os.close(descriptor)
        except (OSError, UnboundLocalError):
            pass
        _reject('root-invalid')
    return path, descriptor


class _PinnedRoot:
    def __init__(self, path):
        self.path, self.fd = _open_directory(path)
        self.details = os.fstat(self.fd)
        if not stat.S_ISDIR(self.details.st_mode):
            os.close(self.fd)
            _reject('root-invalid')

    def close(self):
        os.close(self.fd)

    def verify(self):
        _, probe = _open_directory(self.path)
        try:
            if (_directory_identity(os.fstat(probe))
                    != _directory_identity(self.details)):
                _reject('root-identity-changed')
        finally:
            os.close(probe)

    def open_file(self, logical):
        parts = PurePosixPath(_logical(logical)).parts
        directory = os.dup(self.fd)
        dir_flags = (os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
                     | getattr(os, 'O_NOFOLLOW', 0)
                     | getattr(os, 'O_CLOEXEC', 0))
        try:
            for component in parts[:-1]:
                child = os.open(component, dir_flags, dir_fd=directory)
                os.close(directory)
                directory = child
            flags = (os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
                     | getattr(os, 'O_CLOEXEC', 0))
            descriptor = os.open(parts[-1], flags, dir_fd=directory)
            named = os.stat(parts[-1], dir_fd=directory, follow_symlinks=False)
        except OSError:
            _reject('evidence-file-invalid')
        finally:
            os.close(directory)
        if (not stat.S_ISREG(named.st_mode)
                or _identity(named) != _identity(os.fstat(descriptor))):
            os.close(descriptor)
            _reject('evidence-file-invalid')
        return descriptor

    def verify_file_name(self, logical, expected):
        parts = PurePosixPath(_logical(logical)).parts
        directory = os.dup(self.fd)
        flags = (os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
                 | getattr(os, 'O_NOFOLLOW', 0)
                 | getattr(os, 'O_CLOEXEC', 0))
        try:
            for component in parts[:-1]:
                child = os.open(component, flags, dir_fd=directory)
                os.close(directory)
                directory = child
            current = os.stat(
                parts[-1], dir_fd=directory, follow_symlinks=False)
        except OSError:
            _reject('evidence-name-changed')
        finally:
            os.close(directory)
        if _identity(current) != _identity(expected):
            _reject('evidence-name-changed')


def _read_descriptor(descriptor):
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        _reject('evidence-file-invalid')
    content = bytearray()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        content.extend(chunk)
    after = os.fstat(descriptor)
    if _identity(before) != _identity(after):
        _reject('evidence-identity-changed')
    return bytes(content), after


def _entry_shape(entry, tree, label):
    expected = _TREE_ENTRY_KEYS if tree else _ENTRY_KEYS
    if type(entry) is not dict or set(entry) != expected:
        _reject(f'{label}-schema-invalid')
    digest = entry['sha256']
    if type(digest) is not str or _SHA256.fullmatch(digest) is None:
        _reject(f'{label}-hash-invalid')


def _read_entry(entry, root, logical, tree, label, inode_seen):
    _entry_shape(entry, tree, label)
    logical = _logical(logical)
    if tree and entry['logical_path'] != logical:
        _reject(f'{label}-logical-mismatch')
    if _absolute(entry['root'], f'{label}-root-invalid') != root.path:
        _reject(f'{label}-root-mismatch')
    if _absolute(entry['path'], f'{label}-path-invalid') != root.path / logical:
        _reject(f'{label}-path-escape')
    root.verify()
    descriptor = root.open_file(logical)
    try:
        content, details = _read_descriptor(descriptor)
        root.verify_file_name(logical, details)
    finally:
        os.close(descriptor)
    root.verify()
    inode = (details.st_dev, details.st_ino)
    if inode in inode_seen:
        _reject('duplicate-evidence-file')
    inode_seen.add(inode)
    digest = hashlib.sha256(content).hexdigest()
    if digest != entry['sha256']:
        _reject(f'{label}-hash-mismatch')
    return content, digest


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON member: {key}')
        result[key] = value
    return result


def _json_value(content, label, require_object=True):
    def invalid_constant(_):
        raise ValueError('non-finite JSON number')

    try:
        value = json.loads(
            content.decode('utf-8'), object_pairs_hook=_strict_object,
            parse_constant=invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        _reject(f'{label}-json-invalid')
    if require_object and type(value) is not dict:
        _reject(f'{label}-schema-invalid')
    return value


def _schema_matches(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(
            _schema_matches(actual[key], expected[key]) for key in expected
        )
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _schema_matches(left, right) for left, right in zip(actual, expected)
        )
    return True


def _config_value(name, value, expected):
    kind = OPTION_SCHEMA.get(name)
    if name == 'unlearn_after_tasks':
        if (type(value) is not list or len(value) != 1
                or type(value[0]) is not int):
            raise ValueError('invalid producer unlearn task list')
        return value[0]
    if name == 'unlearn_classes':
        if (type(value) is not list or len(value) != 1
                or type(value[0]) is not list or len(value[0]) != 1
                or type(value[0][0]) is not int):
            raise ValueError('invalid producer unlearn class list')
        return value[0][0]
    if kind == 'bool':
        if name == 'formal_deferred_evaluation':
            if type(value) is not bool:
                raise ValueError('invalid producer formal boolean')
            return value
        if type(value) is not int or value not in (0, 1):
            raise ValueError('invalid producer boolean')
        return bool(value)
    if kind == 'text_bool':
        if type(value) is not bool:
            raise ValueError('invalid producer text boolean')
        return value
    if kind == 'int':
        if type(value) is not int:
            raise ValueError('invalid producer integer')
        return value
    if kind == 'float':
        if type(value) is not float or not math.isfinite(value):
            raise ValueError('invalid producer float')
        return value
    if kind == 'str':
        if type(value) is not str:
            raise ValueError('invalid producer string')
        return value
    if kind == 'tasks':
        if type(value) is not str:
            raise ValueError('invalid producer task classes')
        return [
            [int(class_id) for class_id in task.split(',')]
            for task in value.split('|')
        ]
    if type(value) is not type(expected):
        raise ValueError('unsupported producer config value')
    return value


def _producer_config_schema(protocol):
    tree = ast.parse(Path(__file__).with_name('config.py').read_text(
        encoding='utf-8-sig'
    ))
    schema = {}
    for node in ast.walk(tree):
        if (not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Attribute)
                or node.func.attr != 'add_argument'):
            continue
        options = [
            argument.value for argument in node.args
            if isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
            and argument.value.startswith('--')
        ]
        if not options:
            continue
        destination = next((
            keyword.value.value for keyword in node.keywords
            if keyword.arg == 'dest'
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        ), None)
        name = destination or options[0][2:].replace('-', '_')
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        action = keywords.get('action')
        converter = keywords.get('type')
        default = keywords.get('default')
        if (isinstance(action, ast.Constant)
                and action.value == 'store_true'):
            expected_type = bool
        elif isinstance(default, ast.Constant):
            expected_type = type(default.value)
        elif isinstance(converter, ast.Name) and converter.id in {
                'int', 'float', 'str'}:
            expected_type = {'int': int, 'float': float, 'str': str}[
                converter.id]
        elif isinstance(converter, ast.Lambda):
            expected_type = bool
        else:
            expected_type = type(None)
        schema[name] = expected_type
    emitted = {
        'bool': int, 'text_bool': bool, 'int': int, 'float': float,
        'tasks': str, 'path': str, 'str': str,
    }
    for name in protocol['base_options']:
        if name in OPTION_SCHEMA:
            schema[name] = emitted[OPTION_SCHEMA[name]]
    schema.update({
        'formal_deferred_evaluation': bool,
        'unlearn_after_tasks': list,
        'unlearn_classes': list,
        'party_widths': type(None),
        'party_weight_manifest_hash': str,
        'img_size': int,
        'output_dir': str,
    })
    if protocol['base_options']['data'] == 'tabvfl':
        schema['party_col_ranges'] = type(None)
    return schema


def _validate_producer_config_types(config, protocol):
    schema = _producer_config_schema(protocol)
    if type(config) is not dict or set(config) != set(schema):
        _reject('config-schema-invalid')
    for name, expected_type in schema.items():
        value = config[name]
        if name == 'unlearn_after_tasks':
            valid = (type(value) is list and bool(value)
                     and all(type(item) is int for item in value))
        elif name == 'unlearn_classes':
            valid = (type(value) is list and bool(value)
                     and all(type(group) is list and bool(group)
                             and all(type(item) is int for item in group)
                             for group in value))
        else:
            valid = type(value) is expected_type
        if not valid:
            _reject('config-schema-invalid')


def _validate_producer_config(config, protocol, run_root, data_root,
                              data_files):
    _validate_producer_config_types(config, protocol)
    options = protocol['base_options']
    deployment = {'data_path', 'vector_npz', 'device', 'results_dir', 'exp_name'}
    try:
        scientific = {
            key: _config_value(key, config[key], expected)
            for key, expected in options.items() if key not in deployment
        }
    except (KeyError, TypeError, ValueError):
        _reject('config-schema-invalid')
    expected_scientific = {
        key: value for key, value in options.items() if key not in deployment
    }
    if scientific != expected_scientific:
        _rerun('protocol-mismatch')
    if (config.get('formal_deferred_evaluation') is not True
            or type(config.get('output_dir')) is not str
            or _absolute(config['output_dir'], 'config-output-invalid')
                != run_root.path
            or type(config.get('resume_run_dir')) is not str
            or config['resume_run_dir']
            or type(config.get('data_path')) is not str
            or _absolute(config['data_path'], 'config-data-invalid')
                != data_root.path
            or type(config.get('device')) is not str
            or not config['device']
            or type(config.get('results_dir')) is not str
            or not Path(config['results_dir']).is_absolute()
            or type(config.get('exp_name')) is not str
            or not config['exp_name']
            or run_root.path.parent != Path(config['results_dir'])
            or run_root.path.name != config['exp_name']):
        _reject('config-schema-invalid')
    vector = next((entry for entry in data_files
                   if entry.get('logical_path', '').endswith('.npz')), None)
    if (vector is not None
            and (type(config.get('vector_npz')) is not str
                 or _absolute(config['vector_npz'], 'config-vector-invalid')
                    != Path(vector['path']))):
        _reject('config-schema-invalid')


def _formal_access_plan(protocol):
    spec = FormalSpec(
        protocol['dataset'], protocol['method'], protocol['seed'],
        protocol['explanation'],
    )
    return (
        [('validation', 'lambda_validation',
          'final_validation_pre_install')]
        if validation_access_for(spec) else []
    ) + (
        [('calibration', 'bic_calibration',
          'final_bic_calibration_post_freeze')]
        if (protocol['base_options'].get('bic_enabled') is True
            and not validation_access_for(spec)) else []
    ) + [('test', 'test', 'final_test_post_install')]


def _formal_artifact_logicals(protocol):
    count = len(_task_classes(protocol))
    logicals = []
    for event_idx in range(count):
        logicals.extend((
            f'checkpoints/event_{event_idx}_CIL.pt',
            f'formal_snapshots/event_{event_idx}_CIL.pt',
        ))
    logicals.extend(
        f'formal_access/{split}.consumed.json'
        for split, _prefix, _phase in _formal_access_plan(protocol)
    )
    logicals.extend((
        'FORMAL_STATE_FROZEN.json',
        'FORMAL_EVALUATION_PENDING.json',
        'FORMAL_EVALUATION_CONSUMING.json',
        'FORMAL_EVALUATION_COMPLETE.json',
        'FORMAL_EVALUATION_SEALED.json',
        'FORMAL_EVALUATION_PUBLISHING.json',
        'FORMAL_EVALUATION_PUBLISHED.json',
    ))
    return tuple(logicals)


def _verify_formal_artifacts(entries, root, protocol, inode_seen, artifacts):
    expected = _formal_artifact_logicals(protocol)
    if (type(entries) is not list or len(entries) != len(expected)
            or any(type(entry) is not dict for entry in entries)):
        _reject('formal-artifacts-schema-invalid')
    by_logical = {entry.get('logical_path'): entry for entry in entries}
    if len(by_logical) != len(entries) or set(by_logical) != set(expected):
        _reject('formal-artifacts-schema-invalid')
    contents = {}
    for logical in expected:
        content, digest = _read_entry(
            by_logical[logical], root, logical, True,
            'formal-artifact', inode_seen,
        )
        contents[logical] = content
        artifacts[f'formal:{logical}'] = digest
    return contents


def _formal_checkpoint_protocol(protocol, source_provenance):
    options = protocol['base_options']
    value = {
        'seed': protocol['seed'], 'data': options['data'],
        'cl_method': options['cl_method'],
        'num_tasks': options['num_tasks'],
        'num_parties': options['num_parties'],
        'head_consolidation_enabled': int(
            options.get('head_consolidation_enabled', False)),
        'head_consolidation_mode': options.get('head_consolidation_mode'),
        'formal_deferred_evaluation': True,
        'source_provenance': source_provenance,
    }
    if options['cl_method'] == 'er_ace':
        value.update({
            'num_classes': options['num_classes'],
            'er_ace_buffer_size': options['er_ace_buffer_size'],
            'er_ace_batch': options['er_ace_batch'],
        })
    if options['cl_method'] == 'fedprotip_vfl':
        value.update({
            'fedprotip_tip_threshold': float(
                options['fedprotip_tip_threshold']),
            'fedprotip_max_batches': int(options['fedprotip_max_batches']),
        })
    if options.get('bic_enabled') is True:
        value.update({
            'bic_enabled': True,
            'bic_fit_mode': options.get('bic_fit_mode'),
            'bic_lr': float(options['bic_lr']),
            'bic_steps': int(options['bic_steps']),
            'bic_per_class': int(options['bic_per_class']),
            'lambda_validation_enabled': bool(
                options.get('lambda_validation_enabled')),
            'lambda_validation_per_class': int(
                options['lambda_validation_per_class']),
            'lambda_validation_split_seed': int(
                options['lambda_validation_split_seed']),
        })
    return value


def _validate_cache_identity(value):
    if (type(value) is not dict
            or set(value) != {'batch_count', 'sample_count', 'batches'}
            or type(value['batch_count']) is not int
            or type(value['sample_count']) is not int
            or type(value['batches']) is not list
            or value['batch_count'] != len(value['batches'])
            or value['batch_count'] <= 0 or value['sample_count'] <= 0):
        _reject('formal-access-cache-invalid')
    samples = 0
    for batch in value['batches']:
        if (type(batch) is not dict or set(batch) != {
                'input_sha256', 'label_sha256', 'input_shape', 'label_shape',
                'input_dtype', 'label_dtype'}
                or any(type(batch[key]) is not str
                       or _SHA256.fullmatch(batch[key]) is None
                       for key in ('input_sha256', 'label_sha256'))
                or any(type(batch[key]) is not list
                       or not batch[key]
                       or any(type(item) is not int or item < 0
                              for item in batch[key])
                       for key in ('input_shape', 'label_shape'))
                or len(batch['label_shape']) != 1
                or batch['input_shape'][0] != batch['label_shape'][0]
                or any(type(batch[key]) is not str or not batch[key]
                       for key in ('input_dtype', 'label_dtype'))):
            _reject('formal-access-cache-invalid')
        samples += batch['label_shape'][0]
    if samples != value['sample_count']:
        _reject('formal-access-cache-invalid')


def _validate_formal_access_artifacts(
        contents, protocol, data_flow_records, source_provenance):
    plan = _formal_access_plan(protocol)
    classes = [
        class_id for task in _task_classes(protocol) for class_id in task
    ]
    final = len(_task_classes(protocol)) - 1
    expected_accesses = []
    for split, prefix, phase in plan:
        access = {
            'event': 'first_iteration',
            'loader_key': repr((prefix, tuple(classes))),
            'split': split, 'phase': phase,
            'event_idx': final, 'task_id': final,
            'timeline_step': f'event_{final}_CIL',
            'classes': classes,
        }
        marker = _json_value(
            contents[f'formal_access/{split}.consumed.json'],
            f'formal-access-{split}',
        )
        if marker != {
                'schema_version': 1, 'status': 'consumed', 'access': access}:
            _reject('formal-access-marker-invalid')
        if sum(record == access for record in data_flow_records) != 1:
            _reject('formal-access-data-flow-mismatch')
        expected_accesses.append(access)
    actual_accesses = [
        record for record in data_flow_records
        if record.get('event') == 'first_iteration'
        and 'phase' in record
    ]
    if actual_accesses != expected_accesses:
        _reject('formal-access-data-flow-mismatch')

    pending = _json_value(
        contents['FORMAL_EVALUATION_PENDING.json'], 'formal-pending')
    consuming = _json_value(
        contents['FORMAL_EVALUATION_CONSUMING.json'], 'formal-consuming')
    complete = _json_value(
        contents['FORMAL_EVALUATION_COMPLETE.json'], 'formal-complete')
    if (set(pending) != {
            'schema_version', 'status', 'transaction_sha256', 'identity',
            'freeze'}
            or set(consuming) != {
                'schema_version', 'status', 'transaction_sha256', 'identity',
                'freeze', 'cache_identity'}
            or pending.get('schema_version') != 1
            or consuming.get('schema_version') != 1
            or complete.get('schema_version') != 1
            or pending.get('status') != 'pending'
            or consuming.get('status') != 'consuming'
            or complete.get('status') != 'complete'
            or type(pending.get('transaction_sha256')) is not str
            or _SHA256.fullmatch(pending['transaction_sha256']) is None
            or pending['transaction_sha256']
                != consuming.get('transaction_sha256')
            or pending['transaction_sha256']
                != complete.get('transaction_sha256')
            or pending['identity'] != consuming.get('identity')
            or pending['identity'] != complete.get('identity')
            or pending['freeze'] != consuming.get('freeze')
            or pending['freeze'] != complete.get('freeze')
            or consuming['cache_identity'] != complete.get('cache_identity')):
        _reject('formal-access-transaction-invalid')
    identity = pending['identity']
    if (type(identity) is not dict
            or identity.get('protocol')
                != _formal_checkpoint_protocol(protocol, source_provenance)
            or identity.get('source_provenance') != source_provenance
            or identity.get('task_classes') != {
                str(task_id): list(task)
                for task_id, task in enumerate(_task_classes(protocol))
            }):
        _reject('formal-access-transaction-invalid')
    cache = complete['cache_identity']
    if protocol['base_options'].get('bic_enabled') is True:
        if type(cache) is not dict or set(cache) != {'test', 'calibration'}:
            _reject('formal-access-cache-invalid')
        _validate_cache_identity(cache['test'])
        calibration = cache['calibration']
        expected_split, _prefix, expected_phase = plan[-2]
        if (type(calibration) is not dict
                or set(calibration) != {'split', 'phase', 'cache'}
                or calibration['split'] != expected_split
                or calibration['phase'] != expected_phase
                or calibration != complete.get('calibration_cache_identity')):
            _reject('formal-access-cache-invalid')
        _validate_cache_identity(calibration['cache'])
    else:
        _validate_cache_identity(cache)


def _git(root, *arguments):
    root.verify()
    try:
        result = subprocess.run(
            ['git', '-C', f'/proc/self/fd/{root.fd}', *arguments], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            pass_fds=(root.fd,),
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        _reject('source-commit-invalid')
    root.verify()
    return result


def _verify_sources(root, authority, commit, declared, inode_seen, artifacts):
    if type(declared) is not list:
        _reject('source-files-schema-invalid')
    logicals = [item.get('logical_path') if type(item) is dict else None
                for item in declared]
    if (len(logicals) != len(set(logicals))
            or set(logicals) != set(_SOURCE_INVENTORY)):
        _reject('source-inventory-mismatch')
    resolved = _git(
        authority, 'rev-parse', f'{commit}^{{commit}}').decode().strip()
    if resolved != commit:
        _reject('source-commit-invalid')
    _git(authority, 'merge-base', '--is-ancestor', commit, 'HEAD')
    by_name = {item['logical_path']: item for item in declared}
    combined = hashlib.sha256(bytes.fromhex(commit))
    source_sha256 = {}
    for logical in sorted(_SOURCE_INVENTORY):
        content, digest = _read_entry(
            by_name[logical], root, logical, True, 'source', inode_seen
        )
        committed = _git(authority, 'show', f'{commit}:{logical}')
        tree_line = _git(
            authority, 'ls-tree', commit, '--', logical).decode().strip()
        match = re.fullmatch(r'100\d{3} blob ([0-9a-f]+)\t(.+)', tree_line)
        if committed != content or match is None or match.group(2) != logical:
            _reject('source-git-blob-mismatch')
        token = logical.encode()
        combined.update(len(token).to_bytes(8, 'big'))
        combined.update(token)
        combined.update(match.group(1).encode())
        combined.update(bytes.fromhex(digest))
        artifacts[f'source:{logical}'] = digest
        source_sha256[logical] = digest
    root.verify()
    return combined.hexdigest(), {
        'schema_version': 1,
        'source_commit': commit,
        'source_sha256': source_sha256,
    }


def _verify_data(root, dataset, declared, inode_seen, artifacts):
    expected = _AUTHORITATIVE_DATA.get(dataset)
    if type(expected) is not dict or not expected or type(declared) is not list:
        _reject('data-inventory-mismatch')
    logicals = [item.get('logical_path') if type(item) is dict else None
                for item in declared]
    if len(logicals) != len(set(logicals)) or set(logicals) != set(expected):
        _reject('data-inventory-mismatch')
    by_name = {item['logical_path']: item for item in declared}
    contents = {}
    for logical in sorted(expected):
        content, digest = _read_entry(
            by_name[logical], root, logical, True, 'data', inode_seen
        )
        if digest != expected[logical]:
            _reject('data-inventory-mismatch')
        contents[logical] = content
        artifacts[f'data:{logical}'] = digest
    return contents


def _task_classes(protocol):
    options = protocol['base_options']
    custom = options.get('custom_tasks')
    if custom is not None:
        return tuple(tuple(task) for task in custom)
    count, size, total = (
        options.get('num_tasks'), options.get('classes_per_task'),
        options.get('num_classes'),
    )
    if any(type(value) is not int or value <= 0
           for value in (count, size, total)):
        _reject('registry-protocol-invalid')
    return tuple(tuple(range(index * size, min((index + 1) * size, total)))
                 for index in range(count))


def _validate_manifest(manifest, dataset, protocol):
    expected = _AUTHORITATIVE_MANIFEST.get(dataset)
    ordered_key = 'ordered_sample_ids' if 'ordered_sample_ids' in manifest \
        else 'ordered_indices'
    if (type(expected) is not dict or set(manifest) != {
            'dataset', 'seed', 'per_class', 'by_class', ordered_key, 'sha256'}
            or manifest['dataset'] != expected['dataset']
            or type(manifest['seed']) is not int
            or manifest['seed'] != expected['seed']
            or manifest['seed'] != protocol['validation_split_seed']
            or type(manifest['per_class']) is not int
            or manifest['per_class'] != expected['per_class']
            or type(manifest['by_class']) is not dict
            or type(manifest[ordered_key]) is not list
            or manifest['sha256'] != expected['sha256']):
        _reject('validation-manifest-invalid')
    labels = {str(label) for task in _task_classes(protocol) for label in task}
    by_class = manifest['by_class']
    if (set(by_class) != labels
            or any(type(values) is not list
                   or len(values) != manifest['per_class']
                   or len(set(values)) != len(values)
                   or any(type(value) not in (int, str) for value in values)
                   for values in by_class.values())):
        _reject('validation-manifest-invalid')
    ordered = [value for key in sorted(by_class, key=int) for value in by_class[key]]
    if (manifest[ordered_key] != ordered
            or hashlib.sha256(_canonical_json(ordered)).hexdigest()
            != manifest['sha256']):
        _reject('validation-manifest-invalid')


def _metric_number(value):
    return (type(value) in (int, float) and math.isfinite(value))


def _validate_selection(value, manifest, protocol):
    if type(value) is not dict:
        _reject('selection-audit-invalid')
    expected_source = (
        'cifar100-train-validation'
        if protocol['dataset'] == 'cifar100' else 'vector-train-validation'
    )
    options = protocol['base_options']
    bic_enabled = options.get('bic_enabled') is True
    expected_calibration_per_class = (
        options['bic_per_class'] if bic_enabled else {})
    expected_calibration_count = (
        options['bic_per_class'] * len(manifest['by_class'])
        if bic_enabled else 0)
    expected_training_count = None
    if protocol['dataset'] == 'cifar100':
        expected_training_count = (
            500 - options['bic_per_class'] - manifest['per_class']
        ) * len(manifest['by_class'])
    calibration_sha = value.get('calibration_manifest_sha256')
    if (set(value) != _SELECTION_KEYS
            or value['passed'] is not True
            or value['test_used_for_selection'] is not False
            or value['evaluation_source'] != expected_source
            or value['validation_manifest_sha256'] != manifest['sha256']
            or (bic_enabled and (
                type(calibration_sha) is not str
                or _SHA256.fullmatch(calibration_sha) is None))
            or (not bic_enabled and calibration_sha is not None)
            or value['calibration_per_class'] != expected_calibration_per_class
            or value['validation_per_class'] != manifest['per_class']
            or type(value['training_count']) is not int
            or type(value['calibration_count']) is not int
            or value['calibration_count'] != expected_calibration_count
            or type(value['validation_count']) is not int
            or value['training_count'] <= 0
            or (expected_training_count is not None
                and value['training_count'] != expected_training_count)
            or value['validation_count'] != (
                manifest['per_class'] * len(manifest['by_class']))
            or any(type(value[key]) is not int or value[key] != 0 for key in (
                'training_calibration_overlap_count',
                'training_validation_overlap_count',
                'calibration_validation_overlap_count'))):
        _reject('selection-audit-invalid')


def _bic_float(value, minimum=None, maximum=None):
    return (type(value) is float and math.isfinite(value)
            and (minimum is None or value >= minimum)
            and (maximum is None or value <= maximum))


def _float32(value):
    return struct.unpack('!f', struct.pack('!f', value))[0]


def _validate_bic_summary(value, protocol, task_id):
    task_classes = _task_classes(protocol)[:task_id + 1]
    task_keys = {f'task_{value}' for value in range(task_id + 1)}
    scalar_keys = {'overall_accuracy', 'ece'}
    mapping_keys = {
        'per_task_accuracy', 'task_il', 'task_prediction_fraction',
    }
    if (type(value) is not dict
            or set(value) != scalar_keys | mapping_keys
            or any(not _bic_float(value[key], 0.0, 1.0)
                   for key in scalar_keys)
            or any(type(value[key]) is not dict
                   or set(value[key]) != task_keys
                   or any(not _bic_float(item, 0.0, 1.0)
                          for item in value[key].values())
                   for key in mapping_keys)
            or not math.isclose(
                sum(value['task_prediction_fraction'].values()), 1.0,
                rel_tol=0.0, abs_tol=1e-6,
            )):
        _reject('bic-results-invalid')
    correct, total = 0, 0
    for index, classes in enumerate(task_classes):
        samples = 100 * len(classes)
        accuracy = value['per_task_accuracy'][f'task_{index}']
        task_correct = round(accuracy * samples)
        if (not 0 <= task_correct <= samples
                or accuracy != _float32(task_correct / samples)):
            _reject('bic-results-invalid')
        correct += task_correct
        total += samples
    if value['overall_accuracy'] != _float32(correct / total):
        _reject('bic-results-invalid')


def _validate_calibration(value, protocol, manifest, selection):
    keys = {
        'passed', 'manifest_sha256', 'per_class', 'calibration_count',
        'training_count', 'overlap_count', 'test_used_for_fit',
    }
    options = protocol['base_options']
    class_count = len(manifest['by_class'])
    expected_count = options['bic_per_class'] * class_count
    validation_count = (
        manifest['per_class'] * class_count
        if options.get('lambda_validation_enabled') is True else 0
    )
    expected_training = 500 * class_count - expected_count - validation_count
    if (type(value) is not dict or set(value) != keys
            or protocol['dataset'] != 'cifar100'
            or value['passed'] is not True
            or type(value['manifest_sha256']) is not str
            or _SHA256.fullmatch(value['manifest_sha256']) is None
            or type(value['per_class']) is not int
            or value['per_class'] != options['bic_per_class']
            or type(value['calibration_count']) is not int
            or value['calibration_count'] != expected_count
            or type(value['training_count']) is not int
            or value['training_count'] != expected_training
            or type(value['overlap_count']) is not int
            or value['overlap_count'] != 0
            or value['test_used_for_fit'] is not False
            or (selection is not None and (
                value['manifest_sha256']
                    != selection['calibration_manifest_sha256']
                or value['per_class'] != selection['calibration_per_class']
                or value['calibration_count']
                    != selection['calibration_count']
                or value['training_count'] != selection['training_count']))):
        _reject('calibration-audit-invalid')


def _validate_bic_record(
        record, task_id, protocol, calibration, fit_corpus):
    keys = {
        'step', 'task_id', 'fit', 'paired', 'parameters',
        'calibration_audit', 'fit_corpus', 'disabled_identity_max_abs_diff',
        'task_il_max_abs_delta', 'privacy_audit',
    }
    options = protocol['base_options']
    task_keys = {f'task_{value}' for value in range(task_id + 1)}
    if (type(record) is not dict or set(record) != keys
            or record['step'] != f'event_{task_id}_CIL'
            or type(record['task_id']) is not int
            or record['task_id'] != task_id
            or record['calibration_audit'] != calibration
            or record['fit_corpus'] != fit_corpus
            or not _bic_float(
                record['disabled_identity_max_abs_diff'], 0.0, 0.0)
            or not _bic_float(record['task_il_max_abs_delta'], 0.0, 1e-6)):
        _reject('bic-results-invalid')
    fit = record['fit']
    if (type(fit) is not dict or set(fit) != {
            'mode', 'lr', 'steps', 'loss_before', 'loss_after'}
            or fit['mode'] != 'joint_alpha_beta'
            or type(fit['lr']) is not float
            or fit['lr'] != options['bic_lr']
            or type(fit['steps']) is not int
            or fit['steps'] != options['bic_steps']
            or not _bic_float(fit['loss_before'], 0.0)
            or not _bic_float(fit['loss_after'], 0.0)):
        _reject('bic-results-invalid')
    paired = record['paired']
    if type(paired) is not dict or set(paired) != {'raw', 'calibrated'}:
        _reject('bic-results-invalid')
    _validate_bic_summary(paired['raw'], protocol, task_id)
    _validate_bic_summary(paired['calibrated'], protocol, task_id)
    parameters = record['parameters']
    if (type(parameters) is not dict
            or set(parameters) != {str(value) for value in range(task_id + 1)}
            or any(type(item) is not dict or set(item) != {'alpha', 'beta'}
                   or not _bic_float(item['alpha'], 0.0)
                   or item['alpha'] == 0.0
                   or not _bic_float(item['beta'])
                   for item in parameters.values())):
        _reject('bic-results-invalid')
    privacy = record['privacy_audit']
    if (type(privacy) is not dict or set(privacy) != {
            'passed', 'test_used_for_fit', 'raw_images_saved',
            'party_embeddings_saved'}
            or privacy != {
                'passed': True, 'test_used_for_fit': False,
                'raw_images_saved': False, 'party_embeddings_saved': False,
            }):
        _reject('bic-results-invalid')
    delta = max(abs(
        paired['calibrated']['task_il'][key]
        - paired['raw']['task_il'][key]
    ) for key in task_keys)
    if record['task_il_max_abs_delta'] != delta:
        _reject('bic-results-invalid')


def _validate_bic_results(results, protocol, manifest, selection):
    options = protocol['base_options']
    history = results['bic_history']
    if (options.get('bic_fit_mode') != 'joint_each_stage'
            or type(history) is not list
            or len(history) != options['num_tasks']
            or results['bic_final'] != history[-1]):
        _reject('bic-results-invalid')
    calibration = results['calibration_audit']
    fit_corpus = results['bic_fit_corpus']
    from adaptive_consolidation_audit import (
        _validate_formal_bic_fit_corpus,
    )
    try:
        _validate_formal_bic_fit_corpus(
            fit_corpus, {'protocol': protocol['base_options']}
        )
    except ValueError:
        _reject('bic-fit-corpus-invalid')
    _validate_calibration(calibration, protocol, manifest, selection)
    for task_id, record in enumerate(history):
        _validate_bic_record(
            record, task_id, protocol, calibration, fit_corpus
        )


def _validate_bic_checkpoint_state(state, protocol, bic_final):
    from bic_calibration import TaskAffineCalibrator

    tasks = _task_classes(protocol)
    expected_keys = {str(task_id) for task_id in range(len(tasks))}
    values = state.get('tasks') if type(state) is dict else None
    if (type(state) is not dict or set(state) != {'tasks'}
            or type(values) is not dict or set(values) != expected_keys
            or type(bic_final) is not dict
            or type(bic_final.get('parameters')) is not dict
            or any(type(values[str(task_id)]) is not dict
                   or set(values[str(task_id)]) != {
                       'classes', 'raw_alpha', 'beta'}
                   or values[str(task_id)]['classes'] != list(classes)
                   or not _bic_float(values[str(task_id)]['raw_alpha'])
                   or not _bic_float(values[str(task_id)]['beta'])
                   for task_id, classes in enumerate(tasks))):
        _reject('bic-state-invalid')
    calibrator = TaskAffineCalibrator()
    try:
        calibrator.load_state_dict(state)
        canonical = calibrator.state_dict()
        parameters = {
            str(task_id): calibrator.parameters_for(task_id)
            for task_id in range(len(tasks))
        }
    except (KeyError, TypeError, ValueError):
        _reject('bic-state-invalid')
    if canonical != state or parameters != bic_final['parameters']:
        _reject('bic-state-invalid')


def _validate_formal_bic_producer_bundle(
        *, history, state, calibration_audit, task_classes, options,
        validation_manifest, selection_audit, fit_corpus):
    """Apply the canonical Task 5 BiC producer validators to formal output."""
    normalized = [
        [int(class_id) for class_id in task_classes[task_id]]
        for task_id in sorted(task_classes)
    ]
    protocol = {
        'dataset': 'cifar100',
        'validation_split_seed': options['lambda_validation_split_seed'],
        'base_options': {
            'custom_tasks': normalized,
            'num_tasks': len(normalized),
            'bic_enabled': True,
            'cl_method': options['cl_method'],
            'head_consolidation_enabled':
                options['head_consolidation_enabled'],
            'head_consolidation_mode': options['head_consolidation_mode'],
            'bic_fit_mode': options['bic_fit_mode'],
            'bic_lr': options['bic_lr'],
            'bic_steps': options['bic_steps'],
            'bic_per_class': options['bic_per_class'],
            'lambda_validation_enabled': options['lambda_validation_enabled'],
            'lambda_validation_per_class':
                options['lambda_validation_per_class'],
        },
    }
    results = {
        'bic_history': history,
        'bic_final': history[-1] if type(history) is list and history else None,
        'calibration_audit': calibration_audit,
        'bic_fit_corpus': fit_corpus,
    }
    _validate_manifest(validation_manifest, 'cifar100', protocol)
    _validate_selection(selection_audit, validation_manifest, protocol)
    _validate_bic_results(
        results, protocol, validation_manifest, selection_audit
    )
    _validate_bic_checkpoint_state(state, protocol, results['bic_final'])
    return True


def _validate_results(results, protocol, manifest, source_provenance=None):
    options = protocol['base_options']
    expected_keys = set(_BASE_RESULT_KEYS)
    if options.get('bic_enabled') is True:
        expected_keys.update(_BIC_RESULT_KEYS)
    if options.get('lambda_validation_enabled') is True:
        expected_keys.add('selection_audit')
    if type(results) is not dict or set(results) != expected_keys:
        _reject('results-schema-invalid')
    if (type(results['source_provenance']) is not dict
            or (source_provenance is not None
                and results['source_provenance'] != source_provenance)):
        _reject('results-source-provenance-invalid')
    expected_config = {
        'cl_method': protocol['base_options']['cl_method'],
        'ul_method': protocol['base_options']['ul_method'],
        'data': protocol['base_options']['data'],
        'num_tasks': protocol['base_options']['num_tasks'],
        'seed': protocol['seed'],
    }
    if (type(results['config']) is not dict
            or set(results['config']) != _RESULT_CONFIG_KEYS
            or not _schema_matches(results['config'], expected_config)
            or results['config'] != expected_config):
        _reject('results-config-invalid')
    selection = None
    if options.get('lambda_validation_enabled') is True:
        selection = results['selection_audit']
        _validate_selection(selection, manifest, protocol)
    if options.get('bic_enabled') is True:
        _validate_bic_results(results, protocol, manifest, selection)
    for key in ('ul_metrics', 'comm_stats', 'timing', 'step_results'):
        if type(results[key]) is not list:
            _reject('results-schema-invalid')
    if type(results['cl_metrics']) is not dict:
        _reject('results-schema-invalid')
    from runner import _valid_tracker_checkpoint_state
    tracker_state = {
        key: results[key] for key in _TRACKER_KEYS
    }
    checkpoint_protocol = {
        'seed': protocol['seed'],
        'data': options['data'],
        'cl_method': options['cl_method'],
        'num_tasks': options['num_tasks'],
        'num_parties': options['num_parties'],
        'head_consolidation_enabled': int(
            options.get('head_consolidation_enabled', False)),
        'head_consolidation_mode': options.get('head_consolidation_mode'),
    }
    if options.get('formal_deferred_evaluation') is True:
        checkpoint_protocol['formal_deferred_evaluation'] = True
    if not _valid_tracker_checkpoint_state(tracker_state, checkpoint_protocol):
        _reject('results-producer-state-invalid')
    rows = results['task_acc_history']
    task_classes = _task_classes(protocol)
    if type(rows) is not list or len(rows) != len(task_classes):
        _reject('trajectory-invalid')
    deferred = options.get('formal_deferred_evaluation') is True
    class_rows, taskil_rows, diagonals = [], [], {}
    base_keys = {'step', 'per_task_accs', 'overall_acc', 'per_task_accs_taskil'}
    optional = {'per_task_accs_debiased', 'companion_readouts'}
    for index, row in enumerate(rows):
        deferred_keys = (
            {'step', 'per_task_accs', 'overall_acc', 'deferred_diagonal'}
            if index < len(rows) - 1 else base_keys | {
                'deferred_diagonal', 'deferred_final',
                'deferred_final_task',
            })
        if (type(row) is not dict
                or ((not deferred_keys.issubset(row)
                     or bool(set(row).difference(deferred_keys | optional)))
                    if deferred else (
                    not base_keys.issubset(row)
                    or bool(set(row).difference(base_keys | optional))))
                or row['step'] != f'event_{index}_CIL'
                or not _metric_number(row['overall_acc'])):
            _reject('trajectory-invalid')
        if deferred:
            expected_key = f'task_{index}'
            if index < len(rows) - 1:
                if (row['per_task_accs'] != row['deferred_diagonal']
                        or set(row['per_task_accs']) != {expected_key}):
                    _reject('trajectory-invalid')
                diagonals[expected_key] = row['per_task_accs'][expected_key]
                class_values = dict(diagonals)
                taskil_values = dict(diagonals)
            else:
                if (row['deferred_final'] is not True
                        or row['deferred_final_task'] != expected_key
                        or set(row['deferred_diagonal'])
                        != {f'task_{task}' for task in range(len(rows))}):
                    _reject('trajectory-invalid')
                class_values = row['per_task_accs']
                taskil_values = row['per_task_accs_taskil']
        else:
            class_values = row['per_task_accs']
            taskil_values = row['per_task_accs_taskil']
        class_rows.append({'step': row['step'], 'values': class_values})
        taskil_rows.append({
            'step': row['step'], 'values': taskil_values,
        })
    try:
        metrics = reconstruct_metrics(
            class_rows, taskil_rows, tuple(range(len(task_classes)))
        )
    except ValueError:
        _reject('trajectory-invalid')
    trajectory = {'class_il': class_rows, 'task_il': taskil_rows}
    trajectory_sha256 = hashlib.sha256(_canonical_json(trajectory)).hexdigest()
    return metrics, trajectory_sha256


def _data_flow_records(content):
    records = []
    for line in content.splitlines():
        if not line.strip():
            continue
        records.append(_json_value(line, 'data-flow'))
    if not records:
        _reject('data-flow-invalid')
    return records


def _validate_data_flow(content, protocol):
    records = _data_flow_records(content)
    expected = [tuple(task) for task in _task_classes(protocol)]
    final = len(expected) - 1
    final_step = f'event_{final}_CIL'
    transitions = []
    saw_test = False
    for record in records:
        if set(record) == {
                'event', 'loader_key', 'split', 'phase', 'event_idx',
                'task_id', 'timeline_step', 'classes'}:
            phases = {
                'validation': 'final_validation_pre_install',
                'calibration': 'final_bic_calibration_post_freeze',
                'test': 'final_test_post_install',
            }
            if (record['event'] != 'first_iteration'
                    or record['split'] not in phases
                    or type(record['loader_key']) is not str
                    or record['phase'] != phases[record['split']]
                    or type(record['event_idx']) is not int
                    or record['event_idx'] != final
                    or type(record['task_id']) is not int
                    or record['task_id'] != final
                    or record['timeline_step'] != final_step
                    or record['classes'] != [
                        class_id for task in _task_classes(protocol)
                        for class_id in task
                    ]
                    or not transitions):
                _reject('data-flow-invalid')
            if record['split'] == 'test':
                saw_test = True
            continue
        if set(record) != {
                'loader_key', 'loader_iteration', 'batch', 'indices', 'dtype',
                'shape', 'batch_sha256'}:
            _reject('data-flow-invalid')
        try:
            key = ast.literal_eval(record['loader_key'])
        except (ValueError, SyntaxError):
            _reject('data-flow-invalid')
        if (type(key) is not tuple or len(key) != 2 or key[0] != 'train'
                or type(key[1]) is not tuple
                or any(type(value) is not int for value in key[1])
                or type(record['loader_iteration']) is not int
                or type(record['batch']) is not int
                or type(record['indices']) is not list
                or any(type(value) is not int for value in record['indices'])
                or record['dtype'] != 'torch.float32'
                or type(record['shape']) is not list
                or any(type(value) is not int for value in record['shape'])
                or type(record['batch_sha256']) is not str
                or _SHA256.fullmatch(record['batch_sha256']) is None):
            _reject('data-flow-invalid')
        if not transitions or transitions[-1] != key[1]:
            transitions.append(key[1])
    if transitions != expected or not saw_test:
        _reject('data-flow-invalid')
    return records


def _probe_args(protocol, manifest, scratch, data_paths):
    from types import SimpleNamespace

    options = dict(protocol['base_options'])
    if options.get('lambda_validation_enabled'):
        options['lambda_validation_per_class'] = manifest['per_class']
        options['lambda_validation_split_seed'] = manifest['seed']
    options.update({
        'device': 'cpu', 'output_dir': str(scratch / 'output'),
        'resume_run_dir': str(scratch / 'output'),
        'head_consolidation_enabled': int(
            options.get('head_consolidation_enabled', False)),
        'head_consolidation_mode': options.get('head_consolidation_mode'),
        'embed_dim': 128 if options.get('model_type') == 'mlp' else 512,
        # Image augmentation must replay the producer's worker RNG streams.
        'num_workers': (0 if options['data'] == 'tabvfl'
                        else options['num_workers']),
        'cosine_head': False, 'party_col_ranges': None,
        'party_view_groups': None,
    })
    custom = options.get('custom_tasks')
    options['custom_tasks'] = ('|'.join(
        ','.join(str(value) for value in task) for task in custom
    ) if custom else '')
    after = options.get('unlearn_after_tasks')
    classes = options.get('unlearn_classes')
    options['unlearn_after_tasks'] = [after] if type(after) is int else list(after)
    options['unlearn_classes'] = [[classes]] if type(classes) is int else list(classes)
    options['data_path'] = str(scratch / 'data')
    vector = next((path for logical, path in data_paths.items()
                   if logical.endswith('.npz')), None)
    if vector is not None:
        options['vector_npz'] = str(vector)
    return SimpleNamespace(**options)


def _hidden_loader_prepasses(method, protocol, task_id, epoch):
    if protocol['method_contract']['cl_method'] != 'proto_evolve':
        return 0
    from cl_methods.proto_evolve import ProtoEvolveCL

    if type(method) is not ProtoEvolveCL:
        raise ValueError('invalid adaptive SDC method')
    use_sdc = method.use_sdc
    interval = method.sdc_interval
    if type(use_sdc) is not bool:
        raise ValueError('invalid adaptive SDC enable flag')
    if type(interval) is not int or interval <= 0:
        raise ValueError('invalid adaptive SDC interval')
    return int(task_id > 0 and use_sdc and epoch % interval == 0)


def _strict_data_flow(dataset, args, records, task_classes,
                      checkpoint_boundary, protocol, method):
    from determinism import tensor_sha256

    position = 0

    for task_index, classes in enumerate(task_classes):
        loader = dataset.get_train_loader(classes)
        for epoch in range(args.epochs_per_task):
            # Fully advance sampler and worker RNG streams, not SDC computation.
            for _ in range(_hidden_loader_prepasses(
                    method, protocol, task_index, epoch)):
                for _batch in loader:
                    pass
            for batch_index, (batch_x, _batch_y) in enumerate(loader):
                if position >= len(records):
                    raise ValueError('producer data flow is incomplete')
                record = records[position]
                expected_indices = loader.audit_sampler.epoch_orders[-1][
                    batch_index * loader.batch_size:
                    (batch_index + 1) * loader.batch_size]
                audited_x = batch_x
                if args.model_type == 'mlp':
                    import torch
                    audited_x = torch.stack([
                        dataset.trainset[index][0]
                        for index in expected_indices
                    ])
                if (record.get('loader_key') != loader.audit_key
                        or record.get('loader_iteration')
                        != len(loader.audit_sampler.epoch_orders) - 1
                        or record.get('batch') != batch_index
                        or record.get('indices') != expected_indices
                        or record.get('dtype') != str(audited_x.dtype)
                        or record.get('shape') != list(audited_x.shape)
                        or record.get('batch_sha256')
                        != tensor_sha256(audited_x.cpu())):
                    raise ValueError('producer data flow does not match loader')
                position += 1

    final = len(task_classes) - 1
    expected_boundary = {
        'event_idx': final,
        'task_id': final,
        'timeline_step': f'event_{final}_CIL',
    }
    if checkpoint_boundary != expected_boundary:
        raise ValueError('checkpoint is not the final CIL boundary')
    expected_classes = tuple(
        class_id for classes in task_classes for class_id in classes)
    spec = FormalSpec(
        protocol['dataset'], protocol['method'], protocol['seed'],
        protocol['explanation'],
    )
    expected_accesses = (
        [('validation', 'lambda_validation',
          'final_validation_pre_install')]
        if validation_access_for(spec) else []
    ) + (
        [('calibration', 'bic_calibration',
          'final_bic_calibration_post_freeze')]
        if (protocol['base_options'].get('bic_enabled') is True
            and not validation_access_for(spec)) else []
    ) + [('test', 'test', 'final_test_post_install')]
    for split, prefix, phase in expected_accesses:
        if position >= len(records):
            raise ValueError('producer split evidence is incomplete')
        record = records[position]
        try:
            key = ast.literal_eval(record['loader_key'])
        except (KeyError, ValueError, SyntaxError) as error:
            raise ValueError('invalid producer split key') from error
        if (record.get('event') != 'first_iteration'
                or record.get('split') != split
                or record.get('phase') != phase
                or {name: record.get(name) for name in expected_boundary}
                    != expected_boundary
                or record.get('classes') != list(expected_classes)
                or type(key) is not tuple or len(key) != 2
                or key != (prefix, expected_classes)):
            raise ValueError('invalid producer split boundary')
        position += 1
    if position != len(records):
        raise ValueError('producer split evidence is incomplete')


def _reconstruct_formal_caches(dataset, args, protocol):
    from adaptive_consolidation_audit import _formal_cache_identity
    from metrics import cache_formal_batches

    plan = _formal_access_plan(protocol)
    final = len(_task_classes(protocol)) - 1
    classes = [
        class_id for task in _task_classes(protocol) for class_id in task
    ]
    boundary = {
        'event_idx': final, 'task_id': final,
        'timeline_step': f'event_{final}_CIL', 'classes': classes,
    }
    calibration = None
    calibration_batches = None
    for split, _prefix, phase in plan[:-1]:
        if split == 'validation':
            dataset.authorize_formal_access(
                split=split, phase=phase, **boundary)
            cached = cache_formal_batches(
                dataset.get_validation_loader(classes))
        elif split == 'calibration':
            dataset.authorize_formal_calibration_access(
                phase=phase, **boundary)
            cached = cache_formal_batches(
                dataset.get_calibration_loader(classes))
        else:
            raise ValueError('invalid formal cache reconstruction plan')
        if protocol['base_options'].get('bic_enabled') is True:
            calibration_batches = cached
            calibration = {
                'split': split, 'phase': phase,
                'cache': _formal_cache_identity(cached),
            }
    test_split, _test_prefix, test_phase = plan[-1]
    if test_split != 'test':
        raise ValueError('formal cache reconstruction has no test phase')
    dataset.authorize_formal_access(
        split=test_split, phase=test_phase, **boundary)
    test_batches = cache_formal_batches(dataset.get_test_loader(classes))
    test = _formal_cache_identity(test_batches)
    if protocol['base_options'].get('bic_enabled') is True:
        if calibration is None:
            raise ValueError('formal calibration cache was not reconstructed')
        return (
            {'test': test, 'calibration': calibration},
            test_batches, calibration_batches,
        )
    return test, test_batches, None


def _validate_authoritative_formal_cache(contents, expected):
    consuming = _json_value(
        contents['FORMAL_EVALUATION_CONSUMING.json'], 'probe-consuming')
    complete = _json_value(
        contents['FORMAL_EVALUATION_COMPLETE.json'], 'probe-complete')
    seal = _json_value(
        contents['FORMAL_EVALUATION_SEALED.json'], 'probe-seal')
    if (consuming.get('cache_identity') != expected
            or complete.get('cache_identity') != expected
            or seal.get('cache_identity') != expected):
        raise ValueError('formal cache differs from authoritative loader')
    calibration = (
        expected.get('calibration') if type(expected) is dict else None
    )
    if calibration is not None and complete.get(
            'calibration_cache_identity') != calibration:
        raise ValueError('formal calibration cache identity mismatch')


def _internal_runtime(protocol):
    method = protocol['method']
    fixed = {
        'fixed_full': 'full',
        'fixed_bias': 'bias',
    }
    if method in fixed:
        from adaptive_tinyimagenet_heldout import _fixed_branch_runtime
        return _fixed_branch_runtime(fixed[method])
    ablation = {
        'fixed_half': 'fixed_half_ablation',
        'sample_mean_nll': 'sample_mean_nll',
    }.get(method)
    if ablation is not None:
        from adaptive_dual_branch_validation import _ablation_runtime
        return _ablation_runtime(ablation)
    return nullcontext()


def _strict_method_contract(method, protocol, trainer=None, loaded=False,
                            manifest=None):
    contract = protocol['method_contract']
    cl_method = contract['cl_method']
    if cl_method == 'lwf':
        actual = (
            method.temperature, method.alpha, method.lwf_lambda,
            method.ce_newonly,
        )
        expected = tuple(contract[key] for key in (
            'lwf_temperature', 'lwf_alpha', 'lwf_lambda', 'lwf_ce_newonly'))
    elif cl_method == 'gpm':
        actual, expected = (method.threshold,), (contract['gpm_threshold'],)
    elif cl_method == 'fedprotip_vfl':
        from cl_methods.fedprotip_vfl import TIP_MAX_BATCHES
        actual = (method.tip_threshold, TIP_MAX_BATCHES)
        expected = (
            contract['fedprotip_tip_threshold'],
            contract['fedprotip_max_batches'],
        )
    elif cl_method == 'er':
        actual = (
            method.per_class_size, method.buffer_batch_size, method.alpha)
        expected = tuple(contract[key] for key in (
            'er_per_class', 'er_batch', 'er_alpha'))
    elif cl_method != 'proto_evolve':
        return
    else:
        if not loaded:
            return
        state = method.get_state()
        top = trainer.top_model
        history = state['head_consolidation_history']
        if protocol['method'] == 'no_consolidation':
            if (method.head_consolidation_enabled
                    or bool(top._adaptive_enabled) or history
                    or state['adaptive_audit_bundle'] is not None
                    or state['head_validation_sha256']):
                raise ValueError(
                    'no-consolidation state contains installed head evidence')
            return
        from adaptive_head_consolidation import (
            BIAS_BRANCH_CONFIG, FULL_BRANCH_CONFIG, INACTIVE_BRANCH_CONFIG,
            AdaptiveConsolidationResult,
        )
        from head_consolidation import hash_top_state
        from runner import _checkpoint_values_equal
        if len(history) != 1 or not bool(top._adaptive_enabled):
            raise ValueError('internal consolidation state is incomplete')
        result = AdaptiveConsolidationResult.from_dict(history[0]).to_dict()
        bundle = state['adaptive_audit_bundle']
        classes = [class_id for task in _task_classes(protocol)
                   for class_id in task]
        gate_rule, expected_gate, configs = {
            'fixed_full': (
                'fixed_full', 1.0,
                {'full': FULL_BRANCH_CONFIG, 'bias': INACTIVE_BRANCH_CONFIG}),
            'fixed_bias': (
                'fixed_bias', 0.0,
                {'full': INACTIVE_BRANCH_CONFIG, 'bias': BIAS_BRANCH_CONFIG}),
            'adaptive': (
                'class_balanced', None,
                {'full': FULL_BRANCH_CONFIG, 'bias': BIAS_BRANCH_CONFIG}),
            'fixed_half': (
                'fixed_half_ablation', 0.5,
                {'full': FULL_BRANCH_CONFIG, 'bias': BIAS_BRANCH_CONFIG}),
            'sample_mean_nll': (
                'sample_mean_ablation', None,
                {'full': FULL_BRANCH_CONFIG, 'bias': BIAS_BRANCH_CONFIG}),
        }[protocol['method']]
        gate = result['gate']
        full_state = bundle.get('full_state') if type(bundle) is dict else None
        bias_state = bundle.get('bias_state') if type(bundle) is dict else None
        expected_nonempty = {
            'full': configs['full'] != INACTIVE_BRANCH_CONFIG,
            'bias': configs['bias'] != INACTIVE_BRANCH_CONFIG,
        }
        if (result['ordered_classes'] != classes
                or result['candidate_configs'] != configs
                or gate['gate_rule'] != gate_rule
                or (expected_gate is not None and gate['g'] != expected_gate)
                or float(top._adaptive_gate) != gate['g']
                or top._adaptive_class_order.tolist() != classes
                or result['validation_manifest'] != manifest
                or state['head_validation_sha256'] != manifest['sha256']
                or bundle.get('result') != result
                or bundle.get('installed_state') is None
                or not _checkpoint_values_equal(
                    bundle['installed_state'], trainer.get_state()['top_model'])
                or any(bool(value) != expected_nonempty[name]
                       for name, value in (
                           ('full', full_state), ('bias', bias_state)))
                or result['candidate_hashes']['full']
                    != hash_top_state(full_state)
                or result['candidate_hashes']['bias']
                    != hash_top_state(bias_state)):
            raise ValueError('internal consolidation contract mismatch')
        return
    if actual != expected:
        raise ValueError('loaded method violates canonical runtime contract')


def _strict_probe_main(checkpoint_path, protocol_path, data_root):
    import copy

    from adaptive_consolidation_audit import (
        _restricted_torch_load,
        evaluate_formal_deferred_trajectory,
        prepare_formal_deferred_evaluation,
    )
    from bic_calibration import TaskAffineCalibrator
    from cl_methods import get_cl_method
    from data_utils import TaskManager, VFLDataset
    from metrics import MetricsTracker
    from models import build_models
    from runner import (
        _checkpoint_values_equal, _load_resume_checkpoint, _safe_torch_load,
    )
    from vfl_trainer import VFLTrainer

    bundle = _json_value(Path(protocol_path).read_bytes(), 'probe-protocol')
    if type(bundle) is not dict or set(bundle) != {
            'protocol', 'manifest', 'tracker_state', 'bic_history',
            'data_flow', 'calibration_audit', 'selection_audit',
            'run_dir', 'formal_artifacts', 'results', 'source_provenance'}:
        raise ValueError('invalid strict probe bundle')
    protocol = bundle['protocol']
    scratch = Path(data_root).parent
    data_paths = {
        path.relative_to(data_root).as_posix(): path
        for path in Path(data_root).rglob('*') if path.is_file()
    }
    args = _probe_args(protocol, bundle['manifest'], scratch, data_paths)
    args.formal_source_provenance = copy.deepcopy(
        bundle['source_provenance']
    )
    try:
        expected = _safe_torch_load(checkpoint_path)
    except Exception as error:
        raise RuntimeError('UNSAFE_CHECKPOINT') from error
    if (type(expected) is not dict or expected.get('schema_version') != 4):
        raise ValueError('checkpoint is not runner schema-v4')
    dataset = VFLDataset(args)
    options = protocol['base_options']
    calibration_audit = (
        dataset.calibration_audit()
        if options.get('bic_enabled') is True else None
    )
    selection_audit = (
        dataset.selection_audit()
        if options.get('lambda_validation_enabled') is True else None
    )
    if (calibration_audit != bundle['calibration_audit']
            or selection_audit != bundle['selection_audit']
            or (selection_audit is not None
                and dataset.validation_manifest != bundle['manifest'])):
        raise ValueError('producer manifest audit mismatch')
    bic_enabled = options.get('bic_enabled') is True
    if bic_enabled:
        if not bundle['bic_history']:
            raise ValueError('BiC history is missing')
        _validate_bic_checkpoint_state(
            expected['bic_state'], protocol, bundle['bic_history'][-1])
    elif expected['bic_state'] is not None:
        raise ValueError('non-BiC checkpoint contains calibrator state')
    task_manager = TaskManager(args)
    bottoms, top = build_models(args)
    trainer = VFLTrainer(bottoms, top, args)
    trainer.dataset_ref = dataset
    with _internal_runtime(protocol):
        method = get_cl_method(args.cl_method, trainer, args)
        _strict_method_contract(method, protocol)
        tracker = MetricsTracker()
        calibrator = TaskAffineCalibrator()
        start, seen, restored_bic_history = _load_resume_checkpoint(
            args, trainer, method, task_manager, tracker,
            calibrator,
        )
    timeline = task_manager.get_timeline()
    expected_tasks = sorted(event['task_id'] for event in timeline
                            if event['type'] == 'CIL')
    if start != len(timeline) or sorted(seen) != expected_tasks:
        raise ValueError('resume did not restore completed timeline')
    if (not _checkpoint_values_equal(trainer.get_state(), expected['trainer_state'])
            or not _checkpoint_values_equal(method.get_state(), expected['cl_state'])
            or not _checkpoint_values_equal(
                tracker.to_dict(), expected['tracker_state'])
            or not _checkpoint_values_equal(
                expected['tracker_state'], bundle['tracker_state'])
            or not _checkpoint_values_equal(
                expected['bic_history'], bundle['bic_history'])
            or not _checkpoint_values_equal(
                restored_bic_history, bundle['bic_history'])
            or (bic_enabled and not _checkpoint_values_equal(
                calibrator.state_dict(), expected['bic_state']))):
        raise ValueError('resume load-to-get-state mismatch')
    if bic_enabled:
        _validate_bic_checkpoint_state(
            calibrator.state_dict(), protocol, bundle['bic_history'][-1])
    with _internal_runtime(protocol):
        _strict_method_contract(
            method, protocol, trainer, loaded=True,
            manifest=bundle['manifest'],
        )
    _strict_data_flow(
        dataset, args, bundle['data_flow'],
        [tuple(task) for task in _task_classes(protocol)],
        {'event_idx': expected.get('event_idx'),
         'task_id': expected.get('task_id'),
         'timeline_step': expected.get('step')},
        protocol, method,
    )
    authoritative_cache, test_cache, calibration_cache = \
        _reconstruct_formal_caches(
        dataset, args, protocol
    )
    producer_root = _PinnedRoot(bundle['run_dir'])
    try:
        formal_contents = _verify_formal_artifacts(
            bundle['formal_artifacts'], producer_root, protocol, set(), {}
        )
        _validate_authoritative_formal_cache(
            formal_contents, authoritative_cache
        )
        producer_args = copy.deepcopy(args)
        producer_args.output_dir = str(producer_root.path)
        producer_args.resume_run_dir = str(producer_root.path)
        task_classes = {
            task_id: list(classes)
            for task_id, classes in enumerate(_task_classes(protocol))
        }
        snapshot_payloads = [
            formal_contents[f'formal_snapshots/event_{event_idx}_CIL.pt']
            for event_idx in range(len(task_classes))
        ]
        with _internal_runtime(protocol):
            recomputed = evaluate_formal_deferred_trajectory(
                args=producer_args,
                snapshot_paths=[Path(str(index)) for index in task_classes],
                final_checkpoint=Path(checkpoint_path),
                task_classes=task_classes,
                cached_test_batches=test_cache,
                cached_calibration_batches=calibration_cache,
                calibration_audit=calibration_audit,
                validation_manifest=bundle['manifest'],
                selection_audit=selection_audit,
                output_dir=producer_root.path,
                recompute_payloads={
                    'snapshots': snapshot_payloads,
                    'final': expected,
                },
            )
        expected_tracker = {
            key: bundle['results'][key] for key in _TRACKER_KEYS
        }
        expected_recomputed = expected_tracker
        if bic_enabled:
            expected_recomputed = {
                'tracker_state': expected_tracker,
                'bic_history': bundle['results']['bic_history'],
                'bic_state': expected['bic_state'],
                'calibration_audit': bundle['results']['calibration_audit'],
                'bic_fit_corpus': bundle['results']['bic_fit_corpus'],
            }
        if not _checkpoint_values_equal(recomputed, expected_recomputed):
            raise ValueError(
                'stored formal metrics differ from frozen-state recomputation'
            )
        with _internal_runtime(protocol):
            transaction = prepare_formal_deferred_evaluation(
                args=producer_args,
                snapshot_paths=[
                    producer_root.path / 'formal_snapshots'
                    / f'event_{event_idx}_CIL.pt'
                    for event_idx in range(len(task_classes))
                ],
                final_checkpoint=(
                    producer_root.path / 'checkpoints' / 'formal_final.pt'
                ),
                task_classes=task_classes,
                output_dir=producer_root.path,
            )
        result = transaction.get('result')
        if transaction.get('status') != 'published':
            raise ValueError('formal producer transaction is not published')
        if protocol['base_options'].get('bic_enabled') is True:
            if (type(result) is not dict
                    or result.get('tracker_state') != expected_tracker
                    or result.get('bic_history')
                        != bundle['results'].get('bic_history')
                    or result.get('calibration_audit')
                        != bundle['results'].get('calibration_audit')):
                raise ValueError('formal producer result identity mismatch')
        elif result != expected_tracker:
            raise ValueError('formal producer result identity mismatch')
        producer_root.verify()
    finally:
        producer_root.close()


def _probe_checkpoint(content, protocol, manifest, data_contents,
                      tracker_state, bic_history, data_flow_records,
                      calibration_audit, selection_audit, run_dir,
                      formal_artifacts, results, source_provenance):
    with tempfile.TemporaryDirectory(prefix='formal_resume_probe_') as temporary:
        scratch = Path(temporary)
        output = scratch / 'output' / 'checkpoints'
        output.mkdir(parents=True)
        final = len(_task_classes(protocol)) - 1
        checkpoint = output / f'event_{final}_CIL.pt'
        checkpoint.write_bytes(content)
        protocol_path = scratch / 'protocol.json'
        protocol_path.write_bytes(_canonical_json({
            'protocol': protocol,
            'manifest': manifest,
            'tracker_state': tracker_state,
            'bic_history': bic_history,
            'data_flow': data_flow_records,
            'calibration_audit': calibration_audit,
            'selection_audit': selection_audit,
            'run_dir': str(run_dir),
            'formal_artifacts': formal_artifacts,
            'results': results,
            'source_provenance': source_provenance,
        }))
        data_root = scratch / 'data'
        for logical, payload in data_contents.items():
            target = data_root / logical
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), '--strict-probe',
             str(checkpoint), str(protocol_path), str(data_root)],
            cwd=str(Path(__file__).resolve().parent),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
    lines = completed.stdout.splitlines()
    if completed.returncode == 0 and lines and lines[-1] == 'PROBE_OK':
        return
    if 'UNSAFE_CHECKPOINT' in lines:
        _reject('unsafe-checkpoint')
    _rerun('strict-resume-failed')


def _validate_declaration_shapes(declaration, dataset):
    if set(declaration) != _DECLARATION_KEYS:
        _reject('declaration-schema-invalid')
    if (type(declaration['results']) is not dict
            or set(declaration['results']) != _RESULT_DECLARATION_KEYS
            or type(declaration['results']['data_flow']) is not dict
            or set(declaration['results']['data_flow']) != _ENTRY_KEYS):
        _reject('results-declaration-schema-invalid')
    expected_manifest = _AUTHORITATIVE_MANIFEST.get(dataset)
    if type(expected_manifest) is not dict:
        _reject('validation-manifest-invalid')
    for name in ('config', 'checkpoint'):
        _entry_shape(declaration[name], False, name)
    _entry_shape(declaration['validation_manifest'], True, 'validation-manifest')
    _entry_shape(
        {key: declaration['results'][key] for key in _ENTRY_KEYS},
        False, 'results',
    )
    _entry_shape(declaration['results']['data_flow'], False, 'data-flow')
    if (type(declaration['formal_artifacts']) is not list
            or not declaration['formal_artifacts']):
        _reject('formal-artifacts-schema-invalid')
    for entry in declaration['formal_artifacts']:
        _entry_shape(entry, True, 'formal-artifact')


def audit_candidate(spec, declaration) -> AdmissionRecord:
    """Admit one declared candidate without discovery or candidate mutation."""
    normalized_spec = _normalized_spec(spec)
    protocol, protocol_sha256 = _protocol(spec)
    if type(declaration) is not dict:
        raise TypeError('declaration must be a dict')
    try:
        _validate_declaration_shapes(declaration, spec.dataset)
        if (not _schema_matches(declaration['spec'], normalized_spec)
                or declaration['spec'] != normalized_spec):
            _reject('declaration-spec-mismatch')
        commit = declaration['code_commit']
        if type(commit) is not str or _COMMIT.fullmatch(commit) is None:
            _reject('source-commit-invalid')
        with ExitStack() as stack:
            run_root = _PinnedRoot(declaration['run_dir'])
            stack.callback(run_root.close)
            source_files = declaration['source_files']
            data_files = declaration['data_files']
            if not source_files or type(source_files[0]) is not dict:
                _reject('source-inventory-mismatch')
            if not data_files or type(data_files[0]) is not dict:
                _reject('data-inventory-mismatch')
            source_root = _PinnedRoot(source_files[0].get('root'))
            data_root = _PinnedRoot(data_files[0].get('root'))
            authority_root = _PinnedRoot(Path(__file__).resolve().parent)
            stack.callback(source_root.close)
            stack.callback(data_root.close)
            stack.callback(authority_root.close)
            inode_seen, artifacts = set(), {}
            source_sha256, source_provenance = _verify_sources(
                source_root, authority_root, commit, source_files,
                inode_seen, artifacts,
            )
            data_contents = _verify_data(
                data_root, spec.dataset, data_files, inode_seen, artifacts
            )
            expected_manifest = _AUTHORITATIVE_MANIFEST[spec.dataset]
            manifest_content, digest = _read_entry(
                declaration['validation_manifest'], run_root,
                expected_manifest['logical_path'], True,
                'validation-manifest', inode_seen,
            )
            artifacts['validation_manifest'] = digest
            manifest = _json_value(manifest_content, 'validation-manifest')
            _validate_manifest(manifest, spec.dataset, protocol)
            config_content, digest = _read_entry(
                declaration['config'], run_root, 'config.json', False,
                'config', inode_seen,
            )
            artifacts['config'] = digest
            config = _json_value(config_content, 'config')
            _validate_producer_config(
                config, protocol, run_root, data_root, data_files
            )
            formal_contents = _verify_formal_artifacts(
                declaration['formal_artifacts'], run_root, protocol,
                inode_seen, artifacts,
            )
            data_flow_content, digest = _read_entry(
                declaration['results']['data_flow'], run_root,
                'data_flow_audit.jsonl', False, 'data-flow', inode_seen,
            )
            artifacts['data_flow'] = digest
            data_flow_records = _validate_data_flow(
                data_flow_content, protocol)
            _validate_formal_access_artifacts(
                formal_contents, protocol, data_flow_records,
                source_provenance,
            )
            del formal_contents
            results_entry = {
                key: declaration['results'][key] for key in _ENTRY_KEYS
            }
            results_content, digest = _read_entry(
                results_entry, run_root, 'results.json', False,
                'results', inode_seen,
            )
            artifacts['results'] = digest
            results = _json_value(results_content, 'results')
            metrics, trajectory_sha256 = _validate_results(
                results, protocol, manifest, source_provenance
            )
            artifacts['trajectory'] = trajectory_sha256
            final = len(_task_classes(protocol)) - 1
            checkpoint_content, digest = _read_entry(
                declaration['checkpoint'], run_root,
                'checkpoints/formal_final.pt', False,
                'checkpoint', inode_seen,
            )
            artifacts['checkpoint'] = digest
            _probe_checkpoint(
                checkpoint_content, protocol, manifest, data_contents,
                {key: results[key] for key in _TRACKER_KEYS},
                results.get('bic_history', []), data_flow_records,
                results.get('calibration_audit'),
                results.get('selection_audit'),
                run_root.path, declaration['formal_artifacts'], results,
                source_provenance,
            )
            run_root.verify()
            source_root.verify()
            data_root.verify()
        return AdmissionRecord(
            'REUSABLE', 'admitted', normalized_spec, protocol_sha256,
            source_sha256, dict(sorted(artifacts.items())), metrics,
            FORMULA_VERSION, trajectory_sha256,
        )
    except _EvidenceError as error:
        return _failure(error.status, error.reason,
                        normalized_spec, protocol_sha256)
    except (OSError, ValueError, TypeError):
        return _failure('REJECTED', 'evidence-invalid',
                        normalized_spec, protocol_sha256)


def _completed_spec_key(spec):
    return f'{spec.dataset}:{spec.method}:{spec.seed}'


def _completed_digest(value):
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _completed_hash(value, label):
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _reject(f'{label}-invalid')


def _completed_text(value, label):
    if type(value) is not str or not value or '\x00' in value:
        _reject(f'{label}-invalid')


def _completed_root_identity(value):
    if type(value) is not dict or set(value) != _ROOT_IDENTITY_KEYS:
        _reject('root-identity-invalid')
    for name in ('dev', 'inode', 'ctime_ns', 'size'):
        if type(value[name]) is not int or isinstance(value[name], bool) \
                or value[name] < 0:
            _reject('root-identity-invalid')
    _completed_hash(value['hash'], 'root-identity-hash')


def _completed_profile_binding():
    if experiment_profile() == RECOVERY_PROFILE:
        return {'experiment_profile': RECOVERY_PROFILE}
    if experiment_profile() in (
            SINGLE_DATASET_PROFILE, CONTINUATION_PROFILE,
            METHOD_SHARD_PROFILE):
        binding = {'experiment_profile': experiment_profile(),
                   'formal_dataset': selected_formal_dataset()}
        if experiment_profile() == METHOD_SHARD_PROFILE:
            binding['formal_method'] = selected_formal_method()
        return binding
    return {}


def _validate_completed_plan_identity(plan):
    binding = _completed_profile_binding()
    if type(plan) is not dict or set(plan) != _COMPLETED_PLAN_KEYS | set(binding):
        _reject('completed-plan-schema-invalid')
    if any(plan[name] != expected for name, expected in binding.items()):
        _reject('completed-plan-profile-invalid')
    formal = [_completed_spec_key(item) for item in formal_specs()]
    explanations = [_completed_spec_key(item) for item in explanation_specs()]
    if (plan['kind'] != 'formal_plan'
            or plan['registry_sha256'] != registry_sha256()
            or plan['metric_formula_version'] != FORMULA_VERSION
            or plan['formal_cells'] != formal
            or plan['explanation_cells'] != explanations):
        _reject('completed-plan-identity-invalid')
    missing = plan['missing_jobs']
    if (type(missing) is not list
            or any(type(value) is not str for value in missing)
            or len(missing) != len(set(missing))
            or missing != [value for value in formal if value in set(missing)]):
        _reject('completed-plan-membership-invalid')
    _completed_hash(plan['census_sha256'], 'completed-plan-census-hash')
    return _completed_digest(plan)


def _validate_completed_plan(plan, spec):
    plan_sha256 = _validate_completed_plan_identity(plan)
    key = _completed_spec_key(spec)
    if spec.explanation:
        if spec not in explanation_specs() or key not in plan['explanation_cells']:
            _reject('completed-run-not-planned')
    elif spec not in formal_specs() or key not in plan['missing_jobs']:
        _reject('completed-run-not-planned')
    return plan_sha256


def _read_completed_file(root, logical, require_json=False,
                         require_mode=False):
    root.verify()
    descriptor = root.open_file(logical)
    try:
        content, details = _read_descriptor(descriptor)
        root.verify_file_name(logical, details)
    finally:
        os.close(descriptor)
    root.verify()
    if details.st_nlink != 1:
        _reject('completed-evidence-not-immutable')
    if require_mode and stat.S_IMODE(details.st_mode) != 0o444:
        _reject('completed-control-not-immutable')
    value = None
    if require_json:
        value = _json_value(content, logical)
        if content != _canonical_json(value) + b'\n':
            _reject('completed-control-not-canonical')
    return content, details, value


def _validate_completed_resource(value, checkpoint_size):
    if type(value) is not dict or set(value) != _RESOURCE_KEYS:
        _reject('resource-schema-invalid')
    hardware = value['hardware_identity']
    if type(hardware) is not dict or set(hardware) != _HARDWARE_KEYS:
        _reject('hardware-identity-invalid')
    for name in ('gpu_name', 'cuda', 'torch', 'driver'):
        _completed_text(hardware[name], f'hardware-{name}')
    if (type(hardware['gpu_count']) is not int
            or isinstance(hardware['gpu_count'], bool)
            or hardware['gpu_count'] <= 0):
        _reject('hardware-gpu-count-invalid')
    _completed_text(value['instrumentation'], 'resource-instrumentation')
    runtime = value['runtime_seconds']
    if type(runtime) is not float or not math.isfinite(runtime) or runtime < 0:
        _reject('resource-runtime-invalid')
    for name in (
            'peak_gpu_memory_bytes', 'added_parameters', 'communication_bytes',
            'raw_examples_per_class', 'persistent_embeddings'):
        item = value[name]
        if type(item) is not int or isinstance(item, bool) or item < 0:
            _reject(f'resource-{name}-invalid')
    size = value['checkpoint_size_bytes']
    if (type(size) is not int or isinstance(size, bool) or size <= 0
            or size != checkpoint_size):
        _reject('resource-checkpoint-size-invalid')
    for name in ('replay_type', 'privacy_label'):
        _completed_text(value[name], f'resource-{name}')


def _completed_run_evidence(spec, run_dir, plan, expected_formal_root=None):
    normalized_spec = _normalized_spec(spec)
    protocol, _ = _protocol(spec)
    run_path = _absolute(os.fspath(run_dir), 'completed-run-not-absolute')
    if (run_path.parent.name != 'runs'
            or run_path.name != safe_spec_name(spec)):
        _reject('completed-run-layout-invalid')
    formal_root = _PinnedRoot(run_path.parent.parent)
    if expected_formal_root is not None:
        expected_root = _PinnedRoot(expected_formal_root)
        try:
            if (_directory_identity(expected_root.details)
                    != _directory_identity(formal_root.details)):
                _reject('completed-cli-root-mismatch')
            expected_root.verify()
        finally:
            expected_root.close()
    root = None
    identities = {}
    try:
        import three_dataset_formal_driver as driver
        installed_plan, _ = driver._load_installed_plan(formal_root.path)
        if (type(plan) is not dict or not _schema_matches(plan, installed_plan)
                or plan != installed_plan
                or _canonical_json(plan) != _canonical_json(installed_plan)):
            _reject('completed-plan-authority-invalid')
        plan_sha256 = _validate_completed_plan(installed_plan, spec)
        actual_root_identity = driver._root_identity(formal_root.path)
        formal_root.verify()
        root = _PinnedRoot(run_path)
        run_identity = _directory_identity(root.details)

        def read(logical, require_json=False, require_mode=False):
            content, details, value = _read_completed_file(
                root, logical, require_json=require_json,
                require_mode=require_mode)
            identities[logical] = _identity(details)
            return content, details, value

        job_content, job_details, job = read(
            'FORMAL_JOB_SPEC.json', require_json=True, require_mode=True)
        claim_content, claim_details, claim = read(
            'CLAIM_OWNER.json', require_json=True, require_mode=True)
        launch_content, launch_details, launch = read(
            'LAUNCH_STARTED.json', require_json=True, require_mode=True)
        resource_content, resource_details, resource_evidence = read(
            'RESOURCE_EVIDENCE.json', require_json=True, require_mode=True)
        if type(job) is not dict or set(job) != _JOB_SPEC_KEYS:
            _reject('job-spec-schema-invalid')
        key = _completed_spec_key(spec)
        if (job['kind'] != 'formal_job_spec' or job['spec_key'] != key
                or job['spec'] != normalized_spec
                or not _schema_matches(job['spec'], normalized_spec)
                or job['registry_sha256'] != registry_sha256()
                or job['metric_formula_version'] != FORMULA_VERSION
                or job['plan_sha256'] != plan_sha256
                or _absolute(job['run_dir'], 'job-run-dir-invalid') != run_path
                or job['root_identity'] != actual_root_identity):
            _reject('job-spec-identity-invalid')
        commit = job['source_commit']
        if type(commit) is not str or _COMMIT.fullmatch(commit) is None:
            _reject('job-source-commit-invalid')
        authority = _PinnedRoot(Path(__file__).resolve().parent)
        try:
            if _git(authority, 'rev-parse', 'HEAD').decode().strip() != commit:
                _reject('job-source-commit-invalid')
        finally:
            authority.close()
        source_sha256 = job['source_sha256']
        if (type(source_sha256) is not dict
                or set(source_sha256) != set(_SOURCE_INVENTORY)):
            _reject('job-source-map-invalid')
        for digest in source_sha256.values():
            _completed_hash(digest, 'job-source-hash')
        command = job['command']
        if (type(command) is not list or not command
                or any(type(token) is not str or not token or '\x00' in token
                       for token in command)):
            _reject('job-command-invalid')
        _completed_hash(job['command_sha256'], 'job-command-hash')
        if job['command_sha256'] != _completed_digest(command):
            _reject('job-command-hash-mismatch')
        config_content, _, _ = read('config.json')
        config = _json_value(config_content, 'config')
        if (_absolute(config.get('output_dir'), 'config-output-invalid')
                != run_path
                or _absolute(config.get('results_dir'), 'config-results-invalid')
                != run_path.parent):
            _reject('job-command-deployment-mismatch')
        try:
            expected_command = list(command_for(
                spec, config.get('device'), str(run_path.parent), smoke=False))
            parsed_protocol(command, spec)
        except (OSError, TypeError, ValueError):
            _reject('job-command-authority-invalid')
        if command != expected_command:
            _reject('job-command-authority-invalid')
        _completed_root_identity(job['root_identity'])

        if type(claim) is not dict or set(claim) != _CLAIM_OWNER_KEYS:
            _reject('claim-owner-schema-invalid')
        if (claim['kind'] != 'formal_job_claim' or claim['job'] != key
                or claim['source_commit'] != commit
                or claim['root_identity'] != job['root_identity']
                or claim['phase'] != (
                    'explanation' if spec.explanation else 'formal')):
            _reject('claim-owner-identity-invalid')
        for name in ('launcher_token', 'worker_role', 'process_start_time'):
            _completed_text(claim[name], f'claim-{name}')
        if type(claim['pid']) is not int or isinstance(claim['pid'], bool) \
                or claim['pid'] <= 0:
            _reject('claim-pid-invalid')
        if type(claim['pgid']) is not int or isinstance(claim['pgid'], bool) \
                or claim['pgid'] <= 0:
            _reject('claim-pgid-invalid')
        _completed_root_identity(claim['root_identity'])

        job_sha256 = hashlib.sha256(job_content).hexdigest()
        claim_sha256 = hashlib.sha256(claim_content).hexdigest()
        if type(launch) is not dict or set(launch) != _LAUNCH_STARTED_KEYS:
            _reject('launch-started-schema-invalid')
        if (launch['kind'] != 'formal_launch_started'
                or launch['spec_key'] != key
                or launch['plan_sha256'] != plan_sha256
                or launch['job_spec_sha256'] != job_sha256
                or launch['claim_sha256'] != claim_sha256
                or launch['command_sha256'] != job['command_sha256']
                or launch['source_commit'] != commit
                or launch['root_identity'] != claim['root_identity']
                or launch['worker_role'] != claim['worker_role']
                or launch['phase'] != claim['phase']
                or launch['pid'] != claim['pid']
                or launch['pgid'] != claim['pgid']
                or launch['process_start_time'] != claim['process_start_time']):
            _reject('launch-started-identity-invalid')
        launch_sha256 = hashlib.sha256(launch_content).hexdigest()
        installed_claim_path = (
            formal_root.path / 'claims' / driver._claim_name(key))
        try:
            installed_owner, installed_started = driver._read_started_claim(
                installed_claim_path, key, actual_root_identity, commit)
            expected_started = driver._started_payload(
                claim, run_path, plan_sha256, job['command_sha256'],
                installed_started.get('disk_reservation_sha256'))
        except (OSError, TypeError, ValueError):
            _reject('installed-claim-invalid')
        if (installed_owner != claim
                or not _schema_matches(installed_owner, claim)
                or installed_started != expected_started
                or not _schema_matches(installed_started, expected_started)):
            _reject('installed-claim-mismatch')

        if (type(resource_evidence) is not dict
                or set(resource_evidence) != _RESOURCE_EVIDENCE_KEYS):
            _reject('resource-evidence-schema-invalid')
        if (resource_evidence['kind'] != 'formal_resource_evidence'
                or resource_evidence['spec_key'] != key
                or resource_evidence['plan_sha256'] != plan_sha256
                or resource_evidence['job_spec_sha256'] != job_sha256
                or resource_evidence['claim_sha256'] != claim_sha256
                or resource_evidence['launch_sha256'] != launch_sha256
                or resource_evidence['command_sha256']
                    != job['command_sha256']):
            _reject('resource-evidence-identity-invalid')

        formal_logicals = _formal_artifact_logicals(protocol)
        artifact_logicals = (
            'config.json', 'results.json', 'data_flow_audit.jsonl',
            'validation/validation_manifest.json',
            'checkpoints/formal_final.pt', 'job.log', *formal_logicals,
        )
        declared_artifacts = resource_evidence['artifact_sha256']
        if (type(declared_artifacts) is not dict
                or set(declared_artifacts) != set(artifact_logicals)):
            _reject('resource-artifact-map-invalid')
        artifact_details = {}
        for logical in artifact_logicals:
            content, details, _ = read(
                logical, require_mode=(logical == 'job.log'))
            artifact_details[logical] = details
            _completed_hash(declared_artifacts[logical], 'resource-artifact-hash')
            if hashlib.sha256(content).hexdigest() != declared_artifacts[logical]:
                _reject('resource-artifact-hash-mismatch')
            if logical == 'job.log' and not content:
                _reject('job-log-empty')
        _validate_completed_resource(
            resource_evidence['resource'],
            artifact_details['checkpoints/formal_final.pt'].st_size,
        )
        if (job_details.st_mtime_ns > launch_details.st_mtime_ns
                or claim_details.st_mtime_ns > launch_details.st_mtime_ns
                or launch_details.st_mtime_ns > min(
                    details.st_mtime_ns for details in artifact_details.values())
                or resource_details.st_mtime_ns < max(
                    details.st_mtime_ns for details in artifact_details.values())):
            _reject('completed-control-time-invalid')

        data_root = _absolute(config.get('data_path'), 'config-data-invalid')
        expected_data = _AUTHORITATIVE_DATA.get(spec.dataset)
        if type(expected_data) is not dict or not expected_data:
            _reject('data-inventory-mismatch')
        source_root = Path(__file__).resolve().parent

        def entry(path, entry_root, digest, logical=None):
            value = {
                'path': str(path), 'root': str(entry_root), 'sha256': digest,
            }
            if logical is not None:
                value['logical_path'] = logical
            return value

        results = entry(
            run_path / 'results.json', run_path,
            declared_artifacts['results.json'])
        results['data_flow'] = entry(
            run_path / 'data_flow_audit.jsonl', run_path,
            declared_artifacts['data_flow_audit.jsonl'])
        declaration = {
            'spec': normalized_spec,
            'run_dir': str(run_path),
            'code_commit': commit,
            'source_files': [
                entry(source_root / logical, source_root,
                      source_sha256[logical], logical)
                for logical in _SOURCE_INVENTORY
            ],
            'data_files': [
                entry(data_root / logical, data_root,
                      expected_data[logical], logical)
                for logical in expected_data
            ],
            'validation_manifest': entry(
                run_path / 'validation/validation_manifest.json', run_path,
                declared_artifacts['validation/validation_manifest.json'],
                'validation/validation_manifest.json'),
            'config': entry(
                run_path / 'config.json', run_path,
                declared_artifacts['config.json']),
            'results': results,
            'checkpoint': entry(
                run_path / 'checkpoints/formal_final.pt', run_path,
                declared_artifacts['checkpoints/formal_final.pt']),
            'formal_artifacts': [
                entry(run_path / logical, run_path,
                      declared_artifacts[logical], logical)
                for logical in formal_logicals
            ],
        }
        root.verify()
        return {
            'declaration': declaration,
            'job': job,
            'claim': claim,
            'launch': launch,
            'resource': resource_evidence['resource'],
            'plan_sha256': plan_sha256,
            'job_sha256': job_sha256,
            'claim_sha256': claim_sha256,
            'launch_sha256': launch_sha256,
            'log_sha256': declared_artifacts['job.log'],
            'run_identity': run_identity,
            'identities': identities,
            'formal_root': str(formal_root.path),
            'formal_root_identity': _directory_identity(formal_root.details),
            'root_identity': actual_root_identity,
            'installed_plan': installed_plan,
            'installed_claim_path': str(installed_claim_path),
            'installed_owner': installed_owner,
            'installed_started': installed_started,
        }
    finally:
        if root is not None:
            root.close()
        formal_root.close()


def _verify_completed_snapshot(run_dir, evidence):
    import three_dataset_formal_driver as driver
    formal_root = _PinnedRoot(evidence['formal_root'])
    try:
        if (_directory_identity(formal_root.details)
                != evidence['formal_root_identity']
                or driver._root_identity(formal_root.path)
                    != evidence['root_identity']
                or driver._load_installed_plan(formal_root.path)[0]
                    != evidence['installed_plan']):
            _reject('completed-plan-authority-changed')
        try:
            owner, started = driver._read_started_claim(
                evidence['installed_claim_path'],
                evidence['installed_owner']['job'],
                evidence['root_identity'],
                evidence['installed_owner']['source_commit'])
        except (OSError, TypeError, ValueError):
            _reject('installed-claim-changed')
        if (owner != evidence['installed_owner']
                or started != evidence['installed_started']):
            _reject('installed-claim-changed')
        formal_root.verify()
    finally:
        formal_root.close()
    root = _PinnedRoot(run_dir)
    try:
        if _directory_identity(root.details) != evidence['run_identity']:
            _reject('completed-run-identity-changed')
        for logical, expected in evidence['identities'].items():
            descriptor = root.open_file(logical)
            try:
                details = os.fstat(descriptor)
                root.verify_file_name(logical, details)
            finally:
                os.close(descriptor)
            if _identity(details) != expected:
                _reject('completed-evidence-identity-changed')
        root.verify()
    finally:
        root.close()


def declaration_from_new_run(spec, run_dir, plan) -> dict:
    """Construct the historical declaration from exact completed-run paths."""
    return _completed_run_evidence(spec, run_dir, plan)['declaration']


def audit_completed_run(spec, run_dir, plan,
                        formal_root=None) -> AdmissionRecord:
    """Admit one completed planned run through immutable control evidence."""
    normalized_spec = _normalized_spec(spec)
    _, protocol_sha256 = _protocol(spec)
    try:
        evidence = _completed_run_evidence(
            spec, run_dir, plan, expected_formal_root=formal_root)
        record = audit_candidate(spec, evidence['declaration'])
        _verify_completed_snapshot(run_dir, evidence)
        if record.status == 'REUSABLE' and (
                record.spec != normalized_spec
                or record.protocol_sha256 != protocol_sha256
                or record.metric_formula_version != FORMULA_VERSION):
            _reject('completed-admission-identity-invalid')
        return record
    except _EvidenceError as error:
        return _failure(error.status, error.reason,
                        normalized_spec, protocol_sha256)
    except (OSError, ValueError, TypeError):
        return _failure('REJECTED', 'completed-run-control-invalid',
                        normalized_spec, protocol_sha256)


def _owned_name(parent_fd, name, details):
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return stat.S_ISREG(current.st_mode) and (
        current.st_dev, current.st_ino
    ) == (details.st_dev, details.st_ino)


def write_admission(record, destination) -> None:
    """Crash-safe exclusive installation through one pinned parent descriptor."""
    if not isinstance(record, AdmissionRecord):
        raise TypeError('record must be an AdmissionRecord')
    path = _absolute(os.fspath(destination), 'destination-not-absolute')
    parent = _PinnedRoot(path.parent)
    payload = _canonical_json(asdict(record)) + b'\n'
    temporary = f'.{path.name}.{secrets.token_hex(16)}.tmp'
    temp_fd = None
    temp_details = None
    linked = False
    try:
        flags = (os.O_RDWR | os.O_CREAT | os.O_EXCL
                 | getattr(os, 'O_NOFOLLOW', 0)
                 | getattr(os, 'O_CLOEXEC', 0))
        temp_fd = os.open(temporary, flags, 0o400, dir_fd=parent.fd)
        temp_details = os.fstat(temp_fd)
        if not stat.S_ISREG(temp_details.st_mode):
            raise RuntimeError('admission temp is not regular')
        view = memoryview(payload)
        while view:
            written = os.write(temp_fd, view)
            if written <= 0:
                raise OSError('short admission write')
            view = view[written:]
        os.fchmod(temp_fd, 0o444)
        os.fsync(temp_fd)
        after = os.fstat(temp_fd)
        os.lseek(temp_fd, 0, os.SEEK_SET)
        installed = bytearray()
        while True:
            chunk = os.read(temp_fd, 1024 * 1024)
            if not chunk:
                break
            installed.extend(chunk)
        if (_identity(after) != _identity(os.fstat(temp_fd))
                or after.st_size != len(payload) or bytes(installed) != payload
                or stat.S_IMODE(after.st_mode) != 0o444):
            raise RuntimeError('admission temp verification failed')
        parent.verify()
        os.link(
            temporary, path.name, src_dir_fd=parent.fd,
            dst_dir_fd=parent.fd, follow_symlinks=False,
        )
        linked = True
        os.unlink(temporary, dir_fd=parent.fd)
        os.fsync(parent.fd)
        final = os.stat(path.name, dir_fd=parent.fd, follow_symlinks=False)
        if ((final.st_dev, final.st_ino) != (after.st_dev, after.st_ino)
                or final.st_size != len(payload)
                or stat.S_IMODE(final.st_mode) != 0o444):
            raise RuntimeError('installed admission identity mismatch')
        verify_fd = os.open(
            path.name,
            os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
            | getattr(os, 'O_CLOEXEC', 0),
            dir_fd=parent.fd,
        )
        try:
            verified, verified_details = _read_descriptor(verify_fd)
            verified_name = os.stat(
                path.name, dir_fd=parent.fd, follow_symlinks=False)
        finally:
            os.close(verify_fd)
        identity_fields = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns')
        if (tuple(getattr(verified_name, name) for name in identity_fields)
                != tuple(getattr(verified_details, name)
                         for name in identity_fields)
                or (verified_details.st_dev, verified_details.st_ino)
                != (after.st_dev, after.st_ino)
                or hashlib.sha256(verified).digest()
                != hashlib.sha256(payload).digest()):
            raise RuntimeError('installed admission hash mismatch')
        parent.verify()
    except Exception:
        if linked and temp_details is not None \
                and _owned_name(parent.fd, path.name, temp_details):
            os.unlink(path.name, dir_fd=parent.fd)
        if temp_details is not None \
                and _owned_name(parent.fd, temporary, temp_details):
            os.unlink(temporary, dir_fd=parent.fd)
        try:
            os.fsync(parent.fd)
        except OSError:
            pass
        raise
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        parent.close()


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == '--strict-probe':
        try:
            _strict_probe_main(sys.argv[2], sys.argv[3], sys.argv[4])
        except RuntimeError as error:
            if str(error) == 'UNSAFE_CHECKPOINT':
                print('UNSAFE_CHECKPOINT')
                raise SystemExit(3)
            print('STRICT_PROBE_FAILED')
            raise SystemExit(4)
        except Exception:
            print('STRICT_PROBE_FAILED')
            raise SystemExit(4)
        print('PROBE_OK')
    else:
        raise SystemExit('internal module; no command-line interface')
