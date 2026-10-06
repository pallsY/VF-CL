"""Immutable census, plan, and owner-bound claims for formal comparisons."""
from __future__ import annotations
import argparse
import contextlib
import csv
from dataclasses import asdict
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import re
import runpy
import secrets
import stat
import statistics
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import zipfile

import three_dataset_formal_registry as formal_registry

from three_dataset_formal_audit import (
    AdmissionRecord, _PinnedRoot, _canonical_json, _directory_identity,
    _identity, _json_value, _owned_name, _read_descriptor,
    _completed_run_evidence,
    _formal_artifact_logicals, _SOURCE_INVENTORY,
    _validate_completed_plan_identity,
    _verify_completed_snapshot, audit_candidate, audit_completed_run,
    declaration_from_new_run,
)
from three_dataset_formal_metrics import FORMULA_VERSION, FormalMetrics
from three_dataset_formal_registry import (
    FormalSpec, command_for, explanation_specs, formal_specs, protocol_for,
    experiment_profile, profile_cardinality, FORMAL_PROFILE, PILOT_PROFILE,
    RECOVERY_PROFILE, FULL_MATRIX_PROFILE, SINGLE_DATASET_PROFILE,
    CONTINUATION_PROFILE, METHOD_SHARD_PROFILE,
    selected_formal_dataset, selected_formal_method,
    registered_spec_key, registry_sha256, safe_spec_key, safe_spec_name,
    spec_for_key as registry_spec_for_key,
)


MAX_PIPELINE_INFLIGHT = 3


def pipeline_inflight_limit():
    if experiment_profile() in (
            SINGLE_DATASET_PROFILE, CONTINUATION_PROFILE,
            METHOD_SHARD_PROFILE):
        count = os.environ.get('VFCL_GPU_COUNT', '1')
        if count not in ('1', '2'):
            raise ValueError('single-dataset GPU count must be 1 or 2')
        return int(count)
    return 1 if experiment_profile() == RECOVERY_PROFILE else MAX_PIPELINE_INFLIGHT


class PipelineBackpressure(BlockingIOError):
    """All pipeline reservations are occupied; poll without claiming a job."""


_CENSUS_KIND = 'formal_compatibility_census'
_PLAN_KIND = 'formal_plan'
_REGISTRY_KIND = 'formal_registry'
_JOBS_KIND = 'formal_missing_jobs'
_SHA256 = re.compile(r'[0-9a-f]{64}')
_COMMIT = re.compile(r'[0-9a-f]{40}')
_CENSUS_KEYS = {
    'kind', 'registry_sha256', 'metric_formula_version', 'records',
}
_PLAN_KEYS = {
    'kind', 'registry_sha256', 'metric_formula_version', 'formal_cells',
    'missing_jobs', 'explanation_cells', 'census_sha256',
}
_ADMISSION_KEYS = {
    'status', 'reason', 'spec', 'protocol_sha256', 'source_sha256',
    'artifact_sha256', 'metrics', 'metric_formula_version',
    'trajectory_sha256',
}
_RECORD_KEYS = _ADMISSION_KEYS | {
    'spec_key', 'admission_record_sha256',
}
_OWNER_KEYS = {
    'kind', 'job', 'launcher_token', 'worker_role', 'pid',
    'pgid', 'phase', 'process_start_time', 'source_commit', 'root_identity',
}
_STARTED_KEYS = {
    'kind', 'job', 'owner_sha256', 'plan_sha256', 'source_commit',
    'command_sha256', 'run_dir', 'root_identity', 'worker_role', 'pid',
    'process_start_time', 'pgid', 'phase',
}
_ROOT_IDENTITY_KEYS = {'dev', 'inode', 'ctime_ns', 'size', 'hash'}
_ROOT_RECORD_KEYS = {
    'kind', 'token', 'root_dev', 'root_inode', 'root_ctime_ns', 'root_size',
    'registry_sha256', 'plan_sha256', 'missing_jobs_sha256', 'source_commit',
}
_FORMAL_OUTPUT_NAMES = frozenset({
    'FORMAL_REGISTRY.json', 'COMPATIBILITY_CENSUS.json',
    'FORMAL_PLAN.json', 'MISSING_JOBS.json', 'FORMAL_ROOT_IDENTITY.json',
})
_METRIC_KEYS = {
    'aa_final', 'bwt', 'taskil_final', 'aa_trajectory', 'class_final',
    'taskil_final_by_task',
}
_COMPLETED_RECORD_KEYS = {
    'kind', 'spec_key', 'dataset', 'method', 'seed', 'explanation',
    'registry_sha256', 'metric_formula_version', 'plan_sha256',
    'source_commit', 'protocol_sha256', 'trajectory_sha256',
    'admission_record_sha256', 'artifact_sha256', 'metrics',
    'command_sha256', 'log_sha256', 'claim_sha256', 'launch_sha256',
    'resource', 'record_sha256',
}
_RESOURCE_KEYS = {
    'hardware_identity', 'instrumentation', 'runtime_seconds',
    'peak_gpu_memory_bytes', 'checkpoint_size_bytes', 'added_parameters',
    'communication_bytes', 'replay_type', 'raw_examples_per_class',
    'persistent_embeddings', 'privacy_label',
}
_HARDWARE_KEYS = {'gpu_name', 'gpu_count', 'cuda', 'torch', 'driver'}
_GPU_OWNER_KEYS = _OWNER_KEYS | {'physical_gpu'}
_TABLE_NAMES = (
    'FORMAL_PER_RUN.csv', 'FORMAL_TABLE.csv',
    'RESOURCE_PRIVACY_TABLE.csv', 'MECHANISM_TABLE.csv',
)
if experiment_profile() == PILOT_PROFILE:
    _TABLE_NAMES = (
        'FORMAL_PER_RUN.csv', 'PILOT_TABLE.csv', 'RESOURCE_PRIVACY_TABLE.csv',
    )
elif experiment_profile() == RECOVERY_PROFILE:
    _TABLE_NAMES = (
        'FORMAL_PER_RUN.csv', 'RECOVERY_TABLE.csv',
        'RESOURCE_PRIVACY_TABLE.csv',
    )
elif experiment_profile() in (SINGLE_DATASET_PROFILE, METHOD_SHARD_PROFILE):
    _TABLE_NAMES = (
        'FORMAL_PER_RUN.csv', 'FORMAL_TABLE.csv',
        'RESOURCE_PRIVACY_TABLE.csv',
    )
_CONTINUATION_TABLE_NAMES = (
    'CIFAR_CONTINUATION_PER_RUN.csv', 'CIFAR_CONTINUATION_TABLE.csv',
    'CIFAR_CONTINUATION_RESOURCE.csv', 'CIFAR_CONTINUATION_AUDIT.json',
)
_SMOKE_KIND = 'three_dataset_generated_smoke_plan'
_SMOKE_JOB_SHAPES = (
    ('image-replay-free', 'image', 'replay_free', 'cifar100', 'finetune'),
    ('vector-raw-replay', 'vector', 'raw_replay', 'isolet', 'er'),
    ('vector-fixed-endpoint', 'vector', 'fixed_endpoint', 'isolet', 'fixed_full'),
    ('vector-adaptive', 'vector', 'adaptive', 'isolet', 'adaptive'),
)
_SMOKE_TASKS = '0,1|2,3'
_FULL_SMOKE_CIFAR_TASKS = (
    '0,1|2,3|4,5|6,7|8,9|10,11|12,13|14,15|16,17|18,19'
)
_FULL_SMOKE_CIFAR_CLASSES = 20
_FULL_SMOKE_CIFAR_TASK_COUNT = 10
_SMOKE_FORMAL_NAMES = frozenset({
    *_FORMAL_OUTPUT_NAMES,
    'FORMAL_PHASE_SUCCESS', 'EXPLANATION_PHASE_SUCCESS',
    'FORMAL_EXECUTION_SUCCESS', 'FORMAL_STOPPED', 'FAILED_JOB',
    'PILOT_PHASE_SUCCESS', 'PILOT_EXECUTION_SUCCESS',
    'RECOVERY_PHASE_SUCCESS', 'RECOVERY_EXECUTION_SUCCESS',
    'DATASET_PHASE_SUCCESS', 'DATASET_EXECUTION_SUCCESS',
    'DATASET_CONTINUATION_PHASE_SUCCESS', 'DATASET_CONTINUATION_SUCCESS',
    'METHOD_SHARD_PHASE_SUCCESS', 'METHOD_SHARD_SUCCESS',
    'FULL_MATRIX_REUSE.json', 'FULL_MATRIX_PHASE_SUCCESS',
    'FULL_MATRIX_EXECUTION_SUCCESS',
})
_SMOKE_CONTROL_DIRS = frozenset({
    'fixtures', 'runs', 'logs', 'records', 'control',
})


def _digest(value):
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _profile_binding():
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


def _validate_profile_binding(value):
    binding = _profile_binding()
    if any(value.get(name) != expected for name, expected in binding.items()):
        raise ValueError('experiment profile authority differs')


def _exact_equal(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _exact_equal(left[key], right[key]) for key in left
        )
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(
            _exact_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _hash_string(value, label):
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f'{label} must be a lowercase SHA-256')


def _nonempty_string(value, label):
    if type(value) is not str or not value or '\x00' in value:
        raise ValueError(f'{label} must be a nonempty string')


def _path(value, label):
    try:
        raw = os.fspath(value)
    except TypeError as error:
        raise TypeError(f'{label} must be path-like') from error
    if type(raw) is not str or not raw or '\x00' in raw:
        raise ValueError(f'{label} must be a nonempty path')
    if not Path(raw).is_absolute():
        raise ValueError(f'{label} must be absolute')
    if '..' in raw.split(os.path.sep):
        raise ValueError(f'{label} must not escape through ..')
    path = Path(os.path.abspath(raw))
    if path == Path(path.anchor) or not path.name:
        raise ValueError(f'{label} must name a child path')
    return path


def spec_key(spec) -> str:
    return registered_spec_key(spec)


def _protocol_sha256(spec):
    protocol = json.loads(json.dumps(protocol_for(spec)))
    return _digest(protocol)


def _validate_metrics(metrics, spec, sequence_type):
    if type(metrics) is not dict or set(metrics) != _METRIC_KEYS:
        raise ValueError('formal metrics projection is invalid')
    for name in ('aa_final', 'bwt', 'taskil_final'):
        value = metrics[name]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f'formal metric {name} is invalid')
    task_count = protocol_for(spec)['base_options']['num_tasks']
    for name in ('aa_trajectory', 'class_final', 'taskil_final_by_task'):
        values = metrics[name]
        if type(values) is not sequence_type or len(values) != task_count:
            raise ValueError(f'formal metric {name} task count differs')
        if any(type(value) is not float or not math.isfinite(value)
               for value in values):
            raise ValueError(f'formal metric {name} contains an invalid value')
    if metrics['aa_final'] != statistics.fmean(metrics['class_final']):
        raise ValueError('formal AA final does not match class-final mean')
    if (metrics['taskil_final']
            != statistics.fmean(metrics['taskil_final_by_task'])):
        raise ValueError('formal Task-IL final does not match task mean')
    if metrics['aa_trajectory'][-1] != metrics['aa_final']:
        raise ValueError('formal AA trajectory does not end at AA final')


def _validate_admission(spec, record):
    if type(record) is not AdmissionRecord:
        raise ValueError('audit_candidate returned an incomplete admission record')
    raw = asdict(record)
    if set(raw) != _ADMISSION_KEYS:
        raise ValueError('admission record schema is invalid')
    if not _exact_equal(raw['spec'], asdict(spec)):
        raise ValueError('admission record spec differs from registry cell')
    if raw['status'] not in {'REUSABLE', 'REJECTED', 'RERUN_REQUIRED'}:
        raise ValueError('admission status is invalid')
    _nonempty_string(raw['reason'], 'admission reason')
    if raw['protocol_sha256'] != _protocol_sha256(spec):
        raise ValueError('admission protocol identity differs')
    if raw['metric_formula_version'] != FORMULA_VERSION:
        raise ValueError('admission metric formula differs')
    if type(raw['artifact_sha256']) is not dict:
        raise ValueError('admission artifact identity is invalid')
    for name, digest in raw['artifact_sha256'].items():
        _nonempty_string(name, 'artifact name')
        _hash_string(digest, 'artifact digest')
    reusable = raw['status'] == 'REUSABLE'
    if reusable:
        if type(record.metrics) is not FormalMetrics:
            raise ValueError('reusable admission metrics type is invalid')
        if raw['reason'] != 'admitted':
            raise ValueError('reusable admission reason is invalid')
        _hash_string(raw['source_sha256'], 'source identity')
        _hash_string(raw['trajectory_sha256'], 'trajectory identity')
        if not raw['artifact_sha256'] or type(raw['metrics']) is not dict:
            raise ValueError('reusable admission evidence is incomplete')
        _validate_metrics(raw['metrics'], spec, tuple)
    elif (raw['source_sha256'] != '' or raw['artifact_sha256'] != {}
          or raw['metrics'] is not None or raw['trajectory_sha256'] != ''):
        raise ValueError('non-reusable admission carries evidence')
    encoded = _canonical_json(raw)
    safe = json.loads(encoded)
    return {
        'spec_key': spec_key(spec),
        **safe,
        'admission_record_sha256': hashlib.sha256(encoded).hexdigest(),
    }


def _missing_admission(spec):
    record = AdmissionRecord(
        status='RERUN_REQUIRED',
        reason='no_declared_candidate',
        spec=asdict(spec),
        protocol_sha256=_protocol_sha256(spec),
        source_sha256='',
        artifact_sha256={},
        metrics=None,
        metric_formula_version=FORMULA_VERSION,
        trajectory_sha256='',
    )
    return _validate_admission(spec, record)


def build_census(declarations) -> dict:
    if type(declarations) is not dict:
        raise TypeError('declarations must be an exact object')
    if experiment_profile() in (
            SINGLE_DATASET_PROFILE, METHOD_SHARD_PROFILE) and declarations:
        raise ValueError('single-dataset formal roots forbid reuse declarations')
    if experiment_profile() == CONTINUATION_PROFILE and declarations:
        raise ValueError('continuation census requires a verified reuse bundle')
    specs = formal_specs()
    by_key = {spec_key(spec): spec for spec in specs}
    formal_count, _ = profile_cardinality()
    if len(by_key) != formal_count or len(by_key) != len(specs):
        raise ValueError(f'formal registry must contain {formal_count} unique cells')
    for key, declaration in declarations.items():
        if type(key) is not str or key not in by_key:
            raise ValueError(f'unknown formal declaration key: {key!r}')
        if type(declaration) is not dict:
            raise TypeError(f'declaration {key} must be an exact object')
    records = []
    for spec in specs:
        key = spec_key(spec)
        if key not in declarations:
            records.append(_missing_admission(spec))
        else:
            records.append(_validate_admission(
                spec, audit_candidate(spec, declarations[key])))
    return {
        'kind': _CENSUS_KIND,
        **_profile_binding(),
        'registry_sha256': registry_sha256(),
        'metric_formula_version': FORMULA_VERSION,
        'records': records,
    }


def _validate_record_payload(record, spec):
    if type(record) is not dict or set(record) != _RECORD_KEYS:
        raise ValueError('census record schema is invalid')
    if record['spec_key'] != spec_key(spec):
        raise ValueError('census record order or membership differs')
    if not _exact_equal(record['spec'], asdict(spec)):
        raise ValueError('census record spec differs')
    if record['status'] not in {'REUSABLE', 'REJECTED', 'RERUN_REQUIRED'}:
        raise ValueError('census record status is invalid')
    _nonempty_string(record['reason'], 'census reason')
    if record['protocol_sha256'] != _protocol_sha256(spec):
        raise ValueError('census protocol identity differs')
    if record['metric_formula_version'] != FORMULA_VERSION:
        raise ValueError('census metric formula differs')
    if type(record['artifact_sha256']) is not dict:
        raise ValueError('census artifact evidence is invalid')
    for name, digest in record['artifact_sha256'].items():
        _nonempty_string(name, 'artifact name')
        _hash_string(digest, 'artifact digest')
    reusable = record['status'] == 'REUSABLE'
    if reusable:
        if record['reason'] != 'admitted':
            raise ValueError('reusable census reason is invalid')
        _hash_string(record['source_sha256'], 'source identity')
        _hash_string(record['trajectory_sha256'], 'trajectory identity')
        if not record['artifact_sha256'] or type(record['metrics']) is not dict:
            raise ValueError('reusable census evidence is incomplete')
        _validate_metrics(record['metrics'], spec, list)
    elif (record['source_sha256'] != '' or record['artifact_sha256'] != {}
          or record['metrics'] is not None or record['trajectory_sha256'] != ''):
        raise ValueError('non-reusable census carries evidence')
    raw = {key: record[key] for key in _ADMISSION_KEYS}
    _hash_string(record['admission_record_sha256'], 'admission record digest')
    if record['admission_record_sha256'] != _digest(raw):
        raise ValueError('admission record digest differs')
    _canonical_json(record)


def _validate_census(census):
    if (type(census) is not dict
            or set(census) != _CENSUS_KEYS | set(_profile_binding())):
        raise ValueError('census schema is invalid')
    _validate_profile_binding(census)
    if census['kind'] != _CENSUS_KIND:
        raise ValueError('census kind differs')
    if census['registry_sha256'] != registry_sha256():
        raise ValueError('census registry identity differs')
    if census['metric_formula_version'] != FORMULA_VERSION:
        raise ValueError('census metric formula differs')
    specs = formal_specs()
    records = census['records']
    formal_count, _ = profile_cardinality()
    if (type(records) is not list or len(records) != len(specs)
            or len(specs) != formal_count):
        raise ValueError(f'census must cover all {formal_count} formal cells')
    for record, spec in zip(records, specs):
        _validate_record_payload(record, spec)
    if (experiment_profile() in (SINGLE_DATASET_PROFILE, METHOD_SHARD_PROFILE)
            and any(record['status'] != 'RERUN_REQUIRED' for record in records)):
        raise ValueError('single-dataset formal census forbids reuse')
    if experiment_profile() == CONTINUATION_PROFILE:
        from three_dataset_cifar_continuation_reconcile import origin_keys
        admitted = {record['spec_key'] for record in records
                    if record['status'] == 'REUSABLE'}
        if admitted != set(origin_keys()):
            raise ValueError('CIFAR continuation reuse membership differs')
    if len({record['spec_key'] for record in records}) != len(records):
        raise ValueError('census contains duplicate cells')


def build_plan(census) -> dict:
    _validate_census(census)
    formal = [spec_key(spec) for spec in formal_specs()]
    explanations = [spec_key(spec) for spec in explanation_specs()]
    _, explanation_count = profile_cardinality()
    if (len(explanations) != explanation_count
            or len(set(explanations)) != explanation_count):
        raise ValueError(
            f'explanation registry must contain {explanation_count} unique cells')
    return {
        'kind': _PLAN_KIND,
        **_profile_binding(),
        'registry_sha256': registry_sha256(),
        'metric_formula_version': FORMULA_VERSION,
        'formal_cells': formal,
        'missing_jobs': [
            record['spec_key'] for record in census['records']
            if record['status'] != 'REUSABLE'
        ],
        'explanation_cells': explanations,
        'census_sha256': _digest(census),
    }


def _registry_payload():
    return {
        'kind': _REGISTRY_KIND,
        **_profile_binding(),
        'registry_sha256': registry_sha256(),
        'metric_formula_version': FORMULA_VERSION,
        'formal_cells': [spec_key(spec) for spec in formal_specs()],
        'explanation_cells': [spec_key(spec) for spec in explanation_specs()],
    }


def _missing_jobs_payload(plan, census):
    _validate_plan_shape(plan, census)
    return {
        'kind': _JOBS_KIND,
        'registry_sha256': plan['registry_sha256'],
        'metric_formula_version': plan['metric_formula_version'],
        'missing_jobs': list(plan['missing_jobs']),
        'plan_sha256': _digest(plan),
    }


def _validate_plan_shape(plan, census):
    if (type(plan) is not dict
            or set(plan) != _PLAN_KEYS | set(_profile_binding())):
        raise ValueError('formal plan schema is invalid')
    _validate_profile_binding(plan)
    if plan['kind'] != _PLAN_KIND:
        raise ValueError('formal plan kind differs')
    _validate_census(census)
    if plan['registry_sha256'] != registry_sha256():
        raise ValueError('formal plan registry identity differs')
    if plan['metric_formula_version'] != FORMULA_VERSION:
        raise ValueError('formal plan metric formula differs')
    formal = [spec_key(spec) for spec in formal_specs()]
    explanations = [spec_key(spec) for spec in explanation_specs()]
    missing = [
        record['spec_key'] for record in census['records']
        if record['status'] != 'REUSABLE'
    ]
    if not _exact_equal(plan['formal_cells'], formal):
        raise ValueError('formal plan cells differ from current registry')
    if not _exact_equal(plan['explanation_cells'], explanations):
        raise ValueError('formal explanation cells differ from current registry')
    if not _exact_equal(plan['missing_jobs'], missing):
        raise ValueError('formal missing jobs differ from verified census')
    if (experiment_profile() in (SINGLE_DATASET_PROFILE, METHOD_SHARD_PROFILE)
            and not _exact_equal(missing, formal)):
        raise ValueError('dataset-scoped formal plan must train every cell')
    if experiment_profile() == CONTINUATION_PROFILE:
        from three_dataset_cifar_continuation_reconcile import origin_keys
        expected = [key for key in formal if key not in set(origin_keys())]
        if missing != expected or len(missing) != 24:
            raise ValueError('CIFAR continuation missing jobs differ')
    if plan['census_sha256'] != _digest(census):
        raise ValueError('formal plan census identity differs')
    _canonical_json(plan)


def _validate_record_resource(resource):
    if type(resource) is not dict or set(resource) != _RESOURCE_KEYS:
        raise ValueError('completed resource schema is invalid')
    hardware = resource['hardware_identity']
    if type(hardware) is not dict or set(hardware) != _HARDWARE_KEYS:
        raise ValueError('completed hardware identity is invalid')
    for name in ('gpu_name', 'cuda', 'torch', 'driver'):
        _nonempty_string(hardware[name], f'hardware {name}')
    if (type(hardware['gpu_count']) is not int
            or isinstance(hardware['gpu_count'], bool)
            or hardware['gpu_count'] <= 0):
        raise ValueError('hardware gpu_count is invalid')
    _nonempty_string(resource['instrumentation'], 'resource instrumentation')
    runtime = resource['runtime_seconds']
    if type(runtime) is not float or not math.isfinite(runtime) or runtime < 0:
        raise ValueError('resource runtime is invalid')
    for name in (
            'peak_gpu_memory_bytes', 'added_parameters', 'communication_bytes',
            'raw_examples_per_class', 'persistent_embeddings'):
        value = resource[name]
        if type(value) is not int or isinstance(value, bool) or value < 0:
            raise ValueError(f'resource {name} is invalid')
    size = resource['checkpoint_size_bytes']
    if type(size) is not int or isinstance(size, bool) or size <= 0:
        raise ValueError('resource checkpoint size is invalid')
    for name in ('replay_type', 'privacy_label'):
        _nonempty_string(resource[name], f'resource {name}')


def completed_run_record(spec, run_dir, plan, formal_root=None) -> dict:
    """Return the canonical record for one strictly admitted completed run."""
    evidence = _completed_run_evidence(
        spec, run_dir, plan, expected_formal_root=formal_root)
    admission = audit_completed_run(
        spec, run_dir, plan, formal_root=formal_root)
    _verify_completed_snapshot(run_dir, evidence)
    if admission.status != 'REUSABLE':
        raise ValueError(
            f'completed run is not reusable: {admission.reason}')
    validated = _validate_admission(spec, admission)
    record = {
        'kind': 'formal_completed_run',
        **_profile_binding(),
        'spec_key': spec_key(spec),
        'dataset': spec.dataset,
        'method': spec.method,
        'seed': spec.seed,
        'explanation': spec.explanation,
        'registry_sha256': registry_sha256(),
        'metric_formula_version': FORMULA_VERSION,
        'plan_sha256': evidence['plan_sha256'],
        'source_commit': evidence['job']['source_commit'],
        'protocol_sha256': validated['protocol_sha256'],
        'trajectory_sha256': validated['trajectory_sha256'],
        'admission_record_sha256': validated['admission_record_sha256'],
        'artifact_sha256': validated['artifact_sha256'],
        'metrics': validated['metrics'],
        'command_sha256': evidence['job']['command_sha256'],
        'log_sha256': evidence['log_sha256'],
        'claim_sha256': evidence['claim_sha256'],
        'launch_sha256': evidence['launch_sha256'],
        'resource': evidence['resource'],
    }
    record['record_sha256'] = _digest(record)
    return record


def _validate_completed_record(record, spec):
    if (type(record) is not dict
            or set(record) != _COMPLETED_RECORD_KEYS | set(_profile_binding())):
        raise ValueError('completed record schema is invalid')
    _validate_profile_binding(record)
    if (record['kind'] != 'formal_completed_run'
            or record['spec_key'] != spec_key(spec)
            or record['dataset'] != spec.dataset
            or record['method'] != spec.method
            or type(record['seed']) is not int
            or record['seed'] != spec.seed
            or type(record['explanation']) is not bool
            or record['explanation'] != spec.explanation):
        raise ValueError('completed record spec identity is invalid')
    if record['registry_sha256'] != registry_sha256():
        raise ValueError('completed record registry identity differs')
    if record['metric_formula_version'] != FORMULA_VERSION:
        raise ValueError('completed record metric formula differs')
    if type(record['source_commit']) is not str \
            or _COMMIT.fullmatch(record['source_commit']) is None:
        raise ValueError('completed record source commit is invalid')
    for name in (
            'plan_sha256', 'protocol_sha256', 'trajectory_sha256',
            'admission_record_sha256', 'command_sha256', 'log_sha256',
            'claim_sha256', 'launch_sha256', 'record_sha256'):
        _hash_string(record[name], f'completed record {name}')
    if record['protocol_sha256'] != _protocol_sha256(spec):
        raise ValueError('completed record protocol identity differs')
    artifacts = record['artifact_sha256']
    if type(artifacts) is not dict or not artifacts:
        raise ValueError('completed record artifact identity is invalid')
    for name, digest in artifacts.items():
        _nonempty_string(name, 'completed artifact name')
        _hash_string(digest, 'completed artifact digest')
    _validate_metrics(record['metrics'], spec, list)
    _validate_record_resource(record['resource'])
    payload = {key: value for key, value in record.items()
               if key != 'record_sha256'}
    if record['record_sha256'] != _digest(payload):
        raise ValueError('completed record digest differs')
    _canonical_json(record)


def summarize_records(records) -> dict:
    """Validate and summarize exactly the active profile's completed records."""
    if type(records) is not list:
        raise TypeError('records must be an exact list')
    specs = (*formal_specs(), *explanation_specs())
    formal_count, explanation_count = profile_cardinality()
    if len(records) != len(specs) or len(specs) != formal_count + explanation_count:
        raise ValueError(
            f'summary requires exactly {formal_count} formal and '
            f'{explanation_count} explanation records')
    expected = {spec_key(spec): spec for spec in specs}
    by_key = {}
    for record in records:
        if type(record) is not dict:
            raise TypeError('completed record must be an exact object')
        key = record.get('spec_key')
        if type(key) is not str or key not in expected or key in by_key:
            raise ValueError('summary record membership is invalid')
        _validate_completed_record(record, expected[key])
        by_key[key] = record
    if set(by_key) != set(expected):
        raise ValueError('summary record membership is incomplete')
    ordered = [json.loads(_canonical_json(by_key[spec_key(spec)]))
               for spec in specs]
    for name in (
            'registry_sha256', 'metric_formula_version', 'source_commit',
            'plan_sha256'):
        if len({record[name] for record in ordered}) != 1:
            raise ValueError(f'summary mixes {name}')

    resource_records = [
        by_key[spec_key(spec)] for spec in formal_specs() if spec.seed == 42
    ]
    hardware = resource_records[0]['resource']['hardware_identity']
    instrumentation = resource_records[0]['resource']['instrumentation']
    if any(
            not _exact_equal(record['resource']['hardware_identity'], hardware)
            or record['resource']['instrumentation'] != instrumentation
            for record in resource_records):
        raise ValueError('resource table mixes hardware or instrumentation')

    if experiment_profile() == RECOVERY_PROFILE:
        return {
            'kind': 'seed42_adaptive_recovery_completed_summary',
            **_profile_binding(),
            'registry_sha256': ordered[0]['registry_sha256'],
            'metric_formula_version': ordered[0]['metric_formula_version'],
            'plan_sha256': ordered[0]['plan_sha256'],
            'source_commit': ordered[0]['source_commit'],
            'records_sha256': _digest(ordered),
            'per_run_records': ordered,
            'recovery_rows': [{
                'dataset': record['dataset'], 'method': record['method'],
                'seed': record['seed'],
                'aa_final': record['metrics']['aa_final'],
                'bwt': record['metrics']['bwt'],
                'taskil_final': record['metrics']['taskil_final'],
            } for record in ordered],
            'formal_rows': [],
            'resource_rows': [json.loads(_canonical_json(record))
                              for record in resource_records],
            'mechanism_rows': [],
        }

    if experiment_profile() == PILOT_PROFILE:
        return {
            'kind': 'seed42_pilot_completed_summary',
            'registry_sha256': ordered[0]['registry_sha256'],
            'metric_formula_version': ordered[0]['metric_formula_version'],
            'plan_sha256': ordered[0]['plan_sha256'],
            'source_commit': ordered[0]['source_commit'],
            'records_sha256': _digest(ordered),
            'per_run_records': ordered,
            'pilot_rows': [{
                'dataset': record['dataset'], 'method': record['method'],
                'seed': record['seed'],
                'aa_final': record['metrics']['aa_final'],
                'bwt': record['metrics']['bwt'],
                'taskil_final': record['metrics']['taskil_final'],
            } for record in ordered],
            'formal_rows': [],
            'resource_rows': [json.loads(_canonical_json(record))
                              for record in resource_records],
            'mechanism_rows': [],
        }

    formal_rows = []
    for spec in formal_specs()[::3]:
        group = [
            by_key[spec_key(FormalSpec(spec.dataset, spec.method, seed))]
            for seed in (42, 43, 44)
        ]
        row = {'dataset': spec.dataset, 'method': spec.method}
        for name in ('aa_final', 'bwt', 'taskil_final'):
            values = [record['metrics'][name] for record in group]
            row[f'{name}_mean'] = statistics.fmean(values)
            row[f'{name}_std'] = statistics.stdev(values)
        formal_rows.append(row)

    if experiment_profile() in (SINGLE_DATASET_PROFILE, METHOD_SHARD_PROFILE):
        return {
            'kind': 'single_dataset_formal_completed_summary',
            **_profile_binding(),
            'registry_sha256': ordered[0]['registry_sha256'],
            'metric_formula_version': ordered[0]['metric_formula_version'],
            'plan_sha256': ordered[0]['plan_sha256'],
            'source_commit': ordered[0]['source_commit'],
            'records_sha256': _digest(ordered),
            'per_run_records': ordered,
            'formal_rows': formal_rows,
            'resource_rows': [json.loads(_canonical_json(record))
                              for record in resource_records],
            'mechanism_rows': [],
        }

    mechanism_methods = {
        'no_consolidation', 'fixed_full', 'fixed_bias', 'adaptive',
    }
    mechanism_records = [
        by_key[spec_key(spec)] for spec in formal_specs()
        if spec.seed == 42 and spec.method in mechanism_methods
    ] + [by_key[spec_key(spec)] for spec in explanation_specs()]
    mechanism_records.sort(key=lambda record: next(
        index for index, spec in enumerate(specs)
        if spec_key(spec) == record['spec_key']))
    # Frozen presentation order keeps the two explanations beside each
    # dataset's four registered mechanisms.
    mechanism_rows = []
    for dataset in dict.fromkeys(spec.dataset for spec in formal_specs()):
        for method in (
                'no_consolidation', 'fixed_full', 'fixed_bias', 'adaptive',
                'fixed_half', 'sample_mean_nll'):
            record = next(item for item in mechanism_records
                          if item['dataset'] == dataset
                          and item['method'] == method)
            mechanism_rows.append({
                'record': json.loads(_canonical_json(record)),
                'explanation_scope': (
                    'explanation' if record['explanation'] else 'formal'),
            })
    return {
        'kind': 'formal_completed_summary',
        'registry_sha256': ordered[0]['registry_sha256'],
        'metric_formula_version': ordered[0]['metric_formula_version'],
        'plan_sha256': ordered[0]['plan_sha256'],
        'source_commit': ordered[0]['source_commit'],
        'records_sha256': _digest(ordered),
        'per_run_records': ordered,
        'formal_rows': formal_rows,
        'resource_rows': [json.loads(_canonical_json(record))
                          for record in resource_records],
        'mechanism_rows': mechanism_rows,
    }


def _csv_bytes(header, rows):
    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode('utf-8')


def _decimal(value):
    return format(value, '.6f')


def render_tables(records) -> dict[str, bytes]:
    """Render all deterministic formal CSV tables in frozen registry order."""
    summary = summarize_records(records)
    per_run_header = (
        'kind', 'spec_key', 'dataset', 'method', 'seed', 'explanation',
        'registry_sha256', 'metric_formula_version', 'plan_sha256',
        'source_commit', 'protocol_sha256', 'trajectory_sha256',
        'admission_record_sha256', 'artifact_sha256', 'aa_final', 'bwt',
        'taskil_final', 'command_sha256', 'log_sha256', 'claim_sha256',
        'launch_sha256', 'record_sha256',
    )
    per_run_rows = []
    for record in summary['per_run_records']:
        per_run_rows.append((
            record['kind'], record['spec_key'], record['dataset'],
            record['method'], record['seed'],
            'true' if record['explanation'] else 'false',
            record['registry_sha256'], record['metric_formula_version'],
            record['plan_sha256'], record['source_commit'],
            record['protocol_sha256'], record['trajectory_sha256'],
            record['admission_record_sha256'],
            _canonical_json(record['artifact_sha256']).decode(),
            _decimal(record['metrics']['aa_final']),
            _decimal(record['metrics']['bwt']),
            _decimal(record['metrics']['taskil_final']),
            record['command_sha256'], record['log_sha256'],
            record['claim_sha256'], record['launch_sha256'],
            record['record_sha256'],
        ))
    formal_header = (
        'dataset', 'method', 'aa_final_mean', 'aa_final_std', 'bwt_mean',
        'bwt_std', 'taskil_final_mean', 'taskil_final_std',
    )
    formal_rows = [(
        row['dataset'], row['method'], _decimal(row['aa_final_mean']),
        _decimal(row['aa_final_std']), _decimal(row['bwt_mean']),
        _decimal(row['bwt_std']), _decimal(row['taskil_final_mean']),
        _decimal(row['taskil_final_std']),
    ) for row in summary['formal_rows']]
    resource_header = (
        'dataset', 'method', 'seed', 'gpu_name', 'gpu_count', 'cuda',
        'torch', 'driver', 'instrumentation', 'runtime_seconds',
        'peak_gpu_memory_bytes', 'checkpoint_size_bytes', 'added_parameters',
        'communication_bytes', 'replay_type', 'raw_examples_per_class',
        'persistent_embeddings', 'privacy_label',
    )
    resource_rows = []
    for record in summary['resource_rows']:
        resource = record['resource']
        hardware = resource['hardware_identity']
        resource_rows.append((
            record['dataset'], record['method'], record['seed'],
            hardware['gpu_name'], hardware['gpu_count'], hardware['cuda'],
            hardware['torch'], hardware['driver'],
            resource['instrumentation'], _decimal(resource['runtime_seconds']),
            resource['peak_gpu_memory_bytes'], resource['checkpoint_size_bytes'],
            resource['added_parameters'], resource['communication_bytes'],
            resource['replay_type'], resource['raw_examples_per_class'],
            resource['persistent_embeddings'], resource['privacy_label'],
        ))
    if experiment_profile() == RECOVERY_PROFILE:
        recovery_header = (
            'dataset', 'method', 'seed', 'aa_final', 'bwt', 'taskil_final')
        recovery_rows = [(
            row['dataset'], row['method'], row['seed'],
            _decimal(row['aa_final']), _decimal(row['bwt']),
            _decimal(row['taskil_final']),
        ) for row in summary['recovery_rows']]
        return {
            'FORMAL_PER_RUN.csv': _csv_bytes(per_run_header, per_run_rows),
            'RECOVERY_TABLE.csv': _csv_bytes(
                recovery_header, recovery_rows),
            'RESOURCE_PRIVACY_TABLE.csv': _csv_bytes(
                resource_header, resource_rows),
        }
    if experiment_profile() == PILOT_PROFILE:
        pilot_header = ('dataset', 'method', 'seed', 'aa_final', 'bwt', 'taskil_final')
        pilot_rows = [(
            row['dataset'], row['method'], row['seed'],
            _decimal(row['aa_final']), _decimal(row['bwt']),
            _decimal(row['taskil_final']),
        ) for row in summary['pilot_rows']]
        return {
            'FORMAL_PER_RUN.csv': _csv_bytes(per_run_header, per_run_rows),
            'PILOT_TABLE.csv': _csv_bytes(pilot_header, pilot_rows),
            'RESOURCE_PRIVACY_TABLE.csv': _csv_bytes(resource_header, resource_rows),
        }
    if experiment_profile() in (SINGLE_DATASET_PROFILE, METHOD_SHARD_PROFILE):
        return {
            'FORMAL_PER_RUN.csv': _csv_bytes(per_run_header, per_run_rows),
            'FORMAL_TABLE.csv': _csv_bytes(formal_header, formal_rows),
            'RESOURCE_PRIVACY_TABLE.csv': _csv_bytes(resource_header, resource_rows),
        }
    mechanism_header = (
        'dataset', 'method', 'seed', 'explanation_scope', 'aa_final', 'bwt',
        'taskil_final', 'trajectory_sha256', 'record_sha256',
    )
    mechanism_rows = []
    for row in summary['mechanism_rows']:
        record = row['record']
        mechanism_rows.append((
            record['dataset'], record['method'], record['seed'],
            row['explanation_scope'], _decimal(record['metrics']['aa_final']),
            _decimal(record['metrics']['bwt']),
            _decimal(record['metrics']['taskil_final']),
            record['trajectory_sha256'], record['record_sha256'],
        ))
    return {
        'FORMAL_PER_RUN.csv': _csv_bytes(per_run_header, per_run_rows),
        'FORMAL_TABLE.csv': _csv_bytes(formal_header, formal_rows),
        'RESOURCE_PRIVACY_TABLE.csv': _csv_bytes(
            resource_header, resource_rows),
        'MECHANISM_TABLE.csv': _csv_bytes(mechanism_header, mechanism_rows),
    }


def _validate_formal_root_path(value, create):
    path = _path(value, 'formal root')
    parent = _PinnedRoot(path.parent)
    try:
        parent.verify()
        try:
            details = os.stat(path.name, dir_fd=parent.fd,
                              follow_symlinks=False)
        except FileNotFoundError:
            if not create:
                parent.verify()
                return path
            os.mkdir(path.name, 0o700, dir_fd=parent.fd)
            os.fsync(parent.fd)
            details = os.stat(path.name, dir_fd=parent.fd,
                              follow_symlinks=False)
        if not stat.S_ISDIR(details.st_mode):
            raise ValueError('formal root is not a directory')
        parent.verify()
        root = _PinnedRoot(path)
        try:
            if ((root.details.st_dev, root.details.st_ino)
                    != (details.st_dev, details.st_ino)):
                raise ValueError('formal root identity changed')
            root.verify()
        finally:
            root.close()
        return path
    finally:
        parent.close()


def validate_formal_root(path) -> Path:
    return _validate_formal_root_path(path, create=True)


def install_json_exclusive(path, payload) -> None:
    """Install canonical JSON once through the audit's pinned trust helpers."""
    path = _path(path, 'destination')
    canonical = _canonical_json(payload) + b'\n'
    parent = _PinnedRoot(path.parent)
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
            raise RuntimeError('JSON temp is not regular')
        view = memoryview(canonical)
        while view:
            written = os.write(temp_fd, view)
            if written <= 0:
                raise OSError('short JSON write')
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
                or after.st_size != len(canonical)
                or bytes(installed) != canonical
                or stat.S_IMODE(after.st_mode) != 0o444):
            raise RuntimeError('JSON temp verification failed')
        parent.verify()
        os.link(temporary, path.name, src_dir_fd=parent.fd,
                dst_dir_fd=parent.fd, follow_symlinks=False)
        linked = True
        os.unlink(temporary, dir_fd=parent.fd)
        os.fsync(parent.fd)
        final = os.stat(path.name, dir_fd=parent.fd, follow_symlinks=False)
        if ((final.st_dev, final.st_ino) != (after.st_dev, after.st_ino)
                or final.st_size != len(canonical)
                or stat.S_IMODE(final.st_mode) != 0o444):
            raise RuntimeError('installed JSON identity mismatch')
        verify_fd = os.open(
            path.name,
            os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
            | getattr(os, 'O_CLOEXEC', 0), dir_fd=parent.fd)
        try:
            verified, verified_details = _read_descriptor(verify_fd)
            named = os.stat(path.name, dir_fd=parent.fd,
                            follow_symlinks=False)
        finally:
            os.close(verify_fd)
        fields = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns')
        if (tuple(getattr(named, name) for name in fields)
                != tuple(getattr(verified_details, name) for name in fields)
                or (verified_details.st_dev, verified_details.st_ino)
                != (after.st_dev, after.st_ino)
                or verified != canonical):
            raise RuntimeError('installed JSON verification failed')
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


def _install_table_exclusive(parent, name, payload):
    temporary = f'.{name}.{secrets.token_hex(16)}.tmp'
    descriptor = None
    details = None
    linked = False
    try:
        flags = (os.O_RDWR | os.O_CREAT | os.O_EXCL
                 | getattr(os, 'O_NOFOLLOW', 0)
                 | getattr(os, 'O_CLOEXEC', 0))
        descriptor = os.open(temporary, flags, 0o400, dir_fd=parent.fd)
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise RuntimeError('table temp is not regular')
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError('short table write')
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        installed = bytearray()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            installed.extend(chunk)
        if (_identity(after) != _identity(os.fstat(descriptor))
                or bytes(installed) != payload
                or after.st_size != len(payload)
                or stat.S_IMODE(after.st_mode) != 0o444):
            raise RuntimeError('table temp verification failed')
        parent.verify()
        os.link(temporary, name, src_dir_fd=parent.fd,
                dst_dir_fd=parent.fd, follow_symlinks=False)
        linked = True
        os.unlink(temporary, dir_fd=parent.fd)
        os.fsync(parent.fd)
        final = os.stat(name, dir_fd=parent.fd, follow_symlinks=False)
        if ((final.st_dev, final.st_ino) != (after.st_dev, after.st_ino)
                or final.st_size != len(payload)
                or stat.S_IMODE(final.st_mode) != 0o444):
            raise RuntimeError('installed table identity mismatch')
        verify = os.open(
            name, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
            | getattr(os, 'O_CLOEXEC', 0), dir_fd=parent.fd)
        try:
            content, verified = _read_descriptor(verify)
            parent.verify_file_name(name, verified)
        finally:
            os.close(verify)
        if (_identity(verified) != _identity(final) or content != payload):
            raise RuntimeError('installed table verification failed')
        parent.verify()
        return final
    except Exception:
        if linked and details is not None \
                and _owned_name(parent.fd, name, details):
            os.unlink(name, dir_fd=parent.fd)
        if details is not None \
                and _owned_name(parent.fd, temporary, details):
            os.unlink(temporary, dir_fd=parent.fd)
        try:
            os.fsync(parent.fd)
        except OSError:
            pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def install_tables(records, destination) -> dict:
    """Install the complete deterministic table set without overwrite."""
    rendered = render_tables(records)
    if tuple(rendered) != _TABLE_NAMES:
        raise ValueError('rendered table membership differs')
    destination = _validate_formal_root_path(destination, create=True)
    parent = _PinnedRoot(destination)
    installed = {}
    try:
        for name in _TABLE_NAMES:
            try:
                os.stat(name, dir_fd=parent.fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise FileExistsError(name)
        for name in _TABLE_NAMES:
            installed[name] = _install_table_exclusive(
                parent, name, rendered[name])
        return {
            name: hashlib.sha256(rendered[name]).hexdigest()
            for name in _TABLE_NAMES
        }
    except Exception:
        for name, details in installed.items():
            if _owned_name(parent.fd, name, details):
                os.unlink(name, dir_fd=parent.fd)
        try:
            os.fsync(parent.fd)
        except OSError:
            pass
        raise
    finally:
        parent.close()


def _read_json_from_pinned(root, name, canonical=False):
    if (type(name) is not str or not name or '/' in name
            or name in ('.', '..')):
        raise ValueError('JSON file name is invalid')
    root.verify()
    descriptor = root.open_file(name)
    try:
        content, details = _read_descriptor(descriptor)
        root.verify_file_name(name, details)
    finally:
        os.close(descriptor)
    root.verify()
    value = _json_value(content, name)
    if canonical and (stat.S_IMODE(details.st_mode) != 0o444
                      or content != _canonical_json(value) + b'\n'):
        raise ValueError(f'{name} is not immutable canonical JSON')
    return value


def _load_json_document(path, require_object):
    path = _path(path, 'JSON input')
    parent = _PinnedRoot(path.parent)
    try:
        parent.verify()
        descriptor = parent.open_file(path.name)
        try:
            content, details = _read_descriptor(descriptor)
            parent.verify_file_name(path.name, details)
        finally:
            os.close(descriptor)
        parent.verify()
        return _json_value(
            content, path.name, require_object=require_object)
    finally:
        parent.close()


def _load_json_file(path):
    return _load_json_document(path, require_object=True)


def _load_records_file(path):
    return _load_json_document(path, require_object=False)


def _read_installed(root_path, name):
    if name not in _FORMAL_OUTPUT_NAMES:
        raise ValueError('formal output name is not registered')
    root_path = _validate_formal_root_path(root_path, create=False)
    root = _PinnedRoot(root_path)
    try:
        return _read_json_from_pinned(root, name, canonical=True)
    finally:
        root.close()


def _read_root_identity_record(pinned, plan, manifest):
    pinned.verify()
    descriptor = pinned.open_file('FORMAL_ROOT_IDENTITY.json')
    try:
        content, details = _read_descriptor(descriptor)
        pinned.verify_file_name('FORMAL_ROOT_IDENTITY.json', details)
    finally:
        os.close(descriptor)
    pinned.verify()
    record = _json_value(content, 'FORMAL_ROOT_IDENTITY.json')
    if (stat.S_IMODE(details.st_mode) != 0o444 or details.st_nlink != 1
            or content != _canonical_json(record) + b'\n'):
        raise ValueError('formal root identity is not immutable canonical JSON')
    if (type(record) is not dict
            or set(record) != _ROOT_RECORD_KEYS | set(_profile_binding()) | set(_full_authority_binding(pinned))):
        raise ValueError('formal root identity schema is invalid')
    _validate_profile_binding(record)
    binding = _full_authority_binding(pinned)
    if any(record[name] != value for name, value in binding.items()):
        raise ValueError('full matrix reuse authority differs')
    if record['kind'] != 'formal_root_identity':
        raise ValueError('formal root identity kind differs')
    _hash_string(record['token'], 'formal root identity token')
    for name in ('root_dev', 'root_inode', 'root_ctime_ns', 'root_size'):
        if (type(record[name]) is not int or isinstance(record[name], bool)
                or record[name] < 0):
            raise ValueError(f'formal root identity {name} is invalid')
    for name in ('registry_sha256', 'plan_sha256', 'missing_jobs_sha256'):
        _hash_string(record[name], f'formal root identity {name}')
    if (record['root_dev'] != pinned.details.st_dev
            or record['root_inode'] != pinned.details.st_ino
            or record['registry_sha256'] != registry_sha256()
            or record['plan_sha256'] != _digest(plan)
            or record['missing_jobs_sha256'] != _digest(manifest)
            or record['source_commit'] != _source_commit()):
        raise ValueError('formal root identity authority differs')
    return record, content, details


def _load_installed_plan(root, require_identity=True):
    root = _validate_formal_root_path(root, create=False)
    pinned = _PinnedRoot(root)
    try:
        registry = _read_json_from_pinned(
            pinned, 'FORMAL_REGISTRY.json', canonical=True)
        census = _read_json_from_pinned(
            pinned, 'COMPATIBILITY_CENSUS.json', canonical=True)
        plan = _read_json_from_pinned(
            pinned, 'FORMAL_PLAN.json', canonical=True)
        manifest = _read_json_from_pinned(
            pinned, 'MISSING_JOBS.json', canonical=True)
        if experiment_profile() == FULL_MATRIX_PROFILE:
            from three_dataset_full_matrix_report import reuse_census
            reuse = _read_json_from_pinned(pinned, 'FULL_MATRIX_REUSE.json', canonical=True)
            if not _exact_equal(census, reuse_census(reuse)):
                raise ValueError('installed full matrix census differs from reuse bundle')
        elif experiment_profile() == CONTINUATION_PROFILE:
            from three_dataset_cifar_continuation_profile import reuse_census
            from three_dataset_cifar_continuation_reconcile import load_bundle
            reuse = load_bundle(root / 'reuse')
            if not _exact_equal(census, reuse_census(reuse)):
                raise ValueError('installed CIFAR continuation census differs')
        pinned.verify()
    finally:
        pinned.close()
    if not _exact_equal(registry, _registry_payload()):
        raise ValueError('installed registry provenance differs')
    _validate_plan_shape(plan, census)
    _validate_completed_plan_identity(plan)
    expected = _missing_jobs_payload(plan, census)
    if not _exact_equal(manifest, expected):
        raise ValueError('missing-jobs manifest differs from installed plan')
    if require_identity:
        pinned = _PinnedRoot(root)
        try:
            _read_root_identity_record(pinned, plan, manifest)
        finally:
            pinned.close()
    return plan, manifest


def _installed_jobs(root):
    plan, _ = _load_installed_plan(root)
    return list(plan['missing_jobs'])


def _mkdir_exclusive(parent_path, name):
    parent_path = _path(parent_path, 'directory parent')
    if type(name) is not str or not name or '/' in name or name in ('.', '..'):
        raise ValueError('directory name is invalid')
    parent = _PinnedRoot(parent_path)
    try:
        parent.verify()
        os.mkdir(name, 0o700, dir_fd=parent.fd)
        os.fsync(parent.fd)
        details = os.stat(name, dir_fd=parent.fd, follow_symlinks=False)
        if not stat.S_ISDIR(details.st_mode):
            raise RuntimeError('installed directory is invalid')
        parent.verify()
    finally:
        parent.close()


def _source_commit():
    completed = subprocess.run(
        ['git', '-C', str(Path(__file__).resolve().parent),
         'rev-parse', 'HEAD'],
        check=True, capture_output=True, text=True,
    )
    value = completed.stdout.strip()
    if _COMMIT.fullmatch(value) is None:
        raise RuntimeError('source commit is invalid')
    return value


def _source_branch():
    completed = subprocess.run(
        ['git', '-C', str(Path(__file__).resolve().parent),
         'branch', '--show-current'],
        check=True, capture_output=True, text=True,
    )
    value = completed.stdout.strip()
    if not value or '\x00' in value:
        raise RuntimeError('source branch is invalid')
    return value


def install_formal_root_identity(root, plan):
    root = _validate_formal_root_path(root, create=False)
    installed, manifest = _load_installed_plan(root, require_identity=False)
    if not _exact_equal(plan, installed):
        raise ValueError('formal root identity plan differs')
    pinned = _PinnedRoot(root)
    try:
        pinned.verify()
        details = pinned.details
        record = {
            'kind': 'formal_root_identity',
            **_profile_binding(),
            **_full_authority_binding(pinned),
            'token': secrets.token_hex(32),
            'root_dev': details.st_dev,
            'root_inode': details.st_ino,
            'root_ctime_ns': details.st_ctime_ns,
            'root_size': details.st_size,
            'registry_sha256': registry_sha256(),
            'plan_sha256': _digest(installed),
            'missing_jobs_sha256': _digest(manifest),
            'source_commit': _source_commit(),
        }
    finally:
        pinned.close()
    install_json_exclusive(root / 'FORMAL_ROOT_IDENTITY.json', record)
    return _root_identity(root)


def _full_authority_binding(pinned):
    if experiment_profile() == CONTINUATION_PROFILE:
        from three_dataset_cifar_continuation_reconcile import load_bundle
        reuse = load_bundle(pinned.path / 'reuse')
        return {
            'experiment_profile': CONTINUATION_PROFILE,
            'reuse_bundle_sha256': _sha256_bytes(
                _canonical_json(reuse) + b'\n'),
        }
    if experiment_profile() != FULL_MATRIX_PROFILE:
        return {}
    from three_dataset_full_matrix_report import validate_reuse
    reuse = _read_json_from_pinned(pinned, 'FULL_MATRIX_REUSE.json', canonical=True)
    validate_reuse(reuse)
    return {'experiment_profile': FULL_MATRIX_PROFILE,
            'reuse_bundle_sha256': _sha256_bytes(_canonical_json(reuse) + b'\n')}


def _process_start_time(pid):
    if type(pid) is not int or isinstance(pid, bool) or pid <= 0:
        raise ValueError('pid must be an exact positive integer')
    try:
        content = Path(f'/proc/{pid}/stat').read_text()
        value = content.rsplit(')', 1)[1].split()[19]
    except (FileNotFoundError, IndexError, OSError) as error:
        raise ValueError('process identity is unavailable') from error
    if not value:
        raise ValueError('process start time is empty')
    return value


def _root_identity(root):
    root = _validate_formal_root_path(root, create=False)
    plan, manifest = _load_installed_plan(root)
    pinned = _PinnedRoot(root)
    try:
        _, content, identity_details = _read_root_identity_record(
            pinned, plan, manifest)
        return {
            'dev': pinned.details.st_dev,
            'inode': pinned.details.st_ino,
            'ctime_ns': identity_details.st_ctime_ns,
            'size': identity_details.st_size,
            'hash': hashlib.sha256(content).hexdigest(),
        }
    finally:
        pinned.close()


def _validate_root_identity(value):
    if type(value) is not dict or set(value) != _ROOT_IDENTITY_KEYS:
        raise ValueError('root identity schema is invalid')
    for name in ('dev', 'inode', 'ctime_ns', 'size'):
        if type(value[name]) is not int or isinstance(value[name], bool) \
                or value[name] < 0:
            raise ValueError(f'root identity {name} is invalid')
    _hash_string(value['hash'], 'root identity hash')


def _validate_owner(owner, allow_empty_job=False, require_live=False):
    if type(owner) is not dict or set(owner) != _OWNER_KEYS:
        raise ValueError('claim owner schema is invalid')
    if owner['kind'] != 'formal_job_claim':
        raise ValueError('claim owner kind differs')
    if allow_empty_job and owner['job'] == '':
        pass
    else:
        _nonempty_string(owner['job'], 'claim job')
    for name in ('launcher_token', 'worker_role', 'process_start_time'):
        _nonempty_string(owner[name], name)
    if owner['phase'] not in {'formal', 'explanation'}:
        raise ValueError('claim phase is invalid')
    if type(owner['pid']) is not int or isinstance(owner['pid'], bool) \
            or owner['pid'] <= 0:
        raise ValueError('claim pid must be an exact positive integer')
    if type(owner['pgid']) is not int or isinstance(owner['pgid'], bool) \
            or owner['pgid'] <= 0:
        raise ValueError('claim pgid must be an exact positive integer')
    if type(owner['source_commit']) is not str \
            or _COMMIT.fullmatch(owner['source_commit']) is None:
        raise ValueError('claim source commit is invalid')
    _validate_root_identity(owner['root_identity'])
    _canonical_json(owner)
    if require_live:
        if (_process_start_time(owner['pid']) != owner['process_start_time']
                or os.getpgid(owner['pid']) != owner['pgid']):
            raise ValueError('claim process identity differs')


def _started_payload(owner, run_dir, plan_sha256, command_sha256,
                     disk_reservation_sha256=None):
    _validate_owner(owner)
    run_dir = _path(run_dir, 'started run directory')
    _hash_string(plan_sha256, 'started plan hash')
    _hash_string(command_sha256, 'started command hash')
    binding = {}
    if experiment_profile() == FULL_MATRIX_PROFILE:
        _hash_string(disk_reservation_sha256, 'started disk reservation hash')
        binding['disk_reservation_sha256'] = disk_reservation_sha256
    return {
        **binding,
        'kind': 'formal_job_started',
        'job': owner['job'],
        'owner_sha256': hashlib.sha256(
            _canonical_json(owner) + b'\n').hexdigest(),
        'plan_sha256': plan_sha256,
        'source_commit': owner['source_commit'],
        'command_sha256': command_sha256,
        'run_dir': str(run_dir),
        'root_identity': owner['root_identity'],
        'worker_role': owner['worker_role'],
        'phase': owner['phase'],
        'pid': owner['pid'],
        'pgid': owner['pgid'],
        'process_start_time': owner['process_start_time'],
    }


def _validate_started(started, owner):
    keys = _STARTED_KEYS | ({'disk_reservation_sha256'}
                           if experiment_profile() == FULL_MATRIX_PROFILE else set())
    if type(started) is not dict or set(started) != keys:
        raise ValueError('claim started schema is invalid')
    expected = _started_payload(
        owner, started['run_dir'], started['plan_sha256'],
        started['command_sha256'], started.get('disk_reservation_sha256'))
    if not _exact_equal(started, expected):
        raise ValueError('claim started identity differs')
    _canonical_json(started)


def spec_for_key(key, phase=None):
    spec = registry_spec_for_key(key)
    if phase is not None:
        if phase not in {'formal', 'explanation'}:
            raise ValueError('spec phase is invalid')
        expected = 'explanation' if spec.explanation else 'formal'
        if phase != expected:
            raise ValueError('spec key crosses the requested phase')
    return spec


def owner_for(root, phase, launcher_token, worker_role, pid, pgid):
    if phase not in {'formal', 'explanation'}:
        raise ValueError('owner phase is invalid')
    owner = {
        'kind': 'formal_job_claim',
        'job': '',
        'launcher_token': launcher_token,
        'worker_role': worker_role,
        'phase': phase,
        'pid': pid,
        'pgid': pgid,
        'process_start_time': _process_start_time(pid),
        'source_commit': _source_commit(),
        'root_identity': _root_identity(root),
    }
    _validate_owner(owner, allow_empty_job=True, require_live=True)
    return owner


def command_for_run(key, run_dir):
    spec = spec_for_key(key)
    run_dir = _path(run_dir, 'command run directory')
    if (run_dir.parent.name != 'runs'
            or run_dir.name != safe_spec_name(spec)):
        raise ValueError('command run directory is not the exact safe spec path')
    command = command_for(spec, 'cuda:0', str(run_dir.parent), smoke=False)
    if command[command.index('--exp_name') + 1] != run_dir.name:
        raise RuntimeError('command output directory differs from exact run')
    return command


def _claim_name(job):
    _nonempty_string(job, 'claim job')
    return safe_spec_key(job)


def _read_claim_bundle(claim_path, job, current_root_identity, current_commit):
    claim = _PinnedRoot(claim_path)
    try:
        names = set(os.listdir(claim.fd))
        if 'owner.json' not in names:
            raise ValueError('claim is incomplete')
        allowed = {'owner.json', 'started.json'}
        if experiment_profile() == FULL_MATRIX_PROFILE:
            allowed.add('disk-reservation.json')
        if not names.issubset(allowed):
            raise ValueError('claim contains unsafe evidence')
        owner = _read_json_from_pinned(claim, 'owner.json', canonical=True)
        _validate_owner(owner)
        if (owner['job'] != job
                or not _exact_equal(owner['root_identity'],
                                    current_root_identity)
                or owner['source_commit'] != current_commit):
            raise ValueError('claim provenance differs')
        receipt = None
        if 'disk-reservation.json' in names:
            from three_dataset_resource_gate import validate_receipt
            receipt = _read_json_from_pinned(claim, 'disk-reservation.json', canonical=True)
            validate_receipt(receipt, owner)
        started = None
        if 'started.json' in names:
            started = _read_json_from_pinned(
                claim, 'started.json', canonical=True)
            _validate_started(started, owner)
            if experiment_profile() == FULL_MATRIX_PROFILE and (
                    receipt is None or started['disk_reservation_sha256'] != _file_digest(receipt)):
                raise ValueError('started disk reservation identity differs')
        claim.verify()
        return owner, names, started
    finally:
        claim.close()


def _read_claim(claim_path, job, current_root_identity, current_commit):
    owner, names, _ = _read_claim_bundle(
        claim_path, job, current_root_identity, current_commit)
    return owner, names


def _read_started_claim(claim_path, job, current_root_identity,
                        current_commit):
    owner, names, started = _read_claim_bundle(
        claim_path, job, current_root_identity, current_commit)
    expected = {'owner.json', 'started.json'}
    if experiment_profile() == FULL_MATRIX_PROFILE:
        expected.add('disk-reservation.json')
    if names != expected:
        raise ValueError('claim has no durable started evidence')
    return owner, started


def installed_claim_owner(root, key):
    root = _validate_formal_root_path(root, create=False)
    spec_for_key(key)
    identity = _root_identity(root)
    commit = _source_commit()
    owner, _ = _read_claim(
        root / 'claims' / _claim_name(key), key, identity, commit)
    return owner


def claim_next(plan, claims_root, phase, owner, pipeline=False) -> str | None:
    claims_root = _path(claims_root, 'claims root')
    if claims_root.name != 'claims':
        raise ValueError('claims root must be the formal claims directory')
    installed, _ = _load_installed_plan(claims_root.parent)
    if not _exact_equal(plan, installed):
        raise ValueError('claim plan differs from installed immutable plan')
    if phase not in {'formal', 'explanation'}:
        raise ValueError('claim phase is invalid')
    _validate_owner(owner, allow_empty_job=True, require_live=True)
    if owner['phase'] != phase:
        raise ValueError('claim owner phase differs')
    current_identity = _root_identity(claims_root.parent)
    current_commit = _source_commit()
    if (not _exact_equal(owner['root_identity'], current_identity)
            or owner['source_commit'] != current_commit):
        raise ValueError('claim owner provenance differs')
    if phase == 'formal':
        phase_jobs = tuple(installed['missing_jobs'])
    else:
        if not phase_ready(claims_root.parent, 'explanation'):
            raise ValueError('explanation phase is not ready')
        phase_jobs = tuple(installed['explanation_cells'])
    if owner['job']:
        if owner['job'] not in phase_jobs:
            raise ValueError('claim job is outside the requested phase')
        jobs = (owner['job'],)
    else:
        jobs = phase_jobs
    claimed_names = tuple((job, _claim_name(job)) for job in jobs)
    claims = _PinnedRoot(claims_root)
    try:
        fcntl.flock(claims.fd, fcntl.LOCK_EX)
        if pipeline:
            _pipeline_running(claims_root.parent)
            _validate_pipeline_launcher(
                claims_root.parent, installed, owner)
            queue_path = _ensure_directory(claims_root.parent, 'audit_queue')
            queue = _PinnedRoot(queue_path)
            try:
                queued, active = _audit_state(claims_root.parent, installed, queue)
                if phase == 'explanation' and not _audit_drained(
                        claims_root.parent, installed, 'formal', queued, active):
                    raise ValueError('formal audit queue is not drained')
            finally:
                queue.close()
            inflight = 0
            available = []
            names = set(os.listdir(claims.fd))
            for job in phase_jobs:
                name = _claim_name(job)
                if name not in names:
                    if job in jobs:
                        available.append(job)
                    continue
                _read_claim(claims_root / name, job,
                            current_identity, current_commit)
                if (experiment_profile() in (
                        RECOVERY_PROFILE, SINGLE_DATASET_PROFILE,
                        CONTINUATION_PROFILE)
                        and job in queued
                        or _installed_completed_record(
                        claims_root.parent, spec_for_key(job, phase),
                        installed) is None):
                    inflight += 1
            if not available:
                return None
            if inflight >= pipeline_inflight_limit():
                raise PipelineBackpressure('formal pipeline capacity is full')
        for job, claimed_name in claimed_names:
            try:
                existing = os.stat(claimed_name, dir_fd=claims.fd,
                                   follow_symlinks=False)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                if not stat.S_ISDIR(existing.st_mode):
                    raise ValueError('claim path is unsafe')
                _read_claim(claims_root / claimed_name, job,
                            current_identity, current_commit)
                continue
            if experiment_profile() == FULL_MATRIX_PROFILE:
                from three_dataset_resource_gate import _disk_status_locked
                status = _disk_status_locked(claims_root.parent, 1, owner)
                if not status['safe']:
                    raise PipelineBackpressure('formal disk reservation capacity is full')
            claims.verify()
            os.mkdir(claimed_name, 0o700, dir_fd=claims.fd)
            os.fsync(claims.fd)
            installed_owner = dict(owner)
            installed_owner['job'] = job
            install_json_exclusive(
                claims_root / claimed_name / 'owner.json', installed_owner)
            if experiment_profile() == FULL_MATRIX_PROFILE:
                from three_dataset_resource_gate import reservation_receipt
                install_json_exclusive(
                    claims_root / claimed_name / 'disk-reservation.json',
                    reservation_receipt(installed_owner, installed))
            claimed = _PinnedRoot(claims_root / claimed_name)
            try:
                os.fchmod(claimed.fd, 0o700)
                os.fsync(claimed.fd)
                claimed.verify()
            finally:
                claimed.close()
            claims.verify()
            return job
        return None
    finally:
        try:
            fcntl.flock(claims.fd, fcntl.LOCK_UN)
        finally:
            claims.close()


def mark_claim_started(claim_path, owner, run_dir, plan,
                       command_sha256) -> dict:
    """Durably bind one installed owner to its exact launched run."""
    claim_path = _path(claim_path, 'claim path')
    if claim_path.parent.name != 'claims':
        raise ValueError('claim path is outside the formal claims directory')
    claims = _PinnedRoot(claim_path.parent)
    try:
        fcntl.flock(claims.fd, fcntl.LOCK_EX)
        started = _mark_claim_started_locked(
            claim_path, owner, run_dir, plan, command_sha256)
        claims.verify()
        return started
    finally:
        try:
            fcntl.flock(claims.fd, fcntl.LOCK_UN)
        finally:
            claims.close()


def _mark_claim_started_locked(claim_path, owner, run_dir, plan,
                               command_sha256) -> dict:
    """Publish started evidence while the caller holds the claims flock."""
    claim_path = _path(claim_path, 'claim path')
    claims_root = claim_path.parent
    if claims_root.name != 'claims':
        raise ValueError('claim path is outside the formal claims directory')
    _validate_owner(owner, require_live=True)
    if claim_path.name != _claim_name(owner['job']):
        raise ValueError('claim path does not encode the owner job')
    installed_plan, _ = _load_installed_plan(claims_root.parent)
    if not _exact_equal(plan, installed_plan):
        raise ValueError('started plan differs from installed immutable plan')
    current_identity = _root_identity(claims_root.parent)
    current_commit = _source_commit()
    if (not _exact_equal(owner['root_identity'], current_identity)
            or owner['source_commit'] != current_commit):
        raise ValueError('started owner provenance differs')
    installed_owner, names = _read_claim(
        claim_path, owner['job'], current_identity, current_commit)
    expected = {'owner.json'}
    disk_hash = None
    if experiment_profile() == FULL_MATRIX_PROFILE:
        from three_dataset_resource_gate import validate_receipt
        expected.add('disk-reservation.json')
        claim = _PinnedRoot(claim_path)
        try:
            receipt = _read_json_from_pinned(claim, 'disk-reservation.json', canonical=True)
            validate_receipt(receipt, owner, installed_plan)
            disk_hash = _file_digest(receipt)
        finally:
            claim.close()
    if names != expected or not _exact_equal(installed_owner, owner):
        raise ValueError('started owner differs from installed owner')
    run_dir = _path(run_dir, 'started run directory')
    if run_dir.parent.name != 'runs':
        raise ValueError('started run layout is invalid')
    formal = _PinnedRoot(claims_root.parent)
    run_formal = _PinnedRoot(run_dir.parent.parent)
    run = _PinnedRoot(run_dir)
    try:
        if (_directory_identity(formal.details)
                != _directory_identity(run_formal.details)):
            raise ValueError('started run uses a different formal root')
        formal.verify()
        run_formal.verify()
        run.verify()
    finally:
        run.close()
        run_formal.close()
        formal.close()
    started = _started_payload(
        owner, run_dir, _digest(installed_plan), command_sha256, disk_hash)
    install_json_exclusive(claim_path / 'started.json', started)
    verified_owner, verified_started = _read_started_claim(
        claim_path, owner['job'], current_identity, current_commit)
    if (not _exact_equal(verified_owner, owner)
            or not _exact_equal(verified_started, started)):
        raise RuntimeError('installed started claim verification failed')
    return started


def release_prelaunch_claim(claim_path, owner) -> None:
    claim_path = _path(claim_path, 'claim path')
    claims_root = claim_path.parent
    if claims_root.name != 'claims':
        raise ValueError('claim path is outside the formal claims directory')
    _validate_owner(owner)
    claimed_name = _claim_name(owner['job'])
    if claim_path.name != claimed_name:
        raise ValueError('claim path does not encode the owner job')
    current_identity = _root_identity(claims_root.parent)
    current_commit = _source_commit()
    if (not _exact_equal(owner['root_identity'], current_identity)
            or owner['source_commit'] != current_commit):
        raise ValueError('release owner provenance differs')
    claims = _PinnedRoot(claims_root)
    try:
        fcntl.flock(claims.fd, fcntl.LOCK_EX)
        installed, names = _read_claim(
            claim_path, owner['job'], current_identity, current_commit)
        expected = {'owner.json'}
        if experiment_profile() == FULL_MATRIX_PROFILE:
            expected.add('disk-reservation.json')
        if names != expected:
            raise ValueError('started claim cannot be released pre-launch')
        if not _exact_equal(installed, owner):
            raise ValueError('release owner differs from installed owner')
        if experiment_profile() == FULL_MATRIX_PROFILE:
            for physical_gpu in (0, 1):
                with _audit_gpu(claims_root.parent, physical_gpu) as gpu_owner:
                    if gpu_owner is not None and gpu_owner['job'] == owner['job']:
                        raise ValueError('GPU-owned disk reservation cannot be released')
        claim = _PinnedRoot(claim_path)
        try:
            named = os.stat(claimed_name, dir_fd=claims.fd,
                            follow_symlinks=False)
            if ((named.st_dev, named.st_ino)
                    != (claim.details.st_dev, claim.details.st_ino)):
                raise ValueError('claim directory identity changed')
            os.fchmod(claim.fd, 0o700)
            if 'disk-reservation.json' in names:
                os.unlink('disk-reservation.json', dir_fd=claim.fd)
            os.unlink('owner.json', dir_fd=claim.fd)
            os.fsync(claim.fd)
        finally:
            claim.close()
        os.rmdir(claimed_name, dir_fd=claims.fd)
        os.fsync(claims.fd)
        claims.verify()
    finally:
        try:
            fcntl.flock(claims.fd, fcntl.LOCK_UN)
        finally:
            claims.close()


def _ensure_directory(parent_path, name):
    parent_path = _path(parent_path, 'directory parent')
    try:
        _mkdir_exclusive(parent_path, name)
    except FileExistsError:
        parent = _PinnedRoot(parent_path)
        try:
            details = os.stat(name, dir_fd=parent.fd, follow_symlinks=False)
            if not stat.S_ISDIR(details.st_mode):
                raise ValueError('installed directory path is unsafe')
            child = _PinnedRoot(parent_path / name)
            try:
                child.verify()
            finally:
                child.close()
            parent.verify()
        finally:
            parent.close()
    return parent_path / name


def _source_hashes():
    source = _PinnedRoot(Path(__file__).resolve().parent)
    hashes = {}
    try:
        for logical in sorted(_SOURCE_INVENTORY):
            source.verify()
            descriptor = source.open_file(logical)
            try:
                content, details = _read_descriptor(descriptor)
                source.verify_file_name(logical, details)
            finally:
                os.close(descriptor)
            hashes[logical] = hashlib.sha256(content).hexdigest()
        source.verify()
    finally:
        source.close()
    return hashes


def prepare_run(root, key, owner):
    """Install one exact pre-launch run and all immutable control evidence."""
    root = _validate_formal_root_path(root, create=False)
    claims = _PinnedRoot(root / 'claims')
    try:
        fcntl.flock(claims.fd, fcntl.LOCK_EX)
        run_dir = _prepare_run_locked(root, key, owner)
        claims.verify()
        return run_dir
    finally:
        try:
            fcntl.flock(claims.fd, fcntl.LOCK_UN)
        finally:
            claims.close()


def _prepare_run_locked(root, key, owner):
    """Publish immutable run controls under the claims flock, before launch."""
    plan, _ = _load_installed_plan(root)
    _validate_owner(owner, require_live=True)
    spec = spec_for_key(key, owner['phase'])
    if owner['job'] != key:
        raise ValueError('prepare owner job differs from requested spec')
    installed = installed_claim_owner(root, key)
    if not _exact_equal(installed, owner):
        raise ValueError('prepare owner differs from installed claim')
    runs = _ensure_directory(root, 'runs')
    run_dir = runs / safe_spec_name(spec)
    _mkdir_exclusive(runs, run_dir.name)
    command = command_for_run(key, run_dir)
    command_sha256 = _digest(command)
    job = {
        'kind': 'formal_job_spec',
        'spec_key': key,
        'spec': asdict(spec),
        'registry_sha256': registry_sha256(),
        'metric_formula_version': FORMULA_VERSION,
        'plan_sha256': _digest(plan),
        'run_dir': str(run_dir),
        'source_commit': owner['source_commit'],
        'source_sha256': _source_hashes(),
        'command': list(command),
        'command_sha256': command_sha256,
        'root_identity': owner['root_identity'],
    }
    install_json_exclusive(run_dir / 'FORMAL_JOB_SPEC.json', job)
    install_json_exclusive(run_dir / 'CLAIM_OWNER.json', owner)
    _mark_claim_started_locked(
        root / 'claims' / _claim_name(key), owner, run_dir, plan,
        command_sha256)
    launch = {
        'kind': 'formal_launch_started',
        'spec_key': key,
        'plan_sha256': _digest(plan),
        'job_spec_sha256': hashlib.sha256(
            _canonical_json(job) + b'\n').hexdigest(),
        'claim_sha256': hashlib.sha256(
            _canonical_json(owner) + b'\n').hexdigest(),
        'command_sha256': command_sha256,
        'source_commit': owner['source_commit'],
        'root_identity': owner['root_identity'],
        'worker_role': owner['worker_role'],
        'phase': owner['phase'],
        'pid': owner['pid'],
        'pgid': owner['pgid'],
        'process_start_time': owner['process_start_time'],
    }
    install_json_exclusive(run_dir / 'LAUNCH_STARTED.json', launch)
    return run_dir


def resource_artifact_names(spec):
    protocol = protocol_for(spec)
    return (
        'config.json', 'results.json', 'data_flow_audit.jsonl',
        'validation/validation_manifest.json',
        'checkpoints/formal_final.pt', 'job.log',
        *_formal_artifact_logicals(protocol),
    )


def _read_run_control(run, name):
    return _read_json_from_pinned(run, name, canonical=True)


def _method_resource_state(spec, checkpoint_path):
    """Classify persisted method state; never instantiate models or use a GPU."""
    import numpy as np
    import torch
    from runner import _decode_checkpoint_value, _valid_method_checkpoint_state

    def require(condition, detail):
        if not condition:
            raise ValueError(f'resource method state is invalid: {detail}')

    def finite(value):
        if isinstance(value, torch.Tensor):
            require(value.layout == torch.strided
                    and torch.isfinite(value).all().item(), 'nonfinite tensor')
        elif isinstance(value, (np.ndarray, np.generic)):
            require(np.issubdtype(value.dtype, np.number)
                    and np.isfinite(value).all(), 'nonfinite array')
        elif isinstance(value, dict):
            for key, item in value.items():
                finite(key)
                finite(item)
        elif type(value) in (list, tuple):
            for item in value:
                finite(item)
        else:
            require(value is None or type(value) in (str, bool, int)
                    or (type(value) is float and math.isfinite(value)),
                    'unsupported or nonfinite value')

    def tensor(value, ndim=None):
        require(isinstance(value, torch.Tensor) and value.is_floating_point()
                and value.numel() > 0 and (ndim is None or value.ndim == ndim),
                'tensor shape or dtype')
        return value

    def classes(value):
        require(type(value) is dict and all(
            type(c) is int and 0 <= c < num_classes for c in value), 'class mapping')
        return value

    def parties(value):
        require(type(value) is list and len(value) == num_parties
                and all(type(item) is dict for item in value), 'party mappings')
        return value

    def raw_count(data, capacity):
        total, shape = 0, None
        for samples in classes(data).values():
            tensor(samples)
            require(samples.ndim >= 2 and samples.size(0) <= capacity, 'raw sample count')
            current_shape = (samples.shape[1:], samples.dtype)
            require(shape is None or shape == current_shape, 'raw sample shape')
            shape = current_shape
            total += samples.size(0)
        return total

    def weights(value, shapes):
        require(isinstance(value, dict) and set(value) == set(shapes), 'network keys')
        for name, shape in shapes.items():
            require(tensor(value[name]).shape == shape
                    and value[name].dtype == torch.float32, 'network shape or dtype')
        return sum(item.numel() for item in value.values())

    def classifier_weight():
        trainer = checkpoint.get('trainer_state')
        require(isinstance(trainer, dict) and isinstance(trainer.get('top_model'), dict),
                'checkpoint top model')
        value = tensor(trainer['top_model'].get('classifier.weight'), 2)
        finite(value)
        require(value.dtype == torch.float32 and value.size(0) == num_classes,
                'unsupported checkpoint classifier dtype or classes')
        return value

    protocol = protocol_for(spec)
    contract = protocol['method_contract']
    method = contract['cl_method']
    num_classes = protocol['base_options']['num_classes']
    num_parties = protocol['base_options']['num_parties']
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    require(type(checkpoint) is dict and type(checkpoint.get('schema_version')) is int
            and checkpoint['schema_version'] == 4, 'checkpoint schema version')
    recorded = checkpoint.get('protocol')
    require(type(recorded) is dict and recorded.get('cl_method') == method,
            'checkpoint protocol method')
    try:
        state = _decode_checkpoint_value(checkpoint['cl_state'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('resource method state cannot be decoded') from error
    require(type(state) is dict, 'method mapping')
    schemas = {
        'finetune': set(), 'lwf': set(), 'lwf_wa': set(),
        'er': {'per_class_size', 'data', 'seen_count'},
        'der_pp': {'buffer_size', 'num_seen', 'examples', 'labels', 'logits'},
        'er_ace': {'buffer_size', 'num_seen', 'examples', 'labels'},
        'proto_fedspace': {'protos', 'radius'},
        'adagauss': {'gaussians', 'adapters'},
        'target': {'task_classes', 'forgotten', 'fim_masks', 'generators'},
        'ewc': {'fisher', 'old_params'},
        'afc': {'importance', 'n_seen'},
        'gpm': {'feature_lists', 'head_basis', 'threshold'},
        'fedprotip_vfl': {'feature_lists', 'head_basis', 'threshold',
                         'tip_threshold', 'max_batches', 'task_means',
                         'task_bases', 'task_classes'},
    }
    if method == 'proto_evolve':
        from adaptive_head_consolidation import ADAPTIVE_METHOD_VERSION
        schema_state = state
        if contract['head_consolidation_mode'] != 'adaptive_dual_branch':
            require(state.get('adaptive_method_version') is None, 'nonadaptive version')
            schema_state = {**state, 'adaptive_method_version': 0}
        else:
            require(state.get('adaptive_method_version') == ADAPTIVE_METHOD_VERSION
                    and state.get('adaptive_top_version') in (0, ADAPTIVE_METHOD_VERSION),
                    'unsupported adaptive version')
        require(_valid_method_checkpoint_state(method, schema_state, True, num_parties),
                'prototype continuation schema')
    else:
        require(method in schemas and set(state) == schemas[method], 'method keys')
    finite(state)
    result = {
        'replay_type': 'none', 'raw_examples_per_class': 0,
        'persistent_embeddings': 0, 'added_parameters': 0,
        'privacy_label': 'no-persistent-raw-or-embedding-replay',
    }
    raw_total = 0
    if method in {'er_ace', 'der_pp'}:
        from cl_methods.er_ace import ReservoirBuffer
        option = 'der_buffer_size' if method == 'der_pp' else 'er_ace_buffer_size'
        buffer = ReservoirBuffer(contract[option] or 20 * num_classes, num_classes)
        buffer.load_state({key: state[key] for key in
                           ('buffer_size', 'num_seen', 'examples', 'labels')})
        raw_total = buffer.size()
        require(raw_total > 0, 'empty formal reservoir')
        if method == 'der_pp':
            require(tensor(state['logits'], 2).shape == (raw_total, num_classes),
                    'reservoir logits')
        result['replay_type'] = ('reservoir-raw-examples-and-logits'
                                 if method == 'der_pp' else 'reservoir-raw-examples')
    elif method == 'er':
        require(type(state['per_class_size']) is int
                and state['per_class_size'] == contract['er_per_class'], 'ER capacity')
        data, seen = classes(state['data']), classes(state['seen_count'])
        require(data and set(data) == set(seen), 'ER class membership')
        raw_total = raw_count(data, state['per_class_size'])
        for c, samples in data.items():
            require(type(seen[c]) is int and seen[c] > 0
                    and samples.size(0) == min(state['per_class_size'], seen[c]),
                    'ER sample count')
        result['replay_type'] = 'raw-examples'
    elif method in {'proto_fedspace', 'proto_evolve', 'adagauss'}:
        key = {'proto_fedspace': 'protos', 'proto_evolve': 'global_protos',
               'adagauss': 'gaussians'}[method]
        entries = classes(state[key])
        require(entries, 'empty formal prototype/statistics store')
        dim, dtype = None, None
        stores = [entries]
        if method == 'proto_fedspace':
            from cl_methods.proto_fedspace import ProtoFedSpaceCL
            ProtoFedSpaceCL._normalized_radius(state['radius'])
            top_weight = classifier_weight()
            dim, dtype = top_weight.size(1), top_weight.dtype
        elif method == 'proto_evolve':
            stores.append(classes(state['prev_protos']))
            raw = classes(state['head_raw_replay'])
            require(set(stores[-1]).issubset(entries) and set(raw).issubset(entries),
                    'prototype cache membership')
            require(not raw or contract['head_consolidation_enabled'],
                    'raw replay without head consolidation')
            raw_total = raw_count(raw, contract['head_consolidation_samples_per_class'])
        for store in stores:
            for entry in store.values():
                if method == 'proto_fedspace':
                    mean = tensor(entry, 1)
                else:
                    spread = 'cov' if method == 'adagauss' else 'std'
                    require(type(entry) is dict and set(entry) == {'mean', spread},
                            'prototype/statistics keys')
                    mean = tensor(entry['mean'], 1)
                    value = tensor(entry[spread], 2 if spread == 'cov' else 1)
                    require(value.shape == ((mean.numel(), mean.numel())
                                            if spread == 'cov' else mean.shape),
                            'prototype/statistics shape')
                    require(value.dtype == mean.dtype, 'prototype/statistics dtype')
                    require((value.diag() if spread == 'cov' else value).ge(0).all().item(),
                            'negative prototype spread')
                    if spread == 'cov':
                        require(torch.allclose(value, value.t()), 'asymmetric covariance')
                        # Match AdaGauss's degenerate-covariance Cholesky retry.
                        jittered = value + 1e-4 * torch.eye(
                            mean.numel(), dtype=value.dtype)
                        require(torch.linalg.cholesky_ex(jittered).info.item() == 0,
                                'covariance is not positive semidefinite within jitter')
                require(dim is None or dim == mean.numel(), 'embedding width')
                require(dtype is None or dtype == mean.dtype, 'embedding dtype')
                dim = mean.numel()
                dtype = mean.dtype
        result['persistent_embeddings'] = len(entries)
        result['replay_type'] = ('class-gaussian-statistics' if method == 'adagauss'
                                 else 'class-prototype-embeddings')
        result['privacy_label'] = ('persistent-derived-statistics-replay'
                                   if method == 'adagauss'
                                   else 'persistent-derived-embedding-replay')
        if method == 'adagauss':
            require(type(state['adapters']) is list, 'adapter list')
            for adapter in state['adapters']:
                result['added_parameters'] += weights(adapter, {
                    'net.0.weight': (2 * dim, dim), 'net.0.bias': (2 * dim,),
                    'net.2.weight': (dim, 2 * dim), 'net.2.bias': (dim,),
                })
    elif method == 'target':
        tasks, generators = state['task_classes'], state['generators']
        require(type(tasks) is dict and type(generators) is dict
                and generators and set(tasks) == set(generators), 'generator membership')
        occupied = set()
        for tid, labels in tasks.items():
            require(type(tid) is int and tid >= 0 and type(labels) is list and labels
                    and all(type(c) is int and 0 <= c < num_classes for c in labels)
                    and len(set(labels)) == len(labels) and not occupied.intersection(labels),
                    'generator task classes')
            occupied.update(labels)
        require(type(state['forgotten']) is list
                and all(type(c) is int for c in state['forgotten'])
                and len(set(state['forgotten'])) == len(state['forgotten'])
                and set(state['forgotten']).issubset(occupied), 'generator forgotten classes')
        for mask in parties(state['fim_masks']):
            require(all(type(name) is str and type(value) is bool
                        for name, value in mask.items()), 'FIM mask')
        dim = None
        for tid, record in generators.items():
            metadata = {'num_classes', 'embed_dim', 'noise_dim', 'hidden'}
            require(type(record) is dict and set(record) == metadata | {'state_dict'}
                    and all(type(record[k]) is int and record[k] > 0 for k in metadata)
                    and record['num_classes'] == len(tasks[tid]), 'generator metadata')
            c, d, z, h = (record[k] for k in ('num_classes', 'embed_dim', 'noise_dim', 'hidden'))
            require(dim is None or dim == d, 'generator embedding width')
            dim = d
            result['added_parameters'] += weights(record['state_dict'], {
                'class_emb.weight': (c, 64), 'net.0.weight': (h, 64 + z),
                'net.0.bias': (h,), 'net.2.weight': (h, h), 'net.2.bias': (h,),
                'net.4.weight': (d, h), 'net.4.bias': (d,),
            })
        result['replay_type'] = 'synthetic-generator'
        result['privacy_label'] = 'persistent-synthetic-generator-replay'
    elif method == 'ewc':
        for fisher, anchors in zip(parties(state['fisher']), parties(state['old_params'])):
            require(set(fisher) == set(anchors), 'Fisher/anchor membership')
            for name, value in fisher.items():
                require(type(name) is str and tensor(value).ge(0).all().item()
                        and tensor(anchors[name]).shape == value.shape, 'Fisher/anchor tensor')
        from cl_methods.ewc import _normalize_fisher
        try:
            _normalize_fisher(state['fisher'], 1)
        except (ValueError, FloatingPointError) as error:
            raise ValueError('resource EWC final Fisher is invalid') from error
    elif method in {'gpm', 'fedprotip_vfl'}:
        expected = contract['gpm_threshold'] if method == 'gpm' else contract['fedprotip_tip_threshold']
        require(type(state['threshold']) is float and state['threshold'] == expected,
                'projection threshold')
        for party in parties(state['feature_lists']):
            for name, value in party.items():
                require(type(name) is str and tensor(value, 2).size(1) <= value.size(0),
                        'projection basis')
        if state['head_basis'] is not None:
            basis = tensor(state['head_basis'], 2)
            require(basis.size(1) <= basis.size(0), 'head projection basis')
        if method == 'fedprotip_vfl':
            from cl_methods.fedprotip_vfl import FedProTIPVFLCL, FORMAL_EVALUATION_STATE_KEYS
            probe = SimpleNamespace(args=SimpleNamespace(num_parties=num_parties),
                                    tip_threshold=expected,
                                    max_batches=contract['fedprotip_max_batches'])
            FedProTIPVFLCL._validate_formal_evaluation_state(probe, {
                key: state[key] for key in FORMAL_EVALUATION_STATE_KEYS})
            for labels in state['task_classes'].values():
                classes(dict.fromkeys(labels))
            top_weight = classifier_weight()
            divisor = num_parties if protocol['base_options']['aggregation'] == 'concat' else 1
            party_width, remainder = divmod(top_weight.size(1), divisor)
            require(not remainder and party_width > 0, 'party embedding width')
            for means, bases in zip(state['task_means'], state['task_bases']):
                for tid, mean in means.items():
                    require(mean.numel() == party_width and bases[tid].size(1) <= party_width,
                            'task reference basis/model dimensions')
                    require(mean.dtype == bases[tid].dtype == top_weight.dtype,
                            'task reference basis/model dtype')
    elif method == 'afc':
        require(type(state['n_seen']) is int and 0 < state['n_seen'] <= num_classes,
                'AFC seen classes')
        top_weight = classifier_weight()
        importance = tensor(state['importance'], 1)
        require(importance.shape == (top_weight.size(1),)
                and importance.dtype == top_weight.dtype
                and importance.ge(0).all().item(), 'AFC importance/model compatibility')
    if raw_total:
        require(raw_total % num_classes == 0, 'fractional raw examples per class')
        result['raw_examples_per_class'] = raw_total // num_classes
        if method == 'proto_evolve':
            result['replay_type'] = 'raw-examples-and-class-prototype-embeddings'
            result['privacy_label'] = 'persistent-raw-and-derived-embedding-replay'
        else:
            result['privacy_label'] = 'persistent-raw-example-replay'
    return result


def _resource_measurements(spec, run_dir, runtime_seconds,
                           peak_gpu_memory_bytes):
    if type(runtime_seconds) is not float \
            or not math.isfinite(runtime_seconds) or runtime_seconds < 0:
        raise ValueError('measured runtime is invalid')
    if (type(peak_gpu_memory_bytes) is not int
            or isinstance(peak_gpu_memory_bytes, bool)
            or peak_gpu_memory_bytes < 0):
        raise ValueError('measured peak GPU memory is invalid')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('resource measurement requires one visible GPU')
    driver_lines = subprocess.run(
        ['nvidia-smi', '--query-gpu=driver_version',
         '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip().splitlines()
    driver_versions = {value.strip() for value in driver_lines if value.strip()}
    if len(driver_versions) != 1:
        raise RuntimeError('GPU driver identity is unavailable')
    driver_version = next(iter(driver_versions))
    results = _load_json_file(Path(run_dir) / 'results.json')
    comm = results.get('comm_stats')
    if type(comm) is not list:
        raise ValueError('resource communication evidence is invalid')
    transmitted = 0.0
    for row in comm:
        value = row.get('megabytes_transmitted') if type(row) is dict else None
        if type(value) not in (int, float) or isinstance(value, bool) \
                or not math.isfinite(value) or value < 0:
            raise ValueError('resource communication evidence is invalid')
        transmitted += float(value)
    communication_bytes = int(round(transmitted * 1024 * 1024))
    return {
        'hardware_identity': {
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_count': torch.cuda.device_count(),
            'cuda': str(torch.version.cuda),
            'torch': str(torch.__version__),
            'driver': driver_version,
        },
        'runtime_seconds': runtime_seconds,
        'peak_gpu_memory_bytes': peak_gpu_memory_bytes,
        'communication_bytes': communication_bytes,
        **_method_resource_state(spec, Path(run_dir) / 'checkpoints/formal_final.pt'),
    }


def resource_record(root, key, run_dir, measurements=None,
                    runtime_seconds=None, peak_gpu_memory_bytes=None):
    """Hash exact artifacts and install the canonical post-exit resource."""
    root = _validate_formal_root_path(root, create=False)
    plan, _ = _load_installed_plan(root)
    spec = spec_for_key(key)
    run_dir = _path(run_dir, 'resource run directory')
    if (run_dir.parent.parent != root
            or run_dir.parent.name != 'runs'
            or run_dir.name != safe_spec_name(spec)):
        raise ValueError('resource run directory differs from exact spec path')
    if measurements is None:
        measurements = _resource_measurements(
            spec, run_dir, runtime_seconds, peak_gpu_memory_bytes)
    if type(measurements) is not dict or set(measurements) != {
            'hardware_identity', 'runtime_seconds', 'peak_gpu_memory_bytes',
            'added_parameters', 'communication_bytes', 'replay_type',
            'raw_examples_per_class', 'persistent_embeddings',
            'privacy_label'}:
        raise ValueError('resource measurements schema is invalid')
    run = _PinnedRoot(run_dir)
    try:
        job = _read_run_control(run, 'FORMAL_JOB_SPEC.json')
        owner = _read_run_control(run, 'CLAIM_OWNER.json')
        launch = _read_run_control(run, 'LAUNCH_STARTED.json')
        if (job.get('spec_key') != key or owner.get('job') != key
                or launch.get('spec_key') != key
                or job.get('plan_sha256') != _digest(plan)):
            raise ValueError('resource control identity differs')
        artifacts = {}
        checkpoint_size = None
        for logical in resource_artifact_names(spec):
            run.verify()
            descriptor = run.open_file(logical)
            try:
                content, details = _read_descriptor(descriptor)
                run.verify_file_name(logical, details)
            finally:
                os.close(descriptor)
            if logical == 'job.log' and (
                    not content or stat.S_IMODE(details.st_mode) != 0o444):
                raise ValueError('resource job log is not immutable evidence')
            if logical == 'checkpoints/formal_final.pt':
                checkpoint_size = details.st_size
                # Bind semantics to the same pinned bytes as the artifact hash.
                method_state = _method_resource_state(spec, io.BytesIO(content))
            artifacts[logical] = hashlib.sha256(content).hexdigest()
        run.verify()
    finally:
        run.close()
    if not _exact_equal({name: measurements[name] for name in method_state}, method_state):
        raise ValueError('resource measurements differ from checkpoint method state')
    resource = {
        'hardware_identity': measurements['hardware_identity'],
        'instrumentation': 'formal-resource-v1',
        'runtime_seconds': measurements['runtime_seconds'],
        'peak_gpu_memory_bytes': measurements['peak_gpu_memory_bytes'],
        'checkpoint_size_bytes': checkpoint_size,
        'communication_bytes': measurements['communication_bytes'],
        **method_state,
    }
    _validate_record_resource(resource)
    evidence = {
        'kind': 'formal_resource_evidence',
        'spec_key': key,
        'plan_sha256': _digest(plan),
        'job_spec_sha256': hashlib.sha256(
            _canonical_json(job) + b'\n').hexdigest(),
        'claim_sha256': hashlib.sha256(
            _canonical_json(owner) + b'\n').hexdigest(),
        'launch_sha256': hashlib.sha256(
            _canonical_json(launch) + b'\n').hexdigest(),
        'command_sha256': job['command_sha256'],
        'artifact_sha256': artifacts,
        'resource': resource,
    }
    install_json_exclusive(run_dir / 'RESOURCE_EVIDENCE.json', evidence)
    return evidence


def install_completed_record(root, key, run_dir):
    root = _validate_formal_root_path(root, create=False)
    plan, _ = _load_installed_plan(root)
    spec = spec_for_key(key)
    record = completed_run_record(spec, run_dir, plan, formal_root=root)
    # Only final publication is serialized; scientific admission stays outside.
    claims = _PinnedRoot(root / 'claims')
    try:
        fcntl.flock(claims.fd, fcntl.LOCK_EX)
        _validate_completed_record(record, spec)
        records = _ensure_directory(root, 'records')
        install_json_exclusive(records / f'{safe_spec_name(spec)}.json', record)
        claims.verify()
        return record
    finally:
        try:
            fcntl.flock(claims.fd, fcntl.LOCK_UN)
        finally:
            claims.close()


def _installed_completed_record(root, spec, plan):
    records_path = Path(root) / 'records'
    try:
        details = records_path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(details.st_mode):
        raise ValueError('completed records directory is unsafe')
    records = _PinnedRoot(records_path)
    try:
        name = f'{safe_spec_name(spec)}.json'
        try:
            named = os.stat(name, dir_fd=records.fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(named.st_mode):
            raise ValueError('completed record path is unsafe')
        record = _read_json_from_pinned(records, name, canonical=True)
    finally:
        records.close()
    _validate_completed_record(record, spec)
    if (record['plan_sha256'] != _digest(plan)
            or record['source_commit'] != _source_commit()):
        raise ValueError('completed record authority differs')
    return record


def phase_ready(root, phase):
    if phase != 'explanation':
        raise ValueError('only the explanation barrier is registered')
    root = _validate_formal_root_path(root, create=False)
    plan, _ = _load_installed_plan(root)
    return all(
        _installed_completed_record(root, spec, plan) is not None
        for spec in formal_specs()
    )


def finalize_installed(root):
    root = _validate_formal_root_path(root, create=False)
    full = experiment_profile() == FULL_MATRIX_PROFILE
    dataset_only = experiment_profile() in (
        SINGLE_DATASET_PROFILE, METHOD_SHARD_PROFILE)
    continuation = experiment_profile() == CONTINUATION_PROFILE
    _require_pipeline_drained(root, ('formal',) if full or dataset_only or continuation
                              else ('formal', 'explanation'))
    plan, manifest = _load_installed_plan(root)
    if continuation:
        from three_dataset_cifar_continuation_reconcile import load_bundle
        from three_dataset_cifar_continuation_report import (
            combined_rows, render_continuation_tables,
        )
        pinned = _PinnedRoot(root)
        try:
            census = _read_json_from_pinned(
                pinned, 'COMPATIBILITY_CENSUS.json', canonical=True)
            identity, _, _ = _read_root_identity_record(pinned, plan, manifest)
            pinned.verify()
        finally:
            pinned.close()
        bundle = load_bundle(root / 'reuse')
        bundle_hash = _sha256_bytes(_canonical_json(bundle) + b'\n')
        if identity['reuse_bundle_sha256'] != bundle_hash:
            raise ValueError('continuation reuse authority snapshot differs')
        rows = combined_rows(root, plan, census, bundle)
        tables = render_continuation_tables(rows)
        if tuple(tables) != _CONTINUATION_TABLE_NAMES:
            raise ValueError('continuation table membership differs')
        audit = json.loads(tables['CIFAR_CONTINUATION_AUDIT.json'])
        audit['input_sha256'] = {
            'plan': _digest(plan), 'census': _digest(census),
            'reuse_bundle': bundle_hash,
        }
        audit['producer_commits'] = {
            'reused': sorted({row['origin_source_commit'] for row in rows
                              if row.get('origin') == 'reused'}),
            'new': sorted({row['origin_source_commit'] for row in rows
                           if row.get('origin') == 'new'}),
        }
        tables['CIFAR_CONTINUATION_AUDIT.json'] = _canonical_json(audit) + b'\n'
        destination = _validate_formal_root_path(root / 'tables', create=True)
        parent = _PinnedRoot(destination)
        installed = {}
        try:
            if any(os.path.lexists(destination / name) for name in tables):
                raise FileExistsError('continuation table already exists')
            for name, payload in tables.items():
                installed[name] = _install_table_exclusive(parent, name, payload)
            for name, payload in tables.items():
                descriptor = parent.open_file(name)
                try:
                    content, details = _read_descriptor(descriptor)
                    parent.verify_file_name(name, details)
                finally:
                    os.close(descriptor)
                if content != payload:
                    raise ValueError('installed continuation table bytes differ')
            parent.verify()
            return {name: _sha256_bytes(data) for name, data in tables.items()}
        except Exception:
            for name, details in reversed(list(installed.items())):
                if _owned_name(parent.fd, name, details):
                    os.unlink(name, dir_fd=parent.fd)
            os.fsync(parent.fd)
            raise
        finally:
            parent.close()
    if full:
        from three_dataset_full_matrix_report import combined_rows, render_full_matrix_tables
        pinned = _PinnedRoot(root)
        try:
            reuse = _read_json_from_pinned(pinned, 'FULL_MATRIX_REUSE.json', canonical=True)
            census = _read_json_from_pinned(pinned, 'COMPATIBILITY_CENSUS.json', canonical=True)
            identity, _, _ = _read_root_identity_record(pinned, plan, manifest)
            if identity['reuse_bundle_sha256'] != _sha256_bytes(_canonical_json(reuse) + b'\n'):
                raise ValueError('full matrix reuse authority snapshot differs')
        finally:
            pinned.close()
        tables = render_full_matrix_tables(combined_rows(root, plan, census, reuse))
        audit = json.loads(tables['FULL_MATRIX_AUDIT.json'])
        audit['input_sha256'].update(plan=_digest(plan), census=_digest(census),
                                     reuse_bundle=_sha256_bytes(_canonical_json(reuse) + b'\n'))
        tables['FULL_MATRIX_AUDIT.json'] = _canonical_json(audit) + b'\n'
        destination = _validate_formal_root_path(root / 'tables', create=True)
        parent = _PinnedRoot(destination)
        installed = {}
        try:
            if any(os.path.lexists(destination / name) for name in tables):
                raise FileExistsError('full matrix table already exists')
            for name, payload in tables.items():
                installed[name] = _install_table_exclusive(parent, name, payload)
            for name, payload in tables.items():
                descriptor = parent.open_file(name)
                try:
                    content, details = _read_descriptor(descriptor)
                    parent.verify_file_name(name, details)
                finally:
                    os.close(descriptor)
                if content != payload:
                    raise ValueError('installed full matrix table bytes differ')
            parent.verify()
            return {name: _sha256_bytes(data) for name, data in tables.items()}
        except Exception:
            for name, details in reversed(list(installed.items())):
                if _owned_name(parent.fd, name, details):
                    os.unlink(name, dir_fd=parent.fd)
            os.fsync(parent.fd)
            raise
        finally:
            parent.close()
    records = []
    for spec in (*formal_specs(), *explanation_specs()):
        record = _installed_completed_record(root, spec, plan)
        if record is None:
            raise ValueError('completed record set is incomplete')
        records.append(record)
    return install_tables(records, root / 'tables')


_MARKERS = frozenset({
    'FAILED_JOB', 'FORMAL_STOPPED', 'FORMAL_PHASE_SUCCESS',
    'EXPLANATION_PHASE_SUCCESS', 'FORMAL_EXECUTION_SUCCESS',
    'PILOT_PHASE_SUCCESS', 'PILOT_EXECUTION_SUCCESS',
    'RECOVERY_PHASE_SUCCESS', 'RECOVERY_EXECUTION_SUCCESS',
    'FULL_MATRIX_PHASE_SUCCESS', 'FULL_MATRIX_EXECUTION_SUCCESS',
    'DATASET_PHASE_SUCCESS', 'DATASET_EXECUTION_SUCCESS',
    'DATASET_CONTINUATION_PHASE_SUCCESS', 'DATASET_CONTINUATION_SUCCESS',
    'METHOD_SHARD_PHASE_SUCCESS', 'METHOD_SHARD_SUCCESS',
})
_MARKER_KINDS = {
    'FAILED_JOB': {
        'failed_job', 'failed_resource', 'failed_audit', 'failed_monitor',
        'failed_setup', 'failed_retention'},
    'FORMAL_STOPPED': {'formal_stopped'},
    'FORMAL_PHASE_SUCCESS': {'formal_phase_success'},
    'EXPLANATION_PHASE_SUCCESS': {'explanation_phase_success'},
    'FORMAL_EXECUTION_SUCCESS': {'formal_execution_success'},
    'PILOT_PHASE_SUCCESS': {'pilot_phase_success'},
    'PILOT_EXECUTION_SUCCESS': {'pilot_execution_success'},
    'RECOVERY_PHASE_SUCCESS': {'recovery_phase_success'},
    'RECOVERY_EXECUTION_SUCCESS': {'recovery_execution_success'},
    'FULL_MATRIX_PHASE_SUCCESS': {'full_matrix_phase_success'},
    'FULL_MATRIX_EXECUTION_SUCCESS': {'full_matrix_execution_success'},
    'DATASET_PHASE_SUCCESS': {'dataset_phase_success'},
    'DATASET_EXECUTION_SUCCESS': {'dataset_execution_success'},
    'DATASET_CONTINUATION_PHASE_SUCCESS': {
        'dataset_continuation_phase_success'},
    'DATASET_CONTINUATION_SUCCESS': {'dataset_continuation_success'},
    'METHOD_SHARD_PHASE_SUCCESS': {'method_shard_phase_success'},
    'METHOD_SHARD_SUCCESS': {'method_shard_success'},
}

_SUCCESS_MARKER_PROFILES = {
    'FORMAL_PHASE_SUCCESS': FORMAL_PROFILE,
    'EXPLANATION_PHASE_SUCCESS': FORMAL_PROFILE,
    'FORMAL_EXECUTION_SUCCESS': FORMAL_PROFILE,
    'PILOT_PHASE_SUCCESS': PILOT_PROFILE,
    'PILOT_EXECUTION_SUCCESS': PILOT_PROFILE,
    'RECOVERY_PHASE_SUCCESS': RECOVERY_PROFILE,
    'RECOVERY_EXECUTION_SUCCESS': RECOVERY_PROFILE,
    'FULL_MATRIX_PHASE_SUCCESS': FULL_MATRIX_PROFILE,
    'FULL_MATRIX_EXECUTION_SUCCESS': FULL_MATRIX_PROFILE,
    'DATASET_PHASE_SUCCESS': SINGLE_DATASET_PROFILE,
    'DATASET_EXECUTION_SUCCESS': SINGLE_DATASET_PROFILE,
    'DATASET_CONTINUATION_PHASE_SUCCESS': CONTINUATION_PROFILE,
    'DATASET_CONTINUATION_SUCCESS': CONTINUATION_PROFILE,
    'METHOD_SHARD_PHASE_SUCCESS': METHOD_SHARD_PROFILE,
    'METHOD_SHARD_SUCCESS': METHOD_SHARD_PROFILE,
}


def _validate_installed_success_marker_profiles(root):
    root = _validate_formal_root_path(root, create=False)
    pinned = _PinnedRoot(root)
    try:
        names = set(os.listdir(pinned.fd))
        foreign = sorted(
            name for name, profile in _SUCCESS_MARKER_PROFILES.items()
            if name in names and profile != experiment_profile())
        pinned.verify()
    finally:
        pinned.close()
    if foreign:
        raise ValueError(
            f'installed success marker profile differs: {foreign[0]}')


def install_marker(root, name, payload):
    root = _validate_formal_root_path(root, create=False)
    if name not in _MARKERS:
        raise ValueError('formal marker name is not registered')
    if type(payload) is not dict or set(payload) != {
            'kind', 'role', 'spec_key', 'exit_code'}:
        raise ValueError('formal marker schema is invalid')
    if payload['kind'] not in _MARKER_KINDS[name]:
        raise ValueError('formal marker kind differs')
    _nonempty_string(payload['role'], 'formal marker role')
    if type(payload['spec_key']) is not str or '\x00' in payload['spec_key']:
        raise ValueError('formal marker spec key is invalid')
    if (type(payload['exit_code']) is not int
            or isinstance(payload['exit_code'], bool)
            or payload['exit_code'] < 0):
        raise ValueError('formal marker exit code is invalid')
    success = name.endswith('_SUCCESS')
    if ((success and payload['exit_code'] != 0)
            or (not success and payload['exit_code'] == 0)):
        raise ValueError('formal marker exit status differs')
    if name == 'FAILED_JOB':
        if not (experiment_profile() in (
                FULL_MATRIX_PROFILE, SINGLE_DATASET_PROFILE,
                CONTINUATION_PROFILE, METHOD_SHARD_PROFILE)
                and payload['kind'] == 'failed_setup' and payload['spec_key'] == ''):
            spec_for_key(payload['spec_key'])
    elif payload['spec_key']:
        raise ValueError('formal phase marker must not name a spec')
    _validate_installed_success_marker_profiles(root)
    if success:
        if _SUCCESS_MARKER_PROFILES.get(name) != experiment_profile():
            raise ValueError('success marker profile differs')
        phases = (('formal',) if name in {
            'FORMAL_PHASE_SUCCESS', 'PILOT_PHASE_SUCCESS',
            'PILOT_EXECUTION_SUCCESS', 'RECOVERY_PHASE_SUCCESS',
            'RECOVERY_EXECUTION_SUCCESS', 'FULL_MATRIX_PHASE_SUCCESS',
            'FULL_MATRIX_EXECUTION_SUCCESS', 'DATASET_PHASE_SUCCESS',
            'DATASET_EXECUTION_SUCCESS',
            'DATASET_CONTINUATION_PHASE_SUCCESS',
            'DATASET_CONTINUATION_SUCCESS',
            'METHOD_SHARD_PHASE_SUCCESS', 'METHOD_SHARD_SUCCESS'}
                  else ('formal', 'explanation'))
        _require_pipeline_drained(root, phases)
    if not success and os.path.lexists(root / 'claims'):
        # A valid dispatch root already has a stable claims directory, even
        # before its first pipeline reservation creates audit_queue.
        claims = _PinnedRoot(root / 'claims')
        try:
            fcntl.flock(claims.fd, fcntl.LOCK_EX)
            claims.verify()
            install_json_exclusive(root / name, payload)
            claims.verify()
        finally:
            try:
                fcntl.flock(claims.fd, fcntl.LOCK_UN)
            finally:
                claims.close()
    else:
        # Marker-only legacy roots cannot dispatch; preserve their layout.
        install_json_exclusive(root / name, payload)
    return payload


def _gpu_owner(owner, physical_gpu, require_live=True):
    _validate_owner(owner, require_live=require_live)
    if (type(physical_gpu) is not int or isinstance(physical_gpu, bool)
            or physical_gpu not in (0, 1)):
        raise ValueError('physical GPU is outside the reviewed pair')
    value = dict(owner)
    value['kind'] = 'formal_gpu_claim'
    value['physical_gpu'] = physical_gpu
    if set(value) != _GPU_OWNER_KEYS:
        raise AssertionError('GPU claim schema differs')
    _canonical_json(value)
    return value


def claim_gpu(root, physical_gpu, owner):
    if experiment_profile() != FULL_MATRIX_PROFILE:
        return _claim_gpu(root, physical_gpu, owner)
    root = _validate_formal_root_path(root, create=False)
    claims = _PinnedRoot(root / 'claims')
    try:
        # Serialize reservation validation with exact-owner prelaunch cleanup.
        fcntl.flock(claims.fd, fcntl.LOCK_EX)
        return _claim_gpu(root, physical_gpu, owner)
    finally:
        try:
            fcntl.flock(claims.fd, fcntl.LOCK_UN)
        finally:
            claims.close()


def _claim_gpu(root, physical_gpu, owner):
    root = _validate_formal_root_path(root, create=False)
    expected = _gpu_owner(owner, physical_gpu)
    if (owner['root_identity'] != _root_identity(root)
            or owner['source_commit'] != _source_commit()):
        raise ValueError('GPU owner authority differs')
    if experiment_profile() == FULL_MATRIX_PROFILE:
        plan, _ = _load_installed_plan(root)
        from three_dataset_resource_gate import validate_receipt
        claim_path = root / 'claims' / _claim_name(owner['job'])
        installed, _ = _read_claim(claim_path, owner['job'], owner['root_identity'], owner['source_commit'])
        if not _exact_equal(installed, owner):
            raise ValueError('GPU disk reservation owner differs')
        claim = _PinnedRoot(claim_path)
        try:
            validate_receipt(_read_json_from_pinned(
                claim, 'disk-reservation.json', canonical=True), owner, plan)
        finally:
            claim.close()
    claims_path = _ensure_directory(root, 'gpu_claims')
    claims = _PinnedRoot(claims_path)
    name = f'gpu-{physical_gpu}'
    try:
        fcntl.flock(claims.fd, fcntl.LOCK_EX)
        try:
            os.stat(name, dir_fd=claims.fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            return False
        claims.verify()
        os.mkdir(name, 0o700, dir_fd=claims.fd)
        os.fsync(claims.fd)
        install_json_exclusive(claims_path / name / 'owner.json', expected)
        claims.verify()
        return True
    finally:
        try:
            fcntl.flock(claims.fd, fcntl.LOCK_UN)
        finally:
            claims.close()


def release_gpu(root, physical_gpu, owner):
    root = _validate_formal_root_path(root, create=False)
    expected = _gpu_owner(owner, physical_gpu, require_live=False)
    if (not _exact_equal(owner['root_identity'], _root_identity(root))
            or owner['source_commit'] != _source_commit()):
        raise ValueError('GPU release owner authority differs')
    claims_path = root / 'gpu_claims'
    claims = _PinnedRoot(claims_path)
    name = f'gpu-{physical_gpu}'
    try:
        fcntl.flock(claims.fd, fcntl.LOCK_EX)
        claim = _PinnedRoot(claims_path / name)
        try:
            installed = _read_json_from_pinned(
                claim, 'owner.json', canonical=True)
            if not _exact_equal(installed, expected):
                raise ValueError('GPU release owner differs')
            named = os.stat(name, dir_fd=claims.fd, follow_symlinks=False)
            if ((named.st_dev, named.st_ino)
                    != (claim.details.st_dev, claim.details.st_ino)):
                raise ValueError('GPU claim directory identity changed')
            try:
                _validate_owner(owner, require_live=True)
            except (ValueError, ProcessLookupError):
                try:
                    current_start = _process_start_time(owner['pid'])
                except ValueError as error:
                    if not isinstance(error.__cause__, FileNotFoundError):
                        raise
                    current_start = None
                # Never acquire claims while holding gpu_claims. This marker
                # reader validates immutable terminal evidence without a lock.
                if (current_start == owner['process_start_time']
                        or not _pipeline_terminal(root)):
                    raise ValueError('dead GPU cleanup requires valid terminal authority')
            os.fchmod(claim.fd, 0o700)
            os.unlink('owner.json', dir_fd=claim.fd)
            os.fsync(claim.fd)
        finally:
            claim.close()
        os.rmdir(name, dir_fd=claims.fd)
        os.fsync(claims.fd)
        claims.verify()
    finally:
        try:
            fcntl.flock(claims.fd, fcntl.LOCK_UN)
        finally:
            claims.close()


def _pipeline_terminal(root):
    """Validate terminal markers; cleanup may proceed only with real evidence."""
    pinned = _PinnedRoot(root)
    found = False
    try:
        names = set(os.listdir(pinned.fd))
        for name in ('FAILED_JOB', 'FORMAL_STOPPED'):
            if name not in names:
                continue
            value = _read_json_from_pinned(pinned, name, canonical=True)
            if (type(value) is not dict or set(value) != {
                    'kind', 'role', 'spec_key', 'exit_code'}
                    or value['kind'] not in _MARKER_KINDS[name]
                    or type(value['exit_code']) is not int
                    or value['exit_code'] <= 0):
                raise ValueError('pipeline terminal marker is invalid')
            _nonempty_string(value['role'], 'terminal marker role')
            if name == 'FAILED_JOB':
                if not (experiment_profile() in (
                        FULL_MATRIX_PROFILE, SINGLE_DATASET_PROFILE,
                        CONTINUATION_PROFILE, METHOD_SHARD_PROFILE)
                        and value['kind'] == 'failed_setup' and value['spec_key'] == ''):
                    spec_for_key(value['spec_key'])
            elif value['spec_key'] != '':
                raise ValueError('stopped marker must not name a spec')
            found = True
        pinned.verify()
    finally:
        pinned.close()
    return found


def _pipeline_running(root):
    _validate_installed_success_marker_profiles(root)
    if _pipeline_terminal(root):
        raise ValueError('formal pipeline is terminal')


def _validate_pipeline_launcher(root, plan, owner):
    """Under the caller's claims flock, reject claims from any earlier launch."""
    _validate_owner(owner, allow_empty_job=True)
    identity = _root_identity(root)
    commit = _source_commit()
    registered = {_claim_name(key): key for key in (
        *plan['formal_cells'], *plan['explanation_cells'])}
    claims = _PinnedRoot(root / 'claims')
    try:
        names = set(os.listdir(claims.fd))
        if not names.issubset(registered):
            raise ValueError('pipeline claims contain an unknown entry')
        for name in sorted(names):
            installed, _ = _read_claim(
                claims.path / name, registered[name], identity, commit)
            spec_for_key(installed['job'], installed['phase'])
            if installed['launcher_token'] != owner['launcher_token']:
                raise ValueError('pipeline launcher differs from installed claim')
        claims.verify()
    finally:
        claims.close()


@contextlib.contextmanager
def _audit_control(root, owner=None):
    root = _validate_formal_root_path(root, create=False)
    plan, _ = _load_installed_plan(root)
    claims = _PinnedRoot(root / 'claims')
    try:
        # One lock order: claims -> gpu_claims. Queue operations never re-lock.
        fcntl.flock(claims.fd, fcntl.LOCK_EX)
        if owner is not None:
            _validate_pipeline_launcher(root, plan, owner)
        queue = _PinnedRoot(_ensure_directory(root, 'audit_queue'))
        try:
            yield root, plan, queue
            queue.verify()
            claims.verify()
        finally:
            queue.close()
    finally:
        try:
            fcntl.flock(claims.fd, fcntl.LOCK_UN)
        finally:
            claims.close()


def _file_digest(value):
    return hashlib.sha256(_canonical_json(value) + b'\n').hexdigest()


def _audit_handoff(root, plan, key, physical_gpu):
    spec = spec_for_key(key)
    phase = 'explanation' if spec.explanation else 'formal'
    jobs = plan['explanation_cells'] if spec.explanation else plan['missing_jobs']
    if key not in jobs:
        raise ValueError('audit spec is not a planned job')
    if type(physical_gpu) is not int or physical_gpu not in (0, 1):
        raise ValueError('audit physical GPU is outside the reviewed pair')
    identity = _root_identity(root)
    commit = _source_commit()
    producer, started = _read_started_claim(
        root / 'claims' / _claim_name(key), key, identity, commit)
    run_dir = root / 'runs' / safe_spec_name(spec)
    command = command_for_run(key, run_dir)
    if (producer['phase'] != phase or not _exact_equal(
            started, _started_payload(
                producer, run_dir, _digest(plan), _digest(command),
                started.get('disk_reservation_sha256')))):
        raise ValueError('audit started claim differs')
    run = _PinnedRoot(run_dir)
    try:
        job = _read_run_control(run, 'FORMAL_JOB_SPEC.json')
        owner = _read_run_control(run, 'CLAIM_OWNER.json')
        launch = _read_run_control(run, 'LAUNCH_STARTED.json')
        resource = _read_run_control(run, 'RESOURCE_EVIDENCE.json')
        expected_job = {
            'kind': 'formal_job_spec', 'spec_key': key, 'spec': asdict(spec),
            'registry_sha256': registry_sha256(),
            'metric_formula_version': FORMULA_VERSION,
            'plan_sha256': _digest(plan), 'run_dir': str(run_dir),
            'source_commit': commit, 'source_sha256': _source_hashes(),
            'command': list(command), 'command_sha256': _digest(command),
            'root_identity': identity,
        }
        expected_launch = {
            'kind': 'formal_launch_started', 'spec_key': key,
            'plan_sha256': _digest(plan),
            'job_spec_sha256': _file_digest(job),
            'claim_sha256': _file_digest(producer),
            'command_sha256': _digest(command), 'source_commit': commit,
            'root_identity': identity,
            **{name: producer[name] for name in (
                'worker_role', 'phase', 'pid', 'pgid', 'process_start_time')},
        }
        if (not _exact_equal(job, expected_job)
                or not _exact_equal(owner, producer)
                or not _exact_equal(launch, expected_launch)):
            raise ValueError('audit producer control evidence differs')
        if type(resource) is not dict or set(resource) != {
                'kind', 'spec_key', 'plan_sha256', 'job_spec_sha256',
                'claim_sha256', 'launch_sha256', 'command_sha256',
                'artifact_sha256', 'resource'}:
            raise ValueError('audit resource evidence schema differs')
        expected_resource = {
            'kind': 'formal_resource_evidence', 'spec_key': key,
            'plan_sha256': _digest(plan),
            'job_spec_sha256': _file_digest(job),
            'claim_sha256': _file_digest(producer),
            'launch_sha256': _file_digest(launch),
            'command_sha256': _digest(command),
            'artifact_sha256': resource['artifact_sha256'],
            'resource': resource['resource'],
        }
        if not _exact_equal(resource, expected_resource):
            raise ValueError('audit resource authority differs')
        _validate_record_resource(resource['resource'])
        artifacts = resource['artifact_sha256']
        if (type(artifacts) is not dict
                or set(artifacts) != set(resource_artifact_names(spec))):
            raise ValueError('audit resource artifact map differs')
        for digest in artifacts.values():
            _hash_string(digest, 'audit resource artifact digest')
        run.verify()
    finally:
        run.close()
    return {
        'kind': 'formal_audit_handoff', 'spec_key': key, 'spec': asdict(spec),
        'phase': phase, 'root_identity': identity, 'plan_sha256': _digest(plan),
        'source_commit': commit, 'owner': producer,
        'started_sha256': _file_digest(started),
        'gpu_claim_sha256': _file_digest({
            **producer, 'kind': 'formal_gpu_claim', 'physical_gpu': physical_gpu}),
        'resource_sha256': _file_digest(resource),
        'physical_gpu': physical_gpu, 'seed': spec.seed, 'run_dir': str(run_dir),
    }


def _audit_owner(root, owner, phase, require_live=True):
    _validate_owner(owner, allow_empty_job=True, require_live=require_live)
    if (owner['job'] != '' or owner['phase'] != phase
            or not _exact_equal(owner['root_identity'], _root_identity(root))
            or owner['source_commit'] != _source_commit()):
        raise ValueError('audit owner template authority differs')


def _audit_state(root, plan, queue, require_live=True):
    """Read every entry, including other phases; corruption never means empty."""
    registered = {f'{_claim_name(key)}.json': key for key in (
        *plan['missing_jobs'], *plan['explanation_cells'])}
    names = set(os.listdir(queue.fd))
    if not names.issubset(set(registered) | {'active.json'}):
        raise ValueError('audit queue contains an unknown entry')
    queued = {}
    for name in sorted(names - {'active.json'}):
        handoff = _read_json_from_pinned(queue, name, canonical=True)
        if type(handoff) is not dict:
            raise ValueError('audit handoff schema differs')
        expected = _audit_handoff(
            root, plan, registered[name], handoff.get('physical_gpu'))
        if not _exact_equal(handoff, expected):
            raise ValueError('audit handoff identity differs')
        queued[registered[name]] = handoff
    active = None
    if 'active.json' in names:
        active = _read_json_from_pinned(queue, 'active.json', canonical=True)
        if (type(active) is not dict or set(active) != {
                'kind', 'spec_key', 'handoff_sha256', 'owner'}
                or active['kind'] != 'formal_active_audit'
                or type(active['spec_key']) is not str
                or active['spec_key'] not in queued):
            raise ValueError('active audit schema differs')
        handoff = queued[active['spec_key']]
        _audit_owner(root, active['owner'], handoff['phase'], require_live)
        if (active['handoff_sha256'] != _file_digest(handoff)
                or active['owner']['launcher_token']
                != handoff['owner']['launcher_token']
                or _exact_equal(active['owner'],
                                {**handoff['owner'], 'job': ''})):
            raise ValueError('active audit identity differs')
    queue.verify()
    return queued, active


@contextlib.contextmanager
def _audit_gpu(root, physical_gpu):
    claims = _PinnedRoot(_ensure_directory(root, 'gpu_claims'))
    try:
        fcntl.flock(claims.fd, fcntl.LOCK_EX)
        name = f'gpu-{physical_gpu}'
        installed = None
        if name in os.listdir(claims.fd):
            claim = _PinnedRoot(claims.path / name)
            try:
                if set(os.listdir(claim.fd)) != {'owner.json'}:
                    raise ValueError('GPU claim contains unsafe evidence')
                installed = _read_json_from_pinned(
                    claim, 'owner.json', canonical=True)
                if (type(installed) is not dict
                        or set(installed) != _GPU_OWNER_KEYS
                        or installed['kind'] != 'formal_gpu_claim'
                        or type(installed['physical_gpu']) is not int
                        or installed['physical_gpu'] != physical_gpu):
                    raise ValueError('GPU claim schema differs')
                producer = {k: v for k, v in installed.items()
                            if k != 'physical_gpu'}
                producer['kind'] = 'formal_job_claim'
                _validate_owner(producer)
                spec_for_key(producer['job'], producer['phase'])
                if (not _exact_equal(producer['root_identity'],
                                     _root_identity(root))
                        or producer['source_commit'] != _source_commit()):
                    raise ValueError('GPU claim authority differs')
                claim.verify()
            finally:
                claim.close()
        claims.verify()
        yield installed
        claims.verify()
    finally:
        try:
            fcntl.flock(claims.fd, fcntl.LOCK_UN)
        finally:
            claims.close()


def queue_audit(root, key, physical_gpu, owner):
    with _audit_control(root, owner) as (root, plan, queue):
        _pipeline_running(root)
        _validate_owner(owner, require_live=True)
        queued, _ = _audit_state(root, plan, queue)
        handoff = _audit_handoff(root, plan, key, physical_gpu)
        if not _exact_equal(handoff['owner'], owner):
            raise ValueError('audit handoff producer owner differs')
        if key in queued or _installed_completed_record(
                root, spec_for_key(key), plan) is not None:
            raise ValueError('audit handoff already exists or job is completed')
        if experiment_profile() in (
                RECOVERY_PROFILE, SINGLE_DATASET_PROFILE,
                CONTINUATION_PROFILE, METHOD_SHARD_PROFILE):
            # A handoff retains its reservation; it does not add a second job.
            inflight = set(queued)
            for job in plan['missing_jobs']:
                if ((root / 'claims' / _claim_name(job)).exists()
                        and _installed_completed_record(
                            root, spec_for_key(job), plan) is None):
                    inflight.add(job)
            if len(inflight) > pipeline_inflight_limit():
                raise ValueError('pipeline capacity is full')
        with _audit_gpu(root, physical_gpu) as installed:
            if not _exact_equal(installed, _gpu_owner(owner, physical_gpu)):
                raise ValueError('audit handoff requires its owned GPU claim')
            install_json_exclusive(
                queue.path / f'{_claim_name(key)}.json', handoff)
        return handoff


def next_audit(root, phase, owner):
    with _audit_control(root, owner) as (root, plan, queue):
        _pipeline_running(root)
        _audit_owner(root, owner, phase)
        queued, active = _audit_state(root, plan, queue)
        if active is not None:
            return None
        jobs = plan['missing_jobs'] if phase == 'formal' else plan['explanation_cells']
        for key in jobs:
            if key not in queued:
                continue
            handoff = queued[key]
            if _installed_completed_record(root, spec_for_key(key), plan) is not None:
                raise ValueError('completed handoff has no matching active audit')
            if _exact_equal(owner, {**handoff['owner'], 'job': ''}):
                raise ValueError('auditor must differ from original producer')
            with _audit_gpu(root, handoff['physical_gpu']) as installed:
                if installed is not None and installed['job'] == key:
                    expected = {**handoff['owner'], 'kind': 'formal_gpu_claim',
                                'physical_gpu': handoff['physical_gpu']}
                    if not _exact_equal(installed, expected):
                        raise ValueError('original job GPU owner changed')
                    continue
                install_json_exclusive(queue.path / 'active.json', {
                    'kind': 'formal_active_audit', 'spec_key': key,
                    'handoff_sha256': _file_digest(handoff), 'owner': owner,
                })
            return {name: handoff[name] for name in (
                'spec_key', 'run_dir', 'physical_gpu', 'seed')}
        return None


def _retention_audit_context_locked(root, plan, queue, key, owner):
    """Validate one retention context while the caller holds audit control."""
    spec = spec_for_key(key)
    phase = 'explanation' if spec.explanation else 'formal'
    _audit_owner(root, owner, phase)
    queued, active = _audit_state(root, plan, queue)
    if key not in queued or (experiment_profile() not in (
            FULL_MATRIX_PROFILE, SINGLE_DATASET_PROFILE,
            CONTINUATION_PROFILE, METHOD_SHARD_PROFILE)
            and set(queued) != {key}):
        raise ValueError('retention requires exact audit queue membership')
    if (active is None or active['spec_key'] != key
            or not _exact_equal(active['owner'], owner)):
        raise ValueError('retention requires its exact active audit')
    record = _installed_completed_record(root, spec, plan)
    if record is None:
        raise ValueError('retention completed record is missing')
    handoff = queued[key]
    run_dir = root / 'runs' / safe_spec_name(spec)
    if handoff['run_dir'] != str(run_dir):
        raise ValueError('retention run directory differs')
    run = _PinnedRoot(run_dir)
    try:
        resource = _read_run_control(run, 'RESOURCE_EVIDENCE.json')
        if (any(not _exact_equal(record[name], resource[name]) for name in (
                'claim_sha256', 'launch_sha256', 'command_sha256',
                'resource'))
                or record['log_sha256']
                != resource['artifact_sha256']['job.log']):
            raise ValueError(
                'retention completed record producer evidence differs')
        run.verify()
    finally:
        run.close()
    for physical_gpu in (0, 1):
        with _audit_gpu(root, physical_gpu) as installed:
            if installed is not None and installed['job'] == key:
                raise ValueError('retention run still owns a GPU')
    num_tasks = protocol_for(spec)['base_options']['num_tasks']
    if (type(num_tasks) is not int or isinstance(num_tasks, bool)
            or num_tasks <= 0):
        raise ValueError('retention task count is invalid')
    clone = lambda value: json.loads(_canonical_json(value))
    return {
        'plan': clone(plan), 'handoff': clone(handoff),
        'record': clone(record), 'resource': clone(resource),
        'run_dir': str(run_dir), 'num_tasks': num_tasks,
    }


@contextlib.contextmanager
def _retention_audit_transaction(root, key, owner):
    """Hold audit capacity and yield a non-relocking context reader."""
    with _audit_control(root) as (root, plan, queue):
        yield lambda: _retention_audit_context_locked(
            root, plan, queue, key, owner)


def retention_audit_context(root, key, owner):
    """Return the exact active completed audit without releasing its slot."""
    with _retention_audit_transaction(root, key, owner) as current:
        return current()


def complete_audit(root, key, owner):
    with _audit_control(root) as (root, plan, queue):
        phase = 'explanation' if spec_for_key(key).explanation else 'formal'
        _audit_owner(root, owner, phase)
        queued, active = _audit_state(root, plan, queue)
        if (active is None or active['spec_key'] != key
                or not _exact_equal(active['owner'], owner)):
            raise ValueError('audit completion owner or active identity differs')
        record = _installed_completed_record(root, spec_for_key(key), plan)
        if record is None:
            raise ValueError('audit completed record is missing')
        run = _PinnedRoot(queued[key]['run_dir'])
        try:
            evidence = _read_run_control(run, 'RESOURCE_EVIDENCE.json')
        finally:
            run.close()
        if (any(not _exact_equal(record[name], evidence[name]) for name in (
                'claim_sha256', 'launch_sha256', 'command_sha256', 'resource'))
                or record['log_sha256'] != evidence['artifact_sha256']['job.log']):
            raise ValueError('audit completed record producer evidence differs')
        queue.verify()
        os.unlink('active.json', dir_fd=queue.fd)
        os.fsync(queue.fd)
        os.unlink(f'{_claim_name(key)}.json', dir_fd=queue.fd)
        os.fsync(queue.fd)


def cancel_audit(root, owner):
    _validate_owner(owner, allow_empty_job=True)
    with _audit_control(root) as (root, plan, queue):
        _audit_owner(root, owner, owner['phase'], require_live=False)
        _, active = _audit_state(root, plan, queue, require_live=False)
        if active is None:
            return
        if not _exact_equal(active['owner'], owner):
            raise ValueError('audit cancellation owner differs')
        if not _pipeline_terminal(root):
            raise ValueError('audit cancellation requires a terminal marker')
        queue.verify()
        os.unlink('active.json', dir_fd=queue.fd)
        os.fsync(queue.fd)


def _audit_drained(root, plan, phase, queued, active):
    if phase not in {'formal', 'explanation'}:
        raise ValueError('audit phase is invalid')
    specs = formal_specs() if phase == 'formal' else explanation_specs()
    if experiment_profile() in (FULL_MATRIX_PROFILE, CONTINUATION_PROFILE):
        specs = [spec for spec in specs if spec_key(spec) in plan['missing_jobs']]
    # Do not short-circuit validation of later installed records.
    records = [_installed_completed_record(root, spec, plan) for spec in specs]
    return (all(record is not None for record in records)
            and not any(item['phase'] == phase for item in queued.values())
            and (active is None or active['owner']['phase'] != phase))


def audit_phase_ready(root, phase):
    with _audit_control(root) as (root, plan, queue):
        _pipeline_running(root)
        queued, active = _audit_state(root, plan, queue)
        return _audit_drained(root, plan, phase, queued, active)


def _require_pipeline_drained(root, phases):
    _validate_installed_success_marker_profiles(root)
    # Marker-only formal roots are legacy; installed authorities fail closed.
    if not os.path.lexists(root / 'audit_queue'):
        if experiment_profile() != FORMAL_PROFILE:
            raise ValueError(
                'non-formal success requires an installed audit queue')
        if any(os.path.lexists(root / name) for name in _FORMAL_OUTPUT_NAMES):
            plan, _ = _load_installed_plan(root)
            if not all(_audit_drained(root, plan, phase, {}, None)
                       for phase in phases):
                raise ValueError('pipeline audit phase is not drained')
        return
    with _audit_control(root) as (root, plan, queue):
        _pipeline_running(root)
        queued, active = _audit_state(root, plan, queue)
        if not all(_audit_drained(root, plan, phase, queued, active)
                   for phase in phases):
            raise ValueError('pipeline audit phase is not drained')


def _emit(value):
    print(_canonical_json(value).decode())


def _inline_json(value, label):
    _nonempty_string(value, label)
    return _json_value(value.encode(), label, require_object=True)


def _install_census(root, declarations):
    if experiment_profile() == FULL_MATRIX_PROFILE:
        raise ValueError('full matrix profile requires census-full')
    if experiment_profile() == CONTINUATION_PROFILE:
        raise ValueError('continuation profile requires census-continuation')
    root = validate_formal_root(root)
    census = build_census(declarations)
    install_json_exclusive(root / 'FORMAL_REGISTRY.json', _registry_payload())
    install_json_exclusive(root / 'COMPATIBILITY_CENSUS.json', census)
    return census


def _install_full_census(root, reuse_bundle):
    from three_dataset_full_matrix_report import load_reuse_bundle, reuse_census
    import three_dataset_full_matrix_reconcile as reconcile
    import three_dataset_seed42_reconcile as legacy
    raw = os.fspath(root)
    if type(raw) is not str or raw.startswith('//') or Path(raw).anchor != '/':
        raise ValueError('full census root requires exactly one leading slash')
    root = _path(raw, 'full census root')
    reuse = load_reuse_bundle(reuse_bundle)
    protected = (legacy.PILOT_ROOT, reconcile.ADAPTIVE_ROOT, reconcile.FORMAL_ROOT,
                 Path(reuse_bundle).parent, *(source['root'] for source in reuse['sources']))
    for origin in protected:
        origin = Path('/' + os.fspath(origin).lstrip('/'))
        if root == origin or origin in root.parents:
            raise ValueError('full census root cannot be inside a historical or reuse input root')
    census = reuse_census(reuse)
    if not os.path.lexists(root):
        root = validate_formal_root(root)
    pinned = _PinnedRoot(root)
    try:
        if (pinned.details.st_uid != os.getuid()
                or stat.S_IMODE(pinned.details.st_mode) != 0o700
                or os.listdir(pinned.fd)):
            raise ValueError('full census root must be an owned mode-0700 empty directory')
        for name, payload in (
                ('FULL_MATRIX_REUSE.json', reuse),
                ('FORMAL_REGISTRY.json', _registry_payload()),
                ('COMPATIBILITY_CENSUS.json', census)):
            pinned.verify()
            install_json_exclusive(root / name, payload)
            pinned.verify()
    finally:
        pinned.close()
    return census


def _install_continuation_census(root, reuse_bundle_dir):
    from three_dataset_cifar_continuation_profile import reuse_census
    from three_dataset_cifar_continuation_reconcile import (
        _ORIGIN_ROOT, _OLD_WORKTREE, _audit_payload, _success_payload,
        load_bundle,
    )

    raw = os.fspath(root)
    if type(raw) is not str or raw.startswith('//') or Path(raw).anchor != '/':
        raise ValueError('continuation root requires exactly one leading slash')
    root = _path(raw, 'continuation census root')
    source = _path(reuse_bundle_dir, 'continuation reuse source')
    for protected in (_ORIGIN_ROOT, _OLD_WORKTREE, source,
                      Path(__file__).resolve().parent):
        if root == protected or protected in root.parents:
            raise ValueError('continuation root is inside a protected input')
    reuse = load_bundle(source)
    if reuse['current_commit'] != _source_commit():
        raise ValueError('continuation reuse commit differs')
    census = reuse_census(reuse)
    if not os.path.lexists(root):
        root = validate_formal_root(root)
    pinned = _PinnedRoot(root)
    try:
        if (pinned.details.st_uid != os.getuid()
                or stat.S_IMODE(pinned.details.st_mode) != 0o700
                or os.listdir(pinned.fd)):
            raise ValueError('continuation root must be an owned empty mode-0700 directory')
        _mkdir_exclusive(root, 'reuse')
        audit = _audit_payload(reuse)
        for name, payload in (
                ('CIFAR_CONTINUATION_REUSE.json', reuse),
                ('CIFAR_CONTINUATION_REUSE_AUDIT.json', audit),
                ('CIFAR_CONTINUATION_REUSE_SUCCESS',
                 _success_payload(reuse, audit))):
            install_json_exclusive(root / 'reuse' / name, payload)
        install_json_exclusive(root / 'FORMAL_REGISTRY.json', _registry_payload())
        install_json_exclusive(root / 'COMPATIBILITY_CENSUS.json', census)
        pinned.verify()
    finally:
        pinned.close()
    return census


def _install_plan(root):
    root = _validate_formal_root_path(root, create=False)
    registry = _read_installed(root, 'FORMAL_REGISTRY.json')
    expected_registry = _registry_payload()
    if not _exact_equal(registry, expected_registry):
        raise ValueError('installed registry provenance differs')
    census = _read_installed(root, 'COMPATIBILITY_CENSUS.json')
    plan = build_plan(census)
    install_json_exclusive(root / 'FORMAL_PLAN.json', plan)
    install_json_exclusive(
        root / 'MISSING_JOBS.json', _missing_jobs_payload(plan, census))
    _mkdir_exclusive(root, 'claims')
    install_formal_root_identity(root, plan)
    return plan


def _validate_smoke_root(value, require_empty=False):
    root = _path(value, 'generated smoke root')
    pinned = _PinnedRoot(root)
    try:
        names = set(os.listdir(pinned.fd))
        if require_empty and names:
            raise ValueError('generated smoke root must be empty')
        pinned.verify()
    finally:
        pinned.close()
    return root


def _install_bytes_exclusive(path, payload):
    path = _path(path, 'generated fixture destination')
    if type(payload) is not bytes:
        raise TypeError('generated fixture payload must be bytes')
    parent = _PinnedRoot(path.parent)
    try:
        return _install_table_exclusive(parent, path.name, payload)
    finally:
        parent.close()


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _md5_bytes(payload):
    return hashlib.md5(payload).hexdigest()


def _pickle_bytes(payload):
    return pickle.dumps(payload, protocol=4)


def _npy_bytes(value):
    import numpy as np
    output = io.BytesIO()
    np.lib.format.write_array(output, np.asarray(value), allow_pickle=False)
    return output.getvalue()


def _deterministic_npz_bytes(values):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(values):
            info = zipfile.ZipInfo(f'{name}.npy', (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o444 << 16
            archive.writestr(info, _npy_bytes(values[name]))
    return output.getvalue()


def _generated_image_fixture(root):
    import numpy as np
    full = experiment_profile() == FULL_MATRIX_PROFILE
    num_classes = _FULL_SMOKE_CIFAR_CLASSES if full else 4
    train_per_class, test_per_class = (500, 100) if full else (52, 2)
    fixture = root / 'fixtures' / 'image'
    base = fixture / 'cifar-100-python'
    _mkdir_exclusive(root / 'fixtures', 'image')
    _mkdir_exclusive(fixture, 'cifar-100-python')

    def split_payload(split, per_class, offset):
        rows, labels, filenames = [], [], []
        columns = np.arange(3072, dtype=np.uint32)
        for class_id in range(num_classes):
            for within in range(per_class):
                sample_id = offset + class_id * per_class + within
                if full:
                    row = np.random.default_rng(sample_id).integers(
                        0, 256, 3072, dtype=np.uint8,
                    )
                else:
                    row = ((columns * 37 + sample_id * 53 + class_id * 11)
                           % 256).astype(np.uint8)
                row[:4] = np.frombuffer(
                    int(sample_id).to_bytes(4, 'little'), dtype=np.uint8)
                rows.append(row)
                labels.append(class_id)
                filenames.append(
                    f'generated-{split}-c{class_id:02d}-{within:03d}.png')
        return {
            'data': np.stack(rows),
            'fine_labels': labels,
            'coarse_labels': [0] * len(labels),
            'filenames': filenames,
            'batch_label': f'generated {split}',
        }

    payloads = {
        'train': _pickle_bytes(split_payload('train', train_per_class, 0)),
        'test': _pickle_bytes(split_payload('test', test_per_class, 10000)),
        'meta': _pickle_bytes({
            'fine_label_names': [f'generated_class_{index}' for index in range(num_classes)],
            'coarse_label_names': ['generated_superclass'],
        }),
    }
    for name, payload in payloads.items():
        _install_bytes_exclusive(base / name, payload)
    manifest = {
        'schema_version': 1,
        'kind': 'generated_cifar_pickle_fixture',
        'generated_only': True,
        'classes': list(range(num_classes)),
        'train_per_class': train_per_class,
        'test_per_class': test_per_class,
        'task_classes': [[i, i + 1] for i in range(0, num_classes, 2)],
        'files': {
            name: {
                'path': f'cifar-100-python/{name}',
                'sha256': _sha256_bytes(payload),
                'md5': _md5_bytes(payload),
                'size': len(payload),
            }
            for name, payload in sorted(payloads.items())
        },
    }
    manifest_bytes = _canonical_json(manifest) + b'\n'
    _install_bytes_exclusive(fixture / 'fixture_manifest.json', manifest_bytes)
    payloads['manifest'] = manifest_bytes
    return {
        'kind': manifest['kind'],
        'generated_only': True,
        'train_per_class': train_per_class,
        'remaining_train_per_class': train_per_class - 50,
        'files': {
            name: {
                'path': str(path.relative_to(root)),
                'sha256': _sha256_bytes(payloads[name]),
                'size': len(payloads[name]),
            }
            for name, path in (
                ('train', base / 'train'), ('test', base / 'test'),
                ('meta', base / 'meta'),
                ('manifest', fixture / 'fixture_manifest.json'),
            )
        },
    }


def _smoke_vector_fixture_counts():
    if experiment_profile() != FULL_MATRIX_PROFILE:
        return {'train_per_class': 42, 'remaining_train_per_class': 2}
    options = {
        dataset: formal_registry.protocol_for(
            FormalSpec(dataset, 'adaptive', 42))['base_options']
        for dataset in formal_registry.DATASETS if dataset != 'cifar100'
    }
    # Vector protocols forbid BiC; their sole holdout is validation.
    if any(value['bic_enabled'] for value in options.values()):
        raise ValueError('generated vector smoke requires BiC-disabled protocol')
    heldout = {dataset: (value['lambda_validation_per_class']
                         if value['lambda_validation_enabled'] else 0)
               for dataset, value in options.items()}
    train_per_class = 2 + max(heldout.values())
    return {
        'train_per_class': train_per_class,
        'remaining_train_per_class_by_dataset': {
            dataset: train_per_class - count for dataset, count in heldout.items()},
        'minimum_remaining_train_per_class': 2,
    }


def _generated_vector_fixture(root):
    import numpy as np
    fixture = root / 'fixtures' / 'vector'
    _mkdir_exclusive(root / 'fixtures', 'vector')
    counts = _smoke_vector_fixture_counts()
    train_labels = np.repeat(np.arange(4, dtype=np.int64), counts['train_per_class'])
    test_labels = np.repeat(np.arange(4, dtype=np.int64), 2)
    labels = np.concatenate((train_labels, test_labels))
    sample_count = int(labels.size)
    row_ids = np.arange(sample_count, dtype=np.float32)[:, None]
    columns = np.arange(22, dtype=np.float32)[None, :]
    features = (
        row_ids * np.float32(0.03125)
        + columns * np.float32(0.0078125)
        + labels[:, None].astype(np.float32) * np.float32(0.5)
    ).astype(np.float32)
    features[:, 0] = np.arange(sample_count, dtype=np.float32)
    sample_ids = np.asarray([
        ('generated-train' if index < train_labels.size else 'generated-test')
        + f'-{index:04d}-c{int(labels[index]):02d}'
        for index in range(sample_count)
    ])
    values = {
        'X': features,
        'y': labels,
        'train_idx': np.arange(train_labels.size, dtype=np.int64),
        'test_idx': np.arange(train_labels.size, sample_count, dtype=np.int64),
        'view_names': np.asarray([f'view_{index}' for index in range(4)]),
        'range_lo': np.asarray([0, 4, 9, 15], dtype=np.int64),
        'range_hi': np.asarray([4, 9, 15, 22], dtype=np.int64),
        'sample_ids': sample_ids,
        'generated_only': np.asarray(True),
    }
    npz = _deterministic_npz_bytes(values)
    metadata = {
        'schema_version': 1,
        'kind': 'generated_four_view_vector_fixture',
        'generated_only': True,
        'classes': [0, 1, 2, 3],
        'task_classes': [[0, 1], [2, 3]],
        **counts,
        'test_per_class': 2,
        'view_names': ['view_0', 'view_1', 'view_2', 'view_3'],
        'view_ranges': [[0, 4], [4, 9], [9, 15], [15, 22]],
        'party_widths': [4, 5, 6, 7],
        'train_identity_sha256': _digest(sample_ids[:train_labels.size].tolist()),
        'test_identity_sha256': _digest(sample_ids[train_labels.size:].tolist()),
        'label_sha256': _sha256_bytes(labels.tobytes()),
    }
    metadata_bytes = _canonical_json(metadata) + b'\n'
    paths = {
        'npz': (fixture / 'isolet_vfl.npz', npz),
        'metadata': (fixture / 'isolet_vfl.metadata.json', metadata_bytes),
    }
    for path, payload in paths.values():
        _install_bytes_exclusive(path, payload)
    return {
        'kind': metadata['kind'],
        'generated_only': True,
        **counts,
        'party_widths': [4, 5, 6, 7],
        'files': {
            name: {
                'path': str(path.relative_to(root)),
                'sha256': _sha256_bytes(payload),
                'size': len(payload),
            }
            for name, (path, payload) in paths.items()
        },
    }


def _smoke_command_options(command):
    command = tuple(command)
    try:
        start = next(index for index, token in enumerate(command)
                     if token.startswith('--'))
    except StopIteration as error:
        raise ValueError('generated smoke command has no options') from error
    tokens = command[start:]
    if len(tokens) % 2 or any(not token.startswith('--')
                              for token in tokens[0::2]):
        raise ValueError('generated smoke command options are malformed')
    options = dict(zip(tokens[0::2], tokens[1::2]))
    if len(options) * 2 != len(tokens):
        raise ValueError('generated smoke command repeats an option')
    return options


def _smoke_command_for(root, dataset, method, job_id):
    spec = FormalSpec(dataset, method, 42)
    full = experiment_profile() == FULL_MATRIX_PROFILE
    base = formal_registry._base_command(
        spec, 'cuda:0', str(root / 'runs'), not full)
    options = formal_registry._without_inherited_method_options(
        formal_registry._option_map(base, 2))
    raw_contract = {
        **formal_registry._holdout_contract(spec),
        **formal_registry.method_contract_for(spec),
    }
    for option, value in raw_contract.items():
        if option in formal_registry.OPTION_SCHEMA:
            options[option] = formal_registry._option_token(option, value)
    options.update({
        'num_classes': '4', 'num_tasks': '2',
        'custom_tasks': _SMOKE_TASKS, 'classes_per_task': '2',
        'num_parties': options['num_parties'] if full else '4', 'epochs_per_task': '1',
        'batch_size': '8', 'num_workers': '2', 'device': 'cuda:0',
        'formal_deferred_evaluation': '1',
        'save_task_checkpoints': '3', 'seed': '42',
        'results_dir': str(root / 'runs'), 'exp_name': job_id,
    })
    if full:
        for option in ('bic_steps', 'head_consolidation_steps',
                       'adagauss_adapter_epochs'):
            if option in options:
                options[option] = '1'
    else:
        options.update(bic_steps='1', head_consolidation_steps='1')
    _common, worktree, python = formal_registry._deployment_paths()
    if dataset == 'cifar100':
        fixture = root / 'fixtures' / 'image'
        options['data_path'] = str(fixture)
        options.pop('vector_npz', None)
        if full:
            options.update(
                num_classes=str(_FULL_SMOKE_CIFAR_CLASSES),
                num_tasks=str(_FULL_SMOKE_CIFAR_TASK_COUNT),
                custom_tasks=_FULL_SMOKE_CIFAR_TASKS)
        else:
            options['bic_enabled'] = '0'
            options['lambda_validation_enabled'] = '0'
        wrapper = (
            'import sys; from three_dataset_formal_driver import '
            '_generated_image_entry; '
            '_generated_image_entry(sys.argv[1], sys.argv[2])'
        )
        prefix = (str(python), '-c', wrapper, str(fixture),
                  str(worktree / 'main.py'))
    else:
        options['data_path'] = str(root / 'fixtures' / 'vector')
        options['vector_npz'] = str(
            root / 'fixtures' / 'vector' / 'isolet_vfl.npz')
        if method in formal_registry.EXTERNAL_METHODS:
            prefix = (str(python), str(worktree / 'main.py'))
        else:
            prefix = (str(python), '-c', formal_registry._wrapper_for(method))
    tokens = tuple(
        token for option, value in options.items()
        for token in (f'--{option}', str(value))
    )
    command = (*prefix, *tokens)
    if _smoke_command_options(command)['--exp_name'] != job_id:
        raise RuntimeError('generated smoke command output identity differs')
    return command


def _smoke_job_shapes():
    if experiment_profile() != FULL_MATRIX_PROFILE:
        return _SMOKE_JOB_SHAPES
    return tuple(
        (f'{dataset}-{method}'.replace('_', '-'),
         'image' if dataset == 'cifar100' else 'vector',
         method, dataset, method)
        for dataset in formal_registry.DATASETS
        for method in formal_registry.FULL_MATRIX_METHODS
    )


def _smoke_plan_payload(root, fixtures):
    jobs = []
    for index, (job_id, shape, method_shape, dataset, method) in enumerate(
            _smoke_job_shapes()):
        command = _smoke_command_for(root, dataset, method, job_id)
        jobs.append({
            'job_id': job_id,
            'data_shape': shape,
            'method_shape': method_shape,
            'dataset': dataset,
            'method': method,
            'generated_only': True,
            'scientific_gate': False,
            'fixture': shape,
            'preferred_gpu': index % 2,
            'run_dir': str(Path('runs') / job_id),
            'command': list(command),
            'command_sha256': _digest(list(command)),
            **({'seed': 42} if experiment_profile() == FULL_MATRIX_PROFILE else {}),
        })
    return {
        'schema_version': 2,
        'kind': _SMOKE_KIND,
        'generated_only': True,
        'scientific_gate': False,
        'source_commit': _source_commit(),
        'source_branch': _source_branch(),
        'source_sha256': _source_hashes(),
        'control_sha256': _smoke_control_hashes(),
        'fixtures': fixtures,
        'jobs': jobs,
    }


def plan_smoke(root):
    root = _validate_smoke_root(root, require_empty=True)
    _mkdir_exclusive(root, 'fixtures')
    for name in ('runs', 'logs', 'records', 'control'):
        _mkdir_exclusive(root, name)
    fixtures = {
        'image': _generated_image_fixture(root),
        'vector': _generated_vector_fixture(root),
    }
    plan = _smoke_plan_payload(root, fixtures)
    install_json_exclusive(root / 'SMOKE_PLAN.json', plan)
    return plan


def _read_regular_file(path, require_read_only=False):
    path = _path(path, 'generated smoke file')
    parent = _PinnedRoot(path.parent)
    descriptor = None
    try:
        descriptor = parent.open_file(path.name)
        content, details = _read_descriptor(descriptor)
        parent.verify_file_name(path.name, details)
        if require_read_only and stat.S_IMODE(details.st_mode) != 0o444:
            raise ValueError('generated fixture is not immutable')
        parent.verify()
        return content, details
    finally:
        if descriptor is not None:
            os.close(descriptor)
        parent.close()


def _smoke_control_hashes():
    source = Path(__file__).resolve().parent
    return {
        name: _sha256_bytes(_read_regular_file(source / name)[0])
        for name in (
            'three_dataset_formal_driver.py',
            'run_three_dataset_formal_comparison.sh',
        )
    }


def _fixture_entries(root):
    fixture_root = root / 'fixtures'
    entries = {'fixtures/'}
    for current, directories, files in os.walk(fixture_root, followlinks=False):
        current = Path(current)
        for name in (*directories, *files):
            path = current / name
            details = os.lstat(path)
            if stat.S_ISLNK(details.st_mode) or not (
                    stat.S_ISDIR(details.st_mode)
                    or stat.S_ISREG(details.st_mode)):
                raise ValueError('generated fixture tree contains an unsafe entry')
            relative = path.relative_to(root).as_posix()
            entries.add(relative + ('/' if stat.S_ISDIR(details.st_mode) else ''))
    return entries


def _validate_smoke_plan(root):
    root = _validate_smoke_root(root)
    content, details = _read_regular_file(
        root / 'SMOKE_PLAN.json', require_read_only=True)
    plan = _json_value(content, 'generated smoke plan', require_object=True)
    if content != _canonical_json(plan) + b'\n':
        raise ValueError('generated smoke plan is not canonical')
    if (set(plan) != {
            'schema_version', 'kind', 'generated_only', 'scientific_gate',
            'source_commit', 'source_branch', 'source_sha256',
            'control_sha256', 'fixtures', 'jobs'}
            or plan['schema_version'] != 2 or plan['kind'] != _SMOKE_KIND
            or plan['generated_only'] is not True
            or plan['scientific_gate'] is not False
            or type(plan['source_commit']) is not str
            or _COMMIT.fullmatch(plan['source_commit']) is None
            or type(plan['source_branch']) is not str
            or not plan['source_branch'] or '\x00' in plan['source_branch']
            or type(plan['source_sha256']) is not dict
            or not plan['source_sha256']
            or any(type(name) is not str or not name
                   or type(value) is not str or _SHA256.fullmatch(value) is None
                   for name, value in plan['source_sha256'].items())
            or type(plan['control_sha256']) is not dict
            or set(plan['control_sha256']) != {
                'three_dataset_formal_driver.py',
                'run_three_dataset_formal_comparison.sh'}
            or any(type(value) is not str or _SHA256.fullmatch(value) is None
                   for value in plan['control_sha256'].values())
            or set(plan['fixtures']) != {'image', 'vector'}
            or type(plan['jobs']) is not list
            or len(plan['jobs']) != len(_smoke_job_shapes())):
        raise ValueError('generated smoke plan schema is invalid')
    current_authority = {
        'source_commit': _source_commit(),
        'source_branch': _source_branch(),
        'source_sha256': _source_hashes(),
        'control_sha256': _smoke_control_hashes(),
    }
    if any(plan[name] != value for name, value in current_authority.items()):
        raise ValueError('generated smoke plan source authority changed')
    expected_files = set()
    for fixture_name, fixture in plan['fixtures'].items():
        if (type(fixture) is not dict or fixture.get('generated_only') is not True
                or type(fixture.get('files')) is not dict):
            raise ValueError('generated fixture manifest is invalid')
        for record in fixture['files'].values():
            if (type(record) is not dict
                    or set(record) != {'path', 'sha256', 'size'}
                    or type(record['path']) is not str
                    or Path(record['path']).is_absolute()
                    or '..' in Path(record['path']).parts
                    or type(record['size']) is not int
                    or isinstance(record['size'], bool)
                    or record['size'] < 0):
                raise ValueError('generated fixture file record is invalid')
            _hash_string(record['sha256'], 'generated fixture hash')
            path = root / record['path']
            if not path.is_relative_to(root / 'fixtures' / fixture_name):
                raise ValueError('generated fixture file escaped its fixture root')
            payload, file_details = _read_regular_file(
                path, require_read_only=True)
            if (len(payload) != record['size']
                    or _sha256_bytes(payload) != record['sha256']):
                raise ValueError('generated fixture content differs')
            expected_files.add(record['path'])
    expected_directories = set()
    for name in expected_files:
        parent = Path(name).parent
        while parent != Path('.'):
            expected_directories.add(parent.as_posix() + '/')
            parent = parent.parent
    if _fixture_entries(root) != expected_files | expected_directories:
        raise ValueError('generated fixture tree has extra or missing entries')
    if experiment_profile() == FULL_MATRIX_PROFILE:
        vector = plan['fixtures']['vector']
        metadata = _load_json_document(
            root / vector['files']['metadata']['path'], require_object=True)
        counts = _smoke_vector_fixture_counts()
        for value in (vector, metadata):
            if ('remaining_train_per_class' in value or not _exact_equal(
                    {name: value.get(name) for name in counts}, counts)):
                raise ValueError('generated vector fixture counts differ')

    seen = set()
    for index, (job, expected) in enumerate(zip(plan['jobs'], _smoke_job_shapes())):
        job_id, shape, method_shape, dataset, method = expected
        if (type(job) is not dict or set(job) != {
                'job_id', 'data_shape', 'method_shape', 'dataset', 'method',
                'generated_only', 'scientific_gate', 'fixture', 'preferred_gpu',
                'run_dir', 'command', 'command_sha256'} | (
                    {'seed'} if experiment_profile() == FULL_MATRIX_PROFILE else set())
                or job['job_id'] != job_id or job['data_shape'] != shape
                or job['method_shape'] != method_shape
                or job['dataset'] != dataset or job['method'] != method
                or job['generated_only'] is not True
                or job['scientific_gate'] is not False
                or job['fixture'] != shape
                or type(job['preferred_gpu']) is not int
                or job['preferred_gpu'] not in (0, 1)
                or job['preferred_gpu'] != index % 2
                or job['run_dir'] != str(Path('runs') / job_id)
                or job_id in seen):
            raise ValueError('generated smoke job matrix differs')
        if experiment_profile() == FULL_MATRIX_PROFILE and (
                type(job['seed']) is not int or job['seed'] != 42):
            raise ValueError('generated smoke job seed differs')
        seen.add(job_id)
        expected_command = list(_smoke_command_for(
            root, dataset, method, job_id))
        if (job['command'] != expected_command
                or job['command_sha256'] != _digest(expected_command)):
            raise ValueError('generated smoke command differs')
        flags = _smoke_command_options(job['command'])
        if (flags.get('--epochs_per_task') != '1'
                or (experiment_profile() != FULL_MATRIX_PROFILE
                    and flags.get('--num_parties') != '4')
                or flags.get('--device') != 'cuda:0'):
            raise ValueError('generated smoke command is not exactly bounded')
        for option in ('--data_path', '--vector_npz', '--results_dir'):
            if option in flags and not Path(flags[option]).is_relative_to(root):
                raise ValueError('generated smoke command escaped its root')
        flattened = '\n'.join(job['command']).lower()
        if ('tiny' in flattened or '/data/isolet' in flattened
                or '/data/cifar' in flattened):
            raise ValueError('generated smoke command names official data')
    if details.st_size != len(content):
        raise RuntimeError('generated smoke plan changed during validation')
    return plan


def _smoke_job(plan, job_id):
    matches = [job for job in plan['jobs'] if job['job_id'] == job_id]
    if len(matches) != 1:
        raise ValueError('unknown generated smoke job')
    return matches[0]


def smoke_check(root):
    root = _validate_smoke_root(root)
    plan = _validate_smoke_plan(root)
    forbidden = _SMOKE_FORMAL_NAMES | {
        'SMOKE_FAILED', 'SMOKE_STOPPED', 'SMOKE_AUDIT.json',
        'SMOKE_EXECUTION_SUCCESS',
    }
    if any(os.path.lexists(root / name) for name in forbidden):
        raise ValueError('generated smoke root contains a forbidden marker')
    return plan


def smoke_command(root, job_id, run_dir):
    root = _validate_smoke_root(root)
    job = _smoke_job(_validate_smoke_plan(root), job_id)
    run_dir = _path(run_dir, 'generated smoke run directory')
    if run_dir != root / job['run_dir']:
        raise ValueError('generated smoke run directory differs')
    return tuple(job['command'])


def smoke_begin(root, job_id, physical_gpu):
    root = _validate_smoke_root(root)
    plan = smoke_check(root)
    job = _smoke_job(plan, job_id)
    if type(physical_gpu) is not int or physical_gpu not in (0, 1):
        raise ValueError('generated smoke physical GPU must be exactly 0 or 1')
    launch = {
        'kind': 'generated_smoke_launch',
        'job_id': job_id,
        'generated_only': True,
        'scientific_gate': False,
        'preferred_gpu': job['preferred_gpu'],
        'physical_gpu': physical_gpu,
        'source_commit': plan['source_commit'],
        'source_branch': plan['source_branch'],
        'smoke_plan_sha256': _sha256_bytes(
            (root / 'SMOKE_PLAN.json').read_bytes()),
        'command_sha256': job['command_sha256'],
    }
    install_json_exclusive(root / 'control' / f'{job_id}.json', launch)
    return launch


def _checkpoint_tree_snapshot(source):
    source = _path(source, 'resume checkpoint source')
    pinned = _PinnedRoot(source)
    snapshot = {}
    try:
        names = sorted(os.listdir(pinned.fd))
        if not names:
            raise ValueError('resume checkpoint tree is empty')
        for name in names:
            if ('/' in name or name in ('.', '..') or not name.endswith('.pt')
                    or name.endswith('.pt.tmp')):
                raise ValueError('resume checkpoint tree contains an unexpected entry')
            descriptor = pinned.open_file(name)
            try:
                content, details = _read_descriptor(descriptor)
                pinned.verify_file_name(name, details)
            finally:
                os.close(descriptor)
            snapshot[name] = {
                'sha256': _sha256_bytes(content),
                'size': details.st_size,
                'mode': stat.S_IMODE(details.st_mode),
                'dev': details.st_dev,
                'inode': details.st_ino,
                'mtime_ns': details.st_mtime_ns,
                'ctime_ns': details.st_ctime_ns,
                'content': content,
            }
        pinned.verify()
    finally:
        pinned.close()
    return snapshot


def _copy_resume_checkpoint_tree(source, target):
    source = _path(source, 'resume checkpoint source')
    target = _path(target, 'resume checkpoint scratch target')
    if os.path.lexists(target):
        raise ValueError('resume checkpoint scratch target already exists')
    snapshot = _checkpoint_tree_snapshot(source)
    _mkdir_exclusive(target.parent, target.name)
    target_root = _PinnedRoot(target)
    try:
        for name, record in snapshot.items():
            _install_table_exclusive(target_root, name, record['content'])
        target_root.verify()
    finally:
        target_root.close()
    if _checkpoint_tree_snapshot(source) != snapshot:
        raise RuntimeError('completed checkpoint source changed during copy')
    copied = _checkpoint_tree_snapshot(target)
    if (set(copied) != set(snapshot)
            or any(copied[name]['sha256'] != snapshot[name]['sha256']
                   or copied[name]['size'] != snapshot[name]['size']
                   or copied[name]['mode'] != 0o444 for name in snapshot)):
        raise RuntimeError('resume checkpoint scratch copy differs')
    return snapshot


@contextlib.contextmanager
def _generated_cifar_runtime(data_path):
    fixture = _path(data_path, 'generated image fixture')
    manifest = _load_json_document(
        fixture / 'fixture_manifest.json', require_object=True)
    if (manifest.get('kind') != 'generated_cifar_pickle_fixture'
            or manifest.get('generated_only') is not True
            or set(manifest.get('files', {})) != {'train', 'test', 'meta'}):
        raise ValueError('generated image fixture manifest is invalid')
    for record in manifest['files'].values():
        payload, _ = _read_regular_file(fixture / record['path'])
        if (_sha256_bytes(payload) != record.get('sha256')
                or len(payload) != record.get('size')):
            raise ValueError('generated image fixture file differs')
    from torchvision.datasets import CIFAR100
    previous = CIFAR100.train_list, CIFAR100.test_list, CIFAR100.meta
    CIFAR100.train_list = [[
        'train', manifest['files']['train']['md5']]]
    CIFAR100.test_list = [[
        'test', manifest['files']['test']['md5']]]
    CIFAR100.meta = {
        'filename': 'meta', 'key': 'fine_label_names',
        'md5': manifest['files']['meta']['md5'],
    }
    try:
        yield
    finally:
        CIFAR100.train_list, CIFAR100.test_list, CIFAR100.meta = previous


@contextlib.contextmanager
def _generated_cifar_manifest_expectation(root, job, manifest):
    """Bind the full generated child to its own deterministic holdout only."""
    import numpy as np
    from calibration_split import build_manifest
    import three_dataset_formal_audit as audit

    if experiment_profile() != FULL_MATRIX_PROFILE:
        raise ValueError('generated CIFAR expectation requires full profile')
    raw_root = os.fspath(root)
    if type(raw_root) is not str or raw_root.startswith('//') or Path(raw_root).anchor != '/':
        raise ValueError('generated CIFAR expectation root requires exactly one leading slash')
    root = _path(root, 'generated CIFAR expectation root')
    if raw_root != str(root):
        raise ValueError('generated CIFAR expectation root is not canonical')
    pinned = _PinnedRoot(root)
    try:
        if (stat.S_IMODE(pinned.details.st_mode) != 0o700
                or pinned.details.st_uid != os.getuid()):
            raise ValueError('generated CIFAR expectation root is not private')
        pinned.verify()
    finally:
        pinned.close()
    plan = _validate_smoke_plan(root)
    if (type(job) is not dict or job.get('dataset') != 'cifar100'
            or not _exact_equal(job, _smoke_job(plan, job.get('job_id')))):
        raise ValueError('generated CIFAR expectation job is not plan-bound')
    plan_bytes, plan_details = _read_regular_file(
        root / 'SMOKE_PLAN.json', require_read_only=True)
    if (plan_details.st_nlink != 1
            or plan_bytes != _canonical_json(plan) + b'\n'):
        raise ValueError('generated CIFAR expectation plan identity differs')
    fixture = root / 'fixtures' / 'image'
    image = plan['fixtures']['image']
    expected_metadata = {
        'schema_version': 1, 'kind': 'generated_cifar_pickle_fixture',
        'generated_only': True, 'classes': list(range(20)),
        'train_per_class': 500, 'test_per_class': 100,
        'task_classes': [[i, i + 1] for i in range(0, 20, 2)],
    }
    if (type(manifest) is not dict
            or set(manifest) != set(expected_metadata) | {'files'}
            or not _exact_equal({k: manifest[k] for k in expected_metadata},
                                expected_metadata)
            or type(manifest['files']) is not dict
            or set(manifest['files']) != {'train', 'test', 'meta'}
            or not _exact_equal({k: image.get(k) for k in (
                'kind', 'generated_only', 'train_per_class', 'remaining_train_per_class')},
                {'kind': expected_metadata['kind'], 'generated_only': True,
                 'train_per_class': 500, 'remaining_train_per_class': 450})
            or set(image['files']) != {'train', 'test', 'meta', 'manifest'}):
        raise ValueError('generated CIFAR expectation fixture shape differs')
    payloads = {}
    for name in ('manifest', 'train', 'test', 'meta'):
        relative = ('fixture_manifest.json' if name == 'manifest'
                    else f'cifar-100-python/{name}')
        record = image['files'][name]
        payload, details = _read_regular_file(fixture / relative, require_read_only=True)
        if (record['path'] != f'fixtures/image/{relative}'
                or details.st_nlink != 1 or record['size'] != len(payload)
                or record['sha256'] != _sha256_bytes(payload)):
            raise ValueError('generated CIFAR expectation fixture identity differs')
        if name == 'manifest':
            if payload != _canonical_json(manifest) + b'\n':
                raise ValueError('generated CIFAR expectation manifest is not bound canonical JSON')
        elif not _exact_equal(manifest['files'][name], {
                'path': relative, 'sha256': record['sha256'],
                'md5': _md5_bytes(payload), 'size': len(payload)}):
            raise ValueError('generated CIFAR expectation fixture hash differs')
        payloads[name] = payload

    # The checkpoint loader handles torch archives, not raw CIFAR NumPy pickles.
    # Keep this whitelist local to the generated child; never execute pickle globals.
    class FixtureUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'numpy' and name in {'ndarray', 'dtype'}:
                return getattr(np, name)
            if (module in {'numpy.core.multiarray', 'numpy._core.multiarray'}
                    and name == '_reconstruct'):
                return np._core.multiarray._reconstruct
            raise ValueError('generated CIFAR fixture pickle global is forbidden')

    loaded = {}
    for name in ('train', 'test', 'meta'):
        stream = io.BytesIO(payloads[name])
        try:
            value = FixtureUnpickler(stream).load()
        except (pickle.UnpicklingError, EOFError, TypeError) as error:
            raise ValueError('generated CIFAR fixture pickle is invalid') from error
        if stream.read() or type(value) is not dict:
            raise ValueError('generated CIFAR fixture pickle shape differs')
        if name == 'meta':
            valid = _exact_equal(value, {
                'fine_label_names': [f'generated_class_{i}' for i in range(20)],
                'coarse_label_names': ['generated_superclass']})
        else:
            count = 500 if name == 'train' else 100
            labels = [i for i in range(20) for _ in range(count)]
            valid = (
                set(value) == {'data', 'fine_labels', 'coarse_labels',
                               'filenames', 'batch_label'}
                and type(value['data']) is np.ndarray
                and value['data'].dtype == np.uint8
                and value['data'].shape == (20 * count, 3072)
                and _exact_equal(value['fine_labels'], labels)
                and _exact_equal(value['coarse_labels'], [0] * len(labels))
                and value['filenames'] == [f'generated-{name}-c{i:02d}-{j:03d}.png'
                                          for i in range(20) for j in range(count)]
                and value['batch_label'] == f'generated {name}')
        if not valid:
            raise ValueError('generated CIFAR fixture pickle shape differs')
        loaded[name] = value

    flags = _smoke_command_options(job['command'])
    labels = loaded['train']['fine_labels']
    calibration = build_manifest(labels, 25, int(flags['--bic_split_seed']))
    calibration_ids = set(calibration['ordered_indices'])
    validation = build_manifest(
        labels, 25, int(flags['--lambda_validation_split_seed']),
        excluded_indices=calibration_ids)
    for split, seed in ((calibration, int(flags['--bic_split_seed'])),
                        (validation, int(flags['--lambda_validation_split_seed']))):
        if (set(split) != {'dataset', 'seed', 'per_class', 'by_class',
                          'ordered_indices', 'sha256'}
                or split['dataset'] != 'cifar100-train'
                or type(split['seed']) is not int or split['seed'] != seed
                or type(split['per_class']) is not int or split['per_class'] != 25
                or type(split['by_class']) is not dict
                or set(split['by_class']) != {str(i) for i in range(20)}):
            raise ValueError('generated CIFAR rebuilt split shape differs')
        ordered = []
        for i in range(20):
            indices = split['by_class'][str(i)]
            if (type(indices) is not list or len(indices) != 25
                    or any(type(index) is not int or not i * 500 <= index < (i + 1) * 500
                           for index in indices)
                    or len(set(indices)) != 25):
                raise ValueError('generated CIFAR rebuilt split membership differs')
            ordered.extend(indices)
        if (not _exact_equal(split['ordered_indices'], ordered)
                or split['sha256'] != _digest(ordered)):
            raise ValueError('generated CIFAR rebuilt split self-hash differs')
    validation_ids = set(validation['ordered_indices'])
    train_ids = set(range(len(labels))) - calibration_ids - validation_ids
    if (calibration_ids & validation_ids
            or any(sum(labels[index] == i for index in train_ids) != 450
                   for i in range(20))):
        raise ValueError('generated CIFAR rebuilt splits overlap or train count differs')
    # ponytail: this child is single-job/single-thread; no general runtime setter.
    original = audit._AUTHORITATIVE_MANIFEST['cifar100']
    audit._AUTHORITATIVE_MANIFEST['cifar100'] = {
        'logical_path': 'validation/validation_manifest.json',
        **{key: validation[key] for key in ('dataset', 'seed', 'per_class', 'sha256')},
    }
    try:
        yield validation
    finally:
        audit._AUTHORITATIVE_MANIFEST['cifar100'] = original


def _generated_image_entry(fixture, main_path):
    raw_fixture, raw_main = os.fspath(fixture), os.fspath(main_path)
    if experiment_profile() == FULL_MATRIX_PROFILE and any(
            type(raw) is not str or raw.startswith('//') or Path(raw).anchor != '/'
            for raw in (raw_fixture, raw_main)):
        raise ValueError('generated image entry paths require exactly one leading slash')
    fixture = _path(fixture, 'generated image fixture')
    main_path = _path(main_path, 'generated image entrypoint')
    if experiment_profile() == FULL_MATRIX_PROFILE:
        if raw_fixture != str(fixture) or raw_main != str(main_path):
            raise ValueError('generated image entry paths are not canonical')
        root = fixture.parent.parent
        plan = _validate_smoke_plan(root)
        flags = _smoke_command_options(sys.argv)
        job = _smoke_job(plan, flags.get('--exp_name'))
        if (fixture != root / 'fixtures' / 'image'
                or main_path != Path(__file__).resolve().parent / 'main.py'
                or job['command'][0] != sys.executable
                or sys.argv != ['-c', *job['command'][3:]]):
            raise ValueError('generated image entry command is not plan-bound')
        manifest = _load_json_document(
            fixture / 'fixture_manifest.json', require_object=True)
        with _generated_cifar_runtime(fixture), \
                _generated_cifar_manifest_expectation(root, job, manifest):
            sys.argv = [str(main_path), *sys.argv[3:]]
            runpy.run_path(str(main_path), run_name='__main__')
        return
    with _generated_cifar_runtime(fixture):
        sys.argv = [str(main_path), *sys.argv[3:]]
        runpy.run_path(str(main_path), run_name='__main__')


def _strict_resume_probe_main(run_dir):
    from adaptive_consolidation_audit import _restricted_torch_load
    from bic_calibration import TaskAffineCalibrator
    from cl_methods import get_cl_method
    from data_utils import TaskManager, VFLDataset
    from metrics import MetricsTracker
    from models import build_models
    import runner
    from vfl_trainer import VFLTrainer

    run = _path(run_dir, 'completed generated smoke run')
    pinned = _PinnedRoot(run)
    try:
        pinned.verify()
    finally:
        pinned.close()
    config = _load_json_document(run / 'config.json', require_object=True)
    if not config:
        raise ValueError('resume probe config is malformed')
    source = run / 'checkpoints'
    source_snapshot = _checkpoint_tree_snapshot(source)
    args = SimpleNamespace(**config)
    with tempfile.TemporaryDirectory(prefix='formal_resume_probe_') as temporary:
        scratch = Path(temporary)
        args.output_dir = str(scratch)
        args.resume_run_dir = str(run)
        runtime = (_generated_cifar_runtime(args.data_path)
                   if getattr(args, 'data', None) == 'cifar100'
                   else contextlib.nullcontext())
        with runtime:
            dataset = VFLDataset(args)
        dataset.get_test_loader = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError('resume probe forbids test-loader access'))
        _copy_resume_checkpoint_tree(source, scratch / 'checkpoints')
        task_manager = TaskManager(args)
        timeline = task_manager.get_timeline()
        if (not timeline or any(event.get('type') != 'CIL'
                                for event in timeline)):
            raise ValueError('resume probe requires an exact CIL timeline')
        final_event = len(timeline) - 1
        final_name = f'event_{final_event}_CIL.pt'
        if final_name not in source_snapshot:
            raise ValueError('resume probe latest completed checkpoint is missing')
        checkpoint = runner._decode_checkpoint_value(
            _restricted_torch_load(source_snapshot[final_name]['content']))
        if (checkpoint.get('event_idx') != final_event
                or checkpoint.get('step') != f'event_{final_event}_CIL'):
            raise ValueError('resume probe latest completed boundary differs')
        bottoms, top = build_models(args)
        trainer = VFLTrainer(bottoms, top, args)
        trainer.dataset_ref = dataset
        method = get_cl_method(args.cl_method, trainer, args)
        tracker = MetricsTracker()
        calibrator = TaskAffineCalibrator()
        start, seen, history = runner._load_resume_checkpoint(
            args, trainer, method, task_manager, tracker, calibrator)
        expected_seen = {
            int(event['task_id']): [int(value) for value in event['new_classes']]
            for event in timeline
        }
        if (start != len(timeline) or seen != expected_seen
                or checkpoint.get('seen_task_classes') != expected_seen
                or history != checkpoint.get('bic_history', [])):
            raise ValueError('resume probe did not restore the completed timeline')
        checks = {
            'trainer': runner._checkpoint_values_equal(
                trainer.get_state(), checkpoint['trainer_state']),
            'method': runner._checkpoint_values_equal(
                method.get_state() if hasattr(method, 'get_state') else {},
                checkpoint.get('cl_state', {})),
            'tracker': runner._checkpoint_values_equal(
                tracker.to_dict(), checkpoint.get('tracker_state', {})),
            'rng': runner._checkpoint_values_equal(
                runner._capture_rng_state(), checkpoint['rng_state']),
        }
        if checkpoint.get('bic_state') is not None:
            checks['bic'] = runner._checkpoint_values_equal(
                calibrator.state_dict(), checkpoint['bic_state'])
        if not all(checks.values()):
            raise ValueError(f'resume probe state roundtrip differs: {checks}')
    if _checkpoint_tree_snapshot(source) != source_snapshot:
        raise RuntimeError('resume probe mutated the completed checkpoint source')
    result = {
        'status': 'RESUME_PROBE_SUCCESS',
        'start_event_idx': start,
        'seen_tasks': {str(key): value for key, value in sorted(seen.items())},
        'checkpoint_sha256': source_snapshot[final_name]['sha256'],
        'source_unchanged': True,
        'test_loader_opened': False,
    }
    print('RESUME_PROBE_SUCCESS')
    return result


def resume_probe(run_dir, physical_gpu):
    if (type(physical_gpu) is not int
            or physical_gpu not in (0, 1)):
        raise ValueError('resume probe physical GPU must be exactly 0 or 1')
    run = _path(run_dir, 'completed generated smoke run')
    wrapper = (
        'import json,sys; from three_dataset_formal_driver import '
        '_strict_resume_probe_main; '
        'print(json.dumps(_strict_resume_probe_main(sys.argv[1]), '
        'sort_keys=True,separators=(",",":")))'
    )
    environment = dict(os.environ)
    environment['CUDA_VISIBLE_DEVICES'] = str(physical_gpu)
    completed = subprocess.run(
        [sys.executable, '-c', wrapper, str(run)],
        cwd=Path(__file__).resolve().parent,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=environment,
    )
    lines = completed.stdout.splitlines()
    try:
        result = next(json.loads(line) for line in reversed(lines)
                      if line.startswith('{'))
    except (StopIteration, json.JSONDecodeError):
        result = None
    if (completed.returncode != 0 or type(result) is not dict
            or result.get('status') != 'RESUME_PROBE_SUCCESS'):
        raise ValueError('resume probe failed: ' + completed.stdout[-2000:])
    return result


def _assert_no_unsafe_tree_entries(root):
    for current, directories, files in os.walk(root, followlinks=False):
        for name in (*directories, *files):
            details = os.lstat(Path(current) / name)
            if stat.S_ISLNK(details.st_mode) or not (
                    stat.S_ISDIR(details.st_mode)
                    or stat.S_ISREG(details.st_mode)):
                raise ValueError('generated smoke tree contains an unsafe entry')


def _strict_smoke_producer_audit(run, job):
    import adaptive_consolidation_audit as producer
    expectation = contextlib.nullcontext()
    if experiment_profile() == FULL_MATRIX_PROFILE and job.get('dataset') == 'cifar100':
        raw_run = os.fspath(run)
        if type(raw_run) is not str or raw_run.startswith('//') or Path(raw_run).anchor != '/':
            raise ValueError('generated CIFAR producer run requires exactly one leading slash')
        run = _path(run, 'generated CIFAR producer run')
        root = run.parent.parent
        plan = _validate_smoke_plan(root)
        if (raw_run != str(run) or not _exact_equal(
                job, _smoke_job(plan, job.get('job_id')))
                or run != root / job['run_dir']):
            raise ValueError('generated CIFAR producer run is not plan-bound')
        config = _load_json_document(run / 'config.json', require_object=True)
        options = _smoke_command_options(job['command'])
        if config.get('output_dir') != str(run):
            raise ValueError('generated CIFAR producer config output differs')
        for option, token in options.items():
            name, kind = option[2:], formal_registry.OPTION_SCHEMA[option[2:]]
            expected = (token if kind == 'tasks' else
                        formal_registry._normalize_value(kind, token))
            if name == 'unlearn_after_tasks':
                expected = [expected]
            elif name == 'unlearn_classes':
                expected = [[expected]]
            if name not in config or config[name] != expected:
                raise ValueError(f'generated CIFAR producer config differs: {name}')
        manifest = _load_json_document(
            root / 'fixtures/image/fixture_manifest.json', require_object=True)
        expectation = _generated_cifar_manifest_expectation(root, job, manifest)
    with expectation:
        for name in (
                'FORMAL_EVALUATION_PUBLISHING.json',
                'FORMAL_EVALUATION_PUBLISHED.json'):
            _read_regular_file(run / name)
        complete, seal = producer._load_sealed_complete_artifact(run)
        identity = complete.get('identity') if type(complete) is dict else None
        protocol = identity.get('protocol') if type(identity) is dict else None
        if type(protocol) is not dict:
            raise ValueError('generated smoke producer protocol is missing')
        producer._validate_formal_published(
            run, identity, complete, seal)
    expected_method = {
        'fixed_full': 'proto_evolve',
        'adaptive': 'proto_evolve',
    }.get(job['method'], job['method'])
    expected_data = (
        'cifar100' if job['data_shape'] == 'image' else 'tabvfl')
    expected_head = job['method_shape'] in {'fixed_endpoint', 'adaptive'}
    expected_mode = (
        'adaptive_dual_branch' if expected_head else 'full_classifier')
    expected_parties = (int(_smoke_command_options(job['command'])['--num_parties'])
                        if experiment_profile() == FULL_MATRIX_PROFILE else 4)
    if (protocol.get('seed') != 42
            or protocol.get('data') != expected_data
            or protocol.get('cl_method') != expected_method
            or protocol.get('num_tasks') != int(
                _smoke_command_options(job['command'])['--num_tasks'])
            or protocol.get('num_parties') != expected_parties
            or bool(protocol.get('head_consolidation_enabled')) != expected_head
            or protocol.get('head_consolidation_mode') != expected_mode
            or protocol.get('formal_deferred_evaluation') is not True):
        raise ValueError('generated smoke producer protocol differs')
    provenance = protocol.get('source_provenance')
    if (type(provenance) is not dict
            or type(provenance.get('source_commit')) is not str
            or _COMMIT.fullmatch(provenance['source_commit']) is None
            or type(provenance.get('source_sha256')) is not dict):
        raise ValueError('generated smoke producer source identity is invalid')
    return {
        'status': 'FORMAL_SMOKE_PRODUCER_VERIFIED',
        'published': True,
        'transaction_sha256': complete['transaction_sha256'],
        'identity_sha256': seal['identity_sha256'],
        'protocol_sha256': seal['protocol_sha256'],
        'source_commit': provenance['source_commit'],
        'source_sha256': provenance['source_sha256'],
    }


def _pinned_directory_entries(path):
    pinned = _PinnedRoot(path)
    entries = {}
    try:
        for name in sorted(os.listdir(pinned.fd)):
            details = os.stat(name, dir_fd=pinned.fd, follow_symlinks=False)
            if not (stat.S_ISDIR(details.st_mode)
                    or stat.S_ISREG(details.st_mode)):
                raise ValueError('generated smoke control entry is unsafe')
            entries[name] = 'directory' if stat.S_ISDIR(details.st_mode) else 'file'
        pinned.verify()
    finally:
        pinned.close()
    return entries


def _smoke_live_processes(root):
    proc = Path('/proc')
    if not proc.is_dir():
        raise RuntimeError('generated smoke process audit requires procfs')
    ancestors = {os.getpid()}
    parent = os.getppid()
    while parent > 1 and parent not in ancestors:
        ancestors.add(parent)
        try:
            rows = (proc / str(parent) / 'status').read_text().splitlines()
            parent = int(next(
                row.split(':', 1)[1] for row in rows
                if row.startswith('PPid:')))
        except (OSError, StopIteration, ValueError):
            break
    needle = str(root).encode()
    live = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in ancestors:
            continue
        try:
            command = (entry / 'cmdline').read_bytes()
            status = (entry / 'status').read_text()
        except OSError:
            continue
        if needle in command and '\nState:\tZ' not in status:
            live.append(pid)
    return tuple(sorted(live))


def _smoke_runtime_state(root, plan):
    expected_jobs = [job['job_id'] for job in plan['jobs']]
    root_entries = _pinned_directory_entries(root)
    claim_entries = 0
    gpu_claim_entries = 0
    for name, field in (
            ('claims', 'claim'), ('gpu_claims', 'GPU claim')):
        if name in root_entries:
            if root_entries[name] != 'directory':
                raise ValueError(f'generated smoke {field} evidence is unsafe')
            count = len(_pinned_directory_entries(root / name))
            if name == 'claims':
                claim_entries = count
            else:
                gpu_claim_entries = count
    expected_root = _SMOKE_CONTROL_DIRS | {'SMOKE_PLAN.json'}
    if set(root_entries) != expected_root:
        raise ValueError('generated smoke root has missing or extra control entries')
    expected_runs = {job['job_id']: 'directory' for job in plan['jobs']}
    expected_files = {f'{job_id}.json': 'file' for job_id in expected_jobs}
    expected_logs = {f'{job_id}.log': 'file' for job_id in expected_jobs}
    if (_pinned_directory_entries(root / 'runs') != expected_runs
            or _pinned_directory_entries(root / 'control') != expected_files
            or _pinned_directory_entries(root / 'records') != expected_files
            or _pinned_directory_entries(root / 'logs') != expected_logs):
        raise ValueError('generated smoke run/control membership differs')
    live = _smoke_live_processes(root)
    state = {
        'claim_entries': claim_entries,
        'gpu_claim_entries': gpu_claim_entries,
        'live_processes': len(live),
    }
    if any(state.values()):
        raise ValueError(f'generated smoke has active runtime state: {state}')
    return state


def _smoke_run_record(root, job_id, run_dir):
    root = _validate_smoke_root(root)
    plan = _validate_smoke_plan(root)
    job = _smoke_job(plan, job_id)
    run = _path(run_dir, 'completed generated smoke run')
    if run != root / job['run_dir']:
        raise ValueError('completed generated smoke run escaped its root')
    _assert_no_unsafe_tree_entries(run)
    launch = _load_json_document(
        root / 'control' / f'{job_id}.json', require_object=True)
    expected_launch = {
        'kind': 'generated_smoke_launch', 'job_id': job_id,
        'generated_only': True, 'scientific_gate': False,
        'source_commit': plan['source_commit'],
        'source_branch': plan['source_branch'],
        'smoke_plan_sha256': _sha256_bytes(
            (root / 'SMOKE_PLAN.json').read_bytes()),
        'command_sha256': job['command_sha256'],
    }
    if (set(launch) != set(expected_launch) | {
            'preferred_gpu', 'physical_gpu'}
            or any(launch[name] != value
                   for name, value in expected_launch.items())
            or type(launch['preferred_gpu']) is not int
            or launch['preferred_gpu'] not in (0, 1)
            or launch['preferred_gpu'] != job['preferred_gpu']
            or type(launch['physical_gpu']) is not int
            or launch['physical_gpu'] not in (0, 1)):
        raise ValueError('generated smoke launch evidence differs')
    options = _smoke_command_options(job['command'])
    num_tasks = int(options['--num_tasks'])
    task_classes = [[int(value) for value in task.split(',')]
                    for task in options['--custom_tasks'].split('|')]
    config = _load_json_document(run / 'config.json', require_object=True)
    critical = {
        'num_classes': int(options['--num_classes']), 'num_tasks': num_tasks,
        'custom_tasks': options['--custom_tasks'],
        'num_parties': (int(options['--num_parties'])
                        if experiment_profile() == FULL_MATRIX_PROFILE else 4),
        'epochs_per_task': 1,
        'formal_deferred_evaluation': True, 'save_task_checkpoints': 3,
        'seed': 42, 'device': 'cuda:0', 'output_dir': str(run),
        'bic_enabled': int(options['--bic_enabled']),
        'lambda_validation_enabled': int(
            options['--lambda_validation_enabled']),
    }
    if (any(config.get(name) != value for name, value in critical.items())
            or any(type(config.get(name)) is not int for name in (
                'bic_enabled', 'lambda_validation_enabled'))):
        raise ValueError('generated smoke runtime config differs')
    if experiment_profile() == FULL_MATRIX_PROFILE:
        for option, token in options.items():
            name = option[2:]
            kind = formal_registry.OPTION_SCHEMA[name]
            expected = (token if kind == 'tasks' else
                        formal_registry._normalize_value(kind, token))
            if name == 'unlearn_after_tasks':
                expected = [expected]
            elif name == 'unlearn_classes':
                expected = [[expected]]
            if name not in config or config[name] != expected:
                raise ValueError(f'generated smoke runtime config differs: {name}')
            if name in {'unlearn_after_tasks', 'unlearn_classes'} and not _exact_equal(
                    config[name], expected):
                raise ValueError(f'generated smoke runtime config differs: {name}')
    for option in ('--data_path', '--vector_npz'):
        if option in options and config.get(option[2:]) != options[option]:
            raise ValueError('generated smoke runtime fixture path differs')
    required = [
        'config.json', 'results.json', 'data_flow_audit.jsonl',
        *(f'checkpoints/event_{i}_CIL.pt' for i in range(num_tasks)),
        'checkpoints/formal_final.pt',
        *(f'formal_snapshots/event_{i}_CIL.pt' for i in range(num_tasks)),
        'FORMAL_STATE_FROZEN.json', 'FORMAL_EVALUATION_PENDING.json',
        'FORMAL_EVALUATION_CONSUMING.json',
        'FORMAL_EVALUATION_COMPLETE.json',
        'FORMAL_EVALUATION_SEALED.json',
        'FORMAL_EVALUATION_PUBLISHING.json',
        'FORMAL_EVALUATION_PUBLISHED.json',
        'formal_access/test.consumed.json',
    ]
    validation_access = job['method_shape'] in {'fixed_endpoint', 'adaptive'}
    if experiment_profile() == FULL_MATRIX_PROFILE:
        validation_access = formal_registry.validation_access_for(FormalSpec(
            job['dataset'], job['method'], 42))
    if validation_access:
        required.append('formal_access/validation.consumed.json')
    if int(options['--bic_enabled']) and not validation_access:
        required.append('formal_access/calibration.consumed.json')
    if (job['data_shape'] == 'image'
            and (not int(options['--bic_enabled']) or validation_access) and (
            run / 'formal_access/calibration.consumed.json').exists()):
        raise ValueError('generated image smoke has unexpected calibration access')
    access = {Path(name).name: 'file' for name in required
              if name.startswith('formal_access/')}
    if (experiment_profile() == FULL_MATRIX_PROFILE
            and _pinned_directory_entries(run / 'formal_access') != access):
        raise ValueError('generated smoke formal access membership differs')
    for folder in ('checkpoints', 'formal_snapshots'):
        expected = {Path(name).name: 'file' for name in required
                    if name.startswith(folder + '/')}
        entries = _pinned_directory_entries(run / folder)
        # The unchanged runner may retain its save_task_checkpoints=3 rolling copy.
        if folder == 'checkpoints' and entries.get('resume_latest.pt') == 'file':
            entries.pop('resume_latest.pt')
        if entries != expected:
            raise ValueError('generated smoke event/snapshot membership differs')
    hashes = {}
    for logical in required:
        payload, _ = _read_regular_file(run / logical)
        hashes[logical] = _sha256_bytes(payload)
    flow, _ = _read_regular_file(run / 'data_flow_audit.jsonl')
    try:
        accesses = [json.loads(line) for line in flow.decode().splitlines()
                    if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('generated smoke data-flow evidence is malformed') from error
    tests = [record for record in accesses
             if type(record) is dict and record.get('split') == 'test']
    if (len(tests) != 1
            or tests[0].get('phase') != 'final_test_post_install'
            or tests[0].get('event_idx') != num_tasks - 1
            or tests[0].get('task_id') != num_tasks - 1
            or tests[0].get('timeline_step') != f'event_{num_tasks - 1}_CIL'
            or tests[0].get('classes') != [value for task in task_classes for value in task]):
        raise ValueError('generated smoke did not use one post-freeze test access')
    producer = _strict_smoke_producer_audit(run, job)
    if (producer.get('source_commit') != plan['source_commit']
            or producer.get('source_sha256') != plan['source_sha256']
            or producer.get('published') is not True):
        raise ValueError('generated smoke producer source identity differs')
    probe = resume_probe(run, launch['physical_gpu'])
    log, log_details = _read_regular_file(root / 'logs' / f'{job_id}.log')
    if stat.S_IMODE(log_details.st_mode) != 0o444:
        raise ValueError('generated smoke job log is not immutable')
    return {
        'schema_version': 2,
        'kind': 'generated_smoke_completed',
        'job_id': job_id,
        'data_shape': job['data_shape'],
        'method_shape': job['method_shape'],
        'generated_only': True,
        'scientific_gate': False,
        'source_commit': plan['source_commit'],
        'source_branch': plan['source_branch'],
        'source_sha256': plan['source_sha256'],
        'control_sha256': plan['control_sha256'],
        'smoke_plan_sha256': expected_launch['smoke_plan_sha256'],
        'command_sha256': job['command_sha256'],
        'preferred_gpu': launch['preferred_gpu'],
        'physical_gpu': launch['physical_gpu'],
        'fixture_sha256': {
            name: record['sha256'] for name, record
            in plan['fixtures'][job['fixture']]['files'].items()
        },
        'artifact_sha256': hashes,
        'job_log_sha256': _sha256_bytes(log),
        'resume_probe': probe,
        'producer_audit': producer,
        'train_only_stage_snapshots': num_tasks,
        'post_freeze_test_accesses': 1,
        'passed': True,
    }


def install_smoke_record(root, job_id, run_dir):
    root = _validate_smoke_root(root)
    record = _smoke_run_record(root, job_id, run_dir)
    install_json_exclusive(root / 'records' / f'{job_id}.json', record)
    return record


def install_smoke_marker(root, name, kind, exit_code):
    root = _validate_smoke_root(root)
    if name not in {'SMOKE_FAILED', 'SMOKE_STOPPED'}:
        raise ValueError('unsupported generated smoke marker')
    if kind not in {'failed', 'stopped'}:
        raise ValueError('generated smoke marker kind is invalid')
    if (type(exit_code) is not int or isinstance(exit_code, bool)
            or exit_code == 0):
        raise ValueError('generated smoke marker exit code is invalid')
    marker = {
        'schema_version': 1, 'kind': kind, 'exit_code': exit_code,
        'generated_only': True, 'scientific_gate': False,
    }
    install_json_exclusive(root / name, marker)
    return marker


def audit_smoke(root):
    root = _validate_smoke_root(root)
    plan = _validate_smoke_plan(root)
    _assert_no_unsafe_tree_entries(root)
    forbidden = _SMOKE_FORMAL_NAMES | {'SMOKE_FAILED', 'SMOKE_STOPPED'}
    if any(os.path.lexists(root / name) for name in forbidden):
        raise ValueError('scientific or failure marker is forbidden in smoke audit')
    plan_bytes, _ = _read_regular_file(root / 'SMOKE_PLAN.json')
    runtime_state = _smoke_runtime_state(root, plan)
    records = {}
    for job in plan['jobs']:
        path = root / 'records' / f"{job['job_id']}.json"
        record = _load_json_document(path, require_object=True)
        derived = _smoke_run_record(
            root, job['job_id'], root / job['run_dir'])
        if record != derived:
            raise ValueError('generated smoke installed record differs from run')
        if (record.get('job_id') != path.stem
                or record.get('kind') != 'generated_smoke_completed'
                or record.get('generated_only') is not True
                or record.get('scientific_gate') is not False
                or record.get('source_commit') != plan['source_commit']
                or record.get('source_branch') != plan['source_branch']
                or record.get('passed') is not True
                or record.get('train_only_stage_snapshots') != int(
                    _smoke_command_options(job['command'])['--num_tasks'])
                or record.get('post_freeze_test_accesses') != 1
                or record.get('resume_probe', {}).get('status')
                != 'RESUME_PROBE_SUCCESS'
                or record.get('producer_audit', {}).get('status')
                != 'FORMAL_SMOKE_PRODUCER_VERIFIED'):
            raise ValueError('generated smoke record failed its audit contract')
        records[path.stem] = _sha256_bytes(_read_regular_file(path)[0])
    if _smoke_runtime_state(root, plan) != runtime_state:
        raise RuntimeError('generated smoke runtime state changed during audit')
    if _validate_smoke_plan(root) != plan:
        raise RuntimeError('generated smoke plan changed during audit')
    report = {
        'schema_version': 1,
        'status': 'SMOKE_EXECUTION_SUCCESS',
        'kind': 'three_dataset_generated_smoke_audit',
        'generated_only': True,
        'scientific_gate': False,
        'source_commit': plan['source_commit'],
        'source_branch': plan['source_branch'],
        'source_sha256': plan['source_sha256'],
        'smoke_plan_sha256': _sha256_bytes(plan_bytes),
        'job_records': records,
        'job_count': len(records),
        'runtime_state': runtime_state,
    }
    install_json_exclusive(root / 'SMOKE_AUDIT.json', report)
    marker = {
        'schema_version': 1,
        'status': 'SMOKE_EXECUTION_SUCCESS',
        'generated_only': True,
        'scientific_gate': False,
        'smoke_audit_sha256': _sha256_bytes(
            (root / 'SMOKE_AUDIT.json').read_bytes()),
        'smoke_plan_sha256': report['smoke_plan_sha256'],
    }
    install_json_exclusive(root / 'SMOKE_EXECUTION_SUCCESS', marker)
    return report


def _parser():
    parser = argparse.ArgumentParser(prog='three_dataset_formal_driver')
    actions = parser.add_subparsers(dest='action', required=True)
    check = actions.add_parser('check')
    check.add_argument('--root', required=True)
    census = actions.add_parser('census')
    census.add_argument('--root', required=True)
    census.add_argument('--declarations', required=True)
    census_full = actions.add_parser('census-full')
    census_full.add_argument('--root', required=True)
    census_full.add_argument('--reuse-bundle', required=True)
    census_continuation = actions.add_parser('census-continuation')
    census_continuation.add_argument('--root', required=True)
    census_continuation.add_argument('--reuse-bundle', required=True)
    plan = actions.add_parser('plan')
    plan.add_argument('--root', required=True)
    jobs = actions.add_parser('jobs')
    jobs.add_argument('--root', required=True)
    disk = actions.add_parser('disk-status')
    disk.add_argument('--root', required=True)
    disk.add_argument('--requested-slots', required=True, type=int, choices=(1, 2))
    claim = actions.add_parser('claim')
    claim.add_argument('--root', required=True)
    claim.add_argument('--phase', required=True,
                       choices=('formal', 'explanation'))
    claim.add_argument('--owner-json', required=True)
    claim.add_argument('--pipeline', action='store_true')
    claim.add_argument('--format', choices=('text', 'json'), default='text')
    owner = actions.add_parser('owner')
    owner.add_argument('--root', required=True)
    owner.add_argument('--phase', required=True,
                       choices=('formal', 'explanation'))
    owner.add_argument('--launcher-token', required=True)
    owner.add_argument('--worker-role', required=True)
    owner.add_argument('--pid', required=True, type=int)
    owner.add_argument('--pgid', required=True, type=int)
    claim_owner = actions.add_parser('claim-owner')
    claim_owner.add_argument('--root', required=True)
    claim_owner.add_argument('--spec', required=True)
    release = actions.add_parser('release-claim')
    release.add_argument('--root', required=True)
    release.add_argument('--spec', required=True)
    release.add_argument('--owner-json', required=True)
    mark_started = actions.add_parser('mark-started')
    mark_started.add_argument('--root', required=True)
    mark_started.add_argument('--spec', required=True)
    mark_started.add_argument('--run-dir', required=True)
    mark_started.add_argument('--owner-json', required=True)
    command = actions.add_parser('command')
    command.add_argument('--spec', required=True)
    command.add_argument('--run-dir', required=True)
    command.add_argument('--format', required=True, choices=('nul', 'json'))
    prepare = actions.add_parser('prepare-run')
    prepare.add_argument('--root', required=True)
    prepare.add_argument('--spec', required=True)
    prepare.add_argument('--owner-json', required=True)
    resource = actions.add_parser('resource-record')
    resource.add_argument('--root', required=True)
    resource.add_argument('--spec', required=True)
    resource.add_argument('--run-dir', required=True)
    resource.add_argument('--runtime-seconds', type=float)
    resource.add_argument('--peak-gpu-memory-bytes', type=int)
    ready = actions.add_parser('phase-ready')
    ready.add_argument('--root', required=True)
    ready.add_argument('--phase', required=True, choices=('explanation',))
    gpu_claim = actions.add_parser('gpu-claim')
    gpu_claim.add_argument('--root', required=True)
    gpu_claim.add_argument('--physical-gpu', required=True, type=int)
    gpu_claim.add_argument('--owner-json', required=True)
    gpu_release = actions.add_parser('gpu-release')
    gpu_release.add_argument('--root', required=True)
    gpu_release.add_argument('--physical-gpu', required=True, type=int)
    gpu_release.add_argument('--owner-json', required=True)
    audit_run = actions.add_parser('audit-run')
    for action in ('queue-audit', 'next-audit', 'complete-audit',
                   'cancel-audit', 'audit-phase-ready'):
        control = actions.add_parser(action)
        control.add_argument('--root', required=True)
        if action != 'audit-phase-ready':
            control.add_argument('--owner-json', required=True)
        if action in ('queue-audit', 'complete-audit'):
            control.add_argument('--spec', required=True)
        if action == 'queue-audit':
            control.add_argument('--physical-gpu', required=True, type=int)
        if action in ('next-audit', 'audit-phase-ready'):
            control.add_argument('--phase', required=True,
                                 choices=('formal', 'explanation'))
    audit_run.add_argument('--root', required=True)
    audit_run.add_argument('--spec-key', required=True)
    audit_run.add_argument('--run-dir', required=True)
    summarize = actions.add_parser('summarize')
    summarize.add_argument('--records', required=True)
    summarize.add_argument('--destination', required=True)
    marker = actions.add_parser('mark')
    marker.add_argument('--root', required=True)
    marker.add_argument('--name', required=True)
    marker.add_argument('--payload-json', required=True)
    finalize = actions.add_parser('finalize')
    finalize.add_argument('--root', required=True)
    smoke_plan = actions.add_parser('smoke-plan')
    smoke_plan.add_argument('--root', required=True)
    smoke_check_parser = actions.add_parser('smoke-check')
    smoke_check_parser.add_argument('--root', required=True)
    smoke_jobs = actions.add_parser('smoke-jobs')
    smoke_jobs.add_argument('--root', required=True)
    smoke_command_parser = actions.add_parser('smoke-command')
    smoke_command_parser.add_argument('--root', required=True)
    smoke_command_parser.add_argument('--job', required=True)
    smoke_command_parser.add_argument('--run-dir', required=True)
    smoke_command_parser.add_argument(
        '--format', required=True, choices=('nul', 'json'))
    smoke_begin_parser = actions.add_parser('smoke-begin')
    smoke_begin_parser.add_argument('--root', required=True)
    smoke_begin_parser.add_argument('--job', required=True)
    smoke_begin_parser.add_argument('--physical-gpu', required=True, type=int)
    smoke_record_parser = actions.add_parser('smoke-record')
    smoke_record_parser.add_argument('--root', required=True)
    smoke_record_parser.add_argument('--job', required=True)
    smoke_record_parser.add_argument('--run-dir', required=True)
    smoke_audit = actions.add_parser('audit-smoke')
    smoke_audit.add_argument('--root', required=True)
    smoke_marker = actions.add_parser('smoke-mark')
    smoke_marker.add_argument('--root', required=True)
    smoke_marker.add_argument(
        '--name', required=True, choices=('SMOKE_FAILED', 'SMOKE_STOPPED'))
    smoke_marker.add_argument('--kind', required=True, choices=('failed', 'stopped'))
    smoke_marker.add_argument('--exit-code', required=True, type=int)
    resume = actions.add_parser('resume-probe')
    resume.add_argument('--run-dir', required=True)
    resume.add_argument(
        '--physical-gpu', required=True, type=int, choices=(0, 1))
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(argv)
    if args.action in {
            'check', 'census', 'plan', 'jobs', 'disk-status', 'claim', 'owner',
            'claim-owner', 'release-claim', 'mark-started', 'prepare-run',
            'resource-record', 'phase-ready', 'gpu-claim', 'gpu-release',
            'queue-audit', 'next-audit', 'complete-audit', 'cancel-audit',
            'audit-phase-ready',
            'audit-run', 'mark', 'finalize', 'smoke-plan', 'smoke-check',
            'smoke-jobs', 'smoke-command', 'smoke-begin', 'smoke-record',
            'audit-smoke', 'smoke-mark'}:
        root = _path(args.root, 'formal root')
    if args.action == 'check':
        plan, manifest = _load_installed_plan(root)
        _emit({
            'registry': _registry_payload(),
            'plan_sha256': _digest(plan),
            'missing_jobs_sha256': _digest(manifest),
            'root_identity': _root_identity(root),
        })
    elif args.action == 'disk-status':
        if experiment_profile() in (
                SINGLE_DATASET_PROFILE, CONTINUATION_PROFILE,
                METHOD_SHARD_PROFILE):
            from three_dataset_resource_gate import scoped_disk_status
            _emit(scoped_disk_status(root, args.requested_slots))
        else:
            from three_dataset_resource_gate import disk_status
            _emit(disk_status(root, args.requested_slots))
    elif args.action == 'census':
        _emit(_install_census(root, _load_json_file(args.declarations)))
    elif args.action == 'census-full':
        _emit(_install_full_census(args.root, args.reuse_bundle))
    elif args.action == 'census-continuation':
        _emit(_install_continuation_census(args.root, args.reuse_bundle))
    elif args.action == 'plan':
        _emit(_install_plan(root))
    elif args.action == 'jobs':
        _emit(_installed_jobs(root))
    elif args.action == 'claim':
        plan, _ = _load_installed_plan(root)
        requested = _inline_json(args.owner_json, 'claim owner JSON')
        try:
            key = claim_next(
                plan, root / 'claims', args.phase, requested,
                pipeline=args.pipeline)
        except PipelineBackpressure:
            return 75
        if args.format == 'text':
            print('' if key is None else key)
        else:
            _emit(None if key is None else installed_claim_owner(root, key))
    elif args.action == 'owner':
        _emit(owner_for(
            root, args.phase, args.launcher_token, args.worker_role,
            args.pid, args.pgid))
    elif args.action == 'claim-owner':
        _emit(installed_claim_owner(root, args.spec))
    elif args.action == 'release-claim':
        owner = _inline_json(args.owner_json, 'release owner JSON')
        if owner['job'] != args.spec:
            raise ValueError('release owner spec differs')
        release_prelaunch_claim(
            root / 'claims' / _claim_name(args.spec), owner)
        _emit({'released': args.spec})
    elif args.action == 'mark-started':
        owner = _inline_json(args.owner_json, 'started owner JSON')
        if owner['job'] != args.spec:
            raise ValueError('started owner spec differs')
        plan, _ = _load_installed_plan(root)
        command = command_for_run(
            args.spec, _path(args.run_dir, 'started run directory'))
        _emit(mark_claim_started(
            root / 'claims' / _claim_name(args.spec), owner,
            _path(args.run_dir, 'started run directory'), plan,
            _digest(command)))
    elif args.action == 'command':
        command = command_for_run(
            args.spec, _path(args.run_dir, 'command run directory'))
        if args.format == 'nul':
            sys.stdout.buffer.write(
                b''.join(token.encode() + b'\0' for token in command))
        else:
            _emit(command)
    elif args.action == 'prepare-run':
        print(prepare_run(
            root, args.spec,
            _inline_json(args.owner_json, 'prepare owner JSON')))
    elif args.action == 'resource-record':
        _emit(resource_record(
            root, args.spec, _path(args.run_dir, 'resource run directory'),
            runtime_seconds=args.runtime_seconds,
            peak_gpu_memory_bytes=args.peak_gpu_memory_bytes,
        ))
    elif args.action == 'phase-ready':
        ready = phase_ready(root, args.phase)
        print('true' if ready else 'false')
        if not ready:
            return 3
    elif args.action == 'gpu-claim':
        claimed = claim_gpu(
            root, args.physical_gpu,
            _inline_json(args.owner_json, 'GPU owner JSON'))
        print('true' if claimed else 'false')
        if not claimed:
            return 4
    elif args.action == 'gpu-release':
        release_gpu(
            root, args.physical_gpu,
            _inline_json(args.owner_json, 'GPU owner JSON'))
        _emit({'released_gpu': args.physical_gpu})
    elif args.action == 'audit-run':
        _emit(install_completed_record(
            root, args.spec_key, _path(args.run_dir, 'completed run')))
    elif args.action == 'queue-audit':
        _emit(queue_audit(root, args.spec, args.physical_gpu,
                         _inline_json(args.owner_json, 'producer owner JSON')))
    elif args.action == 'next-audit':
        _emit(next_audit(root, args.phase,
                        _inline_json(args.owner_json, 'auditor owner JSON')))
    elif args.action == 'complete-audit':
        complete_audit(root, args.spec,
                       _inline_json(args.owner_json, 'auditor owner JSON'))
        _emit({'completed_audit': args.spec})
    elif args.action == 'cancel-audit':
        cancel_audit(root, _inline_json(args.owner_json, 'auditor owner JSON'))
        _emit({'cancelled_audit': True})
    elif args.action == 'audit-phase-ready':
        _emit(audit_phase_ready(root, args.phase))
    elif args.action == 'summarize':
        _emit(install_tables(
            _load_records_file(args.records),
            _path(args.destination, 'table destination'),
        ))
    elif args.action == 'mark':
        _emit(install_marker(
            root, args.name,
            _inline_json(args.payload_json, 'formal marker JSON')))
    elif args.action == 'finalize':
        _emit(finalize_installed(root))
    elif args.action == 'smoke-plan':
        _emit(plan_smoke(root))
    elif args.action == 'smoke-check':
        _emit(smoke_check(root))
    elif args.action == 'smoke-jobs':
        for job in _validate_smoke_plan(root)['jobs']:
            print(job['job_id'])
    elif args.action == 'smoke-command':
        command = smoke_command(root, args.job, args.run_dir)
        if args.format == 'nul':
            sys.stdout.buffer.write(
                b''.join(token.encode() + b'\0' for token in command))
        else:
            _emit(list(command))
    elif args.action == 'smoke-begin':
        _emit(smoke_begin(root, args.job, args.physical_gpu))
    elif args.action == 'smoke-record':
        _emit(install_smoke_record(root, args.job, args.run_dir))
    elif args.action == 'audit-smoke':
        _emit(audit_smoke(root))
    elif args.action == 'smoke-mark':
        _emit(install_smoke_marker(
            root, args.name, args.kind, args.exit_code))
    elif args.action == 'resume-probe':
        _emit(resume_probe(_path(
            args.run_dir, 'completed generated smoke run'), args.physical_gpu))
    else:
        raise AssertionError('unreachable action')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
