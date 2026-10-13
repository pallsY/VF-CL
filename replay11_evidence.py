"""Read-only, portable checks for replay-11 result identities."""

import hashlib
import ast
import json
import math
import os
import re
import subprocess
import tempfile
import types
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import replay11_matrix as matrix
from adaptive_consolidation_audit import FORMAL_SOURCE_FILES


PRODUCER_COMMIT = '99e81b28e0544e182f777a34d7e7a786a1c3b920'
CORRECTED_COMMIT = '467611a603c9604ad57861920495a3ad05612037'
FROZEN_RESULTS_ROOT = Path(
    '/brf/users/yangxx/VFCL-results/replay11-paper-v3-20261013/full-3080')
FROZEN_DATA_ROOT = Path('/brf/users/yangxx/VF-CL/data')
PRODUCER_COMMITS = {PRODUCER_COMMIT, CORRECTED_COMMIT}
LEGACY_REUSE_METHODS = frozenset({
    'balanced_contrast', 'er', 'der_pp', 'er_ace', 'adagauss',
})
TASKS = {'cifar100': 10, 'isolet': 13, 'upmc_food101': 10}
PRODUCER_ENTRYPOINTS = {
    '/home/chase/Yangxx/VF-CL-worktrees/github-target-clone/main.py',
    '/home/c3080/YangXiaoXiang/VF-CL-worktrees/replay11-clip-test/main.py',
    '/tmp/vfcl-replay11-code99/main.py',
}
CORRECTED_ENTRYPOINTS = {
    '/tmp/vfcl-replay11-paper-v3-full/main.py',
}


def _file(path):
    path = Path(path)
    if (any(part.is_symlink() for part in (path, *path.parents))
            or not path.is_file() or not path.stat().st_size):
        raise ValueError(f'missing regular evidence file: {path}')
    return path


def _json(path):
    value = json.loads(_file(path).read_text(encoding='utf-8'))
    if type(value) is not dict:
        raise ValueError(f'evidence JSON is not an object: {path}')
    return value


def _verify_authoritative_data(root, expected):
    root = Path(root)
    actual = {}
    for relative, digest in expected.items():
        actual[relative] = matrix.sha256_file(_file(root / relative))
        if actual[relative] != digest:
            raise ValueError(f'authoritative data hash differs: {relative}')
    return actual


def _canonical_sha(value):
    content = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False,
        separators=(',', ': ')) + '\n'
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


@lru_cache(maxsize=2)
def _producer_source_sha256(commit=PRODUCER_COMMIT):
    root = str(Path(matrix.__file__).parent)
    return {
        name: hashlib.sha256(subprocess.check_output([
            'git', '-C', root, 'show', f'{commit}:{name}',
        ])).hexdigest()
        for name in FORMAL_SOURCE_FILES
    }


@lru_cache(maxsize=2)
def _committed_matrix(commit):
    root = str(Path(matrix.__file__).parent)
    source = subprocess.check_output([
        'git', '-C', root, 'show', f'{commit}:replay11_matrix.py',
    ])
    module = types.ModuleType(f'replay11_matrix_{commit[:12]}')
    module.__file__ = str(Path(root) / 'replay11_matrix.py')
    module.__source_sha256__ = hashlib.sha256(source).hexdigest()
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


def _normalized_command(command):
    options = _options(command)
    if 'vector_npz' in options:
        root = options['data_path'].replace('\\', '/').rstrip('/') + '/'
        vector = options['vector_npz'].replace('\\', '/')
        if not vector.startswith(root):
            raise ValueError('vector cache is outside data root')
        options['vector_npz'] = '<data_path>/' + vector[len(root):]
    for key in ('results_dir', 'device', 'data_path'):
        if key in options:
            options[key] = f'<{key}>'
    return options


def equivalence_certificate(job, command):
    """Attest an eligible legacy method against corrected formal code."""
    dataset, method, seed = job
    if job not in matrix.jobs() or method not in LEGACY_REUSE_METHODS:
        raise ValueError('legacy method is not eligible for reuse')
    actual = _normalized_command(command)
    normalized = []
    for commit in (PRODUCER_COMMIT, CORRECTED_COMMIT):
        generator = _committed_matrix(commit)
        expected = generator.command_for(
            job, Path('/audit/results'), python=Path('/audit/python'),
            worktree=Path('/audit/source'), device='audit-device',
            data_root=Path('/audit/data'))
        normalized.append(_normalized_command(expected))
    if actual != normalized[0] or normalized[0] != normalized[1]:
        raise ValueError('formal commands are not equivalent')
    method_sources = {f'cl_methods/{method}.py'}
    if method == 'balanced_contrast':
        method_sources.add('cl_methods/er.py')
    scope = [name for name in FORMAL_SOURCE_FILES
             if not name.startswith('cl_methods/')
             or name == 'cl_methods/__init__.py'
             or name in method_sources]
    old = _producer_source_sha256(PRODUCER_COMMIT)
    new = _producer_source_sha256(CORRECTED_COMMIT)
    if any(old[name] != new[name] for name in scope):
        raise ValueError('executed source blobs are not equivalent')
    return {
        'equivalent': True, 'method': method,
        'old_commit': PRODUCER_COMMIT,
        'corrected_commit': CORRECTED_COMMIT,
        'command_generator_sha256': {
            commit: _committed_matrix(commit).__source_sha256__
            for commit in (PRODUCER_COMMIT, CORRECTED_COMMIT)
        },
        'normalized_command': normalized[0],
        'normalized_command_sha256': _canonical_sha(normalized[0]),
        'source_sha256': {name: old[name] for name in scope},
    }


def _options(command):
    if type(command) is not list or len(command) < 4:
        raise ValueError('invalid job command')
    result = {}
    index = 2
    while index < len(command):
        option = command[index]
        if type(option) is not str or not option.startswith('--'):
            raise ValueError('invalid job option')
        name = option[2:]
        if name in result:
            raise ValueError('duplicate job option')
        if name == 'cosine_head':
            result[name] = '1'
            index += 1
        else:
            if index + 1 >= len(command):
                raise ValueError('missing job option value')
            result[name] = str(command[index + 1])
            index += 2
    return result


@lru_cache(maxsize=99)
def _expected_certificate(job):
    command = _committed_matrix(PRODUCER_COMMIT).command_for(
        job, Path('/audit/results'), python=Path('/audit/python'),
        worktree=Path('/audit/source'), device='audit-device',
        data_root=Path('/audit/data'))
    return equivalence_certificate(job, command)


def _expected_protocol(config, provenance):
    protocol = {
        key: config.get(key) for key in (
            'seed', 'data', 'cl_method', 'num_tasks', 'num_parties',
            'head_consolidation_enabled', 'head_consolidation_mode',
        )
    }
    protocol.update(
        formal_deferred_evaluation=True,
        source_provenance=provenance)
    if config['cl_method'] == 'er_ace':
        protocol.update({
            'num_classes': config['num_classes'],
            'er_ace_buffer_size': config['er_ace_buffer_size'],
            'er_ace_batch': config['er_ace_batch'],
        })
    if bool(config.get('bic_enabled')):
        protocol.update({
            'bic_enabled': True,
            'bic_fit_mode': config['bic_fit_mode'],
            'bic_lr': float(config['bic_lr']),
            'bic_steps': int(config['bic_steps']),
            'bic_per_class': int(config['bic_per_class']),
            'lambda_validation_enabled':
                bool(config['lambda_validation_enabled']),
            'lambda_validation_per_class':
                int(config['lambda_validation_per_class']),
            'lambda_validation_split_seed':
                int(config['lambda_validation_split_seed']),
        })
    return protocol


def _validate_cache_record(record):
    required = {
        'input_sha256', 'label_sha256', 'input_shape', 'label_shape',
        'input_dtype', 'label_dtype',
    }
    if (type(record) is not dict
            or set(record) != {
                'batch_count', 'sample_count', 'batches'}
            or type(record['batch_count']) is not int
            or type(record['sample_count']) is not int
            or type(record['batches']) is not list
            or record['batch_count'] != len(record['batches'])
            or record['batch_count'] < 1
            or record['sample_count'] < 1):
        raise ValueError('formal cache identity schema differs')
    total = 0
    for batch in record['batches']:
        if (type(batch) is not dict or set(batch) != required
                or any(type(batch[key]) is not str
                       or re.fullmatch(r'[0-9a-f]{64}', batch[key]) is None
                       for key in ('input_sha256', 'label_sha256'))
                or type(batch['input_shape']) is not list
                or type(batch['label_shape']) is not list
                or not batch['input_shape'] or len(batch['label_shape']) != 1
                or batch['input_shape'][0] != batch['label_shape'][0]
                or type(batch['label_shape'][0]) is not int
                or batch['label_shape'][0] < 1):
            raise ValueError('formal cache batch identity differs')
        total += batch['label_shape'][0]
    if total != record['sample_count']:
        raise ValueError('formal cache sample count differs')


def _verify_data_flow(run, expected_test, expected_tasks):
    try:
        path = _file(run / 'data_flow_audit.jsonl')
    except ValueError as error:
        raise ValueError('training data-flow evidence is missing') from error
    digest = hashlib.sha256()
    train_groups = set()
    test_count = 0
    formal_started = False
    try:
        with path.open('rb') as handle:
            for line in handle:
                digest.update(line)
                record = json.loads(line)
                if type(record) is not dict:
                    raise ValueError('data-flow record is not an object')
                if 'split' in record:
                    formal_started = True
                    if record.get('split') == 'test':
                        test_count += 1
                        if record != expected_test:
                            raise ValueError('data-flow test access differs')
                    elif record.get('split') not in ('validation', 'calibration'):
                        raise ValueError('data-flow split is invalid')
                    elif test_count:
                        raise ValueError('data-flow access follows test')
                    continue
                if formal_started:
                    raise ValueError('data-flow training follows formal access')
                key = ast.literal_eval(record.get('loader_key', ''))
                if (type(key) is not tuple or len(key) != 2
                        or key[0] != 'train' or type(key[1]) is not tuple
                        or not key[1] or type(record.get('batch')) is not int
                        or type(record.get('loader_iteration')) is not int
                        or type(record.get('indices')) is not list
                        or not record['indices']
                        or any(type(index) is not int or index < 0
                               for index in record['indices'])
                        or record.get('dtype') != 'torch.float32'
                        or type(record.get('shape')) is not list
                        or not record['shape']
                        or record['shape'][0] != len(record['indices'])
                        or re.fullmatch(r'[0-9a-f]{64}',
                                        record.get('batch_sha256', '')) is None):
                    raise ValueError('training data-flow batch is invalid')
                train_groups.add(key[1])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError,
            SyntaxError, TypeError) as error:
        raise ValueError('training data-flow is unreadable') from error
    if len(train_groups) != expected_tasks or test_count != 1:
        raise ValueError('training data-flow task or test count differs')
    return digest.hexdigest()


def _verify_training_flow_strict(run, config, data_root, expected_tasks):
    """Rebuild producer loaders from authoritative data and compare each epoch."""
    from data_utils import TaskManager, VFLDataset
    from determinism import tensor_sha256

    if any(not (Path(data_root) / relative).is_file()
           for relative in matrix.expected_data(
               'cifar100' if config['data'] == 'cifar100' else
               'upmc_food101' if config['num_classes'] == 101 else 'isolet')):
        raise ValueError('training data-flow reconstruction lacks data files')
    with tempfile.TemporaryDirectory() as scratch:
        args = SimpleNamespace(**config)
        args.data_path = str(data_root)
        args.output_dir = scratch
        if config['data'] == 'tabvfl':
            relative = (matrix.UPMC_FOUR_CACHE
                        if config['num_classes'] == 101
                        else 'isolet/isolet_vfl.npz')
            args.vector_npz = str(Path(data_root) / relative)
        try:
            dataset = VFLDataset(args)
            tasks = TaskManager(args)
            if len(tasks.task_classes) != expected_tasks:
                raise ValueError('task count differs')
            batch_count = 0
            with _file(run / 'data_flow_audit.jsonl').open('r',
                                                            encoding='utf-8') as handle:
                records = (json.loads(line) for line in handle if line.strip())
                for task_id in range(expected_tasks):
                    tasks.advance_task(task_id)
                    classes = (tasks.get_effective_classes()
                               if args.replay_mode == 'full'
                               else tasks.task_classes[task_id])
                    loader = dataset.get_train_loader(classes)
                    if loader.audit_sampler is None:
                        raise ValueError('deterministic audit sampler is absent')
                    for epoch in range(args.epochs_per_task):
                        for batch, (batch_x, _batch_y) in enumerate(loader):
                            actual = next(records)
                            order = loader.audit_sampler.epoch_orders[-1]
                            start = batch * loader.batch_size
                            expected = {
                                'loader_key': loader.audit_key,
                                'loader_iteration': epoch,
                                'batch': batch,
                                'indices': order[start:start + len(batch_x)],
                                'dtype': str(batch_x.dtype),
                                'shape': list(batch_x.shape),
                            }
                            if (type(actual) is not dict
                                    or any(actual.get(key) != value
                                           for key, value in expected.items())
                                    or actual.get('batch_sha256')
                                    != tensor_sha256(batch_x)):
                                raise ValueError('batch identity differs')
                            batch_count += 1
                remaining = list(records)
                if (not remaining
                        or any(type(record) is not dict
                               or 'split' not in record
                               for record in remaining)):
                    raise ValueError('training batch count differs')
        except (OSError, StopIteration, KeyError, TypeError,
                json.JSONDecodeError, ValueError) as error:
            raise ValueError('training data-flow reconstruction differs') from error
    return {'batches': batch_count, 'batch_bytes_replayed': True}


def _verify_checkpoint_method_state(path, config, expected_tasks):
    """Restricted decode and method-specific state contract for final weights."""
    import torch
    from adaptive_consolidation_audit import _safe_torch_load
    from runner import (_decode_checkpoint_value,
                        _valid_method_checkpoint_state,
                        _valid_tracker_checkpoint_state)

    base = {'per_class_size', 'data', 'seen_count'}
    schemas = {
        'er': base, 'mir': base, 'lpr': base,
        'balanced_contrast': base | {'task_number'},
        'ocar': base | {'factor_schema_version', 'tau', 'steps', 'factors'},
        'alsar': base | {'usage', 'step', 'fisher', 'top_fisher',
                         'class_similarity', 'last_gradient_mean'},
        'icarl': base | {'total_budget', 'new_classes', 'old_classes',
                         'teacher_bottoms', 'teacher_top'},
        'hpcr': {'per_class_size', 'memory', 'seen_count',
                 'seen_classes', 'step'},
        'der_pp': {'buffer_size', 'num_seen', 'examples',
                   'labels', 'logits'},
        'er_ace': {'buffer_size', 'num_seen', 'examples', 'labels'},
        'adagauss': {'gaussians', 'adapters'},
    }
    method = config['cl_method']
    try:
        checkpoint = _safe_torch_load(path)
        state = _decode_checkpoint_value(checkpoint['cl_state'])
        protocol = checkpoint['protocol']
        if config.get('custom_tasks'):
            task_classes = [[int(value) for value in group.split(',')]
                            for group in config['custom_tasks'].split('|')]
        else:
            per_task = config['classes_per_task']
            task_classes = [list(range(i * per_task, (i + 1) * per_task))
                            for i in range(expected_tasks)]
        bottoms = checkpoint['trainer_state']['bottoms']
        top = checkpoint['trainer_state']['top_model']
        if (checkpoint.get('schema_version') != 4
                or checkpoint.get('task_id') != expected_tasks - 1
                or checkpoint.get('event_idx') != expected_tasks - 1
                or checkpoint.get('step')
                != f'event_{expected_tasks - 1}_CIL'
                or checkpoint.get('seen_task_classes')
                != dict(enumerate(task_classes))
                or protocol != _expected_protocol(
                    config, protocol.get('source_provenance'))
                or type(bottoms) is not list or len(bottoms) != 4
                or any(type(bottom) is not dict or not bottom
                       for bottom in bottoms)
                or type(top) is not dict or not top
                or method not in schemas or type(state) is not dict
                or set(state) != schemas[method]
                or not _valid_method_checkpoint_state(
                    method, state, False, 4)
                or not _valid_tracker_checkpoint_state(
                    checkpoint['tracker_state'], protocol)):
            raise ValueError('checkpoint schema or method fields differ')
        if 'per_class_size' in state:
            memory = state.get('data', state.get('memory'))
            if (type(state['per_class_size']) is not int
                    or state['per_class_size'] < 1
                    or type(memory) is not dict
                    or not memory):
                raise ValueError('method exemplar memory differs')
        def check_tensors(value, path=()):
            if isinstance(value, torch.Tensor):
                if value.is_floating_point():
                    hpcr_masked_logits = (method == 'hpcr'
                                          and len(path) == 4
                                          and path[0] == 'memory'
                                          and path[-1] == 2)
                    invalid = (torch.isnan(value) | torch.isposinf(value)
                               if hpcr_masked_logits
                               else ~torch.isfinite(value))
                    if bool(invalid.any()):
                        raise ValueError('checkpoint tensor is invalid')
            elif type(value) is dict:
                for key, item in value.items():
                    check_tensors(item, path + (key,))
            elif type(value) in (list, tuple):
                for index, item in enumerate(value):
                    check_tensors(item, path + (index,))
        check_tensors(bottoms)
        check_tensors(top)
        check_tensors(state)
    except Exception as error:
        raise ValueError('checkpoint method state is invalid') from error
    return {'method': method, 'state_keys': sorted(state),
            'task_id': checkpoint['task_id'], 'parties': len(bottoms)}


def _verify_transaction(run, frozen, published, provenance, result):
    names = (
        'PENDING', 'CONSUMING', 'COMPLETE', 'SEALED', 'PUBLISHING')
    markers = {
        name: _json(run / f'FORMAL_EVALUATION_{name}.json')
        for name in names
    }
    identity = frozen.get('identity')
    bic = identity.get('protocol', {}).get('bic_enabled') is True
    base = {'schema_version', 'status', 'transaction_sha256'}
    schemas = {
        'PENDING': base | {'identity', 'freeze'},
        'CONSUMING': base | {'identity', 'freeze', 'cache_identity'},
        'COMPLETE': base | {
            'identity', 'freeze', 'cache_identity',
            'evaluation', 'evaluation_sha256',
        } | ({'calibration_cache_identity', 'bic', 'bic_sha256'}
             if bic else set()),
        'SEALED': base | {
            'identity_sha256', 'protocol_sha256', 'freeze',
            'cache_identity', 'source_provenance',
            'pending', 'consuming', 'complete',
        },
        'PUBLISHING': base | {
            'evaluation_sha256', 'results_sha256',
            'seal', 'source_provenance',
        },
        'PUBLISHED': base | {
            'evaluation_sha256', 'results_sha256', 'seal',
            'source_provenance', 'checkpoint', 'results',
        },
    }
    if (set(frozen) != {'schema_version', 'status', 'identity'}
            or any(set(marker) != schemas[name]
                   for name, marker in (
                       *markers.items(), ('PUBLISHED', published)))):
        raise ValueError('formal transaction schema differs')
    freeze_hash = matrix.sha256_file(
        _file(run / 'FORMAL_STATE_FROZEN.json'))
    transaction = markers['PENDING'].get('transaction_sha256')
    evaluation = markers['COMPLETE'].get('evaluation_sha256')
    if not all(type(value) is str and len(value) == 64
               for value in (transaction, evaluation)):
        raise ValueError('formal transaction hash is missing')
    if transaction != _canonical_sha({
            'identity': frozen['identity'],
            'freeze': markers['PENDING']['freeze']}):
        raise ValueError('formal transaction digest differs')
    complete = markers['COMPLETE']
    if (not isinstance(complete.get('evaluation'), dict)
            or evaluation != _canonical_sha(complete['evaluation'])
            or (bic and complete.get('bic_sha256')
                != _canonical_sha(complete['bic']))):
        raise ValueError('formal evaluation digest differs')
    if (complete['evaluation'].get('task_acc_history')
            != result.get('task_acc_history')
            or any(result.get(key) != value
                   for key, value in complete['evaluation'].items())):
        raise ValueError('sealed evaluation history or payload differs from results')
    if (markers['CONSUMING']['cache_identity']
            != complete['cache_identity']
            or markers['SEALED']['cache_identity']
            != complete['cache_identity']
            or markers['SEALED']['identity_sha256']
            != _canonical_sha(identity)
            or markers['SEALED']['protocol_sha256']
            != identity.get('protocol_sha256')):
        raise ValueError('formal cache or seal identity differs')
    cache = complete['cache_identity']
    if bic:
        if (type(cache) is not dict
                or set(cache) != {'test', 'calibration'}
                or cache['calibration']
                != complete['calibration_cache_identity']):
            raise ValueError('formal BiC cache identity differs')
        _validate_cache_record(cache['test'])
        calibration = cache['calibration']
        if (type(calibration) is not dict
                or set(calibration) != {'split', 'phase', 'cache'}
                or calibration['split'] != 'calibration'
                or calibration['phase']
                != 'final_bic_calibration_post_freeze'):
            raise ValueError('formal calibration cache identity differs')
        _validate_cache_record(calibration['cache'])
    else:
        _validate_cache_record(cache)
    for name, marker in (*markers.items(), ('PUBLISHED', published)):
        if (marker.get('schema_version') != 1
                or marker.get('status') != name.lower()
                or marker.get('transaction_sha256') != transaction):
            raise ValueError('formal transaction stage differs')
    for name in ('PENDING', 'CONSUMING', 'COMPLETE'):
        marker = markers[name]
        if marker.get('identity') != frozen.get('identity'):
            raise ValueError('formal frozen identity differs')
    for name in ('PENDING', 'CONSUMING', 'COMPLETE', 'SEALED'):
        freeze = markers[name].get('freeze', {})
        if (freeze.get('path') != 'FORMAL_STATE_FROZEN.json'
                or freeze.get('sha256') != freeze_hash):
            raise ValueError('formal freeze link differs')
    sealed = markers['SEALED']
    for name in ('PENDING', 'CONSUMING', 'COMPLETE'):
        link = sealed.get(name.lower(), {})
        path = f'FORMAL_EVALUATION_{name}.json'
        if (link.get('path') != path
                or link.get('sha256')
                != matrix.sha256_file(_file(run / path))):
            raise ValueError('formal seal link differs')
    seal_hash = matrix.sha256_file(
        _file(run / 'FORMAL_EVALUATION_SEALED.json'))
    publishing = markers['PUBLISHING']
    for marker in (publishing, published):
        seal = marker.get('seal', {})
        if (seal.get('path') != 'FORMAL_EVALUATION_SEALED.json'
                or seal.get('sha256') != seal_hash
                or marker.get('evaluation_sha256') != evaluation
                or marker.get('source_provenance') != provenance):
            raise ValueError('formal publication link differs')
    if (sealed.get('source_provenance') != provenance
            or publishing.get('results_sha256')
            != published.get('results_sha256')
            or published.get('results_sha256') != _canonical_sha(result)):
        raise ValueError('formal publication identity differs')


def verify_local_job(job_dir, *, data_root=None, archive_manifest=None):
    """Check one completed job without relying on inode identity."""
    job_dir = Path(job_dir)
    job = (job_dir.parent.parent.name, job_dir.parent.name,
           int(job_dir.name))
    if job_dir.is_symlink() or job not in matrix.jobs():
        raise ValueError('unregistered job directory')
    dataset, method, seed = job
    plan = _json(job_dir / 'JOB_PLAN.json')
    success = _json(job_dir / 'JOB_SUCCESS.json')
    if (plan.get('pilot_deferred', False) is not False
            or plan.get('pilot_bic', False) is not False
            or success.get('pilot_deferred', False) is not False
            or success.get('pilot_bic', False) is not False):
        raise ValueError('pilot result is excluded from formal matrix')
    producer_commit = plan.get('code_commit')
    if (success.get('smoke') is not False
            or any(success.get(key) != value for key, value in plan.items())
            or (plan.get('dataset'), plan.get('method'), plan.get('seed'))
            != job
            or producer_commit not in PRODUCER_COMMITS
            or plan.get('data_hashes') != matrix.expected_data(dataset)):
        raise ValueError('job plan or authority differs')
    verified_data = (_verify_authoritative_data(data_root, plan['data_hashes'])
                     if data_root is not None else None)
    options = _options(plan.get('command'))
    entrypoint = plan['command'][1].replace('\\', '/')
    allowed_entrypoints = (PRODUCER_ENTRYPOINTS
                           if producer_commit == PRODUCER_COMMIT
                           else CORRECTED_ENTRYPOINTS)
    if entrypoint not in allowed_entrypoints:
        raise ValueError('training entrypoint differs')
    if any(options.get(key) != expected for key, expected in {
            'cl_method': method, 'seed': str(seed),
            'num_tasks': str(TASKS[dataset]), 'num_parties': '4',
            'formal_deferred_evaluation': '1',
    }.items()):
        raise ValueError('job command differs')
    if dataset in ('isolet', 'upmc_food101'):
        relative = (matrix.UPMC_FOUR_CACHE if dataset == 'upmc_food101'
                    else 'isolet/isolet_vfl.npz')
        expected_vector = (
            options['data_path'].replace('\\', '/').rstrip('/')
            + '/' + relative)
        if options.get('vector_npz', '').replace('\\', '/') != expected_vector:
            raise ValueError('vector cache path differs')
    expected_options = _options(_committed_matrix(producer_commit).command_for(
        job, Path(options['results_dir']),
        python=Path(plan['command'][0]),
        worktree=Path(plan['command'][1]).parent,
        device=options['device'],
        data_root=Path(options['data_path'])))
    if (set(options) != set(expected_options)
            or any(options[name] != expected_options[name]
                   for name in options
                   if name not in {
                       'data_path', 'vector_npz', 'results_dir', 'device',
                   })):
        raise ValueError('scientific training protocol differs')

    run = job_dir / 'run'
    if run.is_symlink() or not run.is_dir():
        raise ValueError('missing run directory')
    config = _json(run / 'config.json')
    result = _json(run / 'results.json')
    if any(config.get(key) != expected for key, expected in {
            'cl_method': method, 'seed': seed,
            'num_tasks': TASKS[dataset], 'num_parties': 4,
            'formal_deferred_evaluation': True,
    }.items()):
        raise ValueError('run config differs')
    for name, token in options.items():
        if name not in config:
            raise ValueError(f'run config lacks command option: {name}')
        if name == 'unlearn_after_tasks':
            encoded = ','.join(map(str, config[name]))
        elif name == 'unlearn_classes':
            encoded = ','.join(
                str(class_id) for group in config[name]
                for class_id in group)
        elif name == 'custom_tasks' and isinstance(config[name], str):
            encoded = config[name]
        elif name == 'cosine_head':
            encoded = '1' if config[name] is True else '0'
        elif name in matrix.registry.OPTION_SCHEMA:
            encoded = matrix.registry._option_token(name, config[name])
        else:
            encoded = str(config[name])
        if encoded != token:
            raise ValueError(f'run config differs from command: {name}')
    if any(result.get('config', {}).get(key) != config[key]
           for key in ('cl_method', 'seed', 'num_tasks')):
        raise ValueError('result config differs')
    if (dataset == 'upmc_food101' and not str(
            config.get('vector_npz', '')).endswith(matrix.UPMC_FOUR_CACHE)):
        raise ValueError('UPMC four-party cache differs')

    checkpoint = _file(run / 'checkpoints' / 'formal_final.pt')
    results_file = _file(run / 'results.json')
    if (matrix.sha256_file(checkpoint) != success.get('checkpoint_sha256')
            or matrix.sha256_file(results_file)
            != success.get('results_sha256')):
        raise ValueError('published result bytes differ')
    frozen = _json(run / 'FORMAL_STATE_FROZEN.json')
    published = _json(run / 'FORMAL_EVALUATION_PUBLISHED.json')
    access = _json(run / 'formal_access' / 'test.consumed.json')
    frozen_snapshots = frozen.get('identity', {}).get('snapshots')
    if type(frozen_snapshots) is not list or len(frozen_snapshots) != TASKS[dataset]:
        raise ValueError('frozen snapshot inventory differs')
    for index, entry in enumerate(frozen_snapshots):
        for key, expected_path in (
                ('file', f'formal_snapshots/event_{index}_CIL.pt'),
                ('source_file', f'checkpoints/event_{index}_CIL.pt')):
            record = entry.get(key, {})
            if (record.get('path') != expected_path
                    or matrix.sha256_file(_file(run / expected_path))
                    != record.get('sha256')):
                raise ValueError('frozen checkpoint bytes differ')
    provenance = result.get('source_provenance')
    _verify_transaction(run, frozen, published, provenance, result)
    source_hashes = provenance.get('source_sha256') if type(provenance) is dict else None
    expected_protocol = _expected_protocol(config, provenance)
    if (frozen['identity'].get('protocol') != expected_protocol
            or frozen['identity'].get('protocol_sha256')
            != _canonical_sha(expected_protocol)):
        raise ValueError('frozen scientific protocol differs')
    access_record = access.get('access')
    data_flow_sha256 = _verify_data_flow(run, access_record, TASKS[dataset])
    expected_classes = list(range(config['num_classes']))
    if (set(access) != {'schema_version', 'status', 'access'}
            or access.get('schema_version') != 1
            or type(access_record) is not dict
            or set(access_record) != {
                'event', 'loader_key', 'split', 'phase',
                'event_idx', 'task_id', 'timeline_step', 'classes',
            }
            or access_record.get('event') != 'first_iteration'
            or access_record.get('classes') != expected_classes
            or access_record.get('task_id') != TASKS[dataset] - 1
            or access_record.get('timeline_step')
            != f'event_{TASKS[dataset] - 1}_CIL'
            or access_record.get('loader_key')
            != repr(('test', tuple(expected_classes)))):
        raise ValueError('formal test access identity differs')
    if (frozen.get('status') != 'frozen'
            or frozen.get('identity', {}).get('source_commit')
            != producer_commit
            or published.get('status') != 'published'
            or published.get('source_provenance') != provenance
            or type(provenance) is not dict
            or provenance.get('schema_version') != 1
            or provenance.get('source_commit') != producer_commit
            or type(source_hashes) is not dict
            or source_hashes != _producer_source_sha256(producer_commit)
            or frozen['identity'].get('source_provenance') != provenance
            or published.get('checkpoint', {}).get('path')
            != 'checkpoints/formal_final.pt'
            or published['checkpoint'].get('sha256')
            != success['checkpoint_sha256']
            or published.get('results', {}).get('path') != 'results.json'
            or published['results'].get('sha256')
            != success['results_sha256']
            or access.get('status') != 'consumed'
            or access.get('access', {}).get('split') != 'test'
            or access['access'].get('phase') != 'final_test_post_install'
            or access['access'].get('event_idx') != TASKS[dataset] - 1):
        raise ValueError('formal transaction differs')
    snapshots = run / 'formal_snapshots'
    if (set(path.name for path in snapshots.glob('*.pt'))
            != {f'event_{i}_CIL.pt' for i in range(TASKS[dataset])}):
        raise ValueError('formal snapshot sequence differs')
    for path in snapshots.glob('*.pt'):
        _file(path)
    metrics = matrix.metrics_from_history(
        result.get('task_acc_history'), TASKS[dataset], deferred=True)
    if metrics != success.get('metrics'):
        raise ValueError('metric formula differs')
    original = _is_planned_original(job_dir)
    archive_evidence = (verify_archive_manifest(job_dir, archive_manifest)
                        if not original and archive_manifest else None)
    source_proven = (original or archive_evidence is not None
                     and archive_evidence['status'] == 'SOURCE_EQUIVALENT')
    attestation = {
        'job': f'{dataset}/{method}/{seed}',
        'code_commit': producer_commit,
        'admission': (
            'historical' if producer_commit == PRODUCER_COMMIT
            and method not in LEGACY_REUSE_METHODS else
            'SOURCE_EQUIVALENCE_UNPROVEN' if not source_proven else
            'formal' if verified_data is not None else 'EVIDENCE_INCOMPLETE'),
        'source_sha256': source_hashes,
        'command_sha256': _canonical_sha(plan['command']),
        'data_hashes': plan['data_hashes'],
        'verified_data_hashes': verified_data,
        'archive_evidence': archive_evidence,
        'data_flow_sha256': data_flow_sha256,
        'transaction_sha256': published['transaction_sha256'],
        'metrics': metrics,
        'results_sha256': success['results_sha256'],
        'checkpoint_sha256': success['checkpoint_sha256'],
    }
    if producer_commit == PRODUCER_COMMIT and method in LEGACY_REUSE_METHODS:
        attestation['reuse_certificate'] = equivalence_certificate(
            job, plan['command'])
    return attestation


def _tree_identity(root):
    rows = []
    for path in sorted(
            root.rglob('*'),
            key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(f'archive contains symlink: {path}')
        if path.is_file():
            rows.append([
                path.relative_to(root).as_posix(),
                path.stat().st_size, matrix.sha256_file(path),
            ])
    digest = hashlib.sha256(json.dumps(
        rows, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    return len(rows), sum(row[1] for row in rows), digest


def _archive_job_scope(job_dir, record, destination):
    job_dir = Path(os.path.abspath(job_dir))
    if (any(part.is_symlink() for part in (job_dir, *job_dir.parents))
            or not job_dir.is_dir()):
        raise ValueError('archive job path differs')
    job = f'{job_dir.parent.parent.name}/{job_dir.parent.name}/{job_dir.name}'
    dataset, method, seed = job.split('/')
    plan = _json(job_dir / 'JOB_PLAN.json')
    success = _json(job_dir / 'JOB_SUCCESS.json')
    planned_source = _options(plan.get('command')).get('results_dir')
    source = record.get('source')
    if type(source) is not str or not source:
        raise ValueError('archive source path differs')
    claimed_source = (Path(source) if destination == job_dir
                      else Path(source) / method / seed)
    if (not planned_source
            or os.path.normcase(os.path.abspath(claimed_source))
            != os.path.normcase(os.path.abspath(planned_source))):
        raise ValueError('archive source path differs')
    if (destination != job_dir
            and not (destination.name == dataset
                     and job_dir == destination / method / seed)):
        raise ValueError('archive scope differs')
    if (record.get('code_commit') not in PRODUCER_COMMITS
            or plan.get('code_commit') != record['code_commit']
            or success.get('code_commit') != record['code_commit']
            or ('job' in record and record['job'] != job)
            or ('dataset' in record and record['dataset'] != dataset)
            or ('method' in record and record['method'] != method)
            or ('seed' in record and record['seed'] != int(seed))
            or (isinstance(record.get('methods'), list)
                and method not in record['methods'])
            or (isinstance(record.get('seeds'), list)
                and int(seed) not in record['seeds'])
            or (isinstance(record.get('jobs'), list)
                and f'{method}/{seed}' not in record['jobs']
                and job not in record['jobs'])):
        raise ValueError('archive job identity differs')
    return job


def verify_archive_group(manifest_path, job_dirs):
    """Hash one archived tree once and bind every requested job to it."""
    record = _json(manifest_path)
    destination = Path(os.path.abspath(record['destination']))
    if (any(part.is_symlink()
            for part in (destination, *destination.parents))
            or not destination.is_dir()):
        raise ValueError('archive destination differs')
    job_dirs = list(job_dirs)
    if not job_dirs:
        raise ValueError('archive group is empty')
    jobs = [_archive_job_scope(path, record, destination)
            for path in job_dirs]
    if len(jobs) != len(set(jobs)):
        raise ValueError('duplicate archive job')
    files, size, digest = _tree_identity(destination)
    if record.get('files') != files or record.get('bytes') != size:
        raise ValueError('archive file inventory differs')
    if 'destination_tree_sha256' in record:
        if (record['destination_tree_sha256'] != digest
                or record.get('source_tree_sha256') != digest):
            raise ValueError('archive bytes differ')
    elif record.get('tree_sha256') != digest:
        raise ValueError('archive bytes differ')
    source = Path(record['source'])
    source_verified = False
    if source.is_dir() and source.resolve() != destination.resolve():
        if any(part.is_symlink() for part in (source, *source.parents)):
            raise ValueError('archive source path is a symlink')
        source_files, source_size, source_digest = _tree_identity(source)
        if ((source_files, source_size, source_digest)
                != (files, size, digest)):
            raise ValueError('archive source and destination bytes differ')
        source_verified = True
    return [{
        'job': job, 'code_commit': record['code_commit'],
        'files': files, 'bytes': size, 'tree_sha256': digest,
        'source_tree_sha256': digest if source_verified else None,
        'status': ('SOURCE_EQUIVALENT' if source_verified
                   else 'SOURCE_EQUIVALENCE_UNPROVEN'),
    } for job in jobs]


def verify_archive_manifest(job_dir, manifest_path):
    """Verify one archived job, reusing the group implementation."""
    return verify_archive_group(manifest_path, [job_dir])[0]


def _is_planned_original(job_dir):
    plan = _json(Path(job_dir) / 'JOB_PLAN.json')
    options = _options(plan.get('command'))
    return (os.path.normcase(os.path.abspath(job_dir))
            == os.path.normcase(os.path.abspath(options['results_dir'])))


def _same_path(left, right):
    return (os.path.normcase(os.path.abspath(left))
            == os.path.normcase(os.path.abspath(right)))


def verify_pinned_job(job_dir, *, producer_commit, results_root, data_root):
    """Verify a formal original against caller-frozen external locators."""
    if producer_commit != CORRECTED_COMMIT:
        raise ValueError('formal producer commit differs from frozen v3 commit')
    job_dir = Path(job_dir)
    job = (job_dir.parent.parent.name, job_dir.parent.name, int(job_dir.name))
    if job not in matrix.jobs():
        raise ValueError('unregistered formal job')
    expected = Path(results_root) / job[0] / job[1] / str(job[2])
    if (not _same_path(job_dir, expected)
            or any(part.is_symlink() for part in (job_dir, *job_dir.parents))):
        raise ValueError('pinned result locator differs')
    plan = _json(job_dir / 'JOB_PLAN.json')
    options = _options(plan.get('command'))
    if (plan.get('code_commit') != producer_commit
            or not _same_path(options.get('results_dir', ''), expected)):
        raise ValueError('pinned result locator or producer differs')
    if not _same_path(options.get('data_path', ''), data_root):
        raise ValueError('pinned data root differs from training command')
    record = verify_local_job(job_dir, data_root=data_root)
    if record['admission'] != 'formal' or record['code_commit'] != producer_commit:
        raise ValueError('pinned formal basic evidence is incomplete')
    config = _json(job_dir / 'run' / 'config.json')
    flow = _verify_training_flow_strict(
        job_dir / 'run', config, data_root, TASKS[job[0]])
    method_state = _verify_checkpoint_method_state(
        job_dir / 'run' / 'checkpoints' / 'formal_final.pt',
        config, TASKS[job[0]])
    record['training_flow_replay'] = flow
    record['checkpoint_method_state'] = method_state
    return record


def verify_v3_formal_job(job_dir):
    """Apply the frozen v3 NAS producer, original root, and data root."""
    return verify_pinned_job(
        job_dir, producer_commit=CORRECTED_COMMIT,
        results_root=FROZEN_RESULTS_ROOT, data_root=FROZEN_DATA_ROOT)


def merge_attestations(locations=None):
    """Recheck all 99 corrected originals under one externally pinned root."""
    if locations is not None:
        raise ValueError('catalog requires pinned job path roots')
    records = []
    for dataset, method, seed in matrix.jobs():
        records.append(verify_v3_formal_job(
            FROZEN_RESULTS_ROOT / dataset / method / str(seed)))
    return _merge_verified_attestations(records)


def _merge_verified_attestations(records):
    """Check the complete matrix after all file evidence was reopened."""
    jobs = matrix.jobs()
    expected = {f'{dataset}/{method}/{seed}'
                for dataset, method, seed in jobs}
    found = set()
    producers = {}
    for record in records:
        if type(record) is not dict:
            raise ValueError('invalid attestation')
        job = record.get('job')
        metrics = record.get('metrics')
        if (job not in expected or job in found
                or record.get('code_commit') not in PRODUCER_COMMITS
                or record.get('admission') != 'formal'
                or record.get('smoke', False) is not False
                or record.get('pilot_deferred', False) is not False
                or record.get('pilot_bic', False) is not False
                or type(metrics) is not dict
                or set(metrics) != {
                    'aa_final_cil', 'bwt_cil', 'task_il_final'}
                or any(type(value) not in (int, float)
                       or not math.isfinite(value)
                       for value in metrics.values())
                or any(type(record.get(key)) is not str
                       or re.fullmatch(
                           r'[0-9a-f]{64}', record[key]) is None
                       for key in ('results_sha256', 'checkpoint_sha256'))):
            raise ValueError('invalid or duplicate attestation')
        dataset, method, seed = job.split('/')
        if record['code_commit'] == PRODUCER_COMMIT:
            if (method not in LEGACY_REUSE_METHODS
                    or record.get('reuse_certificate')
                    != _expected_certificate((dataset, method, int(seed)))):
                raise ValueError('legacy result lacks exact equivalence certificate')
        elif record.get('reuse_certificate') is not None:
            raise ValueError('corrected result must not claim legacy reuse')
        found.add(job)
        producers[job] = record['code_commit']
    if found != expected:
        raise ValueError('replay-11 attestation matrix is incomplete')
    return {
        'jobs': len(found),
        'producer_commits': producers,
        'strict_replays': [
            f'{dataset}/{method}/{seed}'
            for dataset, method, seed in jobs if seed == 42
        ],
    }
