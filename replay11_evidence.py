"""Read-only, portable checks for replay-11 result identities."""

import hashlib
import json
import os
from pathlib import Path

import replay11_matrix as matrix


PRODUCER_COMMIT = '99e81b28e0544e182f777a34d7e7a786a1c3b920'
TASKS = {'cifar100': 10, 'isolet': 13, 'upmc_food101': 10}


class InconclusiveEvidence(Exception):
    """Evidence format cannot yet be checked without changing acceptance."""


def _file(path):
    path = Path(path)
    if (any(part.is_symlink() for part in (path, *path.parents))
            or not path.is_file() or not path.stat().st_size):
        raise ValueError(f'missing regular evidence file: {path}')
    return path


def _json(path):
    return json.loads(_file(path).read_text(encoding='utf-8'))


def _options(command):
    if (type(command) is not list or len(command) < 4
            or len(command[2:]) % 2):
        raise ValueError('invalid job command')
    pairs = command[2:]
    if any(type(pairs[i]) is not str or not pairs[i].startswith('--')
           for i in range(0, len(pairs), 2)):
        raise ValueError('invalid job option')
    result = {pairs[i][2:]: str(pairs[i + 1])
              for i in range(0, len(pairs), 2)}
    if len(result) != len(pairs) // 2:
        raise ValueError('duplicate job option')
    return result


def _verify_transaction(run, frozen, published, provenance):
    names = (
        'PENDING', 'CONSUMING', 'COMPLETE', 'SEALED', 'PUBLISHING')
    markers = {
        name: _json(run / f'FORMAL_EVALUATION_{name}.json')
        for name in names
    }
    freeze_hash = matrix.sha256_file(
        _file(run / 'FORMAL_STATE_FROZEN.json'))
    transaction = markers['PENDING'].get('transaction_sha256')
    evaluation = markers['COMPLETE'].get('evaluation_sha256')
    if not all(type(value) is str and len(value) == 64
               for value in (transaction, evaluation)):
        raise ValueError('formal transaction hash is missing')
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
            != published.get('results_sha256')):
        raise ValueError('formal publication identity differs')


def verify_local_job(job_dir):
    """Check one completed job without relying on inode identity."""
    job_dir = Path(job_dir)
    job = (job_dir.parent.parent.name, job_dir.parent.name,
           int(job_dir.name))
    if job_dir.is_symlink() or job not in matrix.jobs():
        raise ValueError('unregistered job directory')
    dataset, method, seed = job
    plan = _json(job_dir / 'JOB_PLAN.json')
    success = _json(job_dir / 'JOB_SUCCESS.json')
    if (success.get('smoke') is not False
            or any(success.get(key) != value for key, value in plan.items())
            or (plan.get('dataset'), plan.get('method'), plan.get('seed'))
            != job
            or plan.get('code_commit') != PRODUCER_COMMIT
            or plan.get('data_hashes') != matrix.expected_data(dataset)):
        raise ValueError('job plan or authority differs')
    options = _options(plan.get('command'))
    if Path(plan['command'][1]).name != 'main.py':
        raise ValueError('training entrypoint differs')
    if any(options.get(key) != expected for key, expected in {
            'cl_method': method, 'seed': str(seed),
            'num_tasks': str(TASKS[dataset]), 'num_parties': '4',
            'formal_deferred_evaluation': '1',
    }.items()):
        raise ValueError('job command differs')
    expected_options = _options(matrix.command_for(
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
    _verify_transaction(run, frozen, published, provenance)
    if (frozen.get('status') != 'frozen'
            or frozen.get('identity', {}).get('source_commit')
            != PRODUCER_COMMIT
            or published.get('status') != 'published'
            or published.get('source_provenance') != provenance
            or type(provenance) is not dict
            or provenance.get('source_commit') != PRODUCER_COMMIT
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
    return {
        'job': f'{dataset}/{method}/{seed}',
        'code_commit': PRODUCER_COMMIT,
        'metrics': metrics,
        'results_sha256': success['results_sha256'],
        'checkpoint_sha256': success['checkpoint_sha256'],
    }


def _tree_identity(root):
    rows = []
    for path in sorted(root.rglob('*')):
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


def verify_archive_manifest(job_dir, manifest_path):
    """Verify an archived job's declared byte tree, not its old inode."""
    job_dir = Path(os.path.abspath(job_dir))
    record = _json(manifest_path)
    destination = Path(os.path.abspath(record['destination']))
    if (any(part.is_symlink() for part in (job_dir, *job_dir.parents))
            or any(part.is_symlink()
                   for part in (destination, *destination.parents))
            or not destination.is_dir()):
        raise ValueError('archive destination differs')
    job = f'{job_dir.parent.parent.name}/{job_dir.parent.name}/{job_dir.name}'
    dataset, method, seed = job.split('/')
    if (destination != job_dir
            and not (destination.name == dataset
                     and job_dir == destination / method / seed)):
        raise ValueError('archive scope differs')
    if (record.get('job') != job
            or record.get('code_commit') != PRODUCER_COMMIT
            or not record.get('source')):
        raise ValueError('archive job identity differs')
    files, size, digest = _tree_identity(destination)
    if record.get('files') != files or record.get('bytes') != size:
        raise ValueError('archive file inventory differs')
    if 'destination_tree_sha256' in record:
        if (record['destination_tree_sha256'] != digest
                or record.get('source_tree_sha256') != digest):
            raise ValueError('archive bytes differ')
    elif record.get('tree_sha256') != digest:
        raise InconclusiveEvidence(
            'legacy archive tree digest algorithm is not yet reproduced')
    return {
        'job': job, 'code_commit': PRODUCER_COMMIT,
        'files': files, 'bytes': size, 'tree_sha256': digest,
    }
