"""Read frozen jobs into a separate, accuracy-free resource ledger."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from adaptive_consolidation_audit import _safe_torch_load
from resource_accounting import account_method_state, payload_bytes
from runner import _decode_checkpoint_value


DATASETS = {'cifar100': (10, 50), 'isolet': (13, 50),
            'upmc_food101': (10, 20)}
SEEDS = (42, 43, 44)
BASELINE_METHODS = ('adagauss', 'alsar', 'balanced_contrast',
                    'der_pp', 'er', 'er_ace', 'hpcr', 'icarl',
                    'lpr', 'mir', 'ocar')
RESULTS = Path('/brf/users/yangxx/VFCL-results')
DATA_ROOT = '/brf/users/yangxx/VF-CL/data'


def _authority(name, folder, jobs_folder, gate_folder, gate_sha,
               producer, auditor, basic, numeric, methods):
    base = RESULTS / folder
    return {
        'name': name, 'base': base, 'jobs_root': base / jobs_folder,
        'gate_path': base / gate_folder / 'FINAL_GATE.json',
        'gate_sha256': gate_sha,
        'producer_commit': producer, 'audit_commit': auditor,
        'basic_passed': basic, 'numeric_passed': numeric,
        'methods': methods, 'data_root': DATA_ROOT,
    }


AUTHORITIES = (
    _authority(
        'baseline', 'replay11-paper-v3-20261013', 'full-3080', 'audit-v3',
        'fc4b555de0451181ac15fe51f6bc94a5183f8d1e633070c74bc79680f6f392ec',
        '467611a603c9604ad57861920495a3ad05612037',
        '8029edebd1b7f981ef75624a90150631c41de5ab',
        99, 33, BASELINE_METHODS),
    _authority(
        'original_adaptive', 'adaptive9-fourparty-v1-20261015',
        'full', 'audit',
        'db3dde9553d9029cf73d09b2ce76f7af51530d642f487d033ec83d98837cd20c',
        '9973018295da270fd6098de07dcf36b9fe65ff61',
        'c0442611ebce23c3428111258cb8a7b731d63863',
        9, 3, ('original_adaptive',)),
    _authority(
        'sample_only', 'adaptive-sample-only-fourparty-v1-20261017',
        'full', 'audit',
        'ff378d1d80abf571ec2d222fdbf5c19f1dc28e529a5ce64d44f7a3a84e05cbad',
        'e2a559dc4945729848225cf21fdc95451762c231',
        '0474731423ca6ce9db03a413792f33e519306ed4',
        9, 3, ('sample_only',)),
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_authority(authority):
    gate_path = authority['gate_path']
    if not gate_path.is_file() or file_sha256(gate_path) != authority['gate_sha256']:
        raise ValueError(f"frozen gate differs: {authority['name']}")
    gate = _json(gate_path)
    if (gate.get('producer_commit') != authority['producer_commit']
            or gate.get('audit_commit') != authority['audit_commit']
            or gate.get('basic_passed') != authority['basic_passed']
            or gate.get('numeric_passed') != authority['numeric_passed']):
        raise ValueError(f"frozen gate identity differs: {authority['name']}")
    ledger_path = gate_path.parent / 'attestations.jsonl'
    if (not ledger_path.is_file()
            or file_sha256(ledger_path) != gate.get('ledger_sha256')):
        raise ValueError(f"frozen ledger differs: {authority['name']}")
    return gate


def _account_checkpoint(path, method, expected_task):
    payload = _safe_torch_load(path)
    trainer = payload.get('trainer_state')
    if (payload.get('schema_version') != 4
            or payload.get('task_id') != expected_task
            or type(trainer) is not dict
            or type(trainer.get('bottoms')) is not list
            or len(trainer['bottoms']) != 4):
        raise ValueError(f'checkpoint schema or four-party state differs: {path}')
    state = _decode_checkpoint_value(payload.get('cl_state'))
    roles = account_method_state(method, state)
    model_bytes = payload_bytes(trainer)
    return {**roles, 'model_bytes': model_bytes,
            'learning_bytes': model_bytes + roles['method_learning_bytes']}


def read_cell(dataset, method, seed, authority):
    """Recount final and task-boundary payloads without using test metrics."""
    if (dataset not in DATASETS or method not in authority['methods']
            or seed not in SEEDS):
        raise ValueError('resource cell is outside the frozen matrix')
    verify_authority(authority)
    task_count, epochs = DATASETS[dataset]
    folder = ('adaptive' if method in ('original_adaptive', 'sample_only')
              else method)
    job = authority['jobs_root'] / dataset / folder / str(seed)
    if job.is_symlink() or not job.is_dir():
        raise ValueError(f'frozen job path differs: {job}')
    plan = _json(job / 'JOB_PLAN.json')
    success = _json(job / 'JOB_SUCCESS.json')
    if (plan.get('dataset') != dataset or plan.get('method') != folder
            or plan.get('seed') != seed
            or plan.get('code_commit') != authority['producer_commit']
            or success.get('code_commit') != authority['producer_commit']):
        raise ValueError(f'frozen producer identity differs: {job}')
    run = job / 'run'
    config = _json(run / 'config.json')
    expected_method = 'proto_evolve' if folder == 'adaptive' else folder
    if (config.get('cl_method') != expected_method
            or config.get('seed') != seed
            or config.get('num_tasks') != task_count
            or config.get('num_parties') != 4
            or config.get('epochs_per_task') != epochs
            or config.get('data_path') != authority['data_root']):
        raise ValueError(f'four-party training configuration differs: {job}')
    frozen = _json(run / 'FORMAL_STATE_FROZEN.json')['identity']
    if frozen.get('protocol', {}).get('source_provenance', {}).get(
            'source_commit') != authority['producer_commit']:
        raise ValueError(f'frozen source provenance differs: {job}')
    final_path = run / 'checkpoints/formal_final.pt'
    final_hash = success.get('checkpoint_sha256')
    if (final_path.is_symlink() or not final_path.is_file()
            or file_sha256(final_path) != final_hash):
        raise ValueError(f'final checkpoint hash differs: {job}')
    final = _account_checkpoint(final_path, expected_method, task_count - 1)
    maxima = {
        'learning_bytes': final['learning_bytes'],
        'raw_examples': final['raw_examples'],
        'raw_bytes': final['raw_bytes'],
    }
    ever_teacher = final['unserialized_teacher']
    snapshots = frozen.get('snapshots')
    if type(snapshots) is not list or len(snapshots) != task_count:
        raise ValueError(f'frozen snapshot inventory differs: {job}')
    for task, entry in enumerate(snapshots):
        relative = f'formal_snapshots/event_{task}_CIL.pt'
        record = entry.get('file', {})
        path = run / relative
        if (record.get('path') != relative or path.is_symlink()
                or not path.is_file()
                or file_sha256(path) != record.get('sha256')):
            raise ValueError(f'frozen snapshot hash differs: {job}/{task}')
        measured = _account_checkpoint(path, expected_method, task)
        for key in maxima:
            maxima[key] = max(maxima[key], measured[key])
        ever_teacher |= measured['unserialized_teacher']
    return {
        'job': f'{dataset}/{method}/{seed}',
        'authority': authority['name'],
        'producer_commit': authority['producer_commit'],
        'final_checkpoint_sha256': final_hash,
        'snapshot_count': task_count,
        'final_model_bytes': final['model_bytes'],
        'final_raw_examples': final['raw_examples'],
        'final_raw_bytes': final['raw_bytes'],
        'final_logits_bytes': final['logits_bytes'],
        'final_feature_bytes': final['feature_bytes'],
        'final_teacher_bytes': final['teacher_bytes'],
        'final_other_bytes': final['other_bytes'],
        'final_audit_only_bytes': final['audit_bytes'],
        'final_learning_bytes': final['learning_bytes'],
        'max_learning_bytes': maxima['learning_bytes'],
        'max_raw_examples': maxima['raw_examples'],
        'max_raw_bytes': maxima['raw_bytes'],
        'unserialized_teacher_present': ever_teacher,
    }


def generate(output_root, seeds=SEEDS):
    output_root = Path(output_root).resolve()
    if any(output_root == authority['base'].resolve()
           or authority['base'].resolve() in output_root.parents
           for authority in AUTHORITIES):
        raise ValueError('resource ledger output overlaps frozen results')
    output_root.mkdir(parents=True, exist_ok=False)
    rows = [read_cell(dataset, method, seed, authority)
            for authority in AUTHORITIES
            for dataset in DATASETS
            for method in authority['methods']
            for seed in seeds]
    if len(rows) != 13 * 3 * len(seeds):
        raise ValueError('resource ledger cell count differs')
    json_path = output_root / 'resource_cells.json'
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n',
                         encoding='utf-8')
    csv_path = output_root / 'resource_cells.csv'
    with csv_path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {'rows': len(rows), 'json_sha256': file_sha256(json_path),
            'csv_sha256': file_sha256(csv_path)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS,
                        default=list(SEEDS))
    args = parser.parse_args()
    print(json.dumps(generate(args.output_root, tuple(args.seeds)),
                     sort_keys=True))
