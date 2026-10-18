"""Count audited current-task loader access and configured steps.

Replay-buffer samples and teacher/model operations are separate costs; this
module never treats current-task loader counts or wall time as total FLOPs.
"""

import ast
import argparse
import csv
import json
from pathlib import Path

from resource_ledger import (AUTHORITIES, DATASETS, SEEDS, file_sha256,
                             verify_authority)


def _manifest_count(run, split):
    relative = ('validation/validation_manifest.json'
                if split == 'validation'
                else 'bic/calibration_manifest.json')
    path = run / relative
    if not path.is_file():
        raise ValueError(f'{split} manifest is missing')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    indices = manifest.get('ordered_indices')
    if (type(indices) is not list or not indices
            or any(type(index) is not int or index < 0 for index in indices)
            or len(indices) != len(set(indices))):
        raise ValueError(f'{split} manifest is invalid')
    return len(indices)


def inventory(run_dir, config):
    """Return exact loader reads and explicitly labelled configured steps."""
    run = Path(run_dir)
    tasks = config['num_tasks']
    epochs = config['epochs_per_task']
    if (type(tasks) is not int or tasks < 1
            or type(epochs) is not int or epochs < 1):
        raise ValueError('task training budget differs')
    by_iteration = {}
    seen_batches = set()
    sample_reads = batch_reads = 0
    with (run / 'data_flow_audit.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            if record.get('split') is not None:
                continue
            key = ast.literal_eval(record['loader_key'])
            iteration = record['loader_iteration']
            batch = record['batch']
            indices = record['indices']
            shape = record['shape']
            if (type(key) is not tuple or len(key) != 2
                    or key[0] != 'train' or type(key[1]) is not tuple
                    or type(iteration) is not int or iteration < 0
                    or type(batch) is not int or batch < 0
                    or type(indices) is not list or not indices
                    or type(shape) is not list or not shape
                    or shape[0] != len(indices)):
                raise ValueError('training loader record differs')
            identity = (key[1], iteration, batch)
            if identity in seen_batches:
                raise ValueError('duplicate training batch record')
            seen_batches.add(identity)
            by_iteration.setdefault(key[1], {}).setdefault(iteration, 0)
            by_iteration[key[1]][iteration] += 1
            batch_reads += 1
            sample_reads += len(indices)
    if len(by_iteration) != tasks:
        raise ValueError('training task loader count differs')
    nominal_steps = sum(counts[min(counts)] * epochs
                        for counts in by_iteration.values())
    loader_passes = sum(len(counts) for counts in by_iteration.values())
    extra_passes = loader_passes - tasks * epochs
    if extra_passes < 0:
        raise ValueError('training loader passes are below epoch budget')

    access_dir = run / 'formal_access'
    splits = {}
    for path in access_dir.glob('*.consumed.json'):
        marker = json.loads(path.read_text(encoding='utf-8'))
        split = path.name.removesuffix('.consumed.json')
        if (split not in ('validation', 'calibration', 'test')
                or marker.get('status') != 'consumed'
                or marker.get('access', {}).get('split') != split
                or split in splits):
            raise ValueError('formal access marker differs')
        splits[split] = marker['access'].get('phase')
    if 'test' not in splits:
        raise ValueError('formal test access marker is missing')
    validation = _manifest_count(run, 'validation') if 'validation' in splits else 0
    calibration = _manifest_count(run, 'calibration') if 'calibration' in splits else 0
    adaptive_head = (config.get('cl_method') == 'proto_evolve'
                     and bool(config.get('head_consolidation_enabled'))
                     and config.get('head_consolidation_schedule') == 'final')
    return {
        'train_loader_passes': loader_passes,
        'extra_loader_passes': extra_passes,
        'train_batch_reads': batch_reads,
        'current_train_loader_sample_reads': sample_reads,
        'nominal_optimizer_steps': nominal_steps,
        'head_candidate_steps': (int(config['head_full_steps'])
                                 + int(config['head_bias_steps'])
                                 if adaptive_head else 0),
        'configured_bic_steps': (int(config['bic_steps']) * tasks
                                 if bool(config.get('bic_enabled')) else 0),
        'validation_examples': validation,
        'calibration_examples': calibration,
        'test_accesses': 1,
        'access_phases': splits,
    }


def read_access_cell(dataset, method, seed, authority):
    if (dataset not in DATASETS or method not in authority['methods']
            or seed not in SEEDS):
        raise ValueError('access cell is outside the frozen matrix')
    verify_authority(authority)
    folder = ('adaptive' if method in ('original_adaptive', 'sample_only')
              else method)
    job = authority['jobs_root'] / dataset / folder / str(seed)
    if job.is_symlink() or not job.is_dir():
        raise ValueError(f'frozen access job differs: {job}')
    plan = json.loads((job / 'JOB_PLAN.json').read_text(encoding='utf-8'))
    if (plan.get('dataset') != dataset or plan.get('method') != folder
            or plan.get('seed') != seed
            or plan.get('code_commit') != authority['producer_commit']):
        raise ValueError(f'frozen access producer differs: {job}')
    run = job / 'run'
    config_path = run / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    task_count, epochs = DATASETS[dataset]
    if (config.get('cl_method') != (
            'proto_evolve' if folder == 'adaptive' else folder)
            or config.get('seed') != seed
            or config.get('num_tasks') != task_count
            or config.get('num_parties') != 4
            or config.get('epochs_per_task') != epochs
            or config.get('data_path') != authority['data_root']):
        raise ValueError(f'frozen access configuration differs: {job}')
    flow_path = run / 'data_flow_audit.jsonl'
    counted = inventory(run, config)
    phases = counted.pop('access_phases')
    return {
        'job': f'{dataset}/{method}/{seed}',
        'authority': authority['name'],
        'producer_commit': authority['producer_commit'],
        'config_sha256': file_sha256(config_path),
        'data_flow_sha256': file_sha256(flow_path),
        **counted,
        'validation_phase': phases.get('validation', ''),
        'calibration_phase': phases.get('calibration', ''),
        'test_phase': phases['test'],
        'flops_measured': False,
    }


def generate_access(output_root, seeds=SEEDS):
    output = Path(output_root).resolve()
    if any(output == authority['base'].resolve()
           or authority['base'].resolve() in output.parents
           for authority in AUTHORITIES):
        raise ValueError('access ledger overlaps frozen results')
    output.mkdir(parents=True, exist_ok=False)
    rows = [read_access_cell(dataset, method, seed, authority)
            for authority in AUTHORITIES
            for dataset in DATASETS
            for method in authority['methods']
            for seed in seeds]
    if len(rows) != 13 * 3 * len(seeds):
        raise ValueError('access ledger cell count differs')
    json_path = output / 'access_cells.json'
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n',
                         encoding='utf-8')
    csv_path = output / 'access_cells.csv'
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
    print(json.dumps(generate_access(args.output_root, tuple(args.seeds)),
                     sort_keys=True))
