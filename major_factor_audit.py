"""Audit shared experimental controls in the frozen 11-baseline comparison."""

import argparse
import csv
import json
from pathlib import Path

from resource_ledger import (
    AUTHORITIES, BASELINE_METHODS, DATASETS, SEEDS,
    file_sha256, verify_authority,
)


RESOURCE_CSV_SHA256 = '0cb12d6a6c66db18a2a56b6f75811aebf81daed4fdf42bd3686b9e1ac420dbdb'
ACCESS_CSV_SHA256 = '28dddbd01377b9e50643c0757da14250d7a8ef4c8de9584256d2c9cb27825b06'
CORE_FIELDS = (
    'data_hashes', 'data_path', 'vector_npz', 'num_classes',
    'num_parties', 'num_tasks', 'custom_tasks', 'classes_per_task',
    'model_type', 'aggregation', 'embed_dim', 'batch_size',
    'num_workers', 'optimizer', 'lr', 'bottom_lr_scale',
    'task_ce_mode', 'momentum', 'weight_decay',
    'epochs_per_task', 'deterministic', 'data_flow_audit',
    'formal_deferred_evaluation', 'bic_enabled',
    'lambda_validation_enabled', 'lambda_validation_per_class',
    'lambda_validation_split_seed', 'train_loader_passes',
    'current_train_loader_sample_reads', 'nominal_optimizer_steps',
    'test_accesses', 'test_phase',
)
ACCESS_FIELDS = (
    'validation_examples', 'calibration_examples',
    'head_candidate_steps', 'configured_bic_steps',
)


def _csv_index(path, expected_sha):
    path = Path(path)
    if file_sha256(path) != expected_sha:
        raise ValueError(f'frozen CSV hash differs: {path}')
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 117:
        raise ValueError('frozen resource/access row count differs')
    indexed = {}
    for row in rows:
        key = (row['authority'], row['job'])
        if key in indexed:
            raise ValueError('duplicate resource/access job')
        indexed[key] = row
    return indexed


def summarize_core(rows, fields=CORE_FIELDS):
    """Keep common controls separate from observed method-specific access."""
    summary = {}
    for dataset in sorted({row['dataset'] for row in rows}):
        group = [row for row in rows if row['dataset'] == dataset]
        common, mismatches = {}, {}
        for field in fields:
            values = {
                json.dumps(row[field], sort_keys=True, ensure_ascii=False)
                for row in group
            }
            decoded = [json.loads(value) for value in sorted(values)]
            if len(decoded) == 1:
                common[field] = decoded[0]
            else:
                mismatches[field] = decoded
        methods = sorted({row['method'] for row in group})
        summary[dataset] = {
            'job_count': len(group),
            'methods': methods,
            'common': common,
            'mismatches': mismatches,
            'method_access': {
                method: sorted({
                    row['validation_examples'] for row in group
                    if row['method'] == method
                }) for method in methods
            },
            'method_calibration': {
                method: sorted({
                    row['calibration_examples'] for row in group
                    if row['method'] == method
                }) for method in methods
            },
        }
    return summary


def collect(resource_csv, access_csv):
    resource = _csv_index(resource_csv, RESOURCE_CSV_SHA256)
    access = _csv_index(access_csv, ACCESS_CSV_SHA256)
    rows = []
    for authority in AUTHORITIES[:2]:
        verify_authority(authority)
        methods = (BASELINE_METHODS if authority['name'] == 'baseline'
                   else ('original_adaptive',))
        for dataset in DATASETS:
            for method in methods:
                folder = 'adaptive' if method == 'original_adaptive' else method
                for seed in SEEDS:
                    identity = f'{dataset}/{method}/{seed}'
                    key = (authority['name'], identity)
                    memory, exposure = resource[key], access[key]
                    job = (authority['jobs_root'] / dataset / folder
                           / str(seed))
                    plan = json.loads((job / 'JOB_PLAN.json').read_text())
                    success = json.loads((job / 'JOB_SUCCESS.json').read_text())
                    config_path = job / 'run/config.json'
                    config = json.loads(config_path.read_text())
                    if (plan.get('data_hashes') != success.get('data_hashes')
                            or plan.get('code_commit')
                            != success.get('code_commit')
                            or plan.get('code_commit')
                            != authority['producer_commit']
                            or plan.get('dataset') != dataset
                            or plan.get('method') != folder
                            or plan.get('seed') != seed
                            or config.get('seed') != seed
                            or exposure['config_sha256']
                            != file_sha256(config_path)
                            or memory['final_checkpoint_sha256']
                            != success.get('checkpoint_sha256')):
                        raise ValueError(f'frozen job identity differs: {identity}')
                    row = {
                        'dataset': dataset, 'method': method, 'seed': seed,
                        'producer_commit': authority['producer_commit'],
                        'config_sha256': exposure['config_sha256'],
                        'data_hashes': plan['data_hashes'],
                    }
                    for field in CORE_FIELDS:
                        if field in row:
                            continue
                        row[field] = (
                            int(exposure[field])
                            if field in ('train_loader_passes',
                                         'current_train_loader_sample_reads',
                                         'nominal_optimizer_steps',
                                         'test_accesses')
                            else exposure[field]
                            if field == 'test_phase'
                            else config.get(field)
                        )
                    for field in ACCESS_FIELDS:
                        row[field] = int(exposure[field])
                    row.update(
                        max_raw_examples=int(memory['max_raw_examples']),
                        max_learning_tensor_bytes=int(
                            memory['max_learning_bytes']),
                        unserialized_teacher_present=(
                            memory['unserialized_teacher_present'] == 'True'),
                    )
                    rows.append(row)
    if len(rows) != 108:
        raise ValueError('main comparison must have 108 audited jobs')
    return rows


def generate(resource_csv, access_csv, output_root):
    output = Path(output_root).resolve()
    if any(output == authority['base'].resolve()
           or authority['base'].resolve() in output.parents
           for authority in AUTHORITIES):
        raise ValueError('audit output overlaps frozen result roots')
    rows = collect(resource_csv, access_csv)
    summary = summarize_core(rows)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'major_factor_cells.json').open('x', encoding='utf-8') as stream:
        json.dump(rows, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write('\n')
    with (output / 'major_factor_summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write('\n')
    with (output / 'major_factor_cells.csv').open(
            'x', newline='', encoding='utf-8') as stream:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({
            key: json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        } for row in rows)
    return {
        'jobs': len(rows),
        'summary_sha256': file_sha256(output / 'major_factor_summary.json'),
        'cells_sha256': file_sha256(output / 'major_factor_cells.csv'),
        'core_mismatches': {
            dataset: list(item['mismatches'])
            for dataset, item in summary.items()
        },
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource-csv', type=Path, required=True)
    parser.add_argument('--access-csv', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.resource_csv, args.access_csv,
                              args.output_root), sort_keys=True))
