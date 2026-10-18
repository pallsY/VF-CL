"""Cross-check exported resource ledgers without opening test metrics."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from resource_ledger import AUTHORITIES, DATASETS, SEEDS


def _load_pair(json_path, csv_path):
    data = json.loads(Path(json_path).read_text(encoding='utf-8'))
    with Path(csv_path).open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if (type(data) is not list or not data or len(rows) != len(data)
            or any(type(record) is not dict or set(record) != set(data[0])
                   for record in data)
            or any(set(row) != set(data[0])
                   or any(row[key] != ('' if record[key] is None
                                       else str(record[key]))
                          for key in record)
                   for row, record in zip(rows, data))):
        raise ValueError('CSV and JSON resource rows differ')
    return data


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_exports(memory_json, memory_csv, access_json, access_csv,
                   *, expected_jobs=None):
    memory = _load_pair(memory_json, memory_csv)
    access = _load_pair(access_json, access_csv)
    if expected_jobs is None:
        expected_jobs = {
            f'{dataset}/{method}/{seed}'
            for authority in AUTHORITIES for dataset in DATASETS
            for method in authority['methods'] for seed in SEEDS
        }
    jobs = [row['job'] for row in memory]
    access_jobs = [row['job'] for row in access]
    if (len(jobs) != len(expected_jobs)
            or len(access_jobs) != len(expected_jobs)
            or set(jobs) != expected_jobs
            or set(access_jobs) != expected_jobs):
        raise ValueError('resource job inventory differs')
    for row in memory:
        components = sum(row[key] for key in (
            'final_model_bytes', 'final_raw_bytes', 'final_logits_bytes',
            'final_feature_bytes', 'final_teacher_bytes',
            'final_other_bytes'))
        if (components != row['final_learning_bytes']
                or row['final_audit_only_bytes'] < 0
                or row['max_learning_bytes'] < components
                or row['max_raw_examples'] < row['final_raw_examples']
                or row['max_raw_bytes'] < row['final_raw_bytes']):
            raise ValueError(f"resource category sum differs: {row['job']}")
    for row in access:
        if (row['test_accesses'] != 1
                or row['nominal_optimizer_steps'] < 1
                or row['validation_examples'] < 0
                or row['calibration_examples'] < 0):
            raise ValueError(f"resource access count differs: {row['job']}")
    return {
        'verified_cells': len(jobs),
        'memory_json_sha256': _sha(memory_json),
        'memory_csv_sha256': _sha(memory_csv),
        'access_json_sha256': _sha(access_json),
        'access_csv_sha256': _sha(access_csv),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('memory_json', type=Path)
    parser.add_argument('memory_csv', type=Path)
    parser.add_argument('access_json', type=Path)
    parser.add_argument('access_csv', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_exports(
        args.memory_json, args.memory_csv,
        args.access_json, args.access_csv), sort_keys=True))
