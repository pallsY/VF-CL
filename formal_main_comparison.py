"""Summarize the frozen 99 baseline and 9 original-adaptive formal jobs."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from resource_ledger import (
    AUTHORITIES, BASELINE_METHODS, DATASETS, SEEDS,
    file_sha256, verify_authority,
)


FACTOR_JSON_SHA256 = 'b84fc16151f1639c8884408cd10eb8770eecc77ccefb13787e7b93ce60eacd8a'
METRICS = ('aa_final_cil', 'bwt_cil', 'task_il_final')


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row['dataset'], row['method']), []).append(row)
    summary = []
    for (dataset, method), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: row['seed'])
        if [row['seed'] for row in group] != list(SEEDS):
            raise ValueError(f'{dataset}/{method} seeds differ')
        result = {'dataset': dataset, 'method': method,
                  'seed_count': len(group)}
        for metric in METRICS:
            values = [float(row['metrics'][metric]) for row in group]
            if any(not math.isfinite(value) or abs(value) > 1.000001
                   for value in values):
                raise ValueError('formal metric is invalid')
            name = metric.replace('_cil', '')
            result[f'{name}_mean_pct'] = 100 * statistics.mean(values)
            result[f'{name}_sd_pct'] = 100 * statistics.stdev(values)
            for row, value in zip(group, values):
                result[f'seed_{row["seed"]}_{name}_pct'] = 100 * value
        for field in ('validation_examples', 'calibration_examples',
                      'head_candidate_steps'):
            values = {row[field] for row in group}
            if len(values) != 1:
                raise ValueError(f'{dataset}/{method} {field} differs by seed')
            result[field] = values.pop()
        result['max_raw_examples'] = max(
            row['max_raw_examples'] for row in group)
        result['max_learning_tensor_bytes'] = max(
            row['max_learning_tensor_bytes'] for row in group)
        summary.append(result)
    for dataset in DATASETS:
        ranking = sorted(
            (row for row in summary if row['dataset'] == dataset),
            key=lambda row: (-row['aa_final_mean_pct'], row['method']),
        )
        for index, row in enumerate(ranking, 1):
            row['aa_rank'] = index
    return summary


def collect(factor_json):
    factor_json = Path(factor_json)
    if file_sha256(factor_json) != FACTOR_JSON_SHA256:
        raise ValueError('key-factor evidence hash differs')
    factors = json.loads(factor_json.read_text(encoding='utf-8'))
    if len(factors) != 108:
        raise ValueError('key-factor job count differs')
    for authority in AUTHORITIES[:2]:
        verify_authority(authority)
    rows = []
    for factor in factors:
        dataset, method, seed = (
            factor['dataset'], factor['method'], factor['seed'])
        if (dataset not in DATASETS or seed not in SEEDS
                or method not in BASELINE_METHODS + ('original_adaptive',)):
            raise ValueError('main comparison job identity differs')
        authority = (AUTHORITIES[1] if method == 'original_adaptive'
                     else AUTHORITIES[0])
        folder = 'adaptive' if method == 'original_adaptive' else method
        job = authority['jobs_root'] / dataset / folder / str(seed)
        success = json.loads((job / 'JOB_SUCCESS.json').read_text())
        results = job / 'run/results.json'
        if (success.get('code_commit') != factor['producer_commit']
                or success.get('results_sha256') != file_sha256(results)
                or set(success.get('metrics', {})) != set(METRICS)):
            raise ValueError(f'frozen metric source differs: {job}')
        reported = json.loads(results.read_text(encoding='utf-8'))
        reported_cl = reported.get('cl_metrics', {})
        if (abs(float(reported_cl.get('AA_final'))
                - float(success['metrics']['aa_final_cil'])) > 0.00011
                or abs(float(reported_cl.get('BWT'))
                - float(success['metrics']['bwt_cil'])) > 0.00011):
            raise ValueError(f'frozen metric summary differs: {job}')
        rows.append({**factor, 'metrics': success['metrics'],
                     'results_sha256': success['results_sha256']})
    if len({(row['dataset'], row['method'], row['seed'])
            for row in rows}) != 108:
        raise ValueError('duplicate formal comparison cell')
    return rows


def generate(factor_json, output_root):
    output = Path(output_root).resolve()
    if any(output == authority['base'].resolve()
           or authority['base'].resolve() in output.parents
           for authority in AUTHORITIES):
        raise ValueError('summary output overlaps frozen results')
    cells = collect(factor_json)
    summary = summarize(cells)
    if len(summary) != 36:
        raise ValueError('main comparison row count differs')
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'formal_main_cells.json').open('x', encoding='utf-8') as stream:
        json.dump(cells, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write('\n')
    with (output / 'formal_main_summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write('\n')
    with (output / 'formal_main_summary.csv').open(
            'x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return {
        'cells': len(cells), 'summary_rows': len(summary),
        'summary_csv_sha256': file_sha256(
            output / 'formal_main_summary.csv'),
        'summary_json_sha256': file_sha256(
            output / 'formal_main_summary.json'),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factor-json', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.factor_json, args.output_root),
                     sort_keys=True))
