"""Independent metadata audit before formal ablation accuracy summarization."""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from frozen_ablation_manifest import (
    AUX_DATA_SHA, BASE_COMMIT, DATASETS, DATA_SHA, PATCH_COMMIT,
    RESULT_ROOT, SEEDS, SOURCE_ANCHOR_SHA, SOURCE_ROOT,
    auxiliary_data_paths, derive_config, expected_keys, semantic_diff,
)

GPU_UUID = {
    0: 'GPU-6c7fee92-a6fe-5525-c554-bddb8e14943f',
    1: 'GPU-6569e526-43ea-74e1-264e-4b66417181e1',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def party_kd_contribution_label(deltas):
    if set(deltas) != set(DATASETS):
        raise ValueError('PartyKD requires three dataset means')
    return ('supported' if sum(delta > 0 for delta in deltas.values()) >= 2
            else 'inconsistent')


def audit_metadata(manifest, digest):
    manifest = Path(manifest).resolve()
    root = Path(RESULT_ROOT)
    if (manifest != root / 'manifests/manifest.jsonl' or
            sha(manifest) != digest or
            (root / 'QUEUE_FAILURE.json').exists()):
        raise ValueError('formal ablation manifest or queue changed')
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if len(rows) != 18 or {row['job_key'] for row in rows} != expected_keys():
        raise ValueError('formal ablation matrix is incomplete')
    queue = json.loads((root / 'QUEUE_COMPLETE.json').read_text())
    ledger_path = root / 'ATTEMPTS.jsonl'
    if (queue.get('status') != 'complete' or
            queue.get('manifest_sha256') != digest or
            queue.get('orchestration_commit') != rows[0]['orchestration_commit'] or
            queue.get('accounted_for') != 18 or
            queue.get('successes') != 18 or
            queue.get('attempts_sha256') != sha(ledger_path)):
        raise ValueError('formal ablation queue is not completely sealed')
    events = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    if len(events) != 36:
        raise ValueError('formal ablation ledger has missing or extra events')
    by_key = defaultdict(list)
    for event in events:
        by_key[event['job_key']].append(event)
    if set(by_key) != expected_keys():
        raise ValueError('formal ablation ledger contains missing or extra jobs')
    source_anchor = SOURCE_ROOT / 'audit/ADAPTIVE_SAMPLE_ONLY_FINAL_ANCHOR.json'
    anchor = json.loads(source_anchor.read_text())
    if (sha(source_anchor) != SOURCE_ANCHOR_SHA or
            anchor.get('producer_commit') != BASE_COMMIT or
            anchor.get('basic_passed') != 9):
        raise ValueError('original formal anchor changed')
    old_ledger = SOURCE_ROOT / 'audit/attestations.jsonl'
    if sha(old_ledger) != anchor['ledger']['sha256']:
        raise ValueError('original formal audit ledger changed')
    old_attest = {}
    for line in old_ledger.read_text().splitlines():
        record = json.loads(line)
        if record.get('stage') == 'basic' and record.get('status') == 'passed':
            if record['job'] in old_attest:
                raise ValueError('duplicate original formal attestation')
            old_attest[record['job']] = record
    if len(old_attest) != 9:
        raise ValueError('original formal attestations are incomplete')
    cache = {}
    def cached_sha(path):
        path = Path(path)
        if path not in cache:
            cache[path] = sha(path)
        return cache[path]
    audited = {}
    for row in rows:
        key = row['job_key']
        dataset, arm, seed_text = key.split('/')
        seed = int(seed_text)
        source = SOURCE_ROOT / 'full' / dataset / 'adaptive' / str(seed) / 'run'
        source_config_path = source / 'config.json'
        source_config = json.loads(source_config_path.read_text())
        old_key = f'{dataset}/adaptive/{seed}'
        original_published = source / 'FORMAL_EVALUATION_PUBLISHED.json'
        if (cached_sha(source_config_path) != row['source_config_sha256'] or
                cached_sha(original_published) != old_attest[old_key]['file_sha256'][
                    'run/FORMAL_EVALUATION_PUBLISHED.json']):
            raise ValueError(f'original formal job changed: {key}')
        old_pub = json.loads(original_published.read_text())
        if cached_sha(source / 'results.json') != old_pub['results_sha256']:
            raise ValueError(f'original formal score source changed: {key}')
        data_path = (Path(source_config['data_path']) / 'cifar-100-python/train'
                     if dataset == 'cifar100' else Path(source_config['vector_npz']))
        if (cached_sha(data_path) != DATA_SHA[dataset] or
                row['source_data_sha256'] != DATA_SHA[dataset] or
                row.get('auxiliary_data_sha256') != AUX_DATA_SHA[dataset]):
            raise ValueError(f'original data differs: {key}')
        for name, path in auxiliary_data_paths(source_config, dataset).items():
            if cached_sha(path) != AUX_DATA_SHA[dataset][name]:
                raise ValueError(f'original auxiliary data differs: {key}, {name}')
        config_path = manifest.parent / row['config_relpath']
        config = json.loads(config_path.read_text())
        code = BASE_COMMIT if arm == 'no_partykd' else PATCH_COMMIT
        gpu = row['physical_gpu']
        if (cached_sha(config_path) != row['config_sha256'] or
                config != derive_config(source_config, dataset, arm, seed) or
                row['config_semantic_diff'] != semantic_diff(source_config, config) or
                row['code_commit'] != code or
                row['source_commit'] != BASE_COMMIT or
                row['source_formal_anchor_sha256'] != SOURCE_ANCHOR_SHA or
                row['gpu_uuid'] != GPU_UUID[gpu] or
                gpu != int(config['device'][-1])):
            raise ValueError(f'formal ablation identity differs: {key}')
        start, finish = by_key[key]
        if (start.get('kind') != 'start' or finish.get('kind') != 'finish' or
                start.get('attempt') != 1 or
                finish.get('attempt') != 1 or
                start.get('config_sha256') != row['config_sha256'] or
                start.get('code_commit') != code or
                start.get('gpu_index') != gpu or
                start.get('gpu_uuid') != GPU_UUID[gpu] or
                finish.get('classification') != 'success' or
                finish.get('returncode') != 0 or
                finish.get('test_accessed') is not True):
            raise ValueError(f'formal ablation attempt differs: {key}')
        job_root = Path(config['output_dir']).parent
        run = job_root / 'attempt-01'
        gate_path = run / 'JOB_SUCCESS.json'
        gate = json.loads(gate_path.read_text())
        log = job_root / 'attempt-01.log'
        if (cached_sha(gate_path) != finish['job_success_sha256'] or
                cached_sha(log) != finish['log_sha256'] or
                gate.get('status') != 'success' or
                gate.get('job_key') != key or
                gate.get('attempt') != 1 or
                gate.get('manifest_sha256') != digest or
                gate.get('config_sha256') != row['config_sha256'] or
                gate.get('training_code_commit') != code):
            raise ValueError(f'formal ablation gate differs: {key}')
        expected_files = {
            'results_sha256': run / 'results.json',
            'checkpoint_sha256': run / 'checkpoints/formal_final.pt',
            'formal_complete_sha256': run / 'FORMAL_EVALUATION_COMPLETE.json',
            'formal_sealed_sha256': run / 'FORMAL_EVALUATION_SEALED.json',
            'formal_published_sha256': run / 'FORMAL_EVALUATION_PUBLISHED.json',
            'test_access_sha256': run / 'formal_access/test.consumed.json',
        }
        if any(cached_sha(path) != gate[field]
               for field, path in expected_files.items()):
            raise ValueError(f'formal ablation result hashes differ: {key}')
        access_files = list((run / 'formal_access').glob('test.*.json'))
        access = json.loads(expected_files['test_access_sha256'].read_text())
        published = json.loads(expected_files['formal_published_sha256'].read_text())
        complete = json.loads(expected_files['formal_complete_sha256'].read_text())
        sealed = json.loads(expected_files['formal_sealed_sha256'].read_text())
        if (len(access_files) != 1 or
                access.get('status') != 'consumed' or
                access.get('access', {}).get('split') != 'test' or
                access.get('access', {}).get('event') != 'first_iteration' or
                complete.get('status') != 'complete' or
                sealed.get('status') != 'sealed' or
                published.get('status') != 'published' or
                published.get('results_sha256') != gate['results_sha256']):
            raise ValueError(f'formal test was not accessed exactly once: {key}')
        if arm == 'fim_off':
            count = len(re.findall(r'FIM applied freeze: 0/\d+ bottom elements '
                                   r'\(fraction=0\.000000\)', log.read_text()))
            if count != config['num_tasks'] or finish.get('fim_zero_freeze_logs') != count:
                raise ValueError(f'FIM-off applied a nonzero or unlogged freeze: {key}')
        audited[key] = {'run': run, 'original': source,
                        'gate_sha256': cached_sha(gate_path)}
    return audited, sha(ledger_path)


def final_score(path):
    result = json.loads(Path(path).read_text())
    value = result['cl_metrics']['AA_final']
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not 0 <= value <= 1):
        raise ValueError('formal final Class-IL is invalid')
    return Fraction(str(value))


def summarize(audited):
    per_seed, summaries = [], []
    means = {}
    for dataset in DATASETS:
        for arm in ('no_partykd', 'fim_off'):
            main_scores, ablated_scores, deltas, exact_deltas = [], [], [], []
            for seed in SEEDS:
                job = audited[f'{dataset}/{arm}/{seed}']
                main = final_score(job['original'] / 'results.json')
                ablated = final_score(job['run'] / 'results.json')
                delta = main - ablated
                main_scores.append(float(main))
                ablated_scores.append(float(ablated))
                deltas.append(float(delta))
                exact_deltas.append(delta)
                per_seed.append({'dataset': dataset, 'arm': arm, 'seed': seed,
                    'main_class_il': float(main),
                    'ablated_class_il': float(ablated),
                    'paired_delta_pp': float(delta * 100)})
            means[(dataset, arm)] = sum(exact_deltas) / 3
            summaries.append({'dataset': dataset, 'arm': arm,
                'main_mean': statistics.mean(main_scores),
                'main_sample_sd': statistics.stdev(main_scores),
                'ablated_mean': statistics.mean(ablated_scores),
                'ablated_sample_sd': statistics.stdev(ablated_scores),
                'paired_delta_mean_pp': 100 * statistics.mean(deltas),
                'paired_delta_sample_sd_pp': 100 * statistics.stdev(deltas)})
    label = party_kd_contribution_label({dataset: means[(dataset, 'no_partykd')]
                                         for dataset in DATASETS})
    return per_seed, summaries, label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    root = Path(RESULT_ROOT)
    audited, ledger_sha = audit_metadata(args.manifest, args.manifest_sha256)
    pass_path = root / 'INDEPENDENT_AUDIT_PASS.json'
    if not args.summarize:
        if pass_path.exists():
            raise FileExistsError('ablation audit already sealed')
        with pass_path.open('x', encoding='utf-8') as stream:
            json.dump({'status': 'passed', 'jobs': 18,
                       'manifest_sha256': args.manifest_sha256,
                       'ledger_sha256': ledger_sha,
                       'job_gate_sha256': {key: item['gate_sha256']
                                           for key, item in audited.items()},
                       'accuracy_opened': False}, stream,
                      indent=2, sort_keys=True)
            stream.write('\n')
        print(json.dumps({'audit': 'passed', 'jobs': 18,
                          'accuracy_opened': False}))
        return
    passed = json.loads(pass_path.read_text())
    if (passed.get('status') != 'passed' or passed.get('jobs') != 18 or
            passed.get('manifest_sha256') != args.manifest_sha256 or
            passed.get('ledger_sha256') != ledger_sha or
            passed.get('job_gate_sha256') != {key: item['gate_sha256']
                                              for key, item in audited.items()}):
        raise ValueError('independent audit seal changed before summary')
    per_seed, summaries, label = summarize(audited)
    for filename, rows in (('ABLATION_PER_SEED.csv', per_seed),
                           ('ABLATION_SUMMARY.csv', summaries)):
        with (root / filename).open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (root / 'ABLATION_INTERPRETATION.json').open('x', encoding='utf-8') as stream:
        json.dump({'party_kd_contribution': label,
                   'main_method_changed': False,
                   'formal_results_seen_before_ablation': True,
                   'significance_claimed': False}, stream,
                  indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'status': 'summarized', 'party_kd_contribution': label}))


if __name__ == '__main__':
    main()
