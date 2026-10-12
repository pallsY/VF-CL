"""Four-party replay pilot commands and job identities.

This runner evaluates after every task and does not provide the repository's
deferred formal-test access audit. Its metrics are pilot measurements only.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from itertools import product
from pathlib import Path

import fair_main_table_3datasets as fair
import formal_cifar100_metrics as formal_metrics
import three_dataset_formal_registry as registry
import three_dataset_formal_audit as audit


DATASETS = ('cifar100', 'isolet', 'upmc_food101')
METHODS = (
    'balanced_contrast', 'lpr', 'hpcr', 'ocar', 'alsar',
    'mir', 'icarl', 'er', 'der_pp', 'er_ace', 'adagauss',
)
SEEDS = (42, 43, 44)
PILOT_TABLE_NAME = 'REPLAY11_PILOT_TABLE.csv'
UPMC_FOUR_CACHE = ('upmc_food101_image_clean_4party_v1'
                   '/upmc_food101_vfl.npz')
_UPMC_FOUR_DATA = {
    'upmc_food101_image_clean_v1/SELECTION_MANIFEST.json':
        '97340df99705f00a2f5201051761b15c519b94ddcc2a5e20cea423491d16e585',
    UPMC_FOUR_CACHE:
        'dbdd3a2979203ba0ee804de7d3b128426f869b0a127e4b87ebbe023e5750023b',
    'upmc_food101_image_clean_4party_v1/upmc_food101_vfl.metadata.json':
        'a9f81a847969dab024d0831e6505e764f8fce58d27828f18eb593d8001358c8c',
}
METHOD_OPTIONS = {
    'balanced_contrast': {'balanced_gamma': 4.0,
                          'balanced_temperature': 0.1},
    'lpr': {'lpr_omega0': 1.0, 'lpr_beta': 1.0,
            'lpr_interval': 10, 'lpr_reference_size': 8,
            'lpr_max_rows': 64},
    'hpcr': {'hpcr_temperature': 0.09, 'hpcr_temp_max': 1.8,
             'hpcr_temp_min': 0.5625, 'hpcr_pcd_alpha': 0.1,
             'hpcr_scd_beta': 0.5},
    'ocar': {'ocar_ema': 0.01, 'ocar_delta_tau': 1e-5,
             'ocar_max_rows': 32},
    'alsar': {'alsar_temperature': 1.0, 'alsar_decay': 0.99,
              'alsar_fisher_rate': 0.01,
              'alsar_gradient_rate': 0.01},
    'mir': {'mir_candidates': 128},
    'icarl': {},
    'er': {},
    'der_pp': {'der_buffer_size': 0, 'der_batch': 64,
               'der_alpha': 0.5, 'der_beta': 0.5},
    'er_ace': {'er_ace_buffer_size': 0, 'er_ace_batch': 64},
    'adagauss': {'adagauss_lambda_ac': 0.2,
                 'adagauss_lambda_pkd': 1.0,
                 'adagauss_shrinkage': 0.1,
                 'adagauss_adapter_epochs': 30,
                 'adagauss_n_samples': 256},
}


def jobs():
    return tuple(product(DATASETS, METHODS, SEEDS))


def expected_data(dataset):
    if dataset not in DATASETS:
        raise ValueError('unregistered dataset')
    return (dict(_UPMC_FOUR_DATA) if dataset == 'upmc_food101'
            else dict(audit._AUTHORITATIVE_DATA[dataset]))


def metrics_from_history(history, expected_tasks):
    return formal_metrics.compute_formal_metrics(
        history, expected_tasks=expected_tasks)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_data(dataset, data_root):
    data_root = Path(data_root)
    for relative, expected_hash in expected_data(dataset).items():
        target = data_root / relative
        if target.is_symlink() or not target.is_file():
            raise FileNotFoundError(f'missing regular dataset file: {target}')
        if sha256_file(target) != expected_hash:
            raise ValueError(f'dataset hash differs: {target}')


def validate_completed_run(job, run_dir, expected_tasks):
    dataset, method, seed = job
    run_dir = Path(run_dir)
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    result = json.loads((run_dir / 'results.json').read_text(encoding='utf-8'))
    if (config.get('cl_method') != method
            or config.get('seed') != seed
            or config.get('num_tasks') != expected_tasks
            or config.get('num_parties') != 4
            or result.get('config', {}).get('cl_method') != method
            or result.get('config', {}).get('seed') != seed
            or result.get('config', {}).get('num_tasks') != expected_tasks):
        raise ValueError('completed run violates job method, seed, task, or party identity')
    if dataset == 'upmc_food101' and not str(
            config.get('vector_npz', '')).endswith(UPMC_FOUR_CACHE):
        raise ValueError('UPMC run did not use the four-party clean cache')
    checkpoint = run_dir / 'checkpoints' / f'event_{expected_tasks - 1}_CIL.pt'
    if checkpoint.is_symlink() or not checkpoint.is_file() or not checkpoint.stat().st_size:
        raise ValueError('final task checkpoint is missing')
    metrics = metrics_from_history(
        result.get('task_acc_history', []), expected_tasks)
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError('nonfinite class-incremental metric')
    return metrics


def verify_success_record(job_dir, record):
    job = (record['dataset'], record['method'], record['seed'])
    if job not in jobs() or record['data_hashes'] != expected_data(job[0]):
        raise ValueError('success record job or data authority differs')
    run_dirs = list(Path(job_dir).glob('run_*'))
    if len(run_dirs) != 1:
        raise ValueError('success record has ambiguous run directories')
    tasks = (2 if record['smoke'] else
             {'cifar100': 10, 'isolet': 13, 'upmc_food101': 10}[job[0]])
    actual = validate_completed_run(job, run_dirs[0], tasks)
    checkpoint = run_dirs[0] / 'checkpoints' / f'event_{tasks - 1}_CIL.pt'
    if (actual != record['metrics']
            or sha256_file(run_dirs[0] / 'results.json')
            != record['results_sha256']
            or sha256_file(checkpoint) != record['checkpoint_sha256']):
        raise ValueError('success record no longer matches result or checkpoint')
    return actual


def jobs_for_shard(index, count):
    if type(index) is not int or type(count) is not int or not 0 <= index < count:
        raise ValueError('invalid shard index or count')
    return tuple(job for position, job in enumerate(jobs())
                 if position % count == index)


def command_for(job, results_root, *, python, worktree,
                device, data_root=None, smoke=False):
    dataset, method, seed = job
    if dataset not in DATASETS or method not in METHODS or seed not in SEEDS:
        raise ValueError('unregistered replay-11 job')
    if type(smoke) is not bool:
        raise TypeError('smoke must be bool')
    # Reuse the reviewed dataset-specific settings. ER supplies the shared
    # replay-class protocol; only the method-specific algorithm changes.
    base = registry.protocol_for(
        registry.FormalSpec(dataset, 'er', seed))['base_options']
    options = dict(base)
    options.update(
        cl_method=method,
        er_per_class=20,
        er_batch=64,
        er_alpha=1.0,
        device=str(device),
        results_dir=str(results_root),
        exp_name='run',
        formal_deferred_evaluation=False,
    )
    if data_root is not None:
        options['data_path'] = str(data_root)
    if dataset == 'upmc_food101':
        options['num_parties'] = 4
        options['vector_npz'] = str(
            Path(options['data_path']) / UPMC_FOUR_CACHE)
    if smoke:
        if 'custom_tasks' in options:
            options['custom_tasks'] = options['custom_tasks'][:2]
        options['num_tasks'] = 2
        options['epochs_per_task'] = 1
    options.update(METHOD_OPTIONS[method])
    command = [str(python), str(Path(worktree) / 'main.py')]
    for option, value in options.items():
        if option in registry.OPTION_SCHEMA:
            encoded = registry._option_token(option, value)
        else:
            encoded = str(value)
        command.extend(('--' + option, encoded))
    return command


def _git_commit(worktree):
    worktree = Path(worktree)
    status = subprocess.check_output(
        ['git', '-C', str(worktree), 'status', '--porcelain'], text=True)
    if status.strip():
        raise ValueError('replay-11 code worktree must be clean')
    return subprocess.check_output(
        ['git', '-C', str(worktree), 'rev-parse', 'HEAD'], text=True).strip()


def run_one(job, results_root, *, data_root, device, smoke=False):
    dataset, method, seed = job
    worktree = Path(__file__).resolve().parent
    commit = _git_commit(worktree)
    verify_data(dataset, data_root)
    root = Path(results_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    minimum_free = 2 * 1024 ** 3 if dataset == 'cifar100' else 512 * 1024 ** 2
    if shutil.disk_usage(root).free < minimum_free:
        raise RuntimeError('insufficient result storage for replay-11 job')
    job_dir = root / dataset / method / str(seed)
    if job_dir.exists():
        raise FileExistsError(f'job root already exists: {job_dir}')
    command = command_for(
        job, job_dir, python=Path(sys.executable), worktree=worktree,
        device=device, data_root=data_root, smoke=smoke)
    job_dir.mkdir(parents=True)
    plan = {
        'dataset': dataset, 'method': method, 'seed': seed,
        'code_commit': commit, 'command': command,
        'data_hashes': expected_data(dataset), 'smoke': smoke,
    }
    (job_dir / 'JOB_PLAN.json').write_text(
        json.dumps(plan, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    env = dict(os.environ)
    env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONHASHSEED=str(seed),
               VFCL_EXPERIMENT_PROFILE='formal')
    env.pop('VFCL_FORMAL_DATASET', None)
    with (job_dir / 'training.log').open('x', encoding='utf-8') as log:
        subprocess.run(command, cwd=worktree, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    run_dirs = list(job_dir.glob('run_*'))
    if len(run_dirs) != 1:
        raise ValueError('job produced an ambiguous result directory')
    tasks = 2 if smoke else {'cifar100': 10, 'isolet': 13,
                             'upmc_food101': 10}[dataset]
    metrics = validate_completed_run(job, run_dirs[0], tasks)
    success = {
        **plan, 'metrics': metrics,
        'results_sha256': sha256_file(run_dirs[0] / 'results.json'),
        'checkpoint_sha256': sha256_file(
            run_dirs[0] / 'checkpoints' / f'event_{tasks - 1}_CIL.pt'),
    }
    (job_dir / 'JOB_SUCCESS.json').write_text(
        json.dumps(success, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    return success


def summarize(results_root):
    root = Path(results_root)
    rows = []
    commits = set()
    for dataset in DATASETS:
        for method in METHODS:
            records = []
            for seed in SEEDS:
                path = root / dataset / method / str(seed) / 'JOB_SUCCESS.json'
                if not path.is_file():
                    raise FileNotFoundError(f'missing audited job: {path}')
                record = json.loads(path.read_text(encoding='utf-8'))
                if ((record['dataset'], record['method'], record['seed'])
                        != (dataset, method, seed)
                        or record['smoke']):
                    raise ValueError('result identity or smoke status differs')
                verify_success_record(path.parent, record)
                commits.add(record['code_commit'])
                records.append(record)
            row = {'dataset': dataset, 'method': method}
            for metric in ('aa_final_cil', 'aa_avg_cil', 'bwt_cil',
                           'task_il_final'):
                values = [record['metrics'][metric] for record in records]
                row[metric + '_mean'] = statistics.fmean(values)
                row[metric + '_std'] = statistics.stdev(values)
            rows.append(row)
    if len(commits) != 1:
        raise ValueError('replay-11 results mix code commits')
    output = root / PILOT_TABLE_NAME
    with output.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('plan', 'check', 'run-one',
                                           'run-shard', 'summarize'))
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--results-root', type=Path)
    parser.add_argument('--data-root', type=Path, default=fair.ROOT / 'data')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard-index', type=int)
    parser.add_argument('--shard-count', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.action == 'plan':
        print(json.dumps(jobs()))
    elif args.action == 'check':
        for dataset in (DATASETS if args.dataset is None else (args.dataset,)):
            verify_data(dataset, args.data_root)
        print('replay11_data_verified')
    elif args.action == 'summarize':
        if args.results_root is None:
            parser.error('--results-root is required')
        print(summarize(args.results_root))
    elif args.action == 'run-one':
        if (args.dataset is None or args.method is None or args.seed is None
                or args.results_root is None):
            parser.error('run-one requires dataset, method, seed, results-root')
        print(json.dumps(run_one(
            (args.dataset, args.method, args.seed), args.results_root,
            data_root=args.data_root, device=args.device, smoke=args.smoke),
            sort_keys=True))
    else:
        if (args.shard_index is None or args.shard_count is None
                or args.results_root is None):
            parser.error('run-shard requires shard-index/count and results-root')
        for job in jobs_for_shard(args.shard_index, args.shard_count):
            if args.dataset is None or job[0] == args.dataset:
                run_one(job, args.results_root, data_root=args.data_root,
                        device=args.device, smoke=args.smoke)


if __name__ == '__main__':
    main()
