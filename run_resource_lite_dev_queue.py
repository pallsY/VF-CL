"""Run the fixed training-only paired development trials when a GPU is idle."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_resource_lite_pilot import PROTOCOL, validate_training_only_artifacts


DATASETS = ('isolet', 'upmc_food101')
SEEDS = (54, 55)
VARIANTS = ('lite18', 'reference20')
ALLOWED_PAIR_DIFFERENCES = {
    'dep_tracking_enabled', 'distill_weight', 'exp_name',
    'expected_party_kd_variant', 'head_consolidation_mode',
    'head_consolidation_samples_per_class', 'lambda_validation_enabled',
    'output_dir', 'party_kd_enabled', 'pilot_variant', 'resource_lite',
}


def jobs(root):
    return [
        (dataset, seed, variant,
         root / f'{dataset}_seed{seed}_{variant}.config.json')
        for dataset in DATASETS for seed in SEEDS for variant in VARIANTS
    ]


def validate_plan(root):
    """Refuse a changed protocol or a trial that already has an output tree."""
    for dataset in DATASETS:
        for seed in SEEDS:
            configs = {}
            for variant in VARIANTS:
                path = root / f'{dataset}_seed{seed}_{variant}.config.json'
                config = json.loads(path.read_text(encoding='utf-8'))
                if (config.get('pilot_protocol') != PROTOCOL
                        or config.get('pilot_smoke') is not False
                        or config.get('pilot_dataset') != dataset
                        or config.get('pilot_variant') != variant
                        or config.get('seed') != seed
                        or config.get('num_parties') != 4
                        or Path(config['output_dir']).exists()):
                    raise ValueError(f'development plan or output differs: {path}')
                configs[variant] = config
            left, right = configs['lite18'], configs['reference20']
            differences = {
                key for key in left
                if left[key] != right[key]
            }
            if differences != ALLOWED_PAIR_DIFFERENCES:
                raise ValueError(
                    f'paired config differences changed: {dataset}/{seed} '
                    f'{sorted(differences)}')


def gpu_is_idle(index):
    query = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True,
    )
    rows = [row.strip().split(', ') for row in query.splitlines() if row.strip()]
    matches = [row for row in rows if int(row[0]) == index]
    if len(matches) != 1:
        raise RuntimeError(f'GPU {index} is missing or ambiguous')
    _id, uuid, memory, utilization = matches[0]
    apps = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid',
         '--format=csv,noheader'], text=True,
    )
    return (uuid not in apps and int(memory) < 800
            and int(utilization) < 10)


def atomic_new_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def checked_commit(repo, expected):
    actual = subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True,
    ).strip()
    dirty = subprocess.check_output(
        ['git', '-C', str(repo), 'status', '--porcelain'], text=True,
    ).strip()
    if actual != expected or dirty:
        raise RuntimeError('development producer checkout changed')


def run_queue(root, repo, expected_commit, interval):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    validate_plan(root)
    checked_commit(repo, expected_commit)
    atomic_new_json(root / 'QUEUE_LOCK.json', {
        'pid': os.getpid(), 'producer_commit': expected_commit,
        'jobs': [str(job[3]) for job in jobs(root)],
    })
    try:
        for dataset, seed, variant, path in jobs(root):
            checked_commit(repo, expected_commit)
            config = json.loads(path.read_text(encoding='utf-8'))
            gpu = int(config['device'].split(':')[1])
            while True:
                if not gpu_is_idle(gpu):
                    print(f'WAIT {dataset}/{seed}/{variant}: GPU {gpu} busy',
                          flush=True)
                    time.sleep(interval)
                    continue
                time.sleep(10)
                if gpu_is_idle(gpu):
                    break
                print(f'WAIT {dataset}/{seed}/{variant}: GPU {gpu} changed',
                      flush=True)
                time.sleep(interval)
            log = root / f'{dataset}_seed{seed}_{variant}.training.log'
            env = os.environ.copy()
            env.update(
                CUBLAS_WORKSPACE_CONFIG=':4096:8',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                PYTHONHASHSEED=str(seed),
            )
            print(f'START {dataset}/{seed}/{variant} GPU {gpu}', flush=True)
            with log.open('x', encoding='utf-8') as stream:
                result = subprocess.run(
                    [sys.executable, str(repo / 'run_resource_lite_pilot.py'),
                     'run', '--config', str(path)],
                    cwd=repo, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, check=False,
                )
            run = Path(config['output_dir'])
            if result.returncode:
                raise RuntimeError(
                    f'{dataset}/{seed}/{variant} exited {result.returncode}')
            validate_training_only_artifacts(run)
            readout = json.loads(
                (run / 'HOLDOUT_READOUT.json').read_text(encoding='utf-8'))
            if (readout.get('status') != 'independent_training_holdout_readout'
                    or readout.get('variant') != variant
                    or readout.get('seed') != seed
                    or readout.get('dataset') != dataset):
                raise RuntimeError(f'{dataset}/{seed}/{variant} readout invalid')
            if variant == 'reference20':
                lite = json.loads((root / f'{dataset}_seed{seed}_lite18'
                                   / 'HOLDOUT_READOUT.json').read_text())
                if lite['holdout_manifest_sha256'] != readout['holdout_manifest_sha256']:
                    raise RuntimeError(f'{dataset}/{seed} holdout mismatch')
            print(f'PASS {dataset}/{seed}/{variant}', flush=True)
        atomic_new_json(root / 'QUEUE_COMPLETE.json', {
            'status': 'development_holdout_complete',
            'producer_commit': expected_commit, 'job_count': 8,
        })
        print('COMPLETE eight training-only development trials', flush=True)
    except Exception as error:
        atomic_new_json(root / 'QUEUE_FAILURE.json', {
            'error': f'{type(error).__name__}: {error}',
            'producer_commit': expected_commit,
        })
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--poll-seconds', type=int, default=180)
    args = parser.parse_args()
    run_queue(args.root, args.repo, args.expected_commit, args.poll_seconds)
