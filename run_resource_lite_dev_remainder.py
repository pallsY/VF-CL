"""Finish seven dev trials after recovering the first frozen checkpoint."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_resource_lite_dev_queue import (
    atomic_new_json, checked_commit, gpu_is_idle, jobs, validate_plan,
)
from run_resource_lite_pilot import (
    sha256, validate_training_only_artifacts,
)


def verified_recovery(root, recovery_path):
    recovery_path = Path(recovery_path).resolve()
    recovered = json.loads(recovery_path.read_text(encoding='utf-8'))
    source = Path(recovered['source_run']).resolve()
    expected_source = Path(
        '/brf/users/yangxx/VFCL-results/resource-lite-dev-full-v1-20261018'
        '/isolet_seed54_lite18').resolve()
    if (source != expected_source
            or recovered.get('status') != 'recovered_training_only_holdout_readout'
            or recovered.get('dataset') != 'isolet'
            or recovered.get('seed') != 54
            or recovered.get('variant') != 'lite18'
            or recovered.get('training_commit')
            != '6829581f1cceb2eaa4dd5d66f760a63c1c3df413'):
        raise ValueError('recovered first trial identity differs')
    validate_training_only_artifacts(source)
    checkpoint = source / 'checkpoints' / 'formal_final.pt'
    if (sha256(checkpoint) != recovered['checkpoint_sha256']
            or sha256(source / 'config.json') != recovered['config_sha256']
            or sha256(source / 'pilot_holdout' / 'manifest.json')
            != recovered['holdout_manifest_sha256']):
        raise ValueError('recovered first trial source changed')
    old = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    new = json.loads((root / 'isolet_seed54_lite18.config.json').read_text())
    for config in (old, new):
        for key in ('results_dir', 'output_dir', 'exp_name'):
            config.pop(key)
    if old != new:
        raise ValueError('replacement first config changes training semantics')
    return recovered


def run_remainder(root, repo, expected_commit, recovery_path, interval):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    validate_plan(root)
    recovered = verified_recovery(root, recovery_path)
    checked_commit(repo, expected_commit)
    atomic_new_json(root / 'QUEUE_LOCK.json', {
        'pid': os.getpid(), 'producer_commit': expected_commit,
        'recovered_first_sha256': sha256(recovery_path),
        'pending_jobs': [str(job[3]) for job in jobs(root)[1:]],
    })
    try:
        for dataset, seed, variant, path in jobs(root)[1:]:
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
                time.sleep(interval)
            env = os.environ.copy()
            env.update(
                CUBLAS_WORKSPACE_CONFIG=':4096:8',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                PYTHONHASHSEED=str(seed),
            )
            print(f'START {dataset}/{seed}/{variant} GPU {gpu}', flush=True)
            log = root / f'{dataset}_seed{seed}_{variant}.training.log'
            with log.open('x', encoding='utf-8') as stream:
                finished = subprocess.run(
                    [sys.executable, str(repo / 'run_resource_lite_pilot.py'),
                     'run', '--config', str(path)],
                    cwd=repo, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, check=False,
                )
            run = Path(config['output_dir'])
            if finished.returncode:
                raise RuntimeError(
                    f'{dataset}/{seed}/{variant} exited {finished.returncode}')
            validate_training_only_artifacts(run)
            readout = json.loads(
                (run / 'HOLDOUT_READOUT.json').read_text(encoding='utf-8'))
            if (readout.get('status') != 'independent_training_holdout_readout'
                    or readout.get('dataset') != dataset
                    or readout.get('seed') != seed
                    or readout.get('variant') != variant):
                raise RuntimeError(f'{dataset}/{seed}/{variant} readout invalid')
            if variant == 'reference20':
                lite = (recovered if (dataset, seed) == ('isolet', 54)
                        else json.loads((root / f'{dataset}_seed{seed}_lite18'
                                         / 'HOLDOUT_READOUT.json').read_text()))
                if lite['holdout_manifest_sha256'] != readout['holdout_manifest_sha256']:
                    raise RuntimeError(f'{dataset}/{seed} holdout mismatch')
            print(f'PASS {dataset}/{seed}/{variant}', flush=True)
        atomic_new_json(root / 'QUEUE_COMPLETE.json', {
            'status': 'development_holdout_complete',
            'producer_commit': expected_commit,
            'new_job_count': 7, 'recovered_job_count': 1,
            'recovered_first_sha256': sha256(recovery_path),
        })
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
    parser.add_argument('--recovery-readout', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=int, default=180)
    args = parser.parse_args()
    run_remainder(args.root, args.repo, args.expected_commit,
                  args.recovery_readout, args.poll_seconds)
