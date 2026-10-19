"""Fail-stop queue for eight paired training-holdout KD development jobs."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from run_sampleonly_fim_queue import gpu_idle


ROOT = Path('/brf/users/yangxx/VFCL-results/oldclass-kd-paired-dev-v1-20261019')
FIM_ROOT = Path('/brf/users/yangxx/VFCL-results/sampleonly-fim-off-dev-v1-20261019')
PLAN_SHA256 = '135e688455f606f3733bb6e71acb53c1905fb49545cb1519a4830656671add7c'
PYTHON = '/brf/users/yangxx/VF-CL-distributed/3fa15e20c636cd123d3a266b065b30e5bd0b57b8/env-cp312/bin/python'
CHECKOUTS = {
    'oldonly20': (Path('/tmp/vfcl-kd-oldonly-v1'),
                  'ad01e8dc180f5cab62a61ec8f78dd8aba5b096b8',
                  Path('/brf/users/yangxx/VFCL-results/oldclass-kd-dev-driver-smoke-20261019')),
    'fullwidth20': (Path('/tmp/vfcl-kd-fullwidth-control-v1'),
                    '40dc7c3588677adea80649d31926910d11bb5fe6',
                    Path('/brf/users/yangxx/VFCL-results/fullwidth-kd-dev-driver-smoke-20261019')),
}


def say(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_cell_keys(cells):
    required = {
        (dataset, seed, variant)
        for dataset in ('isolet', 'upmc_food101')
        for seed in (56, 57)
        for variant in CHECKOUTS
    }
    observed = [(cell['dataset'], cell['seed'], cell['variant']) for cell in cells]
    if len(observed) != 8 or set(observed) != required:
        raise ValueError('pair plan must contain eight unique expected cells')
    return required


def check_checkout(repo, commit):
    head = subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(
        ['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
    if head != commit or dirty:
        raise RuntimeError(f'checkout commit or cleanliness differs: {repo}')


def check_plan():
    path = ROOT / 'PAIR_PLAN.json'
    if sha256(path) != PLAN_SHA256:
        raise RuntimeError('paired plan differs from GitHub anchor')
    plan = json.loads(path.read_text(encoding='utf-8'))
    expected_cell_keys(plan['cells'])
    if (plan.get('protocol') != 'old-class-kd-training-holdout-v1'
            or plan.get('holdout_seed') != 20261019
            or plan.get('status') != 'prepared_not_run'):
        raise RuntimeError('paired plan protocol differs')
    for cell in plan['cells']:
        _, commit, _ = CHECKOUTS[cell['variant']]
        config_path = ROOT / cell['config_file']
        if (cell['code_commit'] != commit
                or sha256(config_path) != cell['config_sha256']):
            raise RuntimeError(f'paired config hash/commit differs: {config_path}')
        config = json.loads(config_path.read_text(encoding='utf-8'))
        if (config['pilot_code_commit'] != commit
                or config['pilot_variant'] != cell['variant']
                or config['seed'] != cell['seed']
                or config['pilot_dataset'] != cell['dataset']
                or config['device'] != cell['device']
                or config['pilot_holdout_seed'] != 20261019
                or Path(config['output_dir']).exists()):
            raise RuntimeError(f'paired job already exists or config differs: {config_path}')
    for repo, commit, _ in CHECKOUTS.values():
        check_checkout(repo, commit)
    for dataset in ('isolet', 'upmc_food101'):
        hashes = []
        for variant, (_, _, smoke_root) in CHECKOUTS.items():
            run = smoke_root / f'{dataset}_seed54_{variant}_smoke'
            gate = json.loads((run / 'PREFLIGHT_GATE.json').read_text())
            if (gate.get('status') != 'training_only_preflight'
                    or gate.get('variant') != variant
                    or not (run / 'FORMAL_EVALUATION_PENDING.json').is_file()
                    or (run / 'results.json').exists()):
                raise RuntimeError(f'{dataset}/{variant} preflight did not pass')
            hashes.append(gate['holdout_manifest_sha256'])
        if hashes[0] != hashes[1]:
            raise RuntimeError(f'{dataset} paired preflight holdout differs')
    return plan


def wait_for_fim():
    lock = json.loads((FIM_ROOT / 'QUEUE_LOCK.json').read_text())
    pid = int(lock['pid'])
    while True:
        if (FIM_ROOT / 'QUEUE_FAILURE.json').exists():
            raise RuntimeError('FIM development queue failed; paired KD remains stopped')
        done = FIM_ROOT / 'QUEUE_COMPLETE.json'
        if done.exists():
            record = json.loads(done.read_text())
            if (record.get('status') != 'complete'
                    or len(record.get('readout_sha256', {})) != 4):
                raise RuntimeError('FIM queue completion is invalid')
            return
        try:
            os.kill(pid, 0)
        except ProcessLookupError as error:
            raise RuntimeError('FIM queue exited without a terminal record') from error
        say('waiting for the existing FIM queue to finish')
        time.sleep(60)


def wait_for_gpus():
    clear = 0
    while clear < 2:
        if gpu_idle(0) and gpu_idle(1):
            clear += 1
        else:
            clear = 0
            say('waiting for both 3080 GPUs; other users retain priority')
        if clear < 2:
            time.sleep(30)


def run_one(cell):
    repo, commit, _ = CHECKOUTS[cell['variant']]
    check_checkout(repo, commit)
    config_path = ROOT / cell['config_file']
    config = json.loads(config_path.read_text())
    output = Path(config['output_dir'])
    if output.exists():
        raise FileExistsError(f'job already exists: {output}')
    log = ROOT / (output.name + '.training.log')
    env = os.environ.copy()
    env.update({
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'PYTHONHASHSEED': str(cell['seed']),
        'PYTHONPATH': str(repo),
    })
    command = ['nice', '-n', '10', PYTHON, '-u',
               str(repo / 'run_old_class_kd_dev.py'), 'run',
               '--config', str(config_path)]
    with log.open('x', encoding='utf-8') as stream:
        completed = subprocess.run(command, cwd=repo, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f'{output.name} exited {completed.returncode}: {log}')
    readout = output / 'HOLDOUT_READOUT.json'
    record = json.loads(readout.read_text())
    if (record.get('status') != 'paired_training_holdout_readout'
            or record.get('code_commit') != commit
            or record.get('config_sha256') != cell['config_sha256']
            or record.get('raw_replay_per_class') != 20
            or record.get('prototype_bytes') != 0
            or not math.isfinite(record['metrics']['cil'])
            or (output / 'results.json').exists()):
        raise RuntimeError(f'{output.name} readout or test-access gate failed')
    say(f'PASSED {output.name}')
    return record, sha256(readout)


def run_pair(plan, dataset, seed):
    wait_for_gpus()
    cells = [cell for cell in plan['cells']
             if cell['dataset'] == dataset and cell['seed'] == seed]
    say(f'starting paired {dataset} seed {seed}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run_one, cell): cell['variant'] for cell in cells}
        results = {}
        errors = []
        for future in concurrent.futures.as_completed(futures):
            try:
                results[futures[future]] = future.result()
            except Exception as error:
                errors.append(str(error))
    if errors:
        raise RuntimeError('; '.join(errors))
    left, right = results['oldonly20'][0], results['fullwidth20'][0]
    if left['holdout_manifest_sha256'] != right['holdout_manifest_sha256']:
        raise RuntimeError(f'{dataset}/{seed} holdout manifests differ')
    pair = {
        'dataset': dataset, 'seed': seed,
        'holdout_manifest_sha256': left['holdout_manifest_sha256'],
        'oldonly_cil': left['metrics']['cil'],
        'fullwidth_cil': right['metrics']['cil'],
        'delta_cil': left['metrics']['cil'] - right['metrics']['cil'],
        'readout_sha256': {key: value[1] for key, value in results.items()},
    }
    with (ROOT / f'PAIR_{dataset}_{seed}.json').open('x') as stream:
        json.dump(pair, stream, indent=2, sort_keys=True)
        stream.write('\n')
    say(f'paired {dataset} seed {seed} passed readout gates')
    return pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-commit', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
            != args.queue_commit
            or subprocess.check_output(
                ['git', 'status', '--porcelain'], text=True).strip()):
        raise RuntimeError('queue checkout commit or cleanliness differs')
    plan = check_plan()
    if args.preflight:
        say('paired plan, checkouts and vector preflights verified')
        return
    if any((ROOT / name).exists() for name in
           ('QUEUE_LOCK.json', 'QUEUE_FAILURE.json', 'QUEUE_COMPLETE.json')):
        raise FileExistsError('paired queue already started or reached a terminal state')
    with (ROOT / 'QUEUE_LOCK.json').open('x') as stream:
        json.dump({'pid': os.getpid(), 'status': 'running',
                   'plan_sha256': PLAN_SHA256,
                   'queue_commit': args.queue_commit}, stream, sort_keys=True)
        stream.write('\n')
    try:
        wait_for_fim()
        pairs = [run_pair(plan, dataset, seed)
                 for seed in (56, 57)
                 for dataset in ('isolet', 'upmc_food101')]
        with (ROOT / 'QUEUE_COMPLETE.json').open('x') as stream:
            json.dump({'status': 'complete', 'pairs': pairs},
                      stream, indent=2, sort_keys=True)
            stream.write('\n')
        say('all eight paired KD development cells passed')
    except Exception as error:
        with (ROOT / 'QUEUE_FAILURE.json').open('x') as stream:
            json.dump({'status': 'failed', 'reason': str(error)},
                      stream, indent=2, sort_keys=True)
            stream.write('\n')
        say(f'FAIL STOP: {error}')
        raise


if __name__ == '__main__':
    main()
