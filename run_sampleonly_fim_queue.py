"""Fail-stop queue for four training-only zero-FIM development cells."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


REPO = Path('/tmp/vfcl-fim-dev-v1')
PYTHON = Path('/brf/users/yangxx/VF-CL-distributed/3fa15e20c636cd123d3a266b065b30e5bd0b57b8/env-cp312/bin/python')
REFERENCE = Path('/brf/users/yangxx/VFCL-results/resource-lite-dev-full-v2-20261018')
SMOKE = Path('/brf/users/yangxx/VFCL-results/sampleonly-fim-off-smoke-v2-20261019')
ROOT = Path('/brf/users/yangxx/VFCL-results/sampleonly-fim-off-dev-v1-20261019')


def say(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_root(root, commit):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    with (root / 'QUEUE_LOCK.json').open('x', encoding='utf-8') as stream:
        json.dump({'producer_commit': commit, 'pid': os.getpid(),
                   'status': 'running'}, stream, sort_keys=True)
        stream.write('\n')


def gpu_idle_from_rows(devices, apps, index):
    parsed = {}
    for row in devices:
        fields = [part.strip() for part in row.split(',')]
        if len(fields) != 4:
            raise ValueError('GPU inventory row has wrong shape')
        parsed[int(fields[0])] = (fields[1], int(fields[2]), int(fields[3]))
    if index not in parsed:
        raise ValueError(f'GPU {index} missing')
    uuid, memory, utilization = parsed[index]
    busy_uuids = {line.split(',')[0].strip() for line in apps if line.strip()}
    return (uuid not in busy_uuids and memory < 512 and utilization < 10)


def gpu_idle(index):
    devices = subprocess.check_output([
        'nvidia-smi',
        '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits',
    ], text=True).splitlines()
    apps = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader',
    ], text=True).splitlines()
    return gpu_idle_from_rows(devices, apps, index)


def wait_for_gpus():
    clear = 0
    while clear < 2:
        if gpu_idle(0) and gpu_idle(1):
            clear += 1
        else:
            clear = 0
            say('waiting for GPU 0 and GPU 1; other users retain priority')
        if clear < 2:
            time.sleep(30)


def checked_source(commit):
    head = subprocess.check_output(
        ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(
        ['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip()
    if head != commit or dirty:
        raise RuntimeError('development checkout commit or cleanliness differs')
    for dataset in ('isolet', 'upmc_food101'):
        smoke = SMOKE / f'{dataset}_seed54_fim_off20_smoke'
        gate = json.loads((smoke / 'SMOKE_GATE.json').read_text())
        if (gate.get('status') != 'full_schedule_one_epoch_smoke'
                or gate.get('variant') != 'fim_off20'
                or not (smoke / 'FORMAL_EVALUATION_PENDING.json').is_file()
                or (smoke / 'results.json').exists()
                or (smoke / 'formal_access/test.consumed.json').exists()):
            raise RuntimeError(f'{dataset} FIM preflight did not pass')


def run_one(dataset, seed, commit):
    name = f'{dataset}_seed{seed}_fim_off20'
    source = REFERENCE / f'{dataset}_seed{seed}_reference20.config.json'
    result_dir = ROOT / name
    if result_dir.exists():
        raise FileExistsError(f'existing development job: {result_dir}')
    env = os.environ.copy()
    env.update({
        'PYTHONPATH': str(REPO), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'PYTHONHASHSEED': str(seed),
    })
    command = [str(PYTHON), str(REPO / 'run_sampleonly_partykd_ablation_dev.py'),
               '--source-config', str(source), '--root', str(ROOT),
               '--mode', 'fim_off']
    log_path = ROOT / f'{name}.training.log'
    with log_path.open('x', encoding='utf-8') as log:
        completed = subprocess.run(command, cwd=REPO, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f'{name} failed; inspect {log_path}')
    freezes = re.findall(r'FIM freeze: (\d+)/', log_path.read_text())
    expected = json.loads(source.read_text())['num_tasks'] - 1
    if len(freezes) != expected or any(int(count) != 0 for count in freezes):
        raise RuntimeError(f'{name} had a nonzero or missing freeze count')
    readout = result_dir / 'HOLDOUT_READOUT.json'
    record = json.loads(readout.read_text())
    reference = json.loads((REFERENCE / f'{dataset}_seed{seed}_reference20'
                            / 'HOLDOUT_READOUT.json').read_text())
    if (record.get('status') != 'independent_training_holdout_readout'
            or record.get('variant') != 'fim_off20'
            or record.get('source_config_sha256') != digest(source)
            or record.get('holdout_manifest_sha256')
            != reference.get('holdout_manifest_sha256')
            or not (result_dir / 'FORMAL_EVALUATION_PENDING.json').is_file()
            or (result_dir / 'results.json').exists()
            or (result_dir / 'formal_access/test.consumed.json').exists()):
        raise RuntimeError(f'{name} readout or test-access gate failed')
    say(f'PASSED {name}')
    return name, digest(readout)


def stage(seed, commit):
    wait_for_gpus()
    checked_source(commit)
    say(f'starting seed {seed} on both idle GPUs')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_one, dataset, seed, commit)
                   for dataset in ('isolet', 'upmc_food101')]
        results = []
        errors = []
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(str(error))
    if errors:
        raise RuntimeError('; '.join(errors))
    return dict(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    checked_source(args.commit)
    prepare_root(ROOT, args.commit)
    try:
        results = {}
        results.update(stage(54, args.commit))
        results.update(stage(55, args.commit))
        with (ROOT / 'QUEUE_COMPLETE.json').open('x', encoding='utf-8') as stream:
            json.dump({'status': 'complete', 'producer_commit': args.commit,
                       'readout_sha256': results}, stream, indent=2,
                      sort_keys=True)
            stream.write('\n')
        say('all four FIM-off development cells passed readout gates')
    except Exception as error:
        with (ROOT / 'QUEUE_FAILURE.json').open('x', encoding='utf-8') as stream:
            json.dump({'status': 'failed', 'reason': str(error)}, stream,
                      indent=2, sort_keys=True)
            stream.write('\n')
        say(f'FAIL STOP: {error}')
        raise


if __name__ == '__main__':
    main()
