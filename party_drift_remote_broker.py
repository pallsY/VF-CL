"""Single-job remote broker. Never run against the production queue without audit.

The broker reports its PID, waits for GO, and directly spawns the frozen worker.
It never claims keys, edits queue locks, or appends ATTEMPTS.jsonl.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from party_drift_balanced_resume import gpu_snapshot


def broker_once(command, env, incoming, outgoing, *, spawn=subprocess.Popen,
                report=None, cwd=None, log_path=None):
    report = {**(report or {}), 'pid': os.getpid()}
    outgoing.write(json.dumps(report, sort_keys=True) + '\n')
    outgoing.flush()
    if incoming.readline() != 'GO\n':
        raise RuntimeError('broker did not receive exact GO authorization')
    if log_path is None:
        process = spawn(command, env=env, cwd=cwd)
        code = process.wait()
    else:
        with Path(log_path).open('x', encoding='utf-8') as stream:
            process = spawn(command, env=env, cwd=cwd, stdout=stream,
                            stderr=subprocess.STDOUT)
            code = process.wait()
    outgoing.write(json.dumps({'kind': 'exit', 'returncode': code},
                              sort_keys=True) + '\n')
    outgoing.flush()
    return code


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer-checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--production-commit', required=True)
    parser.add_argument('--job-key', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--expected-mount-source', required=True)
    parser.add_argument('--min-disk-free-mib', type=int, default=4096)
    parser.add_argument('--result-root', type=Path, required=True)
    args = parser.parse_args()

    producer = args.producer_checkout.resolve()
    manifest = args.manifest.resolve()
    root = args.result_root.resolve()
    if _sha256(manifest) != args.manifest_sha256:
        raise ValueError('remote manifest bytes differ')
    commit = subprocess.check_output(
        ['git', '-C', str(producer), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(
        ['git', '-C', str(producer), 'status', '--porcelain'], text=True).strip()
    if commit != args.production_commit or dirty:
        raise ValueError('remote frozen producer differs or is dirty')
    mount = subprocess.check_output(
        ['findmnt', '-n', '-o', 'SOURCE', '-T', str(root)], text=True).strip()
    if mount != args.expected_mount_source:
        raise ValueError('remote result mount differs')
    disk_free_mib = shutil.disk_usage(root).free // (1024 * 1024)
    if disk_free_mib < args.min_disk_free_mib:
        raise ValueError('remote result filesystem lacks free space')
    gpu = gpu_snapshot(args.gpu_index)
    if gpu['name'] != 'NVIDIA GeForce RTX 4090' or gpu['uuid'] != args.gpu_uuid:
        raise ValueError('remote GPU identity differs')
    driver = subprocess.check_output([
        'nvidia-smi', '-i', str(args.gpu_index),
        '--query-gpu=driver_version', '--format=csv,noheader,nounits',
    ], text=True).strip()
    import torch
    report = {
        'host': os.uname().nodename,
        'producer_commit': commit,
        'manifest_sha256': args.manifest_sha256,
        'mount_source': mount,
        'result_disk_free_mib': disk_free_mib,
        'gpu': gpu,
        'driver': driver,
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
    }
    seed = int(args.job_key.rsplit('/', 1)[1])
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpu_uuid,
           'PYTHONHASHSEED': str(seed), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    command = [sys.executable, '-u', str(producer / 'run_party_drift_development.py'),
               '--job-key', args.job_key, '--manifest', str(manifest),
               '--manifest-sha256', args.manifest_sha256, '--attempt', '1']
    job_root = root / args.job_key
    if not job_root.is_relative_to(root):
        raise ValueError('remote job escaped result root')
    return broker_once(command, env, sys.stdin, sys.stdout, report=report,
                       cwd=producer, log_path=job_root / 'attempt-01.log')


if __name__ == '__main__':
    raise SystemExit(main())
