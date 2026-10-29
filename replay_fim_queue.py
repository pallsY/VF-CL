"""Fail-stop, one-attempt development queue for the final FIM study."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from party_drift_audit_gate import verify_sealed_job
from party_drift_queue import classify_failure, queue_run_lock, reserve_attempt
from replay_fim_manifest import expected_job_keys, load_approved_job
from run_party_drift_development import _sha256

MIN_FREE_MIB = {0: 3072, 1: 5120}
GPU_UUID = {
    0: 'GPU-6c7fee92-a6fe-5525-c554-bddb8e14943f',
    1: 'GPU-6569e526-43ea-74e1-264e-4b66417181e1',
}


def gpu_eligible(snapshot, index):
    return (snapshot['index'] == index and
            snapshot['name'] == 'NVIDIA GeForce RTX 3080' and
            snapshot['uuid'] == GPU_UUID[index] and
            snapshot['free_mib'] >= MIN_FREE_MIB[index])


def must_stop(classification):
    return classification != 'success'


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_once(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def gpu_snapshot(index):
    output = subprocess.run([
        'nvidia-smi', '-i', str(index),
        '--query-gpu=index,name,uuid,memory.free',
        '--format=csv,noheader,nounits',
    ], capture_output=True, text=True, check=True).stdout.strip()
    fields = [item.strip() for item in output.split(',')]
    if len(fields) != 4:
        raise ValueError('unexpected GPU snapshot format')
    return {'index': int(fields[0]), 'name': fields[1],
            'uuid': fields[2], 'free_mib': int(fields[3])}


def wait_headroom(index, stop, *, snapshot_fn=gpu_snapshot,
                  clock_fn=time.monotonic):
    since, uuid = None, None
    while not stop.is_set():
        snapshot = snapshot_fn(index)
        if snapshot['index'] != index or snapshot['uuid'] != GPU_UUID[index]:
            raise ValueError('physical GPU identity changed')
        if uuid is None:
            uuid = snapshot['uuid']
        elif uuid != snapshot['uuid']:
            raise ValueError('GPU UUID changed during admission')
        now = clock_fn()
        since = (now if since is None else since) if gpu_eligible(
            snapshot, index) else None
        if since is not None and now - since >= 120:
            current = snapshot_fn(index)
            if current['uuid'] != uuid:
                raise ValueError('GPU UUID changed before launch')
            if gpu_eligible(current, index):
                return current
            since = None
        stop.wait(20)
    return None


def verify_job_gate(run, key, digest, replay):
    gate = verify_sealed_job(run, key, digest, replay=replay)
    mask = Path(run) / 'FIM_MASK_AUDIT.json'
    expected = _sha256(mask) if mask.is_file() else None
    if (gate.get('fim_mask_audit_sha256') != expected or
            (replay and expected is None) or
            (not replay and expected is not None)):
        raise ValueError('FIM mask audit is absent or changed')
    if replay:
        audit = json.loads(mask.read_text(encoding='utf-8'))
        snapshots = Path(run) / 'formal_snapshots'
        for row in audit['tasks']:
            path = snapshots / f"event_{row['task_id']}_CIL.pt"
            if row.get('snapshot_sha256') != _sha256(path):
                raise ValueError('FIM snapshot differs from mask audit')
    return gate


def run_queue(manifest, digest, result_root, *, stage, selected_policy=None):
    checkout = Path(__file__).resolve().parent
    commit = subprocess.check_output([
        'git', '-C', str(checkout), 'rev-parse', 'HEAD'],
        text=True).strip()
    manifest = Path(manifest).resolve()
    result_root = Path(result_root).resolve()
    if _sha256(manifest) != digest:
        raise ValueError('queue manifest differs from SHA anchor')
    expected = expected_job_keys(stage, selected_policy)
    rows = [json.loads(line) for line in manifest.read_text(
        encoding='utf-8').splitlines()]
    if len(rows) != len(expected) or {row.get('job_key') for row in rows} != expected:
        raise ValueError('queue job allowlist is incomplete or duplicated')
    jobs = []
    for row in rows:
        load_approved_job(row['job_key'], manifest, digest,
                          code_commit=commit, stage=stage,
                          selected_policy=selected_policy)
        config_path = (manifest.parent / row['config_relpath']).resolve()
        config = json.loads(config_path.read_text(encoding='utf-8'))
        job_root = Path(config['output_dir']).resolve()
        if (not job_root.is_relative_to(result_root) or
                config['development_job_key'] != row['job_key'] or
                config['device'] != 'cuda:0' or
                config['study_physical_gpu'] not in (0, 1) or
                config['study_physical_gpu'] != row['physical_gpu']):
            raise ValueError('approved job result or GPU binding differs')
        jobs.append((row, config, job_root))
    if result_root.exists():
        raise FileExistsError('study result root already exists; no automatic restart')
    with queue_run_lock(result_root):
        result_root.mkdir(parents=True, exist_ok=False)
        _write_once(result_root / 'QUEUE_LOCK.json', {
            'pid': os.getpid(), 'status': 'running', 'stage': stage,
            'selected_policy': selected_policy, 'manifest_sha256': digest,
            'production_commit': commit, 'started_utc': _now(),
        })
        events = result_root / 'ATTEMPTS.jsonl'
        with events.open('x', encoding='utf-8'):
            pass
        mutex, stop = threading.Lock(), threading.Event()
        outcomes = {}

        def append(value):
            with mutex, events.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(value, sort_keys=True) + '\n')
                stream.flush()
                os.fsync(stream.fileno())

        def lane(index):
            assigned = sorted((job for job in jobs
                               if job[1]['study_physical_gpu'] == index),
                              key=lambda job: job[0]['job_key'])
            for row, config, job_root in assigned:
                if stop.is_set():
                    return
                while not stop.is_set():
                    admission = wait_headroom(index, stop)
                    if admission is None:
                        return
                    launch_snapshot = gpu_snapshot(index)
                    if (launch_snapshot['uuid'] != admission['uuid']):
                        raise ValueError('GPU UUID changed at worker launch')
                    if gpu_eligible(launch_snapshot, index):
                        break
                if stop.is_set():
                    return
                key = row['job_key']
                reserve_attempt(job_root, 1, key)
                log = job_root / 'attempt-01.log'
                append({'kind': 'start', 'job_key': key, 'attempt': 1,
                        'config_sha256': row['config_sha256'],
                        'gpu_index': index, 'gpu_uuid': admission['uuid'],
                        'time_utc': _now()})
                environment = os.environ.copy()
                environment.update({
                    'CUDA_VISIBLE_DEVICES': admission['uuid'],
                    'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                    'PYTHONHASHSEED': str(config['seed']),
                    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                })
                with log.open('x', encoding='utf-8') as stream:
                    process = subprocess.run([
                        sys.executable, '-u',
                        str(checkout / 'run_replay_fim_development.py'),
                        '--job-key', key, '--manifest', str(manifest),
                        '--manifest-sha256', digest, '--stage', stage,
                        *(['--selected-policy', selected_policy]
                          if selected_policy else []),
                    ], cwd=checkout, env=environment, stdout=stream,
                       stderr=subprocess.STDOUT)
                gate_path = job_root / 'attempt-01' / 'JOB_SUCCESS.json'
                if process.returncode == 0 and gate_path.is_file():
                    try:
                        verify_job_gate(gate_path.parent, key, digest,
                                        bool(config['party_drift_enabled']))
                        classification = 'success'
                    except (ValueError, KeyError, FileNotFoundError):
                        classification = 'unknown'
                else:
                    tail = log.read_text(encoding='utf-8', errors='replace')[-100000:]
                    classification = classify_failure(process.returncode or 1, tail)
                with mutex:
                    outcomes[key] = classification
                append({'kind': 'finish', 'job_key': key, 'attempt': 1,
                        'classification': classification,
                        'returncode': process.returncode,
                        'log_sha256': _sha256(log),
                        'job_success_sha256': (_sha256(gate_path)
                                               if classification == 'success' else None),
                        'time_utc': _now()})
                if must_stop(classification):
                    stop.set()
                    return

        error = None
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(lane, index) for index in (0, 1)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    stop.set()
                    error = exc
        complete = not stop.is_set() and len(outcomes) == len(expected) and all(
            value == 'success' for value in outcomes.values())
        record = {
            'status': 'complete' if complete else 'failed_or_interrupted',
            'stage': stage, 'manifest_sha256': digest,
            'code_commit': commit, 'accounted_for': len(outcomes),
            'successes': sum(value == 'success' for value in outcomes.values()),
            'attempts_sha256': _sha256(events), 'ended_utc': _now(),
            'error': str(error) if error else None,
        }
        _write_once(result_root / ('QUEUE_COMPLETE.json' if complete else
                                   'QUEUE_FAILURE.json'), record)
        if not complete:
            raise RuntimeError('study queue stopped; preserve all attempts and logs')
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--stage', choices=('A', 'B'), required=True)
    parser.add_argument('--selected-policy', choices=('f0', 'f1', 'f_off'))
    args = parser.parse_args()
    print(json.dumps(run_queue(args.manifest, args.manifest_sha256,
                               args.result_root, stage=args.stage,
                               selected_policy=args.selected_policy),
                     sort_keys=True))


if __name__ == '__main__':
    main()
