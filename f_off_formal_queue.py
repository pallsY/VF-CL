"""One-attempt, two-GPU fail-stop queue for the disclosed F-off formal runs."""

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from f_off_formal_manifest import (
    ROOT, STAGE_A_SHA, digest, expected_keys, load_approved_job,
    pinned_checkout, validate_stage_a,
)
from f_off_formal_worker import validate_sources
from party_drift_queue import classify_failure, queue_run_lock, reserve_attempt
from replay_fim_queue import GPU_UUID, _write_once, gpu_eligible, gpu_snapshot, wait_headroom

STAGE_A_DECISION = Path('/brf/users/yangxx/VFCL-results/allparty-fim-final-attempt-v1-20261029/adjudication/STAGE_A_DECISION.json')


def validate_rows(rows):
    if (len(rows) != 9 or {row.get('job_key') for row in rows} != expected_keys()
            or any(row.get('physical_gpu') not in (0, 1) for row in rows)):
        raise ValueError('formal queue allowlist is incomplete or invalid')


def verify_success(run, row, manifest_sha):
    gate_path = run / 'JOB_SUCCESS.json'
    gate = json.loads(gate_path.read_text(encoding='utf-8'))
    if (gate.get('status') != 'success' or gate.get('job_key') != row['job_key']
            or gate.get('stage') != 'formal_followup'
            or gate.get('manifest_sha256') != manifest_sha):
        raise ValueError('formal job success identity differs')
    for field, path in (
        ('results_sha256', run / 'results.json'),
        ('checkpoint_sha256', run / 'checkpoints/formal_final.pt'),
        ('published_sha256', run / 'FORMAL_EVALUATION_PUBLISHED.json'),
        ('complete_sha256', run / 'FORMAL_EVALUATION_COMPLETE.json'),
        ('sealed_sha256', run / 'FORMAL_EVALUATION_SEALED.json'),
        ('fim_mask_audit_sha256', run / 'FIM_MASK_AUDIT.json'),
        ('data_flow_sha256', run / 'data_flow_audit.jsonl'),
    ):
        if digest(path) != gate.get(field):
            raise ValueError(f'formal job artifact differs: {field}')
    return gate_path


def run_queue(manifest, manifest_sha, *, preflight=False):
    checkout = Path(__file__).resolve().parent
    commit = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'],
                                     text=True).strip()
    pinned_checkout(commit)
    manifest = Path(manifest).resolve()
    result_root = Path(ROOT / 'formal')
    if (manifest != Path(ROOT / 'manifests/formal/manifest.jsonl')
            or digest(manifest) != manifest_sha):
        raise ValueError('formal manifest differs from sealed location or SHA')
    validate_stage_a(STAGE_A_DECISION, STAGE_A_SHA)
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    validate_rows(rows)
    jobs = []
    for row in rows:
        approved = load_approved_job(row['job_key'], manifest, manifest_sha, commit)
        config_path = (manifest.parent / row['config_relpath']).resolve()
        config = json.loads(config_path.read_text(encoding='utf-8'))
        validate_sources(config, approved)
        run = Path(config['output_dir'])
        if (not run.is_relative_to(result_root)
                or config['study_physical_gpu'] != row['physical_gpu']):
            raise ValueError('formal result root or physical GPU differs')
        jobs.append((row, config, run))
    if result_root.exists():
        raise FileExistsError('formal result root already exists; no restart')
    if preflight:
        return {'status': 'preflight_passed', 'jobs': len(jobs),
                'manifest_sha256': manifest_sha, 'code_commit': commit}

    with queue_run_lock(result_root):
        result_root.mkdir(parents=True, exist_ok=False)
        _write_once(result_root / 'QUEUE_LOCK.json', {
            'pid': os.getpid(), 'stage': 'formal_followup',
            'status': 'running', 'manifest_sha256': manifest_sha,
            'production_commit': commit,
            'started_utc': datetime.now(timezone.utc).isoformat(),
        })
        events = result_root / 'ATTEMPTS.jsonl'
        events.open('x', encoding='utf-8').close()
        mutex, stop = threading.Lock(), threading.Event()
        outcomes = {}

        def append(value):
            with mutex, events.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(value, sort_keys=True) + '\n')
                stream.flush()
                os.fsync(stream.fileno())

        def lane(index):
            assigned = sorted((job for job in jobs
                               if job[0]['physical_gpu'] == index),
                              key=lambda job: job[0]['job_key'])
            for row, config, job_root in assigned:
                if stop.is_set():
                    return
                while not stop.is_set():
                    admission = wait_headroom(index, stop)
                    if admission is None:
                        return
                    current = gpu_snapshot(index)
                    if current['uuid'] == GPU_UUID[index] and gpu_eligible(current, index):
                        break
                if stop.is_set():
                    return
                key = row['job_key']
                reserve_attempt(job_root, 1, key)
                log = job_root / 'attempt-01.log'
                append({'kind': 'start', 'job_key': key, 'attempt': 1,
                        'config_sha256': row['config_sha256'],
                        'gpu_index': index, 'gpu_uuid': GPU_UUID[index],
                        'time_utc': datetime.now(timezone.utc).isoformat()})
                environment = os.environ.copy()
                environment.update({
                    'CUDA_VISIBLE_DEVICES': GPU_UUID[index],
                    'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                    'PYTHONHASHSEED': str(config['seed']),
                    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                })
                with log.open('x', encoding='utf-8') as stream:
                    process = subprocess.run([
                        sys.executable, '-u', str(checkout / 'f_off_formal_worker.py'),
                        '--job-key', key, '--manifest', str(manifest),
                        '--manifest-sha256', manifest_sha,
                    ], cwd=checkout, env=environment, stdout=stream,
                       stderr=subprocess.STDOUT)
                gate_path = job_root / 'attempt-01/JOB_SUCCESS.json'
                if process.returncode == 0 and gate_path.is_file():
                    try:
                        verify_success(gate_path.parent, row, manifest_sha)
                        classification = 'success'
                    except (ValueError, KeyError, FileNotFoundError):
                        classification = 'unknown'
                else:
                    classification = classify_failure(process.returncode or 1,
                        log.read_text(encoding='utf-8', errors='replace')[-100000:])
                with mutex:
                    outcomes[key] = classification
                append({'kind': 'finish', 'job_key': key, 'attempt': 1,
                        'classification': classification,
                        'returncode': process.returncode,
                        'log_sha256': digest(log),
                        'job_success_sha256': (digest(gate_path)
                                               if classification == 'success' else None),
                        'time_utc': datetime.now(timezone.utc).isoformat()})
                if classification != 'success':
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
        complete = not stop.is_set() and len(outcomes) == 9 and all(
            value == 'success' for value in outcomes.values())
        record = {
            'status': 'complete' if complete else 'failed_or_interrupted',
            'stage': 'formal_followup', 'manifest_sha256': manifest_sha,
            'code_commit': commit, 'accounted_for': len(outcomes),
            'successes': sum(value == 'success' for value in outcomes.values()),
            'attempts_sha256': digest(events),
            'ended_utc': datetime.now(timezone.utc).isoformat(),
            'error': str(error) if error else None,
        }
        _write_once(result_root / ('QUEUE_COMPLETE.json' if complete else
                                   'QUEUE_FAILURE.json'), record)
        if not complete:
            raise RuntimeError('formal queue stopped; preserve all attempts and logs')
        return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run_queue(args.manifest, args.manifest_sha256,
                               preflight=args.preflight), sort_keys=True))
