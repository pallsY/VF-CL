"""Scheduler-only, append-only resume for the frozen party-drift matrix."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from party_drift_audit_gate import verify_sealed_job
from party_drift_manifest import REPLAY_ARMS, expected_job_keys, load_approved_job
import party_drift_queue as frozen_queue


def idle_snapshot(snapshot):
    return (snapshot['memory_mib'] < 256 and
            snapshot['utilization'] < 10 and
            not snapshot['compute_pids'])


def parse_gpu_snapshot(gpu_line, app_lines):
    fields = [field.strip() for field in gpu_line.split(',')]
    if len(fields) != 5:
        raise ValueError('unexpected GPU inventory format')
    index, name, uuid, memory, utilization = fields
    pids = []
    for line in app_lines.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 2:
            raise ValueError('unexpected compute process record')
        if parts[0] == uuid:
            pids.append(int(parts[1]))
    return {'index': int(index), 'name': name, 'uuid': uuid,
            'memory_mib': int(memory), 'utilization': int(utilization),
            'compute_pids': pids}


def gpu_snapshot(index):
    gpu = subprocess.run([
        'nvidia-smi', '-i', str(index),
        '--query-gpu=index,name,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits',
    ], capture_output=True, text=True, check=True)
    apps = subprocess.run([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits',
    ], capture_output=True, text=True, check=True)
    result = parse_gpu_snapshot(gpu.stdout.strip(), apps.stdout)
    if result['index'] != index:
        raise ValueError('physical GPU index changed')
    return result


def wait_ready(gpu, pending, stop, *, snapshot_fn=None,
               clock_fn=time.monotonic, sleep_fn=None):
    snapshot_fn = snapshot_fn or gpu_snapshot
    sleep_fn = sleep_fn or stop.wait
    required = 600 if gpu == 1 else 30
    idle_since = None
    while pending and not stop.is_set():
        snapshot = snapshot_fn(gpu)
        if (snapshot['name'] != 'NVIDIA GeForce RTX 3080' or
                not snapshot['uuid'].startswith('GPU-')):
            raise ValueError('balanced queue needs the two qualified RTX 3080 cards')
        now = clock_fn()
        idle_since = (now if idle_since is None else idle_since) if idle_snapshot(
            snapshot) else None
        if idle_since is not None and now - idle_since >= required:
            return True
        sleep_fn(30)
    return False


def worker_environment(base, physical_gpu, seed):
    if physical_gpu not in (0, 1):
        raise ValueError('only the two qualified RTX 3080 cards are approved')
    return {**base, 'CUDA_VISIBLE_DEVICES': str(physical_gpu),
            'PYTHONHASHSEED': str(seed),
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


def validate_anchor(value, *, producer, scheduler, manifest, ledger,
                    gpu_uuids):
    expected = {
        'producer_commit': producer, 'scheduler_commit': scheduler,
        'manifest_sha256': manifest, 'source_ledger_sha256': ledger,
        'gpu_uuids': list(gpu_uuids),
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError('balanced resume anchor differs from live inputs')


def validate_manifest_rows(rows, configs, result_root, *, producer, expected):
    keys = [row.get('job_key') for row in rows]
    if len(keys) != len(expected) or set(keys) != set(expected):
        raise ValueError('balanced resume manifest keys differ')
    root = Path(result_root).resolve()
    for row in rows:
        key = row['job_key']
        config = configs[key]
        if (row.get('production_commit') != producer or
                config.get('development_job_key') != key or
                Path(config['output_dir']).resolve() != root / key or
                config.get('seed') != int(key.split('/')[-1]) or
                config.get('device') not in ('cuda:0', 'cuda:1')):
            raise ValueError(f'balanced resume manifest row differs: {key}')


def assert_original_queue_gone(root, manifest, *, proc_root=Path('/proc')):
    lock = frozen_queue.active_queue_lock(root)
    cmdline = Path(proc_root) / str(int(lock['pid'])) / 'cmdline'
    if cmdline.is_file() and b'party_drift_queue.py' in cmdline.read_bytes():
        raise RuntimeError('original queue coordinator is still running')
    workers = frozen_queue.running_worker_keys(manifest, proc_root)
    if workers:
        raise RuntimeError(f'original workers are still running: {sorted(workers)}')


def start_record(job_key, config_sha256, physical_gpu, gpu_uuid,
                 scheduler_commit, producer_commit):
    if physical_gpu not in (0, 1) or not gpu_uuid.startswith('GPU-'):
        raise ValueError('physical GPU assignment is invalid')
    return {
        'kind': 'start', 'job_key': job_key, 'attempt': 1,
        'time_utc': datetime.now(timezone.utc).isoformat(),
        'config_sha256': config_sha256,
        'physical_gpu_index': physical_gpu,
        'physical_gpu_uuid': gpu_uuid,
        'scheduler_commit': scheduler_commit,
        'production_commit': producer_commit,
    }


def claim_next(pending, mutex, *, ready):
    if not ready:
        return None
    with mutex:
        return pending.pop(0) if pending else None


def dispatch_pending(keys, *, ready, run_one):
    pending = list(keys)
    mutex = threading.Lock()
    stop = threading.Event()
    outcomes = {}

    def lane(gpu):
        while not stop.is_set():
            if not ready(gpu, pending, stop):
                return
            key = claim_next(pending, mutex, ready=True)
            if key is None:
                return
            try:
                classification = run_one(key, gpu)
            except BaseException:
                stop.set()
                raise
            with mutex:
                outcomes[key] = classification
            if classification not in ('success', 'training_anomaly'):
                stop.set()
                return

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(lane, gpu) for gpu in (0, 1)]
        for future in futures:
            future.result()
    return outcomes


def execute_one_job(row, config, producer_checkout, manifest, result_root, events,
                    manifest_sha256, scheduler_commit, producer_commit,
                    physical_gpu, gpu_uuid, mutex, *, run_process=subprocess.run):
    key = row['job_key']
    if config['device'] != 'cuda:0' or config['seed'] not in (70, 72):
        raise ValueError('balanced resume only accepts unstarted logical cuda:0 jobs')
    job_root = Path(config['output_dir']).resolve()
    if not job_root.is_relative_to(Path(result_root).resolve()):
        raise ValueError('job output escaped approved result root')
    frozen_queue.reserve_attempt(job_root, 1, key)
    log = job_root / 'attempt-01.log'
    frozen_queue._append_event(events, start_record(
        key, row['config_sha256'], physical_gpu, gpu_uuid,
        scheduler_commit, producer_commit), mutex)
    command = [sys.executable, '-u', str(Path(producer_checkout) /
                                         'run_party_drift_development.py'),
               '--job-key', key, '--manifest', str(manifest),
               '--manifest-sha256', manifest_sha256, '--attempt', '1']
    env = worker_environment(os.environ.copy(), physical_gpu, config['seed'])
    with log.open('x', encoding='utf-8') as stream:
        process = run_process(command, cwd=producer_checkout, env=env,
                              stdout=stream, stderr=subprocess.STDOUT)
    gate = job_root / 'attempt-01' / 'JOB_SUCCESS.json'
    if process.returncode == 0 and gate.is_file():
        try:
            verify_sealed_job(gate.parent, key, manifest_sha256,
                              replay=row['method'] in REPLAY_ARMS)
            classification = 'success'
        except (ValueError, FileNotFoundError, KeyError):
            classification = 'unknown'
    else:
        tail = log.read_text(encoding='utf-8', errors='replace')[-100000:]
        classification = frozen_queue.classify_failure(
            process.returncode or 1, tail)
    frozen_queue._append_event(events, {
        'kind': 'finish', 'job_key': key, 'attempt': 1,
        'time_utc': datetime.now(timezone.utc).isoformat(),
        'returncode': process.returncode,
        'classification': classification,
        'log_sha256': frozen_queue._sha256(log),
        'job_success_sha256': (frozen_queue._sha256(gate)
                               if classification == 'success' else None),
    }, mutex)
    return classification


def completion_record(state, expected_count, manifest_sha256, attempts_sha256):
    finished = state['finished']
    if (state['pending'] or state['reconstructable'] or state.get('active') or
            len(finished) != expected_count or any(
                value not in ('success', 'training_anomaly')
                for value in finished.values())):
        raise ValueError('balanced queue is not complete')
    return {
        'status': 'complete', 'accounted_for': expected_count,
        'successes': sum(value == 'success' for value in finished.values()),
        'training_anomalies': sum(value == 'training_anomaly'
                                  for value in finished.values()),
        'manifest_sha256': manifest_sha256,
        'attempts_sha256': attempts_sha256,
        'ended_utc': datetime.now(timezone.utc).isoformat(),
    }


def execute_balanced_in_root(rows, configs, manifest, result_root,
                             producer_checkout, manifest_sha256,
                             scheduler_commit, producer_commit, gpu_uuids,
                             *, anchor_sha256=None, ready=None,
                             run_process=subprocess.run):
    root = Path(result_root).resolve()
    events_path = root / 'ATTEMPTS.jsonl'
    if (root / 'QUEUE_COMPLETE.json').exists():
        raise ValueError('completed development queue cannot be resumed')
    prior_lock = frozen_queue.active_queue_lock(root)
    if (prior_lock.get('manifest_sha256') != manifest_sha256 or
            prior_lock.get('production_commit') != producer_commit):
        raise ValueError('original queue lock differs from frozen protocol')
    source_ledger_sha = frozen_queue._sha256(events_path)
    source_events = [json.loads(line) for line in events_path.read_text(
        encoding='utf-8').splitlines()]
    source_state = validate_history(rows, source_events, root,
                                    manifest_sha256)
    frozen_queue.recover_interrupted_attempts(
        root, manifest_sha256, 'verified scheduler-only handoff',
        {row['job_key']: row['config_sha256'] for row in rows})
    recovered_ledger_sha = frozen_queue._sha256(events_path)
    recovered_events = [json.loads(line) for line in events_path.read_text(
        encoding='utf-8').splitlines()]
    state = validate_history(rows, recovered_events, root, manifest_sha256)
    if any(configs[key]['device'] != 'cuda:0' for key in state['pending']):
        raise ValueError('pending job uses an unsupported logical GPU')
    frozen_queue._write_once(root / 'BALANCED_HANDOFF.json', {
        'status': 'scheduler_only_handoff',
        'manifest_sha256': manifest_sha256,
        'production_commit': producer_commit,
        'scheduler_commit': scheduler_commit,
        'original_queue_pid': prior_lock['pid'],
        'source_ledger_sha256': source_ledger_sha,
        'recovered_ledger_sha256': recovered_ledger_sha,
        'anchor_sha256': anchor_sha256,
        'gpu_uuids': list(gpu_uuids),
        'pending_keys': state['pending'],
        'reconstructed_keys': source_state['reconstructable'],
        'started_utc': datetime.now(timezone.utc).isoformat(),
    })
    frozen_queue._write_once(frozen_queue._next_record(
        root, 'QUEUE_LOCK_RESUME'), {
            'pid': os.getpid(), 'status': 'running',
            'manifest_sha256': manifest_sha256,
            'production_commit': producer_commit,
            'scheduler_commit': scheduler_commit,
            'started_utc': datetime.now(timezone.utc).isoformat(),
        })
    by_key = {row['job_key']: row for row in rows}
    ready = ready or (lambda gpu, pending, stop: wait_ready(gpu, pending, stop))
    mutex = threading.Lock()

    def run_one(key, gpu):
        return execute_one_job(
            by_key[key], configs[key], producer_checkout, manifest,
            root, events_path, manifest_sha256, scheduler_commit,
            producer_commit, gpu, gpu_uuids[gpu], mutex,
            run_process=run_process)

    try:
        dispatch_pending(state['pending'], ready=ready, run_one=run_one)
        final_events = [json.loads(line) for line in events_path.read_text(
            encoding='utf-8').splitlines()]
        final_state = validate_history(rows, final_events, root, manifest_sha256)
        gate = completion_record(final_state, len(rows), manifest_sha256,
                                 frozen_queue._sha256(events_path))
        frozen_queue._write_once(root / 'QUEUE_COMPLETE.json', gate)
        return gate
    except BaseException as error:
        path = (root / 'QUEUE_FAILURE.json' if not
                (root / 'QUEUE_FAILURE.json').exists() else
                frozen_queue._next_record(root, 'QUEUE_FAILURE'))
        frozen_queue._write_once(path, {
            'status': 'balanced_queue_failed', 'type': type(error).__name__,
            'message': str(error), 'manifest_sha256': manifest_sha256,
            'attempts_sha256': frozen_queue._sha256(events_path),
            'ended_utc': datetime.now(timezone.utc).isoformat(),
        })
        raise


def validate_history(manifest_rows, events, result_root, manifest_sha256,
                     *, allow_active=False):
    """Classify sealed, reconstructable, and unstarted jobs without scores."""
    root = Path(result_root)
    rows = {row['job_key']: row for row in manifest_rows}
    if len(rows) != len(manifest_rows):
        raise ValueError('manifest contains duplicate job keys')
    latest = {}
    for event in events:
        key = event.get('job_key')
        if key not in rows:
            raise ValueError('ledger contains an unlisted job')
        kind = event.get('kind')
        attempt = event.get('attempt')
        if type(attempt) is not int or attempt < 1:
            raise ValueError('ledger has invalid attempt')
        if kind == 'start':
            if (event.get('config_sha256') != rows[key]['config_sha256'] or
                    (key in latest and latest[key]['kind'] != 'finish') or
                    (key in latest and latest[key]['classification'] !=
                     'infrastructure')):
                raise ValueError('ledger start differs from approved retry')
        elif kind == 'finish':
            if (key not in latest or latest[key]['kind'] != 'start' or
                    latest[key]['attempt'] != attempt or
                    event.get('classification') not in
                    ('success', 'training_anomaly', 'infrastructure')):
                raise ValueError('ledger finish does not match start')
        else:
            raise ValueError('ledger contains an unknown event')
        latest[key] = event
    pending, reconstructable, active, anomalies, finished = [], [], [], [], {}
    for row in manifest_rows:
        key = row['job_key']
        event = latest.get(key)
        if event is None:
            pending.append(key)
            continue
        if event['kind'] == 'finish' and event['classification'] == 'training_anomaly':
            anomalies.append(key)
            finished[key] = 'training_anomaly'
            continue
        if event['kind'] == 'finish' and event['classification'] == 'infrastructure':
            raise ValueError('unresolved infrastructure failure requires its own audit')
        run = root / key / f"attempt-{event['attempt']:02d}"
        if (allow_active and event['kind'] == 'start' and
                not (run / 'JOB_SUCCESS.json').is_file()):
            active.append(key)
            continue
        gate = verify_sealed_job(
            run, key, manifest_sha256, replay=row['method'] in REPLAY_ARMS)
        digest = hashlib.sha256((run / 'JOB_SUCCESS.json').read_bytes()).hexdigest()
        if event['kind'] == 'start':
            reconstructable.append(key)
        elif event.get('job_success_sha256') != digest:
            raise ValueError('finished gate differs from append-only ledger')
        finished[key] = 'success'
    return {'pending': pending, 'reconstructable': reconstructable,
            'active': active,
            'anomalies': anomalies, 'finished': finished}


def _git_commit(checkout, *, require_clean):
    checkout = Path(checkout).resolve()
    commit = subprocess.check_output(
        ['git', '-C', str(checkout), 'rev-parse', 'HEAD'],
        text=True).strip()
    if require_clean and subprocess.check_output(
            ['git', '-C', str(checkout), 'status', '--porcelain'],
            text=True).strip():
        raise ValueError(f'checkout is not clean: {checkout}')
    return commit


def load_run_inputs(manifest, manifest_sha256, result_root,
                    producer_checkout, *, allow_active):
    manifest = Path(manifest).resolve()
    root = Path(result_root).resolve()
    producer_commit = _git_commit(producer_checkout, require_clean=True)
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != manifest_sha256:
        raise ValueError('balanced queue manifest differs from external anchor')
    rows = [json.loads(line) for line in manifest.read_text(
        encoding='utf-8').splitlines()]
    if len(rows) != 183 or {row.get('job_key') for row in rows} != expected_job_keys():
        raise ValueError('balanced queue does not have exactly 183 approved jobs')
    configs = {}
    for row in rows:
        load_approved_job(row['job_key'], manifest, manifest_sha256,
                          production_commit=producer_commit)
        path = (manifest.parent / row['config_relpath']).resolve()
        configs[row['job_key']] = json.loads(path.read_text(encoding='utf-8'))
    validate_manifest_rows(rows, configs, root, producer=producer_commit,
                           expected=expected_job_keys())
    events = [json.loads(line) for line in (root / 'ATTEMPTS.jsonl').read_text(
        encoding='utf-8').splitlines()]
    state = validate_history(rows, events, root, manifest_sha256,
                             allow_active=allow_active)
    snapshots = [gpu_snapshot(index) for index in (0, 1)]
    if any(value['name'] != 'NVIDIA GeForce RTX 3080'
           for value in snapshots):
        raise ValueError('balanced queue GPUs are not the same RTX 3080 model')
    return rows, configs, state, producer_commit, snapshots


def run_balanced_queue(manifest, manifest_sha256, result_root,
                       producer_checkout, anchor_path, anchor_sha256):
    root = Path(result_root).resolve()
    with frozen_queue.queue_run_lock(root):
        assert_original_queue_gone(root, manifest)
        rows, configs, state, producer_commit, snapshots = load_run_inputs(
            manifest, manifest_sha256, root, producer_checkout,
            allow_active=False)
        if state['active'] or not state['pending']:
            raise ValueError('handoff has no safe unstarted jobs')
        scheduler_commit = _git_commit(Path(__file__).resolve().parent,
                                       require_clean=True)
        anchor_path = Path(anchor_path)
        if hashlib.sha256(anchor_path.read_bytes()).hexdigest() != anchor_sha256:
            raise ValueError('scheduler addendum anchor bytes differ')
        validate_anchor(
            json.loads(anchor_path.read_text(encoding='utf-8')),
            producer=producer_commit, scheduler=scheduler_commit,
            manifest=manifest_sha256,
            ledger=frozen_queue._sha256(root / 'ATTEMPTS.jsonl'),
            gpu_uuids=[item['uuid'] for item in snapshots])
        return execute_balanced_in_root(
            rows, configs, manifest, root, producer_checkout,
            manifest_sha256, scheduler_commit, producer_commit,
            [item['uuid'] for item in snapshots],
            anchor_sha256=anchor_sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--producer-checkout', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--anchor', type=Path)
    parser.add_argument('--anchor-sha256')
    args = parser.parse_args()
    if args.check:
        rows, _configs, state, producer, snapshots = load_run_inputs(
            args.manifest, args.manifest_sha256, args.result_root,
            args.producer_checkout, allow_active=True)
        print(json.dumps({
            'status': 'read_only_check', 'expected': len(rows),
            'completed': len(state['finished']),
            'pending': len(state['pending']),
            'active': state['active'],
            'reconstructable': state['reconstructable'],
            'producer_commit': producer,
            'gpu': snapshots,
            'accuracy_used_for_selection': False,
        }, sort_keys=True))
    else:
        if not args.anchor or not args.anchor_sha256:
            raise ValueError('resume requires the externally anchored addendum')
        result = run_balanced_queue(
            args.manifest, args.manifest_sha256, args.result_root,
            args.producer_checkout, args.anchor, args.anchor_sha256)
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
