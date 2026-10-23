"""Queued execution of the immutable development manifest."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from party_drift_manifest import expected_job_keys, load_approved_job


DATASET_ORDER = ('upmc_food101', 'isolet', 'cifar100')


def classify_failure(exit_code, log_text):
    if exit_code == 0:
        return 'success'
    text = log_text.lower()
    if any(word in text for word in (
            'floatingpointerror', 'non-finite', 'nonfinite',
            'loss diverged', 'nan loss')):
        return 'training_anomaly'
    if (exit_code in (-9, 137) or any(word in text for word in (
            'outofmemoryerror', 'out of memory', 'connection reset',
            'no space left', 'host unreachable', 'keyboardinterrupt'))):
        return 'infrastructure'
    return 'unknown'


def ordered_jobs(rows):
    return sorted(rows, key=lambda row: (
        DATASET_ORDER.index(row['dataset']), row['job_key']))


def gpu_state_is_idle(line):
    try:
        memory, utilization = [int(value.strip()) for value in line.split(',')]
    except (ValueError, TypeError) as error:
        raise ValueError('invalid GPU state record') from error
    return memory < 256 and utilization < 10


def reserve_attempt(job_root, attempt, job_key):
    if type(attempt) is not int or attempt < 1:
        raise ValueError('attempt must be a positive integer')
    job_root = Path(job_root)
    job_root.mkdir(parents=True, exist_ok=True)
    path = job_root / f'attempt-{attempt:02d}.plan.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump({'attempt': attempt, 'job_key': job_key}, stream,
                  sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return path


def retry_attempt(events, job_key, reason):
    if not reason or not reason.strip():
        raise ValueError('infrastructure retry needs a recorded reason')
    finished = [event for event in events if event.get('kind') == 'finish'
                and event.get('job_key') == job_key]
    if not finished or finished[-1].get('classification') != 'infrastructure':
        raise ValueError('only a documented infrastructure failure may retry')
    return int(finished[-1]['attempt']) + 1


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _gpu_idle(device):
    index = int(device.split(':')[1])
    result = subprocess.run([
        'nvidia-smi', '-i', str(index),
        '--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits',
    ], text=True, capture_output=True, check=True)
    return gpu_state_is_idle(result.stdout.strip())


def _wait_for_gpu(device, stop):
    consecutive = 0
    while not stop.is_set():
        consecutive = consecutive + 1 if _gpu_idle(device) else 0
        if consecutive >= 2:
            return True
        stop.wait(30)
    return False


def _append_event(path, event, mutex):
    with mutex:
        with Path(path).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def active_queue_lock(root):
    root = Path(root)
    locks = sorted(root.glob('QUEUE_LOCK_RESUME_*.json'))
    path = locks[-1] if locks else root / 'QUEUE_LOCK.json'
    return json.loads(path.read_text(encoding='utf-8'))


def _next_record(root, stem):
    number = 1
    while (Path(root) / f'{stem}_{number:02d}.json').exists():
        number += 1
    return Path(root) / f'{stem}_{number:02d}.json'


def record_resume(job_root, attempt, job_key, manifest_sha256, reason):
    path = Path(job_root) / f'attempt-{attempt:02d}.resume.json'
    if path.is_file():
        prior = json.loads(path.read_text(encoding='utf-8'))
        if (prior.get('job_key') != job_key or prior.get('attempt') != attempt or
                prior.get('manifest_sha256') != manifest_sha256 or
                not prior.get('reason')):
            raise ValueError('existing retry authorization differs')
    else:
        _write_once(path, {
            'job_key': job_key, 'attempt': attempt, 'reason': reason,
            'time_utc': _utc_now(), 'manifest_sha256': manifest_sha256,
        })
    return path


def running_worker_keys(manifest, proc_root=Path('/proc')):
    found = set()
    proc_root = Path(proc_root)
    if not proc_root.is_dir():
        return found
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        try:
            parts = (process / 'cmdline').read_bytes().decode(
                'utf-8', errors='replace').split('\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (not any(part.endswith('run_party_drift_development.py')
                    for part in parts) or '--manifest' not in parts or
                '--job-key' not in parts):
            continue
        try:
            if Path(parts[parts.index('--manifest') + 1]).resolve() == Path(
                    manifest).resolve():
                found.add(parts[parts.index('--job-key') + 1])
        except (IndexError, ValueError):
            continue
    return found


def recover_interrupted_attempts(root, manifest_sha256, reason,
                                 config_sha256_by_key):
    from party_drift_audit_gate import verify_sealed_job

    if not reason or not reason.strip():
        raise ValueError('interrupted queue recovery needs an explicit reason')
    root = Path(root)
    events_path = root / 'ATTEMPTS.jsonl'
    if not events_path.exists():
        with events_path.open('x', encoding='utf-8'):
            pass
    history = [json.loads(line) for line in events_path.read_text(
        encoding='utf-8').splitlines() if line]
    latest = {}
    for event in history:
        key = event.get('job_key')
        if key not in config_sha256_by_key:
            raise ValueError('interrupted queue contains an unlisted job')
        if event.get('kind') == 'start':
            if event.get('config_sha256') != config_sha256_by_key[key]:
                raise ValueError('interrupted job config differs from manifest')
            latest[key] = event
        elif event.get('kind') == 'finish':
            if (key not in latest or latest[key].get('kind') != 'start' or
                    latest[key]['attempt'] != event.get('attempt')):
                raise ValueError('interrupted queue events do not pair')
            latest[key] = event
        else:
            raise ValueError('unknown queue event in interrupted history')
    for plan_path in sorted(root.rglob('attempt-*.plan.json')):
        job_root = plan_path.parent
        key = job_root.relative_to(root).as_posix()
        if key not in config_sha256_by_key:
            raise ValueError('interrupted plan has an unlisted job')
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        attempt = plan.get('attempt')
        if (plan != {'attempt': attempt, 'job_key': key} or
                type(attempt) is not int or attempt < 1 or
                plan_path.name != f'attempt-{attempt:02d}.plan.json'):
            raise ValueError('interrupted job plan differs')
        if not any(event.get('job_key') == key and
                   event.get('attempt') == attempt and
                   event.get('kind') == 'start' for event in history):
            if key in latest and (latest[key].get('kind') != 'finish' or
                                  latest[key].get('attempt') != attempt - 1 or
                                  latest[key].get('classification') != 'infrastructure'):
                raise ValueError('orphan plan is not an authorized retry')
            start = {'kind': 'start', 'job_key': key, 'attempt': attempt,
                     'time_utc': _utc_now(),
                     'config_sha256': config_sha256_by_key[key],
                     'reconstructed_before_worker': True}
            _append_event(events_path, start, threading.Lock())
            history.append(start)
            latest[key] = start
    recovered = []
    for key, event in sorted(latest.items()):
        if event['kind'] != 'start':
            continue
        attempt = event['attempt']
        job_root = root / key
        plan = json.loads((job_root / f'attempt-{attempt:02d}.plan.json').read_text(
            encoding='utf-8'))
        if plan != {'attempt': attempt, 'job_key': key}:
            raise ValueError('interrupted job plan differs')
        log = job_root / f'attempt-{attempt:02d}.log'
        tail = log.read_text(encoding='utf-8', errors='replace')[-100000:] if log.is_file() else ''
        gate = job_root / f'attempt-{attempt:02d}' / 'JOB_SUCCESS.json'
        readout = gate.parent / 'HOLDOUT_READOUT.json'
        success = False
        if gate.is_file() and readout.is_file():
            try:
                verify_sealed_job(gate.parent, key, manifest_sha256,
                                  replay=key.split('/')[1] in (
                                      'all_party', 'uniform', 'static', 'drift',
                                      'cmcdr', 'cmcdr_budget'))
                success = True
            except ValueError:
                pass
        classification = ('success' if success else
                          'training_anomaly' if classify_failure(1, tail) ==
                          'training_anomaly' else 'infrastructure')
        finish = {
            'kind': 'finish', 'job_key': key, 'attempt': attempt,
            'time_utc': _utc_now(), 'returncode': None,
            'classification': classification,
            'recovery_reason': reason,
            'log_sha256': _sha256(log) if log.is_file() else None,
            'job_success_sha256': _sha256(gate) if success else None,
        }
        _append_event(events_path, finish, threading.Lock())
        latest[key] = finish
        recovered.append(finish)
    outcomes = {key: event['classification'] for key, event in latest.items()
                if event['kind'] == 'finish'}
    retries = {key: event['attempt'] + 1 for key, event in latest.items()
               if event['kind'] == 'finish' and
               event['classification'] == 'infrastructure'}
    _write_once(_next_record(root, 'QUEUE_RECOVERY'), {
        'reason': reason, 'manifest_sha256': manifest_sha256,
        'reconstructed': recovered, 'retry_attempts': retries,
        'attempts_sha256': _sha256(events_path), 'time_utc': _utc_now(),
    })
    return {'outcomes': outcomes, 'retry_attempts': retries}


@contextmanager
def queue_run_lock(result_root):
    """Hold one OS-released lock across the entire queue or recovery run."""
    if os.name != 'posix' or not Path('/proc').is_dir():
        raise RuntimeError('development queue requires the verified Linux host')
    import fcntl

    root = Path(result_root).resolve()
    if not root.parent.is_dir():
        raise ValueError('queue result parent does not exist')
    path = root.parent / f'.{root.name}.queue.lock'
    with path.open('a+', encoding='utf-8') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('development queue or recovery is already running') from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_queue(manifest, manifest_sha256, result_root, *,
              retry_infra_key=None, retry_reason=None,
              recover_interruption_reason=None):
    with queue_run_lock(result_root):
        return _run_queue_locked(
            manifest, manifest_sha256, result_root,
            retry_infra_key=retry_infra_key, retry_reason=retry_reason,
            recover_interruption_reason=recover_interruption_reason)


def _run_queue_locked(manifest, manifest_sha256, result_root, *,
                      retry_infra_key=None, retry_reason=None,
                      recover_interruption_reason=None):
    if retry_infra_key is not None and recover_interruption_reason is not None:
        raise ValueError('choose one explicit queue recovery mode')
    checkout = Path(__file__).resolve().parent
    commit = subprocess.check_output(
        ['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    manifest = Path(manifest).resolve()
    if _sha256(manifest) != manifest_sha256:
        raise ValueError('queue manifest differs from external anchor')
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    if len(rows) != 183 or {row.get('job_key') for row in rows} != expected_job_keys():
        raise ValueError('queue does not have exactly 183 approved jobs')
    root = Path(result_root).resolve()
    jobs = []
    for row in ordered_jobs(rows):
        load_approved_job(row['job_key'], manifest, manifest_sha256,
                          production_commit=commit)
        config_path = (manifest.parent / row['config_relpath']).resolve()
        config = json.loads(config_path.read_text(encoding='utf-8'))
        job_root = Path(config['output_dir']).resolve()
        if (not job_root.is_relative_to(root) or
                config['development_job_key'] != row['job_key'] or
                config['device'] not in ('cuda:0', 'cuda:1')):
            raise ValueError('approved job output or GPU differs')
        jobs.append((row, config, job_root))
    events = root / 'ATTEMPTS.jsonl'
    attempts_by_key = {}
    outcomes = {}
    if not root.exists():
        if retry_infra_key is not None or recover_interruption_reason is not None:
            raise ValueError('cannot retry an absent queue')
        root.mkdir(parents=True, exist_ok=False)
        _write_once(root / 'QUEUE_LOCK.json', {
            'pid': os.getpid(), 'status': 'running',
            'manifest_sha256': manifest_sha256,
            'production_commit': commit, 'started_utc': _utc_now(),
        })
        with events.open('x', encoding='utf-8'):
            pass
    else:
        if (recover_interruption_reason is None and
                (retry_infra_key is None or
                 not list(root.glob('QUEUE_FAILURE*.json')))):
            raise FileExistsError('existing queue needs an explicit infrastructure retry')
        if (root / 'QUEUE_COMPLETE.json').exists():
            raise ValueError('completed queue cannot be restarted')
        lock = active_queue_lock(root)
        if (lock.get('manifest_sha256') != manifest_sha256 or
                lock.get('production_commit') != commit):
            raise ValueError('retry code or manifest differs from original queue')
        pid = int(lock['pid'])
        cmdline = Path(f'/proc/{pid}/cmdline')
        if cmdline.is_file() and b'party_drift_queue.py' in cmdline.read_bytes():
            raise RuntimeError('original queue process is still running')
        active_workers = running_worker_keys(manifest)
        if active_workers:
            raise RuntimeError(f'original workers still active: {sorted(active_workers)}')
        if recover_interruption_reason is not None:
            recovered = recover_interrupted_attempts(
                root, manifest_sha256, recover_interruption_reason,
                {row['job_key']: row['config_sha256'] for row in rows})
            outcomes.update(recovered['outcomes'])
            attempts_by_key.update(recovered['retry_attempts'])
        else:
            prior = [json.loads(line) for line in events.read_text(
                encoding='utf-8').splitlines()]
            retry_number = retry_attempt(prior, retry_infra_key, retry_reason)
            attempts_by_key[retry_infra_key] = retry_number
            for event in prior:
                if event.get('kind') == 'finish':
                    outcomes[event['job_key']] = event['classification']
        if any(status == 'unknown' for status in outcomes.values()) or any(
                status == 'infrastructure' and key not in attempts_by_key
                for key, status in outcomes.items()):
            raise ValueError('another unresolved failure blocks queue resume')
        for key, attempt in attempts_by_key.items():
            outcomes.pop(key, None)
            reason = (recover_interruption_reason if recover_interruption_reason
                      is not None else retry_reason)
            record_resume(root / key, attempt, key, manifest_sha256, reason)
        _write_once(_next_record(root, 'QUEUE_LOCK_RESUME'), {
            'pid': os.getpid(), 'status': 'running',
            'manifest_sha256': manifest_sha256,
            'production_commit': commit, 'started_utc': _utc_now(),
            'recovery_reason': recover_interruption_reason or retry_reason,
        })
    jobs = [job for job in jobs if job[0]['job_key'] not in outcomes]
    mutex = threading.Lock()
    stop = threading.Event()

    def lane(items):
        from party_drift_audit_gate import verify_sealed_job

        for row, config, job_root in items:
            if stop.is_set() or not _wait_for_gpu(config['device'], stop):
                return
            key = row['job_key']
            attempt = attempts_by_key.get(key, 1)
            reserve_attempt(job_root, attempt, key)
            log = job_root / f'attempt-{attempt:02d}.log'
            _append_event(events, {
                'kind': 'start', 'job_key': key, 'attempt': attempt,
                'time_utc': _utc_now(), 'config_sha256': row['config_sha256'],
            }, mutex)
            env = os.environ.copy()
            env.update({
                'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                'PYTHONHASHSEED': str(config['seed']),
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
            })
            with log.open('x', encoding='utf-8') as stream:
                process = subprocess.run([
                    sys.executable, '-u',
                    str(checkout / 'run_party_drift_development.py'),
                    '--job-key', key, '--manifest', str(manifest),
                    '--manifest-sha256', manifest_sha256,
                    '--attempt', str(attempt),
                ], cwd=checkout, env=env, stdout=stream,
                   stderr=subprocess.STDOUT)
            gate = job_root / f'attempt-{attempt:02d}' / 'JOB_SUCCESS.json'
            if process.returncode == 0 and gate.is_file():
                try:
                    verify_sealed_job(gate.parent, key, manifest_sha256,
                                      replay=bool(config['party_drift_enabled']))
                    classification = 'success'
                except (ValueError, FileNotFoundError, KeyError):
                    classification = 'unknown'
            else:
                tail = log.read_text(encoding='utf-8', errors='replace')[-100000:]
                classification = classify_failure(process.returncode or 1, tail)
            with mutex:
                outcomes[key] = classification
            _append_event(events, {
                'kind': 'finish', 'job_key': key, 'attempt': attempt,
                'time_utc': _utc_now(), 'returncode': process.returncode,
                'classification': classification, 'log_sha256': _sha256(log),
                'job_success_sha256': (_sha256(gate) if classification ==
                                       'success' else None),
            }, mutex)
            if classification in ('infrastructure', 'unknown'):
                stop.set()
                return

    try:
        for dataset in DATASET_ORDER:
            group = [job for job in jobs if job[0]['dataset'] == dataset]
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(lane, [job for job in group
                                              if job[1]['device'] == f'cuda:{gpu}'])
                           for gpu in (0, 1)]
                failures = []
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as error:
                        stop.set()
                        failures.append(error)
                if failures:
                    raise failures[0]
            if stop.is_set():
                break
    except Exception as error:
        filename = (root / 'QUEUE_FAILURE.json' if not
                    (root / 'QUEUE_FAILURE.json').exists() else
                    _next_record(root, 'QUEUE_FAILURE'))
        _write_once(filename, {
            'status': 'queue_exception', 'type': type(error).__name__,
            'message': str(error), 'accounted_for': len(outcomes),
            'manifest_sha256': manifest_sha256,
            'attempts_sha256': _sha256(events), 'ended_utc': _utc_now(),
        })
        raise
    status = ('complete' if len(outcomes) == 183 and all(
        value in ('success', 'training_anomaly') for value in outcomes.values())
        else 'failed_or_interrupted')
    record = {
        'status': status, 'accounted_for': len(outcomes),
        'successes': sum(value == 'success' for value in outcomes.values()),
        'training_anomalies': sum(value == 'training_anomaly'
                                  for value in outcomes.values()),
        'manifest_sha256': manifest_sha256,
        'attempts_sha256': _sha256(events), 'ended_utc': _utc_now(),
    }
    if status == 'complete':
        gate_name = 'QUEUE_COMPLETE.json'
    else:
        gate_name = ('QUEUE_FAILURE.json' if not
                     (root / 'QUEUE_FAILURE.json').exists() else
                     _next_record(root, 'QUEUE_FAILURE').name)
    _write_once(root / gate_name, record)
    if status != 'complete':
        raise RuntimeError('development queue stopped with unresolved work')
    print(json.dumps(record, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--retry-infra-key')
    parser.add_argument('--retry-reason')
    parser.add_argument('--recover-interruption-reason')
    args = parser.parse_args()
    run_queue(args.manifest, args.manifest_sha256, args.result_root,
              retry_infra_key=args.retry_infra_key,
              retry_reason=args.retry_reason,
              recover_interruption_reason=args.recover_interruption_reason)


if __name__ == '__main__':
    main()
