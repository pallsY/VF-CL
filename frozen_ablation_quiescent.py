"""Read-only ledger observation and pidfd-bound retirement at an idle point.

This module never creates a writer, worker, epoch, plan, or result. A failed
pre-retirement fence can therefore resume only the same bound old controller.
"""

import ctypes
import hashlib
import json
import os
import select
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from frozen_ablation_manifest import expected_keys

RESTART_ANCHOR_SHA256 = '61d6e229783f12a3d0d02af81b158efea459830d6fe83667df5aab1037d2f00b'


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON field')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON value')
    try:
        return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('invalid metadata JSON') from exc


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'missing or symlinked metadata file: {path}')
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f'cannot read metadata file: {path}') from exc


def _sha(path):
    return hashlib.sha256(_read(path)).hexdigest()


def _directory(path):
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f'unknown or symlinked job directory: {path}')


def _verify_restart_anchor(root):
    try:
        digest = _sha(root / 'manifests/RESTART_ANCHOR.json')
    except ValueError as exc:
        raise ValueError('required restart anchor is missing or unsafe') from exc
    if digest != RESTART_ANCHOR_SHA256:
        raise ValueError('restart anchor SHA differs')


def assign_pending(keys):
    """Assign only the seven authorized candidates, with no duplicate claim."""
    keys = list(keys)
    isolet = {f'isolet/{arm}/{seed}' for arm in ('no_partykd', 'fim_off')
              for seed in (42, 43, 44)}
    cifar = 'cifar100/no_partykd/43'
    if len(keys) != len(set(keys)) or not set(keys) <= isolet | {cifar}:
        raise ValueError('pending keys exceed the original candidate allowlist')
    return {key: (int(key == cifar) if cifar in keys else i % 2)
            for i, key in enumerate(sorted(keys))}


def ledger_state(root, rows):
    """Classify the original eighteen jobs; only hash completed gate/log bytes."""
    root, rows = Path(root), list(rows)
    _directory(root)
    _verify_restart_anchor(root)
    by_key = {}
    for row in rows:
        key = row.get('job_key')
        if key not in expected_keys() or key in by_key:
            raise ValueError('manifest must contain eighteen unique original jobs')
        dataset, arm, seed = key.split('/')
        digest = row.get('config_sha256')
        if (row.get('dataset') != dataset or row.get('arm') != arm or
                type(row.get('seed')) is not int or row['seed'] != int(seed) or
                type(row.get('physical_gpu')) is not int or row['physical_gpu'] not in (0, 1) or
                not isinstance(digest, str) or len(digest) != 64 or
                any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('manifest job identity differs')
        by_key[key] = row
    if set(by_key) != expected_keys():
        raise ValueError('manifest must contain eighteen unique original jobs')
    raw = _read(root / 'ATTEMPTS.jsonl')
    if raw and not raw.endswith(b'\n'):
        raise ValueError('partial ledger line')
    events = {}
    for line in raw.splitlines():
        if not line.strip():
            raise ValueError('blank ledger line')
        event = _json(line)
        if (not isinstance(event, dict) or event.get('job_key') not in by_key or
                type(event.get('attempt')) is not int or event['attempt'] != 1):
            raise ValueError('unknown ledger event or attempt')
        key = event['job_key']
        own = events.setdefault(key, [])
        if event.get('kind') != ('start' if not own else 'finish') or len(own) >= 2:
            raise ValueError('duplicate or out-of-order ledger claim')
        if not own and event.get('config_sha256') != by_key[key]['config_sha256']:
            raise ValueError('start config SHA differs')
        row = by_key[key]
        if not own and 'code_commit' in row and event.get('code_commit') != row['code_commit']:
            raise ValueError('start code commit differs')
        if not own and 'gpu_uuid' in row and 'execution_epoch_sha256' not in event:
            if (type(event.get('gpu_index')) is not int or event['gpu_index'] != row['physical_gpu'] or
                    event.get('gpu_uuid') != row['gpu_uuid']):
                raise ValueError('legacy start physical GPU differs')
        own.append(event)

    # Inspect only the job hierarchy: never open accuracy-bearing run artifacts.
    for entry in root.iterdir():
        if entry.name in ('manifests', 'gpu1-rebalance-v1'):
            _directory(entry)
            if entry.name == 'manifests':
                for item in entry.iterdir():
                    if item.is_symlink() or item.name not in ('manifest.jsonl', 'manifest.sha256', 'RESTART_ANCHOR.json',
                                                             'preflight', 'configs'):
                        raise ValueError('unknown manifest artifact')
                    if item.name in ('preflight', 'configs'):
                        _directory(item)
                    elif not item.is_file():
                        raise ValueError('invalid manifest metadata')
            if entry.name == 'gpu1-rebalance-v1':
                for item in entry.iterdir():
                    if item.is_symlink() or item.name not in ('ARMED.json', 'EXECUTION_EPOCH.json', 'workers'):
                        raise ValueError('unknown execution extension artifact')
                    if item.name == 'workers':
                        _directory(item)
                        for worker in item.iterdir():
                            _directory(worker)
                            if worker.name not in {key.replace('/', '__') for key in by_key}:
                                raise ValueError('unknown extension worker')
                            for protocol in worker.iterdir():
                                if (protocol.is_symlink() or not protocol.is_file() or
                                        protocol.name not in ('READY.json', 'GO.json')):
                                    raise ValueError('unknown extension worker protocol')
                    elif not item.is_file():
                        raise ValueError('invalid execution extension metadata')
        elif entry.is_dir() or entry.is_symlink():
            if entry.name not in ('no_partykd', 'fim_off'):
                raise ValueError('unknown root directory')
            _directory(entry)
            for dataset in entry.iterdir():
                _directory(dataset)
                if dataset.name not in ('cifar100', 'isolet', 'upmc_food101'):
                    raise ValueError('unknown dataset directory')
                for job in dataset.iterdir():
                    _directory(job)
                    key = f'{dataset.name}/{entry.name}/{job.name}'
                    if key not in by_key or key not in events:
                        raise ValueError('pending or unknown job directory already exists')
                    if any(path.name not in ('attempt-01.plan.json', 'attempt-01.log', 'attempt-01')
                           or path.is_symlink() for path in job.iterdir()):
                        raise ValueError('unknown attempt artifact')
        elif entry.name not in ('ATTEMPTS.jsonl', 'QUEUE_LOCK.json', 'QUEUE_COMPLETE.json', 'QUEUE_FAILURE.json'):
            raise ValueError('unknown root artifact')

    state = dict(raw_sha256=hashlib.sha256(raw).hexdigest(), raw_bytes_length=len(raw),
                 success=[], unfinished=[], pending=[], failed=[])
    for key, row in sorted(by_key.items()):
        job = root / row['arm'] / row['dataset'] / str(row['seed'])
        own = events.get(key, [])
        if not own:
            if job.exists() or job.is_symlink():
                raise ValueError('pending job has an existing root or plan')
            state['pending'].append(key)
            continue
        _directory(job)
        plan = _json(_read(job / 'attempt-01.plan.json'))
        if (not isinstance(plan, dict) or set(plan) != {'job_key', 'attempt'} or plan.get('job_key') != key or
                type(plan.get('attempt')) is not int or plan['attempt'] != 1):
            raise ValueError('attempt plan differs')
        if len(own) == 1:
            state['unfinished'].append(key)
            continue
        finish = own[1]
        classification = finish.get('classification')
        if classification == 'success':
            _directory(job / 'attempt-01')
            if (type(finish.get('returncode')) is not int or finish['returncode'] != 0 or
                    finish.get('test_accessed') is not True or
                    finish.get('job_success_sha256') != _sha(job / 'attempt-01/JOB_SUCCESS.json') or
                    finish.get('log_sha256') != _sha(job / 'attempt-01.log')):
                raise ValueError('successful finish raw gate/log SHA differs')
            state['success'].append(key)
        elif classification in ('infrastructure_candidate', 'training_or_audit_failure',
                                'cancelled_by_peer_failure'):
            state['failed'].append(key)
        else:
            raise ValueError('unknown finish classification')
    if _read(root / 'ATTEMPTS.jsonl') != raw:
        raise ValueError('ledger changed during observation')
    _verify_restart_anchor(root)
    return state


def _stat(path):
    raw = path.read_text()
    end = raw.rfind(')')
    fields = raw[end + 2:].split()
    if end < 0 or len(fields) < 20:
        raise ValueError('invalid proc stat')
    return dict(pid=int(raw.split(' ', 1)[0]), state=fields[0], ppid=int(fields[1]),
                pgrp=int(fields[2]), session=int(fields[3]), start_ticks=int(fields[19]))


def identity(pid):
    """Snapshot same-host Linux process identity, guarding a changing /proc read."""
    if sys.platform != 'linux' or type(pid) is not int or pid <= 0:
        raise ValueError('identity requires a positive same-host Linux PID')
    proc = Path('/proc') / str(pid)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    before = _stat(proc / 'stat')
    command = [os.fsdecode(part) for part in (proc / 'cmdline').read_bytes().split(b'\0') if part]
    after = _stat(proc / 'stat')
    if any(before[field] != after[field] for field in ('pid', 'ppid', 'pgrp', 'session', 'start_ticks')):
        raise ValueError('process identity changed during observation')
    return dict(after, boot_id=boot, cmdline=command)


def same_identity(expected, actual):
    """State may change; every bound identity field must remain identical."""
    fields = ('pid', 'ppid', 'pgrp', 'session', 'start_ticks', 'boot_id', 'cmdline')
    return (isinstance(expected, dict) and isinstance(actual, dict) and
            all(field in expected and field in actual and expected[field] == actual[field]
                for field in fields) and
            type(expected['pid']) is int and expected['pid'] > 0 and
            type(expected['start_ticks']) is int and expected['start_ticks'] > 0 and
            bool(expected['boot_id']) and bool(expected['cmdline']))


def _threads(pid):
    task = Path('/proc') / str(pid) / 'task'
    before = sorted(task.iterdir())
    if not before:
        raise ValueError('controller has no observable threads')
    snapshot = []
    for thread in before:
        info = _stat(thread / 'stat')
        children = tuple(int(value) for value in (thread / 'children').read_text().split())
        snapshot.append((int(thread.name), info['start_ticks'], info['state'], children))
    if before != sorted(task.iterdir()):
        raise ValueError('controller threads changed during observation')
    return tuple(snapshot)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _pidfd_open(pid):
    native = getattr(os, 'pidfd_open', None)
    if native is not None:
        return native(pid)
    # Some supported Python builds omit the wrapper although libc/kernel have it.
    try:
        function = ctypes.CDLL(None, use_errno=True).pidfd_open
    except AttributeError as exc:
        raise RuntimeError('Linux libc has no pidfd_open') from exc
    function.argtypes = (ctypes.c_int, ctypes.c_uint)
    function.restype = ctypes.c_int
    fd = function(pid, 0)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def _no_new_writer(root):
    extension = root / 'gpu1-rebalance-v1'
    return not any(path.exists() or path.is_symlink() for path in
                   (extension / 'EXECUTION_EPOCH.json', extension / 'workers'))


def _metadata_snapshot(root, rows):
    paths = [root / name for name in ('QUEUE_LOCK.json', 'QUEUE_COMPLETE.json', 'QUEUE_FAILURE.json')]
    paths += [root / 'manifests' / name for name in ('manifest.jsonl', 'manifest.sha256', 'RESTART_ANCHOR.json')]
    paths += [root / 'gpu1-rebalance-v1/ARMED.json']
    paths += [root / row['arm'] / row['dataset'] / str(row['seed']) / 'attempt-01.plan.json'
              for row in rows]
    return {path.relative_to(root).as_posix(): _sha(path) if path.exists() or path.is_symlink() else None
            for path in paths}


def try_quiescent_fence(root, rows, old_identity):
    """Return a retirement receipt only at a verified settled, child-free idle point.

    Active original queues return None without signals. Invalid evidence raises;
    a race after SIGSTOP resumes the same parent and returns None. No live child
    is ever adopted or signalled. pidfds eliminate check-to-signal PID reuse.
    """
    root, rows = Path(root), list(rows)
    initial = ledger_state(root, rows)
    if initial['failed'] or (root / 'QUEUE_FAILURE.json').exists():
        raise ValueError('failed original queue cannot be continued')
    if initial['unfinished'] or len(initial['pending']) < 2:
        return None
    if len(set(assign_pending(initial['pending']).values())) < 2:
        return None
    if (root / 'QUEUE_COMPLETE.json').exists() or not _no_new_writer(root):
        raise ValueError('queue is already complete or has an execution epoch')
    metadata = _metadata_snapshot(root, rows)
    if sys.platform != 'linux' or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('quiescent fence requires Linux pidfd support')
    pid = old_identity.get('pid')
    if type(pid) is not int or pid <= 1 or pid == os.getpid():
        raise ValueError('unsafe old controller PID')
    fd = _pidfd_open(pid)
    stopped, retired = False, False
    try:
        if not same_identity(old_identity, identity(pid)):
            raise ValueError('old controller identity differs')
        threads = _threads(pid)
        if any(thread[3] for thread in threads):
            return None
        if any(thread[2] not in ('R', 'S', 'D', 'I') for thread in threads):
            raise ValueError('old controller has an unknown or externally stopped state')
        if not same_identity(old_identity, identity(pid)):
            raise ValueError('old controller identity changed before stop')
        signal.pidfd_send_signal(fd, signal.SIGSTOP)
        stopped = True
        stopped_utc = _now()
        try:
            deadline = time.monotonic() + 3
            while True:
                threads = _threads(pid)
                if any(thread[3] for thread in threads):
                    return None
                if all(thread[2] == 'T' for thread in threads):
                    break
                if time.monotonic() >= deadline:
                    return None
                time.sleep(0.01)
            previous, snapshots = None, []
            for _ in range(2):
                current = identity(pid)
                threads = _threads(pid)
                state = ledger_state(root, rows)
                if (not same_identity(old_identity, current) or current['state'] != 'T' or
                        any(thread[2] != 'T' or thread[3] for thread in threads) or
                        state != initial or _metadata_snapshot(root, rows) != metadata or
                        not _no_new_writer(root) or (previous is not None and threads != previous)):
                    return None
                previous = threads
                snapshots.append(dict(observed_utc=_now(), identity=current, threads=threads,
                                      ledger_sha256=state['raw_sha256'],
                                      ledger_bytes_length=state['raw_bytes_length']))
                time.sleep(0.01)
            # Last identity/tree check immediately precedes irreversible retirement.
            if (not same_identity(old_identity, identity(pid)) or _threads(pid) != previous or
                    ledger_state(root, rows) != initial or
                    _metadata_snapshot(root, rows) != metadata or not _no_new_writer(root)):
                return None
        except (OSError, ValueError):
            return None
        signal.pidfd_send_signal(fd, signal.SIGKILL)
        retired = True
        if not select.select([fd], [], [], 5)[0]:
            raise RuntimeError('old controller retirement is not proven')
        try:
            dead = identity(pid)
        except FileNotFoundError:
            dead = {**old_identity, 'state': 'gone'}
        if (dead['state'] not in ('Z', 'X', 'gone') or
                any(dead[field] != old_identity[field] for field in ('pid', 'start_ticks', 'boot_id'))):
            raise RuntimeError('old controller retirement identity differs')
        if (ledger_state(root, rows) != initial or _metadata_snapshot(root, rows) != metadata or
                not _no_new_writer(root)):
            raise ValueError('ledger changed after old controller retirement')
        return dict(ledger_prefix_sha256=initial['raw_sha256'],
                    ledger_prefix_bytes_length=initial['raw_bytes_length'],
                    raw_sha256=initial['raw_sha256'], raw_bytes_length=initial['raw_bytes_length'],
                    success_keys=initial['success'], pending_keys=initial['pending'],
                    old_identity=dict(old_identity), retired_identity=dead,
                    stopped_utc=stopped_utc, retired_utc=_now(),
                    metadata_sha256=hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
                    metadata_files_sha256=metadata,
                    stopped_snapshots=snapshots)
    finally:
        try:
            if stopped and not retired:
                if not _no_new_writer(root):
                    raise RuntimeError('new writer evidence appeared; old controller left stopped')
                try:
                    current = identity(pid)
                except FileNotFoundError:
                    current = None
                if current is not None:
                    if not same_identity(old_identity, current):
                        raise RuntimeError('cannot resume changed old controller identity')
                    signal.pidfd_send_signal(fd, signal.SIGCONT)
        finally:
            os.close(fd)
