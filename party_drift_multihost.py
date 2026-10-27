"""Isolated, fail-stop authorization gate for an optional remote queue lane.

This module does not start production jobs. The caller must hold queue_run_lock
and route every local claim, launch, and ledger append through HandoffGate.
"""

import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from party_drift_queue import _append_event, _write_once, reserve_attempt


class FailStop(RuntimeError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def active_lock(root):
    records = sorted(Path(root).glob('QUEUE_LOCK_RESUME_*.json'))
    if not records:
        raise FileNotFoundError('no resume lock')
    return json.loads(records[-1].read_text(encoding='utf-8'))


def next_lock(root):
    numbers = [int(path.stem.rsplit('_', 1)[1]) for path in
               Path(root).glob('QUEUE_LOCK_RESUME_*.json')]
    number = max(numbers, default=0) + 1
    if number >= 100:
        raise FailStop('resume lock sequence exhausted')
    return Path(root) / f'QUEUE_LOCK_RESUME_{number:02d}.json'


def validate_remote_gpu(snapshot, uuid):
    if (snapshot.get('name') != 'NVIDIA GeForce RTX 4090' or
            snapshot.get('uuid') != uuid or not uuid.startswith('GPU-') or
            snapshot.get('free_mib', 0) < 6144 or
            snapshot.get('utilization', 100) >= 10 or
            not isinstance(snapshot.get('compute_pids'), list)):
        raise ValueError('remote GPU is not admitted')


def wait_remote_ready(snapshot, uuid, *, clock=time.monotonic, sleep=time.sleep):
    """Opportunity gate: 120 continuous seconds of low use and free memory."""
    since = None
    while True:
        current = snapshot()
        now = clock()
        try:
            validate_remote_gpu(current, uuid)
        except ValueError:
            since = None
        else:
            if since is None:
                since = now
            if now - since >= 120:
                return current
        sleep(30)


class JsonBroker:
    """Line protocol over a caller-owned SSH process's pipes."""

    def __init__(self, incoming, outgoing, poll):
        self.incoming = incoming
        self.outgoing = outgoing
        self.poll = poll

    def receive_report(self):
        line = self.incoming.readline()
        if not line:
            raise FailStop('remote broker disconnected before PID report')
        return json.loads(line)

    def receive_pid(self):
        return self.receive_report()['pid']

    def send_go(self):
        self.outgoing.write('GO\n')
        self.outgoing.flush()

    def alive(self):
        return self.poll() is None


class HandoffGate:
    """One coordinator owns pending keys, append operations, and PID epochs."""

    def __init__(self, root, events, pending, *, manifest_sha256,
                 producer_commit, scheduler_commit):
        self.root = Path(root)
        self.events = Path(events)
        self.pending = list(pending)
        self.manifest_sha256 = manifest_sha256
        self.producer_commit = producer_commit
        self.scheduler_commit = scheduler_commit
        self.mutex = threading.RLock()
        self.failed = False

    def _check(self):
        if self.failed:
            raise FailStop('queue stopped after uncertain remote handoff')

    def claim(self):
        with self.mutex:
            self._check()
            return self.pending.pop(0) if self.pending else None

    def append(self, event):
        with self.mutex:
            self._check()
            _append_event(self.events, event, self.mutex)

    def local_launch(self, launch):
        """Launch must return promptly with a process handle; wait outside."""
        with self.mutex:
            self._check()
            return launch()

    def _wait_ack(self, broker, attempt_dir, approved, key, *, seconds=900):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not broker.alive():
                raise FailStop('remote broker connection lost before ACK')
            if ((attempt_dir / 'APPROVED_CONFIG.json').is_file() and
                    (attempt_dir / 'JOB_RUNTIME.json').is_file()):
                return
            time.sleep(1)
        raise FailStop('remote authorization ACK absent; key remains claimed')

    def authorize_remote(self, key, job_root, approved, broker, *, gpu_uuid,
                         gpu_snapshot, ack_wait=None):
        """Report PID -> lock/plan/start -> GO -> worker ACK -> local PID lock."""
        validate_remote_gpu(gpu_snapshot, gpu_uuid)
        pid = broker.receive_pid()
        if type(pid) is not int or pid <= 1 or not broker.alive():
            raise FailStop('remote broker PID is invalid or disconnected')
        job_root = Path(job_root).resolve()
        approved = Path(approved).resolve()
        attempt_dir = job_root / 'attempt-01'
        with self.mutex:
            self._check()
            if key not in self.pending or attempt_dir.exists():
                raise FailStop('remote key is already claimed or attempted')
            # Keep the gate locked through ACK: local starts and all ledger writes wait.
            self.pending.remove(key)
            try:
                _write_once(next_lock(self.root), {
                    'pid': pid, 'status': 'remote_authorization',
                    'manifest_sha256': self.manifest_sha256,
                    'production_commit': self.producer_commit,
                    'scheduler_commit': self.scheduler_commit,
                    'started_utc': datetime.now(timezone.utc).isoformat(),
                })
                reserve_attempt(job_root, 1, key)
                _append_event(self.events, {
                    'kind': 'start', 'job_key': key, 'attempt': 1,
                    'time_utc': datetime.now(timezone.utc).isoformat(),
                    'config_sha256': sha256(approved),
                    'physical_gpu_index': 0,
                    'physical_gpu_uuid': gpu_uuid,
                    'physical_gpu_name': gpu_snapshot['name'],
                    'host': '40902',
                    'free_memory_mib': gpu_snapshot['free_mib'],
                    'co_tenant_pids': list(gpu_snapshot['compute_pids']),
                    'scheduler_commit': self.scheduler_commit,
                    'production_commit': self.producer_commit,
                }, self.mutex)
                broker.send_go()
                if ack_wait is None:
                    self._wait_ack(broker, attempt_dir, approved, key)
                else:
                    ack_wait()
                if not broker.alive():
                    raise FailStop('remote broker connection lost at ACK')
                if (not (attempt_dir / 'APPROVED_CONFIG.json').is_file() or
                        (attempt_dir / 'APPROVED_CONFIG.json').read_bytes() !=
                        approved.read_bytes()):
                    raise FailStop('approved config ACK differs')
                runtime = json.loads((attempt_dir / 'JOB_RUNTIME.json').read_text(
                    encoding='utf-8'))
                expected = {
                    'job_key': key, 'attempt': 1,
                    'manifest_sha256': self.manifest_sha256,
                    'config_sha256': sha256(approved),
                    'production_commit': self.producer_commit,
                    'actual_output_dir': str(attempt_dir),
                    'test_allowed': False,
                }
                if any(runtime.get(name) != value for name, value in expected.items()):
                    raise FailStop('runtime ACK differs')
                _write_once(next_lock(self.root), {
                    'pid': os.getpid(), 'status': 'running',
                    'manifest_sha256': self.manifest_sha256,
                    'production_commit': self.producer_commit,
                    'scheduler_commit': self.scheduler_commit,
                    'started_utc': datetime.now(timezone.utc).isoformat(),
                })
            except BaseException as error:
                self.failed = True
                raise FailStop('remote authorization uncertain; do not retry key') from error
