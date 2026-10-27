"""Isolated, fail-stop authorization gate for an optional remote queue lane.

This module does not start production jobs. The caller must hold queue_run_lock
and route every local claim, launch, and ledger append through HandoffGate.
"""

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

from party_drift_queue import (_append_event, _write_once, classify_failure,
                               reserve_attempt)
from party_drift_audit_gate import verify_sealed_job
from party_drift_manifest import REPLAY_ARMS


class FailStop(RuntimeError):
    pass


class NotAdmitted(ValueError):
    """An idle opportunity disappeared before a key was claimed."""


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


def require_two_lock_slots(root):
    path = next_lock(root)
    if int(path.stem.rsplit('_', 1)[1]) > 98:
        raise FailStop('remote handoff needs two remaining resume lock slots')


def validate_remote_gpu(snapshot, uuid):
    if (snapshot.get('name') != 'NVIDIA GeForce RTX 4090' or
            snapshot.get('uuid') != uuid or not uuid.startswith('GPU-') or
            snapshot.get('free_mib', 0) < 6144 or
            type(snapshot.get('utilization')) is not int or
            not 0 <= snapshot['utilization'] <= 100 or
            not isinstance(snapshot.get('compute_pids'), list)):
        raise ValueError('remote GPU is not admitted')


def validate_broker_report(report, *, host, mount_source, manifest_sha256,
                           producer_commit, gpu_uuid, scheduler_commit=None,
                           environment=None):
    expected = {'host': host, 'mount_source': mount_source,
                'manifest_sha256': manifest_sha256,
                'producer_commit': producer_commit}
    if scheduler_commit is not None:
        expected['scheduler_commit'] = scheduler_commit
    if any(report.get(field) != value for field, value in expected.items()):
        raise ValueError('remote identity or shared mount differs')
    if type(report.get('pid')) is not int or report['pid'] <= 1:
        raise ValueError('remote broker PID is invalid')
    if type(report.get('result_disk_free_mib')) is not int or report[
            'result_disk_free_mib'] < 4096:
        raise ValueError('remote result filesystem lacks free space')
    if (not isinstance(report.get('scratch_path'), str) or
            not report['scratch_path'].startswith('/') or
            type(report.get('scratch_free_mib')) is not int or
            report['scratch_free_mib'] < 4096):
        raise ValueError('remote local scratch filesystem lacks free space')
    for field in ('driver', 'python', 'torch', 'torch_cuda'):
        if not isinstance(report.get(field), str) or not report[field]:
            raise ValueError(f'remote {field} is absent')
    if environment and any(report.get(field) != value
                           for field, value in environment.items()):
        raise ValueError('remote software environment differs')
    gpu = report.get('gpu', {})
    if (gpu.get('name') != 'NVIDIA GeForce RTX 4090' or
            gpu.get('uuid') != gpu_uuid or
            type(gpu.get('utilization')) is not int or
            not 0 <= gpu['utilization'] <= 100 or
            not isinstance(gpu.get('compute_pids'), list)):
        raise ValueError('remote GPU report differs')


def wait_remote_ready(snapshot, uuid, *, clock=time.monotonic, sleep=time.sleep,
                      stop=None):
    """Opportunity gate: 120 continuous seconds of low use and free memory."""
    since = None
    while True:
        if stop is not None and stop.is_set():
            return None
        now = clock()
        try:
            current = snapshot()
            validate_remote_gpu(current, uuid)
        except ValueError:
            since = None
        else:
            if since is None:
                since = now
            if now - since >= 120:
                return current
        if stop is not None:
            stop.wait(30)
        else:
            sleep(30)


def select_remote_keys(pending):
    """Fixed pre-run selection; only unfinished keys are eligible."""
    pending = set(pending)
    chosen = []
    for seed in (70, 72):
        key = f'cifar100/sample_only_off/none/{seed}'
        if key in pending:
            chosen.append(key)
    for alpha in ('0.25', '0.5', '1.0'):
        for seed in (70, 72):
            pair = [f'cifar100/{arm}/{alpha}/{seed}'
                    for arm in ('static', 'uniform')]
            if len(chosen) + 2 <= 4 and all(key in pending for key in pair):
                chosen.extend(pair)
    return chosen


def validate_remote_key_anchor(pending, anchored):
    if type(anchored) is not list or anchored != select_remote_keys(pending):
        raise ValueError('remote key anchor differs from fixed selection')


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

    def claim_local(self, remote_keys):
        with self.mutex:
            self._check()
            for index, key in enumerate(self.pending):
                if key not in remote_keys:
                    return self.pending.pop(index)
            return self.pending.pop(0) if self.pending else None

    def peek_remote(self, remote_keys):
        with self.mutex:
            self._check()
            return next((key for key in self.pending if key in remote_keys), None)

    def requeue_unstarted(self, key):
        with self.mutex:
            self._check()
            if (key in self.pending or
                    (self.root / key / 'attempt-01.plan.json').exists() or
                    any(json.loads(line).get('job_key') == key
                        for line in self.events.read_text(
                            encoding='utf-8').splitlines() if line)):
                self.failed = True
                raise FailStop('cannot requeue a planned or started key')
            self.pending.insert(0, key)

    def append(self, event):
        with self.mutex:
            self._check()
            _append_event(self.events, event, self.mutex)

    def local_launch(self, launch, *, ack_wait):
        """Keep the PID epoch stable until the new local worker authorizes."""
        with self.mutex:
            self._check()
            try:
                process = launch()
                ack_wait(process)
                return process
            except NotAdmitted:
                raise
            except BaseException as error:
                self.failed = True
                raise FailStop('local worker authorization uncertain') from error

    def _wait_ack(self, broker, attempt_dir, approved, key, *, seconds=900,
                  clock=time.monotonic, sleep=time.sleep):
        deadline = clock() + seconds
        approved = Path(approved)
        attempt_dir = Path(attempt_dir)
        approved_bytes = approved.read_bytes()
        expected = {
            'job_key': key, 'attempt': 1,
            'manifest_sha256': self.manifest_sha256,
            'config_sha256': hashlib.sha256(approved_bytes).hexdigest(),
            'production_commit': self.producer_commit,
            'actual_output_dir': str(attempt_dir),
            'test_allowed': False,
        }
        while clock() < deadline:
            if not broker.alive():
                raise FailStop('worker or broker lost before ACK')
            try:
                if (attempt_dir / 'APPROVED_CONFIG.json').read_bytes() == approved_bytes:
                    runtime = json.loads((attempt_dir / 'JOB_RUNTIME.json').read_text(
                        encoding='utf-8'))
                    if runtime == expected:
                        return
            except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
                pass
            sleep(1)
        raise FailStop('worker authorization ACK absent or incomplete')

    def authorize_remote(self, key, job_root, approved, broker, *, gpu_uuid,
                         gpu_snapshot=None, ack_wait=None, fresh_snapshot=None,
                         expected_host='40902',
                         expected_mount_source='shared:/brf',
                         expected_environment=None, ack_seconds=900):
        """Report PID -> lock/plan/start -> GO -> worker ACK -> local PID lock."""
        if gpu_snapshot is not None:
            validate_remote_gpu(gpu_snapshot, gpu_uuid)
        try:
            report = broker.receive_report()
            validate_broker_report(
                report, host=expected_host, mount_source=expected_mount_source,
                manifest_sha256=self.manifest_sha256,
                producer_commit=self.producer_commit,
                scheduler_commit=self.scheduler_commit, gpu_uuid=gpu_uuid,
                environment=expected_environment)
        except BaseException as error:
            self.failed = True
            raise FailStop('remote broker report failed audit') from error
        pid = report['pid']
        if gpu_snapshot is not None and report['gpu'] != gpu_snapshot:
            self.failed = True
            raise FailStop('remote reported GPU differs from admission snapshot')
        gpu_snapshot = report['gpu']
        if not broker.alive():
            self.failed = True
            raise FailStop('remote broker disconnected before handoff')
        job_root = Path(job_root).resolve()
        approved = Path(approved).resolve()
        attempt_dir = job_root / 'attempt-01'
        with self.mutex:
            self._check()
            if key not in self.pending:
                raise NotAdmitted('local lane claimed remote candidate first')
            if attempt_dir.exists():
                raise FailStop('remote key already has an attempt directory')
            require_two_lock_slots(self.root)
            if fresh_snapshot is None:
                raise ValueError('GO requires an independent fresh GPU snapshot')
            try:
                gpu_snapshot = fresh_snapshot()
                validate_remote_gpu(gpu_snapshot, gpu_uuid)
            except ValueError as error:
                raise NotAdmitted('remote GPU lost launch eligibility') from error
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
                    'host': expected_host,
                    'free_memory_mib': gpu_snapshot['free_mib'],
                    'gpu_utilization': gpu_snapshot['utilization'],
                    'temperature_c': gpu_snapshot.get('temperature_c'),
                    'sm_clock_mhz': gpu_snapshot.get('sm_clock_mhz'),
                    'sw_thermal_slowdown': gpu_snapshot.get(
                        'sw_thermal_slowdown'),
                    'driver': report['driver'],
                    'python': report['python'],
                    'torch': report['torch'],
                    'torch_cuda': report['torch_cuda'],
                    'co_tenant_pids': list(gpu_snapshot['compute_pids']),
                    'scheduler_commit': self.scheduler_commit,
                    'production_commit': self.producer_commit,
                }, self.mutex)
                broker.send_go()
                if ack_wait is not None:
                    ack_wait()
                self._wait_ack(broker, attempt_dir, approved, key,
                               seconds=ack_seconds)
                if not broker.alive():
                    raise FailStop('remote broker connection lost at ACK')
                if hasattr(broker, 'validate_launch_check'):
                    broker.validate_launch_check(gpu_uuid)
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


def dispatch_three(gate, remote_keys, *, ready_local, run_local,
                   ready_remote, run_remote):
    """Two unchanged local lanes and one optional remote lane share one key owner."""
    remote_keys = set(remote_keys)
    stop = threading.Event()
    outcomes = {}
    total = len(gate.pending)
    outcome_mutex = threading.Lock()

    def record(key, classification):
        with outcome_mutex:
            outcomes[key] = classification
            if classification not in ('success', 'training_anomaly') or len(outcomes) == total:
                stop.set()

    def local_lane(gpu):
        try:
            while not stop.is_set():
                if not ready_local(gpu, stop) or stop.is_set():
                    return
                key = gate.claim_local(remote_keys)
                if key is None:
                    return
                try:
                    classification = run_local(key, gpu)
                except NotAdmitted:
                    gate.requeue_unstarted(key)
                    continue
                record(key, classification)
        except BaseException:
            gate.failed = True
            stop.set()
            raise

    def remote_lane():
        try:
            while not stop.is_set():
                key = gate.peek_remote(remote_keys)
                if key is None or not ready_remote(stop) or stop.is_set():
                    return
                try:
                    classification = run_remote(key)
                except NotAdmitted:
                    remote_keys.discard(key)
                    continue
                record(key, classification)
        except BaseException:
            gate.failed = True
            stop.set()
            raise

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(local_lane, gpu) for gpu in (0, 1)]
        futures.append(pool.submit(remote_lane))
        for future in futures:
            try:
                future.result()
            except BaseException:
                gate.failed = True
                stop.set()
                raise
    return outcomes


def execute_remote_job(row, config, approved, gate, broker, *, gpu_uuid,
                       host, mount_source, environment=None):
    """One assigned remote job; central alone appends its finish event."""
    key = row['job_key']
    job_root = Path(config['output_dir']).resolve()
    if sha256(approved) != row['config_sha256']:
        raise ValueError('remote approved config differs from manifest')
    gate.authorize_remote(key, job_root, approved, broker,
        gpu_uuid=gpu_uuid, fresh_snapshot=broker.fresh_gpu,
        expected_host=host, expected_mount_source=mount_source,
        expected_environment=environment)
    try:
        code = broker.wait_exit()
        log = job_root / 'attempt-01.log'
        run = job_root / 'attempt-01'
        seal = run / 'JOB_SUCCESS.json'
        if code == 0 and seal.is_file():
            verify_sealed_job(run, key, gate.manifest_sha256,
                              replay=row['method'] in REPLAY_ARMS)
            classification = 'success'
        elif code != 0 and log.is_file():
            classification = classify_failure(code, log.read_text(
                encoding='utf-8', errors='replace')[-100000:])
        else:
            raise FailStop('remote result is missing or ambiguous')
        gate.append({
            'kind': 'finish', 'job_key': key, 'attempt': 1,
            'time_utc': datetime.now(timezone.utc).isoformat(),
            'returncode': code, 'classification': classification,
            'log_sha256': sha256(log),
            'job_success_sha256': sha256(seal) if classification == 'success' else None,
        })
        if classification not in ('success', 'training_anomaly'):
            raise FailStop('remote infrastructure failure; no automatic retry')
        return classification
    except BaseException as error:
        gate.failed = True
        raise FailStop('remote job uncertain; queue stopped') from error


def execute_local_job(row, config, approved, gate, *, producer_checkout,
                      manifest, gpu, gpu_uuid, snapshot_fn,
                      spawn=subprocess.Popen):
    """Launch under the shared PID barrier, wait for worker ACK, then finish."""
    from party_drift_balanced_resume import (shared_headroom_snapshot,
                                             worker_environment)

    key = row['job_key']
    job_root = Path(config['output_dir']).resolve()
    approved = Path(approved).resolve()
    if (sha256(approved) != row['config_sha256'] or
            config['device'] != 'cuda:0' or
            not job_root.is_relative_to(gate.root.resolve())):
        raise ValueError('local job differs from approved pending row')
    attempt_dir = job_root / 'attempt-01'
    log = job_root / 'attempt-01.log'
    stream = None

    def launch():
        nonlocal stream
        snapshot = snapshot_fn(gpu)
        if (snapshot['name'] != 'NVIDIA GeForce RTX 3080' or
                snapshot['uuid'] != gpu_uuid or
                not shared_headroom_snapshot(snapshot, gpu=gpu)):
            raise NotAdmitted('local GPU lost launch eligibility')
        if (attempt_dir.exists() or log.exists() or
                (job_root / 'attempt-01.plan.json').exists()):
            raise FailStop('local attempt already has artifacts before start')
        reserve_attempt(job_root, 1, key)
        start = {
            'kind': 'start', 'job_key': key, 'attempt': 1,
            'time_utc': datetime.now(timezone.utc).isoformat(),
            'config_sha256': row['config_sha256'],
            'physical_gpu_index': gpu, 'physical_gpu_uuid': gpu_uuid,
            'physical_gpu_name': snapshot['name'], 'host': os.uname().nodename
                if hasattr(os, 'uname') else 'local',
            'free_memory_mib': snapshot['free_mib'],
            'gpu_utilization': snapshot['utilization'],
            'co_tenant_pids': list(snapshot['compute_pids']),
            'temperature_c': snapshot.get('temperature_c'),
            'sm_clock_mhz': snapshot.get('sm_clock_mhz'),
            'sw_thermal_slowdown': snapshot.get('sw_thermal_slowdown'),
            'scheduler_commit': gate.scheduler_commit,
            'production_commit': gate.producer_commit,
        }
        gate.append(start)
        command = [sys.executable, '-u', str(Path(producer_checkout) /
                                            'run_party_drift_development.py'),
                   '--job-key', key, '--manifest', str(manifest),
                   '--manifest-sha256', gate.manifest_sha256,
                   '--attempt', '1']
        env = worker_environment(os.environ.copy(), gpu, config['seed'], gpu_uuid)
        stream = log.open('x', encoding='utf-8')
        return spawn(command, cwd=producer_checkout, env=env,
                     stdout=stream, stderr=subprocess.STDOUT)

    class Child:
        def __init__(self, process):
            self.process = process
        def alive(self):
            return self.process.poll() is None

    def await_ack(process):
        gate._wait_ack(Child(process), attempt_dir, approved, key)

    try:
        process = gate.local_launch(launch, ack_wait=await_ack)
        code = process.wait()
    finally:
        if stream is not None:
            stream.close()
    seal = attempt_dir / 'JOB_SUCCESS.json'
    if code == 0 and seal.is_file():
        verify_sealed_job(attempt_dir, key, gate.manifest_sha256,
                          replay=row['method'] in REPLAY_ARMS)
        classification = 'success'
    elif code != 0:
        classification = classify_failure(code, log.read_text(
            encoding='utf-8', errors='replace')[-100000:])
    else:
        gate.failed = True
        raise FailStop('local result is missing or ambiguous')
    gate.append({
        'kind': 'finish', 'job_key': key, 'attempt': 1,
        'time_utc': datetime.now(timezone.utc).isoformat(),
        'returncode': code, 'classification': classification,
        'log_sha256': sha256(log),
        'job_success_sha256': sha256(seal) if classification == 'success' else None,
    })
    if classification not in ('success', 'training_anomaly'):
        gate.failed = True
        raise FailStop('local infrastructure failure; no automatic retry')
    return classification
