"""Append-only NFS mailbox for one central queue writer and one remote broker."""

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path


class MailboxError(RuntimeError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, value):
    path = Path(path)
    descriptor, temp_name = tempfile.mkstemp(prefix=f'.{path.name}.',
                                             suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp_name, path)
    finally:
        os.unlink(temp_name)
    if os.name == 'posix':
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def replace_current(path, value):
    path = Path(path)
    descriptor, temp_name = tempfile.mkstemp(prefix=f'.{path.name}.',
                                             suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    if os.name == 'posix':
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class FileBroker:
    """Central-side adapter; the broker is launched independently on 40902."""

    def __init__(self, control, sequence, *, result_root, key,
                 manifest_sha256, producer_commit, config_sha256,
                 scheduler_commit='s',
                 clock=time.monotonic, heartbeat_seconds=90):
        self.control = Path(control).resolve()
        self.root = Path(result_root).resolve()
        if self.control.is_relative_to(self.root):
            raise ValueError('mailbox must be outside the result root')
        if type(sequence) is not int or sequence < 1:
            raise ValueError('mailbox sequence must be positive')
        self.sequence = sequence
        self.key = key
        self.manifest_sha256 = manifest_sha256
        self.producer_commit = producer_commit
        self.config_sha256 = config_sha256
        self.scheduler_commit = scheduler_commit
        self.clock = clock
        self.heartbeat_seconds = heartbeat_seconds
        self.heartbeat_index = 0
        self.last_heartbeat = clock()

    def path(self, kind):
        return self.control / f'{kind}_{self.sequence:06d}.json'

    def assign(self):
        write_once(self.path('ASSIGN'), {
            'protocol': 1,
            'sequence': self.sequence, 'job_key': self.key,
            'manifest_sha256': self.manifest_sha256,
            'production_commit': self.producer_commit,
            'scheduler_commit': self.scheduler_commit,
            'config_sha256': self.config_sha256,
        })

    def _heartbeat(self):
        path = self.control / 'HEARTBEAT_CURRENT.json'
        if path.is_file():
            heartbeat = json.loads(path.read_text(encoding='utf-8'))
            index = heartbeat.get('sequence')
            if type(index) is not int or index < self.heartbeat_index:
                raise MailboxError('remote heartbeat sequence regressed')
            if index > self.heartbeat_index:
                if heartbeat.get('protocol') != 1 or type(
                        heartbeat.get('pid')) is not int:
                    raise MailboxError('remote heartbeat differs')
                if hasattr(self, 'pid') and heartbeat['pid'] != self.pid:
                    raise MailboxError('remote heartbeat PID changed')
                self.heartbeat_index = index
                self.last_heartbeat = self.clock()
                return heartbeat
        return None

    def alive(self):
        if self.path('ABORT').is_file():
            raise MailboxError('remote broker aborted after GO')
        self._heartbeat()
        return (self.heartbeat_index > 0 and
                self.clock() - self.last_heartbeat <= self.heartbeat_seconds)

    def _wait(self, kind, seconds=900):
        deadline = self.clock() + seconds
        path = self.path(kind)
        while self.clock() < deadline:
            if not self.alive():
                raise MailboxError('remote broker heartbeat expired')
            if path.is_file():
                return json.loads(path.read_text(encoding='utf-8'))
            time.sleep(1)
        raise MailboxError(f'remote {kind} absent; no retry authorized')

    def receive_report(self):
        ready = self._wait('READY')
        if (ready.get('protocol') != 1 or
                ready.get('sequence') != self.sequence or
                ready.get('job_key') != self.key or
                ready.get('manifest_sha256') != self.manifest_sha256 or
                ready.get('production_commit') != self.producer_commit or
                ready.get('scheduler_commit') != self.scheduler_commit or
                ready.get('assignment_sha256') != sha256(self.path('ASSIGN'))):
            raise MailboxError('remote READY differs from assignment')
        self.ready_sha256 = sha256(self.path('READY'))
        self.ready_heartbeat_index = self.heartbeat_index
        report = ready['report']
        self.pid = report['pid']
        path = self.control / 'HEARTBEAT_CURRENT.json'
        heartbeat = json.loads(path.read_text(encoding='utf-8'))
        if heartbeat['pid'] != self.pid:
            raise MailboxError('remote READY PID differs from heartbeat')
        return report

    def fresh_gpu(self):
        deadline = self.clock() + self.heartbeat_seconds
        while self.clock() < deadline:
            heartbeat = self._heartbeat()
            if not self.alive():
                raise MailboxError('remote heartbeat expired before GO')
            if heartbeat and self.heartbeat_index > self.ready_heartbeat_index:
                return heartbeat['gpu']
            time.sleep(1)
        raise MailboxError('no fresh GPU heartbeat before GO')

    def send_go(self):
        write_once(self.path('GO'), {
            'protocol': 1,
            'sequence': self.sequence, 'job_key': self.key,
            'manifest_sha256': self.manifest_sha256,
            'production_commit': self.producer_commit,
            'scheduler_commit': self.scheduler_commit,
            'assignment_sha256': sha256(self.path('ASSIGN')),
            'ready_sha256': self.ready_sha256,
        })

    def cancel(self):
        write_once(self.path('CANCEL'), {
            'protocol': 1, 'sequence': self.sequence, 'job_key': self.key,
            'manifest_sha256': self.manifest_sha256,
            'production_commit': self.producer_commit,
            'scheduler_commit': self.scheduler_commit,
            'assignment_sha256': sha256(self.path('ASSIGN')),
            'ready_sha256': sha256(self.path('READY')),
        })

    def validate_launch_check(self, gpu_uuid):
        from party_drift_multihost import validate_remote_gpu
        record = json.loads(self.path('LAUNCH_CHECK').read_text(encoding='utf-8'))
        expected = {
            'protocol': 1, 'sequence': self.sequence, 'job_key': self.key,
            'manifest_sha256': self.manifest_sha256,
            'production_commit': self.producer_commit,
            'scheduler_commit': self.scheduler_commit,
            'assignment_sha256': sha256(self.path('ASSIGN')),
            'ready_sha256': self.ready_sha256,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise MailboxError('launch check differs from assignment')
        validate_remote_gpu(record.get('gpu', {}), gpu_uuid)
        return record['gpu']

    def wait_exit(self):
        result = self._wait('EXIT', seconds=86400)
        if (result.get('protocol') != 1 or
                result.get('sequence') != self.sequence or
                result.get('job_key') != self.key or
                result.get('manifest_sha256') != self.manifest_sha256 or
                result.get('production_commit') != self.producer_commit or
                result.get('scheduler_commit') != self.scheduler_commit or
                result.get('assignment_sha256') != sha256(self.path('ASSIGN')) or
                type(result.get('returncode')) is not int):
            raise MailboxError('remote EXIT differs from assignment')
        return result['returncode']
