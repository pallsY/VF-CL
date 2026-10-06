#!/usr/bin/env python3
"""Fail-closed audited intermediate retention; CLI uses live authority."""
import argparse
from contextlib import contextmanager, ExitStack
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import secrets
import select
import signal
import stat
import subprocess
import sys
import time

PLAN_NAME = 'PRUNE_PLAN.json'
COMPLETE_NAME = 'PRUNED_EVIDENCE.json'
POLICY = 'completed-run-intermediate-v1'
QUARANTINE_PREFIX = '.prune-quarantine-'
SHA256 = re.compile(r'[0-9a-f]{64}')
COMMIT = re.compile(r'[0-9a-f]{40}')
# 2026-09-12: the user explicitly confirmed this existing SFTP instance was not
# reading/writing VF-CL and accepted ignoring it. This is not a program-name or
# UID exemption and cannot establish anything about its future file activity.
USER_AUTHORIZED_PROCESS_INSTANCE = {
    'boot_id': '45f74e91-2a45-443e-93e3-6a6f1435b74f',
    'pid': 1499350, 'uid': 1000, 'start_time': 51143674,
    'cmdline_sha256': 'a26e51b5c84c3a90917af4c407e079c6f835a039c8c506fd91442650e29fd49f',
    'ppid': 328962, 'parent_start_time': 42107480,
    'parent_cmdline_sha256': '588b8493fd781b8abebe40c33111eebd54b44244c376dcd495d2ab84b00294ef',
}
USER_AUTHORIZED_PROCESS_REASON = (
    '2026-09-12 user confirmed SFTP PID 1499350 with this exact identity was not reading/writing '
    'VF-CL and explicitly accepted skipping its cwd inspection; this does not '
    'prove future inactivity.')
# A separate user confirmation covers only this second complete identity.
USER_AUTHORIZED_SECOND_PROCESS_INSTANCE = {
    'boot_id': '45f74e91-2a45-443e-93e3-6a6f1435b74f',
    'pid': 1560782, 'uid': 1000, 'start_time': 51571438,
    'cmdline_sha256': 'a26e51b5c84c3a90917af4c407e079c6f835a039c8c506fd91442650e29fd49f',
    'ppid': 1560780, 'parent_start_time': 51571420,
    'parent_cmdline_sha256': '8c00a6ad1a9dd00a3933a5301dd50aa69a7dc63e273664ad0da0b1f51e731fc5',
}
USER_AUTHORIZED_SECOND_PROCESS_REASON = (
    '2026-09-12 user separately confirmed SFTP PID 1560782 with this exact identity '
    'was not reading/writing VF-CL and explicitly accepted skipping its cwd '
    'inspection; this does not prove future inactivity.')
# User-approved 2026-09-24 exception: only these two live process instances.
# Their future inactivity cannot be verified; any identity change fails closed.
USER_AUTHORIZED_SCOPED_PROCESS_INSTANCES = (
    ({'boot_id': '51a87e5b-59f7-48b1-aad8-b01d9b47be67',
      'pid': 3205670, 'uid': 1000, 'start_time': 24255716,
      'cmdline_sha256': 'a26e51b5c84c3a90917af4c407e079c6f835a039c8c506fd91442650e29fd49f',
      'ppid': 3205640, 'parent_start_time': 24255668,
      'parent_cmdline_sha256': 'a6f3ee7fca934e051385f0d82a9a2410fde560e4c1cec269b521b0d33a7492db'},
     '2026-09-24 user explicitly accepted ignoring this exact other-user SFTP '
     'instance during VF-CL; future file inactivity is not verified.'),
    ({'boot_id': '86ed0e8a-8e8c-40f6-b7af-0dcb5ae918ac',
      'pid': 176899, 'uid': 1000, 'start_time': 86695586,
      'cmdline_sha256': 'a26e51b5c84c3a90917af4c407e079c6f835a039c8c506fd91442650e29fd49f',
      'ppid': 176898, 'parent_start_time': 86695579,
      'parent_cmdline_sha256': '2f2758c74ddb7f77965e3922c993a52dbe82cdb5557a9a5eafb2b65accb4d247'},
     '2026-09-24 user explicitly accepted ignoring this exact other-user SFTP '
     'instance during VF-CL; future file inactivity is not verified.'),
)
TARGET = re.compile(r'(?:checkpoints/(?:event_[0-9]+_CIL|resume_latest)|formal_snapshots/event_[0-9]+_CIL)\.pt')
RENAME_NOREPLACE = 1
_LIBC = ctypes.CDLL(None, use_errno=True)
_RENAMEAT2 = getattr(_LIBC, 'renameat2', None)
if _RENAMEAT2 is not None:
    _RENAMEAT2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint)
    _RENAMEAT2.restype = ctypes.c_int


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode() + b'\n'


def directory_identity(details):
    return details.st_dev, details.st_ino, details.st_mode


def file_identity(details):
    return (*directory_identity(details), details.st_nlink, details.st_size,
            details.st_mtime_ns, details.st_ctime_ns)


def component(name):
    if type(name) is not str or name in ('', '.', '..') or '/' in name or '\x00' in name:
        raise ValueError('unsafe directory entry name')
    return name


def rename_noreplace(source_parent, source, destination_parent, destination):
    """Atomically move one entry without replacing an existing destination."""
    source, destination = component(source), component(destination)
    source_parent.verify()
    destination_parent.verify()
    if _RENAMEAT2 is None:
        raise OSError(errno.ENOSYS, 'renameat2 is unavailable')
    ctypes.set_errno(0)
    if _RENAMEAT2(source_parent.fd, os.fsencode(source),
                  destination_parent.fd, os.fsencode(destination),
                  RENAME_NOREPLACE) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), f'{source} -> {destination}')


def _term_checked_pidfd(pid, identity, audit=None):
    if audit is None:
        audit = {}
    audit['outcome'] = 'signal_not_sent'
    libc = ctypes.CDLL(None, use_errno=True)
    opened = getattr(libc, 'pidfd_open', None)
    send = getattr(libc, 'pidfd_send_signal', None)
    if opened is None or send is None:
        raise OSError('pidfd libc wrappers are unavailable')
    opened.argtypes, opened.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
    send.argtypes, send.restype = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint], ctypes.c_int
    fd = opened(pid, 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'pidfd_open')
    try:
        first = identity()
        audit['identity'] = first
        if type(first) is not dict or first.get('pid') != pid or first != identity():
            raise ValueError('SFTP instance identity changed')
        if send(fd, signal.SIGTERM, None, 0) < 0:
            raise OSError(ctypes.get_errno(), 'pidfd_send_signal')
        audit['outcome'] = 'term_sent_exit_unconfirmed'
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        if not any(events & select.POLLIN for _, events in poller.poll(5000)):
            raise TimeoutError('SFTP instance did not exit after TERM')
        audit['outcome'] = 'term_sent_exit_observed'
        return first
    finally:
        os.close(fd)


def _pidfd_confirms_exit(pid, proc_root):
    """Read-only confirmation that a PID no longer names a live process."""
    # A synthetic proc tree has no relationship to kernel PIDs.
    if Path(proc_root) != Path('/proc'):
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    opened = getattr(libc, 'pidfd_open', None)
    if opened is None:
        raise OSError('pidfd_open libc wrapper is unavailable')
    opened.argtypes, opened.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
    fd = opened(pid, 0)
    if fd < 0:
        code = ctypes.get_errno()
        if code == errno.ESRCH:
            return True
        raise OSError(code, 'pidfd_open')
    try:
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        events = poller.poll(0)
        if any(flags & (select.POLLERR | select.POLLNVAL) for _, flags in events):
            raise OSError('pidfd poll failed')
        return any(flags & select.POLLIN for _, flags in events)
    finally:
        os.close(fd)


class PinnedDir:
    """Pin every ancestor; never resolve symlinks into deletion authority."""
    def __init__(self, path, parent=None):
        self.fd = None
        self.owns_parent = False
        self.kept = {}
        if parent is None:
            path = Path(path)
            if not path.is_absolute() or '..' in path.parts:
                raise ValueError('pinned path must be exact absolute path')
            if path != Path('/'):
                parent = PinnedDir(path.parent)
                self.owns_parent = True
                name = path.name
            else:
                name = '/'
        else:
            name = component(str(path))
            path = parent.path / name
        self.path, self.parent, self.name = path, parent, name
        try:
            if parent:
                parent.verify()
            named = os.stat(name, dir_fd=parent.fd if parent else None,
                            follow_symlinks=False)
            if not stat.S_ISDIR(named.st_mode):
                raise ValueError('pinned parent is not a regular directory')
            self.fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=parent.fd if parent else None)
            self.details = os.fstat(self.fd)
            if directory_identity(named) != directory_identity(self.details):
                raise ValueError('pinned directory identity changed during open')
            self.verify()
        except BaseException:
            self.close()
            raise

    def verify(self):
        if self.parent:
            self.parent.verify()
        named = os.stat(self.name, dir_fd=self.parent.fd if self.parent else None,
                        follow_symlinks=False)
        if (not stat.S_ISDIR(named.st_mode)
                or directory_identity(named) != directory_identity(self.details)
                or directory_identity(os.fstat(self.fd)) != directory_identity(self.details)):
            raise ValueError('pinned directory identity changed')
        for name, expected in self.kept.items():
            actual = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            if file_identity(actual) != expected:
                raise ValueError('kept file identity changed')

    def present(self, name):
        self.verify()
        try:
            os.stat(component(name), dir_fd=self.fd, follow_symlinks=False)
            return True
        except FileNotFoundError:
            return False

    def names(self):
        self.verify()
        names = os.listdir(self.fd)
        self.verify()
        return names

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        if self.owns_parent:
            self.parent.close()
            self.owns_parent = False

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


@contextmanager
def pinned(value):
    if isinstance(value, PinnedDir):
        value.verify()
        yield value
        value.verify()
    else:
        with PinnedDir(value) as directory:
            yield directory
            directory.verify()


def verify_file(parent, name, descriptor, details):
    parent.verify()
    named = os.stat(name, dir_fd=parent.fd, follow_symlinks=False)
    if (file_identity(named) != file_identity(details)
            or file_identity(os.fstat(descriptor)) != file_identity(details)):
        raise ValueError('pinned file identity changed')


@contextmanager
def opened_file(parent, name, marker=False, remember=False):
    parent.verify()
    name = component(name)
    named = os.stat(name, dir_fd=parent.fd, follow_symlinks=False)
    if not stat.S_ISREG(named.st_mode) or named.st_nlink != 1:
        raise ValueError('target is not a single-link regular file')
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=parent.fd)
    try:
        verify_file(parent, name, descriptor, named)
        if marker and stat.S_IMODE(named.st_mode) != 0o444:
            raise ValueError('installed marker mode differs')
        if remember:
            parent.kept.setdefault(name, file_identity(named))
            parent.verify()
        yield descriptor, named
        verify_file(parent, name, descriptor, named)
    finally:
        os.close(descriptor)


def file_hash(descriptor):
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def read_json(parent, name, marker=True):
    with opened_file(parent, name, marker=marker, remember=True) as (descriptor, _):
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        content = b''.join(chunks)
        value = json.loads(content)
        if content != canonical_bytes(value):
            raise ValueError('installed JSON is not canonical')
        return value


def load_json_canonical(path):
    path = Path(path)
    with PinnedDir(path.parent) as parent:
        return read_json(parent, path.name)


def write_marker(parent, name, value):
    parent.verify()
    component(name)
    temporary = f'.{name}.{secrets.token_hex(16)}.tmp'
    descriptor = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL
                         | os.O_NOFOLLOW, 0o400, dir_fd=parent.fd)
    details = os.fstat(descriptor)
    try:
        payload = memoryview(canonical_bytes(value))
        while payload:
            written = os.write(descriptor, payload)
            if written <= 0:
                raise OSError('short marker write')
            payload = payload[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        details = os.fstat(descriptor)
        verify_file(parent, temporary, descriptor, details)
        os.link(temporary, name, src_dir_fd=parent.fd, dst_dir_fd=parent.fd,
                follow_symlinks=False)
        linked = os.fstat(descriptor)
        if linked.st_nlink != 2:
            raise ValueError('marker link count changed')
        verify_file(parent, name, descriptor, linked)
        verify_file(parent, temporary, descriptor, linked)
        os.unlink(temporary, dir_fd=parent.fd)
        final = os.fstat(descriptor)
        if final.st_nlink != 1 or directory_identity(final) != directory_identity(details):
            raise ValueError('marker final identity changed')
        verify_file(parent, name, descriptor, final)
        os.fsync(parent.fd)
        if read_json(parent, name) != value:
            raise ValueError('marker final content changed')
        verify_file(parent, name, descriptor, final)
    finally:
        # Only clean our temporary inode, never a replacement entry.
        try:
            named = os.stat(temporary, dir_fd=parent.fd, follow_symlinks=False)
            if (named.st_dev, named.st_ino) == (details.st_dev, details.st_ino):
                os.unlink(temporary, dir_fd=parent.fd)
        except FileNotFoundError:
            pass
        os.close(descriptor)


def write_json_exclusive(path, value):
    path = Path(path)
    with PinnedDir(path.parent) as parent:
        write_marker(parent, path.name, value)


def expected_paths(num_tasks):
    if type(num_tasks) is not int or num_tasks <= 0:
        raise ValueError('task count is invalid')
    return sorted([f'{directory}/event_{i}_CIL.pt'
                   for directory in ('checkpoints', 'formal_snapshots')
                   for i in range(num_tasks)] + ['checkpoints/resume_latest.pt'])


@contextmanager
def target_parents(run):
    with pinned(run) as directory, ExitStack() as stack:
        if hasattr(directory, 'target_dirs'):
            yield directory.target_dirs
        else:
            directory.target_dirs = {name: stack.enter_context(PinnedDir(name, directory))
                                     for name in ('checkpoints', 'formal_snapshots')}
            try:
                yield directory.target_dirs
                for parent in directory.target_dirs.values():
                    parent.verify()
            finally:
                del directory.target_dirs


def checked_target(parents, relative, wanted, size=None):
    group, name = relative.split('/')
    with opened_file(parents[group], name, remember=name == 'formal_final.pt') as (descriptor, details):
        if (SHA256.fullmatch(str(wanted)) is None or file_hash(descriptor) != wanted
                or size is not None and details.st_size != size):
            raise ValueError(f'prune target hash/size differs: {relative}')
        return details


def evidence_hashes(evidence, num_tasks):
    if (type(evidence) is not dict or evidence.get('kind') != 'formal_resource_evidence'
            or type(evidence.get('spec_key')) is not str
            or type(evidence.get('artifact_sha256')) is not dict):
        raise ValueError('resource evidence is invalid')
    artifacts = evidence['artifact_sha256']
    paths = expected_paths(num_tasks)
    for group in ('checkpoints/event_', 'formal_snapshots/event_'):
        if sorted(n for n in artifacts if n.startswith(group)) != sorted(n for n in paths if n.startswith(group)):
            raise ValueError('prunable artifact sequence differs')
    hashes = {name: artifacts.get(name) for name in paths}
    hashes['checkpoints/resume_latest.pt'] = artifacts.get(f'checkpoints/event_{num_tasks - 1}_CIL.pt')
    return hashes


def build_prune_plan(run, evidence, record_hash, source_commit, num_tasks):
    hashes = evidence_hashes(evidence, num_tasks)
    with target_parents(run) as parents:
        checked_target(parents, 'checkpoints/formal_final.pt',
                       evidence['artifact_sha256'].get('checkpoints/formal_final.pt'))
        files = []
        for name, digest in sorted(hashes.items()):
            details = checked_target(parents, name, digest)
            files.append({
                'path': name, 'sha256': digest, 'size': details.st_size,
                'device': details.st_dev, 'inode': details.st_ino,
                'mode': stat.S_IMODE(details.st_mode),
            })
    return validate_plan({'kind': 'formal_prune_plan', 'policy': POLICY,
                          'spec_key': evidence['spec_key'], 'source_commit': source_commit,
                          'record_sha256': record_hash, 'files': files,
                          'bytes': sum(item['size'] for item in files)})


def validate_plan(plan):
    if (type(plan) is not dict or set(plan) != {
            'kind', 'policy', 'spec_key', 'source_commit', 'record_sha256', 'files', 'bytes'}
            or plan['kind'] != 'formal_prune_plan' or plan['policy'] != POLICY
            or type(plan['spec_key']) is not str or not plan['spec_key']
            or type(plan['source_commit']) is not str
            or COMMIT.fullmatch(str(plan['source_commit'])) is None
            or type(plan['record_sha256']) is not str
            or SHA256.fullmatch(str(plan['record_sha256'])) is None
            or type(plan['files']) is not list or not plan['files']
            or type(plan['bytes']) is not int or plan['bytes'] <= 0):
        raise ValueError('installed prune plan schema is invalid')
    for item in plan['files']:
        if (type(item) is not dict or set(item) != {
                    'path', 'sha256', 'size', 'device', 'inode', 'mode'}
                or type(item['path']) is not str or TARGET.fullmatch(item['path']) is None
                or type(item['sha256']) is not str
                or SHA256.fullmatch(str(item['sha256'])) is None
                or any(type(item[field]) is not int or isinstance(item[field], bool)
                       for field in ('size', 'device', 'inode', 'mode'))
                or item['size'] <= 0 or item['device'] < 0 or item['inode'] <= 0
                or item['mode'] < 0 or item['mode'] > 0o7777):
            raise ValueError('installed prune file/path schema is invalid')
    names = [i['path'] for i in plan['files']]
    if names != sorted(set(names)) or sum(i['size'] for i in plan['files']) != plan['bytes']:
        raise ValueError('installed prune plan totals differ')
    return plan


def validate_plan_against_evidence(run, plan, evidence, record_hash, source_commit,
                                   num_tasks, completed=False):
    validate_plan(plan)
    hashes = evidence_hashes(evidence, num_tasks)
    if (plan['spec_key'] != evidence['spec_key'] or plan['record_sha256'] != record_hash
            or plan['source_commit'] != source_commit
            or {i['path']: i['sha256'] for i in plan['files']} != hashes):
        raise ValueError('installed plan differs from evidence')
    with target_parents(run) as parents:
        checked_target(parents, 'checkpoints/formal_final.pt',
                       evidence['artifact_sha256'].get('checkpoints/formal_final.pt'))
        for item in plan['files']:
            group, name = item['path'].split('/')
            if completed:
                if parents[group].present(name):
                    raise ValueError('completed prune target is present')
            else:
                details = checked_target(
                    parents, item['path'], item['sha256'], item['size'])
                if (details.st_dev != item['device']
                        or details.st_ino != item['inode']
                        or stat.S_IMODE(details.st_mode) != item['mode']):
                    raise ValueError('installed plan target identity differs')
    return plan


def install_plan(run, plan):
    with pinned(run) as directory:
        write_marker(directory, PLAN_NAME, validate_plan(plan))


def load_installed_plan(run):
    with pinned(run) as directory:
        return validate_plan(read_json(directory, PLAN_NAME))


@contextmanager
def quarantine(run):
    """Create and pin one private same-filesystem forensic quarantine."""
    with pinned(run) as directory:
        name = component(QUARANTINE_PREFIX + secrets.token_hex(16))
        directory.verify()
        os.mkdir(name, 0o700, dir_fd=directory.fd)
        os.fsync(directory.fd)
        descriptor = os.open(
            name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=directory.fd)
        try:
            named = os.stat(name, dir_fd=directory.fd, follow_symlinks=False)
            opened = os.fstat(descriptor)
            if (not stat.S_ISDIR(named.st_mode)
                    or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)):
                raise ValueError('quarantine identity changed during creation')
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
            opened = os.fstat(descriptor)
            if stat.S_IMODE(opened.st_mode) != 0o700 or opened.st_nlink != 2:
                raise ValueError('quarantine mode/link count differs')
        finally:
            os.close(descriptor)
        child = PinnedDir(name, directory)
        try:
            if (stat.S_IMODE(child.details.st_mode) != 0o700
                    or child.details.st_nlink != 2
                    or child.names()):
                raise ValueError('quarantine authority differs')
            yield child
            child.verify()
            os.fsync(child.fd)
        finally:
            child.close()
            directory.verify()
            os.fsync(directory.fd)


def _matches_plan(details, item, *, size):
    return (stat.S_ISREG(details.st_mode) and details.st_nlink == 1
            and details.st_dev == item['device']
            and details.st_ino == item['inode'] and details.st_size == size)


def _quarantine_truncate(parent, group, name, item, holding):
    """Move one exact inode and release only that inode's data through its fd."""
    parent.verify()
    named = os.stat(name, dir_fd=parent.fd, follow_symlinks=False)
    if (not _matches_plan(named, item, size=item['size'])
            or stat.S_IMODE(named.st_mode) != item['mode']):
        raise ValueError('prune target identity differs from immutable plan')
    descriptor = os.open(
        name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
        dir_fd=parent.fd)
    try:
        verify_file(parent, name, descriptor, named)
        if file_hash(descriptor) != item['sha256']:
            raise ValueError('prune target hash differs')
        verify_file(parent, name, descriptor, named)
        held_name = component(f'{group}.{name}')
        rename_noreplace(parent, name, holding, held_name)
        os.fsync(parent.fd)
        os.fsync(holding.fd)
        moved = os.stat(held_name, dir_fd=holding.fd, follow_symlinks=False)
        moved_descriptor = os.open(
            held_name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=holding.fd)
        try:
            verify_file(holding, held_name, moved_descriptor, moved)
            original = os.fstat(descriptor)
            if (not _matches_plan(moved, item, size=item['size'])
                    or not _matches_plan(original, item, size=item['size'])
                    or stat.S_IMODE(moved.st_mode) != item['mode']
                    or file_hash(moved_descriptor) != item['sha256']):
                raise ValueError('quarantined prune target differs')
            verify_file(holding, held_name, moved_descriptor, moved)
            if parent.present(name):
                raise ValueError('prune target name was recreated')

            writable_mode = item['mode'] | stat.S_IWUSR
            os.fchmod(moved_descriptor, writable_mode)
            mode_needs_restore = True
            try:
                ready = os.stat(
                    held_name, dir_fd=holding.fd, follow_symlinks=False)
                if (not _matches_plan(ready, item, size=item['size'])
                        or stat.S_IMODE(ready.st_mode) != writable_mode):
                    raise ValueError('quarantine changed before writable reopen')
                writable = os.open(
                    held_name, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
                    dir_fd=holding.fd)
                try:
                    opened = os.fstat(writable)
                    current = os.stat(
                        held_name, dir_fd=holding.fd, follow_symlinks=False)
                    if (not _matches_plan(opened, item, size=item['size'])
                            or not _matches_plan(current, item, size=item['size'])
                            or (opened.st_dev, opened.st_ino)
                            != (current.st_dev, current.st_ino)
                            or stat.S_IMODE(opened.st_mode) != writable_mode):
                        raise ValueError('writable quarantine inode differs')
                    os.ftruncate(writable, 0)
                    os.fchmod(writable, item['mode'])
                    mode_needs_restore = False
                    os.fsync(writable)
                    after = os.fstat(writable)
                    final = os.stat(
                        held_name, dir_fd=holding.fd, follow_symlinks=False)
                    if (not _matches_plan(after, item, size=0)
                            or not _matches_plan(final, item, size=0)
                            or (after.st_dev, after.st_ino)
                            != (final.st_dev, final.st_ino)
                            or stat.S_IMODE(after.st_mode) != item['mode']):
                        raise ValueError('quarantine tombstone identity differs')
                finally:
                    os.close(writable)
            except BaseException as primary:
                if mode_needs_restore:
                    try:
                        os.fchmod(moved_descriptor, item['mode'])
                        restored = os.fstat(moved_descriptor)
                        if stat.S_IMODE(restored.st_mode) != item['mode']:
                            raise ValueError('restored quarantine mode differs')
                    except BaseException as restoration:
                        failure = RuntimeError(
                            'quarantine mode restoration failed; '
                            f'primary={type(primary).__name__}: {primary}; '
                            f'restoration={type(restoration).__name__}: {restoration}')
                        failure.primary_error = primary
                        failure.restoration_error = restoration
                        raise failure from primary
                raise
            os.fsync(holding.fd)
            return {
                'path': item['path'], 'tombstone': held_name,
                'inode': item['inode'], 'original_size': item['size'],
                'sha256': item['sha256'], 'final_size': 0,
            }
        finally:
            os.close(moved_descriptor)
    finally:
        os.close(descriptor)


def apply_prune_plan(run, plan):
    validate_plan(plan)
    with pinned(run) as directory, target_parents(directory) as parents:
        # ponytail: no resume ledger; a partial operation needs manual review.
        for item in plan['files']:
            details = checked_target(
                parents, item['path'], item['sha256'], item['size'])
            if (details.st_dev != item['device']
                    or details.st_ino != item['inode']
                    or stat.S_IMODE(details.st_mode) != item['mode']):
                raise ValueError('prune target differs from immutable plan')
        if any(name.startswith(QUARANTINE_PREFIX) for name in directory.names()):
            raise ValueError('existing quarantine requires manual review')
        with quarantine(directory) as holding:
            tombstones = []
            for item in plan['files']:
                group, name = item['path'].split('/')
                tombstones.append(_quarantine_truncate(
                    parents[group], group, name, item, holding))
                parents[group].verify()
            if sorted(holding.names()) != sorted(
                    item['tombstone'] for item in tombstones):
                raise ValueError('quarantine tombstone set differs')
            os.fsync(holding.fd)
        for parent in parents.values():
            os.fsync(parent.fd)
    return {
        'quarantine': holding.name,
        'files': tombstones,
        'freed_bytes': sum(item['original_size'] for item in tombstones),
    }


def _validate_apply_result(plan, applied):
    validate_plan(plan)
    if (type(applied) is not dict
            or set(applied) != {'quarantine', 'files', 'freed_bytes'}
            or type(applied['quarantine']) is not str
            or not applied['quarantine'].startswith(QUARANTINE_PREFIX)
            or component(applied['quarantine']) != applied['quarantine']
            or type(applied['files']) is not list
            or type(applied['freed_bytes']) is not int
            or isinstance(applied['freed_bytes'], bool)):
        raise ValueError('prune apply evidence schema differs')
    expected = []
    for item in plan['files']:
        group, name = item['path'].split('/')
        expected.append({
            'path': item['path'], 'tombstone': f'{group}.{name}',
            'inode': item['inode'], 'original_size': item['size'],
            'sha256': item['sha256'], 'final_size': 0,
        })
    if (canonical_bytes(applied['files']) != canonical_bytes(expected)
            or applied['freed_bytes'] != plan['bytes']):
        raise ValueError('prune apply evidence differs from immutable plan')
    return applied


def completion_for(plan, applied):
    applied = _validate_apply_result(plan, applied)
    return {
        'kind': 'formal_pruned_evidence', 'policy': POLICY,
        'spec_key': plan['spec_key'], 'source_commit': plan['source_commit'],
        'record_sha256': plan['record_sha256'],
        'plan_sha256': hashlib.sha256(canonical_bytes(plan)).hexdigest(),
        'quarantine': applied['quarantine'], 'files': applied['files'],
        'freed_bytes': applied['freed_bytes'],
    }


def _validate_tombstones(run, value, plan):
    with pinned(run) as directory, target_parents(directory) as parents:
        for item in plan['files']:
            group, name = item['path'].split('/')
            if parents[group].present(name):
                raise ValueError('completed prune target is present')
        with PinnedDir(value['quarantine'], directory) as holding:
            if stat.S_IMODE(holding.details.st_mode) != 0o700:
                raise ValueError('completed quarantine mode differs')
            if sorted(holding.names()) != sorted(
                    item['tombstone'] for item in value['files']):
                raise ValueError('completed quarantine entries differ')
            by_path = {item['path']: item for item in plan['files']}
            for evidence in value['files']:
                item = by_path[evidence['path']]
                with opened_file(
                        holding, evidence['tombstone']) as (descriptor, details):
                    if (not _matches_plan(details, item, size=0)
                            or stat.S_IMODE(details.st_mode) != item['mode']):
                        raise ValueError('completed quarantine tombstone differs')
            os.fsync(holding.fd)


def validate_completion(value, plan, run=None):
    if (type(value) is not dict or set(value) != {
            'kind', 'policy', 'spec_key', 'source_commit', 'record_sha256',
            'plan_sha256', 'quarantine', 'files', 'freed_bytes'}):
        raise ValueError('installed completion exact schema differs')
    applied = {name: value[name]
               for name in ('quarantine', 'files', 'freed_bytes')}
    if canonical_bytes(value) != canonical_bytes(completion_for(plan, applied)):
        raise ValueError('installed completion exact schema differs')
    if run is not None:
        _validate_tombstones(run, value, plan)
    return value


def install_completion(run, plan, applied):
    applied = _validate_apply_result(plan, applied)
    with pinned(run) as directory, target_parents(directory) as parents:
        for item in plan['files']:
            group, name = item['path'].split('/')
            if parents[group].present(name):
                raise ValueError('prune completion has remaining target')
        value = completion_for(plan, applied)
        _validate_tombstones(directory, value, plan)
        write_marker(directory, COMPLETE_NAME, value)
        if validate_completion(read_json(directory, COMPLETE_NAME), plan, directory) != value:
            raise ValueError('installed completion changed')
        return value


def process_matches(arguments, run, cwd):
    run = Path(run)
    if str(run) in arguments:
        return True
    options = {}
    for index, argument in enumerate(arguments):
        for option in ('--run-dir', '--resume_run_dir', '--results_dir', '--exp_name'):
            if argument == option:
                if index + 1 >= len(arguments):
                    raise ValueError('incomplete process run option')
                options.setdefault(option, []).append(arguments[index + 1])
            elif argument.startswith(option + '='):
                options.setdefault(option, []).append(argument[len(option) + 1:])
    def candidate(value):
        path = Path(value)
        if not path.is_absolute():
            path = Path(cwd) / path
        # Inspect before lexical normalization: alias/../run must not hide a link.
        current = Path('/')
        for part in path.parts[1:]:
            current = current / part
            try:
                details = current.lstat()
            except FileNotFoundError:
                break
            if stat.S_ISLNK(details.st_mode):
                raise ValueError('process run option contains symlink')
        path = Path(os.path.abspath(path))
        return path
    for option in ('--run-dir', '--resume_run_dir'):
        if any(candidate(value) == run for value in options.get(option, [])):
            return True
    return any(candidate(parent) == run.parent and name == run.name
               for parent in options.get('--results_dir', [])
               for name in options.get('--exp_name', []))


def proc_status(text):
    return {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
            for line in text.splitlines() if ':' in line}


def unrelated_daemon(arguments, status, proc_root):
    """Only reviewed user systemd/PAM/sshd chains can omit cwd inspection."""
    uid = os.getuid()
    if (re.search(r'vf.?cl|formal|main\.py|run_three_dataset', ' '.join(arguments), re.I)
            or status['Uid'].split() != [str(uid)] * 4):
        return False
    if arguments == ['/usr/lib/systemd/systemd', '--user']:
        return status['Name'] == 'systemd' and status['PPid'] == '1'
    username = pwd.getpwuid(uid).pw_name
    pam = arguments == ['(sd-pam)'] and status['Name'] == '(sd-pam)'
    ssh = (len(arguments) == 1 and status['Name'] == 'sshd'
           and re.fullmatch(r'sshd: ' + re.escape(username)
                            + r'@(?:notty|pts/[0-9]+(?:,pts/[0-9]+)*)',
                            arguments[0]) is not None)
    if not pam and not ssh:
        return False
    if not status['PPid'].isdigit() or int(status['PPid']) <= 0:
        return False
    parent = Path(proc_root) / status['PPid']
    owner_uid = parent.stat().st_uid
    parent_status = proc_status((parent / 'status').read_text())
    parent_comm = (parent / 'comm').read_text().strip()
    parent_args = [os.fsdecode(a) for a in (parent / 'cmdline').read_bytes().split(b'\0') if a]
    if pam:
        return (owner_uid == uid and parent_comm == 'systemd'
                and unrelated_daemon(parent_args, parent_status, proc_root))
    return (owner_uid == 0 and parent_status['Uid'].split() == ['0'] * 4
            and parent_status['Name'] == parent_comm == 'sshd'
            and parent_args == [f'sshd: {username} [priv]'])


def verified_sftp_child(entry, arguments, status, proc_root):
    uid = os.getuid()
    if (arguments != ['/usr/lib/openssh/sftp-server']
            or status.get('Name') != 'sftp-server'
            or status.get('Uid', '').split() != [str(uid)] * 4
            or (entry / 'comm').read_text().removesuffix('\n') != 'sftp-server'
            or not status.get('PPid', '').isdigit()
            or int(status['PPid']) <= 0):
        return False
    try:
        # argv and comm can be forged. The kernel-provided exe link cannot;
        # if procfs denies it, keep the original fail-closed behavior.
        if os.readlink(entry / 'exe') != '/usr/lib/openssh/sftp-server':
            return False
    except OSError:
        return False
    parent = Path(proc_root) / status['PPid']
    if parent.stat().st_uid != uid:
        return False
    parent_status = proc_status((parent / 'status').read_text())
    parent_comm = (parent / 'comm').read_text().removesuffix('\n')
    parent_args = [os.fsdecode(a) for a in
                   (parent / 'cmdline').read_bytes().split(b'\0') if a]
    return (parent_comm == 'sshd'
            and unrelated_daemon(parent_args, parent_status, proc_root))


def _sftp_termination_identity(entry, proc_root):
    try:
        exe = os.readlink(entry / 'exe')
    except PermissionError:
        exe = None
    else:
        if exe != '/usr/lib/openssh/sftp-server':
            raise ValueError('SFTP executable identity differs')
    uid = os.getuid()
    status = proc_status((entry / 'status').read_text())
    command = (entry / 'cmdline').read_bytes()
    argv = [os.fsdecode(arg) for arg in command.split(b'\0') if arg]
    if (entry.stat().st_uid != uid
            or status['Pid'] != entry.name
            or status['Uid'].split() != [str(uid)] * 4
            or status['Name'] != 'sftp-server'
            or (entry / 'comm').read_text().strip() != 'sftp-server'
            or argv != ['/usr/lib/openssh/sftp-server']
            or status['State'].startswith(('Z', 'X'))
            or not status['PPid'].isdigit() or int(status['PPid']) <= 0):
        raise ValueError('SFTP candidate identity differs')
    parent = Path(proc_root) / status['PPid']
    parent_status = proc_status((parent / 'status').read_text())
    parent_command = (parent / 'cmdline').read_bytes()
    parent_args = [os.fsdecode(arg) for arg in parent_command.split(b'\0') if arg]
    if (parent.stat().st_uid != uid
            or parent_status['Pid'] != parent.name
            or parent_status['Uid'].split() != [str(uid)] * 4
            or (parent / 'comm').read_text().strip() != 'sshd'
            or not unrelated_daemon(parent_args, parent_status, proc_root)):
        raise ValueError('SFTP SSH parent identity differs')
    child_stat = (entry / 'stat').read_text()
    parent_stat = (parent / 'stat').read_text()
    child_fields = child_stat.rsplit(') ', 1)[1].split()
    parent_fields = parent_stat.rsplit(') ', 1)[1].split()
    if (int(child_stat.split(' ', 1)[0]) != int(entry.name)
            or int(child_fields[1]) != int(status['PPid'])
            or int(parent_stat.split(' ', 1)[0]) != int(parent.name)
            or int(child_fields[19]) <= 0 or int(parent_fields[19]) <= 0):
        raise ValueError('SFTP process stat identity differs')
    boot_id = (Path(proc_root) / 'sys/kernel/random/boot_id').read_text().strip()
    if not boot_id:
        raise ValueError('SFTP boot identity is empty')
    return {
        'boot_id': boot_id, 'pid': int(entry.name), 'uid': uid,
        'exe': exe, 'exe_status': 'permission_denied' if exe is None else 'verified',
        'start_time': int(child_fields[19]),
        'cmdline_sha256': hashlib.sha256(command).hexdigest(),
        'ppid': int(status['PPid']),
        'parent_start_time': int(parent_fields[19]),
        'parent_cmdline_sha256': hashlib.sha256(parent_command).hexdigest(),
    }


def user_authorized_process(entry, proc_root):
    for expected, reason in (
            (USER_AUTHORIZED_PROCESS_INSTANCE, USER_AUTHORIZED_PROCESS_REASON),
            (USER_AUTHORIZED_SECOND_PROCESS_INSTANCE, USER_AUTHORIZED_SECOND_PROCESS_REASON),
            *USER_AUTHORIZED_SCOPED_PROCESS_INSTANCES):
        if entry.name == str(expected['pid']):
            break
    else:
        return False
    try:
        # Recheck all fields: a reused PID, changed arguments/parent, reboot or
        # unreadable identity must return to the original process scan.
        for _ in range(2):
            child_stat = (entry / 'stat').read_text()
            fields = child_stat.rsplit(') ', 1)[1].split()
            status = proc_status((entry / 'status').read_text())
            if (status['Pid'] != str(expected['pid'])
                    or status['PPid'] != str(expected['ppid'])
                    or status['Uid'].split() != [str(expected['uid'])] * 4):
                return False
            parent = Path(proc_root) / str(expected['ppid'])
            parent_stat = (parent / 'stat').read_text()
            if parent_stat.split(' ', 1)[0] != str(expected['ppid']):
                return False
            observed = {
                'boot_id': (Path(proc_root) / 'sys/kernel/random/boot_id').read_text().strip(),
                'pid': int(child_stat.split(' ', 1)[0]), 'uid': entry.stat().st_uid,
                'start_time': int(fields[19]), 'ppid': int(fields[1]),
                'cmdline_sha256': hashlib.sha256((entry / 'cmdline').read_bytes()).hexdigest(),
                'parent_start_time': int(parent_stat.rsplit(') ', 1)[1].split()[19]),
                'parent_cmdline_sha256': hashlib.sha256((parent / 'cmdline').read_bytes()).hexdigest(),
            }
            if observed != expected:
                return False
    except (OSError, ValueError, IndexError, KeyError):
        return False
    return {'identity': expected, 'reason': reason}


def active_processes(run, proc_root=Path('/proc'), *, terminate_sftp=False):
    matches = []
    for entry in Path(proc_root).iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            details = entry.stat()
            if details.st_uid != os.getuid():
                continue
        except FileNotFoundError:
            continue
        try:
            exemption = user_authorized_process(entry, proc_root)
            if exemption:
                print(canonical_bytes({'kind': 'user_authorized_process_exemption',
                    **exemption, 'run': str(run)}).decode(),
                    file=sys.stderr, end='')
                continue
            arguments = [os.fsdecode(a) for a in (entry / 'cmdline').read_bytes().split(b'\0') if a]
            status = proc_status((entry / 'status').read_text())
            state = status['State']
            if not arguments and not state.startswith('Z'):
                # A fork/exec transition can expose a live PID before argv.
                # Retry only that transient; persistent opacity fails closed.
                for _ in range(3):
                    time.sleep(0.02)
                    current = entry.stat()
                    if (current.st_dev, current.st_ino) != (
                            details.st_dev, details.st_ino):
                        raise ValueError('same-UID process identity changed')
                    arguments = [os.fsdecode(a) for a in
                                 (entry / 'cmdline').read_bytes().split(b'\0') if a]
                    status = proc_status((entry / 'status').read_text())
                    state = status['State']
                    if arguments or state.startswith('Z'):
                        break
            if not arguments:
                if state.startswith('Z'):
                    continue
                raise ValueError('same-UID process has unreadable/empty argv')
            if unrelated_daemon(arguments, status, proc_root):
                if (status['Name'] == 'sshd'
                        and (entry / 'comm').read_text().removesuffix('\n') != 'sshd'):
                    raise ValueError('SSH child comm differs')
                continue
            try:
                cwd = os.readlink(entry / 'cwd')
            except PermissionError:
                if verified_sftp_child(entry, arguments, status, proc_root):
                    print(canonical_bytes({
                        'kind': 'verified_sftp_cwd_exemption',
                        'pid': int(entry.name), 'parent_pid': int(status['PPid']),
                        'run': str(run),
                    }).decode(), file=sys.stderr, end='')
                    continue
                if terminate_sftp:
                    audit = {'kind': 'sftp_termination_for_retention',
                             'pid': int(entry.name), 'identity': None,
                             'run': str(run), 'outcome': 'signal_not_sent'}
                    try:
                        candidate = _sftp_termination_identity(entry, proc_root)
                        audit['identity'] = candidate
                        def verify_candidate():
                            observed = _sftp_termination_identity(entry, proc_root)
                            if observed != candidate:
                                raise ValueError('SFTP candidate identity changed')
                            return observed
                        _term_checked_pidfd(int(entry.name), verify_candidate, audit)
                        audit['kind'] = 'sftp_terminated_for_retention'
                    except (FileNotFoundError, ProcessLookupError) as error:
                        raise ValueError('SFTP termination target disappeared') from error
                    finally:
                        print(canonical_bytes(audit).decode(), file=sys.stderr, end='')
                    return active_processes(run, proc_root, terminate_sftp=False)
                raise
            if process_matches(arguments, run, cwd):
                matches.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            try:
                current = entry.lstat()
            except (FileNotFoundError, ProcessLookupError):
                continue
            if (current.st_dev, current.st_ino, current.st_uid) != (
                    details.st_dev, details.st_ino, details.st_uid):
                raise ValueError('same-UID process identity changed')
            try:
                refreshed = proc_status((entry / 'status').read_text())
                state = refreshed['State']
            except (FileNotFoundError, ProcessLookupError) as error:
                try:
                    if _pidfd_confirms_exit(int(entry.name), proc_root):
                        continue
                except OSError as confirm_error:
                    raise ValueError('same-UID process state is unreadable') from confirm_error
                raise ValueError('same-UID process state is unreadable') from error
            except (PermissionError, IndexError, KeyError) as error:
                raise ValueError('same-UID process state is unreadable') from error
            if state.startswith(('Z', 'X', 'x')):
                continue
            raise ValueError('same-UID process state is incomplete')
        except (PermissionError, IndexError, KeyError) as error:
            raise ValueError('same-UID process state is unreadable') from error
    return matches


def bind_record(record, evidence, expected_artifacts):
    if (type(evidence) is not dict or set(evidence) != {
            'kind', 'spec_key', 'plan_sha256', 'job_spec_sha256', 'claim_sha256',
            'launch_sha256', 'command_sha256', 'artifact_sha256', 'resource'}
            or evidence['kind'] != 'formal_resource_evidence'):
        raise ValueError('resource evidence exact schema differs')
    for name in ('spec_key', 'plan_sha256', 'claim_sha256',
                 'launch_sha256', 'command_sha256', 'resource'):
        if canonical_bytes(record[name]) != canonical_bytes(evidence[name]):
            raise ValueError(f'completed/resource evidence differs: {name}')
    artifacts = evidence['artifact_sha256']
    if type(artifacts) is not dict or set(artifacts) != set(expected_artifacts):
        raise ValueError('resource artifact keys differ from live authority')
    if any(type(value) is not str or SHA256.fullmatch(value) is None
           for value in artifacts.values()):
        raise ValueError('resource artifact hash is invalid')
    mapping = {'checkpoints/formal_final.pt': 'checkpoint', 'config.json': 'config',
               'data_flow_audit.jsonl': 'data_flow', 'results.json': 'results',
               'validation/validation_manifest.json': 'validation_manifest'}
    if not set(mapping).issubset(artifacts) or 'job.log' not in artifacts:
        raise ValueError('resource artifact core mapping is incomplete')
    for name in set(artifacts) - set(mapping) - {'job.log'}:
        if re.fullmatch(r'(?:checkpoints/event_[0-9]+_CIL\.pt|formal_snapshots/event_[0-9]+_CIL\.pt|formal_access/[^/]+\.json|FORMAL_EVALUATION_[A-Z_]+\.json|FORMAL_STATE_FROZEN\.json)', name) is None:
            raise ValueError('resource artifact has no approved record mapping')
        mapping[name] = 'formal:' + name
    audited = record['artifact_sha256']
    if (type(audited) is not dict
            or {key for key in audited if key.startswith('formal:')}
            != {key for key in mapping.values() if key.startswith('formal:')}
            or any(audited.get(logical) != artifacts[name] for name, logical in mapping.items())):
        raise ValueError('completed/resource artifact projection differs')
    if record['log_sha256'] != artifacts['job.log']:
        raise ValueError('completed log evidence differs')


def load_authority(worktree, expected_head):
    worktree = Path(worktree)
    with PinnedDir(worktree):
        if subprocess.check_output(['git', '-C', str(worktree), 'rev-parse', 'HEAD'], text=True).strip() != expected_head:
            raise ValueError('worktree HEAD differs')
        if subprocess.check_output(['git', '-C', str(worktree), 'status', '--porcelain'], text=True):
            raise ValueError('worktree is dirty')
    os.chdir(worktree)
    sys.path.insert(0, str(worktree))
    sys.dont_write_bytecode = True
    import three_dataset_formal_driver as driver
    import three_dataset_formal_registry as registry
    if Path(driver.__file__).parent != worktree or driver._source_commit() != expected_head:
        raise ValueError('driver source authority differs')
    return driver, registry


def _checked_active_context(driver, root, key, owner, expected_head,
                            context=None, terminate_sftp=False):
    if context is None:
        context = driver.retention_audit_context(root, key, owner)
    if (type(context) is not dict or set(context) != {
            'plan', 'handoff', 'record', 'resource', 'run_dir', 'num_tasks'}
            or type(context['run_dir']) is not str
            or type(context['num_tasks']) is not int
            or isinstance(context['num_tasks'], bool)
            or context['num_tasks'] <= 0):
        raise ValueError('retention audit context schema differs')
    record, evidence = context['record'], context['resource']
    if (record['spec_key'] != key or evidence['spec_key'] != key
            or record['source_commit'] != expected_head
            or context['handoff']['source_commit'] != expected_head):
        raise ValueError('retention audit source/spec differs')
    bind_record(record, evidence,
                driver.resource_artifact_names(driver.spec_for_key(key)))
    run = Path(context['run_dir'])
    expected_run = root / 'runs' / component(driver.safe_spec_name(
        driver.spec_for_key(key)))
    if run != expected_run:
        raise ValueError('retention run exact path differs')
    if active_processes(run, terminate_sftp=terminate_sftp):
        raise ValueError(f'completed run still has a process: {key}')
    return context


def _active_summary(key, plan, applied, completion=None):
    return {
        'spec_key': key,
        'candidate_count': len(plan['files']),
        'reclaimable_bytes': plan['bytes'],
        'applied': applied,
        'plan_sha256': hashlib.sha256(canonical_bytes(plan)).hexdigest(),
        'completion_sha256': (None if completion is None else
            hashlib.sha256(canonical_bytes(completion)).hexdigest()),
    }


def prune_active_audit(root, worktree, expected_head, key, owner, *,
                       apply=False, terminate_sftp=False):
    if (type(expected_head) is not str or COMMIT.fullmatch(expected_head) is None
            or type(apply) is not bool or type(terminate_sftp) is not bool):
        raise ValueError('active retention invocation is invalid')
    driver, registry = load_authority(worktree, expected_head)
    profile = driver.experiment_profile()
    if profile not in (
            driver.RECOVERY_PROFILE, driver.FULL_MATRIX_PROFILE,
            driver.SINGLE_DATASET_PROFILE, driver.CONTINUATION_PROFILE,
            driver.METHOD_SHARD_PROFILE):
        raise ValueError('active retention requires a retention-enabled profile')
    if terminate_sftp and profile not in (
            driver.SINGLE_DATASET_PROFILE, driver.CONTINUATION_PROFILE,
            driver.METHOD_SHARD_PROFILE):
        raise ValueError('SFTP termination requires dataset-scoped formal profile')
    root = Path(root)
    if driver._validate_formal_root_path(root, create=False) != root:
        raise ValueError('retention root exact path differs')
    spec = driver.spec_for_key(key)
    expected_run = root / 'runs' / component(registry.safe_spec_name(spec))
    with PinnedDir(expected_run) as run:
        if not apply:
            initial = _checked_active_context(
                driver, root, key, owner, expected_head,
                terminate_sftp=terminate_sftp)
            run.verify()
            if initial['run_dir'] != str(run.path):
                raise ValueError('retention run exact path differs')
            if run.present(COMPLETE_NAME):
                raise ValueError('existing prune completion is terminal')
            if run.present(PLAN_NAME):
                raise ValueError('partial prune plan requires manual review')
            record, evidence = initial['record'], initial['resource']
            selected = build_prune_plan(
                run, evidence, record['record_sha256'], expected_head,
                initial['num_tasks'])
            return _active_summary(key, selected, False)

        with driver._retention_audit_transaction(
                root, key, owner) as current_context:
            initial = _checked_active_context(
                driver, root, key, owner, expected_head,
                context=current_context(), terminate_sftp=terminate_sftp)
            run.verify()
            if initial['run_dir'] != str(run.path):
                raise ValueError('retention run exact path differs')
            if run.present(COMPLETE_NAME):
                raise ValueError('existing prune completion is terminal')
            if run.present(PLAN_NAME):
                raise ValueError('partial prune plan requires manual review')
            record, evidence = initial['record'], initial['resource']
            selected = build_prune_plan(
                run, evidence, record['record_sha256'], expected_head,
                initial['num_tasks'])

            terminal = False
            try:
                before_plan = _checked_active_context(
                    driver, root, key, owner, expected_head,
                    context=current_context(), terminate_sftp=terminate_sftp)
                run.verify()
                if canonical_bytes(before_plan) != canonical_bytes(initial):
                    raise ValueError('retention audit context changed before plan')
                terminal = True
                install_plan(run, selected)

                installed = load_installed_plan(run)
                validate_plan_against_evidence(
                    run, installed, evidence, record['record_sha256'],
                    expected_head, initial['num_tasks'])
                before_unlink = _checked_active_context(
                    driver, root, key, owner, expected_head,
                    context=current_context(), terminate_sftp=terminate_sftp)
                run.verify()
                if (canonical_bytes(before_unlink) != canonical_bytes(initial)
                        or canonical_bytes(installed) != canonical_bytes(selected)):
                    raise ValueError('retention authority changed before unlink')
                applied = apply_prune_plan(run, installed)

                validate_plan_against_evidence(
                    run, installed, evidence, record['record_sha256'],
                    expected_head, initial['num_tasks'], completed=True)
                before_completion = _checked_active_context(
                    driver, root, key, owner, expected_head,
                    context=current_context(), terminate_sftp=terminate_sftp)
                run.verify()
                if canonical_bytes(before_completion) != canonical_bytes(initial):
                    raise ValueError('retention authority changed before completion')
                completion = install_completion(run, installed, applied)
            except Exception as error:
                if terminal:
                    raise RuntimeError(
                        'retention apply is terminal and requires manual review') from error
                raise
        return _active_summary(key, selected, True, completion)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--worktree', required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--spec-key', required=True)
    parser.add_argument('--owner-json', required=True)
    parser.add_argument('--terminate-blocking-sftp', action='store_true')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    if COMMIT.fullmatch(args.expected_head) is None:
        raise SystemExit('expected HEAD is invalid')
    driver, _ = load_authority(args.worktree, args.expected_head)
    if (args.terminate_blocking_sftp
            and driver.experiment_profile() not in (
                driver.SINGLE_DATASET_PROFILE, driver.CONTINUATION_PROFILE,
                driver.METHOD_SHARD_PROFILE)):
        raise SystemExit('SFTP termination requires dataset-scoped formal profile')
    owner = driver._inline_json(args.owner_json, 'retention owner JSON')
    summary = prune_active_audit(
        args.root, args.worktree, args.expected_head, args.spec_key,
        owner, apply=args.apply,
        terminate_sftp=args.terminate_blocking_sftp)
    print(canonical_bytes(summary).decode(), end='')
    return 0


if __name__ == '__main__':
    main()
