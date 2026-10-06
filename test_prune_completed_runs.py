import hashlib
import inspect
import json
import os
import pwd
import io
import signal
import stat
import subprocess
import sys
import threading
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import prune_completed_runs as prune
import three_dataset_formal_driver as driver
import three_dataset_formal_registry as registry
from test_three_dataset_formal_driver import _completed_record, _write_resource_checkpoint


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PruneCompletedRunsTests(unittest.TestCase):
    def test_pidfd_failure_outcomes_and_descriptor_cleanup(self):
        expected = {'pid': 123, 'exe': None, 'exe_status': 'permission_denied'}
        for case in ('missing_open', 'missing_send', 'open', 'identity',
                     'send', 'timeout', 'poll', 'success'):
            with self.subTest(case=case):
                opened = Mock(return_value=-1 if case == 'open' else 42)
                send = Mock(return_value=-1 if case == 'send' else 0)
                libc = SimpleNamespace(pidfd_open=opened, pidfd_send_signal=send)
                if case.startswith('missing_'):
                    delattr(libc, 'pidfd_' + ('open' if case == 'missing_open' else 'send_signal'))
                identity = Mock(return_value=expected)
                if case == 'identity':
                    identity.side_effect = [expected, ValueError('changed')]
                poller = Mock()
                poller.poll.return_value = [] if case == 'timeout' else [(42, prune.select.POLLIN)]
                if case == 'poll':
                    poller.poll.side_effect = OSError('poll failed')
                audit = {}
                with patch.object(prune.ctypes, 'CDLL', return_value=libc), \
                        patch.object(prune.select, 'poll', return_value=poller), \
                        patch.object(prune.os, 'close') as close:
                    if case == 'success':
                        self.assertEqual(expected, prune._term_checked_pidfd(123, identity, audit))
                    else:
                        with self.assertRaises((OSError, ValueError)):
                            prune._term_checked_pidfd(123, identity, audit)
                if case in ('missing_open', 'missing_send', 'open'):
                    close.assert_not_called()
                    identity.assert_not_called()
                else:
                    close.assert_called_once_with(42)
                sent = case in ('timeout', 'poll', 'success')
                self.assertEqual('term_sent_exit_observed' if case == 'success' else
                                 'term_sent_exit_unconfirmed' if sent else 'signal_not_sent',
                                 audit['outcome'])
                if case in ('send', 'timeout', 'poll', 'success'):
                    send.assert_called_once_with(42, signal.SIGTERM, None, 0)
                    self.assertEqual(expected, audit['identity'])
                else:
                    send.assert_not_called()

    @unittest.skipUnless(sys.platform.startswith('linux'), 'pidfd requires Linux')
    def test_checked_pidfd_term(self):
        self.assertTrue(callable(getattr(prune, '_term_checked_pidfd', None)))

        def child_process():
            child = subprocess.Popen(['sleep', '60'])
            def cleanup():
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=5)
            self.addCleanup(cleanup)
            return child

        child = child_process()
        expected = {'pid': child.pid, 'start': 1}
        seen = []
        def identity():
            seen.append(expected.copy())
            return seen[-1]
        self.assertEqual(expected, prune._term_checked_pidfd(child.pid, identity))
        self.assertEqual([expected, expected], seen)
        self.assertEqual(-signal.SIGTERM, child.wait(timeout=5))

        unchanged_child = child_process()
        calls = []
        def changed_identity():
            calls.append({'pid': unchanged_child.pid, 'start': len(calls) + 1})
            return calls[-1]
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            prune._term_checked_pidfd(unchanged_child.pid, changed_identity)
        self.assertEqual(2, len(calls))
        self.assertIsNone(unchanged_child.poll())

    def test_new_safety_contract_is_exposed(self):
        for name in ('PinnedDir', 'bind_record', 'validate_completion',
                     'process_matches'):
            self.assertTrue(callable(getattr(prune, name, None)), name)

    def test_process_formal_argv_and_relative_resume_are_recognized(self):
        run, _ = self.fixture()
        self.assertTrue(callable(getattr(prune, 'process_matches', None)))
        for argv in ([str(run)], ['--run-dir=' + str(run)],
                     ['--run-dir', str(run)], ['--resume_run_dir', 'run'],
                     ['--results_dir', str(run.parent), '--exp_name', 'run']):
            self.assertTrue(prune.process_matches(argv, run, run.parent), argv)

    def test_same_uid_unreadable_proc_fails_closed(self):
        run, _ = self.fixture()
        proc = run.parent / 'proc'
        (proc / '999999999').mkdir(parents=True)
        real_read = Path.read_bytes
        def unreadable(path):
            if path.name == 'cmdline':
                raise PermissionError('injected same UID unreadable proc')
            return real_read(path)
        with patch.object(Path, 'read_bytes', unreadable):
            with self.assertRaises((ValueError, PermissionError)):
                prune.active_processes(run, proc)

    def test_transient_empty_argv_is_retried_without_hiding_run_process(self):
        run, _ = self.fixture()
        proc = run.parent / 'proc'
        entry = proc / '999999998'
        entry.mkdir(parents=True)
        (entry / 'status').write_text(
            f'Name:\tpython\nState:\tR (running)\nPPid:\t1\nUid:\t'
            + '\t'.join([str(os.getuid())] * 4) + '\n')
        command = b'python\0--run-dir\0' + os.fsencode(run) + b'\0'
        (entry / 'cmdline').write_bytes(command)
        original = Path.read_bytes
        reads = 0

        def transient(path):
            nonlocal reads
            if path == entry / 'cmdline':
                reads += 1
                if reads == 1:
                    return b''
            return original(path)

        with patch.object(Path, 'read_bytes', transient), \
                patch.object(prune.os, 'readlink', return_value=str(run.parent)):
            self.assertEqual([int(entry.name)], prune.active_processes(run, proc))
        self.assertGreaterEqual(reads, 2)
        (entry / 'cmdline').write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'empty argv'):
            prune.active_processes(run, proc)

    def process_missing_cwd_fixture(self):
        run, _ = self.fixture()
        proc = run.parent / 'proc'
        entry = proc / '999999997'
        entry.mkdir(parents=True)
        (entry / 'cmdline').write_bytes(
            b'python\0--run-dir\0' + os.fsencode(run) + b'\0')
        running = (f'Name:\tpython\nState:\tR (running)\nPPid:\t1\nUid:\t'
                   + '\t'.join([str(os.getuid())] * 4) + '\n')
        zombie = running.replace('R (running)', 'Z (zombie)')
        (entry / 'status').write_text(running)
        return run, proc, entry, running, zombie

    def test_missing_cwd_is_ignored_after_verified_terminal_transition(self):
        run, proc, entry, running, zombie = self.process_missing_cwd_fixture()
        original = Path.read_text
        reads = 0

        def transitioning(path, *args, **kwargs):
            nonlocal reads
            if path == entry / 'status':
                reads += 1
                return running if reads == 1 else zombie
            return original(path, *args, **kwargs)

        with patch.object(Path, 'read_text', transitioning), \
                patch.object(prune.os, 'readlink',
                             side_effect=FileNotFoundError('exiting process')):
            self.assertEqual([], prune.active_processes(run, proc))
        self.assertEqual(2, reads)

    def test_missing_cwd_for_refreshed_live_process_stays_fail_closed(self):
        run, proc, _, _, _ = self.process_missing_cwd_fixture()
        with patch.object(prune.os, 'readlink',
                          side_effect=FileNotFoundError('missing cwd')):
            with self.assertRaisesRegex(ValueError, 'state is incomplete'):
                prune.active_processes(run, proc)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'pidfd requires Linux')
    def test_read_only_pidfd_distinguishes_owned_live_and_exited_pid(self):
        child = subprocess.Popen(
            [sys.executable, '-c', 'import sys; sys.stdin.buffer.read(1)'],
            stdin=subprocess.PIPE)
        try:
            self.assertFalse(prune._pidfd_confirms_exit(child.pid, Path('/proc')))
            child.stdin.write(b'x')
            child.stdin.close()
            os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
            self.assertTrue(prune._pidfd_confirms_exit(child.pid, Path('/proc')))
            self.assertEqual(0, child.wait(timeout=5))
            self.assertTrue(prune._pidfd_confirms_exit(child.pid, Path('/proc')))
        finally:
            if not child.stdin.closed:
                child.stdin.close()
            child.wait(timeout=5)

    def test_read_only_pidfd_never_probes_a_synthetic_proc_tree(self):
        with patch.object(prune.ctypes, 'CDLL', side_effect=AssertionError('kernel probe')):
            self.assertFalse(prune._pidfd_confirms_exit(999999997, Path('/tmp/fake-proc')))

    def test_read_only_pidfd_closes_descriptor_if_poll_fails(self):
        libc = SimpleNamespace(pidfd_open=Mock(return_value=42))
        poller = Mock()
        poller.poll.side_effect = OSError('poll failed')
        with patch.object(prune.ctypes, 'CDLL', return_value=libc), \
                patch.object(prune.select, 'poll', return_value=poller), \
                patch.object(prune.os, 'close') as close:
            with self.assertRaisesRegex(OSError, 'poll failed'):
                prune._pidfd_confirms_exit(123, Path('/proc'))
        close.assert_called_once_with(42)

    def test_missing_cwd_then_status_esrch_skips_only_confirmed_exit(self):
        run, proc, entry, running, _ = self.process_missing_cwd_fixture()
        original = Path.read_text
        reads = 0

        def exiting(path, *args, **kwargs):
            nonlocal reads
            if path == entry / 'status':
                reads += 1
                if reads == 1:
                    return running
                raise ProcessLookupError('process exited')
            return original(path, *args, **kwargs)

        with patch.object(Path, 'read_text', exiting), \
                patch.object(prune.os, 'readlink',
                             side_effect=FileNotFoundError('cwd gone')), \
                patch.object(prune, '_pidfd_confirms_exit', return_value=True) as exited:
            self.assertEqual([], prune.active_processes(run, proc))
        self.assertEqual(2, reads)
        exited.assert_called_once_with(int(entry.name), proc)

    def test_missing_cwd_then_status_esrch_live_pid_stays_fail_closed(self):
        run, proc, entry, running, _ = self.process_missing_cwd_fixture()
        original = Path.read_text
        reads = 0

        def unreadable(path, *args, **kwargs):
            nonlocal reads
            if path == entry / 'status':
                reads += 1
                if reads == 1:
                    return running
                raise ProcessLookupError('still present but unreadable')
            return original(path, *args, **kwargs)

        with patch.object(Path, 'read_text', unreadable), \
                patch.object(prune.os, 'readlink',
                             side_effect=FileNotFoundError('cwd gone')), \
                patch.object(prune, '_pidfd_confirms_exit', return_value=False):
            with self.assertRaisesRegex(ValueError, 'state is unreadable'):
                prune.active_processes(run, proc)

    def test_missing_cwd_then_status_esrch_pidfd_error_stays_fail_closed(self):
        run, proc, entry, running, _ = self.process_missing_cwd_fixture()
        original = Path.read_text
        reads = 0

        def unreadable(path, *args, **kwargs):
            nonlocal reads
            if path == entry / 'status':
                reads += 1
                if reads == 1:
                    return running
                raise ProcessLookupError('state vanished')
            return original(path, *args, **kwargs)

        with patch.object(Path, 'read_text', unreadable), \
                patch.object(prune.os, 'readlink',
                             side_effect=FileNotFoundError('cwd gone')), \
                patch.object(prune, '_pidfd_confirms_exit',
                             side_effect=OSError('pidfd unavailable')):
            with self.assertRaisesRegex(ValueError, 'state is unreadable'):
                prune.active_processes(run, proc)

    def test_missing_cwd_then_status_esrch_changed_identity_stays_fail_closed(self):
        run, proc, entry, running, _ = self.process_missing_cwd_fixture()
        original = Path.lstat

        def changed(path):
            result = original(path)
            if path == entry:
                return SimpleNamespace(st_dev=result.st_dev, st_ino=result.st_ino + 1,
                                       st_uid=result.st_uid)
            return result

        with patch.object(Path, 'lstat', changed), \
                patch.object(prune.os, 'readlink',
                             side_effect=FileNotFoundError('cwd gone')), \
                patch.object(prune, '_pidfd_confirms_exit', return_value=True) as exited:
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                prune.active_processes(run, proc)
        exited.assert_not_called()

    def test_missing_cwd_then_status_permission_error_stays_fail_closed(self):
        run, proc, entry, running, _ = self.process_missing_cwd_fixture()
        original = Path.read_text
        reads = 0

        def unreadable(path, *args, **kwargs):
            nonlocal reads
            if path == entry / 'status':
                reads += 1
                if reads == 1:
                    return running
                raise PermissionError('status denied')
            return original(path, *args, **kwargs)

        with patch.object(Path, 'read_text', unreadable), \
                patch.object(prune.os, 'readlink',
                             side_effect=FileNotFoundError('cwd gone')), \
                patch.object(prune, '_pidfd_confirms_exit', return_value=True) as exited:
            with self.assertRaisesRegex(ValueError, 'state is unreadable'):
                prune.active_processes(run, proc)
        exited.assert_not_called()

    def authorized_process_fixture(self, pid=1499350):
        run, _ = self.fixture()
        proc = run.parent / 'proc'
        ppid, start, parent_start = {
            1499350: (328962, 51143674, 42107480),
            1560782: (1560780, 51571438, 51571420),
        }[pid]
        entry, parent = proc / str(pid), proc / str(ppid)
        entry.mkdir(parents=True)
        parent.mkdir()
        boot = proc / 'sys/kernel/random/boot_id'
        boot.parent.mkdir(parents=True)
        boot.write_text('45f74e91-2a45-443e-93e3-6a6f1435b74f\n')
        child_command = b'sftp-server\0generated-fixture\0'
        parent_command = f'sshd\0generated-parent-{ppid}\0'.encode()
        (entry / 'cmdline').write_bytes(child_command)
        (parent / 'cmdline').write_bytes(parent_command)
        identity = {'boot_id': boot.read_text().strip(), 'pid': pid, 'uid': os.getuid(),
                    'start_time': start, 'cmdline_sha256': hashlib.sha256(child_command).hexdigest(),
                    'ppid': ppid, 'parent_start_time': parent_start,
                    'parent_cmdline_sha256': hashlib.sha256(parent_command).hexdigest()}
        self.write_proc_stat(entry, identity['pid'], identity['ppid'], identity['start_time'])
        self.write_proc_stat(parent, identity['ppid'], 1, identity['parent_start_time'])
        (entry / 'status').write_text(
            f'Name:\tsftp-server\nState:\tS\nPid:\t{pid}\nPPid:\t{ppid}\nUid:\t'
            + '\t'.join([str(os.getuid())] * 4) + '\n')
        original_stat = Path.stat
        def skip_parent(path, *args, **kwargs):
            if path == parent:
                return SimpleNamespace(st_uid=os.getuid() + 1)
            return original_stat(path, *args, **kwargs)
        context = ExitStack()
        self.addCleanup(context.close)
        context.enter_context(patch.object(Path, 'stat', skip_parent))
        name = ('USER_AUTHORIZED_PROCESS_INSTANCE' if pid == 1499350
                else 'USER_AUTHORIZED_SECOND_PROCESS_INSTANCE')
        context.enter_context(patch.object(prune, name, identity, create=True))
        return run, proc, entry, parent, identity

    @staticmethod
    def write_proc_stat(entry, pid, ppid, start):
        (entry / 'stat').write_text(
            f'{pid} (generated process) ' + ' '.join(['S', str(ppid), *(['0'] * 17), str(start)]) + '\n')

    def test_exact_user_authorized_instance_is_audited_without_cwd_for_each_profile(self):
        for pid in (1499350, 1560782):
            with self.subTest(pid=pid):
                self.check_authorized_instance(pid)

    def test_scoped_sftp_instances_use_existing_exact_identity_gate(self):
        scoped = getattr(prune, 'USER_AUTHORIZED_SCOPED_PROCESS_INSTANCES', ())
        self.assertEqual({3205670, 176899},
                         {identity['pid'] for identity, _ in scoped})
        run, proc, _, _, identity = self.authorized_process_fixture()
        with patch.object(prune, 'USER_AUTHORIZED_PROCESS_INSTANCE', {'pid': -1}), \
                patch.object(prune, 'USER_AUTHORIZED_SECOND_PROCESS_INSTANCE', {'pid': -1}), \
                patch.object(prune, 'USER_AUTHORIZED_SCOPED_PROCESS_INSTANCES',
                             ((identity, 'user accepted precise SFTP risk for VF-CL'),)), \
                patch.object(prune.os, 'readlink',
                             side_effect=PermissionError('opaque cwd')), \
                redirect_stderr(io.StringIO()) as output:
            self.assertEqual([], prune.active_processes(run, proc))
            self.assertEqual(identity, json.loads(output.getvalue())['identity'])
        for field in identity:
            wrong = (identity[field] + 1 if type(identity[field]) is int else
                     '0' * 64 if field.endswith('_sha256') else
                     '00000000-0000-4000-8000-000000000000')
            with self.subTest(field=field), \
                    patch.object(prune, 'USER_AUTHORIZED_PROCESS_INSTANCE', {'pid': -1}), \
                    patch.object(prune, 'USER_AUTHORIZED_SECOND_PROCESS_INSTANCE', {'pid': -1}), \
                    patch.object(prune, 'USER_AUTHORIZED_SCOPED_PROCESS_INSTANCES',
                                 (({**identity, field: wrong}, 'user accepted risk'),)), \
                    patch.object(prune.os, 'readlink',
                                 side_effect=PermissionError('opaque cwd')):
                with self.assertRaisesRegex(ValueError, 'unreadable'):
                    prune.active_processes(run, proc)

    def check_authorized_instance(self, pid):
        run, proc, _, _, identity = self.authorized_process_fixture(pid)
        for profile in (None, 'seed42-adaptive-recovery', 'full-public-matrix'):
            with self.subTest(profile=profile), patch.dict(os.environ), \
                    patch.object(prune.os, 'readlink', side_effect=PermissionError('cwd must not be read')) as readlink:
                if profile is None:
                    os.environ.pop('VFCL_EXPERIMENT_PROFILE', None)
                else:
                    os.environ['VFCL_EXPERIMENT_PROFILE'] = profile
                output = io.StringIO()
                try:
                    with redirect_stderr(output):
                        result = prune.active_processes(run, proc)
                except ValueError as error:
                    self.fail('exact user-authorized process was rejected: ' + str(error))
                self.assertEqual([], result)
                readlink.assert_not_called()
                event = json.loads(output.getvalue())
                self.assertEqual('user_authorized_process_exemption', event['kind'])
                self.assertEqual(identity, event['identity'])
                self.assertEqual(str(run), event['run'])
                self.assertIn('user', event['reason'].lower())
                self.assertIn('VF-CL', event['reason'])
                reason = (prune.USER_AUTHORIZED_PROCESS_REASON if pid == 1499350
                          else prune.USER_AUTHORIZED_SECOND_PROCESS_REASON)
                self.assertEqual(reason, event['reason'])

    def test_user_authorized_instance_pin_changes_missing_fields_and_reuse_fail_closed(self):
        for pid in (1499350, 1560782):
            with self.subTest(pid=pid):
                self.check_authorized_instance_rejections(pid)

    def check_authorized_instance_rejections(self, pid):
        run, proc, entry, parent, identity = self.authorized_process_fixture(pid)
        name = ('USER_AUTHORIZED_PROCESS_INSTANCE' if pid == 1499350
                else 'USER_AUTHORIZED_SECOND_PROCESS_INSTANCE')
        paths = (entry / 'stat', entry / 'status', entry / 'cmdline', parent / 'stat',
                 parent / 'cmdline', proc / 'sys/kernel/random/boot_id')
        original = {path: path.read_bytes() for path in paths}
        for profile in ('formal', 'seed42-adaptive-recovery', 'full-public-matrix'):
            for field in identity:
                with self.subTest(profile=profile, field=field), \
                        patch.dict(os.environ, VFCL_EXPERIMENT_PROFILE=profile):
                    wrong = (identity[field] + 1 if type(identity[field]) is int else
                             '0' * 64 if field.endswith('_sha256') else '00000000-0000-4000-8000-000000000000')
                    changed = {**identity, field: wrong}
                    with patch.object(prune, name, changed), \
                            patch.object(prune.os, 'readlink', side_effect=PermissionError('unreadable cwd')), \
                            redirect_stderr(io.StringIO()) as output:
                        with self.assertRaises(ValueError):
                            prune.active_processes(run, proc)
                        self.assertEqual('', output.getvalue())
        for path in paths:
            with self.subTest(missing=path.name):
                path.unlink()
                with patch.object(prune.os, 'readlink', side_effect=PermissionError('unreadable cwd')), \
                        self.assertRaises(ValueError):
                    prune.active_processes(run, proc)
                path.write_bytes(original[path])
        for path in paths:
            method = 'read_bytes' if path.name == 'cmdline' else 'read_text'
            original_read = getattr(Path, method)
            def unreadable(target, *args, **kwargs):
                if target == path:
                    raise PermissionError('unreadable identity field')
                return original_read(target, *args, **kwargs)
            with self.subTest(unreadable=str(path)), patch.object(Path, method, unreadable), \
                    patch.object(prune.os, 'readlink', side_effect=PermissionError('unreadable cwd')), \
                    self.assertRaises(ValueError):
                prune.active_processes(run, proc)
        self.write_proc_stat(parent, identity['ppid'], 1, identity['parent_start_time'] + 1)
        with patch.object(prune.os, 'readlink', side_effect=PermissionError('reused parent session')), \
                self.assertRaises(ValueError):
            prune.active_processes(run, proc)
        (parent / 'stat').write_bytes(original[parent / 'stat'])
        self.write_proc_stat(entry, identity['pid'], identity['ppid'], identity['start_time'] + 1)
        with patch.object(prune.os, 'readlink', side_effect=PermissionError('reused PID')), \
                self.assertRaises(ValueError):
            prune.active_processes(run, proc)
        (entry / 'stat').write_bytes(original[entry / 'stat'])
        entry.rename(proc / '1499351')
        with patch.object(prune.os, 'readlink', side_effect=PermissionError('another SFTP PID')), \
                self.assertRaises(ValueError):
            prune.active_processes(run, proc)

    def test_user_authorized_identity_is_rechecked_before_exemption(self):
        for pid in (1499350, 1560782):
            with self.subTest(pid=pid):
                self.check_authorized_identity_recheck(pid)

    def check_authorized_identity_recheck(self, pid):
        run, proc, entry, _, _ = self.authorized_process_fixture(pid)
        original_read = Path.read_bytes
        reads = 0
        def changed_command(path):
            nonlocal reads
            value = original_read(path)
            if path == entry / 'cmdline':
                reads += 1
                if reads > 1:
                    return b'changed executable\0'
            return value
        with patch.object(Path, 'read_bytes', changed_command), \
                patch.object(prune.os, 'readlink', side_effect=PermissionError('changed identity')), \
                redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(ValueError):
                prune.active_processes(run, proc)
            self.assertEqual('', output.getvalue())

    def test_unreadable_cwd_only_known_unrelated_systemd_is_allowed(self):
        run, _ = self.fixture()
        proc = run.parent / 'proc'
        entry = proc / '999999999'
        entry.mkdir(parents=True)
        (entry / 'status').write_text(f'Name:\tsystemd\nState:\tS (sleeping)\nPPid:\t1\nUid:\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\n')
        for argv, allowed in ((['/usr/lib/systemd/systemd', '--user'], True),
                              (['python', 'main.py', '--run-dir', str(run)], False),
                              (['bash', 'relative-script.sh'], False)):
            with self.subTest(argv=argv):
                (entry / 'cmdline').write_bytes(b'\0'.join(os.fsencode(a) for a in argv) + b'\0')
                with patch.object(prune.os, 'readlink', side_effect=PermissionError('injected cwd')):
                    if allowed:
                        self.assertEqual([], prune.active_processes(run, proc))
                    else:
                        with self.assertRaises(ValueError):
                            prune.active_processes(run, proc)

    def test_daemon_parent_chain_positive_and_spoof_rejection(self):
        self.assertTrue(callable(getattr(prune, 'unrelated_daemon', None)))
        run, _ = self.fixture()
        proc = run.parent / 'proc'
        parent = proc / '888888888'
        parent.mkdir(parents=True)
        uid = os.getuid()
        username = pwd.getpwuid(uid).pw_name
        def status(name, user, ppid):
            return dict(Name=name, State='S (sleeping)', PPid=str(ppid),
                        Uid='\t'.join([str(user)] * 4))
        def write_parent(name, user, argv):
            value = status(name, user, 1)
            (parent / 'status').write_text('\n'.join(k + ':\t' + v for k, v in value.items()))
            (parent / 'comm').write_text(name + '\n')
            (parent / 'cmdline').write_bytes(b'\0'.join(os.fsencode(a) for a in argv) + b'\0')
        write_parent('systemd', uid, ['/usr/lib/systemd/systemd', '--user'])
        child = status('(sd-pam)', uid, int(parent.name))
        self.assertTrue(prune.unrelated_daemon(['(sd-pam)'], child, proc))
        write_parent('systemd', uid + 1, ['/usr/lib/systemd/systemd', '--user'])
        self.assertFalse(prune.unrelated_daemon(['(sd-pam)'], child, proc))
        write_parent('sshd', 0, [f'sshd: {username} [priv]'])
        child = status('sshd', uid, int(parent.name))
        original_stat = Path.stat
        def root_stat(path, *args, **kwargs):
            if path == parent:
                return SimpleNamespace(st_uid=0)
            return original_stat(path, *args, **kwargs)
        with patch.object(Path, 'stat', root_stat):
            self.assertTrue(prune.unrelated_daemon([f'sshd: {username}@notty'], child, proc))
            self.assertFalse(prune.unrelated_daemon(['sshd: wrong-user@notty'], child, proc))
            write_parent('bash', 0, ['bash', 'relative.sh'])
            self.assertFalse(prune.unrelated_daemon([f'sshd: {username}@notty'], child, proc))
        write_parent('sshd', 0, [f'sshd: {username} [priv]'])
        self.assertFalse(prune.unrelated_daemon([f'sshd: {username}@notty'], child, proc))

    def test_interactive_sshd_requires_exact_terminal_and_verified_parent(self):
        run, _ = self.fixture()
        proc = run.parent / 'proc'
        entry, parent = proc / '999999999', proc / '888888888'
        entry.mkdir(parents=True)
        parent.mkdir()
        uid = os.getuid()
        username = pwd.getpwuid(uid).pw_name
        original_stat, original_read = Path.stat, Path.read_bytes
        def root_stat(path, *args, **kwargs):
            if path == parent:
                return SimpleNamespace(st_uid=parent_uid)
            return original_stat(path, *args, **kwargs)
        def readable_cmdline(path):
            if path == entry / 'cmdline' and scenario == 'unreadable_cmdline':
                raise PermissionError('injected child cmdline')
            return original_read(path)
        scenarios = (
            ('notty', 'notty', True), ('single_pts', 'pts/1', True),
            ('multi_pts', 'pts/1,pts/2', True),
            ('zero_pts', 'pts/0', True), ('wrong_username', 'pts/1', False),
            ('space', 'pts/1 ', False), ('suffix', 'pts/1extra', False),
            ('relative', 'pts/1,../main.py', False), ('negative', 'pts/-1', False),
            ('nonnumeric', 'pts/a', False), ('empty_pts', 'pts/', False),
            ('empty_entry', 'pts/1,', False), ('double_comma', 'pts/1,,pts/2', False),
            ('missing_prefix', 'pts/1,2', False), ('nonascii_digits', 'pts/١', False),
            ('extra_argument', 'pts/1', False), ('forbidden', 'pts/1,formal', False),
            ('wrong_child_name', 'pts/1', False), ('wrong_child_comm', 'notty', False),
            ('spaced_child_comm', 'notty', False), ('wrong_parent_uid', 'pts/1', False),
            ('wrong_parent_status_uid', 'pts/1', False), ('wrong_parent_name', 'pts/1', False),
            ('wrong_parent_comm', 'pts/1', False), ('wrong_ppid', 'pts/1', False),
            ('wrong_parent_cmdline', 'pts/1', False), ('unreadable_cmdline', 'pts/1', False),
        )
        for scenario, terminal, allowed in scenarios:
            with self.subTest(scenario=scenario):
                parent_uid = uid if scenario == 'wrong_parent_uid' else 0
                child_name = 'bash' if scenario == 'wrong_child_name' else 'sshd'
                child_comm = ('bash' if scenario == 'wrong_child_comm' else
                              ' sshd ' if scenario == 'spaced_child_comm' else 'sshd')
                parent_name = 'bash' if scenario == 'wrong_parent_name' else 'sshd'
                parent_comm = 'bash' if scenario == 'wrong_parent_comm' else 'sshd'
                parent_status_uid = uid if scenario == 'wrong_parent_status_uid' else 0
                ppid = '0' if scenario == 'wrong_ppid' else parent.name
                (entry / 'status').write_text(f'Name:\t{child_name}\nState:\tS\nPPid:\t{ppid}\nUid:\t' + '\t'.join([str(uid)] * 4) + '\n')
                (entry / 'comm').write_text(child_comm + '\n')
                user = 'wrong-user' if scenario == 'wrong_username' else username
                child_args = [f'sshd: {user}@{terminal}']
                if scenario == 'extra_argument':
                    child_args.append('relative-script.sh')
                (entry / 'cmdline').write_bytes(b'\0'.join(os.fsencode(a) for a in child_args) + b'\0')
                (parent / 'status').write_text(f'Name:\t{parent_name}\nState:\tS\nPPid:\t1\nUid:\t' + '\t'.join([str(parent_status_uid)] * 4) + '\n')
                (parent / 'comm').write_text(parent_comm + '\n')
                parent_arg = 'bash relative-script.sh' if scenario == 'wrong_parent_cmdline' else f'sshd: {username} [priv]'
                (parent / 'cmdline').write_bytes(os.fsencode(parent_arg) + b'\0')
                with patch.object(Path, 'stat', root_stat), \
                     patch.object(Path, 'read_bytes', readable_cmdline), \
                     patch.object(prune.os, 'readlink', side_effect=PermissionError('injected cwd')):
                    if allowed:
                        try:
                            result = prune.active_processes(run, proc)
                        except ValueError as error:
                            self.fail(f'valid verified SSH terminal rejected: {error}')
                        self.assertEqual([], result)
                    else:
                        with self.assertRaises(ValueError):
                            prune.active_processes(run, proc)

    def test_unreadable_verified_sftp_child_is_exempt_but_spoofs_fail_closed(self):
        self.assertTrue(callable(getattr(prune, '_sftp_termination_identity', None)))
        run, _ = self.fixture()
        proc = run.parent / 'proc'
        child, parent, privileged = (proc / name for name in
                                     ('999999997', '888888887', '777777777'))
        for entry in (child, parent, privileged):
            entry.mkdir(parents=True)
        uid = os.getuid()
        username = pwd.getpwuid(uid).pw_name

        def write_process(entry, name, ppid, owner, args):
            (entry / 'status').write_text(
                f'Name:\t{name}\nState:\tS\nPid:\t{entry.name}\nPPid:\t{ppid}\nUid:\t'
                + '\t'.join([str(owner)] * 4) + '\n')
            (entry / 'comm').write_text(name + '\n')
            (entry / 'cmdline').write_bytes(
                b'\0'.join(os.fsencode(arg) for arg in args) + b'\0')
            self.write_proc_stat(entry, int(entry.name), int(ppid), 100 + int(entry.name))

        write_process(privileged, 'sshd', 1, 0, [f'sshd: {username} [priv]'])
        write_process(parent, 'sshd', privileged.name, uid,
                      [f'sshd: {username}@notty'])
        write_process(child, 'sftp-server', parent.name, uid,
                      ['/usr/lib/openssh/sftp-server'])
        boot = proc / 'sys/kernel/random/boot_id'
        boot.parent.mkdir(parents=True)
        boot.write_text('synthetic-boot-id\n')
        original_stat = Path.stat

        def root_stat(path, *args, **kwargs):
            if path == privileged:
                return SimpleNamespace(st_uid=0)
            return original_stat(path, *args, **kwargs)

        with patch.object(Path, 'stat', root_stat), \
                patch.object(prune.os, 'readlink',
                             side_effect=PermissionError('SFTP identity denied')):
            with self.assertRaisesRegex(ValueError, 'unreadable'):
                prune.active_processes(run, proc)

        for exe in ('/usr/bin/python3', PermissionError('denied'),
                    FileNotFoundError('gone'), OSError('I/O error')):
            with self.subTest(exe=exe), patch.object(Path, 'stat', root_stat), \
                    patch.object(prune.os, 'readlink', side_effect=
                                 exe if isinstance(exe, Exception) else None,
                                 return_value=exe):
                if isinstance(exe, PermissionError):
                    observed = prune._sftp_termination_identity(child, proc)
                    self.assertIsNone(observed['exe'])
                    self.assertEqual('permission_denied', observed['exe_status'])
                else:
                    with self.assertRaises((ValueError, OSError)):
                        prune._sftp_termination_identity(child, proc)

        for stage in (0, 1, 2):
            exe_reads = []
            def changing_exe(path):
                if path.name == 'cwd':
                    raise PermissionError('cwd denied')
                exe_reads.append(path)
                if len(exe_reads) > stage:
                    return '/usr/bin/python3'
                raise PermissionError('exe denied')
            libc = SimpleNamespace(pidfd_open=Mock(return_value=42),
                                   pidfd_send_signal=Mock(return_value=0))
            with self.subTest(stage=stage), patch.object(Path, 'stat', root_stat), \
                    patch.object(prune, 'verified_sftp_child', return_value=False), \
                    patch.object(prune.os, 'readlink', side_effect=changing_exe), \
                    patch.object(prune.ctypes, 'CDLL', return_value=libc), \
                    patch.object(prune.os, 'close') as close, \
                    redirect_stderr(io.StringIO()) as output:
                with self.assertRaises(ValueError):
                    prune.active_processes(run, proc, terminate_sftp=True)
            libc.pidfd_send_signal.assert_not_called()
            event = json.loads(output.getvalue())
            self.assertEqual('signal_not_sent', event['outcome'])
            self.assertEqual(int(child.name), event['pid'])
            if stage:
                close.assert_called_once_with(42)
            else:
                close.assert_not_called()

        signals = []
        for failure in ('send', 'timeout', 'poll'):
            libc = SimpleNamespace(pidfd_open=Mock(return_value=42),
                                   pidfd_send_signal=Mock(return_value=-1 if failure == 'send' else 0))
            poller = Mock()
            poller.poll.return_value = []
            if failure == 'poll':
                poller.poll.side_effect = OSError('poll failed')
            with self.subTest(failure=failure), patch.object(Path, 'stat', root_stat), \
                    patch.object(prune.os, 'readlink', side_effect=PermissionError('denied')), \
                    patch.object(prune.ctypes, 'CDLL', return_value=libc), \
                    patch.object(prune.select, 'poll', return_value=poller), \
                    patch.object(prune.os, 'close') as close, \
                    redirect_stderr(io.StringIO()) as output:
                with self.assertRaises(OSError):
                    prune.active_processes(run, proc, terminate_sftp=True)
            event = json.loads(output.getvalue())
            self.assertEqual('signal_not_sent' if failure == 'send' else
                             'term_sent_exit_unconfirmed', event['outcome'])
            self.assertEqual(int(child.name), event['identity']['pid'])
            self.assertEqual('permission_denied', event['identity']['exe_status'])
            self.assertIsNone(event['identity']['exe'])
            libc.pidfd_send_signal.assert_called_once_with(42, signal.SIGTERM, None, 0)
            close.assert_called_once_with(42)

        def fake_term(pid, verifier, audit):
            first = verifier()
            self.assertEqual(first, verifier())
            signals.append((pid, first))
            audit.update(identity=first, outcome='term_sent_exit_observed')
            for part in child.iterdir():
                part.unlink()
            child.rmdir()
            return first
        with patch.object(Path, 'stat', root_stat), \
                patch.object(prune, '_term_checked_pidfd', side_effect=fake_term), \
                patch.object(prune.os, 'readlink', side_effect=PermissionError('opaque cwd')), \
                redirect_stderr(io.StringIO()) as output:
            self.assertEqual([], prune.active_processes(run, proc, terminate_sftp=True))
        self.assertEqual(1, len(signals))
        self.assertEqual(int(child.name), signals[0][0])
        event = json.loads(output.getvalue())
        self.assertEqual('sftp_terminated_for_retention', event['kind'])
        self.assertEqual('term_sent_exit_observed', event['outcome'])
        self.assertEqual(signals[0][1], event['identity'])
        child.mkdir()
        write_process(child, 'sftp-server', parent.name, uid,
                      ['/usr/lib/openssh/sftp-server'])

        with patch.object(Path, 'stat', root_stat), \
                patch.object(prune, '_term_checked_pidfd') as term, \
                patch.object(prune.os, 'readlink', side_effect=PermissionError('opaque cwd')):
            with self.assertRaisesRegex(ValueError, 'unreadable'):
                prune.active_processes(run, proc)
            term.assert_not_called()

        original = {path: path.read_bytes() for path in (
            child / 'status', child / 'comm', child / 'cmdline', child / 'stat',
            parent / 'status', parent / 'comm', parent / 'cmdline', parent / 'stat')}
        for label, path, changed in (
                ('argv', child / 'cmdline', b'python\0'),
                ('comm', child / 'comm', b'python\n'),
                ('uid', child / 'status', original[child / 'status'].replace(
                    f'Uid:\t{uid}'.encode(), b'Uid:\t999999', 1)),
                ('parent', child / 'stat', original[child / 'stat'].replace(
                    f'S {parent.name} '.encode(), b'S 1 ', 1)),
                ('parent_comm', parent / 'comm', b'python\n')):
            with self.subTest(label=label):
                path.write_bytes(changed)
                with patch.object(Path, 'stat', root_stat), \
                        patch.object(prune, '_term_checked_pidfd', Mock()) as term, \
                        patch.object(prune.os, 'readlink', side_effect=PermissionError('opaque cwd')):
                    with self.assertRaises(ValueError):
                        prune.active_processes(run, proc, terminate_sftp=True)
                    term.assert_not_called()
                path.write_bytes(original[path])

        signaled_after_change = []
        def changed_start_after_pidfd(pid, verifier, audit):
            (child / 'stat').write_text(original[child / 'stat'].decode().replace(
                str(100 + int(child.name)), '2345678901'))
            verifier()
            signaled_after_change.append(pid)
        with patch.object(Path, 'stat', root_stat), \
                patch.object(prune, '_term_checked_pidfd', side_effect=changed_start_after_pidfd), \
                patch.object(prune.os, 'readlink', side_effect=PermissionError('opaque cwd')):
            with self.assertRaises(ValueError):
                prune.active_processes(run, proc, terminate_sftp=True)
        self.assertEqual([], signaled_after_change)
        (child / 'stat').write_bytes(original[child / 'stat'])

        (child / 'stat').unlink()
        with patch.object(Path, 'stat', root_stat), \
                patch.object(prune, '_term_checked_pidfd', Mock()) as term, \
                patch.object(prune.os, 'readlink', side_effect=PermissionError('opaque cwd')):
            with self.assertRaises(ValueError):
                prune.active_processes(run, proc, terminate_sftp=True)
            term.assert_not_called()
        (child / 'stat').write_bytes(original[child / 'stat'])

        second = proc / '999999996'
        second.mkdir()
        write_process(second, 'sftp-server', parent.name, uid,
                      ['/usr/lib/openssh/sftp-server'])
        terminated = []
        def remove_one(pid, verifier, audit):
            identity = verifier()
            terminated.append(pid)
            entry = proc / str(pid)
            for part in entry.iterdir():
                part.unlink()
            entry.rmdir()
            return identity
        with patch.object(Path, 'stat', root_stat), \
                patch.object(prune, '_term_checked_pidfd', side_effect=remove_one), \
                patch.object(prune.os, 'readlink', side_effect=PermissionError('opaque cwd')), \
                redirect_stderr(io.StringIO()):
            with self.assertRaises(ValueError):
                prune.active_processes(run, proc, terminate_sftp=True)
        self.assertEqual(1, len(terminated))
        remaining = proc / str(({int(child.name), int(second.name)} - set(terminated)).pop())
        if remaining == second:
            for part in second.iterdir():
                part.unlink()
            second.rmdir()
            child.mkdir()
            write_process(child, 'sftp-server', parent.name, uid,
                          ['/usr/lib/openssh/sftp-server'])

        def trusted_exe_only(path):
            if path == child / 'exe':
                return '/usr/lib/openssh/sftp-server'
            raise PermissionError('SFTP cwd denied')

        with patch.object(Path, 'stat', root_stat), \
                patch.object(prune.os, 'readlink', side_effect=trusted_exe_only), \
                redirect_stderr(io.StringIO()) as output:
            self.assertEqual([], prune.active_processes(run, proc))
            event = json.loads(output.getvalue())
            self.assertEqual('verified_sftp_cwd_exemption', event['kind'])

        def forged_exe(path):
            if path == child / 'exe':
                return '/usr/bin/python3'
            raise PermissionError('opaque cwd')

        with patch.object(Path, 'stat', root_stat), \
                patch.object(prune.os, 'readlink', side_effect=forged_exe):
            with self.assertRaisesRegex(ValueError, 'unreadable'):
                prune.active_processes(run, proc)

        for name, args, owner, parent_args in (
                ('python', ['/usr/lib/openssh/sftp-server'], uid,
                 [f'sshd: {username}@notty']),
                ('sftp-server', ['python', '--run-dir', str(run)], uid,
                 [f'sshd: {username}@notty']),
                ('sftp-server', ['/usr/lib/openssh/sftp-server'], uid + 1,
                 [f'sshd: {username}@notty']),
                ('sftp-server', ['/usr/lib/openssh/sftp-server'], uid,
                 ['sshd: wrong-user@notty'])):
            with self.subTest(name=name, args=args, owner=owner,
                              parent_args=parent_args):
                write_process(child, name, parent.name, owner, args)
                write_process(parent, 'sshd', privileged.name, uid, parent_args)
                with patch.object(Path, 'stat', root_stat), \
                        patch.object(prune.os, 'readlink',
                                     side_effect=PermissionError('opaque cwd')):
                    with self.assertRaisesRegex(ValueError, 'unreadable'):
                        prune.active_processes(run, proc)

        write_process(child, 'sftp-server', parent.name, uid,
                      ['/usr/lib/openssh/sftp-server'])
        write_process(parent, 'sshd', privileged.name, uid,
                      [f'sshd: {username}@notty'])
        with patch.object(Path, 'stat', root_stat), \
                patch.object(prune.os, 'readlink', return_value=str(run)), \
                patch.object(prune, 'process_matches', return_value=True) as matches:
            self.assertEqual([int(child.name)], prune.active_processes(run, proc))
            matches.assert_called_once()

    def test_sftp_pidfd_esrch_stops_scan_even_after_candidate_disappears(self):
        run, _ = self.fixture()
        proc = run.parent / 'proc'
        attempted = []
        for pid in (101, 102):
            entry = proc / str(pid)
            entry.mkdir(parents=True)
            (entry / 'cmdline').write_bytes(b'python\0')
            (entry / 'status').write_text(
                f'Name:\tpython\nState:\tS\nPPid:\t1\nUid:\t'
                + '\t'.join([str(os.getuid())] * 4) + '\n')

        def exited_before_term(pid, verifier, audit):
            attempted.append(pid)
            entry = proc / str(pid)
            for part in entry.iterdir():
                part.unlink()
            entry.rmdir()
            raise ProcessLookupError('pidfd target exited')

        with patch.object(prune, 'unrelated_daemon', return_value=False), \
                patch.object(prune, 'verified_sftp_child', return_value=False), \
                patch.object(prune, '_sftp_termination_identity',
                             side_effect=lambda entry, root: {'pid': int(entry.name)}), \
                patch.object(prune, '_term_checked_pidfd', side_effect=exited_before_term), \
                patch.object(prune.os, 'readlink', side_effect=PermissionError('opaque cwd')):
            with self.assertRaisesRegex(ValueError, 'SFTP termination'):
                prune.active_processes(run, proc, terminate_sftp=True)
        self.assertEqual(1, len(attempted))
        self.assertIn(attempted[0], (101, 102))

    def test_missing_cwd_never_invokes_sftp_terminator(self):
        run, _ = self.fixture()
        proc = run.parent / 'proc'
        entry = proc / '101'
        entry.mkdir(parents=True)
        (entry / 'cmdline').write_bytes(b'python\0')
        (entry / 'status').write_text(
            f'Name:\tpython\nState:\tS\nPPid:\t1\nUid:\t'
            + '\t'.join([str(os.getuid())] * 4) + '\n')
        with patch.object(prune, 'unrelated_daemon', return_value=False), \
                patch.object(prune, '_term_checked_pidfd') as term, \
                patch.object(prune, '_pidfd_confirms_exit', return_value=False), \
                patch.object(prune.os, 'readlink', side_effect=FileNotFoundError('cwd gone')):
            with self.assertRaises(ValueError):
                prune.active_processes(run, proc, terminate_sftp=True)
            term.assert_not_called()

    def test_rename_noreplace_refuses_an_existing_tombstone(self):
        run, evidence = self.fixture()
        plan = prune.build_prune_plan(run, evidence, 'a' * 64, 'b' * 40, 2)
        source = run / plan['files'][0]['path']
        destination = run / 'existing-tombstone'
        destination.write_bytes(b'must-not-be-overwritten')
        with prune.PinnedDir(source.parent) as parent, prune.PinnedDir(run) as holding:
            with self.assertRaises(FileExistsError):
                prune.rename_noreplace(
                    parent, source.name, holding, destination.name)
        self.assertTrue(source.is_file())
        self.assertEqual(b'must-not-be-overwritten', destination.read_bytes())

    def test_candidate_and_quarantine_paths_have_no_name_based_delete(self):
        source = inspect.getsource(prune.quarantine)
        source += inspect.getsource(prune._quarantine_truncate)
        self.assertNotIn('os.unlink', source)
        self.assertNotIn('os.rmdir', source)

    def test_process_symlink_option_fails_closed(self):
        run, _ = self.fixture()
        alias = run.parent / 'alias'
        alias.symlink_to(run)
        with self.assertRaises((ValueError, OSError)):
            prune.process_matches(['--run-dir', str(alias)], run, run.parent)

    def test_kept_marker_identity_remains_pinned_between_reads(self):
        run, _ = self.fixture()
        marker = run / 'RESOURCE_EVIDENCE.json'
        prune.write_json_exclusive(marker, {'x': 1})
        with prune.PinnedDir(run) as directory:
            prune.read_json(directory, marker.name)
            marker.rename(run / 'old-marker')
            prune.write_json_exclusive(marker, {'x': 1})
            with self.assertRaises(ValueError):
                directory.verify()

    def test_record_artifact_map_must_bind_in_full(self):
        _, evidence = self.fixture()
        evidence.update({
            'plan_sha256': '1' * 64,
            'job_spec_sha256': '2' * 64,
            'claim_sha256': '3' * 64,
            'launch_sha256': '4' * 64,
            'command_sha256': '5' * 64,
            'resource': {},
        })
        record = {
            name: evidence[name]
            for name in ('spec_key', 'plan_sha256', 'claim_sha256',
                         'launch_sha256', 'command_sha256', 'resource')
        }
        mapping = {'checkpoints/formal_final.pt': 'checkpoint',
                   'config.json': 'config',
                   'data_flow_audit.jsonl': 'data_flow',
                   'results.json': 'results',
                   'validation/validation_manifest.json': 'validation_manifest'}
        record['artifact_sha256'] = {
            mapping.get(name, 'formal:' + name): value
            for name, value in evidence['artifact_sha256'].items()
            if name != 'job.log'
        }
        record['log_sha256'] = evidence['artifact_sha256']['job.log']
        expected = tuple(evidence['artifact_sha256'])
        self.assertEqual(None, prune.bind_record(record, evidence, expected))
        for change in ('missing', 'extra', 'hash', 'log', 'resource_missing', 'resource_extra', 'schema_extra'):
            with self.subTest(change=change):
                r = json.loads(json.dumps(record))
                e = json.loads(json.dumps(evidence))
                if change == 'missing':
                    del r['artifact_sha256']['formal:checkpoints/event_0_CIL.pt']
                elif change == 'extra':
                    r['artifact_sha256']['formal:checkpoints/event_99_CIL.pt'] = 'a' * 64
                elif change == 'hash':
                    r['artifact_sha256']['formal:checkpoints/event_0_CIL.pt'] = 'a' * 64
                elif change == 'log':
                    r['log_sha256'] = 'a' * 64
                elif change == 'resource_missing':
                    del e['artifact_sha256']['checkpoints/event_0_CIL.pt']
                elif change == 'schema_extra':
                    e['unexpected'] = True
                else:
                    e['artifact_sha256']['checkpoints/event_99_CIL.pt'] = 'a' * 64
                with self.assertRaises(ValueError):
                    prune.bind_record(r, e, expected)

    def test_completion_exact_schema(self):
        self.assertTrue(callable(getattr(prune, 'validate_completion', None)))
        run, evidence = self.fixture()
        plan = prune.build_prune_plan(run, evidence, 'a' * 64, 'b' * 40, 2)
        prune.install_plan(run, plan)
        applied = prune.apply_prune_plan(run, plan)
        value = prune.install_completion(run, plan, applied)
        self.assertEqual(value, prune.validate_completion(value, plan, run))
        for mutation in ({**value, 'extra': 1}, {**value, 'freed_bytes': 0},
                         {**value, 'kind': 'wrong'}, {**value, 'policy': 'wrong'},
                         {**value, 'source_commit': 'f' * 40}):
            with self.assertRaises(ValueError):
                prune.validate_completion(mutation, plan, run)

    def test_plan_hash_fields_require_strings(self):
        run, evidence = self.fixture()
        plan = prune.build_prune_plan(run, evidence, 'a' * 64, 'b' * 40, 2)
        for field, value in (('source_commit', int('1' * 40)),
                             ('record_sha256', int('1' * 64))):
            with self.assertRaises(ValueError):
                prune.validate_plan({**plan, field: value})

    def test_pinned_marker_name_swap_and_run_swap_rejected(self):
        self.assertTrue(callable(getattr(prune, 'PinnedDir', None)))
        run, _ = self.fixture()
        with prune.PinnedDir(run) as pinned:
            run.rename(run.parent / 'moved')
            run.mkdir()
            with self.assertRaises(ValueError):
                pinned.verify()
        run, _ = self.fixture()
        marker = run / prune.PLAN_NAME
        prune.write_json_exclusive(marker, {'x': 1})
        original = prune.os.read
        swapped = False
        def swap(fd, count):
            nonlocal swapped
            content = original(fd, count)
            if not swapped:
                swapped = True
                marker.rename(run / 'original-marker')
                marker.write_bytes(content)
                marker.chmod(0o444)
            return content
        with patch.object(prune.os, 'read', swap):
            with self.assertRaises(ValueError):
                prune.load_json_canonical(marker)

    def test_directory_swap_between_stat_and_open_is_rejected(self):
        run, evidence = self.fixture()
        original_open = prune.os.open
        swapped = False
        def swap(name, flags, *args, **kwargs):
            nonlocal swapped
            if name == 'checkpoints' and not swapped:
                swapped = True
                (run / 'checkpoints').rename(run / 'old-checkpoints')
                (run / 'checkpoints').mkdir()
            return original_open(name, flags, *args, **kwargs)
        with prune.PinnedDir(run) as parent:
            with patch.object(prune.os, 'open', side_effect=swap):
                with self.assertRaisesRegex(ValueError, 'identity'):
                    with prune.PinnedDir('checkpoints', parent):
                        pass

    def fixture(self):
        temporary = TemporaryDirectory(prefix='vfcl-prune-test-')
        self.addCleanup(temporary.cleanup)
        run = Path(temporary.name) / 'run'
        (run / 'checkpoints').mkdir(parents=True)
        (run / 'formal_snapshots').mkdir()
        for index in range(2):
            (run / 'checkpoints' / f'event_{index}_CIL.pt').write_bytes(
                f'checkpoint-{index}'.encode())
            (run / 'formal_snapshots' / f'event_{index}_CIL.pt').write_bytes(
                f'snapshot-{index}'.encode())
        final = run / 'checkpoints/event_1_CIL.pt'
        (run / 'checkpoints/resume_latest.pt').write_bytes(final.read_bytes())
        (run / 'checkpoints/formal_final.pt').write_bytes(b'keep-final')
        (run / 'results.json').write_text('{}\n')
        (run / 'config.json').write_text('{}\n')
        (run / 'data_flow_audit.jsonl').write_text('{}\n')
        (run / 'job.log').write_text('fixture log\n')
        (run / 'validation').mkdir()
        (run / 'validation/validation_manifest.json').write_text('{}\n')
        artifacts = {
            path.relative_to(run).as_posix(): digest(path)
            for path in run.rglob('*') if path.is_file()
            and path.name != 'resume_latest.pt'
        }
        evidence = {
            'kind': 'formal_resource_evidence',
            'spec_key': 'cifar100:finetune:42',
            'artifact_sha256': artifacts,
        }
        return run, evidence

    def test_plan_keeps_final_model_and_prunes_only_heavy_intermediates(self):
        run, evidence = self.fixture()
        plan = prune.build_prune_plan(
            run, evidence, 'a' * 64, 'b' * 40, 2)
        names = [item['path'] for item in plan['files']]
        self.assertEqual([
            'checkpoints/event_0_CIL.pt',
            'checkpoints/event_1_CIL.pt',
            'checkpoints/resume_latest.pt',
            'formal_snapshots/event_0_CIL.pt',
            'formal_snapshots/event_1_CIL.pt',
        ], names)
        for item in plan['files']:
            details = (run / item['path']).stat()
            self.assertEqual(
                {'path', 'sha256', 'size', 'device', 'inode', 'mode'},
                set(item))
            self.assertEqual(details.st_dev, item['device'])
            self.assertEqual(details.st_ino, item['inode'])
            self.assertEqual(stat.S_IMODE(details.st_mode), item['mode'])
        self.assertNotIn('checkpoints/formal_final.pt', names)
        self.assertNotIn('results.json', names)
        prune.install_plan(run, plan)
        applied = prune.apply_prune_plan(run, plan)
        completion = prune.install_completion(run, plan, applied)
        self.assertEqual(plan['bytes'], completion['freed_bytes'])
        self.assertTrue((run / 'checkpoints/formal_final.pt').is_file())
        self.assertTrue((run / 'results.json').is_file())
        self.assertTrue((run / 'PRUNE_PLAN.json').is_file())
        self.assertTrue((run / 'PRUNED_EVIDENCE.json').is_file())
        self.assertTrue(all(not (run / name).exists() for name in names))
        quarantine = run / completion['quarantine']
        self.assertEqual(0o700, stat.S_IMODE(quarantine.stat().st_mode))
        self.assertEqual(
            [item['tombstone'] for item in completion['files']],
            sorted(path.name for path in quarantine.iterdir()))
        self.assertTrue(all(path.stat().st_size == 0 for path in quarantine.iterdir()))

    def test_partial_plan_never_resumes(self):
        run, evidence = self.fixture()
        plan = prune.build_prune_plan(
            run, evidence, 'a' * 64, 'b' * 40, 2)
        prune.install_plan(run, plan)
        first = run / plan['files'][0]['path']
        first.unlink()
        resumed = prune.load_installed_plan(run)
        self.assertEqual(plan, resumed)
        with self.assertRaises((ValueError, FileNotFoundError)):
            prune.apply_prune_plan(run, resumed)
        self.assertTrue((run / plan['files'][1]['path']).is_file())

    def test_symlink_run_is_not_resolved_into_authority(self):
        run, evidence = self.fixture()
        alias = run.parent / 'alias'
        alias.symlink_to(run, target_is_directory=True)
        with self.assertRaises((ValueError, OSError)):
            prune.build_prune_plan(alias, evidence, 'a' * 64, 'b' * 40, 2)

    def test_hardlinked_target_rejected(self):
        run, evidence = self.fixture()
        os.link(run / 'checkpoints/event_0_CIL.pt', run / 'alias.pt')
        with self.assertRaises((ValueError, OSError)):
            prune.build_prune_plan(run, evidence, 'a' * 64, 'b' * 40, 2)

    def test_completion_rejects_dangling_target(self):
        run, evidence = self.fixture()
        plan = prune.build_prune_plan(run, evidence, 'a' * 64, 'b' * 40, 2)
        prune.install_plan(run, plan)
        applied = prune.apply_prune_plan(run, plan)
        (run / plan['files'][0]['path']).symlink_to(run / 'missing')
        with self.assertRaises(ValueError):
            prune.install_completion(run, plan, applied)

    def test_hash_time_parent_swap_never_unlinks_foreign_file(self):
        run, evidence = self.fixture()
        plan = prune.build_prune_plan(run, evidence, 'a' * 64, 'b' * 40, 2)
        prune.install_plan(run, plan)
        foreign = run.parent / 'foreign'
        foreign.mkdir()
        target = foreign / 'event_0_CIL.pt'
        target.write_bytes((run / plan['files'][0]['path']).read_bytes())
        original_hash = prune.file_hash
        swapped = False
        def swap(value):
            nonlocal swapped
            result = original_hash(value)
            if not swapped:
                swapped = True
                (run / 'checkpoints').rename(run / 'original')
                (run / 'checkpoints').symlink_to(foreign, target_is_directory=True)
            return result
        with patch.object(prune, 'file_hash', side_effect=swap):
            with self.assertRaises((ValueError, OSError)):
                prune.apply_prune_plan(run, plan)
        self.assertTrue(target.is_file())

    def test_resume_rejects_a_plan_that_differs_from_resource_evidence(self):
        run, evidence = self.fixture()
        plan = prune.build_prune_plan(
            run, evidence, 'a' * 64, 'b' * 40, 2)
        plan['files'][0]['sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'evidence'):
            prune.validate_plan_against_evidence(
                run, plan, evidence, 'a' * 64, 'b' * 40, 2)

    def test_hash_mismatch_symlink_and_incomplete_sequences_fail_closed(self):
        run, evidence = self.fixture()
        evidence['artifact_sha256']['checkpoints/event_0_CIL.pt'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'hash'):
            prune.build_prune_plan(
                run, evidence, 'a' * 64, 'b' * 40, 2)

        run, evidence = self.fixture()
        target = run / 'formal_snapshots/event_0_CIL.pt'
        target.unlink()
        target.symlink_to(run / 'results.json')
        with self.assertRaisesRegex(ValueError, 'regular'):
            prune.build_prune_plan(
                run, evidence, 'a' * 64, 'b' * 40, 2)

        run, evidence = self.fixture()
        del evidence['artifact_sha256']['formal_snapshots/event_1_CIL.pt']
        with self.assertRaisesRegex(ValueError, 'sequence'):
            prune.build_prune_plan(
                run, evidence, 'a' * 64, 'b' * 40, 2)

    def test_tampered_installed_plan_is_rejected_before_deletion(self):
        run, evidence = self.fixture()
        plan = prune.build_prune_plan(
            run, evidence, 'a' * 64, 'b' * 40, 2)
        prune.install_plan(run, plan)
        path = run / 'PRUNE_PLAN.json'
        path.chmod(0o644)
        value = json.loads(path.read_text())
        value['files'][0]['path'] = '../../escape'
        path.write_text(json.dumps(value) + '\n')
        path.chmod(0o444)
        with self.assertRaisesRegex(ValueError, 'canonical|path|plan'):
            prune.load_installed_plan(run)


@unittest.skipUnless(driver.experiment_profile() == driver.RECOVERY_PROFILE,
                     'recovery retention contract')
class ActiveAuditRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix='vfcl-active-retention-')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.worktree = Path(driver.__file__).resolve().parent
        self.head = driver._source_commit()

    @staticmethod
    def _snapshot(root):
        result = {}
        for path in root.rglob('*'):
            details = path.lstat()
            payload = path.read_bytes() if stat.S_ISREG(details.st_mode) else None
            result[path.relative_to(root).as_posix()] = (
                details.st_mode, details.st_nlink, details.st_size, payload)
        return result

    @staticmethod
    def _owner(root, job, role):
        pid = os.getpid()
        return {
            'kind': 'formal_job_claim', 'job': job,
            'launcher_token': 'active-retention-launch',
            'worker_role': role, 'phase': 'formal', 'pid': pid,
            'pgid': os.getpgid(pid),
            'process_start_time': driver._process_start_time(pid),
            'source_commit': driver._source_commit(),
            'root_identity': driver._root_identity(root),
        }

    def _fixture(self, name='authority'):
        census = driver.build_census({})
        plan = driver.build_plan(census)
        root = driver.validate_formal_root(self.base / name)
        driver.install_json_exclusive(
            root / 'FORMAL_REGISTRY.json', driver._registry_payload())
        driver.install_json_exclusive(root / 'COMPATIBILITY_CENSUS.json', census)
        driver.install_json_exclusive(root / 'FORMAL_PLAN.json', plan)
        driver.install_json_exclusive(
            root / 'MISSING_JOBS.json',
            driver._missing_jobs_payload(plan, census))
        (root / 'claims').mkdir(mode=0o700)
        driver.install_formal_root_identity(root, plan)
        key = plan['missing_jobs'][0]
        producer = self._owner(root, key, 'retention-producer')
        self.assertEqual(key, driver.claim_next(
            plan, root / 'claims', 'formal', producer))
        self.assertTrue(driver.claim_gpu(root, 0, producer))
        run = driver.prepare_run(root, key, producer)
        spec = driver.spec_for_key(key)
        for logical in driver.resource_artifact_names(spec):
            path = run / logical
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('retention:' + logical + '\n').encode())
            path.chmod(0o444)
        measurements = _completed_record(spec, .5)['resource'].copy()
        for field in ('instrumentation', 'checkpoint_size_bytes'):
            measurements.pop(field)
        measurements.update(_write_resource_checkpoint(
            spec, run / 'checkpoints/formal_final.pt'))
        evidence = driver.resource_record(
            root, key, run, measurements=measurements)
        resume = run / 'checkpoints/resume_latest.pt'
        resume.write_bytes((run / 'checkpoints/event_9_CIL.pt').read_bytes())
        resume.chmod(0o444)
        driver.queue_audit(root, key, 0, producer)
        driver.release_gpu(root, 0, producer)
        auditor = self._owner(root, '', 'retention-auditor')
        self.assertEqual(key, driver.next_audit(
            root, 'formal', auditor)['spec_key'])
        record = _completed_record(
            spec, .5, source_commit=driver._source_commit(),
            plan_sha256=driver._digest(plan))
        for field in ('claim_sha256', 'launch_sha256', 'command_sha256',
                      'resource'):
            record[field] = evidence[field]
        mapping = {
            'checkpoints/formal_final.pt': 'checkpoint',
            'config.json': 'config',
            'data_flow_audit.jsonl': 'data_flow',
            'results.json': 'results',
            'validation/validation_manifest.json': 'validation_manifest',
        }
        record['artifact_sha256'] = {
            mapping.get(name, 'formal:' + name): digest
            for name, digest in evidence['artifact_sha256'].items()
            if name != 'job.log'
        }
        record['log_sha256'] = evidence['artifact_sha256']['job.log']
        record['record_sha256'] = driver._digest({
            field: value for field, value in record.items()
            if field != 'record_sha256'})
        prune.bind_record(
            record, evidence, driver.resource_artifact_names(spec))
        records = driver._ensure_directory(root, 'records')
        driver.install_json_exclusive(
            records / f'{driver.safe_spec_name(spec)}.json', record)
        return root, plan, key, producer, auditor, run, evidence, record

    def _authority(self, worktree, expected_head):
        if Path(worktree) != self.worktree or expected_head != self.head:
            raise ValueError('test authority differs')
        return driver, registry

    def _call(self, root, key, owner, *, apply=False, processes=()):
        with patch.object(prune, 'load_authority', side_effect=self._authority), \
                patch.object(prune, 'active_processes', return_value=list(processes)):
            return prune.prune_active_audit(
                root, self.worktree, self.head, key, owner, apply=apply)

    def test_active_dry_run_reports_exact_21_targets_without_mutation(self):
        root, _, key, _, auditor, run, evidence, record = self._fixture()
        before = self._snapshot(root)
        summary = self._call(root, key, auditor)
        expected = prune.expected_paths(10)
        expected_bytes = sum((run / path).stat().st_size for path in expected)
        self.assertEqual(21, len(expected))
        self.assertEqual({
            'spec_key': key, 'candidate_count': 21,
            'reclaimable_bytes': expected_bytes, 'applied': False,
            'plan_sha256': summary['plan_sha256'],
            'completion_sha256': None,
        }, summary)
        self.assertEqual(before, self._snapshot(root))
        self.assertEqual(record, driver._load_json_file(
            root / 'records' / f'{driver._claim_name(key)}.json'))
        self.assertEqual(evidence, driver._load_json_file(
            run / 'RESOURCE_EVIDENCE.json'))

    def test_full_profile_admission_preserves_active_retention_target_policy(self):
        from types import SimpleNamespace

        root, _, key, _, auditor, run, _, _ = self._fixture()
        # Exercise the real pruner with the established active-audit evidence;
        # only the loaded authority's profile boundary differs from recovery.
        authority = SimpleNamespace(**{**vars(driver),
            'experiment_profile': lambda: driver.FULL_MATRIX_PROFILE})
        with patch.object(prune, 'load_authority', return_value=(authority, registry)), \
                patch.object(prune, 'active_processes', return_value=[]):
            before = self._snapshot(root)
            dry = prune.prune_active_audit(root, self.worktree, self.head, key, auditor)
            self.assertEqual(21, dry['candidate_count'])
            self.assertEqual(before, self._snapshot(root))
            applied = prune.prune_active_audit(root, self.worktree, self.head, key, auditor, apply=True)
        self.assertTrue(applied['applied'])
        self.assertEqual(21, applied['candidate_count'])
        self.assertTrue((run / 'checkpoints/formal_final.pt').is_file())
        self.assertTrue((root / 'audit_queue/active.json').is_file())

    def test_sftp_termination_opt_in_is_limited_to_dataset_scoped_profiles(self):
        root, _, key, _, auditor, _, _, _ = self._fixture()
        with patch.object(prune, 'load_authority', side_effect=self._authority), \
                patch.object(prune, 'active_processes') as processes:
            with self.assertRaises(ValueError):
                prune.prune_active_audit(
                    root, self.worktree, self.head, key, auditor,
                    terminate_sftp=True)
            processes.assert_not_called()

        for profile in (driver.SINGLE_DATASET_PROFILE,
                        driver.CONTINUATION_PROFILE):
            with self.subTest(profile=profile):
                authority = SimpleNamespace(**{**vars(driver),
                    'experiment_profile': lambda: profile})
                with patch.object(prune, 'load_authority',
                                  return_value=(authority, registry)), \
                        patch.object(prune, 'active_processes',
                                          return_value=[]) as processes:
                    prune.prune_active_audit(
                        root, self.worktree, self.head, key, auditor,
                        terminate_sftp=True)
                    processes.assert_called_once_with(
                        root / 'runs' / registry.safe_spec_name(
                            driver.spec_for_key(key)), terminate_sftp=True)

    def test_active_real_proc_process_gate_rejects_before_plan(self):
        root, _, key, _, auditor, run, _, _ = self._fixture(
            'real-proc-process-gate')
        targets = prune.expected_paths(10)
        before = self._snapshot(root)
        blocker = subprocess.Popen([
            sys.executable, '-B', '-c',
            "import sys; print('READY', flush=True); sys.stdin.read(1)",
            str(run),
        ], cwd=run, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
           stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual('READY', blocker.stdout.readline().strip())
            self.assertIsNone(blocker.poll())
            self.assertIn(blocker.pid, prune.active_processes(run))
            with patch.object(prune, 'load_authority', side_effect=self._authority):
                with self.assertRaisesRegex(
                        ValueError, 'completed run still has a process'):
                    prune.prune_active_audit(
                        root, self.worktree, self.head, key, auditor,
                        apply=True)
        finally:
            if blocker.poll() is None:
                blocker.terminate()
            blocker.wait(timeout=5)
            blocker.stdout.close()
            blocker.stderr.close()
            blocker.stdin.close()
        self.assertEqual(before, self._snapshot(root))
        self.assertTrue(all((run / path).is_file() for path in targets))
        self.assertFalse((run / prune.PLAN_NAME).exists())
        self.assertTrue((root / 'audit_queue/active.json').is_file())

    def test_active_apply_prunes_exact_targets_and_installs_completion_last(self):
        root, _, key, _, auditor, run, evidence, _ = self._fixture()
        targets = set(prune.expected_paths(10))
        original = {
            path: (run / path).stat()
            for path in targets
        }
        retained = {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob('*')
            if path.is_file()
            and not (path.is_relative_to(run)
                     and path.relative_to(run).as_posix() in targets)
        }
        writes = []
        real_write = prune.write_marker

        def observed(parent, name, value):
            writes.append(name)
            return real_write(parent, name, value)

        with patch.object(prune, 'write_marker', side_effect=observed):
            summary = self._call(root, key, auditor, apply=True)
        self.assertTrue(summary['applied'])
        self.assertEqual(21, summary['candidate_count'])
        self.assertIsNotNone(summary['completion_sha256'])
        self.assertEqual([prune.PLAN_NAME, prune.COMPLETE_NAME], writes)
        self.assertEqual(prune.COMPLETE_NAME, writes[-1])
        self.assertTrue(all(not (run / path).exists() for path in targets))
        self.assertEqual(retained, {
            name: root.joinpath(name).read_bytes() for name in retained})
        self.assertEqual(0o444, stat.S_IMODE((run / prune.PLAN_NAME).stat().st_mode))
        self.assertEqual(0o444, stat.S_IMODE((run / prune.COMPLETE_NAME).stat().st_mode))
        self.assertTrue((root / 'audit_queue/active.json').is_file())
        completion = prune.load_json_canonical(run / prune.COMPLETE_NAME)
        self.assertEqual(prune.QUARANTINE_PREFIX, completion['quarantine'][:len(prune.QUARANTINE_PREFIX)])
        quarantine = run / completion['quarantine']
        self.assertEqual(0o700, stat.S_IMODE(quarantine.stat().st_mode))
        self.assertEqual(21, len(completion['files']))
        by_path = {item['path']: item for item in completion['files']}
        self.assertEqual(targets, set(by_path))
        self.assertEqual(21, len({item['tombstone'] for item in completion['files']}))
        for logical in sorted(targets):
            item = by_path[logical]
            tombstone = quarantine / item['tombstone']
            self.assertTrue(tombstone.is_file())
            self.assertEqual(0, tombstone.stat().st_size)
            self.assertEqual(0, item['final_size'])
            self.assertEqual(original[logical].st_ino, item['inode'])
            self.assertEqual(original[logical].st_size, item['original_size'])
            expected_hash = evidence['artifact_sha256'].get(logical)
            if logical == 'checkpoints/resume_latest.pt':
                expected_hash = evidence['artifact_sha256']['checkpoints/event_9_CIL.pt']
            self.assertEqual(expected_hash, item['sha256'])

    def test_active_preapply_identity_and_runtime_rejections_change_nothing(self):
        cases = (
            ('wrong-key', lambda state: state.update(key=state['plan']['missing_jobs'][1])),
            ('wrong-owner', lambda state: state.update(
                owner={**state['owner'], 'launcher_token': 'foreign'})),
            ('wrong-head', lambda state: state.update(head='0' * 40)),
            ('wrong-root', lambda state: state.update(root=state['root'].parent / 'alias')),
            ('wrong-run', lambda state: state.update(context_run=state['root'])),
            ('no-active', lambda state: (state['root'] / 'audit_queue/active.json').unlink()),
            ('active-gpu', lambda state: driver.claim_gpu(
                state['root'], 0, state['producer'])),
            ('live-producer', lambda state: state.update(processes=(12345,))),
            ('unknown-queue', lambda state: driver.install_json_exclusive(
                state['root'] / 'audit_queue/unknown.json', {})),
        )
        for name, mutate in cases:
            with self.subTest(case=name):
                root, plan, key, producer, owner, run, _, _ = self._fixture(name)
                if name == 'wrong-root':
                    (root.parent / 'alias').symlink_to(root, target_is_directory=True)
                state = dict(root=root, plan=plan, key=key, producer=producer,
                             owner=owner, run=run, head=self.head, processes=(),
                             context_run=None)
                mutate(state)
                before = self._snapshot(root)
                real_context = driver._retention_audit_context_locked

                def context(*args):
                    value = real_context(*args)
                    if state['context_run'] is not None:
                        value = json.loads(json.dumps(value))
                        value['run_dir'] = str(state['context_run'])
                    return value

                authority = lambda worktree, head: (
                    (driver, registry) if head == state['head']
                    else (_ for _ in ()).throw(ValueError('test authority differs')))
                with patch.object(prune, 'load_authority', side_effect=authority), \
                        patch.object(prune, 'active_processes',
                                     return_value=list(state['processes'])), \
                        patch.object(driver, '_retention_audit_context_locked',
                                     side_effect=context):
                    with self.assertRaises((ValueError, OSError, RuntimeError)):
                        prune.prune_active_audit(
                            state['root'], self.worktree, state['head'],
                            state['key'], state['owner'], apply=True)
                self.assertEqual(before, self._snapshot(root))
                self.assertFalse((run / prune.PLAN_NAME).exists())

    def test_active_target_type_content_and_sequence_rejections_are_preapply(self):
        for name in ('symlink', 'hardlink', 'fifo', 'size', 'hash',
                     'missing-sequence', 'extra-sequence'):
            with self.subTest(case=name):
                root, _, key, _, owner, run, _, _ = self._fixture(name)
                target = run / 'checkpoints/event_0_CIL.pt'
                context_change = None
                if name == 'symlink':
                    target.unlink()
                    target.symlink_to(run / 'results.json')
                elif name == 'hardlink':
                    os.link(target, run / 'hardlink.pt')
                elif name == 'fifo':
                    target.unlink()
                    os.mkfifo(target)
                elif name == 'size':
                    target.chmod(0o644)
                    target.write_bytes(b'changed-size')
                    target.chmod(0o444)
                elif name == 'hash':
                    target.chmod(0o644)
                    target.write_bytes(b'x' * target.stat().st_size)
                    target.chmod(0o444)
                else:
                    context_change = name
                before = self._snapshot(root)
                real_context = driver._retention_audit_context_locked

                def context(*args):
                    value = real_context(*args)
                    if context_change is None:
                        return value
                    value = json.loads(json.dumps(value))
                    artifacts = value['resource']['artifact_sha256']
                    if context_change == 'missing-sequence':
                        del artifacts['checkpoints/event_0_CIL.pt']
                    else:
                        artifacts['checkpoints/event_10_CIL.pt'] = '0' * 64
                    return value

                with patch.object(prune, 'load_authority', side_effect=self._authority), \
                        patch.object(prune, 'active_processes', return_value=[]), \
                        patch.object(driver, '_retention_audit_context_locked',
                                     side_effect=context):
                    with self.assertRaises((ValueError, OSError)):
                        prune.prune_active_audit(
                            root, self.worktree, self.head, key, owner, apply=True)
                self.assertEqual(before, self._snapshot(root))
                self.assertFalse((run / prune.PLAN_NAME).exists())

    def test_active_directory_swap_never_deletes_replacement(self):
        root, _, key, _, owner, run, _, _ = self._fixture('directory-swap')
        original_open = prune.os.open
        swapped = False

        def swap(name, flags, *args, **kwargs):
            nonlocal swapped
            if name == 'checkpoints' and not swapped:
                swapped = True
                old = run / 'old-checkpoints'
                (run / 'checkpoints').rename(old)
                (run / 'checkpoints').mkdir()
                for source in old.iterdir():
                    if source.is_file():
                        destination = run / 'checkpoints' / source.name
                        destination.write_bytes(source.read_bytes())
                        destination.chmod(source.stat().st_mode & 0o777)
            return original_open(name, flags, *args, **kwargs)

        with patch.object(prune, 'load_authority', side_effect=self._authority), \
                patch.object(prune, 'active_processes', return_value=[]), \
                patch.object(prune.os, 'open', side_effect=swap):
            with self.assertRaisesRegex(ValueError, 'identity'):
                prune.prune_active_audit(
                    root, self.worktree, self.head, key, owner, apply=True)
        self.assertTrue(swapped)
        self.assertTrue((run / 'checkpoints/event_0_CIL.pt').is_file())
        self.assertFalse((run / prune.PLAN_NAME).exists())

    def test_active_partial_plan_and_existing_completion_are_terminal(self):
        for completed in (False, True):
            with self.subTest(completed=completed):
                root, _, key, _, owner, run, evidence, record = self._fixture(
                    f'existing-{completed}')
                plan = prune.build_prune_plan(
                    run, evidence, record['record_sha256'], self.head, 10)
                prune.install_plan(run, plan)
                if completed:
                    applied = prune.apply_prune_plan(run, plan)
                    prune.install_completion(run, plan, applied)
                before = self._snapshot(root)
                with self.assertRaisesRegex(ValueError, 'partial|completion'):
                    self._call(root, key, owner, apply=True)
                self.assertEqual(before, self._snapshot(root))

    def test_active_apply_failures_are_terminal_manual_and_keep_handoff(self):
        for failure in ('rename', 'reopen', 'truncate', 'fsync', 'completion-write'):
            with self.subTest(failure=failure):
                root, _, key, _, owner, run, _, _ = self._fixture(failure)
                real_rename = prune.rename_noreplace
                real_open = prune.os.open
                real_truncate = prune.os.ftruncate
                real_fsync = prune.os.fsync
                real_write = prune.write_marker
                calls = {'rename': 0, 'truncate': 0, 'truncated': False}

                def rename(*args, **kwargs):
                    calls['rename'] += 1
                    if failure == 'rename' and calls['rename'] == 2:
                        raise OSError('injected rename failure')
                    return real_rename(*args, **kwargs)

                def opened(name, flags, *args, **kwargs):
                    if (failure == 'reopen' and flags & os.O_RDWR
                            and str(name).startswith(('checkpoints.', 'formal_snapshots.'))):
                        raise OSError('injected writable reopen failure')
                    return real_open(name, flags, *args, **kwargs)

                def truncate(fd, size):
                    calls['truncate'] += 1
                    if failure == 'truncate' and calls['truncate'] == 1:
                        raise OSError('injected truncate failure')
                    result = real_truncate(fd, size)
                    calls['truncated'] = True
                    return result

                def fsync(fd):
                    if failure == 'fsync' and calls['truncated']:
                        raise OSError('injected tombstone fsync failure')
                    return real_fsync(fd)

                def write(parent, name, value):
                    if name == prune.COMPLETE_NAME:
                        raise OSError('injected completion write failure')
                    return real_write(parent, name, value)

                patches = [
                    patch.object(prune, 'load_authority', side_effect=self._authority),
                    patch.object(prune, 'active_processes', return_value=[]),
                    patch.object(driver, 'complete_audit', wraps=driver.complete_audit),
                    patch.object(prune, 'rename_noreplace', side_effect=rename),
                    patch.object(prune.os, 'open', side_effect=opened),
                    patch.object(prune.os, 'ftruncate', side_effect=truncate),
                    patch.object(prune.os, 'fsync', side_effect=fsync),
                ]
                if failure == 'completion-write':
                    patches.append(patch.object(prune, 'write_marker', side_effect=write))
                with ExitStack() as stack:
                    mocks = [stack.enter_context(item) for item in patches]
                    with self.assertRaisesRegex(RuntimeError, 'terminal.*manual'):
                        prune.prune_active_audit(
                            root, self.worktree, self.head, key, owner, apply=True)
                mocks[2].assert_not_called()
                self.assertTrue((run / prune.PLAN_NAME).is_file())
                self.assertFalse((run / prune.COMPLETE_NAME).exists())
                self.assertTrue((root / 'audit_queue/active.json').is_file())
                quarantines = [path for path in run.iterdir()
                               if path.name.startswith(prune.QUARANTINE_PREFIX)]
                self.assertEqual(1, len(quarantines))
                self.assertTrue(all(path.is_file() for path in quarantines[0].iterdir()))

    def test_active_apply_holds_audit_capacity_until_completion_evidence(self):
        root, plan, key, _, owner, _, _, _ = self._fixture('locked-transaction')
        next_owner = self._owner(root, '', 'retention-competitor')
        real_install = prune.install_plan
        completed = [threading.Event(), threading.Event()]
        attempted = [threading.Event(), threading.Event()]
        errors = []
        claimed = []
        blocked = []

        def finish_audit():
            try:
                driver.complete_audit(root, key, owner)
            except BaseException as error:
                errors.append(error)
            finally:
                completed[0].set()

        def claim_job():
            try:
                claimed.append(driver.claim_next(
                    plan, root / 'claims', 'formal', next_owner))
            except BaseException as error:
                errors.append(error)
            finally:
                completed[1].set()

        def install_while_competing(run, selected):
            real_flock = driver.fcntl.flock
            threads = [
                threading.Thread(target=finish_audit, name='retention-complete'),
                threading.Thread(target=claim_job, name='retention-claim'),
            ]

            def observed_flock(fd, operation):
                name = threading.current_thread().name
                if operation == driver.fcntl.LOCK_EX and name.startswith('retention-'):
                    attempted[0 if name == 'retention-complete' else 1].set()
                return real_flock(fd, operation)

            with patch.object(driver.fcntl, 'flock', side_effect=observed_flock):
                for thread in threads:
                    thread.start()
                self.assertTrue(all(event.wait(5) for event in attempted))
                blocked.extend(not event.is_set() for event in completed)
            self.competitors = threads
            return real_install(run, selected)

        with patch.object(prune, 'load_authority', side_effect=self._authority), \
                patch.object(prune, 'active_processes', return_value=[]), \
                patch.object(prune, 'install_plan', side_effect=install_while_competing):
            summary = prune.prune_active_audit(
                root, self.worktree, self.head, key, owner, apply=True)
        for thread in self.competitors:
            thread.join(5)
            self.assertFalse(thread.is_alive())
        self.assertEqual([True, True], blocked)
        self.assertEqual([], errors)
        self.assertTrue(summary['applied'])
        self.assertEqual([plan['missing_jobs'][1]], claimed)

    def test_active_post_link_plan_failure_is_terminal_with_installed_evidence(self):
        root, _, key, _, owner, run, _, _ = self._fixture('post-link-plan')
        targets = prune.expected_paths(10)
        real_unlink = prune.os.unlink

        def fail_after_plan_link(name, *args, **kwargs):
            result = real_unlink(name, *args, **kwargs)
            if str(name).startswith(f'.{prune.PLAN_NAME}.'):
                raise OSError('injected post-link plan cleanup failure')
            return result

        with patch.object(prune, 'load_authority', side_effect=self._authority), \
                patch.object(prune, 'active_processes', return_value=[]), \
                patch.object(prune.os, 'unlink', side_effect=fail_after_plan_link):
            with self.assertRaisesRegex(
                    RuntimeError, '^retention apply is terminal and requires manual review$') as raised:
                prune.prune_active_audit(
                    root, self.worktree, self.head, key, owner, apply=True)
        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertEqual(
            'injected post-link plan cleanup failure', str(raised.exception.__cause__))
        self.assertTrue((run / prune.PLAN_NAME).is_file())
        self.assertFalse((run / prune.COMPLETE_NAME).exists())
        self.assertTrue(all((run / path).is_file() for path in targets))
        self.assertTrue((root / 'audit_queue/active.json').is_file())

    def test_active_swap_immediately_before_rename_preserves_replacement_bytes(self):
        root, _, key, _, owner, run, _, _ = self._fixture('atomic-move-swap')
        target = run / 'checkpoints/event_0_CIL.pt'
        displaced = run / 'attacker-original-event-0.pt'
        replacement = b'attacker replacement must survive\n'
        real_rename = prune.rename_noreplace
        swapped = False

        def swap_before_move(parent, source, holding, destination):
            nonlocal swapped
            if source == target.name and not swapped:
                swapped = True
                # Move the planned inode out of the candidate slot, then install
                # an untrusted replacement at that exact source name.
                target.rename(displaced)
                target.write_bytes(replacement)
                target.chmod(0o444)
            return real_rename(parent, source, holding, destination)

        with patch.object(prune, 'load_authority', side_effect=self._authority), \
                patch.object(prune, 'active_processes', return_value=[]), \
                patch.object(prune, 'rename_noreplace', side_effect=swap_before_move):
            with self.assertRaisesRegex(
                    RuntimeError, '^retention apply is terminal and requires manual review$'):
                prune.prune_active_audit(
                    root, self.worktree, self.head, key, owner, apply=True)
        quarantines = [path for path in run.iterdir()
                       if path.name.startswith(prune.QUARANTINE_PREFIX)]
        self.assertTrue(swapped)
        self.assertEqual(1, len(quarantines))
        self.assertEqual(0o700, stat.S_IMODE(quarantines[0].stat().st_mode))
        self.assertIn(replacement, [path.read_bytes()
                                   for path in quarantines[0].iterdir()])
        self.assertTrue(displaced.is_file())
        self.assertTrue((run / prune.PLAN_NAME).is_file())
        self.assertFalse((run / prune.COMPLETE_NAME).exists())
        self.assertTrue((root / 'audit_queue/active.json').is_file())

    def test_active_swap_immediately_before_writable_reopen_truncates_neither_replacement(self):
        root, _, key, _, owner, run, _, _ = self._fixture('writable-reopen-swap')
        target = run / 'checkpoints/event_0_CIL.pt'
        planned_bytes = target.read_bytes()
        displaced = run / 'planned-inode-before-writable-reopen.pt'
        replacement = b'writable reopen replacement must survive\n'
        real_open = prune.os.open
        swapped = False

        def swap_before_writable_reopen(name, flags, *args, **kwargs):
            nonlocal swapped
            if (not swapped and flags & os.O_RDWR
                    and name == 'checkpoints.event_0_CIL.pt'
                    and kwargs.get('dir_fd') is not None):
                swapped = True
                quarantine = Path(os.readlink(
                    f'/proc/self/fd/{kwargs["dir_fd"]}'))
                tombstone = quarantine / name
                tombstone.rename(displaced)
                tombstone.write_bytes(replacement)
                tombstone.chmod(0o444)
            return real_open(name, flags, *args, **kwargs)

        with patch.object(prune, 'load_authority', side_effect=self._authority), \
                patch.object(prune, 'active_processes', return_value=[]), \
                patch.object(prune.os, 'open', side_effect=swap_before_writable_reopen):
            with self.assertRaisesRegex(
                    RuntimeError, '^retention apply is terminal and requires manual review$'):
                prune.prune_active_audit(
                    root, self.worktree, self.head, key, owner, apply=True)
        quarantines = [path for path in run.iterdir()
                       if path.name.startswith(prune.QUARANTINE_PREFIX)]
        self.assertTrue(swapped)
        self.assertEqual(1, len(quarantines))
        self.assertEqual(replacement,
                         (quarantines[0] / 'checkpoints.event_0_CIL.pt').read_bytes())
        self.assertEqual(planned_bytes, displaced.read_bytes())
        self.assertEqual(0o444, stat.S_IMODE(displaced.stat().st_mode))
        self.assertTrue((run / prune.PLAN_NAME).is_file())
        self.assertFalse((run / prune.COMPLETE_NAME).exists())
        self.assertTrue((root / 'audit_queue/active.json').is_file())

    def test_active_mode_restore_failure_preserves_primary_evidence(self):
        root, _, key, _, owner, run, _, _ = self._fixture(
            'mode-restore-failure')
        real_open = prune.os.open
        real_fchmod = prune.os.fchmod
        state = {'writable': False}

        def fail_reopen(name, flags, *args, **kwargs):
            if (flags & os.O_RDWR
                    and name == 'checkpoints.event_0_CIL.pt'
                    and kwargs.get('dir_fd') is not None):
                raise OSError('injected writable reopen failure')
            return real_open(name, flags, *args, **kwargs)

        def fail_restore(descriptor, mode):
            if state['writable'] and mode == 0o444:
                raise OSError('injected mode restore failure')
            result = real_fchmod(descriptor, mode)
            if mode == 0o644:
                state['writable'] = True
            return result

        with patch.object(prune, 'load_authority', side_effect=self._authority), \
                patch.object(prune, 'active_processes', return_value=[]), \
                patch.object(prune.os, 'open', side_effect=fail_reopen), \
                patch.object(prune.os, 'fchmod', side_effect=fail_restore):
            with self.assertRaisesRegex(
                    RuntimeError,
                    '^retention apply is terminal and requires manual review$') as raised:
                prune.prune_active_audit(
                    root, self.worktree, self.head, key, owner, apply=True)
        restoration = raised.exception.__cause__
        primary = restoration.__cause__
        self.assertEqual('injected writable reopen failure', str(primary))
        self.assertIs(restoration.primary_error, primary)
        self.assertEqual('injected mode restore failure',
                         str(restoration.restoration_error))
        self.assertIn('injected mode restore failure', str(restoration))
        self.assertTrue((run / prune.PLAN_NAME).is_file())
        self.assertFalse((run / prune.COMPLETE_NAME).exists())
        self.assertTrue((root / 'audit_queue/active.json').is_file())

    def test_real_load_authority_exact_worktree_loading(self):
        _, _, _, _, _, run, _, _ = self._fixture('real-authority')
        clone = self.base / 'clean-authority'
        subprocess.run([
            'git', 'clone', '--quiet', '--no-hardlinks',
            str(self.worktree), str(clone),
        ], check=True)
        environment = os.environ.copy()
        environment['VFCL_EXPERIMENT_PROFILE'] = driver.RECOVERY_PROFILE
        completed = subprocess.run([
            sys.executable, '-B', '-c',
            ('import json, pathlib, prune_completed_runs as prune, sys; '
             'driver, registry = prune.load_authority(sys.argv[1], sys.argv[2]); '
             'print(json.dumps({"driver": str(pathlib.Path(driver.__file__).parent), '
             '"head": driver._source_commit(), '
             '"processes": prune.active_processes(pathlib.Path(sys.argv[3]))}, '
             'sort_keys=True))'),
            str(clone), self.head, str(run),
        ], check=True, capture_output=True, text=True, env=environment)
        result = json.loads(completed.stdout)
        self.assertEqual(str(clone), result['driver'])
        self.assertEqual(self.head, result['head'])
        self.assertEqual([], result['processes'])

    def test_real_active_process_scan_allows_disposable_completed_run(self):
        _, _, _, _, _, run, _, _ = self._fixture('real-process-scan')
        self.assertEqual([], prune.active_processes(run))

    def test_active_cli_requires_exact_owner_and_mode_and_emits_summary(self):
        root, _, key, _, owner, _, _, _ = self._fixture('cli')
        arguments = [
            '--root', str(root), '--worktree', str(self.worktree),
            '--expected-head', self.head, '--spec-key', key,
            '--owner-json', json.dumps(owner, sort_keys=True, separators=(',', ':')),
        ]
        for invalid in (arguments, [*arguments, '--dry-run', '--apply']):
            with self.subTest(invalid=invalid), self.assertRaises(SystemExit):
                prune.main(invalid)
        with patch.object(prune, 'load_authority', side_effect=self._authority), \
                patch.object(prune, 'active_processes') as processes:
            with self.assertRaises(SystemExit):
                prune.main([*arguments, '--dry-run', '--terminate-blocking-sftp'])
            processes.assert_not_called()
        output = io.StringIO()
        with patch.object(prune, 'load_authority', side_effect=self._authority), \
                patch.object(prune, 'active_processes', return_value=[]), \
                redirect_stdout(output):
            self.assertEqual(0, prune.main([*arguments, '--dry-run']))
        summary = json.loads(output.getvalue())
        self.assertEqual(key, summary['spec_key'])
        self.assertEqual(21, summary['candidate_count'])
        self.assertFalse(summary['applied'])



if __name__ == '__main__':
    unittest.main()
