import hashlib
import ctypes
import errno
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import frozen_ablation_quiescent as q


class QuiescentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rows = [dict(job_key=f'{dataset}/{arm}/{seed}', dataset=dataset,
                          arm=arm, seed=seed, physical_gpu=0,
                          config_sha256='a' * 64)
                     for dataset in ('cifar100', 'isolet', 'upmc_food101')
                     for arm in ('no_partykd', 'fim_off')
                     for seed in (42, 43, 44)]
        self.ledger = self.root / 'ATTEMPTS.jsonl'
        self.ledger.write_bytes(b'')

    def append(self, event):
        with self.ledger.open('ab') as stream:
            stream.write((json.dumps(event, indent=None) + '\n').encode())

    def start(self, row):
        job = self.root / row['arm'] / row['dataset'] / str(row['seed'])
        job.mkdir(parents=True)
        (job / 'attempt-01.plan.json').write_text(json.dumps(
            {'job_key': row['job_key'], 'attempt': 1}))
        event = dict(kind='start', job_key=row['job_key'], attempt=1,
                     config_sha256=row['config_sha256'])
        if 'code_commit' in row:
            event['code_commit'] = row['code_commit']
        if 'gpu_uuid' in row:
            event.update(gpu_index=row['physical_gpu'], gpu_uuid=row['gpu_uuid'])
        self.append(event)
        return job

    def finish(self, row, job):
        run = job / 'attempt-01'
        run.mkdir()
        # Deliberately not JSON: metadata observation must hash, never parse it.
        gate, log = run / 'JOB_SUCCESS.json', job / 'attempt-01.log'
        gate.write_bytes(b'opaque synthetic success seal\n')
        log.write_bytes(b'opaque synthetic worker log\n')
        self.append(dict(kind='finish', job_key=row['job_key'], attempt=1,
                         classification='success', returncode=0, test_accessed=True,
                         job_success_sha256=hashlib.sha256(gate.read_bytes()).hexdigest(),
                         log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()))

    def test_pending_allocation_is_bounded_sorted_and_unique(self):
        isolet = sorted(row['job_key'] for row in self.rows if row['dataset'] == 'isolet')
        self.assertEqual(q.assign_pending(reversed(isolet)),
                         {key: i % 2 for i, key in enumerate(isolet)})
        mixed = q.assign_pending([*isolet, 'cifar100/no_partykd/43'])
        self.assertEqual(mixed['cifar100/no_partykd/43'], 1)
        self.assertTrue(all(mixed[key] == 0 for key in isolet))
        self.assertEqual(q.assign_pending([]), {})
        for keys in (['cifar100/no_partykd/42'], ['isolet/fim_off/45'],
                     ['isolet/fim_off/42'] * 2, ['../../escape']):
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                q.assign_pending(keys)

    def test_complete_raw_prefix_and_unfinished_are_distinct(self):
        row = self.rows[0]
        self.assertEqual(len(q.ledger_state(self.root, self.rows)['pending']), 18)
        job = self.start(row)
        state = q.ledger_state(self.root, self.rows)
        self.assertEqual(state['unfinished'], [row['job_key']])
        self.finish(row, job)
        raw = self.ledger.read_bytes()
        state = q.ledger_state(self.root, self.rows)
        self.assertEqual(state['success'], [row['job_key']])
        self.assertEqual(state['unfinished'], [])
        self.assertEqual(state['raw_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(state['raw_bytes_length'], len(raw))
        self.assertEqual(self.ledger.read_bytes(), raw)

    def test_failed_finish_is_not_success(self):
        row = self.rows[0]
        self.start(row)
        self.append(dict(kind='finish', job_key=row['job_key'], attempt=1,
                         classification='training_or_audit_failure', returncode=1))
        state = q.ledger_state(self.root, self.rows)
        self.assertEqual(state['failed'], [row['job_key']])
        self.assertEqual(state['unfinished'], [])

    def test_rejects_partial_blank_missing_duplicate_and_unknown_events(self):
        row = self.rows[0]
        event = dict(kind='start', job_key=row['job_key'], attempt=1,
                     config_sha256='a' * 64)
        cases = [b'{', b'\n', b'{}\n', b'{"kind":"start","kind":"finish"}\n',
                 (json.dumps({**event, 'job_key': 'unknown'}) + '\n').encode(),
                 (json.dumps({**event, 'attempt': 2}) + '\n').encode()]
        for raw in cases:
            self.ledger.write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                q.ledger_state(self.root, self.rows)
        self.ledger.write_bytes(b'')
        self.start(row)
        self.append(event)
        with self.assertRaises(ValueError):
            q.ledger_state(self.root, self.rows)
        self.ledger.unlink()
        with self.assertRaises(ValueError):
            q.ledger_state(self.root, self.rows)

    def test_rejects_finish_without_start_and_duplicate_finish(self):
        row = self.rows[0]
        event = dict(kind='finish', job_key=row['job_key'], attempt=1,
                     classification='infrastructure_candidate', returncode=1)
        self.append(event)
        with self.assertRaises(ValueError):
            q.ledger_state(self.root, self.rows)
        self.ledger.write_bytes(b'')
        self.start(row)
        self.append(event)
        self.append(event)
        with self.assertRaises(ValueError):
            q.ledger_state(self.root, self.rows)

    def test_rejects_manifest_duplicates_wrong_identity_and_hash_changes(self):
        for rows in (self.rows[:-1], [self.rows[0]] * 18,
                     [{**self.rows[0], 'seed': 1}, *self.rows[1:]]):
            with self.subTest(rows=rows[:1]), self.assertRaises(ValueError):
                q.ledger_state(self.root, rows)
        row = self.rows[0]
        job = self.start(row)
        self.finish(row, job)
        gate = job / 'attempt-01/JOB_SUCCESS.json'
        gate.write_bytes(gate.read_bytes() + b' ')
        with self.assertRaises(ValueError):
            q.ledger_state(self.root, self.rows)

    def test_rejects_wrong_config_unknown_directories_and_pending_plan(self):
        row = self.rows[0]
        job = self.start(row)
        raw = self.ledger.read_bytes()
        self.ledger.write_bytes(raw.replace(b'a' * 64, b'b' * 64))
        with self.assertRaises(ValueError):
            q.ledger_state(self.root, self.rows)
        self.ledger.write_bytes(raw)
        (job / 'attempt-02').mkdir()
        with self.assertRaises(ValueError):
            q.ledger_state(self.root, self.rows)
        (job / 'attempt-02').rmdir()
        self.ledger.write_bytes(b'')
        with self.assertRaises(ValueError):
            q.ledger_state(self.root, self.rows)
        # Unknown directories also fail closed when there is no ledger start.
        (self.root / 'unknown').mkdir()
        with self.assertRaises(ValueError):
            q.ledger_state(self.root, self.rows)

    def test_identity_requires_all_immutable_fields_but_not_process_state(self):
        expected = dict(pid=123, ppid=1, pgrp=123, session=123, state='S',
                        start_ticks=456, boot_id='synthetic-boot', cmdline=['python', 'old'])
        self.assertTrue(q.same_identity(expected, {**expected, 'state': 'T'}))
        for field, value in [('pid', 124), ('start_ticks', 457), ('boot_id', 'other'),
                             ('cmdline', ['python', 'new']), ('pgrp', 124),
                             ('session', 124), ('ppid', 2)]:
            self.assertFalse(q.same_identity(expected, {**expected, field: value}))
        self.assertFalse(q.same_identity({}, {}))

    def test_active_ledger_never_signals_or_reads_process_identity(self):
        self.start(self.rows[0])
        with mock.patch.object(q, 'identity', side_effect=AssertionError('must not inspect')):
            self.assertIsNone(q.try_quiescent_fence(self.root, self.rows, {}))

    def test_all_complete_is_not_a_handoff(self):
        for row in self.rows:
            self.finish(row, self.start(row))
        with mock.patch.object(q, 'identity', side_effect=AssertionError('must not inspect')):
            self.assertIsNone(q.try_quiescent_fence(self.root, self.rows, {}))

    def test_single_pending_candidate_has_no_parallel_gain_and_never_signals(self):
        for row in self.rows:
            if row['job_key'] != 'cifar100/no_partykd/43':
                self.finish(row, self.start(row))
        with mock.patch.object(q, 'identity', side_effect=AssertionError('must not inspect')):
            self.assertIsNone(q.try_quiescent_fence(self.root, self.rows, {}))

    def test_pidfd_open_uses_libc_when_python_has_no_wrapper(self):
        function = mock.Mock(return_value=42)
        with mock.patch.object(q.os, 'pidfd_open', None, create=True), mock.patch.object(
                q.ctypes, 'CDLL', return_value=mock.Mock(pidfd_open=function)):
            self.assertEqual(q._pidfd_open(123), 42)
        function.assert_called_once_with(123, 0)
        function.return_value = -1
        ctypes.set_errno(errno.ENOSYS)
        with mock.patch.object(q.os, 'pidfd_open', None, create=True), mock.patch.object(
                q.ctypes, 'CDLL', return_value=mock.Mock(pidfd_open=function)):
            with self.assertRaises(OSError):
                q._pidfd_open(123)

    def test_unknown_root_file_and_log_mutation_fail_closed(self):
        (self.root / 'rogue.plan').write_bytes(b'unknown')
        with self.assertRaises(ValueError):
            q.ledger_state(self.root, self.rows)
        (self.root / 'rogue.plan').unlink()
        row = self.rows[0]
        job = self.start(row)
        self.finish(row, job)
        (job / 'attempt-01.log').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            q.ledger_state(self.root, self.rows)

    def test_armed_extension_is_metadata_only_but_unknown_contents_rejected(self):
        extension = self.root / 'gpu1-rebalance-v1'
        extension.mkdir()
        (extension / 'ARMED.json').write_bytes(b'opaque arm evidence')
        self.assertEqual(len(q.ledger_state(self.root, self.rows)['pending']), 18)
        (extension / 'unexpected.plan').write_bytes(b'unknown')
        with self.assertRaises(ValueError):
            q.ledger_state(self.root, self.rows)

    def test_legacy_start_source_gpu_and_success_test_access_are_binding(self):
        row = self.rows[0]
        row.update(code_commit='b' * 40, gpu_uuid='synthetic-GPU0')
        job = self.start(row)
        self.finish(row, job)
        raw = self.ledger.read_bytes()
        events = [json.loads(line) for line in raw.splitlines()]
        for field, value in [('code_commit', 'wrong'), ('gpu_index', 1), ('gpu_uuid', 'other')]:
            changed = [{**events[0], field: value}, events[1]]
            self.ledger.write_bytes(b''.join((json.dumps(event) + '\n').encode() for event in changed))
            with self.subTest(field=field), self.assertRaises(ValueError):
                q.ledger_state(self.root, self.rows)
        self.ledger.write_bytes(b''.join((json.dumps(event) + '\n').encode() for event in
                                         [events[0], {**events[1], 'test_accessed': False}]))
        with self.assertRaises(ValueError):
            q.ledger_state(self.root, self.rows)
        self.ledger.write_bytes(raw)
        (job / 'attempt-01.plan.json').write_text(json.dumps(
            {'job_key': row['job_key'], 'attempt': 1, 'unexpected': 'claim'}))
        with self.assertRaises(ValueError):
            q.ledger_state(self.root, self.rows)

    def candidate_ledger(self):
        for row in self.rows:
            if row['dataset'] != 'isolet' and row['job_key'] != 'cifar100/no_partykd/43':
                self.finish(row, self.start(row))

    def fake_fence(self, inject):
        self.candidate_ledger()
        old = dict(pid=123456, ppid=1, pgrp=123456, session=123456, state='S',
                   start_ticks=100, boot_id='synthetic-boot', cmdline=['fake-old-parent'])
        state, sent = ['S'], []
        def send(fd, sig):
            self.assertEqual(fd, 999)
            sent.append(sig)
            state[0] = {signal.SIGSTOP: 'T', signal.SIGCONT: 'S', signal.SIGKILL: 'Z'}[sig]
        def threads(pid):
            inject(state[0])
            return ((pid, 100, state[0], ()), (pid + 1, 101, state[0], ()))
        with mock.patch.object(q.sys, 'platform', 'linux'), mock.patch.object(q, '_pidfd_open', return_value=999), \
                mock.patch.object(q.signal, 'SIGSTOP', 19, create=True), \
                mock.patch.object(q.signal, 'SIGCONT', 18, create=True), \
                mock.patch.object(q.signal, 'SIGKILL', 9, create=True), \
                mock.patch.object(q, 'identity', side_effect=lambda pid: {**old, 'state': state[0]}), \
                mock.patch.object(q, '_threads', side_effect=threads), \
                mock.patch.object(q.signal, 'pidfd_send_signal', side_effect=send, create=True), \
                mock.patch.object(q.os, 'close'), mock.patch.object(q.select, 'select', return_value=([999], [], [])):
            result = q.try_quiescent_fence(self.root, self.rows, old)
        return result, sent

    def test_new_epoch_race_does_not_resume_the_old_parent(self):
        extension = self.root / 'gpu1-rebalance-v1'
        extension.mkdir()
        def inject(state):
            if state == 'T':
                (extension / 'EXECUTION_EPOCH.json').write_bytes(b'other writer')
        with self.assertRaises(RuntimeError):
            self.fake_fence(inject)

    def test_lock_race_resumes_parent_without_retirement(self):
        lock = self.root / 'QUEUE_LOCK.json'
        lock.write_bytes(b'original lock')
        def inject(state):
            if state == 'T':
                lock.write_bytes(b'changed lock')
        result, sent = self.fake_fence(inject)
        self.assertIsNone(result)
        self.assertEqual(sent, [19, 18])


@unittest.skipUnless(sys.platform == 'linux', 'Linux /proc and pidfd signals required')
class LinuxQuiescentTests(QuiescentTests):
    def parent(self, child=False):
        code = '''import pathlib, subprocess, sys, threading, time
if sys.argv[2] == 'child':
    def worker():
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(1.0)'])
        pathlib.Path(sys.argv[1]).write_text(str(child.pid))
        child.wait()
    threading.Thread(target=worker, daemon=True).start()
else:
    threading.Thread(target=lambda: time.sleep(60), daemon=True).start()
pathlib.Path(sys.argv[1] + '.ready').write_text('ready')
time.sleep(60)
'''
        support = tempfile.TemporaryDirectory()
        self.addCleanup(support.cleanup)
        marker = Path(support.name) / 'synthetic-child'
        process = subprocess.Popen([sys.executable, '-c', code, str(marker),
                                    'child' if child else 'idle'], start_new_session=True)
        def cleanup():
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        self.addCleanup(cleanup)
        deadline = time.monotonic() + 5
        while not Path(str(marker) + '.ready').exists():
            if time.monotonic() > deadline:
                self.fail('synthetic parent did not become ready')
            time.sleep(0.01)
        if child:
            while not marker.exists():
                if time.monotonic() > deadline:
                    self.fail('synthetic child did not become ready')
                time.sleep(0.01)
        return process

    def test_linux_identity_and_bound_retirement_preserve_original_bytes(self):
        self.candidate_ledger()
        parent = self.parent()
        old = q.identity(parent.pid)
        self.assertEqual(old['pid'], parent.pid)
        self.assertEqual(old['ppid'], os.getpid())
        self.assertGreater(old['start_ticks'], 0)
        raw = self.ledger.read_bytes()
        receipt = q.try_quiescent_fence(self.root, self.rows, old)
        self.assertIsNotNone(receipt)
        parent.wait(timeout=5)
        self.assertEqual(parent.returncode, -signal.SIGKILL)
        self.assertEqual(receipt['ledger_prefix_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(receipt['ledger_prefix_bytes_length'], len(raw))
        self.assertEqual(self.ledger.read_bytes(), raw)
        self.assertEqual(len(receipt['pending_keys']), 7)
        self.assertIn(receipt['retired_identity']['state'], ('Z', 'X', 'gone'))

    def test_linux_nonleader_thread_child_is_never_signalled_or_adopted(self):
        self.candidate_ledger()
        parent = self.parent(child=True)
        old = q.identity(parent.pid)
        with mock.patch.object(q.signal, 'pidfd_send_signal', wraps=signal.pidfd_send_signal) as send:
            self.assertIsNone(q.try_quiescent_fence(self.root, self.rows, old))
            send.assert_not_called()
        self.assertIsNone(parent.poll())
        # Child is reaped normally by its original parent, then a later idle fence works.
        time.sleep(1.1)
        self.assertIsNotNone(q.try_quiescent_fence(self.root, self.rows, old))

    def test_linux_wrong_pid_identity_cannot_signal(self):
        self.candidate_ledger()
        parent = self.parent()
        old = q.identity(parent.pid)
        with mock.patch.object(q.signal, 'pidfd_send_signal') as send:
            with self.assertRaises(ValueError):
                q.try_quiescent_fence(self.root, self.rows,
                                      {**old, 'start_ticks': old['start_ticks'] + 1})
            send.assert_not_called()
        self.assertIsNone(parent.poll())

    def test_linux_ledger_race_after_stop_resumes_original_parent(self):
        self.candidate_ledger()
        parent = self.parent()
        old = q.identity(parent.pid)
        original = q.ledger_state
        calls = 0
        def race(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.ledger.write_bytes(self.ledger.read_bytes() + b'{')
            return original(*args)
        with mock.patch.object(q, 'ledger_state', side_effect=race):
            self.assertIsNone(q.try_quiescent_fence(self.root, self.rows, old))
        deadline = time.monotonic() + 2
        while q.identity(parent.pid)['state'] == 'T' and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertNotEqual(q.identity(parent.pid)['state'], 'T')
        self.assertIsNone(parent.poll())


if __name__ == '__main__':
    unittest.main()
