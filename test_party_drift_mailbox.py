import json
import threading
import time
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import party_drift_mailbox as mailbox
import party_drift_remote_broker as remote_broker


class MailboxTests(unittest.TestCase):
    def test_assignment_report_go_exit_are_exclusive_and_bound_to_one_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = root / 'control'
            control.mkdir()
            broker = mailbox.FileBroker(control, 1, result_root=root / 'results',
                                        key='cifar100/static/0.25/70',
                                        manifest_sha256='m', producer_commit='p',
                                        config_sha256='c')
            broker.assign()
            assignment = control / 'ASSIGN_000001.json'
            self.assertTrue(assignment.is_file())
            with self.assertRaises(FileExistsError):
                broker.assign()
            mailbox.replace_current(control / 'HEARTBEAT_CURRENT.json',
                               {'protocol': 1, 'pid': 41002, 'sequence': 1,
                                'gpu': {'free_mib': 10000}})
            mailbox.write_once(control / 'READY_000001.json', {
                'protocol': 1, 'sequence': 1,
                'job_key': 'cifar100/static/0.25/70',
                'manifest_sha256': 'm', 'production_commit': 'p',
                'scheduler_commit': 's',
                'assignment_sha256': mailbox.sha256(assignment),
                'report': {'pid': 41002, 'host': '40902'}})
            self.assertEqual(broker.receive_report()['pid'], 41002)
            broker.send_go()
            go = json.loads((control / 'GO_000001.json').read_text())
            self.assertEqual(go['assignment_sha256'], mailbox.sha256(assignment))
            mailbox.write_once(control / 'LAUNCH_CHECK_000001.json', {
                'protocol': 1, 'sequence': 1,
                'job_key': 'cifar100/static/0.25/70',
                'manifest_sha256': 'm', 'production_commit': 'p',
                'scheduler_commit': 's',
                'assignment_sha256': mailbox.sha256(assignment),
                'ready_sha256': mailbox.sha256(control / 'READY_000001.json'),
                'gpu': {'name': 'NVIDIA GeForce RTX 4090',
                        'uuid': 'GPU-r', 'free_mib': 7000,
                        'utilization': 90, 'compute_pids': []}})
            self.assertEqual(broker.validate_launch_check('GPU-r')['free_mib'], 7000)
            mailbox.write_once(control / 'EXIT_000001.json', {
                'protocol': 1, 'sequence': 1,
                'job_key': 'cifar100/static/0.25/70',
                'manifest_sha256': 'm', 'production_commit': 'p',
                'scheduler_commit': 's',
                'assignment_sha256': mailbox.sha256(assignment),
                'returncode': 0})
            self.assertEqual(broker.wait_exit(), 0)

    def test_mailbox_file_becomes_visible_only_after_complete_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'MESSAGE.json'
            writing = threading.Event()
            release = threading.Event()
            original = json.dump
            def delayed(value, stream, **kwargs):
                stream.write('{')
                stream.flush()
                writing.set()
                self.assertTrue(release.wait(3))
                stream.seek(0)
                stream.truncate()
                original(value, stream, **kwargs)
            with patch.object(mailbox.json, 'dump', side_effect=delayed):
                thread = threading.Thread(target=lambda: mailbox.write_once(
                    path, {'complete': True}))
                thread.start()
                try:
                    self.assertTrue(writing.wait(3))
                    self.assertFalse(path.exists())
                finally:
                    release.set()
                    thread.join(3)
            self.assertEqual(json.loads(path.read_text()), {'complete': True})

    def test_current_heartbeat_keeps_previous_complete_value_during_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'HEARTBEAT_CURRENT.json'
            mailbox.replace_current(path, {'sequence': 99})
            writing = threading.Event()
            release = threading.Event()
            original = json.dump
            def delayed(value, stream, **kwargs):
                stream.write('{')
                stream.flush()
                writing.set()
                self.assertTrue(release.wait(3))
                stream.seek(0)
                stream.truncate()
                original(value, stream, **kwargs)
            with patch.object(mailbox.json, 'dump', side_effect=delayed):
                thread = threading.Thread(target=lambda: mailbox.replace_current(
                    path, {'sequence': 100}))
                thread.start()
                try:
                    self.assertTrue(writing.wait(3))
                    self.assertEqual(json.loads(path.read_text())['sequence'], 99)
                finally:
                    release.set()
                    thread.join(3)
            self.assertEqual(json.loads(path.read_text())['sequence'], 100)

    def test_tampered_ready_or_stale_heartbeat_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = root / 'control'
            control.mkdir()
            broker = mailbox.FileBroker(control, 1, result_root=root / 'results',
                                        key='k', manifest_sha256='m',
                                        producer_commit='p', config_sha256='c',
                                        clock=lambda: 100, heartbeat_seconds=10)
            broker.assign()
            mailbox.replace_current(control / 'HEARTBEAT_CURRENT.json',
                               {'protocol': 1, 'pid': 41002,
                                'sequence': 1, 'gpu': {}})
            mailbox.write_once(control / 'READY_000001.json', {
                'protocol': 1, 'sequence': 1, 'job_key': 'k',
                'manifest_sha256': 'm', 'production_commit': 'p',
                'scheduler_commit': 's',
                'assignment_sha256': 'tampered', 'report': {'pid': 41002}})
            with self.assertRaises(mailbox.MailboxError):
                broker.receive_report()
            broker.clock = lambda: 111
            self.assertFalse(broker.alive())

    def test_default_heartbeat_allows_nfs_delay_below_90_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = [0]
            broker = mailbox.FileBroker(root, 1, result_root=root / 'results',
                key='k', manifest_sha256='m', producer_commit='p',
                config_sha256='c', clock=lambda: now[0])
            mailbox.replace_current(root / 'HEARTBEAT_CURRENT.json', {
                'protocol': 1, 'sequence': 1, 'pid': 41002, 'gpu': {}})
            self.assertTrue(broker.alive())
            now[0] = 80
            self.assertTrue(broker.alive())
            now[0] = 91
            self.assertFalse(broker.alive())

    def test_control_directory_must_stay_outside_result_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                mailbox.FileBroker(root / 'results' / 'control', 1,
                    result_root=root / 'results', key='k',
                    manifest_sha256='m', producer_commit='p', config_sha256='c')

    def test_remote_broker_waits_for_bound_go_before_spawning_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = root / 'control'
            control.mkdir()
            manifest = root / 'manifest.jsonl'
            key = 'cifar100/static/0.25/70'
            manifest.write_text(json.dumps({'job_key': key,
                                            'config_sha256': 'c'}) + '\n')
            central = mailbox.FileBroker(control, 1, result_root=root / 'results',
                key=key, manifest_sha256=mailbox.sha256(manifest),
                producer_commit='p', config_sha256='c')
            central.assign()
            result_root = root / 'results'
            job = result_root / key
            job.mkdir(parents=True)
            (result_root / 'QUEUE_LOCK_RESUME_01.json').write_text(json.dumps({
                'pid': remote_broker.os.getpid(),
                'manifest_sha256': mailbox.sha256(manifest),
                'production_commit': 'p'}))
            (job / 'attempt-01.plan.json').write_text(json.dumps({
                'attempt': 1, 'job_key': key}))
            (result_root / 'ATTEMPTS.jsonl').write_text(json.dumps({
                'kind': 'start', 'job_key': key, 'attempt': 1,
                'config_sha256': 'c'}) + '\n')
            launched = []
            class Process:
                def wait(self):
                    return 0
            def run():
                return remote_broker.broker_mailbox_once(
                    control, 1, report={'host': '40902'}, manifest=manifest,
                    manifest_sha256=mailbox.sha256(manifest),
                    producer_commit='p', command=['python', 'worker.py'],
                    env={}, log_path=root / 'attempt.log',
                    result_root=result_root, gpu_uuid='GPU-r',
                    snapshot=lambda: {'name': 'NVIDIA GeForce RTX 4090',
                        'uuid': 'GPU-r', 'free_mib': 7000,
                        'utilization': 90, 'compute_pids': []},
                    spawn=lambda command, **kw: launched.append(command) or Process())
            result = []
            thread = threading.Thread(target=lambda: result.append(run()))
            thread.start()
            ready = control / 'READY_000001.json'
            for _ in range(100):
                if ready.is_file():
                    break
                time.sleep(0.01)
            self.assertTrue(ready.is_file())
            self.assertEqual(launched, [])
            mailbox.write_once(control / 'GO_000001.json', {
                'protocol': 1, 'sequence': 1, 'job_key': key,
                'manifest_sha256': mailbox.sha256(manifest),
                'production_commit': 'p', 'scheduler_commit': 's',
                'assignment_sha256': mailbox.sha256(control / 'ASSIGN_000001.json'),
                'ready_sha256': mailbox.sha256(ready)})
            thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(result, [0])
            self.assertEqual(len(launched), 1)

    def test_heartbeat_current_replaces_atomically_with_increasing_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = Path(tmp)
            writer = remote_broker.HeartbeatWriter(
                control, pid=41002, snapshot=lambda: {'uuid': 'GPU-r',
                                                       'free_mib': 7000})
            writer.sequence = 99
            writer.pulse()
            current = control / 'HEARTBEAT_CURRENT.json'
            self.assertEqual(json.loads(current.read_text())['sequence'], 100)
            writer.pulse()
            self.assertEqual(json.loads(current.read_text())['sequence'], 101)
            self.assertEqual(json.loads(current.read_text())['gpu']['free_mib'], 7000)
            self.assertEqual(len(list(control.glob('HEARTBEAT*'))), 1)

    def test_cancel_before_go_never_spawns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = root / 'control'
            control.mkdir()
            manifest = root / 'manifest.jsonl'
            manifest.write_text('{"job_key":"k","config_sha256":"c"}\n')
            central = mailbox.FileBroker(control, 1, result_root=root / 'results',
                key='k', manifest_sha256=mailbox.sha256(manifest),
                producer_commit='p', config_sha256='c')
            central.assign()
            launched = []
            result = []
            thread = threading.Thread(target=lambda: result.append(
                remote_broker.broker_mailbox_once(control, 1,
                    report={'host': '40902'}, manifest=manifest,
                    manifest_sha256=mailbox.sha256(manifest),
                    producer_commit='p', command=['worker'], env={},
                    log_path=root / 'log', result_root=root / 'results',
                    gpu_uuid='GPU-r',
                    snapshot=lambda: {'name': 'NVIDIA GeForce RTX 4090',
                        'uuid': 'GPU-r', 'free_mib': 7000,
                        'utilization': 90, 'compute_pids': []},
                    spawn=lambda *_a, **_k: launched.append(1))))
            thread.start()
            ready = control / 'READY_000001.json'
            for _ in range(100):
                if ready.is_file():
                    break
                time.sleep(0.01)
            self.assertTrue(ready.is_file())
            central.cancel()
            thread.join(3)
            self.assertEqual(result, [None])
            self.assertEqual(launched, [])

    def test_gpu_thermal_audit_parses_temperature_clock_and_slowdown(self):
        self.assertEqual(remote_broker.parse_thermal_audit('85, 750, Active'), {
            'temperature_c': 85, 'sm_clock_mhz': 750,
            'sw_thermal_slowdown': True})
        self.assertFalse(remote_broker.parse_thermal_audit(
            '70, 1905, Not Active')['sw_thermal_slowdown'])

    def test_broker_checks_its_pid_plan_and_complete_start_before_spawn(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'results'
            key = 'cifar100/static/0.25/70'
            job = root / key
            job.mkdir(parents=True)
            assignment = {'job_key': key, 'manifest_sha256': 'm',
                          'production_commit': 'p', 'config_sha256': 'c'}
            lock = root / 'QUEUE_LOCK_RESUME_01.json'
            plan = job / 'attempt-01.plan.json'
            events = root / 'ATTEMPTS.jsonl'
            lock.write_text(json.dumps({'pid': 41002,
                'manifest_sha256': 'm', 'production_commit': 'p'}))
            plan.write_text(json.dumps({'attempt': 1, 'job_key': key}))
            events.write_text(json.dumps({'kind': 'start', 'job_key': key,
                'attempt': 1, 'config_sha256': 'c'}) + '\n')
            remote_broker.verify_remote_authorization(root, assignment, 41002)
            lock.unlink()
            with self.assertRaises(ValueError):
                remote_broker.verify_remote_authorization(root, assignment, 41002)
            lock.write_text(json.dumps({'pid': 41002,
                'manifest_sha256': 'm', 'production_commit': 'p'}))
            plan.unlink()
            with self.assertRaises(ValueError):
                remote_broker.verify_remote_authorization(root, assignment, 41002)
            plan.write_text(json.dumps({'attempt': 1, 'job_key': key}))
            events.write_text('{"kind":"start"')
            with self.assertRaises(ValueError):
                remote_broker.verify_remote_authorization(root, assignment, 41002)

    def test_go_time_gpu_loss_writes_abort_and_never_spawns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = root / 'control'
            control.mkdir()
            manifest = root / 'manifest.jsonl'
            manifest.write_text('{"job_key":"k","config_sha256":"c"}\n')
            central = mailbox.FileBroker(control, 1, result_root=root / 'results',
                key='k', manifest_sha256=mailbox.sha256(manifest),
                producer_commit='p', config_sha256='c')
            central.assign()
            launched = []
            errors = []
            def run():
                try:
                    remote_broker.broker_mailbox_once(control, 1,
                        report={'host': '40902'}, manifest=manifest,
                        manifest_sha256=mailbox.sha256(manifest),
                        producer_commit='p', command=['worker'], env={},
                        log_path=root / 'log', gpu_uuid='GPU-r',
                        result_root=root / 'results',
                        snapshot=lambda: {'name': 'NVIDIA GeForce RTX 4090',
                                          'uuid': 'GPU-r', 'free_mib': 100,
                                          'utilization': 1, 'compute_pids': []},
                        spawn=lambda *_a, **_k: launched.append(1))
                except RuntimeError as error:
                    errors.append(str(error))
            thread = threading.Thread(target=run)
            thread.start()
            ready = control / 'READY_000001.json'
            for _ in range(100):
                if ready.is_file():
                    break
                time.sleep(0.01)
            self.assertTrue(ready.is_file())
            central.ready_sha256 = mailbox.sha256(ready)
            central.send_go()
            thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertTrue((control / 'ABORT_000001.json').is_file())
            self.assertEqual(launched, [])
            self.assertTrue(errors)

    def test_go_time_missing_plan_writes_abort_and_never_spawns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = root / 'control'
            control.mkdir()
            manifest = root / 'manifest.jsonl'
            key = 'cifar100/static/0.25/70'
            manifest.write_text(json.dumps({'job_key': key,
                                            'config_sha256': 'c'}) + '\n')
            result_root = root / 'results'
            result_root.mkdir()
            (result_root / 'QUEUE_LOCK_RESUME_01.json').write_text(json.dumps({
                'pid': remote_broker.os.getpid(),
                'manifest_sha256': mailbox.sha256(manifest),
                'production_commit': 'p'}))
            (result_root / 'ATTEMPTS.jsonl').write_text(json.dumps({
                'kind': 'start', 'job_key': key, 'attempt': 1,
                'config_sha256': 'c'}) + '\n')
            central = mailbox.FileBroker(control, 1, result_root=result_root,
                key=key, manifest_sha256=mailbox.sha256(manifest),
                producer_commit='p', config_sha256='c')
            central.assign()
            launched = []
            errors = []
            def run():
                try:
                    remote_broker.broker_mailbox_once(control, 1,
                        report={'host': '40902'}, manifest=manifest,
                        manifest_sha256=mailbox.sha256(manifest),
                        producer_commit='p', command=['worker'], env={},
                        log_path=root / 'log', result_root=result_root,
                        gpu_uuid='GPU-r', snapshot=lambda: {
                            'name': 'NVIDIA GeForce RTX 4090',
                            'uuid': 'GPU-r', 'free_mib': 7000,
                            'utilization': 1, 'compute_pids': []},
                        spawn=lambda *_a, **_k: launched.append(1))
                except RuntimeError as error:
                    errors.append(str(error))
            thread = threading.Thread(target=run)
            thread.start()
            ready = control / 'READY_000001.json'
            for _ in range(100):
                if ready.is_file():
                    break
                time.sleep(0.01)
            self.assertTrue(ready.is_file())
            central.ready_sha256 = mailbox.sha256(ready)
            central.send_go()
            thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertTrue((control / 'ABORT_000001.json').is_file())
            self.assertEqual(launched, [])
            self.assertTrue(errors)


if __name__ == '__main__':
    unittest.main()
