import json
import io
import tempfile
import threading
import unittest
from pathlib import Path

import party_drift_multihost as multi
import party_drift_remote_broker as remote_broker


KEY = 'cifar100/sample_only_off/none/70'


class Broker:
    def __init__(self, pid=41002):
        self.pid = pid
        self.go = False
        self.report = {'pid': pid, 'host': '40902', 'producer_commit': 'p',
                       'scheduler_commit': 's', 'manifest_sha256': 'm',
                       'mount_source': 'shared:/brf',
                       'result_disk_free_mib': 5000,
                       'scratch_path': '/tmp', 'scratch_free_mib': 5000,
                       'driver': '575',
                       'python': '3.12.3', 'torch': '2.7.1',
                       'torch_cuda': '12.8',
                       'gpu': {'name': 'NVIDIA GeForce RTX 4090',
                               'uuid': 'GPU-remote', 'free_mib': 18000,
                               'utilization': 1, 'compute_pids': []}}

    def receive_report(self):
        return self.report

    def send_go(self):
        self.go = True

    def alive(self):
        return True


class MultihostTests(unittest.TestCase):
    def fixture(self, root):
        job = root / KEY
        job.mkdir(parents=True)
        events = root / 'ATTEMPTS.jsonl'
        events.write_text('')
        approved = root / 'approved.json'
        approved.write_text('{"seed":70}')
        return job, events, approved

    def test_remote_handoff_writes_parent_lock_plan_start_before_go_and_restores_only_after_ack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, events, approved = self.fixture(root)
            gate = multi.HandoffGate(root, events, [KEY], manifest_sha256='m',
                                     producer_commit='p', scheduler_commit='s')
            broker = Broker()

            def ack():
                self.assertTrue(broker.go)
                self.assertEqual(multi.active_lock(root)['pid'], 41002)
                self.assertEqual(json.loads((job / 'attempt-01.plan.json').read_text()),
                                 {'attempt': 1, 'job_key': KEY})
                start = json.loads(events.read_text().splitlines()[0])
                self.assertEqual(start['kind'], 'start')
                self.assertEqual(start['physical_gpu_uuid'], 'GPU-remote')
                attempt = job / 'attempt-01'
                attempt.mkdir()
                (attempt / 'APPROVED_CONFIG.json').write_bytes(approved.read_bytes())
                (attempt / 'JOB_RUNTIME.json').write_text(json.dumps({
                    'job_key': KEY, 'attempt': 1, 'manifest_sha256': 'm',
                    'config_sha256': multi.sha256(approved),
                    'production_commit': 'p', 'actual_output_dir': str(attempt),
                    'test_allowed': False}))

            gate.authorize_remote(KEY, job, approved, broker,
                                  gpu_uuid='GPU-remote', gpu_snapshot={
                                      'name': 'NVIDIA GeForce RTX 4090',
                                      'uuid': 'GPU-remote', 'free_mib': 18000,
                                      'utilization': 1, 'compute_pids': []},
                                  ack_wait=ack,
                                  fresh_snapshot=lambda: broker.report['gpu'])
            self.assertEqual(multi.active_lock(root)['pid'], multi.os.getpid())
            self.assertEqual(gate.pending, [])

    def test_remote_ack_failure_is_fail_stop_and_does_not_reclaim_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, events, approved = self.fixture(root)
            gate = multi.HandoffGate(root, events, [KEY], manifest_sha256='m',
                                     producer_commit='p', scheduler_commit='s')
            with self.assertRaises(multi.FailStop):
                gate.authorize_remote(KEY, job, approved, Broker(),
                                      gpu_uuid='GPU-remote', gpu_snapshot={
                                          'name': 'NVIDIA GeForce RTX 4090',
                                          'uuid': 'GPU-remote', 'free_mib': 18000,
                                          'utilization': 1, 'compute_pids': []},
                                      ack_wait=lambda: None,
                                      ack_seconds=0,
                                      fresh_snapshot=lambda: Broker().report['gpu'])
            self.assertEqual(multi.active_lock(root)['pid'], 41002)
            self.assertEqual(len(events.read_text().splitlines()), 1)
            with self.assertRaises(multi.FailStop):
                gate.claim()
            with self.assertRaises(multi.FailStop):
                gate.append({'kind': 'finish'})

    def test_local_append_waits_for_ack_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, events, approved = self.fixture(root)
            gate = multi.HandoffGate(root, events, [KEY], manifest_sha256='m',
                                     producer_commit='p', scheduler_commit='s')
            entered = threading.Event()
            release = threading.Event()
            finished = threading.Event()

            def ack():
                entered.set()
                self.assertTrue(release.wait(3))
                attempt = job / 'attempt-01'
                attempt.mkdir()
                (attempt / 'APPROVED_CONFIG.json').write_bytes(approved.read_bytes())
                (attempt / 'JOB_RUNTIME.json').write_text(json.dumps({
                    'job_key': KEY, 'attempt': 1, 'manifest_sha256': 'm',
                    'config_sha256': multi.sha256(approved),
                    'production_commit': 'p', 'actual_output_dir': str(attempt),
                    'test_allowed': False}))

            remote = threading.Thread(target=lambda: gate.authorize_remote(
                KEY, job, approved, Broker(), gpu_uuid='GPU-remote',
                gpu_snapshot={'name': 'NVIDIA GeForce RTX 4090',
                              'uuid': 'GPU-remote', 'free_mib': 18000,
                              'utilization': 1, 'compute_pids': []}, ack_wait=ack,
                fresh_snapshot=lambda: Broker().report['gpu']))
            remote.start()
            self.assertTrue(entered.wait(3))
            local = threading.Thread(target=lambda: (gate.append({
                'kind': 'finish', 'job_key': 'local', 'attempt': 1}),
                finished.set()))
            local.start()
            self.assertFalse(finished.wait(0.1))
            release.set()
            remote.join(3)
            local.join(3)
            self.assertTrue(finished.is_set())

    def test_local_launch_keeps_remote_lock_switch_out_until_local_ack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = multi.HandoffGate(root, root / 'ATTEMPTS.jsonl', [KEY],
                manifest_sha256='m', producer_commit='p', scheduler_commit='s')
            entered = threading.Event()
            release = threading.Event()
            switched = threading.Event()

            def ack(_process):
                entered.set()
                self.assertTrue(release.wait(3))

            local = threading.Thread(target=lambda: gate.local_launch(
                lambda: object(), ack_wait=ack))
            local.start()
            self.assertTrue(entered.wait(3))
            remote = threading.Thread(target=lambda: (gate.peek_remote({KEY}),
                                                      switched.set()))
            remote.start()
            self.assertFalse(switched.wait(0.1))
            release.set()
            local.join(3)
            remote.join(3)
            self.assertTrue(switched.is_set())

    def test_ack_wait_ignores_empty_and_half_written_files_until_exact_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, events, approved = self.fixture(root)
            attempt = job / 'attempt-01'
            attempt.mkdir()
            gate = multi.HandoffGate(root, events, [KEY], manifest_sha256='m',
                                     producer_commit='p', scheduler_commit='s')
            now = [0]
            expected = {'job_key': KEY, 'attempt': 1, 'manifest_sha256': 'm',
                'config_sha256': multi.sha256(approved),
                'production_commit': 'p', 'actual_output_dir': str(attempt),
                'test_allowed': False}
            class Alive:
                def alive(self):
                    return True
            def advance(seconds):
                now[0] += seconds
                if now[0] == 1:
                    (attempt / 'APPROVED_CONFIG.json').write_bytes(b'')
                    (attempt / 'JOB_RUNTIME.json').write_text('{')
                elif now[0] == 2:
                    (attempt / 'APPROVED_CONFIG.json').write_bytes(approved.read_bytes())
                    (attempt / 'JOB_RUNTIME.json').write_text('{"job_key":')
                elif now[0] == 3:
                    (attempt / 'JOB_RUNTIME.json').write_text(json.dumps(expected))
            gate._wait_ack(Alive(), attempt, approved, KEY, seconds=5,
                clock=lambda: now[0], sleep=advance)
            self.assertEqual(now[0], 3)

    def test_ack_wait_fails_if_worker_dies_before_complete_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, events, approved = self.fixture(root)
            attempt = job / 'attempt-01'
            attempt.mkdir()
            (attempt / 'APPROVED_CONFIG.json').write_bytes(b'')
            (attempt / 'JOB_RUNTIME.json').write_text('{')
            gate = multi.HandoffGate(root, events, [KEY], manifest_sha256='m',
                                     producer_commit='p', scheduler_commit='s')
            class Dead:
                def alive(self):
                    return False
            with self.assertRaises(multi.FailStop):
                gate._wait_ack(Dead(), attempt, approved, KEY, seconds=5)

    def test_remote_admission_rejects_busy_or_wrong_gpu_and_uuid(self):
        good = {'name': 'NVIDIA GeForce RTX 4090', 'uuid': 'GPU-remote',
                'free_mib': 18000, 'utilization': 1, 'compute_pids': []}
        multi.validate_remote_gpu(good, 'GPU-remote')
        for changes in ({'utilization': -1}, {'utilization': 101},
                        {'free_mib': 100},
                        {'uuid': 'GPU-other'}, {'name': 'NVIDIA GeForce RTX 3080'}):
            with self.assertRaises(ValueError):
                multi.validate_remote_gpu({**good, **changes}, 'GPU-remote')

    def test_remote_report_rejects_any_identity_or_environment_mismatch(self):
        broker = Broker()
        multi.validate_broker_report(broker.report, host='40902',
                                     mount_source='shared:/brf',
                                     manifest_sha256='m', producer_commit='p',
                                     scheduler_commit='s', gpu_uuid='GPU-remote')
        for field, bad in [('pid', 0), ('host', 'other'),
                           ('producer_commit', 'other'),
                           ('scheduler_commit', 'other'),
                           ('manifest_sha256', 'other'),
                           ('mount_source', 'other'),
                           ('result_disk_free_mib', 100),
                           ('scratch_path', ''), ('scratch_free_mib', 100),
                           ('driver', ''),
                           ('python', ''), ('torch', ''), ('torch_cuda', '')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                multi.validate_broker_report({**broker.report, field: bad},
                    host='40902', mount_source='shared:/brf',
                    manifest_sha256='m', producer_commit='p',
                    scheduler_commit='s',
                    gpu_uuid='GPU-remote')

    def test_go_requires_fresh_gpu_snapshot_after_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, events, approved = self.fixture(root)
            gate = multi.HandoffGate(root, events, [KEY], manifest_sha256='m',
                                     producer_commit='p', scheduler_commit='s')
            broker = Broker()
            with self.assertRaises(multi.NotAdmitted):
                gate.authorize_remote(KEY, job, approved, broker,
                    gpu_uuid='GPU-remote', gpu_snapshot=broker.report['gpu'],
                    fresh_snapshot=lambda: {**broker.report['gpu'],
                                            'free_mib': 100},
                    expected_host='40902', expected_mount_source='shared:/brf')
            self.assertFalse(broker.go)
            self.assertEqual(events.read_text(), '')

    def test_remote_gpu_requires_continuous_two_minute_window(self):
        now = [0]
        snapshot = {'name': 'NVIDIA GeForce RTX 4090', 'uuid': 'GPU-remote',
                    'free_mib': 18000, 'utilization': 1, 'compute_pids': []}
        def read():
            return {**snapshot, 'free_mib': 100 if now[0] == 60 else 18000,
                    'utilization': 90}
        result = multi.wait_remote_ready(
            read, 'GPU-remote', clock=lambda: now[0],
            sleep=lambda n: now.__setitem__(0, now[0] + n))
        self.assertEqual(now[0], 210)
        self.assertEqual(result['uuid'], 'GPU-remote')

    def test_missing_broker_heartbeat_is_optional_before_assignment(self):
        now = [0]
        good = {'name': 'NVIDIA GeForce RTX 4090', 'uuid': 'GPU-remote',
                'free_mib': 7000, 'utilization': 90, 'compute_pids': []}
        def read():
            if now[0] < 30:
                raise ValueError('broker not yet present')
            return good
        multi.wait_remote_ready(read, 'GPU-remote', clock=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
        self.assertEqual(now[0], 150)

    def test_broker_protocol_reads_report_and_sends_go(self):
        incoming = io.StringIO('{"pid":41002,"host":"40902"}\n')
        outgoing = io.StringIO()
        broker = multi.JsonBroker(incoming, outgoing, lambda: None)
        self.assertEqual(broker.receive_report()['pid'], 41002)
        broker.send_go()
        self.assertEqual(outgoing.getvalue(), 'GO\n')

    def test_lock_sequence_stops_before_lexicographic_wrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'QUEUE_LOCK_RESUME_99.json').write_text('{}')
            with self.assertRaises(multi.FailStop):
                multi.next_lock(root)

    def test_broker_waits_for_go_then_becomes_worker_direct_parent(self):
        output = io.StringIO()
        launches = []
        class Input:
            def readline(self):
                report = json.loads(output.getvalue().splitlines()[0])
                self_outer.assertEqual(report['pid'], multi.os.getpid())
                self_outer.assertEqual(launches, [])
                return 'GO\n'
        self_outer = self
        class Process:
            def wait(self):
                return 0
        def spawn(command, **kwargs):
            launches.append((command, kwargs))
            return Process()
        code = remote_broker.broker_once(
            ['python', 'run_party_drift_development.py'], {'CUDA_VISIBLE_DEVICES': 'GPU-r'},
            Input(), output, spawn=spawn, report={'host': '40902'})
        self.assertEqual(code, 0)
        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0][1]['env']['CUDA_VISIBLE_DEVICES'], 'GPU-r')

    def test_broker_selects_nfs_source_from_automount_double_record(self):
        output = ('autofs systemd-1 /brf\n'
                  'nfs4 10.233.0.20:/srv/BRF/brf-gpu /brf\n')
        self.assertEqual(remote_broker.nfs_mount_source(output),
                         '10.233.0.20:/srv/BRF/brf-gpu')
        with self.assertRaises(ValueError):
            remote_broker.nfs_mount_source('autofs systemd-1 /brf\n')


if __name__ == '__main__':
    unittest.main()
