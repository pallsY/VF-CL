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

    def receive_pid(self):
        return self.pid

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
                                  ack_wait=ack)
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
                                      ack_wait=lambda: None)
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
                              'utilization': 1, 'compute_pids': []}, ack_wait=ack))
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

    def test_remote_admission_rejects_busy_or_wrong_gpu_and_uuid(self):
        good = {'name': 'NVIDIA GeForce RTX 4090', 'uuid': 'GPU-remote',
                'free_mib': 18000, 'utilization': 1, 'compute_pids': []}
        multi.validate_remote_gpu(good, 'GPU-remote')
        for changes in ({'utilization': 90}, {'free_mib': 100},
                        {'uuid': 'GPU-other'}, {'name': 'NVIDIA GeForce RTX 3080'}):
            with self.assertRaises(ValueError):
                multi.validate_remote_gpu({**good, **changes}, 'GPU-remote')

    def test_remote_gpu_requires_continuous_two_minute_window(self):
        now = [0]
        snapshot = {'name': 'NVIDIA GeForce RTX 4090', 'uuid': 'GPU-remote',
                    'free_mib': 18000, 'utilization': 1, 'compute_pids': []}
        def read():
            return {**snapshot, 'utilization': 90 if now[0] == 60 else 1}
        result = multi.wait_remote_ready(
            read, 'GPU-remote', clock=lambda: now[0],
            sleep=lambda n: now.__setitem__(0, now[0] + n))
        self.assertEqual(now[0], 210)
        self.assertEqual(result['uuid'], 'GPU-remote')

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


if __name__ == '__main__':
    unittest.main()
