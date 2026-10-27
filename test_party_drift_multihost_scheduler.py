import tempfile
import json
import time
import os
import unittest
from unittest.mock import patch
from pathlib import Path

import party_drift_multihost as multi
import party_drift_multihost_resume as resume
import party_drift_remote_broker as remote_broker
from test_party_drift_balanced_resume import sealed_job
from party_drift_mailbox import sha256 as file_sha


def key(arm, alpha, seed):
    return f'cifar100/{arm}/{alpha}/{seed}'


class SchedulerTests(unittest.TestCase):
    def test_fixed_remote_selection_preserves_pairs_and_ignores_scores(self):
        sample = [key('sample_only_off', 'none', seed) for seed in (70, 72)]
        pair = [key(arm, '0.25', 70) for arm in ('static', 'uniform')]
        pending = sample + pair + [key('static', '0.5', 70)]
        self.assertEqual(multi.select_remote_keys(pending), sample + pair)
        self.assertEqual(multi.select_remote_keys(pending[2:]), pair)
        with self.assertRaises(ValueError):
            multi.validate_remote_key_anchor(pending, sample)

    def test_external_anchor_must_fix_remote_keys_and_environment(self):
        pending = [key('sample_only_off', 'none', 70)]
        anchor = {'remote_job_keys': pending,
                  'remote': {'host': '40902', 'gpu_uuid': 'GPU-r',
                             'mount_source': 'shared:/brf',
                             'environment': {'driver': '575', 'python': '3.12',
                                             'torch': '2.7',
                                             'torch_cuda': '12.8'}}}
        resume.validate_remote_anchor(anchor, pending)
        with self.assertRaises(ValueError):
            resume.validate_remote_anchor({**anchor, 'remote_job_keys': []},
                                          pending)
        with self.assertRaises(ValueError):
            resume.validate_remote_anchor({**anchor, 'remote': {
                **anchor['remote'], 'gpu_uuid': ''}}, pending)

    def test_prepared_control_is_private_and_bound_before_queue_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = root / 'anchor.json'
            anchor.write_text('{"remote_job_keys":[]}')
            control = root / 'control'
            digest = multi.sha256(anchor)
            resume.prepare_control(control, anchor, digest,
                result_root=root / 'results', manifest_sha256='m',
                producer_commit='p', scheduler_commit='s')
            resume.validate_control(control, digest,
                manifest_sha256='m', producer_commit='p', scheduler_commit='s')
            with self.assertRaises(ValueError):
                resume.validate_control(control, 'wrong',
                    manifest_sha256='m', producer_commit='p', scheduler_commit='s')

    def test_prepare_control_rejects_result_subdirectory_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'results'
            root.mkdir()
            anchor = Path(tmp) / 'anchor.json'
            anchor.write_text('{}')
            control = root / 'control'
            with self.assertRaises(ValueError):
                resume.prepare_control(control, anchor, multi.sha256(anchor),
                    result_root=root, manifest_sha256='m',
                    producer_commit='p', scheduler_commit='s')
            self.assertFalse(control.exists())

    def test_lock_budget_reserves_initial_and_two_per_remote_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'QUEUE_LOCK_RESUME_96.json').write_text('{}')
            resume.require_lock_budget(root, remote_count=1)
            (root / 'QUEUE_LOCK_RESUME_97.json').write_text('{}')
            with self.assertRaises(ValueError):
                resume.require_lock_budget(root, remote_count=1)

    def test_stop_failure_does_not_create_complete_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'results'
            root.mkdir()
            events = root / 'ATTEMPTS.jsonl'
            events.write_text('')
            with patch.object(resume, 'write_once', side_effect=OSError('NFS error')):
                with self.assertRaises(OSError):
                    resume.write_completion_and_stop(
                        root, Path(tmp) / 'control', {'status': 'complete'},
                        manifest_sha256='m', producer_commit='p',
                        scheduler_commit='s', events=events)
            self.assertFalse((root / 'QUEUE_COMPLETE.json').exists())

    def test_heartbeat_current_reads_sequence_after_99_to_100(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for number, free in ((99, 100), (100, 7000)):
                (root / 'HEARTBEAT_CURRENT.json').write_text(json.dumps({
                    'protocol': 1, 'sequence': number, 'pid': 41002,
                    'gpu': {'free_mib': free}}))
            self.assertEqual(resume.read_control_gpu(root)['free_mib'], 7000)

    def test_control_heartbeat_age_uses_90_second_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'HEARTBEAT_CURRENT.json'
            path.write_text(json.dumps({'protocol': 1, 'sequence': 1,
                'pid': 41002, 'gpu': {'free_mib': 7000}}))
            os.utime(path, (time.time() - 80, time.time() - 80))
            self.assertEqual(resume.read_control_gpu(root)['free_mib'], 7000)
            os.utime(path, (time.time() - 91, time.time() - 91))
            with self.assertRaises(ValueError):
                resume.read_control_gpu(root)

    def test_two_local_lanes_fallback_when_remote_never_admitted_without_duplicate(self):
        pending = ['local-a', 'local-b', 'remote-a', 'remote-b']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = multi.HandoffGate(root, root / 'ATTEMPTS.jsonl', pending,
                manifest_sha256='m', producer_commit='p', scheduler_commit='s')
            seen = []
            outcomes = multi.dispatch_three(
                gate, {'remote-a', 'remote-b'},
                ready_local=lambda gpu, stop: True,
                run_local=lambda k, gpu: seen.append((k, gpu)) or 'success',
                ready_remote=lambda stop: False,
                run_remote=lambda k: self.fail('remote must stay idle'))
            self.assertEqual(set(outcomes), set(pending))
            self.assertEqual(len(seen), len(pending))
            self.assertEqual(len({k for k, _ in seen}), len(pending))
            self.assertEqual(gate.pending, [])

    def test_remote_failure_stops_other_lanes_before_next_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = multi.HandoffGate(root, root / 'ATTEMPTS.jsonl',
                ['remote-a', 'local-a', 'local-b'], manifest_sha256='m',
                producer_commit='p', scheduler_commit='s')
            seen = []
            def ready_local(_gpu, _stop):
                deadline = time.monotonic() + 0.5
                while not gate.failed and time.monotonic() < deadline:
                    time.sleep(0.001)
                return True
            with self.assertRaises(RuntimeError):
                multi.dispatch_three(gate, {'remote-a'},
                    ready_local=ready_local,
                    run_local=lambda k, _gpu: seen.append(k) or 'success',
                    ready_remote=lambda _stop: True,
                    run_remote=lambda _k: (_ for _ in ()).throw(
                        RuntimeError('broker disconnected')))
            self.assertTrue(gate.failed)
            self.assertEqual(seen, [])

    def test_three_lanes_claim_distinct_keys_while_local_jobs_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = multi.HandoffGate(root, root / 'ATTEMPTS.jsonl',
                ['local-a', 'local-b', 'remote-a'], manifest_sha256='m',
                producer_commit='p', scheduler_commit='s')
            remote_started = multi.threading.Event()
            seen = []
            def local_run(k, _gpu):
                self.assertTrue(remote_started.wait(3))
                seen.append(k)
                return 'success'
            def remote_run(k):
                with gate.mutex:
                    gate.pending.remove(k)
                seen.append(k)
                remote_started.set()
                return 'success'
            outcomes = multi.dispatch_three(gate, {'remote-a'},
                ready_local=lambda _gpu, _stop: True, run_local=local_run,
                ready_remote=lambda _stop: True, run_remote=remote_run)
            self.assertEqual(set(outcomes), {'local-a', 'local-b', 'remote-a'})
            self.assertEqual(len(seen), len(set(seen)))

    def test_remote_success_is_sealed_then_finished_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key_name = key('sample_only_off', 'none', 70)
            job = root / key_name
            job.mkdir(parents=True)
            approved = root / 'approved.json'
            approved.write_text('{"seed":70}')
            events = root / 'ATTEMPTS.jsonl'
            events.write_text('')
            report = {'pid': 41002, 'host': '40902', 'producer_commit': 'p',
                'scheduler_commit': 's',
                'manifest_sha256': 'm', 'mount_source': 'shared:/brf',
                'result_disk_free_mib': 5000, 'scratch_path': '/tmp',
                'scratch_free_mib': 5000, 'driver': '575', 'python': '3.12',
                'torch': '2.7', 'torch_cuda': '12.8',
                'gpu': {'name': 'NVIDIA GeForce RTX 4090', 'uuid': 'GPU-r',
                        'free_mib': 7000, 'utilization': 90,
                        'compute_pids': [123]}}
            class Broker:
                def receive_report(self):
                    return report
                def alive(self):
                    return True
                def fresh_gpu(self):
                    return report['gpu']
                def send_go(self):
                    attempt = job / 'attempt-01'
                    attempt.mkdir()
                    (attempt / 'APPROVED_CONFIG.json').write_bytes(approved.read_bytes())
                    (attempt / 'JOB_RUNTIME.json').write_text(json.dumps({
                        'job_key': key_name, 'attempt': 1,
                        'manifest_sha256': 'm',
                        'config_sha256': multi.sha256(approved),
                        'production_commit': 'p',
                        'actual_output_dir': str(attempt),
                        'test_allowed': False}))
                    (job / 'attempt-01.log').write_text('ok')
                    sealed_job(root, key_name, 'm')
                def wait_exit(self):
                    return 0
            gate = multi.HandoffGate(root, events, [key_name],
                manifest_sha256='m', producer_commit='p', scheduler_commit='s')
            outcome = multi.execute_remote_job(
                {'job_key': key_name, 'config_sha256': multi.sha256(approved),
                 'method': 'sample_only_off'}, {'output_dir': str(job)}, approved,
                gate, Broker(), gpu_uuid='GPU-r', host='40902',
                mount_source='shared:/brf')
            self.assertEqual(outcome, 'success')
            history = [json.loads(line) for line in events.read_text().splitlines()]
            self.assertEqual([item['kind'] for item in history], ['start', 'finish'])
            self.assertEqual(history[1]['job_success_sha256'],
                multi.sha256(job / 'attempt-01' / 'JOB_SUCCESS.json'))

    def test_local_start_waits_for_worker_ack_and_writes_one_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key_name = key('sample_only_off', 'none', 70)
            job = root / key_name
            approved = root / 'approved.json'
            approved.write_text('{"seed":70}')
            events = root / 'ATTEMPTS.jsonl'
            events.write_text('')
            gate = multi.HandoffGate(root, events, [], manifest_sha256='m',
                                     producer_commit='p', scheduler_commit='s')
            class Process:
                def poll(self):
                    return None
                def wait(self):
                    return 0
            def spawn(_command, **_kwargs):
                attempt = job / 'attempt-01'
                attempt.mkdir()
                (attempt / 'APPROVED_CONFIG.json').write_bytes(approved.read_bytes())
                (attempt / 'JOB_RUNTIME.json').write_text(json.dumps({
                    'job_key': key_name, 'attempt': 1,
                    'manifest_sha256': 'm',
                    'config_sha256': multi.sha256(approved),
                    'production_commit': 'p',
                    'actual_output_dir': str(attempt),
                    'test_allowed': False}))
                sealed_job(root, key_name, 'm')
                return Process()
            outcome = multi.execute_local_job(
                {'job_key': key_name, 'config_sha256': multi.sha256(approved),
                 'method': 'sample_only_off'},
                {'output_dir': str(job), 'seed': 70, 'device': 'cuda:0'},
                approved, gate, producer_checkout=root,
                manifest=root / 'manifest.jsonl', gpu=0, gpu_uuid='GPU-local',
                snapshot_fn=lambda _gpu: {'name': 'NVIDIA GeForce RTX 3080',
                    'uuid': 'GPU-local', 'free_mib': 5000, 'utilization': 90,
                    'compute_pids': [11], 'temperature_c': 85,
                    'sm_clock_mhz': 750, 'sw_thermal_slowdown': True},
                spawn=spawn)
            self.assertEqual(outcome, 'success')
            history = [json.loads(line) for line in events.read_text().splitlines()]
            self.assertEqual([item['kind'] for item in history], ['start', 'finish'])
            self.assertEqual(history[0]['temperature_c'], 85)

    def test_local_gpu_loss_before_plan_requeues_key_without_ledger_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key_name = key('sample_only_off', 'none', 70)
            approved = root / 'approved.json'
            approved.write_text('{"seed":70}')
            events = root / 'ATTEMPTS.jsonl'
            events.write_text('')
            gate = multi.HandoffGate(root, events, [key_name],
                manifest_sha256='m', producer_commit='p', scheduler_commit='s')
            tries = [0]
            def run(k, _gpu):
                return multi.execute_local_job(
                    {'job_key': k, 'config_sha256': multi.sha256(approved),
                     'method': 'sample_only_off'},
                    {'output_dir': str(root / k), 'seed': 70,
                     'device': 'cuda:0'}, approved, gate,
                    producer_checkout=root, manifest=root / 'manifest.jsonl',
                    gpu=0, gpu_uuid='GPU-local',
                    snapshot_fn=lambda _gpu: {'name': 'NVIDIA GeForce RTX 3080',
                        'uuid': 'GPU-local', 'free_mib': 100,
                        'utilization': 90, 'compute_pids': []})
            def ready(_gpu, _stop):
                tries[0] += 1
                return tries[0] == 1
            outcomes = multi.dispatch_three(gate, set(), ready_local=ready,
                run_local=run, ready_remote=lambda _stop: False,
                run_remote=lambda _k: self.fail())
            self.assertEqual(outcomes, {})
            self.assertEqual(gate.pending, [key_name])
            self.assertFalse(gate.failed)
            self.assertEqual(events.read_text(), '')
            self.assertFalse((root / key_name / 'attempt-01.plan.json').exists())

    def test_local_rejects_old_exact_ack_directory_before_plan_or_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key_name = key('sample_only_off', 'none', 70)
            job = root / key_name
            attempt = job / 'attempt-01'
            attempt.mkdir(parents=True)
            approved = root / 'approved.json'
            approved.write_text('{"seed":70}')
            (attempt / 'APPROVED_CONFIG.json').write_bytes(approved.read_bytes())
            (attempt / 'JOB_RUNTIME.json').write_text(json.dumps({
                'job_key': key_name, 'attempt': 1, 'manifest_sha256': 'm',
                'config_sha256': multi.sha256(approved),
                'production_commit': 'p', 'actual_output_dir': str(attempt),
                'test_allowed': False}))
            events = root / 'ATTEMPTS.jsonl'
            events.write_text('')
            gate = multi.HandoffGate(root, events, [], manifest_sha256='m',
                producer_commit='p', scheduler_commit='s')
            with self.assertRaises(multi.FailStop):
                multi.execute_local_job(
                    {'job_key': key_name, 'config_sha256': multi.sha256(approved),
                     'method': 'sample_only_off'},
                    {'output_dir': str(job), 'seed': 70, 'device': 'cuda:0'},
                    approved, gate, producer_checkout=root,
                    manifest=root / 'manifest', gpu=0, gpu_uuid='GPU-local',
                    snapshot_fn=lambda _gpu: {'name': 'NVIDIA GeForce RTX 3080',
                        'uuid': 'GPU-local', 'free_mib': 7000,
                        'utilization': 0, 'compute_pids': []},
                    spawn=lambda *_a, **_k: self.fail('worker started'))
            self.assertEqual(events.read_text(), '')
            self.assertFalse((job / 'attempt-01.plan.json').exists())

    def test_central_entry_completes_with_local_fallback_and_writes_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'results'
            root.mkdir()
            control = Path(tmp) / 'control'
            control.mkdir()
            key_name = key('sample_only_off', 'none', 70)
            manifest = Path(tmp) / 'manifest.jsonl'
            approved = Path(tmp) / 'approved.json'
            approved.write_text('{"seed":70}')
            row = {'job_key': key_name, 'config_sha256': multi.sha256(approved),
                   'config_relpath': 'approved.json', 'method': 'sample_only_off'}
            manifest.write_text(json.dumps(row) + '\n')
            job = root / key_name
            config = {'output_dir': str(job), 'seed': 70, 'device': 'cuda:0'}
            (root / 'ATTEMPTS.jsonl').write_text('')
            (root / 'QUEUE_LOCK.json').write_text(json.dumps({
                'pid': 123, 'manifest_sha256': 'm', 'production_commit': 'p'}))
            anchor = {'remote_job_keys': [key_name],
                'remote': {'host': '40902', 'gpu_uuid': 'GPU-r',
                    'mount_source': 'shared:/brf', 'environment': {
                        'driver': '575', 'python': '3.12',
                        'torch': '2.7', 'torch_cuda': '12.8'}}}
            class Process:
                def poll(self):
                    return None
                def wait(self):
                    return 0
            def spawn(_command, **_kwargs):
                attempt = job / 'attempt-01'
                attempt.mkdir()
                (attempt / 'APPROVED_CONFIG.json').write_bytes(approved.read_bytes())
                (attempt / 'JOB_RUNTIME.json').write_text(json.dumps({
                    'job_key': key_name, 'attempt': 1,
                    'manifest_sha256': 'm',
                    'config_sha256': multi.sha256(approved),
                    'production_commit': 'p',
                    'actual_output_dir': str(attempt),
                    'test_allowed': False}))
                sealed_job(root, key_name, 'm')
                return Process()
            completion = resume.execute_multihost_in_root(
                [row], {key_name: config}, manifest, root, Path(tmp),
                'm', 's', 'p', ['GPU-0', 'GPU-1'], anchor, 'anchor-sha', control,
                local_snapshot=lambda gpu: {'name': 'NVIDIA GeForce RTX 3080',
                    'uuid': ['GPU-0', 'GPU-1'][gpu], 'free_mib': 7000,
                    'utilization': 90, 'compute_pids': [],
                    'temperature_c': 85, 'sm_clock_mhz': 750,
                    'sw_thermal_slowdown': True},
                local_spawn=spawn,
                ready_local=lambda _gpu, _stop: True,
                ready_remote=lambda _stop: False)
            self.assertEqual(completion['accounted_for'], 1)
            self.assertTrue((root / 'MULTIHOST_HANDOFF_01.json').is_file())
            self.assertTrue((control / 'STOP.json').is_file())

    def test_prior_multihost_handoff_blocks_recovery_of_remote_active_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'results'
            root.mkdir()
            key_name = key('sample_only_off', 'none', 70)
            (root / 'MULTIHOST_HANDOFF_01.json').write_text('{"status":"started"}')
            (root / 'QUEUE_LOCK_RESUME_01.json').write_text(json.dumps({
                'pid': 41002, 'manifest_sha256': 'm',
                'production_commit': 'p'}))
            (root / 'ATTEMPTS.jsonl').write_text(json.dumps({
                'kind': 'start', 'job_key': key_name, 'attempt': 1,
                'config_sha256': 'a' * 64}) + '\n')
            row = {'job_key': key_name, 'config_sha256': 'a' * 64,
                   'method': 'sample_only_off'}
            with patch.object(resume.queue, 'recover_interrupted_attempts',
                              side_effect=AssertionError('recovery was called')):
                with self.assertRaisesRegex(ValueError, 'manual'):
                    resume.execute_multihost_in_root(
                        [row], {key_name: {'device': 'cuda:0'}},
                        Path(tmp) / 'manifest', root, Path(tmp),
                        'm', 's', 'p', ['GPU-0', 'GPU-1'],
                        {'remote_job_keys': []}, 'anchor', Path(tmp) / 'control')

    def test_central_and_file_broker_complete_one_remote_job_in_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'results'
            root.mkdir()
            control = Path(tmp) / 'control'
            control.mkdir()
            key_name = key('sample_only_off', 'none', 70)
            manifest = Path(tmp) / 'manifest.jsonl'
            approved = Path(tmp) / 'approved.json'
            approved.write_text('{"seed":70}')
            row = {'job_key': key_name, 'config_sha256': multi.sha256(approved),
                   'config_relpath': 'approved.json', 'method': 'sample_only_off'}
            manifest.write_text(json.dumps(row) + '\n')
            digest = multi.sha256(manifest)
            job = root / key_name
            (root / 'ATTEMPTS.jsonl').write_text('')
            (root / 'QUEUE_LOCK.json').write_text(json.dumps({
                'pid': 123, 'manifest_sha256': digest, 'production_commit': 'p'}))
            gpu = {'name': 'NVIDIA GeForce RTX 4090', 'uuid': 'GPU-r',
                   'free_mib': 7000, 'utilization': 90, 'compute_pids': []}
            report = {'pid': multi.os.getpid(), 'host': '40902',
                'producer_commit': 'p', 'scheduler_commit': 's',
                'manifest_sha256': digest,
                'mount_source': 'shared:/brf', 'result_disk_free_mib': 5000,
                'scratch_path': '/tmp', 'scratch_free_mib': 5000,
                'driver': '575', 'python': '3.12', 'torch': '2.7',
                'torch_cuda': '12.8', 'gpu': gpu}
            anchor = {'remote_job_keys': [key_name],
                'remote': {'host': '40902', 'gpu_uuid': 'GPU-r',
                    'mount_source': 'shared:/brf', 'environment': {
                        'driver': '575', 'python': '3.12',
                        'torch': '2.7', 'torch_cuda': '12.8'}}}
            writer = remote_broker.HeartbeatWriter(
                control, pid=multi.os.getpid(), snapshot=lambda: gpu)
            pulse = multi.threading.Thread(target=writer.run, daemon=True)
            pulse.start()
            class Process:
                def wait(self):
                    return 0
            def spawn(_command, **_kwargs):
                attempt = job / 'attempt-01'
                attempt.mkdir()
                (attempt / 'APPROVED_CONFIG.json').write_bytes(approved.read_bytes())
                (attempt / 'JOB_RUNTIME.json').write_text(json.dumps({
                    'job_key': key_name, 'attempt': 1,
                    'manifest_sha256': digest,
                    'config_sha256': multi.sha256(approved),
                    'production_commit': 'p',
                    'actual_output_dir': str(attempt),
                    'test_allowed': False}))
                sealed_job(root, key_name, digest)
                return Process()
            broker_errors = []
            def broker_run():
                try:
                    remote_broker.broker_mailbox_once(
                        control, 1, report=report, manifest=manifest,
                        manifest_sha256=file_sha(manifest),
                        producer_commit='p', command=['worker'], env={},
                        log_path=job / 'attempt-01.log', gpu_uuid='GPU-r',
                        result_root=root, snapshot=lambda: gpu, spawn=spawn)
                except BaseException as error:
                    broker_errors.append(error)
            broker_thread = multi.threading.Thread(target=broker_run)
            broker_thread.start()
            try:
                completion = resume.execute_multihost_in_root(
                    [row], {key_name: {'output_dir': str(job), 'seed': 70,
                                       'device': 'cuda:0'}},
                    manifest, root, Path(tmp), digest, 's', 'p',
                    ['GPU-0', 'GPU-1'], anchor, 'anchor-sha', control,
                    ready_local=lambda _gpu, _stop: False,
                    ready_remote=lambda _stop: True)
                self.assertEqual(completion['successes'], 1)
                self.assertEqual(broker_errors, [])
                self.assertTrue((control / 'LAUNCH_CHECK_000001.json').is_file())
            finally:
                writer.stop.set()
                pulse.join(3)
                broker_thread.join(3)


if __name__ == '__main__':
    unittest.main()
