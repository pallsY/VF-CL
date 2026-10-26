import hashlib
import json
import os
import tempfile
import unittest
import threading
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import party_drift_balanced_resume as balanced


def sealed_job(root, key, digest, attempt=1):
    run = root / key / f'attempt-{attempt:02d}'
    (run / 'checkpoints').mkdir(parents=True)
    checkpoint = run / 'checkpoints' / 'formal_final.pt'
    checkpoint.write_bytes(b'fixed checkpoint')
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    flow = run / 'data_flow_audit.jsonl'
    flow.write_text('{"split":"train"}\n')
    readout = run / 'HOLDOUT_READOUT.json'
    readout.write_text(json.dumps({'checkpoint_sha256': checkpoint_sha}))
    gate = {
        'status': 'success', 'job_key': key, 'manifest_sha256': digest,
        'test_accessed': False,
        'readout_sha256': hashlib.sha256(readout.read_bytes()).hexdigest(),
        'checkpoint_sha256': checkpoint_sha,
        'data_flow_sha256': hashlib.sha256(flow.read_bytes()).hexdigest(),
        'selection_audit_sha256': None, 'replay_steps': 0,
    }
    (run / 'JOB_SUCCESS.json').write_text(json.dumps(gate))
    return hashlib.sha256((run / 'JOB_SUCCESS.json').read_bytes()).hexdigest()


class BalancedAdmissionTests(unittest.TestCase):
    def test_preparation_gpu_wait_has_bounded_timeout(self):
        now = [0]
        busy = {'name': 'NVIDIA GeForce RTX 3080', 'uuid': 'GPU-one',
                'memory_mib': 7000, 'total_mib': 10240,
                'free_mib': 3240,
                'utilization': 50,
                'compute_pids': [123]}
        self.assertFalse(balanced.wait_ready(
            1, ['pending'], threading.Event(), snapshot_fn=lambda _gpu: busy,
            clock_fn=lambda: now[0],
            sleep_fn=lambda seconds: now.__setitem__(0, now[0] + seconds),
            max_wait_seconds=120))
        self.assertLessEqual(now[0], 120)

    def test_prepared_proof_binds_stable_ledger_and_gpu_uuid(self):
        proof = balanced.make_admission_proof(
            manifest='m', ledger='l', producer='p', scheduler='s',
            gpu_uuid='GPU-one', original_pid=123,
            pending_count=24, reconstructable=['job-a'])
        self.assertEqual(proof['admission_duration_seconds'], 120)
        self.assertEqual(proof['min_free_mib'], 5120)
        self.assertEqual(proof['source_ledger_sha256'], 'l')
        self.assertEqual(proof['gpu1_uuid'], 'GPU-one')
        self.assertEqual(proof['original_queue_pid'], 123)
        self.assertEqual(proof['reconstructable'], ['job-a'])

    def test_idle_proof_only_prepared_while_original_coordinator_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'results'
            root.mkdir()
            (root / 'QUEUE_LOCK.json').write_text(json.dumps({'pid': 123}))
            proc = Path(tmp) / 'proc' / '123'
            proc.mkdir(parents=True)
            (proc / 'status').write_text('Name:\tpython\nState:\tS (sleeping)\n')
            with self.assertRaises(ValueError):
                balanced.assert_stopped_coordinator(root,
                    proc_root=Path(tmp) / 'proc')
            (proc / 'status').write_text('Name:\tpython\nState:\tT (stopped)\n')
            balanced.assert_stopped_coordinator(root,
                proc_root=Path(tmp) / 'proc')

    def test_anchor_binds_admission_proof_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            proof = Path(tmp) / 'GPU1_ADMISSION_PROOF.json'
            proof.write_text('{"admission_duration_seconds":120}')
            anchor = {'admission_proof_path': str(proof),
                      'admission_proof_sha256': hashlib.sha256(
                          proof.read_bytes()).hexdigest()}
            self.assertEqual(balanced.load_anchored_admission_proof(
                anchor)['admission_duration_seconds'], 120)
            proof.write_text('{"admission_duration_seconds":0}')
            with self.assertRaises(ValueError):
                balanced.load_anchored_admission_proof(anchor)

    def test_admission_proof_accepts_outside_process_but_rejects_low_headroom(self):
        now = datetime(2026, 10, 27, 1, 0, tzinfo=timezone.utc)
        proof = {'manifest_sha256': 'm', 'source_ledger_sha256': 'l',
                 'producer_commit': 'p', 'scheduler_commit': 's',
                 'gpu1_uuid': 'GPU-one', 'admission_duration_seconds': 120,
                 'min_free_mib': 5120,
                 'completed_utc': now.isoformat()}
        snapshot = {'name': 'NVIDIA GeForce RTX 3080',
                    'uuid': 'GPU-one', 'memory_mib': 4000,
                    'total_mib': 10240, 'free_mib': 6240,
                    'utilization': 100,
                    'compute_pids': [123]}
        balanced.validate_admission_proof(proof, manifest='m', ledger='l',
                                          producer='p', scheduler='s',
                                          snapshot=snapshot, now=now)
        with self.assertRaises(ValueError):
            balanced.validate_admission_proof(
                proof, manifest='m', ledger='l', producer='p', scheduler='s',
                snapshot={**snapshot, 'free_mib': 4240}, now=now)
        with self.assertRaises(ValueError):
            balanced.validate_admission_proof(
                proof, manifest='m', ledger='l', producer='p', scheduler='s',
                snapshot=snapshot, now=now + timedelta(minutes=31))

    def test_gpu_probe_error_stops_other_lane_before_any_claim(self):
        signaled = threading.Event()
        calls = []
        def ready(gpu, _pending, stop):
            if gpu == 1:
                signaled.set()
                raise RuntimeError('GPU probe failed')
            signaled.wait(1)
            stop.wait(0.1)
            return True
        with self.assertRaises(RuntimeError):
            balanced.dispatch_pending(
                ['a', 'b', 'c'], ready=ready,
                run_one=lambda key, gpu: calls.append((key, gpu)) or 'success')
        self.assertEqual(calls, [])

    def test_handoff_rejects_original_queue_or_worker_still_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'results'
            root.mkdir()
            (root / 'QUEUE_LOCK.json').write_text(json.dumps({'pid': 123}))
            proc = Path(tmp) / 'proc'
            (proc / '123').mkdir(parents=True)
            (proc / '123' / 'cmdline').write_bytes(
                b'python\0party_drift_queue.py\0')
            with self.assertRaises(RuntimeError):
                balanced.assert_original_queue_gone(
                    root, '/tmp/manifest.jsonl', proc_root=proc)
            (proc / '123' / 'cmdline').write_bytes(b'')
            (proc / '124').mkdir()
            (proc / '124' / 'cmdline').write_bytes(
                b'python\0run_party_drift_development.py\0--manifest\0'
                b'/tmp/manifest.jsonl\0--job-key\0cifar100/drift/0.5/70\0')
            with self.assertRaises(RuntimeError):
                balanced.assert_original_queue_gone(
                    root, '/tmp/manifest.jsonl', proc_root=proc)
            (proc / '124' / 'cmdline').write_bytes(b'')
            balanced.assert_original_queue_gone(root, '/tmp/manifest.jsonl',
                                                proc_root=proc)

    def test_resumed_queue_preserves_sealed_job_and_completes_pending_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'results'
            root.mkdir()
            keys = [f'cifar100/sample_only_off/none/{seed}'
                    for seed in (71, 70, 72)]
            rows = [{'job_key': key, 'config_sha256': str(i) * 64,
                     'method': 'sample_only_off'}
                    for i, key in enumerate(keys, 1)]
            configs = {key: {'development_job_key': key,
                             'output_dir': str(root / key),
                             'seed': int(key.split('/')[-1]),
                             'device': ('cuda:1' if key.endswith('/71')
                                        else 'cuda:0'),
                             'party_drift_enabled': 0}
                       for key in keys}
            sealed_job(root, keys[0], 'm')
            (root / keys[0] / 'attempt-01.plan.json').write_text(json.dumps({
                'attempt': 1, 'job_key': keys[0]}))
            events = root / 'ATTEMPTS.jsonl'
            events.write_text(''.join(json.dumps(item) + '\n' for item in [
                {'kind': 'start', 'job_key': keys[0], 'attempt': 1,
                 'config_sha256': rows[0]['config_sha256']},
            ]))
            (root / 'QUEUE_LOCK.json').write_text(json.dumps({
                'pid': 999, 'manifest_sha256': 'm',
                'production_commit': 'p'}))
            producer = Path(tmp) / 'producer'
            producer.mkdir()
            launched = []
            def fake_run(command, **_kwargs):
                key = command[command.index('--job-key') + 1]
                launched.append(key)
                sealed_job(root, key, 'm')
                return SimpleNamespace(returncode=0)
            gate = balanced.execute_balanced_in_root(
                rows, configs, Path(tmp) / 'manifest.jsonl', root,
                producer, 'm', 's', 'p', ['GPU-zero', 'GPU-one'],
                ready=lambda _gpu, _pending, _stop: True,
                run_process=fake_run,
                snapshot_fn=lambda gpu: {
                    'name': 'NVIDIA GeForce RTX 3080',
                    'uuid': ['GPU-zero', 'GPU-one'][gpu],
                    'memory_mib': 0, 'total_mib': 10240,
                    'free_mib': 10240,
                    'utilization': 0, 'compute_pids': []})
            self.assertEqual(set(launched), set(keys[1:]))
            self.assertEqual(gate['accounted_for'], 3)
            self.assertTrue((root / 'QUEUE_COMPLETE.json').is_file())
            self.assertTrue((root / 'BALANCED_HANDOFF.json').is_file())
            handoff = json.loads((root / 'BALANCED_HANDOFF.json').read_text())
            self.assertEqual(handoff['reconstructed_keys'], [keys[0]])
            self.assertEqual(handoff['new_scheduler_pid'], os.getpid())
            self.assertEqual(handoff['reason'], 'scheduler_only_balancing')
            self.assertIn('party_drift_balanced_resume.py',
                          handoff['startup_command'])

    def test_manifest_rows_cannot_change_producer_or_output_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = 'cifar100/sample_only_off/none/70'
            row = {'job_key': key, 'config_sha256': 'a' * 64,
                   'production_commit': 'p', 'method': 'sample_only_off'}
            config = {'development_job_key': key, 'output_dir': str(root / key),
                      'device': 'cuda:0', 'seed': 70}
            balanced.validate_manifest_rows([row], {key: config}, root,
                                            producer='p', expected={key})
            row['production_commit'] = 'changed'
            with self.assertRaises(ValueError):
                balanced.validate_manifest_rows([row], {key: config}, root,
                                                producer='p', expected={key})

    def test_read_only_check_reports_live_open_start_without_claiming_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            key = 'cifar100/sample_only_off/none/70'
            rows = [{'job_key': key, 'config_sha256': 'a' * 64,
                     'method': 'sample_only_off'}]
            events = [{'kind': 'start', 'job_key': key, 'attempt': 1,
                       'config_sha256': 'a' * 64}]
            state = balanced.validate_history(rows, events, Path(tmp), 'm',
                                              allow_active=True)
            self.assertEqual(state['active'], [key])
            self.assertEqual(state['pending'], [])
            with self.assertRaises(FileNotFoundError):
                balanced.validate_history(rows, events, Path(tmp), 'm')

    def test_completion_gate_requires_every_listed_job_final(self):
        state = {'pending': ['b'], 'reconstructable': [],
                 'finished': {'a': 'success'}, 'anomalies': []}
        with self.assertRaises(ValueError):
            balanced.completion_record(state, 2, 'manifest', 'ledger')
        state['pending'] = []
        state['finished']['b'] = 'training_anomaly'
        gate = balanced.completion_record(state, 2, 'manifest', 'ledger')
        self.assertEqual(gate['accounted_for'], 2)
        self.assertEqual(gate['training_anomalies'], 1)
        self.assertEqual(gate['attempts_sha256'], 'ledger')

    def test_anchor_must_match_frozen_producer_manifest_and_pre_handoff_ledger(self):
        anchor = {'producer_commit': 'p', 'scheduler_commit': 's',
                  'manifest_sha256': 'm', 'source_ledger_sha256': 'l',
                  'gpu_uuids': ['GPU-zero', 'GPU-one']}
        balanced.validate_anchor(anchor, producer='p', scheduler='s',
                                 manifest='m', ledger='l',
                                 gpu_uuids=['GPU-zero', 'GPU-one'])
        with self.assertRaises(ValueError):
            balanced.validate_anchor(anchor, producer='p', scheduler='s',
                                     manifest='m', ledger='changed',
                                     gpu_uuids=['GPU-zero', 'GPU-one'])

    def test_one_job_uses_frozen_worker_and_seals_physical_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = 'cifar100/sample_only_off/none/70'
            job_root = root / key
            row = {'job_key': key, 'config_sha256': 'a' * 64,
                   'method': 'sample_only_off'}
            config = {'output_dir': str(job_root), 'seed': 70,
                      'device': 'cuda:0', 'party_drift_enabled': 0}
            events = root / 'ATTEMPTS.jsonl'
            events.write_text('')
            producer = root / 'frozen-producer'
            producer.mkdir()
            commands = []
            def fake_run(command, **kwargs):
                commands.append((command, kwargs['env']['CUDA_VISIBLE_DEVICES']))
                sealed_job(root, key, 'm')
                return SimpleNamespace(returncode=0)
            outcome = balanced.execute_one_job(
                row, config, producer, root / 'manifest.jsonl', root, events,
                'm', 'scheduler-sha',
                'producer-sha', 1, 'GPU-one', threading.Lock(),
                run_process=fake_run,
                snapshot_fn=lambda _gpu: {
                    'name': 'NVIDIA GeForce RTX 3080', 'uuid': 'GPU-one',
                    'memory_mib': 4000, 'total_mib': 10240,
                    'free_mib': 6000,
                    'utilization': 100, 'compute_pids': [123]})
            self.assertEqual(outcome, 'success')
            self.assertEqual(commands[0][1], 'GPU-one')
            self.assertEqual(commands[0][0][2], str(
                producer / 'run_party_drift_development.py'))
            history = [json.loads(line) for line in events.read_text().splitlines()]
            self.assertEqual([event['kind'] for event in history],
                             ['start', 'finish'])
            self.assertEqual(history[0]['physical_gpu_uuid'], 'GPU-one')
            self.assertEqual(history[0]['free_memory_mib'], 6000)
            self.assertEqual(history[0]['co_tenant_pids'], [123])

    def test_gpu_one_losing_headroom_does_not_reserve_or_start_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = 'cifar100/sample_only_off/none/70'
            events = root / 'ATTEMPTS.jsonl'
            events.write_text('')
            with self.assertRaisesRegex(RuntimeError, 'launch conditions'):
                balanced.execute_one_job(
                    {'job_key': key, 'config_sha256': 'a' * 64,
                     'method': 'sample_only_off'},
                    {'output_dir': str(root / key), 'seed': 70,
                     'device': 'cuda:0'}, root, root / 'manifest.jsonl',
                    root, events, 'm', 's', 'p', 1, 'GPU-one',
                    threading.Lock(),
                    snapshot_fn=lambda _gpu: {
                        'name': 'NVIDIA GeForce RTX 3080',
                        'uuid': 'GPU-one', 'memory_mib': 6000,
                        'total_mib': 10240, 'free_mib': 4240,
                        'utilization': 50,
                        'compute_pids': [123]})
            self.assertEqual(events.read_text(), '')
            self.assertFalse((root / key).exists())

    def test_gpu_snapshot_uses_uuid_to_reject_only_its_own_compute_process(self):
        value = balanced.parse_gpu_snapshot(
            '1, NVIDIA GeForce RTX 3080, GPU-one, 226, 10240, 9900, 0',
            'GPU-zero, 44, 2418\nGPU-one, 55, 216\n')
        self.assertEqual(value['uuid'], 'GPU-one')
        self.assertEqual(value['compute_pids'], [55])
        self.assertFalse(balanced.idle_snapshot(value))

    def test_gpu_one_uses_reported_free_memory_not_total_minus_used(self):
        snapshot = {'name': 'NVIDIA GeForce RTX 3080', 'uuid': 'GPU-one',
                    'memory_mib': 4000, 'total_mib': 10240,
                    'free_mib': 5000, 'utilization': 0,
                    'compute_pids': [123]}
        self.assertFalse(balanced.shared_headroom_snapshot(snapshot))

    def test_gpu_one_requires_two_continuous_minutes_of_headroom(self):
        now = [0]
        def snapshot(_gpu):
            return {'name': 'NVIDIA GeForce RTX 3080', 'uuid': 'GPU-1',
                    'memory_mib': 7000 if now[0] < 60 else 4000,
                    'total_mib': 10240,
                    'free_mib': 3240 if now[0] < 60 else 6240,
                    'utilization': 100,
                    'compute_pids': [123]}
        self.assertTrue(balanced.wait_ready(
            1, ['pending'], threading.Event(), snapshot_fn=snapshot,
            clock_fn=lambda: now[0],
            sleep_fn=lambda seconds: now.__setitem__(0, now[0] + seconds)))
        self.assertGreaterEqual(now[0], 180)
        self.assertLess(now[0], 240)

    def test_dispatch_keeps_gpu_zero_moving_when_gpu_one_is_busy(self):
        calls = []
        outcomes = balanced.dispatch_pending(
            ['job-a', 'job-b', 'job-c'],
            ready=lambda gpu, pending, stop: gpu == 0,
            run_one=lambda key, gpu: calls.append((key, gpu)) or 'success')
        self.assertEqual(calls, [('job-a', 0), ('job-b', 0), ('job-c', 0)])
        self.assertEqual(outcomes, {'job-a': 'success', 'job-b': 'success',
                                    'job-c': 'success'})

    def test_mapping_keeps_approved_logical_device_and_records_physical_uuid(self):
        config = {'device': 'cuda:0', 'seed': 70}
        env = balanced.worker_environment({'PATH': 'fixed'}, 1,
                                          config['seed'], 'GPU-one')
        self.assertEqual(config['device'], 'cuda:0')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-one')
        self.assertEqual(env['PYTHONHASHSEED'], '70')
        event = balanced.start_record(
            'cifar100/drift/0.5/70', 'a' * 64, 1, 'GPU-1',
            'scheduler-sha', 'producer-sha')
        self.assertEqual(event['physical_gpu_uuid'], 'GPU-1')
        self.assertEqual(event['physical_gpu_index'], 1)

    def test_busy_gpu_cannot_claim_and_each_pending_key_is_claimed_once(self):
        self.assertFalse(balanced.idle_snapshot(
            {'memory_mib': 216, 'utilization': 0, 'compute_pids': [123]}))
        self.assertTrue(balanced.idle_snapshot(
            {'memory_mib': 1, 'utilization': 0, 'compute_pids': []}))
        keys = ['cifar100/drift/0.5/70', 'cifar100/drift/0.5/72']
        mutex = threading.Lock()
        self.assertIsNone(balanced.claim_next(keys, mutex, ready=False))
        self.assertEqual(balanced.claim_next(keys, mutex, ready=True),
                         'cifar100/drift/0.5/70')
        self.assertEqual(balanced.claim_next(keys, mutex, ready=True),
                         'cifar100/drift/0.5/72')
        self.assertIsNone(balanced.claim_next(keys, mutex, ready=True))

    def test_sealed_open_start_is_reconstructable_and_only_unstarted_is_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keys = ('cifar100/sample_only_off/none/70',
                    'cifar100/sample_only_off/none/71',
                    'cifar100/sample_only_off/none/72')
            rows = [{'job_key': key, 'config_sha256': str(i) * 64,
                     'method': 'sample_only_off'} for i, key in enumerate(keys, 1)]
            seal0 = sealed_job(root, keys[0], 'm')
            sealed_job(root, keys[1], 'm')
            events = [
                {'kind': 'start', 'job_key': keys[0], 'attempt': 1,
                 'config_sha256': rows[0]['config_sha256']},
                {'kind': 'finish', 'job_key': keys[0], 'attempt': 1,
                 'classification': 'success', 'job_success_sha256': seal0},
                {'kind': 'start', 'job_key': keys[1], 'attempt': 1,
                 'config_sha256': rows[1]['config_sha256']},
            ]
            state = balanced.validate_history(rows, events, root, 'm')
            self.assertEqual(state['pending'], [keys[2]])
            self.assertEqual(state['reconstructable'], [keys[1]])
            events[2]['config_sha256'] = 'x' * 64
            with self.assertRaises(ValueError):
                balanced.validate_history(rows, events, root, 'm')

    def test_training_anomaly_is_accounted_for_and_never_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            key = 'cifar100/sample_only_off/none/70'
            rows = [{'job_key': key, 'config_sha256': 'a' * 64,
                     'method': 'sample_only_off'}]
            events = [
                {'kind': 'start', 'job_key': key, 'attempt': 1,
                 'config_sha256': 'a' * 64},
                {'kind': 'finish', 'job_key': key, 'attempt': 1,
                 'classification': 'training_anomaly'},
            ]
            state = balanced.validate_history(rows, events, Path(tmp), 'm')
            self.assertEqual(state['pending'], [])
            self.assertEqual(state['anomalies'], [key])


if __name__ == '__main__':
    unittest.main()
