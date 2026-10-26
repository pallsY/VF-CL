import hashlib
import json
import tempfile
import unittest
import threading
from types import SimpleNamespace
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
                run_process=fake_run)
            self.assertEqual(set(launched), set(keys[1:]))
            self.assertEqual(gate['accounted_for'], 3)
            self.assertTrue((root / 'QUEUE_COMPLETE.json').is_file())
            self.assertTrue((root / 'BALANCED_HANDOFF.json').is_file())
            handoff = json.loads((root / 'BALANCED_HANDOFF.json').read_text())
            self.assertEqual(handoff['reconstructed_keys'], [keys[0]])

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
                run_process=fake_run)
            self.assertEqual(outcome, 'success')
            self.assertEqual(commands[0][1], '1')
            self.assertEqual(commands[0][0][2], str(
                producer / 'run_party_drift_development.py'))
            history = [json.loads(line) for line in events.read_text().splitlines()]
            self.assertEqual([event['kind'] for event in history],
                             ['start', 'finish'])
            self.assertEqual(history[0]['physical_gpu_uuid'], 'GPU-one')

    def test_gpu_snapshot_uses_uuid_to_reject_only_its_own_compute_process(self):
        value = balanced.parse_gpu_snapshot(
            '1, NVIDIA GeForce RTX 3080, GPU-one, 226, 0',
            'GPU-zero, 44, 2418\nGPU-one, 55, 216\n')
        self.assertEqual(value['uuid'], 'GPU-one')
        self.assertEqual(value['compute_pids'], [55])
        self.assertFalse(balanced.idle_snapshot(value))

    def test_gpu_one_requires_ten_continuous_minutes_without_compute_process(self):
        now = [0]
        def snapshot(_gpu):
            return {'name': 'NVIDIA GeForce RTX 3080', 'uuid': 'GPU-1',
                    'memory_mib': 216 if now[0] < 60 else 1,
                    'utilization': 0,
                    'compute_pids': [123] if now[0] < 60 else []}
        self.assertTrue(balanced.wait_ready(
            1, ['pending'], threading.Event(), snapshot_fn=snapshot,
            clock_fn=lambda: now[0],
            sleep_fn=lambda seconds: now.__setitem__(0, now[0] + seconds)))
        self.assertGreaterEqual(now[0], 660)

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
        env = balanced.worker_environment({'PATH': 'fixed'}, 1, config['seed'])
        self.assertEqual(config['device'], 'cuda:0')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '1')
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
