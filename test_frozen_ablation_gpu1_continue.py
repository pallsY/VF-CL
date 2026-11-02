import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    continuation = importlib.import_module('frozen_ablation_gpu1_continue')
except ModuleNotFoundError:
    continuation = None


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(continuation, 'continuation implementation is missing')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def identity(self, pid, **extra):
        return dict(pid=pid, ppid=1, pgrp=pid, session=pid, start_ticks=pid + 100,
                    boot_id='synthetic-boot', cmdline=['python', 'synthetic.py'], **extra)

    def test_candidates_can_only_shrink_and_mapping_keeps_two_visible_cards(self):
        keys = {'cifar100/no_partykd/43', 'isolet/no_partykd/42'}
        self.assertEqual(continuation.pending_candidates(keys), sorted(keys))
        with self.assertRaises(ValueError):
            continuation.pending_candidates({'cifar100/no_partykd/44'})
        row = {'job_key': 'cifar100/no_partykd/43', 'config_sha256': 'c' * 64,
               'code_commit': 'd' * 40}
        binding = continuation.execution_binding(row, 1, 'e' * 64, {'pid': 12})
        self.assertEqual(binding['logical_device'], 'cuda:0')
        self.assertEqual(binding['cuda_visible_devices'],
                         [continuation.GPU_UUID[1], continuation.GPU_UUID[0]])
        env = continuation.worker_environment({'seed': 43}, 1)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], ','.join(binding['cuda_visible_devices']))
        self.assertEqual(env['PYTHONHASHSEED'], '43')
        self.assertEqual(env['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
        self.assertEqual(env['OMP_NUM_THREADS'], '1')
        self.assertEqual(env['MKL_NUM_THREADS'], '1')

    def authorization_fixture(self):
        key = 'isolet/fim_off/42'
        config = {'seed': 42, 'device': 'cuda:0', 'num_tasks': 2,
                  'output_dir': str(self.root / 'fim_off/isolet/42/attempt-01')}
        config_path = self.root / 'manifests/configs/job.json'
        config_path.parent.mkdir(parents=True)
        config_path.write_bytes((json.dumps(config, indent=2) + '\n').encode())
        row = {'job_key': key, 'arm': 'fim_off', 'code_commit': 'd' * 40,
               'config_relpath': 'configs/job.json',
               'config_sha256': continuation.sha(config_path)}
        manifest = self.root / 'manifests/manifest.jsonl'
        manifest.write_text(json.dumps(row) + '\n')
        parent = self.identity(123)
        epoch_dir = self.root / 'gpu1-rebalance-v1'
        epoch_dir.mkdir()
        continuation.write_once(epoch_dir / 'ARMED.json', {'fixture': 'armed'})
        prefix = b'{"synthetic":"settled prefix"}\n'
        epoch = {'schema_version': 1, 'manifest_sha256': continuation.sha(manifest),
                 'controller_identity': parent, 'assignments': {key: 1},
                 'armed_sha256': continuation.sha(epoch_dir / 'ARMED.json'),
                 'ledger_prefix_sha256': hashlib.sha256(prefix).hexdigest(),
                 'ledger_prefix_bytes_length': len(prefix),
                 'orchestration_commit': 'a' * 40}
        epoch_path = epoch_dir / 'EXECUTION_EPOCH.json'
        continuation.write_once(epoch_path, epoch)
        epoch_sha = continuation.sha(epoch_path)
        binding = continuation.execution_binding(row, 1, epoch_sha, parent)
        start = dict(kind='start', job_key=key, attempt=1,
                     config_sha256=row['config_sha256'], code_commit=row['code_commit'],
                     **binding)
        (self.root / 'ATTEMPTS.jsonl').write_bytes(prefix + (json.dumps(start) + '\n').encode())
        job_root = Path(config['output_dir']).parent
        job_root.mkdir(parents=True)
        continuation.write_once(job_root / 'attempt-01.plan.json', {'job_key': key, 'attempt': 1})
        return SimpleNamespace(**locals())

    def test_worker_authorization_rejects_duplicate_claim_parent_sha_map_and_config_tamper(self):
        f = self.authorization_fixture()
        with mock.patch.object(continuation, 'process_identity', return_value=f.parent), \
             mock.patch.object(continuation.os, 'getppid', return_value=123):
            auth = continuation.authorize_worker(f.manifest, f.epoch['manifest_sha256'],
                                                f.key, f.epoch_path, f.epoch_sha)
            self.assertEqual(auth['config_path'].read_bytes(), f.config_path.read_bytes())
            for change in ('duplicate', 'parent', 'sha', 'map', 'config'):
                with self.subTest(change=change):
                    ledger = (self.root / 'ATTEMPTS.jsonl').read_bytes()
                    raw_config = f.config_path.read_bytes()
                    original = f.start.copy()
                    try:
                        if change == 'duplicate':
                            (self.root / 'ATTEMPTS.jsonl').write_bytes(ledger + (json.dumps(f.start) + '\n').encode())
                        elif change == 'parent':
                            f.parent['start_ticks'] += 1
                        elif change == 'map':
                            bad = {**f.start, 'gpu_uuid': continuation.GPU_UUID[0]}
                            (self.root / 'ATTEMPTS.jsonl').write_bytes(f.prefix + (json.dumps(bad) + '\n').encode())
                        elif change == 'config':
                            f.config_path.write_bytes(raw_config + b' ')
                        with self.assertRaises(ValueError):
                            continuation.authorize_worker(f.manifest, f.epoch['manifest_sha256'],
                                f.key, f.epoch_path, '0' * 64 if change == 'sha' else f.epoch_sha)
                    finally:
                        (self.root / 'ATTEMPTS.jsonl').write_bytes(ledger)
                        f.config_path.write_bytes(raw_config)
                        if change == 'parent':
                            f.parent['start_ticks'] -= 1

    def test_ready_go_binds_the_exact_worker_and_epoch(self):
        expected = self.identity(24)
        auth = {'job_key': 'isolet/fim_off/42', 'execution_epoch_sha256': 'e' * 64}
        ready = dict(auth, worker_identity=expected)
        with mock.patch.object(continuation, 'process_identity', return_value=expected):
            self.assertEqual(continuation.validate_ready(ready, auth, 24), expected)
            for bad in ({**ready, 'execution_epoch_sha256': 'f' * 64},
                        {**ready, 'worker_identity': {**expected, 'start_ticks': 26}}):
                with self.assertRaises(ValueError):
                    continuation.validate_ready(bad, auth, 24)

    def test_parent_watchdog_kills_only_its_own_session_on_identity_change(self):
        parent = self.identity(3)
        with mock.patch.object(continuation, 'process_identity', return_value={**parent, 'start_ticks': 5}), \
             mock.patch.object(continuation.os, 'getpid', return_value=20), \
             mock.patch.object(continuation.os, 'getsid', return_value=20, create=True), \
             mock.patch.object(continuation.os, 'getpgrp', return_value=20, create=True), \
             mock.patch.object(continuation.os, 'killpg', create=True) as kill, \
             mock.patch.object(continuation.signal, 'SIGKILL', 9, create=True):
            continuation.parent_watchdog(parent, __import__('threading').Event(), interval=0)
            kill.assert_called_once_with(20, continuation.signal.SIGKILL)

    def test_settlement_uses_exact_wait_code_fim_logs_and_original_seal(self):
        run = self.root / 'attempt-01'
        run.mkdir()
        continuation.write_once(run / 'JOB_SUCCESS.json', {'formal_published_sha256': 'a'})
        (run / 'formal_access').mkdir()
        (run / 'formal_access/test.consumed.json').write_text('{}')
        log = self.root / 'attempt-01.log'
        log.write_text('FIM applied freeze: 0/12 bottom elements (fraction=0.000000)\n' * 2)
        proc = mock.Mock(returncode=0)
        proc.wait.return_value = 0
        legacy = SimpleNamespace(verify_formal_seal=mock.Mock(return_value={'formal_published_sha256': 'a'}))
        entry = dict(process=proc, row={'job_key': 'isolet/fim_off/42', 'arm': 'fim_off'},
                     config={'num_tasks': 2}, job_root=self.root, log=log,
                     binding={'execution_epoch_sha256': 'e' * 64})
        self.assertEqual(continuation.settle_worker(entry, legacy)['classification'], 'success')
        proc.wait.assert_called_once_with()
        proc.returncode = -9
        proc.wait.return_value = -9
        self.assertEqual(continuation.settle_worker(entry, legacy)['classification'], 'infrastructure_candidate')
        proc.returncode = 0
        proc.wait.return_value = 0
        legacy.verify_formal_seal.side_effect = ValueError('seal invalid')
        self.assertEqual(continuation.settle_worker(entry, legacy)['classification'], 'training_or_audit_failure')

    def test_cleanup_visits_every_peer_even_if_one_session_is_unverifiable(self):
        first, second = {'process': mock.Mock()}, {'process': mock.Mock()}
        with mock.patch.object(continuation, 'stop_owned_session', side_effect=[ValueError('identity'), None]) as stop:
            errors = continuation.stop_all_owned([first, second])
        self.assertEqual(stop.call_count, 2)
        self.assertEqual(errors, ['identity'])

    def test_worker_preserves_config_and_uses_real_single_publication_seal(self):
        from frozen_ablation_seal import verify_formal_seal
        from test_frozen_ablation_run import AblationRunTests
        producer_case = AblationRunTests()
        self.addCleanup(producer_case.doCleanups)
        fixture = producer_case.producer_fixture()
        key = 'isolet/fim_off/42'
        job_root = self.root / 'fim_off/isolet/42'
        job_root.mkdir(parents=True)
        config = dict(seed=42, device='cuda:0', num_tasks=1,
                      expected_party_kd_variant='original', formal_deferred_evaluation=True,
                      output_dir=str(job_root / 'attempt-01'))
        path = self.root / 'frozen-config.json'
        path.write_bytes((json.dumps(config, indent=2) + '\n').encode())
        original = path.read_bytes()
        row = dict(job_key=key, arm='fim_off', code_commit='d' * 40, config_sha256=continuation.sha(path))
        parent, mine = self.identity(123), self.identity(200)
        mine['ppid'] = parent['pid']
        epoch_path = self.root / 'gpu1-rebalance-v1/EXECUTION_EPOCH.json'
        roots = {'no_partykd': self.root, 'fim_off': self.root}
        auth = dict(row=row, config=config, config_path=path, job_root=job_root,
                    epoch=dict(controller_identity=parent, legacy_root=str(self.root.resolve()),
                               code_roots={k: str(v.resolve()) for k, v in roots.items()}),
                    expected_env=continuation.worker_environment(config, 1),
                    binding=continuation.execution_binding(row, 1, 'e' * 64, parent))
        protocol = epoch_path.parent / 'workers' / key.replace('/', '__')
        protocol.mkdir(parents=True)
        continuation.write_once(protocol / 'GO.json', dict(continuation.protocol_binding(auth), worker_identity=mine))
        def publish(args):
            # Move the real producer's sealed artifacts on the same filesystem,
            # retaining the inode/file records that its publication validates.
            for artifact in fixture.run.iterdir():
                artifact.rename(job_root / 'attempt-01' / artifact.name)
        legacy = SimpleNamespace(load_jobs=mock.Mock(return_value=[(row, config, path, job_root)]),
                                 runtime_identity=lambda: {'fixture': 'runtime'}, verify_formal_seal=verify_formal_seal)
        config_module = SimpleNamespace(validate_party_kd_variant=lambda *a: None,
            validate_sample_replay_only=lambda *a: None, validate_adaptive_head_consolidation=lambda *a: None)
        with mock.patch.object(continuation, 'authorize_worker', return_value=auth), \
             mock.patch.object(continuation, 'load_legacy', return_value=legacy), \
             mock.patch.object(continuation, 'process_identity', return_value=mine), \
             mock.patch.object(continuation.os, 'getpid', return_value=mine['pid']), \
             mock.patch.object(continuation, 'observed_cuda_order', return_value=auth['binding']['cuda_visible_devices']), \
             mock.patch.object(continuation.threading, 'Thread') as watchdog, \
             mock.patch.dict(os.environ, auth['expected_env']), \
             mock.patch.object(sys, 'path', list(sys.path)), \
             mock.patch.dict(sys.modules, {'config': config_module, 'runner': SimpleNamespace(run_experiment=publish)}):
            continuation.worker(self.root / 'manifest', 'm' * 64, key, epoch_path,
                                'e' * 64, self.root, roots)
            watchdog.return_value.start.assert_called_once_with()
        run = job_root / 'attempt-01'
        self.assertEqual((run / 'APPROVED_CONFIG.json').read_bytes(), original)
        self.assertEqual((run / 'config.json').read_bytes(), original)
        gate, runtime = continuation.read_json(run / 'JOB_SUCCESS.json'), continuation.read_json(run / 'JOB_RUNTIME.json')
        sealed = verify_formal_seal(run)
        self.assertTrue(all(gate[k] == value for k, value in sealed.items()))
        self.assertEqual(gate['execution_epoch_sha256'], 'e' * 64)
        self.assertEqual(runtime['observed_cuda_uuid_order'], [continuation.GPU_UUID[1], continuation.GPU_UUID[0]])
        self.assertEqual(gate['worker_identity'], mine)

    def test_old_parent_requires_external_identity_and_exact_pinned_command(self):
        parent = {**self.identity(50),
                  'cmdline': ['python', '-u', str(self.root / 'frozen_ablation_run.py'),
                              '--manifest', '/result/manifests/manifest.jsonl',
                              '--manifest-sha256', continuation.MANIFEST_SHA,
                              '--base-code', '/base', '--patch-code', '/patch']}
        roots = {'no_partykd': Path('/base'), 'fim_off': Path('/patch')}
        with mock.patch.object(continuation, 'process_identity', return_value=parent):
            continuation.validate_old_parent(parent, self.root, '/result/manifests/manifest.jsonl', roots)
            with self.assertRaises(ValueError):
                continuation.validate_old_parent({**parent, 'start_ticks': 61}, self.root,
                                                  '/result/manifests/manifest.jsonl', roots)

    def test_legacy_import_binds_original_manifest_even_when_new_manifest_is_cached(self):
        import frozen_ablation_manifest as current_manifest
        source = Path(continuation.__file__).parent
        old_root = self.root / 'old-origin'
        old_root.mkdir()
        for name in ('frozen_ablation_manifest.py', 'frozen_ablation_run.py'):
            (old_root / name).write_bytes((source / name).read_bytes())
        before_path = sys.path[:]
        with mock.patch.object(continuation, 'pinned_checkout', return_value=old_root), \
             mock.patch.object(sys, 'path', list(sys.path)):
            legacy = continuation.load_legacy(old_root)
        validator = legacy.validate_publication_preflight
        self.assertEqual(Path(validator.__globals__['__file__']), old_root / 'frozen_ablation_manifest.py')
        self.assertEqual(Path(validator.__code__.co_filename), old_root / 'frozen_ablation_manifest.py')
        self.assertEqual(sys.path, before_path)
        self.assertIs(sys.modules['frozen_ablation_manifest'], current_manifest)
        self.assertIsNot(validator, current_manifest.validate_publication_preflight)
        modules = ('test_frozen_ablation_run', 'test_frozen_ablation_audit',
                   'test_frozen_ablation_manifest', 'test_frozen_ablation_smoke',
                   'test_frozen_fim_application')
        for name in modules:
            (old_root / (name + '.py')).write_bytes((source / (name + '.py')).read_bytes())
        publication = self.root / 'publication'
        publication.mkdir()
        log = publication / 'publication-tests.log'
        log.write_text('test_full_linux_producer_fixture_passes_formal_seal '
                       '(test_frozen_ablation_run.AblationRunTests) ... ok\n')
        report = dict(status='passed', returncode=0, platform='linux',
                      linux_producer_fixture_passed=True, skipped_tests=0,
                      orchestration_commit=continuation.ORIGIN_COMMIT, real_test_access=False,
                      log_sha256=continuation.sha(log), test_source_sha256={
                          name: continuation.sha(old_root / (name + '.py')) for name in modules})
        report_path = publication / 'PUBLICATION_PREFLIGHT.json'
        continuation.write_once(report_path, report)
        with mock.patch.object(continuation.subprocess, 'check_output', return_value=continuation.ORIGIN_COMMIT + '\n'):
            self.assertEqual(validator(publication), continuation.sha(report_path))
            report_path.write_text(json.dumps({**report, 'orchestration_commit': '0' * 40}))
            with self.assertRaisesRegex(ValueError, 'did not pass this code'):
                validator(publication)
            report_path.write_text(json.dumps(report))
            (old_root / 'test_frozen_ablation_run.py').write_text('changed source')
            with self.assertRaisesRegex(ValueError, 'test sources changed'):
                validator(publication)

    def test_failed_legacy_import_restores_the_current_manifest_module(self):
        import frozen_ablation_manifest as current_manifest
        (self.root / 'frozen_ablation_manifest.py').write_bytes(Path(current_manifest.__file__).read_bytes())
        (self.root / 'frozen_ablation_run.py').write_text("raise RuntimeError('synthetic import failure')\n")
        with mock.patch.object(continuation, 'pinned_checkout', return_value=self.root), \
             mock.patch.object(sys, 'path', list(sys.path)):
            with self.assertRaisesRegex(RuntimeError, 'synthetic import failure'):
                continuation.load_legacy(self.root)
        self.assertIs(sys.modules['frozen_ablation_manifest'], current_manifest)

    def test_actual_cuda_order_requires_exactly_two_expected_uuids(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2,
            get_device_properties=lambda i: SimpleNamespace(uuid=continuation.GPU_UUID[1-i])))
        with mock.patch.dict('sys.modules', {'torch': fake}):
            self.assertEqual(continuation.observed_cuda_order(1),
                             [continuation.GPU_UUID[1], continuation.GPU_UUID[0]])

        fake.cuda.get_device_properties = lambda i: SimpleNamespace(uuid=continuation.GPU_UUID[1-i][4:])
        with mock.patch.dict('sys.modules', {'torch': fake}):
            self.assertEqual(continuation.observed_cuda_order(1),
                             [continuation.GPU_UUID[1], continuation.GPU_UUID[0]])
            with self.assertRaises(ValueError):
                continuation.observed_cuda_order(0)

    def test_handshake_is_complete_before_becoming_visible_and_never_overwrites(self):
        self.assertTrue(hasattr(continuation, 'publish_once'), 'atomic handshake publisher missing')
        path = self.root / 'READY.json'
        real_link = os.link
        def link(source, target):
            self.assertFalse(path.exists())
            self.assertEqual(json.loads(Path(source).read_text()), {'ready': True})
            return real_link(source, target)
        with mock.patch.object(continuation.os, 'link', side_effect=link):
            continuation.publish_once(path, {'ready': True})
        with self.assertRaises(FileExistsError):
            continuation.publish_once(path, {'changed': True})
        self.assertEqual(json.loads(path.read_text()), {'ready': True})

    def controller_fixture(self, mode='success'):
        from frozen_ablation_manifest import expected_keys
        selected = ['isolet/fim_off/43', 'isolet/no_partykd/44']
        success = sorted(expected_keys() - set(selected))
        prefix = b''.join((json.dumps(dict(kind=kind, job_key=k, attempt=1,
                         classification='success', returncode=0)) + '\n').encode()
                         for k in success for kind in ('start', 'finish'))
        (self.root / 'ATTEMPTS.jsonl').write_bytes(prefix)
        parent = self.identity(100)
        epoch = dict(manifest=str(self.root / 'manifest'), manifest_sha256='f' * 64,
                     assignments={selected[0]: 0, selected[1]: 1}, success_keys=success,
                     controller_identity=parent, orchestration_commit='e' * 40,
                     code_roots={'no_partykd': str(self.root), 'fim_off': str(self.root)},
                     legacy_root=str(self.root), ledger_prefix_bytes_length=len(prefix),
                     ledger_prefix_sha256=hashlib.sha256(prefix).hexdigest())
        extension = self.root / 'gpu1-rebalance-v1'
        extension.mkdir()
        epoch_path = extension / 'EXECUTION_EPOCH.json'
        continuation.write_once(epoch_path, epoch)
        epoch_sha = continuation.sha(epoch_path)
        jobs = []
        for key in selected:
            dataset, arm, seed = key.split('/')
            job_root = self.root / arm / dataset / seed
            config = {'device': 'cuda:0', 'seed': int(seed), 'num_tasks': 2,
                      'output_dir': str(job_root / 'attempt-01')}
            path = self.root / (arm + '-config.json')
            path.write_text(json.dumps(config))
            row = dict(job_key=key, arm=arm, code_commit='d' * 40,
                       config_sha256=continuation.sha(path))
            jobs.append((row, config, path, job_root))
        processes, identities, gos = [], {}, []
        owner = self
        class Process:
            def __init__(self, command, **unused):
                if mode == 'launch_failure':
                    raise OSError('synthetic launch failure')
                self.pid = 200 + len(processes)
                self.returncode = None
                self.command = command
                self.key = command[command.index('--job-key') + 1]
                self.job = next(j for j in jobs if j[0]['job_key'] == self.key)
                self.protocol = extension / 'workers' / self.key.replace('/', '__')
                mine = owner.identity(self.pid)
                mine['ppid'] = parent['pid']
                identities[self.pid] = mine
                binding = continuation.execution_binding(self.job[0], epoch['assignments'][self.key], epoch_sha, parent)
                auth = continuation.protocol_binding(dict(row=self.job[0], binding=binding))
                continuation.write_once(self.protocol / 'READY.json', dict(auth, worker_identity=mine))
                processes.append(self)
            def poll(self):
                if self.returncode is None and (self.protocol / 'GO.json').exists():
                    gos.append(self.key)
                    run = self.job[3] / 'attempt-01'
                    run.mkdir()
                    continuation.write_once(run / 'JOB_SUCCESS.json', {'formal_published_sha256': 'ok'})
                    (run / 'formal_access').mkdir()
                    (run / 'formal_access/test.consumed.json').write_text('{}')
                    (self.job[3] / 'attempt-01.log').write_text(
                        'FIM applied freeze: 0/12 bottom elements (fraction=0.000000)\n' * 2)
                    self.returncode = 0
                return self.returncode
            def wait(self):
                return self.returncode
        def identity(pid):
            if mode == 'identity_failure' and processes:
                raise ValueError('synthetic post-Popen identity failure')
            return identities[pid]
        def snapshot(index):
            if mode == 'peer_failure_during_admission' and processes:
                processes[0].returncode = 23
            return dict(index=index, uuid=continuation.GPU_UUID[index])
        legacy = SimpleNamespace(gpu_snapshot=snapshot, gpu_eligible=lambda s, i: True,
                                 verify_formal_seal=lambda run: {'formal_published_sha256': 'ok'})
        def stop(entry):
            if entry['process'].returncode is None:
                entry['process'].returncode = -9
        return SimpleNamespace(**locals())

    def run_controller_fixture(self, fixture):
        clock = iter(range(0, 100000, 61))
        with mock.patch.object(continuation.subprocess, 'Popen', fixture.Process), \
             mock.patch.object(continuation, 'process_identity', fixture.identity), \
             mock.patch.object(continuation, 'stop_owned_session', fixture.stop), \
             mock.patch.object(continuation.time, 'monotonic', side_effect=lambda: next(clock)), \
             mock.patch.object(continuation.time, 'sleep'):
            return continuation.controller_loop(fixture.jobs, fixture.legacy, self.root,
                                                fixture.epoch_path, fixture.epoch_sha)

    def test_controller_finishes_all_eighteen_and_retains_the_original_prefix(self):
        f = self.controller_fixture()
        result = self.run_controller_fixture(f)
        self.assertEqual(result['successes'], 18)
        raw = (self.root / 'ATTEMPTS.jsonl').read_bytes()
        self.assertTrue(raw.startswith(f.prefix))
        self.assertEqual(len(raw.splitlines()), 36)
        self.assertEqual(len(f.gos), 2)
        self.assertEqual(result['orchestration_commit'], 'e' * 40)
        self.assertEqual(result['original_orchestration_commit'], continuation.ORIGIN_COMMIT)

    def test_failure_seen_during_admission_prevents_any_next_start_or_go(self):
        f = self.controller_fixture('peer_failure_during_admission')
        with self.assertRaises(RuntimeError):
            self.run_controller_fixture(f)
        tail = (self.root / 'ATTEMPTS.jsonl').read_bytes()[len(f.prefix):].splitlines()
        self.assertEqual([json.loads(line)['kind'] for line in tail], ['start', 'finish'])
        self.assertEqual(f.gos, [])
        self.assertEqual(len(f.processes), 1)

    def test_launch_failure_consumes_attempt_and_postlaunch_identity_error_tracks_child(self):
        for mode in ('launch_failure', 'identity_failure'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                original = self.root
                self.root = Path(directory)
                try:
                    f = self.controller_fixture(mode)
                    with self.assertRaises((RuntimeError, ValueError)):
                        self.run_controller_fixture(f)
                    tail = (self.root / 'ATTEMPTS.jsonl').read_bytes()[len(f.prefix):].splitlines()
                    self.assertEqual([json.loads(line)['kind'] for line in tail], ['start', 'finish'])
                    self.assertTrue((self.root / 'QUEUE_FAILURE.json').is_file())
                    self.assertEqual(f.gos, [])
                finally:
                    self.root = original

    def wait_for(self, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.03)
        self.fail('synthetic subprocess did not reach its expected state')

    def alive(self, pid):
        try:
            return continuation.process_identity(pid)['state'] not in ('Z', 'X')
        except FileNotFoundError:
            return False

    @unittest.skipUnless(sys.platform == 'linux', 'Linux synthetic process sessions')
    def test_dead_worker_leader_cleanup_stops_its_surviving_loader(self):
        code = """import json, os, subprocess, sys, time
from pathlib import Path
p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path(sys.argv[1]).write_text(json.dumps({'pid': os.getpid(), 'child': p.pid}))
time.sleep(60)
"""
        path = self.root / 'leader.json'
        process = subprocess.Popen([sys.executable, '-c', code, str(path)], start_new_session=True)
        self.wait_for(path.exists)
        info = json.loads(path.read_text())
        expected = continuation.process_identity(process.pid)
        entry = dict(process=process, identity=expected)
        try:
            process.kill()
            process.wait(timeout=5)
            self.assertTrue(self.alive(info['child']))
            continuation.stop_owned_session(entry)
            self.wait_for(lambda: not self.alive(info['child']))
        finally:
            continuation.stop_owned_session(entry)
            process.wait(timeout=5)

    @unittest.skipUnless(sys.platform == 'linux', 'Linux synthetic process sessions')
    def test_live_worker_cleanup_handles_loader_reparenting_during_signals(self):
        code = """import json, os, subprocess, sys, time
from pathlib import Path
p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path(sys.argv[1]).write_text(json.dumps({'pid': os.getpid(), 'child': p.pid}))
time.sleep(60)
"""
        path = self.root / 'live-leader.json'
        process = subprocess.Popen([sys.executable, '-c', code, str(path)], start_new_session=True)
        self.wait_for(path.exists)
        info = json.loads(path.read_text())
        entry = dict(process=process, identity=continuation.process_identity(process.pid))
        try:
            continuation.stop_owned_session(entry)
            process.wait(timeout=5)
            self.wait_for(lambda: not self.alive(info['child']))
        finally:
            continuation.stop_owned_session(entry)
            process.wait(timeout=5)

    @unittest.skipUnless(sys.platform == 'linux', 'Linux synthetic parent-death watchdog')
    def test_real_parent_death_watchdog_stops_worker_and_loader_session(self):
        worker_code = """import json, os, subprocess, sys, threading, time
from pathlib import Path
import frozen_ablation_gpu1_continue as c
parent = c.process_identity(os.getppid())
finished = threading.Event()
threading.Thread(target=c.parent_watchdog, args=(parent, finished), daemon=True).start()
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path(sys.argv[1]).write_text(json.dumps({'worker': os.getpid(), 'child': child.pid}))
time.sleep(60)
"""
        launcher = """import subprocess, sys, time
from pathlib import Path
subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]], start_new_session=True)
while not Path(sys.argv[3]).exists(): time.sleep(.02)
"""
        info_path, exit_path = self.root / 'watchdog.json', self.root / 'exit'
        parent = subprocess.Popen([sys.executable, '-c', launcher, worker_code,
                                   str(info_path), str(exit_path)], cwd=Path(continuation.__file__).parent)
        try:
            self.wait_for(info_path.exists)
            info = json.loads(info_path.read_text())
            exit_path.touch()
            parent.wait(timeout=5)
            self.wait_for(lambda: not self.alive(info['worker']) and not self.alive(info['child']))
        finally:
            if parent.poll() is None:
                parent.kill()
            parent.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
