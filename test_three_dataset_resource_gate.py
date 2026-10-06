"""Generated authorities only; no training, historical writes, or success markers."""
import copy
import importlib
import importlib.util
import hashlib
import os
from pathlib import Path
import stat
import threading
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import three_dataset_formal_driver as driver
import three_dataset_formal_registry as registry
import test_three_dataset_full_matrix_report as fixtures
from test_three_dataset_formal_driver import _completed_record
import test_three_dataset_formal_driver as driver_fixtures


def _bounded_flock(real_flock, fd, operation, timeout=10):
    """Keep real flock contention, but never strand executor shutdown on a regression."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return real_flock(fd, operation | driver.fcntl.LOCK_NB)
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise TimeoutError('test flock acquisition timed out')
            time.sleep(.01)


class BoundedFlockTests(unittest.TestCase):
    def test_recursive_lock_times_out_and_original_lock_remains_held(self):
        with tempfile.TemporaryDirectory(prefix='generated_flock_timeout_') as root:
            first = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            second = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                driver.fcntl.flock(first, driver.fcntl.LOCK_EX)
                with self.assertRaisesRegex(TimeoutError, 'test flock acquisition timed out'):
                    _bounded_flock(driver.fcntl.flock, second, driver.fcntl.LOCK_EX,
                                   timeout=.02)
                with self.assertRaises(BlockingIOError):
                    driver.fcntl.flock(second, driver.fcntl.LOCK_EX | driver.fcntl.LOCK_NB)
                driver.fcntl.flock(first, driver.fcntl.LOCK_UN)
                _bounded_flock(driver.fcntl.flock, second, driver.fcntl.LOCK_EX)
            finally:
                os.close(second)
                os.close(first)


class SingleDatasetDiskGateTests(unittest.TestCase):
    def test_continuation_reserves_only_missing_twenty_four_with_same_safety(self):
        gate = importlib.import_module('three_dataset_resource_gate')
        with mock.patch.dict(os.environ, {
                registry.PROFILE_ENV: registry.CONTINUATION_PROFILE,
                registry.DATASET_ENV: 'cifar100', 'VFCL_GPU_COUNT': '2',
                'VFCL_PYTHON': str(registry._REVIEWED_VFCL_PYTHON)}), \
                tempfile.TemporaryDirectory(prefix='generated_continuation_disk_') as temp:
            root = Path(temp) / 'formal'
            root.mkdir(mode=0o700)
            missing = [driver.spec_key(spec) for spec in registry.formal_specs()
                       if spec.method not in {
                           'finetune', 'lwf', 'ewc', 'er', 'der_pp', 'er_ace'}]
            plan = {'missing_jobs': missing}
            with mock.patch.object(driver, '_load_installed_plan',
                                   return_value=(plan, {})), \
                    mock.patch.object(driver, '_installed_completed_record',
                                      return_value=None) as completed, \
                    mock.patch.object(os, 'statvfs', return_value=SimpleNamespace(
                        f_bavail=94 * gate.GIB, f_frsize=1)):
                status = gate.scoped_disk_status(root, requested_slots=2)
            self.assertEqual(24, status['remaining_jobs'])
            self.assertEqual(20 * gate.GIB, status['safety_bytes'])
            self.assertEqual(2, status['requested_slots'])
            self.assertTrue(status['safe'])
            self.assertEqual(set(missing), {
                driver.spec_key(call.args[1]) for call in completed.call_args_list})
            self.assertEqual(24, completed.call_count)

    def test_scoped_two_gpu_requires_an_additional_peak_and_matching_slots(self):
        gate = importlib.import_module('three_dataset_resource_gate')
        with mock.patch.dict(os.environ, {
                registry.PROFILE_ENV: registry.SINGLE_DATASET_PROFILE,
                registry.DATASET_ENV: 'cifar100', 'VFCL_GPU_COUNT': '2',
                'VFCL_PYTHON': str(registry._REVIEWED_VFCL_PYTHON)}), \
                tempfile.TemporaryDirectory(prefix='generated_scoped_disk_') as temp:
            root = Path(temp) / 'formal'
            driver._install_census(root, {})
            driver._install_plan(root)
            with mock.patch.object(os, 'statvfs', return_value=SimpleNamespace(
                    f_bavail=100 * gate.GIB, f_frsize=1)):
                with mock.patch.dict(os.environ, {'VFCL_GPU_COUNT': '1'}):
                    one = gate.scoped_disk_status(root, requested_slots=1)
                two = gate.scoped_disk_status(root, requested_slots=2)
            self.assertEqual(2, two['requested_slots'])
            self.assertEqual(30 * gate.GIB, one['safety_bytes'])
            self.assertEqual(20 * gate.GIB, two['safety_bytes'])
            self.assertEqual(two['required_bytes'] - one['required_bytes'],
                             two['active_peaks_bytes'] - one['active_peaks_bytes']
                             - 10 * gate.GIB)
            self.assertGreater(two['required_bytes'], one['required_bytes'])
            self.assertTrue(one['safe'])
            self.assertFalse(two['safe'])
            with self.assertRaisesRegex(ValueError, 'GPU count'):
                gate.scoped_disk_status(root, requested_slots=3)

    def test_scoped_admission_reserves_peak_retention_and_safety(self):
        gate = importlib.import_module('three_dataset_resource_gate')
        for dataset, low, high in (
                ('isolet', 30, 40),
                ('upmc_food101', 40, 50),
                ('cifar100', 90, 110)):
            with self.subTest(dataset=dataset), mock.patch.dict(os.environ, {
                    registry.PROFILE_ENV: registry.SINGLE_DATASET_PROFILE,
                    registry.DATASET_ENV: dataset,
                    'VFCL_PYTHON': str(registry._REVIEWED_VFCL_PYTHON)}), \
                    tempfile.TemporaryDirectory(prefix='generated_scoped_disk_') as temp:
                root = Path(temp) / 'formal'
                driver._install_census(root, {})
                driver._install_plan(root)
                for available, safe in ((low, False), (high, True)):
                    with mock.patch.object(os, 'statvfs', return_value=SimpleNamespace(
                            f_bavail=available * gate.GIB, f_frsize=1)):
                        status = gate.scoped_disk_status(root)
                    self.assertEqual(dataset, status['dataset'])
                    self.assertEqual(42, status['remaining_jobs'])
                    self.assertEqual(30 * gate.GIB, status['safety_bytes'])
                    self.assertIs(safe, status['safe'])

    def test_non_cifar_two_gpu_keeps_30_gib_safety(self):
        gate = importlib.import_module('three_dataset_resource_gate')
        with mock.patch.dict(os.environ, {
                registry.PROFILE_ENV: registry.SINGLE_DATASET_PROFILE,
                registry.DATASET_ENV: 'isolet', 'VFCL_GPU_COUNT': '2',
                'VFCL_PYTHON': str(registry._REVIEWED_VFCL_PYTHON)}), \
                tempfile.TemporaryDirectory(prefix='generated_scoped_disk_') as temp:
            root = Path(temp) / 'formal'
            driver._install_census(root, {})
            driver._install_plan(root)
            with mock.patch.object(os, 'statvfs', return_value=SimpleNamespace(
                    f_bavail=100 * gate.GIB, f_frsize=1)):
                status = gate.scoped_disk_status(root, requested_slots=2)
            self.assertEqual(30 * gate.GIB, status['safety_bytes'])


class ResourceGateTests(unittest.TestCase):
    make_bundle = fixtures.FullMatrixReportTests.make_bundle
    publish = fixtures.FullMatrixReportTests.publish

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('three_dataset_resource_gate'),
                             'disk reservation gate is missing')
        self.gate = importlib.import_module('three_dataset_resource_gate')
        profile = mock.patch.dict(os.environ, {
            registry.PROFILE_ENV: registry.FULL_MATRIX_PROFILE,
            'VFCL_PYTHON': str(registry._REVIEWED_VFCL_PYTHON)})
        profile.start()
        self.addCleanup(profile.stop)
        temp = tempfile.TemporaryDirectory(prefix='generated_disk_gate_')
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.bundle = self.make_bundle(0)
        self.root = self.base / 'authority'
        driver._install_full_census(self.root, self.publish())
        self.plan = driver._install_plan(self.root)
        self.owner = driver.owner_for(self.root, 'formal', 'generated', 'worker-0',
                                      os.getpid(), os.getpgid(os.getpid()))
        self.free = mock.patch.object(os, 'statvfs', return_value=SimpleNamespace(
            f_bavail=1000 * 1024 ** 3, f_frsize=1))
        self.free.start()
        self.addCleanup(self.free.stop)

    def claim(self, owner=None):
        return driver.claim_next(self.plan, self.root / 'claims', 'formal',
                                 self.owner if owner is None else owner)

    def receipt(self, key):
        return self.root / 'claims' / driver._claim_name(key) / 'disk-reservation.json'

    def resource_run(self, method):
        with torch.random.fork_rng(devices=[]):
            spec, checkpoint, added = driver_fixtures.MethodResourceStateTests()._fixture(method)
        key = driver.spec_key(spec)
        owner = {**self.owner, 'job': key, 'worker_role': f'resource-{method}'}
        self.assertEqual(key, self.claim(owner))
        run = driver.prepare_run(self.root, key, owner)
        for logical in driver.resource_artifact_names(spec):
            path = run / logical
            path.parent.mkdir(parents=True, exist_ok=True)
            if logical == 'checkpoints/formal_final.pt':
                torch.save(checkpoint, path)
            elif logical == 'results.json':
                path.write_bytes(driver._canonical_json({
                    'comm_stats': [{'megabytes_transmitted': .25},
                                   {'megabytes_transmitted': .5}],
                    'replay_type': 'none', 'raw_examples_per_class': 999,
                    'persistent_embeddings': 999, 'added_parameters': 999,
                    'privacy_label': 'forged-results-privacy',
                }))
            else:
                path.write_bytes(b'generated resource artifact\n')
            path.chmod(0o444)
        return spec, key, run, added

    def publish_resource(self, key, run):
        real_run = driver.subprocess.run

        def subprocess_run(command, *args, **kwargs):
            if command[0] == 'nvidia-smi':
                return SimpleNamespace(stdout='generated-driver\n')
            return real_run(command, *args, **kwargs)

        with mock.patch.object(torch.cuda, 'is_available', return_value=True), \
                mock.patch.object(torch.cuda, 'device_count', return_value=1), \
                mock.patch.object(torch.cuda, 'get_device_name', return_value='generated-gpu'), \
                mock.patch.object(driver.subprocess, 'run', side_effect=subprocess_run):
            return driver.resource_record(self.root, key, run,
                runtime_seconds=3.25, peak_gpu_memory_bytes=123456)

    def test_resource_record_uses_checkpoint_not_results_and_preserves_measurements(self):
        expected = {
            'er': ('raw-examples', 2, 0, 'persistent-raw-example-replay'),
            'er_ace': ('reservoir-raw-examples', 2, 0, 'persistent-raw-example-replay'),
            'der_pp': ('reservoir-raw-examples-and-logits', 2, 0,
                       'persistent-raw-example-replay'),
            'proto_fedspace': ('class-prototype-embeddings', 0, 3,
                               'persistent-derived-embedding-replay'),
            'adaptive': ('raw-examples-and-class-prototype-embeddings', 2, 3,
                         'persistent-raw-and-derived-embedding-replay'),
            'adagauss': ('class-gaussian-statistics', 0, 3,
                         'persistent-derived-statistics-replay'),
            'target': ('synthetic-generator', 0, 0, 'persistent-synthetic-generator-replay'),
        }
        for method, (replay, raw, embeddings, privacy) in expected.items():
            with self.subTest(method=method):
                spec, key, run, added = self.resource_run(method)
                artifact_hashes = {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                                   for name in driver.resource_artifact_names(spec)}
                evidence = self.publish_resource(key, run)
                self.assertEqual({
                    'hardware_identity': {'gpu_name': 'generated-gpu', 'gpu_count': 1,
                        'cuda': str(torch.version.cuda), 'torch': str(torch.__version__),
                        'driver': 'generated-driver'},
                    'instrumentation': 'formal-resource-v1',
                    'runtime_seconds': 3.25, 'peak_gpu_memory_bytes': 123456,
                    'communication_bytes': 786432,
                    'checkpoint_size_bytes': (run / 'checkpoints/formal_final.pt').stat().st_size,
                    'replay_type': replay, 'raw_examples_per_class': raw,
                    'persistent_embeddings': embeddings, 'added_parameters': added,
                    'privacy_label': privacy,
                }, evidence['resource'])
                self.assertEqual(artifact_hashes, evidence['artifact_sha256'])
                path = run / 'RESOURCE_EVIDENCE.json'
                content, details = path.read_bytes(), path.stat()
                self.assertEqual(driver._canonical_json(evidence) + b'\n', content)
                self.assertEqual(0o444, stat.S_IMODE(details.st_mode))
                self.assertEqual(1, details.st_nlink)
                with self.assertRaises(FileExistsError):
                    self.publish_resource(key, run)
                self.assertEqual(content, path.read_bytes())
                self.assertEqual((details.st_ino, details.st_mtime_ns),
                                 (path.stat().st_ino, path.stat().st_mtime_ns))

    def test_supplied_resource_semantics_must_exactly_match_checkpoint(self):
        spec, key, run, _ = self.resource_run('adagauss')
        expected = driver._method_resource_state(spec, run / 'checkpoints/formal_final.pt')
        supplied = {k: v for k, v in _completed_record(spec, .5)['resource'].items()
                    if k not in {'instrumentation', 'checkpoint_size_bytes'}}
        supplied.update(expected)
        for field, wrong in (('replay_type', 'none'), ('raw_examples_per_class', 1),
                ('persistent_embeddings', 0), ('added_parameters', 0),
                ('privacy_label', 'no-persistent-raw-or-embedding-replay'),
                ('raw_examples_per_class', False), ('added_parameters', float(expected['added_parameters']))):
            with self.subTest(field=field, wrong=wrong):
                path = run / 'RESOURCE_EVIDENCE.json'
                try:
                    with self.assertRaisesRegex(ValueError, 'checkpoint'):
                        driver.resource_record(self.root, key, run,
                            measurements={**supplied, field: wrong})
                    self.assertFalse(path.exists())
                finally:
                    if path.exists():
                        path.unlink()
        before = copy.deepcopy(supplied)
        evidence = driver.resource_record(self.root, key, run, measurements=supplied)
        self.assertEqual(before, supplied)
        self.assertEqual(expected, {k: evidence['resource'][k] for k in expected})
        self.assertEqual(supplied['hardware_identity'], evidence['resource']['hardware_identity'])

    def test_mismatched_checkpoint_rejects_both_resource_publication_paths(self):
        spec, key, run, _ = self.resource_run('target')
        supplied = {k: v for k, v in _completed_record(spec, .5)['resource'].items()
                    if k not in {'instrumentation', 'checkpoint_size_bytes'}}
        checkpoint = run / 'checkpoints/formal_final.pt'
        supplied.update(driver._method_resource_state(spec, checkpoint))
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        payload['protocol']['cl_method'] = 'finetune'
        checkpoint.chmod(0o644)
        torch.save(payload, checkpoint)
        checkpoint.chmod(0o444)
        before = checkpoint.read_bytes()
        for injected in (False, True):
            with self.subTest(injected=injected):
                path = run / 'RESOURCE_EVIDENCE.json'
                try:
                    with self.assertRaisesRegex(ValueError, 'method'):
                        if injected:
                            driver.resource_record(self.root, key, run, measurements=supplied)
                        else:
                            self.publish_resource(key, run)
                    self.assertFalse(path.exists())
                    self.assertEqual(before, checkpoint.read_bytes())
                finally:
                    if path.exists():
                        path.unlink()

    def test_one_slot_safe_two_unsafe_exact_arithmetic(self):
        one = self.gate.disk_status(self.root, 1)
        two = self.gate.disk_status(self.root, 2)
        threshold = sum(one[k] for k in ('safety_bytes', 'active_reservation_bytes',
                                        'requested_reservation_bytes', 'predicted_retained_remainder_bytes'))
        with mock.patch.object(os, 'statvfs', return_value=SimpleNamespace(f_bavail=threshold, f_frsize=1)):
            one = self.gate.disk_status(self.root, 1)
            two = self.gate.disk_status(self.root, 2)
        self.assertTrue(one['safe'])
        self.assertFalse(two['safe'])
        self.assertEqual(set(one), {'kind', 'available_bytes', 'safety_bytes',
            'active_reservation_bytes', 'requested_reservation_bytes',
            'predicted_retained_remainder_bytes', 'requested_slots', 'safe', 'plan_sha256'})
        self.assertEqual(one['kind'], 'full_matrix_disk_status_v1')
        self.assertEqual(one['plan_sha256'], driver._digest(self.plan))
        self.assertEqual(one['safety_bytes'], 30 * 1024 ** 3)

    def test_frozen_dataset_maxima_are_conservative(self):
        self.assertEqual(driver._digest(self.gate.OBSERVATIONS),
                         'a758d2fd6088408141181e9f62cef5d76dc7454ee772a45ebeedb5043052a9e6')
        self.assertEqual(self.gate.dataset_estimates(), {
            'cifar100': (16 * 1024 ** 3, 969779196),
            'isolet': (1024 ** 3, 16544177),
            'upmc_food101': (3 * 1024 ** 3, 199164899)})

    def test_invalid_sizes_membership_and_missing_estimates_fail_closed(self):
        for value in (-1, True, 1.5):
            for field in ('peak_bytes', 'retained_bytes'):
                rows = copy.deepcopy(self.gate.OBSERVATIONS)
                rows[0][field] = value
                with self.subTest(field=field, value=value), mock.patch.object(self.gate, 'OBSERVATIONS', rows):
                    with self.assertRaises(ValueError):
                        self.gate.disk_status(self.root, 1)
        for field, value in [('spec_key', 'cifar100:unknown:42'),
                             ('spec_key', 'unknown:er:42'), ('model_family', 'wrong')]:
            rows = copy.deepcopy(self.gate.OBSERVATIONS)
            rows[0][field] = value
            with self.subTest(field=field), mock.patch.object(self.gate, 'OBSERVATIONS', rows):
                with self.assertRaises(ValueError):
                    self.gate.disk_status(self.root, 1)
        with mock.patch.object(self.gate, 'OBSERVATIONS', []):
            with self.assertRaisesRegex(ValueError, '^disk estimate unavailable$'):
                self.gate.disk_status(self.root, 1)
        for value in (True, 0, 3, -1):
            with self.subTest(slots=value), self.assertRaises(ValueError):
                self.gate.disk_status(self.root, value)

    def test_claim_has_immutable_owner_bound_receipt_and_release_is_exact(self):
        key = self.claim()
        path = self.receipt(key)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)
        receipt = driver._load_json_file(path)
        owner = driver.installed_claim_owner(self.root, key)
        self.assertEqual(receipt['owner_sha256'], driver._file_digest(owner))
        status = self.gate.disk_status(self.root, 1)
        self.assertEqual(status['active_reservation_bytes'], 16 * 1024 ** 3)
        with self.assertRaises(ValueError):
            driver.release_prelaunch_claim(path.parent, {**owner, 'worker_role': 'wrong'})
        self.assertTrue(path.exists())
        driver.release_prelaunch_claim(path.parent, owner)
        self.assertFalse(path.parent.exists())

    def test_duplicate_worker_reservation_rejected(self):
        self.claim()
        with self.assertRaisesRegex(ValueError, 'duplicate.*reservation'):
            self.claim()

    def test_gpu_owned_provisional_reservation_cannot_be_released(self):
        key = self.claim()
        owner = driver.installed_claim_owner(self.root, key)
        self.assertTrue(driver.claim_gpu(self.root, 0, owner))
        with self.assertRaisesRegex(ValueError, 'GPU'):
            driver.release_prelaunch_claim(self.receipt(key).parent, owner)
        self.assertTrue(self.receipt(key).exists())
        driver.release_gpu(self.root, 0, owner)
        driver.release_prelaunch_claim(self.receipt(key).parent, owner)

    def test_gpu_claim_requires_the_spec_reservation(self):
        owner = dict(self.owner, job=self.plan['missing_jobs'][0])
        with self.assertRaises((ValueError, OSError)):
            driver.claim_gpu(self.root, 0, owner)
        self.assertFalse((self.root / 'gpu_claims').exists())

    def test_concurrent_claims_serialize_disk_admission(self):
        one = self.gate.disk_status(self.root, 1)
        threshold = sum(one[k] for k in ('safety_bytes', 'requested_reservation_bytes',
                                        'predicted_retained_remainder_bytes'))
        # Different roles represent two generated worker identities.
        owners = [dict(self.owner, worker_role=f'worker-{i}') for i in range(2)]
        def attempt(owner):
            try:
                return self.claim(owner)
            except driver.PipelineBackpressure:
                return None
        with mock.patch.object(os, 'statvfs', return_value=SimpleNamespace(f_bavail=threshold, f_frsize=1)):
            with ThreadPoolExecutor(max_workers=2) as workers:
                results = list(workers.map(attempt, owners))
        self.assertEqual(sum(key is not None for key in results), 1)
        self.assertEqual(len(list((self.root / 'claims').iterdir())), 1)

    def test_dead_or_stale_owner_is_not_zero_reservation(self):
        self.claim()
        with mock.patch.object(driver, '_process_start_time', return_value='not-the-owner'):
            with self.assertRaises(ValueError):
                self.gate.disk_status(self.root, 1)

    def test_receipt_symlink_noncanonical_and_stale_binding_rejected(self):
        key = self.claim()
        path = self.receipt(key)
        original = path.read_bytes()
        for kind in ('noncanonical', 'plan', 'owner', 'root', 'symlink', 'size'):
            if path.is_symlink():
                path.unlink()
                path.write_bytes(original)
            path.chmod(0o644)
            path.write_bytes(original)
            value = driver._load_json_file(path)
            if kind == 'noncanonical':
                path.write_bytes(b' ' + original)
            elif kind == 'symlink':
                alternate = self.base / 'receipt-copy'
                alternate.write_bytes(original)
                alternate.chmod(0o444)
                path.unlink()
                path.symlink_to(alternate)
            else:
                field = {'plan': 'plan_sha256', 'owner': 'owner_sha256',
                         'root': 'root_identity', 'size': 'reservation_bytes'}[kind]
                value[field] = True if kind == 'size' else ('f' * 64 if kind != 'root' else {})
                path.write_bytes(driver._canonical_json(value) + b'\n')
            path.chmod(0o444)
            with self.subTest(kind=kind), self.assertRaises((ValueError, OSError)):
                self.gate.disk_status(self.root, 1)

    def test_stale_installed_plan_rejected(self):
        path = self.root / 'FORMAL_PLAN.json'
        plan = copy.deepcopy(self.plan)
        plan['missing_jobs'] = plan['missing_jobs'][1:]
        path.chmod(0o644)
        path.write_bytes(driver._canonical_json(plan) + b'\n')
        path.chmod(0o444)
        with self.assertRaises(ValueError):
            self.gate.disk_status(self.root, 1)

    def test_driver_disk_status_action(self):
        with mock.patch.object(driver, '_emit') as emit:
            driver.main(['disk-status', '--root', str(self.root), '--requested-slots', '2'])
        self.assertEqual(emit.call_args.args[0], self.gate.disk_status(self.root, 2))

    def start(self, key):
        path = self.receipt(key)
        owner = driver.installed_claim_owner(self.root, key)
        # Fixture preparation precedes the interleaving-specific flock patches.
        real_flock = driver.fcntl.flock
        with mock.patch.object(driver.fcntl, 'flock', side_effect=lambda fd, operation:
                               _bounded_flock(real_flock, fd, operation)):
            run = driver.prepare_run(self.root, key, owner)
        started = driver._load_json_file(path.parent / 'started.json')
        return owner, run, started

    def complete(self, key, run, foreign_owner=False):
        spec = driver.spec_for_key(key)
        record = _completed_record(spec, .5, source_commit=driver._source_commit(),
                                   plan_sha256=driver._digest(self.plan))
        job = driver._load_json_file(run / 'FORMAL_JOB_SPEC.json')
        owner = driver._load_json_file(run / 'CLAIM_OWNER.json')
        launch = driver._load_json_file(run / 'LAUNCH_STARTED.json')
        record.update(claim_sha256=driver._file_digest(owner),
                      launch_sha256=driver._file_digest(launch), command_sha256=job['command_sha256'])
        if foreign_owner:
            record['claim_sha256'] = 'f' * 64
        record['record_sha256'] = driver._digest({k: v for k, v in record.items() if k != 'record_sha256'})
        records = self.root / 'records'
        records.mkdir(exist_ok=True)
        driver.install_json_exclusive(records / (driver._claim_name(key) + '.json'), record)
        evidence = {'kind': 'formal_resource_evidence', 'spec_key': key,
            'plan_sha256': driver._digest(self.plan), 'job_spec_sha256': driver._file_digest(job),
            **{k: record[k] for k in ('claim_sha256', 'launch_sha256', 'command_sha256', 'resource')},
            'artifact_sha256': {**{name: 'b' * 64 for name in driver.resource_artifact_names(spec)},
                               'job.log': record['log_sha256']}}
        driver.install_json_exclusive(run / 'RESOURCE_EVIDENCE.json', evidence)
        (run / 'checkpoints').mkdir()
        common = {'policy': 'completed-run-intermediate-v1', 'spec_key': key,
                  'source_commit': record['source_commit'], 'record_sha256': record['record_sha256']}
        prune = {'kind': 'formal_prune_plan', **common, 'files': [
            {'path': 'checkpoints/event_0_CIL.pt', 'sha256': 'b' * 64, 'size': 100}], 'bytes': 100}
        driver.install_json_exclusive(run / 'PRUNE_PLAN.json', prune)
        driver.install_json_exclusive(run / 'PRUNED_EVIDENCE.json', {
            'kind': 'formal_pruned_evidence', **common,
            'plan_sha256': driver._file_digest(prune), 'freed_bytes': 100})

    def started_publication_overlap(self, prepare):
        key = self.claim()
        owner = driver.installed_claim_owner(self.root, key)
        claim = self.receipt(key).parent
        run = self.root / 'runs' / driver._claim_name(key)
        if not prepare:
            run.mkdir(parents=True)
        linked, release, attempted = (threading.Event() for _ in range(3))
        blocked = []
        real_link, real_flock = os.link, driver.fcntl.flock

        def link(source, destination, *args, **kwargs):
            result = real_link(source, destination, *args, **kwargs)
            if destination == 'started.json':
                self.assertEqual(os.stat(claim / destination).st_nlink, 2)
                linked.set()
                self.assertTrue(release.wait(10), 'started publication was not released')
            return result

        def flock(fd, operation):
            if (threading.current_thread().name.startswith('disk-reader')
                    and operation == driver.fcntl.LOCK_EX):
                try:
                    real_flock(fd, operation | driver.fcntl.LOCK_NB)
                except BlockingIOError:
                    blocked.append(True)
                    attempted.set()
                    return _bounded_flock(real_flock, fd, operation)
                blocked.append(False)
                attempted.set()
                return None
            return _bounded_flock(real_flock, fd, operation)

        with mock.patch.object(os, 'link', side_effect=link), \
                mock.patch.object(driver.fcntl, 'flock', side_effect=flock), \
                ThreadPoolExecutor(max_workers=1) as publisher, \
                ThreadPoolExecutor(max_workers=1, thread_name_prefix='disk-reader') as reader:
            if prepare:
                published = publisher.submit(driver.prepare_run, self.root, key, owner)
            else:
                published = publisher.submit(driver.mark_claim_started, claim, owner,
                                             run, self.plan, 'a' * 64)
            try:
                self.assertTrue(linked.wait(10), 'started link window was not reached')
                status = reader.submit(self.gate.disk_status, self.root, 1)
                self.assertTrue(attempted.wait(10), 'disk reader did not attempt claims lock')
                # If unprotected, finish the real read before closing the link window.
                if not blocked[0]:
                    try:
                        status.result(timeout=10)
                    except ValueError:
                        pass
            finally:
                release.set()
            published.result(timeout=10)
            try:
                result = status.result(timeout=10)
            except ValueError as error:
                self.fail(f'disk status observed started publication: {error}')
        self.assertEqual(blocked, [True])
        self.assertTrue(result['safe'])
        self.assertEqual(result['active_reservation_bytes'], 16 * 1024 ** 3)
        self.assertEqual({path.name for path in claim.iterdir()},
                         {'owner.json', 'disk-reservation.json', 'started.json'})
        self.assertEqual((claim / 'started.json').stat().st_nlink, 1)
        self.assertEqual(driver._read_started_claim(
            claim, key, driver._root_identity(self.root), driver._source_commit())[0], owner)
        if prepare:
            self.assertEqual({path.name for path in run.iterdir()},
                             {'FORMAL_JOB_SPEC.json', 'CLAIM_OWNER.json', 'LAUNCH_STARTED.json'})

    def test_disk_status_waits_for_direct_started_publication(self):
        self.started_publication_overlap(prepare=False)

    def test_disk_status_waits_for_prepare_started_publication(self):
        self.started_publication_overlap(prepare=True)

    def test_disk_settlement_serializes_legal_sibling_prepare(self):
        key = self.claim()
        _, run, _ = self.start(key)
        self.complete(key, run)
        sibling = self.claim(dict(self.owner, worker_role='worker-1'))
        owner = driver.installed_claim_owner(self.root, sibling)
        sibling_run = self.root / 'runs' / driver._claim_name(sibling)
        pinned, release, attempted = (threading.Event() for _ in range(3))
        blocked = []
        real_pruning = self.gate.pruning
        real_mkdir, real_flock = driver._mkdir_exclusive, driver.fcntl.flock

        def pruning(evidence, *args):
            # RESOURCE_EVIDENCE.json has already pinned the shared runs directory.
            pinned.set()
            self.assertTrue(release.wait(10), 'disk settlement was not released')
            return real_pruning(evidence, *args)

        def mkdir(parent, name):
            result = real_mkdir(parent, name)
            if Path(parent) / name == sibling_run:
                blocked.append(False)
                attempted.set()
            return result

        def flock(fd, operation):
            if (threading.current_thread().name.startswith('sibling-prepare')
                    and operation == driver.fcntl.LOCK_EX):
                try:
                    real_flock(fd, operation | driver.fcntl.LOCK_NB)
                except BlockingIOError:
                    blocked.append(True)
                    attempted.set()
                    return _bounded_flock(real_flock, fd, operation)
                return None
            return _bounded_flock(real_flock, fd, operation)

        with mock.patch.object(self.gate, 'pruning', side_effect=pruning), \
                mock.patch.object(driver, '_mkdir_exclusive', side_effect=mkdir), \
                mock.patch.object(driver.fcntl, 'flock', side_effect=flock), \
                ThreadPoolExecutor(max_workers=1) as reader, \
                ThreadPoolExecutor(max_workers=1, thread_name_prefix='sibling-prepare') as publisher:
            status = reader.submit(self.gate.disk_status, self.root, 1)
            try:
                self.assertTrue(pinned.wait(10), 'settlement did not pin runs')
                prepared = publisher.submit(driver.prepare_run, self.root, sibling, owner)
                self.assertTrue(attempted.wait(10), 'sibling prepare did not reach lock or mkdir')
                if blocked[0]:
                    self.assertFalse(sibling_run.exists())
                else:
                    prepared.result(timeout=10)
            finally:
                release.set()
            self.assertEqual(prepared.result(timeout=10), sibling_run)
            try:
                result = status.result(timeout=10)
            except ValueError as error:
                self.fail(f'disk settlement raced legal sibling prepare: {error}')
        self.assertTrue(blocked[0])
        self.assertEqual(result['active_reservation_bytes'], 16 * 1024 ** 3)
        self.assertEqual(self.gate.disk_status(self.root, 1), result)
        self.assertEqual({path.name for path in sibling_run.iterdir()},
                         {'FORMAL_JOB_SPEC.json', 'CLAIM_OWNER.json', 'LAUNCH_STARTED.json'})

    def test_started_receipt_is_bound_and_never_prelaunch_released(self):
        key = self.claim()
        owner, run, started = self.start(key)
        self.assertEqual(started['disk_reservation_sha256'],
                         driver._sha256_bytes(self.receipt(key).read_bytes()))
        with self.assertRaises(ValueError):
            driver.release_prelaunch_claim(self.receipt(key).parent, owner)
        self.assertTrue(self.receipt(key).exists())

    def test_disk_status_waits_for_completed_record_publication(self):
        key = self.claim()
        _, run, _ = self.start(key)
        self.complete(key, run)
        spec = driver.spec_for_key(key)
        path = self.root / 'records' / (driver._claim_name(key) + '.json')
        record = driver._load_json_file(path)
        path.unlink()  # Keep generated resource/retention evidence for the real reader.
        linked, release, attempted = (threading.Event() for _ in range(3))
        sampled, unlinked = threading.Event(), threading.Event()
        blocked = []
        real_link, real_flock = os.link, driver.fcntl.flock
        real_unlink, real_read = os.unlink, driver._read_descriptor

        def compute_record(*args, **kwargs):
            # Scientific audit must remain outside the publication lock.
            status = self.gate.disk_status(self.root, 1)
            self.assertEqual(status['active_reservation_bytes'], 16 * 1024 ** 3)
            return record

        def link(source, destination, *args, **kwargs):
            result = real_link(source, destination, *args, **kwargs)
            if destination == path.name:
                self.assertEqual(path.stat().st_nlink, 2)
                linked.set()
                self.assertTrue(release.wait(10), 'completed publication was not released')
            return result

        def unlink(name, *args, **kwargs):
            result = real_unlink(name, *args, **kwargs)
            if str(name).startswith('.' + path.name + '.'):
                unlinked.set()
            return result

        def read(descriptor):
            content, details = real_read(descriptor)
            if (threading.current_thread().name.startswith('disk-reader')
                    and details.st_ino == path.stat().st_ino and details.st_nlink == 2):
                # Let publication finish between the real descriptor and name checks.
                sampled.set()
                self.assertTrue(unlinked.wait(10), 'completed temporary link was not removed')
            return content, details

        def flock(fd, operation):
            if (threading.current_thread().name.startswith('disk-reader')
                    and operation == driver.fcntl.LOCK_EX):
                try:
                    real_flock(fd, operation | driver.fcntl.LOCK_NB)
                except BlockingIOError:
                    blocked.append(True)
                    attempted.set()
                    return _bounded_flock(real_flock, fd, operation)
                blocked.append(False)
                attempted.set()
                return None
            return _bounded_flock(real_flock, fd, operation)

        with mock.patch.object(driver, 'completed_run_record', side_effect=compute_record) as audit, \
                mock.patch.object(os, 'link', side_effect=link), \
                mock.patch.object(os, 'unlink', side_effect=unlink), \
                mock.patch.object(driver, '_read_descriptor', side_effect=read), \
                mock.patch.object(driver.fcntl, 'flock', side_effect=flock), \
                ThreadPoolExecutor(max_workers=1) as publisher, \
                ThreadPoolExecutor(max_workers=1, thread_name_prefix='disk-reader') as reader:
            published = publisher.submit(driver.install_completed_record, self.root, key, run)
            try:
                self.assertTrue(linked.wait(10), 'completed link window was not reached')
                status = reader.submit(self.gate.disk_status, self.root, 1)
                self.assertTrue(attempted.wait(10), 'disk reader did not attempt claims lock')
                # On an unprotected publisher, pin the two-link inode before release.
                if not blocked[0]:
                    self.assertTrue(sampled.wait(10), 'disk reader did not read completed inode')
            finally:
                release.set()
            self.assertEqual(published.result(timeout=10), record)
            try:
                result = status.result(timeout=10)
            except ValueError as error:
                self.fail(f'disk status observed completed publication: {error}')
        audit.assert_called_once_with(spec, run, self.plan, formal_root=self.root)
        self.assertEqual(blocked, [True])
        self.assertTrue(result['safe'])
        self.assertEqual(result['active_reservation_bytes'], 0)
        self.assertEqual(list(path.parent.iterdir()), [path])
        self.assertEqual(path.stat().st_nlink, 1)
        self.assertEqual(driver._installed_completed_record(self.root, spec, self.plan), record)

    def test_settled_receipt_remains_but_no_longer_counts(self):
        key = self.claim()
        owner, run, _ = self.start(key)
        self.complete(key, run)
        with mock.patch.object(driver, '_process_start_time', return_value='expired-worker'):
            self.assertEqual(self.gate.disk_status(self.root, 1)['active_reservation_bytes'], 0)
        self.assertTrue(self.receipt(key).exists())
        self.assertFalse(self.gate._settled(self.root, self.plan, key, {key: {}}, None))
        self.assertFalse(self.gate._settled(self.root, self.plan, key, {}, {'spec_key': key}))

    def test_audit_handoff_accepts_the_exact_started_reservation(self):
        key = self.claim()
        _, run, _ = self.start(key)
        self.complete(key, run)
        try:
            handoff = driver._audit_handoff(self.root, self.plan, key, 0)
        except ValueError as error:
            self.fail(f'valid disk-bound audit handoff rejected: {error}')
        self.assertEqual(handoff['spec_key'], key)

    def test_invalid_retention_receipt_cannot_settle(self):
        key = self.claim()
        _, run, _ = self.start(key)
        self.complete(key, run)
        path = run / 'PRUNED_EVIDENCE.json'
        value = driver._load_json_file(path)
        value['freed_bytes'] = True
        path.chmod(0o644)
        path.write_bytes(driver._canonical_json(value) + b'\n')
        path.chmod(0o444)
        with self.assertRaises(ValueError):
            self.gate.disk_status(self.root, 1)

    def test_foreign_completed_owner_cannot_settle_reservation(self):
        key = self.claim()
        _, run, _ = self.start(key)
        self.complete(key, run, foreign_owner=True)
        with self.assertRaises(ValueError):
            self.gate.disk_status(self.root, 1)

    def retention_overlap(self):
        helpers = driver_fixtures.FormalDriverTests()
        auditor = helpers._owner(self.root, '', role='cpu-auditor')
        jobs = []
        for index in range(3):
            producer = helpers._owner(self.root, self.plan['missing_jobs'][index],
                                       role=f'formal-worker-{index % 2}')
            key, owner, run = helpers._pipeline_run(self.root, self.plan, index=index,
                                                    producer=producer, gpu=index % 2)
            driver.queue_audit(self.root, key, index % 2, owner)
            driver.release_gpu(self.root, index % 2, owner)
            jobs.append((key, run))
            if index == 0:
                self.assertEqual(key, driver.next_audit(self.root, 'formal', auditor)['spec_key'])
                record = helpers._pipeline_record(self.root, self.plan, key)
                evidence = driver._load_json_file(run / 'RESOURCE_EVIDENCE.json')
                mapping = {'checkpoints/formal_final.pt': 'checkpoint', 'config.json': 'config',
                           'data_flow_audit.jsonl': 'data_flow', 'results.json': 'results',
                           'validation/validation_manifest.json': 'validation_manifest'}
                record['artifact_sha256'] = {mapping.get(name, 'formal:' + name): digest
                    for name, digest in evidence['artifact_sha256'].items() if name != 'job.log'}
                record['record_sha256'] = driver._digest({k: v for k, v in record.items()
                                                          if k != 'record_sha256'})
                helpers._rewrite(self.root / 'records' / (driver._claim_name(key) + '.json'), record)
                resume = run / 'checkpoints/resume_latest.pt'
                resume.write_bytes((run / 'checkpoints/event_9_CIL.pt').read_bytes())
                resume.chmod(0o444)
        return jobs, auditor

    def test_full_retention_prunes_only_active_with_three_valid_queued(self):
        import prune_completed_runs as prune
        jobs, auditor = self.retention_overlap()
        key, run = jobs[0]
        before = driver_fixtures.FormalDriverTests._pipeline_snapshot(self.root)
        try:
            context = driver.retention_audit_context(self.root, key, auditor)
        except ValueError as error:
            self.fail(f'valid full retention overlap rejected: {error}')
        self.assertEqual(key, context['handoff']['spec_key'])
        self.assertEqual(before, driver_fixtures.FormalDriverTests._pipeline_snapshot(self.root))
        with mock.patch.object(prune, 'load_authority', return_value=(driver, registry)), \
                mock.patch.object(prune, 'active_processes', return_value=[]):
            dry = prune.prune_active_audit(self.root, Path(driver.__file__).parent,
                                           driver._source_commit(), key, auditor)
            self.assertEqual(21, dry['candidate_count'])
            self.assertEqual(before, driver_fixtures.FormalDriverTests._pipeline_snapshot(self.root))
            applied = prune.prune_active_audit(self.root, Path(driver.__file__).parent,
                                               driver._source_commit(), key, auditor, apply=True)
        self.assertTrue(applied['applied'])
        after = driver_fixtures.FormalDriverTests._pipeline_snapshot(self.root)
        prefix = 'runs/' + run.name + '/'
        self.assertEqual({p: v for p, v in before.items() if not p.startswith(prefix)},
                         {p: v for p, v in after.items() if not p.startswith(prefix)})
        self.assertTrue((run / 'checkpoints/formal_final.pt').is_file())
        self.assertEqual(3, len(list((self.root / 'audit_queue').glob('*%3A*.json'))))

    def test_full_retention_rejects_invalid_sibling_missing_target_and_active_owner(self):
        jobs, auditor = self.retention_overlap()
        key = jobs[0][0]
        first = self.root / 'audit_queue' / (driver._claim_name(key) + '.json')
        sibling = self.root / 'audit_queue' / (driver._claim_name(jobs[1][0]) + '.json')
        original = driver._load_json_file(sibling)
        for field, value in (('source_commit', 'f' * 40), ('root_identity', {}),
                             ('owner', {}), ('spec_key', key)):
            with self.subTest(field=field):
                driver_fixtures.FormalDriverTests._rewrite(sibling, {**original, field: value})
                with self.assertRaises(ValueError):
                    driver.retention_audit_context(self.root, key, auditor)
                driver_fixtures.FormalDriverTests._rewrite(sibling, original)
        duplicate = self.root / 'audit_queue/duplicate.json'
        driver.install_json_exclusive(duplicate, original)
        with self.assertRaises(ValueError):
            driver.retention_audit_context(self.root, key, auditor)
        duplicate.unlink()
        saved = first.read_bytes()
        first.unlink()
        with self.assertRaises(ValueError):
            driver.retention_audit_context(self.root, key, auditor)
        first.write_bytes(saved)
        first.chmod(0o444)
        with self.assertRaises(ValueError):
            driver.retention_audit_context(self.root, key, {**auditor, 'worker_role': 'foreign'})
        owner = driver.installed_claim_owner(self.root, key)
        self.assertTrue(driver.claim_gpu(self.root, 0, owner))
        with self.assertRaises(ValueError):
            driver.retention_audit_context(self.root, key, auditor)
        driver.release_gpu(self.root, 0, owner)
        record_path = self.root / 'records' / (driver._claim_name(key) + '.json')
        record = record_path.read_bytes()
        record_path.unlink()
        with self.assertRaises(ValueError):
            driver.retention_audit_context(self.root, key, auditor)
        record_path.write_bytes(record)
        record_path.chmod(0o444)
        import prune_completed_runs as prune
        with mock.patch.object(prune, 'load_authority', return_value=(driver, registry)), \
                mock.patch.object(prune, 'active_processes', return_value=[999999]):
            with self.assertRaisesRegex(ValueError, 'completed run still has a process'):
                prune.prune_active_audit(self.root, Path(driver.__file__).parent,
                                         driver._source_commit(), key, auditor, apply=True)
        self.assertFalse((jobs[0][1] / 'PRUNE_PLAN.json').exists())


if __name__ == '__main__':
    unittest.main()
