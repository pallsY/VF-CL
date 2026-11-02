import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import frozen_ablation_audit as audit
import frozen_ablation_gpu1_continue as continuation


class ExtensionAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.restart_raw = b'synthetic restart anchor\n'
        patch = mock.patch.object(audit, 'RESTART_ANCHOR_SHA256',
                                  hashlib.sha256(self.restart_raw).hexdigest(), create=True)
        patch.start()
        self.addCleanup(patch.stop)

    def fixture(self):
        pending = ['isolet/fim_off/43', 'isolet/no_partykd/44']
        rows = [{'job_key': k, 'physical_gpu': 0, 'config_sha256': 'c' * 64,
                 'code_commit': 'd' * 40} for k in sorted(audit.expected_keys())]
        success = sorted(set(audit.expected_keys()) - set(pending))
        prefix_events = [dict(kind=kind, job_key=k, attempt=1, classification='success',
                              returncode=0) for k in success for kind in ('start', 'finish')]
        prefix = b''.join((json.dumps(e) + '\n').encode() for e in prefix_events)
        old = dict(pid=1, ppid=0, pgrp=1, session=1, start_ticks=10,
                   boot_id='boot', cmdline=['python', 'old.py'], state='S')
        parent = dict(old, pid=2, start_ticks=20, cmdline=['python', 'new.py'])
        extension = self.root / 'gpu1-rebalance-v1'
        extension.mkdir()
        armed = dict(schema_version=1, manifest_sha256='m' * 64,
                     candidate_cap=sorted(continuation.CANDIDATE_CAP), initial_pending=pending,
                     old_identity=old, controller_identity=parent,
                     original_orchestration_commit=continuation.ORIGIN_COMMIT,
                     orchestration_commit='e' * 40,
                     ledger_sha256=hashlib.sha256(prefix).hexdigest(), ledger_bytes_length=len(prefix),
                     gpu_uuid={str(i): continuation.GPU_UUID[i] for i in (0, 1)})
        continuation.write_once(extension / 'ARMED.json', armed)
        (self.root / 'manifests').mkdir()
        (self.root / 'manifests/manifest.jsonl').write_text('synthetic manifest')
        (self.root / 'manifests/manifest.sha256').write_text('synthetic external anchor')
        (self.root / 'manifests/RESTART_ANCHOR.json').write_bytes(self.restart_raw)
        manifest_sha = continuation.sha(self.root / 'manifests/manifest.jsonl')
        armed['manifest_sha256'] = manifest_sha
        (extension / 'ARMED.json').write_text(json.dumps(armed))
        continuation.write_once(self.root / 'QUEUE_LOCK.json', dict(pid=1,
            manifest_sha256=manifest_sha, orchestration_commit=continuation.ORIGIN_COMMIT))
        metadata = {name: continuation.sha(self.root / name) if (self.root / name).is_file() else None
                    for name in ['QUEUE_LOCK.json', 'QUEUE_COMPLETE.json', 'QUEUE_FAILURE.json',
                                 'manifests/manifest.jsonl', 'manifests/manifest.sha256',
                                 'manifests/RESTART_ANCHOR.json',
                                 'gpu1-rebalance-v1/ARMED.json']}
        metadata.update({f'{k.split("/")[1]}/{k.split("/")[0]}/{k.split("/")[2]}/attempt-01.plan.json': None
                         for k in audit.expected_keys()})
        epoch = dict(schema_version=1, manifest_sha256=manifest_sha, success_keys=success,
                     pending_keys=pending, assignments={pending[0]: 0, pending[1]: 1},
                     old_identity=old, retired_identity=dict(old, state='gone'),
                     stopped_utc='2026-11-02T00:00:00+00:00', retired_utc='2026-11-02T00:00:01+00:00',
                     controller_identity=parent, orchestration_commit='e' * 40,
                     original_orchestration_commit=continuation.ORIGIN_COMMIT,
                     armed_sha256=continuation.sha(extension / 'ARMED.json'),
                     ledger_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
                     ledger_prefix_bytes_length=len(prefix), metadata_files_sha256=metadata,
                     metadata_sha256=hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
                     stopped_snapshots=[dict(observed_utc='time', identity=dict(old, state='T'),
                         threads=[[1, 10, 'T', []]], ledger_sha256=hashlib.sha256(prefix).hexdigest(),
                         ledger_bytes_length=len(prefix))] * 2)
        roots = {'no_partykd': '/pinned/base', 'fim_off': '/pinned/patch'}
        armed.update(legacy_root='/pinned/origin', code_roots=roots)
        (extension / 'ARMED.json').write_text(json.dumps(armed))
        metadata['gpu1-rebalance-v1/ARMED.json'] = continuation.sha(extension / 'ARMED.json')
        epoch.update(legacy_root='/pinned/origin', code_roots=roots,
                     armed_sha256=metadata['gpu1-rebalance-v1/ARMED.json'],
                     metadata_sha256=hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest())
        path = extension / 'EXECUTION_EPOCH.json'
        continuation.write_once(path, epoch)
        (self.root / 'ATTEMPTS.jsonl').write_bytes(prefix)
        return rows, path, epoch, prefix

    def test_extension_requires_external_sha_prefix_cap_and_deterministic_mapping(self):
        self.assertTrue(hasattr(audit, 'verify_execution_extension'), 'explicit extension verifier missing')
        rows, path, epoch, prefix = self.fixture()
        actual = audit.verify_execution_extension(self.root, rows, epoch['manifest_sha256'], path, continuation.sha(path))
        self.assertEqual(actual['assignments'], epoch['assignments'])
        with self.assertRaises(ValueError):
            audit.verify_execution_extension(self.root, rows, epoch['manifest_sha256'], path, '0' * 64)
        for field, value in [('ledger_prefix_sha256', '0' * 64),
                             ('assignments', {'cifar100/no_partykd/44': 1}),
                             ('assignments', {key: 0 for key in epoch['assignments']}),
                             ('old_identity', {**epoch['old_identity'], 'start_ticks': 11})]:
            with self.subTest(field=field):
                path.write_text(json.dumps({**epoch, field: value}))
                with self.assertRaises(ValueError):
                    audit.verify_execution_extension(self.root, rows, epoch['manifest_sha256'], path, continuation.sha(path))

    def test_default_audit_never_implicitly_selects_extension(self):
        manifest = self.root / 'manifests/manifest.jsonl'
        manifest.parent.mkdir()
        manifest.write_text('\n'.join(json.dumps({'job_key': k}) for k in audit.expected_keys()))
        with mock.patch.object(audit, 'RESULT_ROOT', self.root), \
             mock.patch.object(audit, 'verify_execution_extension', create=True) as extension:
            with self.assertRaisesRegex(ValueError, 'preflight'):
                audit.audit_metadata(manifest, continuation.sha(manifest))
            extension.assert_not_called()

    def test_restart_anchor_missing_tampered_or_forged_record_is_rejected(self):
        rows, path, epoch, _ = self.fixture()
        anchor = self.root / 'manifests/RESTART_ANCHOR.json'
        for change in ('missing', 'tampered', 'forged_record'):
            changed = json.loads(json.dumps(epoch))
            with self.subTest(change=change):
                try:
                    if change == 'missing':
                        anchor.unlink()
                    else:
                        anchor.write_bytes(b'changed restart anchor\n')
                    if change == 'forged_record':
                        changed['metadata_files_sha256']['manifests/RESTART_ANCHOR.json'] = continuation.sha(anchor)
                        changed['metadata_sha256'] = hashlib.sha256(json.dumps(
                            changed['metadata_files_sha256'], sort_keys=True).encode()).hexdigest()
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        audit.verify_execution_extension(self.root, rows, epoch['manifest_sha256'],
                                                         path, continuation.sha(path))
                finally:
                    anchor.write_bytes(self.restart_raw)

    def test_extension_arguments_are_an_explicit_pair(self):
        with self.assertRaises(ValueError):
            audit.audit_metadata(self.root / 'missing', 'x', execution_epoch=self.root / 'epoch')

    def test_publication_validation_delegates_to_pinned_origin_only_in_extension_mode(self):
        self.assertTrue(hasattr(audit, 'audit_publication_preflight'), 'origin publication delegation missing')
        import frozen_ablation_gpu1_continue as continuation
        legacy = mock.Mock()
        legacy.validate_publication_preflight.return_value = 'origin-publication-sha'
        path = self.root / 'publication'
        epoch = {'legacy_root': '/pinned/origin'}
        with mock.patch.object(continuation, 'load_legacy', return_value=legacy) as load, \
             mock.patch.object(audit, 'validate_publication_preflight', return_value='current-sha') as current:
            self.assertEqual(audit.audit_publication_preflight(path, None), 'current-sha')
            load.assert_not_called()
            self.assertEqual(audit.audit_publication_preflight(path, epoch), 'origin-publication-sha')
            load.assert_called_once_with('/pinned/origin')
            legacy.validate_publication_preflight.assert_called_once_with(path)
            self.assertEqual(current.call_count, 1)
            legacy.validate_publication_preflight.side_effect = ValueError('original source rejection')
            with self.assertRaisesRegex(ValueError, 'original source rejection'):
                audit.audit_publication_preflight(path, epoch)

    def test_execution_job_rejects_runtime_config_copy_and_handshake_tampering(self):
        key = 'isolet/fim_off/43'
        config_path = self.root / 'manifests/configs/job.json'
        config_path.parent.mkdir(parents=True)
        job_root = self.root / 'fim_off/isolet/43'
        config = {'device': 'cuda:0', 'output_dir': str(job_root / 'attempt-01')}
        config_path.write_text(json.dumps(config, indent=2))
        row = dict(job_key=key, config_sha256=continuation.sha(config_path),
                   code_commit='d' * 40, config_relpath='configs/job.json')
        parent = dict(pid=100, start_ticks=20)
        worker = dict(pid=200, ppid=100, pgrp=200, session=200,
                      start_ticks=30, boot_id='boot', cmdline=['python', 'worker.py'])
        epoch = dict(assignments={key: 1}, controller_identity=parent, manifest_sha256='m' * 64)
        binding = continuation.execution_binding(row, 1, 'e' * 64, parent)
        metadata = dict(job_key=key, attempt=1, config_sha256=row['config_sha256'],
                        training_code_commit=row['code_commit'], manifest_sha256='m' * 64,
                        worker_identity=worker, observed_cuda_uuid_order=binding['cuda_visible_devices'], **binding)
        run = job_root / 'attempt-01'
        run.mkdir(parents=True)
        continuation.write_once(job_root / 'attempt-01.plan.json', {'job_key': key, 'attempt': 1})
        for name in ('config.json', 'APPROVED_CONFIG.json'):
            (run / name).write_bytes(config_path.read_bytes())
        protocol = self.root / 'gpu1-rebalance-v1/workers' / key.replace('/', '__')
        protocol.mkdir(parents=True)
        ready = dict(job_key=key, attempt=1, config_sha256=row['config_sha256'],
                     training_code_commit=row['code_commit'], worker_identity=worker, **binding)
        for name in ('READY.json', 'GO.json'):
            continuation.write_once(protocol / name, ready)
        def verify(runtime=metadata):
            audit.verify_execution_job(self.root, row, config, binding, binding,
                                       metadata, runtime, epoch, 'e' * 64)
        verify()
        for field, value in [('job_key', 'wrong'), ('attempt', 2), ('manifest_sha256', 'x'),
                             ('config_sha256', 'x'), ('training_code_commit', 'x'),
                             ('gpu_index', 0), ('observed_cuda_uuid_order', list(reversed(binding['cuda_visible_devices'])))]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify({**metadata, field: value})
        (protocol / 'GO.json').write_text(json.dumps({**ready, 'worker_identity': {**worker, 'pid': 201}}))
        with self.assertRaises(ValueError):
            verify()

        (protocol / 'GO.json').write_text(json.dumps(ready))
        (run / 'config.json').write_bytes(config_path.read_bytes() + b' ')
        with self.assertRaises(ValueError):
            verify()

    def test_execution_audit_requires_the_actual_clean_orchestration_source(self):
        self.assertTrue(hasattr(audit, 'verify_execution_source'), 'source verifier missing')
        names = ('frozen_ablation_gpu1_continue.py', 'frozen_ablation_quiescent.py', 'frozen_ablation_audit.py')
        armed = dict(orchestration_files_sha256={name: continuation.sha(Path(audit.__file__).parent / name) for name in names},
                     training_commits={'no_partykd': audit.BASE_COMMIT, 'fim_off': audit.PATCH_COMMIT})
        with mock.patch.object(audit.subprocess, 'check_output', side_effect=['e' * 40 + '\n', '']):
            audit.verify_execution_source({'orchestration_commit': 'e' * 40}, armed)
        with mock.patch.object(audit.subprocess, 'check_output', side_effect=['e' * 40 + '\n', ' M runner.py']):
            with self.assertRaises(ValueError):
                audit.verify_execution_source({'orchestration_commit': 'e' * 40}, armed)


if __name__ == '__main__':
    unittest.main()
