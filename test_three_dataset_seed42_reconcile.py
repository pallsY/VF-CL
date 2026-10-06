"""Disposable, production-shaped evidence; no live experiment roots are used."""
import csv
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import quote

import prune_completed_runs as prune


DATASETS = ('cifar100', 'isolet', 'upmc_food101')
METHODS = ('finetune', 'lwf', 'er', 'afc', 'adaptive')
FORMULA = 'final-minus-diagonal-v1'
PROFILE = 'seed42-adaptive-recovery'


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def install(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o600)
    path.write_bytes(encoded(value) + b'\n')
    path.chmod(0o444)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(path, *args):
    return subprocess.run(['git', '-C', str(path), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def snapshot(path):
    return {str(p.relative_to(path)): (stat.S_IMODE(p.lstat().st_mode),
            hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None)
            for p in path.rglob('*') if not p.is_symlink()}


def readonly_snapshot(path):
    root = Path(path)
    metadata = {}
    for entry in (root, *root.rglob('*')):
        details = entry.lstat()
        metadata[str(entry.relative_to(root))] = (
            details.st_dev, details.st_ino, details.st_mode, details.st_size,
            details.st_mtime_ns, details.st_ctime_ns)
    return snapshot(root), metadata


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('three_dataset_seed42_reconcile'),
                             'reconciliation module/CLI is missing')
        self.api = importlib.import_module('three_dataset_seed42_reconcile')
        self.temp = tempfile.TemporaryDirectory(prefix='reconcile-fixture-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / 'git worktree ; literal'
        self.repo.mkdir()
        git(self.repo, 'init', '-q')
        git(self.repo, 'config', 'user.name', 'Fixture')
        git(self.repo, 'config', 'user.email', 'fixture@example.invalid')
        (self.repo / 'main.py').write_text('# frozen scientific source\n')
        (self.repo / 'adaptive_consolidation_audit.py').write_bytes(b'# old audit\r\n# byte \xff\n')
        (self.repo / 'runner.py').write_bytes(b'# old runner\r\n')
        (self.repo / 'bic_calibration.py').write_bytes(b'# frozen bic\r\n')
        self.base_commit = self.commit('base')
        (self.repo / 'docs').mkdir()
        (self.repo / 'docs' / 'pilot.md').write_text('pilot\n')
        self.pilot_commit = self.commit('pilot')
        (self.repo / 'three_dataset_formal_audit.py').write_text('# audit only\n')
        self.recovery_commit = self.commit('recovery')
        self.old = self.base / 'pilot'
        self.new = self.base / 'recovery'
        self.out = self.base / 'out'
        self.sources = {'main.py': hashlib.sha256((self.repo / 'main.py').read_bytes()).hexdigest()}
        self.make_root(self.old, False)
        self.make_root(self.new, True)
        self.pin = self.pin_old()
        for name, value in [('PILOT_ROOT', str(self.old)),
                            ('PILOT_COMMIT', self.pilot_commit),
                            ('BASE_COMMIT', self.base_commit),
                            ('PILOT_PIN', self.pin)]:
            patcher = patch.object(self.api, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(self.api, 'PILOT_EVIDENCE_SHA256', self.pin_evidence())
        patcher.start()
        self.addCleanup(patcher.stop)

    def commit(self, message):
        git(self.repo, 'add', '.')
        git(self.repo, 'commit', '-qm', message)
        return git(self.repo, 'rev-parse', 'HEAD')

    def pin_old(self):
        return {name: hashlib.sha256((self.old / name).read_bytes()).hexdigest()
                for name in ('FORMAL_ROOT_IDENTITY.json', 'FORMAL_REGISTRY.json',
                             'FORMAL_PLAN.json', 'MISSING_JOBS.json')}

    def pin_evidence(self):
        paths = [*(self.old / 'records').iterdir(), *self.old.glob('runs/*/PRUNE_PLAN.json'),
                 *self.old.glob('runs/*/PRUNED_EVIDENCE.json')]
        self.assertEqual(len(paths), 34)
        return digest([{'path': str(p.relative_to(self.old)), 'type': 'regular',
                        'mode': stat.S_IMODE(p.lstat().st_mode), 'size': p.stat().st_size,
                        'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in sorted(paths)])

    def make_root(self, root, recovery):
        root.mkdir(mode=0o700)
        binding = {'experiment_profile': PROFILE} if recovery else {}
        commit = self.recovery_commit if recovery else self.pilot_commit
        keys = ([f'{d}:adaptive:42' for d in DATASETS] if recovery else
                [f'{d}:{m}:42' for m in METHODS for d in DATASETS])
        registry_hash = digest({'profile': PROFILE if recovery else 'pilot'})
        common = {'registry_sha256': registry_hash, 'metric_formula_version': FORMULA}
        admissions = []
        for key in keys:
            d, m, _ = key.split(':')
            raw = {'status': 'RERUN_REQUIRED', 'reason': 'no_declared_candidate',
                   'spec': {'dataset': d, 'method': m, 'seed': 42, 'explanation': False},
                   'protocol_sha256': getattr(self, 'adaptive_protocols', {}).get(key, digest({'dataset': d, 'method': m})),
                   'source_sha256': '', 'artifact_sha256': {}, 'metrics': None,
                   'metric_formula_version': FORMULA, 'trajectory_sha256': ''}
            admissions.append({**raw, 'spec_key': key, 'admission_record_sha256': digest(raw)})
        census = {'kind': 'formal_compatibility_census', **binding, **common, 'records': admissions}
        plan = {'kind': 'formal_plan', **binding, **common, 'formal_cells': keys,
                'missing_jobs': keys, 'explanation_cells': [], 'census_sha256': digest(census)}
        manifest = {'kind': 'formal_missing_jobs', **common, 'missing_jobs': keys,
                    'plan_sha256': digest(plan)}
        install(root / 'COMPATIBILITY_CENSUS.json', census)
        install(root / 'FORMAL_REGISTRY.json', {'kind': 'formal_registry', **binding,
                **common, 'formal_cells': keys, 'explanation_cells': []})
        install(root / 'FORMAL_PLAN.json', plan)
        install(root / 'MISSING_JOBS.json', manifest)
        details = root.stat()
        identity = {'kind': 'formal_root_identity', **binding,
                    'token': digest(str(root)), 'root_dev': details.st_dev,
                    'root_inode': details.st_ino, 'root_ctime_ns': details.st_ctime_ns,
                    'root_size': details.st_size, 'registry_sha256': registry_hash,
                    'plan_sha256': digest(plan), 'missing_jobs_sha256': digest(manifest),
                    'source_commit': commit}
        identity_hash = install(root / 'FORMAL_ROOT_IDENTITY.json', identity)
        identity_stat = (root / 'FORMAL_ROOT_IDENTITY.json').stat()
        owner_identity = {'dev': details.st_dev, 'inode': details.st_ino,
                          'ctime_ns': identity_stat.st_ctime_ns, 'size': identity_stat.st_size,
                          'hash': identity_hash}
        for key in keys:
            d, m, _ = key.split(':')
            run = root / 'runs' / quote(key, safe='')
            run.mkdir(parents=True)
            command = [sys.executable, '-c', 'fixture adaptive entrypoint' if m == 'adaptive' else m,
                       '--data', d, '--num_tasks', '2', '--seed', '42', '--device', 'cuda:0',
                       '--lr', '0.003', '--results_dir', str(root / 'runs'),
                       '--exp_name', run.name, '--cl_method', 'proto_evolve' if m == 'adaptive' else m]
            if key in getattr(self, 'adaptive_commands', {}):
                command = list(self.adaptive_commands[key])
                command[command.index('--results_dir') + 1] = str(root / 'runs')
            job = {'kind': 'formal_job_spec', 'spec_key': key,
                   'spec': {'dataset': d, 'method': m, 'seed': 42, 'explanation': False},
                   **common, 'plan_sha256': digest(plan), 'run_dir': str(run),
                   'source_commit': commit, 'source_sha256': self.sources,
                   'command': command, 'command_sha256': digest(command),
                   'root_identity': owner_identity}
            job_hash = install(run / 'FORMAL_JOB_SPEC.json', job)
            if not recovery and m == 'adaptive':
                (run / 'results.json').write_text('{"diagnostic_only": 999}\n')
                continue
            claim = {'kind': 'formal_job_claim', 'job': key, 'launcher_token': 'fixture',
                     'worker_role': 'formal-worker-1', 'pid': 123, 'pgid': 123,
                     'phase': 'formal', 'process_start_time': '1',
                     'source_commit': commit, 'root_identity': owner_identity}
            if recovery:
                claim.update(getattr(self, 'recovery_owner', {'pid': 2147483647}))
            claim_hash = install(run / 'CLAIM_OWNER.json', claim)
            if recovery:
                claims = root / 'claims'
                claims.mkdir(mode=0o700, exist_ok=True)
                bundle = claims / run.name
                bundle.mkdir(mode=0o700)
                install(bundle / 'owner.json', claim)
                install(bundle / 'started.json', {'kind': 'formal_job_started', 'job': key,
                    'owner_sha256': claim_hash, 'plan_sha256': digest(plan), 'command_sha256': digest(command),
                    'run_dir': str(run), **{k: claim[k] for k in ('source_commit', 'root_identity',
                    'worker_role', 'phase', 'pid', 'pgid', 'process_start_time')}})
            launch = {'kind': 'formal_launch_started', 'spec_key': key,
                      'plan_sha256': digest(plan), 'job_spec_sha256': job_hash,
                      'claim_sha256': claim_hash, 'command_sha256': digest(command),
                      **{k: claim[k] for k in ('source_commit', 'root_identity', 'worker_role',
                                              'phase', 'pid', 'pgid', 'process_start_time')}}
            launch_hash = install(run / 'LAUNCH_STARTED.json', launch)
            files = ['checkpoints/formal_final.pt', 'checkpoints/event_0_CIL.pt',
                     'checkpoints/event_1_CIL.pt', 'checkpoints/resume_latest.pt',
                     'formal_snapshots/event_0_CIL.pt', 'formal_snapshots/event_1_CIL.pt',
                     'config.json', 'results.json', 'job.log', 'data_flow_audit.jsonl',
                     'validation/validation_manifest.json', 'FORMAL_EVALUATION_COMPLETE.json']
            artifacts = {}
            for name in files:
                path = run / name
                path.parent.mkdir(parents=True, exist_ok=True)
                content = (key + (name if name != 'checkpoints/resume_latest.pt'
                                  else 'checkpoints/event_1_CIL.pt')).encode()
                path.write_bytes(content)
                artifacts[name] = hashlib.sha256(content).hexdigest()
            resource = {'hardware_identity': {'gpu_name': 'fixture', 'gpu_count': 1,
                        'cuda': 'fixture', 'torch': 'fixture', 'driver': 'fixture'},
                        'instrumentation': 'formal-resource-v1', 'runtime_seconds': 1.25,
                        'peak_gpu_memory_bytes': 1, 'checkpoint_size_bytes': (run / files[0]).stat().st_size,
                        'added_parameters': 0, 'communication_bytes': 0, 'replay_type': 'none',
                        'raw_examples_per_class': 0, 'persistent_embeddings': 0, 'privacy_label': 'fixture'}
            evidence = {'kind': 'formal_resource_evidence', 'spec_key': key,
                        'plan_sha256': digest(plan), 'job_spec_sha256': job_hash,
                        'claim_sha256': claim_hash, 'launch_sha256': launch_hash,
                        'command_sha256': digest(command), 'artifact_sha256': artifacts,
                        'resource': resource}
            install(run / 'RESOURCE_EVIDENCE.json', evidence)
            record_artifacts = {f'formal:{name}': value for name, value in artifacts.items()
                                if name.startswith(('formal_snapshots/', 'FORMAL_', 'checkpoints/event_'))}
            record_artifacts.update({logical: artifacts[path] for logical, path in
                {'checkpoint': files[0], 'config': 'config.json', 'results': 'results.json',
                 'data_flow': 'data_flow_audit.jsonl', 'validation_manifest': 'validation/validation_manifest.json'}.items()})
            record_artifacts.update({f'source:{k}': v for k, v in self.sources.items()})
            record_artifacts['trajectory'] = digest([0.3333333333333333, 0.12345678901234568])
            record = {'kind': 'formal_completed_run', **binding, 'spec_key': key,
                      'dataset': d, 'method': m, 'seed': 42, 'explanation': False, **common,
                      'plan_sha256': digest(plan), 'source_commit': commit,
                      'protocol_sha256': digest({'dataset': d, 'method': m}),
                      'trajectory_sha256': record_artifacts['trajectory'],
                      'admission_record_sha256': digest({'admitted': key}),
                      'artifact_sha256': record_artifacts,
                      'metrics': {'aa_final': 0.12345678901234568, 'bwt': -0.20000000000000004,
                                  'taskil_final': 0.9876543210987654,
                                  'aa_trajectory': [0.3333333333333333, 0.12345678901234568],
                                  'class_final': [0.12345678901234568] * 2,
                                  'taskil_final_by_task': [0.9876543210987654] * 2},
                      'command_sha256': digest(command), 'log_sha256': artifacts['job.log'],
                      'claim_sha256': claim_hash, 'launch_sha256': launch_hash, 'resource': resource}
            record['record_sha256'] = digest(record)
            install(root / 'records' / (run.name + '.json'), record)
            if (not recovery and key != 'cifar100:afc:42') or (recovery and (d != 'isolet' or getattr(self, 'prune_all_recovery', False))):
                removed = [{'path': n, 'sha256': artifacts[n], 'size': (run / n).stat().st_size}
                           for n in files if n.startswith(('checkpoints/event_',
                                                          'formal_snapshots/', 'checkpoints/resume_'))]
                prune = {'kind': 'formal_prune_plan', 'policy': 'completed-run-intermediate-v1',
                         'spec_key': key, 'source_commit': commit, 'record_sha256': record['record_sha256'],
                         'files': sorted(removed, key=lambda x: x['path']), 'bytes': sum(x['size'] for x in removed)}
                prune_hash = install(run / 'PRUNE_PLAN.json', prune)
                install(run / 'PRUNED_EVIDENCE.json', {'kind': 'formal_pruned_evidence',
                        'policy': prune['policy'], 'spec_key': key, 'source_commit': commit,
                        'record_sha256': record['record_sha256'], 'plan_sha256': prune_hash,
                        'freed_bytes': prune['bytes']})
                for item in removed:
                    (run / item['path']).unlink()
        if recovery:
            for name, kind in [('RECOVERY_PHASE_SUCCESS', 'recovery_phase_success'),
                               ('RECOVERY_EXECUTION_SUCCESS', 'recovery_execution_success')]:
                install(root / name, {'kind': kind, 'role': 'launcher', 'spec_key': '', 'exit_code': 0})
            (root / 'audit_queue').mkdir(mode=0o700)
            (root / 'gpu_claims').mkdir(mode=0o700)
        else:
            install(root / 'FAILED_JOB', {'kind': 'failed_audit', 'role': 'formal-worker-2',
                    'spec_key': 'isolet:adaptive:42', 'exit_code': 1})
            install(root / 'FORMAL_STOPPED', {'kind': 'formal_stopped', 'role': 'formal',
                    'spec_key': '', 'exit_code': 1})

    def upgrade_recovery_pruning_to_tombstones(self):
        upgraded = []
        for plan_path in sorted(self.new.glob('runs/*/PRUNE_PLAN.json')):
            run = plan_path.parent
            old_plan = json.loads(plan_path.read_bytes())
            quarantine = run / ('.prune-quarantine-' + '0' * 32)
            quarantine.mkdir(mode=0o700)
            files = []
            tombstones = []
            for old in old_plan['files']:
                group, name = old['path'].split('/')
                tombstone_name = f'{group}.{name}'
                tombstone = quarantine / tombstone_name
                tombstone.write_bytes(b'')
                tombstone.chmod(0o444)
                details = tombstone.stat()
                item = {
                    **old, 'device': details.st_dev, 'inode': details.st_ino,
                    'mode': stat.S_IMODE(details.st_mode),
                }
                files.append(item)
                tombstones.append({
                    'path': old['path'], 'tombstone': tombstone_name,
                    'inode': details.st_ino, 'original_size': old['size'],
                    'sha256': old['sha256'], 'final_size': 0,
                })
            plan = {**old_plan, 'files': files}
            plan_hash = install(plan_path, plan)
            completion = {
                'kind': 'formal_pruned_evidence',
                'policy': plan['policy'], 'spec_key': plan['spec_key'],
                'source_commit': plan['source_commit'],
                'record_sha256': plan['record_sha256'],
                'plan_sha256': plan_hash, 'quarantine': quarantine.name,
                'files': tombstones, 'freed_bytes': plan['bytes'],
            }
            install(run / 'PRUNED_EVIDENCE.json', completion)
            upgraded.append((run, plan_path, run / 'PRUNED_EVIDENCE.json',
                             quarantine))
        self.assertTrue(upgraded)
        return upgraded

    def call(self, **kwargs):
        return self.api.reconcile(kwargs.get('pilot_root', self.old), self.new,
                                  kwargs.get('output_root', self.out), self.repo, self.repo)

    def alter(self, path, change):
        value = json.loads(path.read_bytes())
        change(value)
        install(path, value)

    def reject(self, pattern):
        before = (snapshot(self.old), snapshot(self.new))
        with self.assertRaisesRegex((ValueError, OSError), pattern):
            self.call()
        self.assertEqual((snapshot(self.old), snapshot(self.new)), before)
        self.assertFalse((self.out / 'RECONCILIATION_SUCCESS').exists())

    def test_collection_preserves_report_table_and_success_bytes(self):
        self.assertTrue(hasattr(self.api, 'collect_seed42_rows'), 'pure collector is missing')
        report = self.call()
        table = (self.out / 'PILOT_TABLE.csv').read_bytes()
        success = (self.out / 'RECONCILIATION_SUCCESS').read_bytes()
        collected = self.api.collect_seed42_rows(self.old, self.new, self.repo, self.repo)
        self.assertEqual(encoded(report), encoded(collected))
        for field in ('rows', 'sources', 'git_scope'):
            self.assertEqual(report[field], collected[field])
        self.out = self.base / 'second-output'
        self.assertEqual(report, self.call())
        self.assertEqual(table, (self.out / 'PILOT_TABLE.csv').read_bytes())
        self.assertEqual(success, (self.out / 'RECONCILIATION_SUCCESS').read_bytes())

    def test_publication_matches_independent_golden_bytes(self):
        # Fixed input isolates the pre-extraction CSV/JSON/marker byte contract
        # from fixture inode, timestamp, commit and record-hash variability.
        rows = [{'dataset': d, 'method': m, 'seed': 42,
                 'metrics': {'aa_final': 0.125, 'bwt': -0.25, 'taskil_final': 0.5},
                 'source_root': '/golden/source', 'source_commit': 'a' * 40,
                 'record_sha256': 'b' * 64} for d in DATASETS for m in METHODS]
        frozen = {'rows': rows, 'validation': 'ADMITTED'}
        expected_report = encoded(frozen) + b'\n'
        expected_table = (
            'dataset,method,seed,aa_final,bwt,taskil_final,source_root,source_commit,record_sha256\n'
            + ''.join(f'{d},{m},42,0.125,-0.25,0.5,/golden/source,{"a" * 40},{"b" * 64}\n'
                      for d in DATASETS for m in METHODS)).encode()
        expected_success = (
            '{"artifact_sha256":{"PILOT_RECONCILIATION.json":"'
            + hashlib.sha256(expected_report).hexdigest()
            + '","PILOT_TABLE.csv":"' + hashlib.sha256(expected_table).hexdigest()
            + '"},"kind":"pilot_reconciliation_success","row_count":15}\n').encode()
        with patch.object(self.api, '_collect_seed42_rows', return_value=frozen):
            self.call()
        for name, expected in (('PILOT_RECONCILIATION.json', expected_report),
                               ('PILOT_TABLE.csv', expected_table),
                               ('RECONCILIATION_SUCCESS', expected_success)):
            self.assertEqual(expected, (self.out / name).read_bytes(), name)

    def test_exact_fifteen_rows_are_unrounded_and_sources_read_only(self):
        before = (snapshot(self.old), snapshot(self.new))
        self.call()
        self.assertEqual((snapshot(self.old), snapshot(self.new)), before)
        self.assertEqual(stat.S_IMODE(self.out.stat().st_mode), 0o700)
        self.assertEqual({p.name for p in self.out.iterdir()},
                         {'PILOT_TABLE.csv', 'PILOT_RECONCILIATION.json', 'RECONCILIATION_SUCCESS'})
        for path in self.out.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)
        rows = list(csv.DictReader(io.StringIO((self.out / 'PILOT_TABLE.csv').read_text())))
        self.assertEqual([(r['dataset'], r['method'], r['seed']) for r in rows],
                         [(d, m, '42') for d in DATASETS for m in METHODS])
        self.assertEqual({r['aa_final'] for r in rows}, {'0.12345678901234568'})
        self.assertEqual({r['bwt'] for r in rows}, {'-0.20000000000000004'})
        report = json.loads((self.out / 'PILOT_RECONCILIATION.json').read_bytes())
        self.assertEqual({s['source_commit'] for s in report['sources']},
                         {self.pilot_commit, self.recovery_commit})
        self.assertEqual(len(report['rows']), 15)
        self.assertIsNone(report['git_scope']['approved_source_migration'])
        for row in report['rows']:
            self.assertEqual(row['decision'], 'ADMITTED')
            self.assertEqual(hashlib.sha256(Path(row['record_path']).read_bytes()).hexdigest(),
                             row['record_file_sha256'])
        marker = json.loads((self.out / 'RECONCILIATION_SUCCESS').read_bytes())
        for name, value in marker['artifact_sha256'].items():
            self.assertEqual(hashlib.sha256((self.out / name).read_bytes()).hexdigest(), value)

    def test_new_tombstone_pruning_and_frozen_old_pilot_reconcile_to_fifteen_rows(self):
        upgraded = self.upgrade_recovery_pruning_to_tombstones()
        before = (snapshot(self.old), snapshot(self.new))

        self.call()

        self.assertEqual((snapshot(self.old), snapshot(self.new)), before)
        rows = list(csv.DictReader(io.StringIO(
            (self.out / 'PILOT_TABLE.csv').read_text())))
        self.assertEqual(15, len(rows))
        self.assertTrue(all(row['decision'] == 'ADMITTED' for row in
                            json.loads((self.out / 'PILOT_RECONCILIATION.json').read_bytes())['rows']))
        self.assertEqual(34, len([
            *self.old.joinpath('records').iterdir(),
            *self.old.glob('runs/*/PRUNE_PLAN.json'),
            *self.old.glob('runs/*/PRUNED_EVIDENCE.json'),
        ]))
        self.assertTrue(all(path.stat().st_size == 0
                            for run, _, _, quarantine in upgraded
                            for path in quarantine.iterdir()))

    def test_reject_new_tombstone_pruning_near_miss_matrix(self):
        cases = ('missing', 'extra', 'reordered', 'malformed', 'tampered',
                 'quarantine', 'path', 'hash')
        for index, case in enumerate(cases):
            with self.subTest(case=case):
                upgraded = self.upgrade_recovery_pruning_to_tombstones()
                run, plan_path, evidence_path, quarantine = upgraded[0]
                plan = json.loads(plan_path.read_bytes())
                evidence = json.loads(evidence_path.read_bytes())
                original_plan = json.loads(plan_path.read_bytes())
                original_evidence = json.loads(evidence_path.read_bytes())
                moved = None
                extra = None
                if case == 'missing':
                    target = quarantine / evidence['files'][0]['tombstone']
                    moved = quarantine / '.held-tombstone'
                    target.rename(moved)
                elif case == 'extra':
                    extra = quarantine / 'extra'
                    extra.write_bytes(b'')
                    extra.chmod(0o444)
                elif case == 'reordered':
                    evidence['files'] = list(reversed(evidence['files']))
                    install(evidence_path, evidence)
                elif case == 'malformed':
                    evidence['extra'] = True
                    install(evidence_path, evidence)
                elif case == 'tampered':
                    evidence['files'][0]['final_size'] = 1
                    install(evidence_path, evidence)
                elif case == 'quarantine':
                    quarantine.chmod(0o755)
                elif case == 'path':
                    plan['files'][0]['path'] = '../escape'
                    evidence['plan_sha256'] = install(plan_path, plan)
                    install(evidence_path, evidence)
                else:
                    evidence['files'][0]['sha256'] = '0' * 64
                    install(evidence_path, evidence)
                self.out = self.base / f'out-new-prune-{index}'
                self.reject('prun|quarantine|tombstone|path|hash|schema|input')
                if moved is not None:
                    moved.rename(quarantine / original_evidence['files'][0]['tombstone'])
                if extra is not None:
                    extra.unlink()
                quarantine.chmod(0o700)
                install(plan_path, original_plan)
                install(evidence_path, original_evidence)
                for other_run, _, _, other_quarantine in upgraded[1:]:
                    for tombstone in other_quarantine.iterdir():
                        tombstone.unlink()
                    other_quarantine.rmdir()
                for tombstone in quarantine.iterdir():
                    tombstone.unlink()
                quarantine.rmdir()

    def test_reject_symlink_root_and_ancestor(self):
        link = self.base / 'link'
        link.symlink_to(self.old, target_is_directory=True)
        with self.assertRaisesRegex((ValueError, OSError), 'symlink|symbolic|root'):
            self.call(pilot_root=link)
        link.unlink()
        link.symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex((ValueError, OSError), 'symlink|symbolic|root'):
            self.call(pilot_root=link / 'pilot')

    def test_reject_existing_output_even_empty(self):
        self.out.mkdir(mode=0o700)
        self.reject('output|exist')

    def test_reject_nonempty_output(self):
        self.out.mkdir()
        (self.out / 'keep').write_text('keep')
        self.reject('output|exist')
        self.assertEqual((self.out / 'keep').read_text(), 'keep')

    def test_reject_output_symlink(self):
        self.out.symlink_to(self.old, target_is_directory=True)
        self.reject('output|exist|symlink')

    def test_reject_wrong_pilot_identity(self):
        self.alter(self.old / 'FORMAL_ROOT_IDENTITY.json', lambda v: v.update(token='0' * 64))
        self.reject('identity|pin')

    def test_reject_wrong_recovery_commit(self):
        self.alter(self.new / 'FORMAL_ROOT_IDENTITY.json', lambda v: v.update(source_commit=self.pilot_commit))
        self.reject('commit|identity')

    def test_reject_missing_cell(self):
        next((self.new / 'records').iterdir()).unlink()
        self.reject('membership|record|cell')

    def test_reject_duplicate_record(self):
        path = next((self.old / 'records').iterdir())
        install(path.parent / 'duplicate.json', json.loads(path.read_bytes()))
        self.reject('membership|record|cell')

    def test_reject_old_provisional_adaptive_in_source_set(self):
        path = next((self.new / 'records').iterdir())
        install(self.old / 'records' / path.name, json.loads(path.read_bytes()))
        self.reject('membership|record|adaptive')

    def test_reject_changed_record_hash(self):
        path = next((self.old / 'records').iterdir())
        self.alter(path, lambda v: v['metrics'].update(aa_final=0.99))
        self.reject('record.*hash|record.*digest')

    def test_reject_changed_pruning_evidence(self):
        path = next((self.old / 'runs').glob('*/PRUNED_EVIDENCE.json'))
        self.alter(path, lambda v: v.update(freed_bytes=v['freed_bytes'] + 1))
        self.reject('prun')

    def test_reject_changed_prune_plan_and_rehashed_evidence(self):
        path = next((self.old / 'runs').glob('*/PRUNE_PLAN.json'))
        self.alter(path, lambda v: v['files'][0].update(sha256='1' * 64))
        self.alter(path.with_name('PRUNED_EVIDENCE.json'), lambda v: v.update(
            plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        self.reject('prun|artifact')

    def test_reject_prune_path_escape(self):
        path = next((self.old / 'runs').glob('*/PRUNE_PLAN.json'))
        self.alter(path, lambda v: v['files'][0].update(path='../outside'))
        self.reject('prun|path')

    def test_reject_unpruned_artifact_hash(self):
        path = self.old / 'runs' / quote('cifar100:afc:42', safe='') / 'checkpoints/event_0_CIL.pt'
        path.write_bytes(b'altered')
        self.reject('artifact|hash')

    def test_reject_symlink_record_and_nonregular_evidence(self):
        path = next((self.old / 'records').iterdir())
        saved = self.base / 'saved.json'
        path.rename(saved)
        path.symlink_to(saved)
        self.reject('symlink|regular|record')

    def test_reject_fifo_evidence_without_blocking(self):
        path = next((self.old / 'runs').glob('*/PRUNED_EVIDENCE.json'))
        path.unlink()
        os.mkfifo(path)
        with self.assertRaisesRegex((ValueError, OSError), 'regular|type'):
            self.call()

    def test_reject_writable_control_json(self):
        (self.new / 'FORMAL_PLAN.json').chmod(0o644)
        self.reject('immutable|mode')

    def test_reject_noncanonical_and_duplicate_json_keys(self):
        path = self.new / 'FORMAL_PLAN.json'
        path.chmod(0o600)
        path.write_bytes(path.read_bytes().replace(b'{', b'{"kind":"formal_plan",', 1))
        path.chmod(0o444)
        self.reject('canonical|duplicate')

    def test_reject_changed_adaptive_option_even_rehashed(self):
        path = self.new / 'runs' / quote('cifar100:adaptive:42', safe='') / 'FORMAL_JOB_SPEC.json'
        def change(v):
            v['command'][v['command'].index('--lr') + 1] = '0.004'
            v['command_sha256'] = digest(v['command'])
        self.alter(path, change)
        self.reject('command|contract')

    def test_reject_device_change(self):
        path = self.new / 'runs' / quote('cifar100:adaptive:42', safe='') / 'FORMAL_JOB_SPEC.json'
        def change(v):
            v['command'][v['command'].index('--device') + 1] = 'cuda:1'
            v['command_sha256'] = digest(v['command'])
        self.alter(path, change)
        self.reject('command|contract')

    def test_reject_unexpected_scientific_source_change(self):
        (self.repo / 'main.py').write_text('# changed training\n')
        self.commit('forbidden scientific change')
        self.reject('scope|source|commit')

    def test_allow_only_pinned_prune_operational_source_change(self):
        (self.repo / 'prune_completed_runs.py').write_text('# pinned retention source\n')
        allowed_commit = self.commit('add pinned retention source')
        head, changed, migration = self.api.provenance(self.repo, self.repo)
        self.assertEqual(head, allowed_commit)
        self.assertIn('prune_completed_runs.py', changed)
        self.assertIsNone(migration)

        (self.repo / 'unreviewed_helper.py').write_text('# arbitrary operational source\n')
        self.commit('add arbitrary source')
        with self.assertRaisesRegex(ValueError,
                                    'Git scope forbids source: unreviewed_helper.py'):
            self.api.provenance(self.repo, self.repo)

    def source_transition(self, path, commit, parent):
        def entry(revision):
            fields = git(self.repo, 'ls-tree', revision, '--', path).split()
            self.assertEqual(fields[1], 'blob')
            return fields[0], fields[2]
        def content(revision):
            return subprocess.run(['git', '-C', str(self.repo), 'show', revision + ':' + path],
                                  check=True, capture_output=True).stdout
        old_mode, old_blob = entry(parent)
        new_mode, new_blob = entry(commit)
        return {'commit': commit, 'parent': parent,
                'old_mode': old_mode, 'new_mode': new_mode,
                'old_blob': old_blob, 'new_blob': new_blob,
                'old_sha256': hashlib.sha256(content(parent)).hexdigest(),
                'new_sha256': hashlib.sha256(content(commit)).hexdigest()}

    def reviewed_migrations(self):
        audit = self.repo / 'adaptive_consolidation_audit.py'
        runner = self.repo / 'runner.py'
        parent_one = git(self.repo, 'rev-parse', 'HEAD')
        audit.write_bytes(b'# reviewed audit one\r\n# byte \xfe\n')
        commit_one = self.commit('reviewed audit migration one')
        parent_two = commit_one
        audit.write_bytes(b'# reviewed audit two\r\n')
        runner.write_bytes(b'# reviewed runner\r\n')
        commit_two = self.commit('reviewed audit and runner migration')
        parent_three = commit_two
        audit.write_bytes(b'# reviewed audit final\r\n')
        commit_three = self.commit('reviewed audit migration final')
        self.recovery_commit = commit_three
        return {
            'adaptive_consolidation_audit.py': [
                self.source_transition('adaptive_consolidation_audit.py', commit_one, parent_one),
                self.source_transition('adaptive_consolidation_audit.py', commit_two, parent_two),
                self.source_transition('adaptive_consolidation_audit.py', commit_three, parent_three),
            ],
            'runner.py': [self.source_transition('runner.py', commit_two, parent_two)],
        }

    def migration_roots(self, migrations):
        self.old, self.new = self.base / 'migration-pilot', self.base / 'migration-recovery'
        self.sources = {**self.sources, **{
            path: transitions[0]['old_sha256'] for path, transitions in migrations.items()}}
        self.make_root(self.old, False)
        old = self.sources
        self.sources = {**old, **{
            path: transitions[-1]['new_sha256'] for path, transitions in migrations.items()}}
        self.make_root(self.new, True)
        for name, value in (('PILOT_ROOT', str(self.old)), ('PILOT_PIN', self.pin_old()),
                            ('PILOT_EVIDENCE_SHA256', self.pin_evidence())):
            patcher = patch.object(self.api, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return old, self.sources

    def real_prune_mixed_mode_recovery_cell(self, migrations):
        old, new = self.migration_roots(migrations)
        run = self.new / 'runs' / quote('isolet:adaptive:42', safe='')
        record = json.loads((self.new / 'records' / (run.name + '.json')).read_bytes())
        evidence = json.loads((run / 'RESOURCE_EVIDENCE.json').read_bytes())
        wanted_modes = {
            name: 0o444 if name.startswith('formal_snapshots/') else 0o600
            for name in prune.expected_paths(2)
        }
        for name, mode in wanted_modes.items():
            (run / name).chmod(mode)
        plan = prune.build_prune_plan(
            run, evidence, record['record_sha256'], record['source_commit'], 2)
        self.assertEqual(wanted_modes,
                         {item['path']: item['mode'] for item in plan['files']})
        prune.install_plan(run, plan)
        applied = prune.apply_prune_plan(run, plan)
        completion = prune.install_completion(run, plan, applied)
        return old, new, run, plan, completion

    def test_accept_complete_ordered_source_migration_chains(self):
        migrations = self.reviewed_migrations()
        with patch.object(self.api, 'APPROVED_SOURCE_MIGRATIONS', migrations, create=True):
            try:
                result = self.api.provenance(self.repo, self.repo)
            except ValueError as error:
                self.fail('exact reviewed migration chains were rejected: ' + str(error))
        self.assertEqual(result, (self.recovery_commit,
            ['adaptive_consolidation_audit.py', 'docs/pilot.md', 'runner.py',
             'three_dataset_formal_audit.py'], migrations))
        self.assertIsNot(result[2], migrations)
        self.assertIsNot(result[2]['adaptive_consolidation_audit.py'],
                         migrations['adaptive_consolidation_audit.py'])

    def test_accept_exact_source_hash_migration_chains(self):
        migrations = self.reviewed_migrations()
        old, new = self.migration_roots(migrations)
        before = (snapshot(self.old), snapshot(self.new))
        with self.subTest(boundary='source maps'):
            self.assertTrue(callable(getattr(self.api, '_source_compatibility', None)),
                            'exact old/new source compatibility gate is missing')
            old_jobs = {f'{d}:{m}:42': ({'source_sha256': old},) for d in DATASETS for m in METHODS}
            new_jobs = {f'{d}:adaptive:42': ({'source_sha256': new},) for d in DATASETS}
            actual_old, actual_new = self.api._source_compatibility(old_jobs, new_jobs, migrations)
            self.assertIs(actual_old, old)
            self.assertIs(actual_new, new)
        with self.subTest(boundary='reconciliation'):
            with patch.object(self.api, 'APPROVED_SOURCE_MIGRATIONS', migrations, create=True):
                try:
                    report = self.call()
                except ValueError as error:
                    self.fail('exact source hash migration was rejected: ' + str(error))
            self.assertEqual(report['git_scope']['approved_source_migration'], migrations)
            self.assertEqual(len(report['rows']), 15)
            self.assertTrue((self.out / 'RECONCILIATION_SUCCESS').is_file())
        self.assertEqual((snapshot(self.old), snapshot(self.new)), before)

    def test_real_pruner_mixed_tombstone_modes_reconcile_and_reject_corruption(self):
        migrations = self.reviewed_migrations()
        _, _, run, plan, completion = self.real_prune_mixed_mode_recovery_cell(
            migrations)
        quarantine = run / completion['quarantine']
        by_path = {item['path']: item for item in completion['files']}
        before = (snapshot(self.old), snapshot(self.new))
        with patch.object(self.api, 'APPROVED_SOURCE_MIGRATIONS', migrations,
                          create=True):
            report = self.call()
        self.assertEqual((snapshot(self.old), snapshot(self.new)), before)
        self.assertEqual(15, len(report['rows']))
        self.assertTrue(all(row['decision'] == 'ADMITTED'
                            for row in report['rows']))
        self.assertEqual(migrations,
                         report['git_scope']['approved_source_migration'])
        self.assertTrue((self.out / 'RECONCILIATION_SUCCESS').is_file())

        plan_path = run / prune.PLAN_NAME
        evidence_path = run / prune.COMPLETE_NAME
        original_plan = json.loads(plan_path.read_bytes())
        original_evidence = json.loads(evidence_path.read_bytes())
        checkpoint = next(item for item in plan['files']
                          if item['path'] == 'checkpoints/event_0_CIL.pt')
        snapshot_entry = next(item for item in plan['files']
                              if item['path'] == 'formal_snapshots/event_0_CIL.pt')

        for index, case in enumerate(('checkpoint_mode', 'snapshot_mode',
                                      'device', 'inode', 'nlink', 'size')):
            with self.subTest(corruption=case):
                self.out = self.base / f'out-real-prune-corrupt-{index}'
                cleanup = None
                if case in ('checkpoint_mode', 'snapshot_mode'):
                    entry = checkpoint if case == 'checkpoint_mode' else snapshot_entry
                    tombstone = quarantine / by_path[entry['path']]['tombstone']
                    tombstone.chmod(0o444 if entry['mode'] == 0o600 else 0o600)
                    cleanup = lambda p=tombstone, mode=entry['mode']: p.chmod(mode)
                elif case in ('device', 'inode'):
                    changed_plan = json.loads(plan_path.read_bytes())
                    changed_evidence = json.loads(evidence_path.read_bytes())
                    field = case
                    changed_plan['files'][0][field] += 1
                    if field == 'inode':
                        changed_evidence['files'][0]['inode'] += 1
                    changed_evidence['plan_sha256'] = install(plan_path, changed_plan)
                    install(evidence_path, changed_evidence)
                    cleanup = lambda: (install(plan_path, original_plan),
                                       install(evidence_path, original_evidence))
                elif case == 'nlink':
                    tombstone = quarantine / by_path[checkpoint['path']]['tombstone']
                    hardlink = self.base / 'tombstone-hardlink'
                    os.link(tombstone, hardlink)
                    cleanup = hardlink.unlink
                else:
                    tombstone = quarantine / by_path[checkpoint['path']]['tombstone']
                    tombstone.write_bytes(b'not-empty')
                    cleanup = lambda p=tombstone: p.write_bytes(b'')
                try:
                    with patch.object(self.api, 'APPROVED_SOURCE_MIGRATIONS',
                                      migrations, create=True):
                        self.reject('prun|tombstone|mode|identity|input')
                finally:
                    cleanup()

    def test_reject_source_migration_chain_history_near_misses(self):
        original_repo = self.repo
        cases = {
            'missing': 'history', 'reordered': 'history', 'extra': 'history',
            'parent': 'parent', 'old_blob': 'blob', 'new_blob': 'blob',
            'old_mode': 'mode', 'new_mode': 'mode',
            'old_sha256': 'old hash', 'new_sha256': 'new hash',
            'later_edit': 'history', 'later_revert': 'history',
            'other_inventory': 'Git scope forbids source: bic_calibration.py',
            'merge_parent': 'history', 'merge_reverted_touch': 'history',
        }
        for case, reason in cases.items():
            with self.subTest(case=case):
                self.repo = self.base / ('migration-' + case)
                subprocess.run(['git', 'clone', '--quiet', '--shared', str(original_repo), str(self.repo)],
                               check=True, capture_output=True)
                git(self.repo, 'config', 'user.name', 'Fixture')
                git(self.repo, 'config', 'user.email', 'fixture@example.invalid')
                migrations = self.reviewed_migrations()
                migrations = {path: [dict(item) for item in transitions]
                              for path, transitions in migrations.items()}
                source = self.repo / 'adaptive_consolidation_audit.py'
                approved = source.read_bytes()
                chain = migrations['adaptive_consolidation_audit.py']
                if case == 'missing':
                    chain.pop(1)
                elif case == 'reordered':
                    chain[0], chain[1] = chain[1], chain[0]
                elif case == 'extra':
                    chain.append(dict(chain[-1]))
                elif case == 'parent':
                    chain[1]['parent'] = self.pilot_commit
                elif case in ('old_blob', 'new_blob', 'old_sha256', 'new_sha256'):
                    chain[1][case] = '0' * len(chain[1][case])
                elif case in ('old_mode', 'new_mode'):
                    chain[1][case] = '100755'
                elif case in ('later_edit', 'later_revert'):
                    source.write_bytes(b'# unreviewed edit\n')
                    self.commit('unreviewed edit')
                    if case == 'later_revert':
                        source.write_bytes(approved)
                        self.commit('restore reviewed bytes')
                elif case == 'other_inventory':
                    (self.repo / 'bic_calibration.py').write_text('# unreviewed BiC change\n')
                    self.commit('other inventory source')
                elif case == 'merge_parent':
                    git(self.repo, 'checkout', '-qb', 'side-history')
                    source.write_bytes(b'# merge source\n')
                    self.commit('side edit')
                    git(self.repo, 'checkout', '--detach', self.recovery_commit)
                    (self.repo / 'docs' / 'mainline.md').write_text('mainline\n')
                    first_parent = self.commit('mainline docs')
                    git(self.repo, 'merge', '--no-ff', '-s', 'ours', '-m', 'merge side history', 'side-history')
                    merge = git(self.repo, 'rev-parse', 'HEAD')
                    source.write_bytes(b'# merge source\n')
                    git(self.repo, 'add', str(source))
                    git(self.repo, 'commit', '--amend', '--no-edit')
                    merge = git(self.repo, 'rev-parse', 'HEAD')
                    chain.append(self.source_transition('adaptive_consolidation_audit.py', merge,
                                                        first_parent))
                elif case == 'merge_reverted_touch':
                    git(self.repo, 'checkout', '-qb', 'side-history')
                    source.write_bytes(b'# hidden side-branch edit\n')
                    self.commit('side edit')
                    source.write_bytes(approved)
                    self.commit('side revert')
                    git(self.repo, 'checkout', '--detach', self.recovery_commit)
                    (self.repo / 'docs' / 'mainline.md').write_text('mainline\n')
                    self.commit('mainline docs')
                    git(self.repo, 'merge', '--no-ff', '-s', 'ours', '-m', 'merge side history', 'side-history')
                    self.assertEqual(source.read_bytes(), approved)
                with patch.object(self.api, 'APPROVED_SOURCE_MIGRATIONS', migrations, create=True):
                    with self.assertRaisesRegex(ValueError, reason):
                        self.api.provenance(self.repo, self.repo)
                self.assertFalse((self.out / 'RECONCILIATION_SUCCESS').exists())

    def test_reject_source_map_migration_near_misses(self):
        migrations = self.reviewed_migrations()
        old, new = self.migration_roots(migrations)
        cases = {'old_hash': 'approved scientific source migration differs',
                 'new_hash': 'approved scientific source migration differs',
                 'missing_key': 'scientific source inventory differs',
                 'extra_key': 'scientific source inventory differs',
                 'unapproved_direct_hash': 'scientific source provenance differs outside approved migration',
                 'old_disagreement': 'old scientific source provenance differs',
                 'new_disagreement': 'new scientific source provenance differs',
                 'no_migration': 'scientific source provenance differs'}
        for case, reason in cases.items():
            with self.subTest(case=case):
                self.assertTrue(callable(getattr(self.api, '_source_compatibility', None)),
                                'exact old/new source compatibility gate is missing')
                old_map, new_map = dict(old), dict(new)
                if case == 'old_hash':
                    old_map['adaptive_consolidation_audit.py'] = '0' * 64
                elif case == 'new_hash':
                    new_map['runner.py'] = old_map['runner.py']
                elif case == 'missing_key':
                    new_map.pop('main.py')
                elif case == 'extra_key':
                    new_map['extra.py'] = '0' * 64
                elif case == 'unapproved_direct_hash':
                    new_map['main.py'] = '0' * 64
                old_jobs = {f'{d}:{m}:42': ({'source_sha256': dict(old_map)},)
                            for d in DATASETS for m in METHODS}
                new_jobs = {f'{d}:adaptive:42': ({'source_sha256': dict(new_map)},) for d in DATASETS}
                if case.endswith('disagreement'):
                    jobs = old_jobs if case == 'old_disagreement' else new_jobs
                    jobs['cifar100:adaptive:42'][0]['source_sha256']['main.py'] = '0' * 64
                before = encoded([old_jobs, new_jobs])
                with self.assertRaisesRegex(ValueError, reason):
                    self.api._source_compatibility(old_jobs, new_jobs,
                                                   None if case == 'no_migration' else migrations)
                self.assertEqual(encoded([old_jobs, new_jobs]), before)
                self.assertFalse((self.out / 'RECONCILIATION_SUCCESS').exists())
        with self.subTest(case='record_must_match_own_job'):
            path = self.new / 'records' / (quote('isolet:adaptive:42', safe='') + '.json')
            def change(record):
                record['artifact_sha256']['source:adaptive_consolidation_audit.py'] = (
                    migrations['adaptive_consolidation_audit.py'][0]['old_sha256'])
                record['record_sha256'] = digest({k: v for k, v in record.items() if k != 'record_sha256'})
            self.alter(path, change)
            with patch.object(self.api, 'APPROVED_SOURCE_MIGRATIONS', migrations, create=True):
                self.reject('record source provenance differs')

    def test_reject_change_reverted_in_later_commit(self):
        path = self.repo / 'main.py'
        original = path.read_bytes()
        path.write_text('# changed then reverted\n')
        self.commit('forbidden')
        path.write_bytes(original)
        self.commit('revert')
        self.reject('scope|source|commit')

    def test_reject_failure_marker_in_recovery(self):
        install(self.new / 'FAILED_JOB', {'kind': 'failed_audit', 'role': 'formal-worker-1',
                'spec_key': 'isolet:adaptive:42', 'exit_code': 1})
        self.reject('marker|failure')

    def test_reject_success_marker_mismatch(self):
        self.alter(self.new / 'RECOVERY_EXECUTION_SUCCESS', lambda v: v.update(kind='pilot_execution_success'))
        self.reject('marker|success')

    def test_reject_old_failure_for_baseline(self):
        self.alter(self.old / 'FAILED_JOB', lambda v: v.update(spec_key='isolet:er:42'))
        self.reject('marker|failed')

    def test_reject_nonempty_recovery_audit_queue(self):
        install(self.new / 'audit_queue' / 'pending.json', {'pending': True})
        self.reject('queue|drain')

    def test_cli_requires_exact_arguments_and_rejects_wrong_root(self):
        script = Path(self.api.__file__)
        result = subprocess.run([sys.executable, '-B', str(script), '--pilot-root', str(self.old),
                '--recovery-root', str(self.new), '--output-root', str(self.out),
                '--pilot-worktree', str(self.repo), '--recovery-worktree', str(self.repo)],
                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('pilot', result.stderr.lower())
        self.assertFalse(self.out.exists())

    def test_success_marker_is_installed_last(self):
        observed = []
        original = os.rename
        def observe(src, dst, *args, **kwargs):
            observed.append(Path(dst).name)
            return original(src, dst, *args, **kwargs)
        with patch.object(self.api.os, 'rename', side_effect=observe):
            self.call()
        self.assertEqual(observed[-1], 'RECONCILIATION_SUCCESS')

    def test_reject_boolean_success_exit_code(self):
        self.alter(self.new / 'RECOVERY_EXECUTION_SUCCESS', lambda v: v.update(exit_code=False))
        self.reject('marker|success')

    def test_reject_rehashed_resource_boolean_count(self):
        name = quote('isolet:adaptive:42', safe='')
        record = self.new / 'records' / (name + '.json')
        def change(v):
            v['resource']['hardware_identity']['gpu_count'] = True
            v['record_sha256'] = digest({k: x for k, x in v.items() if k != 'record_sha256'})
        self.alter(record, change)
        self.alter(self.new / 'runs' / name / 'RESOURCE_EVIDENCE.json',
                   lambda v: v['resource']['hardware_identity'].update(gpu_count=True))
        self.reject('resource|hardware|type')

    def test_directory_membership_change_during_validation_rejects(self):
        original = self.api.completed
        changed = False
        def mutate(*args):
            nonlocal changed
            result = original(*args)
            if not changed:
                install(self.old / 'records' / 'injected.json', {'injected': True})
                changed = True
            return result
        with patch.object(self.api, 'completed', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'directory|membership|changed'):
                self.call()
        self.assertFalse((self.out / 'RECONCILIATION_SUCCESS').exists())

    def test_input_mutation_during_publication_has_no_success_marker(self):
        original = os.rename
        def mutate(src, dst, *args, **kwargs):
            result = original(src, dst, *args, **kwargs)
            if Path(dst).name == 'PILOT_RECONCILIATION.json':
                self.alter(self.new / 'RECOVERY_EXECUTION_SUCCESS', lambda v: v.update(exit_code=1))
            return result
        with patch.object(self.api.os, 'rename', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'input|changed'):
                self.call()
        self.assertFalse((self.out / 'RECONCILIATION_SUCCESS').exists())

    def test_large_artifact_hashing_has_bounded_memory(self):
        import tracemalloc
        name = quote('isolet:adaptive:42', safe='')
        run = self.new / 'runs' / name
        content = b'fixture-checkpoint' * (1024 * 1024)
        (run / 'checkpoints/formal_final.pt').write_bytes(content)
        value_hash, size = hashlib.sha256(content).hexdigest(), len(content)
        del content
        def change_record(v):
            v['artifact_sha256']['checkpoint'] = value_hash
            v['resource']['checkpoint_size_bytes'] = size
            v['record_sha256'] = digest({k: x for k, x in v.items() if k != 'record_sha256'})
        self.alter(self.new / 'records' / (name + '.json'), change_record)
        def change_evidence(v):
            v['artifact_sha256']['checkpoints/formal_final.pt'] = value_hash
            v['resource']['checkpoint_size_bytes'] = size
        self.alter(run / 'RESOURCE_EVIDENCE.json', change_evidence)
        tracemalloc.start()
        try:
            self.call()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 5 * 1024 * 1024, 'artifact hashing must stream, not load checkpoints')

    def test_reject_symlink_parent_of_pruned_artifacts(self):
        run = self.old / 'runs' / quote('cifar100:finetune:42', safe='')
        (run / 'formal_snapshots').rmdir()
        outside = self.base / 'outside'
        outside.mkdir()
        (run / 'formal_snapshots').symlink_to(outside, target_is_directory=True)
        with self.assertRaises((ValueError, OSError)):
            self.call()
        self.assertFalse(self.out.exists())

    def test_reject_replaced_pinned_root_and_recovery_symlink(self):
        saved = self.base / 'old-saved'
        self.old.rename(saved)
        self.old.symlink_to(saved, target_is_directory=True)
        with self.assertRaises(OSError):
            self.call()
        self.old.unlink()
        saved.rename(self.old)
        link = self.base / 'new-link'
        link.symlink_to(self.new, target_is_directory=True)
        self.new = link
        with self.assertRaises(OSError):
            self.call()

    def test_reject_hardlinked_record(self):
        path = next((self.new / 'records').iterdir())
        os.link(path, self.base / 'hardlink')
        self.reject('single-link|regular')

    def test_reject_pruned_evidence_float_byte_count(self):
        path = next((self.old / 'runs').glob('*/PRUNED_EVIDENCE.json'))
        self.alter(path, lambda v: v.update(freed_bytes=float(v['freed_bytes'])))
        self.reject('prun')

    def test_reject_rehashed_launch_float_pid(self):
        name = quote('isolet:adaptive:42', safe='')
        run = self.new / 'runs' / name
        launch_path = run / 'LAUNCH_STARTED.json'
        self.alter(launch_path, lambda v: v.update(pid=float(v['pid'])))
        launch_hash = hashlib.sha256(launch_path.read_bytes()).hexdigest()
        self.alter(run / 'RESOURCE_EVIDENCE.json', lambda v: v.update(launch_sha256=launch_hash))
        def change(v):
            v['launch_sha256'] = launch_hash
            v['record_sha256'] = digest({k: x for k, x in v.items() if k != 'record_sha256'})
        self.alter(self.new / 'records' / (name + '.json'), change)
        self.reject('launch|provenance')

    def test_git_inspection_does_not_refresh_input_index(self):
        source = self.repo / 'main.py'
        details = source.stat()
        os.utime(source, ns=(details.st_atime_ns, details.st_mtime_ns + 2000000000))
        index = self.repo / '.git' / 'index'
        before = index.read_bytes()
        self.call()
        self.assertEqual(index.read_bytes(), before, 'read-only Git inspection rewrote its input index')

    def test_scalar_record_digests_require_exact_hex_strings(self):
        path = self.new / 'records' / (quote('isolet:adaptive:42', safe='') + '.json')
        original = json.loads(path.read_bytes())
        fields = [k for k in original if k.endswith('_sha256') and k != 'artifact_sha256']
        for field in fields:
            for index, invalid in enumerate(({}, [], None, False, 42, 'A' * 64, 'a' * 63)):
                with self.subTest(field=field, invalid=invalid):
                    record = {**original, field: invalid}
                    if field != 'record_sha256':
                        record['record_sha256'] = digest({k: v for k, v in record.items() if k != 'record_sha256'})
                    install(path, record)
                    self.out = self.base / ('out-' + field + '-' + str(index))
                    self.reject('record|hash|authority')

    def test_reject_unknown_pruned_artifact_with_null_digest(self):
        path = next((self.old / 'runs').glob('*/PRUNE_PLAN.json'))
        def change(v):
            v['files'].append({'path': 'checkpoints/event_999_CIL.pt', 'sha256': None, 'size': 1})
            v['bytes'] += 1
        self.alter(path, change)
        self.alter(path.with_name('PRUNED_EVIDENCE.json'), lambda v: v.update(
            plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), freed_bytes=v['freed_bytes'] + 1))
        self.reject('prun|hash|artifact')

    def test_reject_output_parent_replaced_during_publication(self):
        before = (snapshot(self.old), snapshot(self.new))
        original = os.rename
        for index, trigger in enumerate(('PILOT_RECONCILIATION.json', 'PILOT_TABLE.csv')):
            for replacement in ('symlink', 'directory'):
                with self.subTest(trigger=trigger, replacement=replacement):
                    parent = self.base / ('publish-' + str(index) + '-' + replacement)
                    moved = parent.with_name(parent.name + '-moved')
                    parent.mkdir()
                    self.out = parent / 'out'
                    def mutate(src, dst, *args, **kwargs):
                        result = original(src, dst, *args, **kwargs)
                        if Path(dst).name == trigger:
                            original(parent, moved)
                            if replacement == 'symlink':
                                parent.symlink_to(moved, target_is_directory=True)
                            else:
                                parent.mkdir()
                                self.out.mkdir()
                        return result
                    with patch.object(self.api.os, 'rename', side_effect=mutate):
                        with self.assertRaises((ValueError, OSError)):
                            self.call()
                    partial = moved / 'out'
                    self.assertFalse((partial / 'RECONCILIATION_SUCCESS').exists())
                    expected = {'PILOT_RECONCILIATION.json'}
                    if trigger == 'PILOT_TABLE.csv':
                        expected.add('PILOT_TABLE.csv')
                    self.assertEqual({p.name for p in partial.iterdir()}, expected)
                    for path in partial.iterdir():
                        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)
                    if replacement == 'directory':
                        self.assertEqual(list(self.out.iterdir()), [])
        self.assertEqual((snapshot(self.old), snapshot(self.new)), before)

    def test_reject_output_parent_replaced_after_temporary_fsync(self):
        parent, moved = self.base / 'publication', self.base / 'publication-moved'
        parent.mkdir()
        self.out = parent / 'out'
        original = os.fsync
        changed = False
        def mutate(fd):
            nonlocal changed
            result = original(fd)
            if not changed and (self.out / '.PILOT_TABLE.csv.tmp').exists():
                parent.rename(moved)
                parent.symlink_to(moved, target_is_directory=True)
                changed = True
            return result
        with patch.object(self.api.os, 'fsync', side_effect=mutate):
            with self.assertRaises((ValueError, OSError)):
                self.call()
        self.assertTrue(changed)
        self.assertEqual({p.name for p in (moved / 'out').iterdir()},
                         {'PILOT_RECONCILIATION.json', '.PILOT_TABLE.csv.tmp'})
        self.assertFalse((moved / 'out' / 'RECONCILIATION_SUCCESS').exists())

    def test_reject_output_parent_replaced_during_final_input_check(self):
        parent, moved = self.base / 'publication', self.base / 'publication-moved'
        parent.mkdir()
        self.out = parent / 'out'
        original = self.api.Evidence.verify
        changed = False
        def mutate(root):
            nonlocal changed
            result = original(root)
            if not changed and (self.out / '.RECONCILIATION_SUCCESS.tmp').exists():
                parent.rename(moved)
                parent.symlink_to(moved, target_is_directory=True)
                changed = True
            return result
        with patch.object(self.api.Evidence, 'verify', autospec=True, side_effect=mutate):
            with self.assertRaises((ValueError, OSError)):
                self.call()
        self.assertTrue(changed)
        self.assertEqual({p.name for p in (moved / 'out').iterdir()},
                         {'PILOT_RECONCILIATION.json', 'PILOT_TABLE.csv', '.RECONCILIATION_SUCCESS.tmp'})
        self.assertFalse((moved / 'out' / 'RECONCILIATION_SUCCESS').exists())

    def test_reject_conflicting_explicit_pruned_resume_digest(self):
        plan = next((self.old / 'runs').glob('*/PRUNE_PLAN.json'))
        self.alter(plan.with_name('RESOURCE_EVIDENCE.json'), lambda v: v['artifact_sha256'].update(
            {'checkpoints/resume_latest.pt': '0' * 64}))
        self.reject('pruning.*hash|resume')

    def test_accept_matching_explicit_pruned_resume_digest(self):
        plan = next((self.old / 'runs').glob('*/PRUNE_PLAN.json'))
        resume = next(e for e in json.loads(plan.read_bytes())['files']
                      if e['path'] == 'checkpoints/resume_latest.pt')
        self.alter(plan.with_name('RESOURCE_EVIDENCE.json'), lambda v: v['artifact_sha256'].update(
            {resume['path']: resume['sha256']}))
        self.call()
        self.assertTrue((self.out / 'RECONCILIATION_SUCCESS').is_file())

    def test_reject_old_unpruned_metric_tampering_with_recomputed_record_hash(self):
        path = self.old / 'records' / (quote('cifar100:afc:42', safe='') + '.json')
        def change(v):
            v['metrics']['aa_final'] = 0.99
            v['record_sha256'] = digest({k: x for k, x in v.items() if k != 'record_sha256'})
        self.alter(path, change)
        self.reject('pilot.*evidence|record.*hash')

    def test_reject_old_prune_tampering_with_recomputed_plan_hash(self):
        path = next((self.old / 'runs').glob('*/PRUNE_PLAN.json'))
        def change(v):
            v['files'][0]['size'] += 1
            v['bytes'] += 1
        self.alter(path, change)
        self.alter(path.with_name('PRUNED_EVIDENCE.json'), lambda v: v.update(
            plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), freed_bytes=v['freed_bytes'] + 1))
        self.reject('pilot.*evidence|pruning')

    def test_accept_deleted_explicit_resume_digest_with_valid_prune_evidence(self):
        plan = next((self.old / 'runs').glob('*/PRUNE_PLAN.json'))
        resume = next(e for e in json.loads(plan.read_bytes())['files']
                      if e['path'] == 'checkpoints/resume_latest.pt')
        resource = plan.with_name('RESOURCE_EVIDENCE.json')
        self.alter(resource, lambda v: v['artifact_sha256'].update({resume['path']: resume['sha256']}))
        self.alter(resource, lambda v: v['artifact_sha256'].pop(resume['path']))
        self.call()
        self.assertTrue((self.out / 'RECONCILIATION_SUCCESS').is_file())

    def test_real_driver_completed_lifecycle_rejects_unapproved_source_inventory(self):
        worktree = Path(__file__).resolve().parent
        lifecycle_base = self.base / 'real-driver'
        lifecycle_base.mkdir()
        script = '''
import json, sys
from pathlib import Path
import three_dataset_formal_driver as driver
from test_three_dataset_formal_driver import FormalDriverTests
suite = FormalDriverTests()
suite.base = Path(sys.argv[1])
root, plan = suite._installed(name='recovery')
auditor = suite._owner(root, '', role='auditor')
for index in range(3):
    key, producer, run = suite._pipeline_run(root, plan, index=index)
    driver.queue_audit(root, key, 0, producer)
    driver.release_gpu(root, 0, producer)
    assert driver.next_audit(root, 'formal', auditor)['spec_key'] == key
    record = suite._pipeline_record(root, plan, key)
    resource = driver._load_json_file(run / 'RESOURCE_EVIDENCE.json')['artifact_sha256']
    special = {'checkpoint':'checkpoints/formal_final.pt', 'config':'config.json',
               'results':'results.json', 'data_flow':'data_flow_audit.jsonl',
               'validation_manifest':'validation/validation_manifest.json'}
    record['artifact_sha256'] = {k:resource[v] for k,v in special.items()}
    record['artifact_sha256'].update({'formal:'+k:v for k,v in resource.items()
                                     if k not in set(special.values()) | {'job.log'}})
    record['artifact_sha256'].update({'source:'+k:v for k,v in driver._source_hashes().items()})
    record['artifact_sha256']['trajectory'] = record['trajectory_sha256']
    record['record_sha256'] = driver._digest({k:v for k,v in record.items() if k != 'record_sha256'})
    suite._rewrite(root / 'records' / (driver.safe_spec_key(key)+'.json'), record)
    driver.complete_audit(root, key, auditor)
assert driver.claim_next(plan, root / 'claims', 'formal', suite._owner(root, ''), pipeline=True) is None
for name in ('RECOVERY_PHASE_SUCCESS', 'RECOVERY_EXECUTION_SUCCESS'):
    driver.install_marker(root, name, {'kind':name.lower(), 'role':'launcher', 'spec_key':'', 'exit_code':0})
assert len(list((root/'claims').iterdir())) == len(list((root/'records').iterdir())) == 3
assert not list((root/'audit_queue').iterdir()) and not list((root/'gpu_claims').iterdir())
jobs = {key: driver._load_json_file(root/'runs'/driver.safe_spec_key(key)/'FORMAL_JOB_SPEC.json')
        for key in plan['missing_jobs']}
census = driver._load_json_file(root/'COMPATIBILITY_CENSUS.json')
print(json.dumps({'root':str(root), 'commit':driver._source_commit(), 'sources':driver._source_hashes(),
                  'commands':{k:v['command'] for k,v in jobs.items()},
                  'protocols':{r['spec_key']:r['protocol_sha256'] for r in census['records']}}))
'''
        result = subprocess.run([sys.executable, '-B', '-c', script, str(lifecycle_base)],
            cwd=worktree, env={**os.environ, 'VFCL_EXPERIMENT_PROFILE': PROFILE, 'VFCL_PYTHON': sys.executable},
            capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        produced = json.loads(result.stdout.splitlines()[-1])
        self.repo = self.base / 'real-git'
        subprocess.run(['git', 'clone', '--quiet', '--shared', str(worktree), str(self.repo)],
                       check=True, capture_output=True, text=True)
        self.old, self.new = self.base / 'compatible-pilot', Path(produced['root'])
        self.pilot_commit = 'f79062f20c54bb43849ecfc43682d643be23afe6'
        self.recovery_commit = produced['commit']
        from adaptive_consolidation_audit import FORMAL_SOURCE_FILES
        self.assertEqual(FORMAL_SOURCE_FILES, (
            'adaptive_consolidation_audit.py', 'adaptive_dual_branch_validation.py',
            'adaptive_head_consolidation.py', 'adaptive_tinyimagenet_heldout.py',
            'bic_calibration.py', 'calibration_split.py', 'cl_methods/__init__.py',
            'cl_methods/adagauss.py', 'cl_methods/afc.py', 'cl_methods/der_pp.py',
            'cl_methods/er.py', 'cl_methods/er_ace.py', 'cl_methods/ewc.py',
            'cl_methods/fedprotip_vfl.py', 'cl_methods/finetune.py',
            'cl_methods/gpm.py', 'cl_methods/lwf.py', 'cl_methods/lwf_wa.py',
            'cl_methods/proto_evolve.py', 'cl_methods/proto_fedspace.py',
            'cl_methods/target.py', 'config.py', 'data_utils.py', 'determinism.py',
            'head_consolidation.py', 'main.py', 'metrics.py', 'models.py', 'runner.py',
            'three_dataset_formal_runtime.py', 'ul_methods/__init__.py',
            'ul_methods/retrain.py', 'vfl_trainer.py',
        ))
        self.assertEqual(set(produced['sources']), set(FORMAL_SOURCE_FILES))
        self.sources = {name: hashlib.sha256(subprocess.run(
            ['git', '-C', str(self.repo), 'show', self.pilot_commit + ':' + name],
            check=True, capture_output=True).stdout).hexdigest() for name in FORMAL_SOURCE_FILES}
        self.assertEqual([name for name in FORMAL_SOURCE_FILES
                          if self.sources[name] != produced['sources'][name]],
                         ['adaptive_consolidation_audit.py', 'cl_methods/er_ace.py',
                          'cl_methods/proto_fedspace.py', 'cl_methods/target.py',
                          'config.py', 'runner.py', 'vfl_trainer.py'])
        self.adaptive_commands, self.adaptive_protocols = produced['commands'], produced['protocols']
        self.make_root(self.old, False)
        before = {root: readonly_snapshot(root) for root in (self.old, self.new)}
        with patch.multiple(self.api, PILOT_ROOT=str(self.old), PILOT_COMMIT=self.pilot_commit,
                BASE_COMMIT='c3ad9214759a5819e682bc50c7db667d41934e1d',
                PILOT_PIN=self.pin_old(), PILOT_EVIDENCE_SHA256=self.pin_evidence()):
            with self.assertRaisesRegex(ValueError, 'approved source migration history differs'):
                self.call()
        self.assertFalse(self.out.exists())
        self.assertEqual(before, {root: readonly_snapshot(root) for root in (self.old, self.new)})

    def test_reject_empty_recovery_claims(self):
        for bundle in (self.new / 'claims').iterdir():
            for path in bundle.iterdir():
                path.unlink()
            bundle.rmdir()
        self.reject('claim')

    def test_reject_missing_or_extra_durable_claim(self):
        claims = self.new / 'claims'
        bundle = next(claims.iterdir())
        moved = self.base / 'saved-claim'
        bundle.rename(moved)
        self.reject('claim.*membership')
        moved.rename(bundle)
        (claims / 'extra').mkdir(mode=0o700)
        self.reject('claim.*membership')

    def test_reject_unstarted_or_active_durable_claim(self):
        bundle = next((self.new / 'claims').iterdir())
        value = json.loads((bundle / 'started.json').read_bytes())
        (bundle / 'started.json').unlink()
        self.reject('claim|started')
        install(bundle / 'started.json', value)
        install(bundle / 'active.json', {'active': True})
        self.reject('claim|active')

    def test_reject_durable_claim_symlink_and_wrong_modes(self):
        claims = self.new / 'claims'
        bundle = next(claims.iterdir())
        moved = self.base / 'saved-claim'
        bundle.rename(moved)
        bundle.symlink_to(moved, target_is_directory=True)
        with self.assertRaises((ValueError, OSError)):
            self.call()
        bundle.unlink()
        moved.rename(bundle)
        for path, bad_mode in ((claims, 0o755), (bundle, 0o755), (bundle / 'owner.json', 0o644),
                               (bundle / 'started.json', 0o644)):
            with self.subTest(path=path.name):
                original = stat.S_IMODE(path.stat().st_mode)
                path.chmod(bad_mode)
                self.reject('mode|immutable|claim')
                path.chmod(original)

    def test_reject_mutated_durable_owner_or_started_fields(self):
        bundle = next((self.new / 'claims').iterdir())
        for name in ('owner.json', 'started.json'):
            path = bundle / name
            original = json.loads(path.read_bytes())
            for field in original:
                with self.subTest(name=name, field=field):
                    install(path, {**original, field: 'altered'})
                    self.reject('claim|started|owner')
            install(path, original)
        for name in ('owner.json', 'started.json'):
            path = bundle / name
            original = json.loads(path.read_bytes())
            install(path, {**original, 'extra': True})
            self.reject('claim|owner|started')
            install(path, original)

    def test_reject_matching_live_completed_claim_process(self):
        self.new = self.base / 'live-recovery'
        start = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]
        self.recovery_owner = {'pid': os.getpid(), 'pgid': os.getpgrp(), 'process_start_time': start}
        self.make_root(self.new, True)
        self.reject('live.*claim|claim.*live|process')

    def test_accept_reused_claim_pid_with_different_start_time(self):
        self.new = self.base / 'reused-pid-recovery'
        self.recovery_owner = {'pid': os.getpid(), 'pgid': os.getpgrp(), 'process_start_time': '1'}
        self.make_root(self.new, True)
        try:
            self.call()
        except ValueError as error:
            self.fail('completed claim with a reused PID was rejected: ' + str(error))
        self.assertTrue((self.out / 'RECONCILIATION_SUCCESS').is_file())


if __name__ == '__main__':
    unittest.main()
