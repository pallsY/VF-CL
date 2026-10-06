"""Fail-closed reuse tests using generated, disposable evidence only."""
import importlib
import copy
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import shutil
import stat
import unittest
from unittest.mock import patch
from urllib.parse import quote

import test_three_dataset_seed42_reconcile as fixtures


class OperationalMigrationPinTests(unittest.TestCase):
    generated_cifar_transitions = [
        {
            'commit': '8cfcdaaed05a090d84bc20412a3ef8b8732b01aa',
            'parent': '36181fcfaac66dca309228627fdf8f86212effce',
            'old_mode': '100644', 'new_mode': '100644',
            'old_blob': '73488fbd18f4c559227e2f5f3c4172b3f50e743b',
            'new_blob': '3f30a727aaee8325f81599b2dc673ab8c1c3a920',
            'old_sha256': '444e4ff90ba3d3f3fd40a5b88dd726bf8ed25c0cb9e27f3f34a5a516a2bf4dcc',
            'new_sha256': '14b8e113b0ab09e2103f9210a2556f43584fc54311c9d6ce9a300b13a45195fa',
        },
        {
            'commit': 'ca5a80b40f5b47b8531cdde19d06cdeffc438af2',
            'parent': '86f2caa4847ca6cdeffe25f79135ac0f2e25df09',
            'old_mode': '100644', 'new_mode': '100644',
            'old_blob': '3f30a727aaee8325f81599b2dc673ab8c1c3a920',
            'new_blob': '21c8d641d2b13a284564ac30c5306adc7caac5bd',
            'old_sha256': '14b8e113b0ab09e2103f9210a2556f43584fc54311c9d6ce9a300b13a45195fa',
            'new_sha256': 'a5afd117f7c2d5d2ca584045d1401bdc730e8ad89d53e345afbbb4c78dfa92fb',
        },
        {
            'commit': 'e0ff10dd00bfb492a96bbf7cd60b9c4827cb89ee',
            'parent': '276f0d702af9c8160f2636ea7f245851ba3a0c0a',
            'old_mode': '100644', 'new_mode': '100644',
            'old_blob': '21c8d641d2b13a284564ac30c5306adc7caac5bd',
            'new_blob': '7e1ac11301de3574ae6de85c9b0a8b7ed3ee7844',
            'old_sha256': 'a5afd117f7c2d5d2ca584045d1401bdc730e8ad89d53e345afbbb4c78dfa92fb',
            'new_sha256': '200ce77f8fb15811d13454aa072610428a2341615ac40c608589ea64e568f416',
        },
    ]

    def test_cifar_bic_smoke_amendment_is_exactly_pinned(self):
        api = importlib.import_module('three_dataset_full_matrix_reconcile')
        self.assertEqual({
            'commit': '86a31ea5ab6ad5254fc06e1328b9dc4dbf30f392',
            'parent': '1c5e556aab145698036a4e28f283592a3c6645b0',
            'old_mode': '100644', 'new_mode': '100644',
            'old_blob': '53194e7cae8466838fd848971de15f21822b30ea',
            'new_blob': '73488fbd18f4c559227e2f5f3c4172b3f50e743b',
            'old_sha256': 'c92ad548cd51967153188ca4d1ce688b2ba2b1075ea85556bf500851d4df5efc',
            'new_sha256': '444e4ff90ba3d3f3fd40a5b88dd726bf8ed25c0cb9e27f3f34a5a516a2bf4dcc',
        }, api.APPROVED_OPERATIONAL_MIGRATIONS['three_dataset_formal_driver.py'][5])

    def test_generated_cifar_manifest_and_alias_fix_are_exactly_pinned(self):
        api = importlib.import_module('three_dataset_full_matrix_reconcile')
        chain = api.APPROVED_OPERATIONAL_MIGRATIONS['three_dataset_formal_driver.py']
        self.assertEqual(9, len(chain))
        self.assertEqual(self.generated_cifar_transitions, chain[-3:])

    def test_generated_cifar_pins_match_git_and_reject_each_field_mutation(self):
        api = importlib.import_module('three_dataset_full_matrix_reconcile')
        worktree = Path(__file__).resolve().parent
        path = 'three_dataset_formal_driver.py'
        pins = api.APPROVED_OPERATIONAL_MIGRATIONS
        origin = pins[path][0]['parent']
        head = self.generated_cifar_transitions[-1]['commit']
        try:
            proof = api._approved_operational_migrations(worktree, origin, head, [path])
        except ValueError as error:
            self.fail(f'reviewed generated CIFAR history must be pinned: {error}')
        self.assertEqual(self.generated_cifar_transitions, proof[path][-3:])
        for offset, item in enumerate(self.generated_cifar_transitions, start=6):
            for field, value in item.items():
                with self.subTest(commit=item['commit'], field=field):
                    bad = copy.deepcopy(pins)
                    bad[path][offset][field] = ('100755' if field.endswith('_mode')
                                               else 'f' * len(value))
                    with patch.object(api, 'APPROVED_OPERATIONAL_MIGRATIONS', bad):
                        with self.assertRaisesRegex(ValueError, 'operational migration'):
                            api._approved_operational_migrations(worktree, origin, head, [path])
            with self.subTest(commit=item['commit'], field='missing_transition'):
                bad = copy.deepcopy(pins)
                del bad[path][offset]
                with patch.object(api, 'APPROVED_OPERATIONAL_MIGRATIONS', bad):
                    with self.assertRaisesRegex(ValueError, 'operational migration history differs'):
                        api._approved_operational_migrations(worktree, origin, head, [path])


class FullMatrixReuseTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('three_dataset_full_matrix_reconcile'),
                             'full matrix reuse builder is missing')
        self.api = importlib.import_module('three_dataset_full_matrix_reconcile')
        self.fixture = fixtures.ReconciliationTests()
        self.fixture.prune_all_recovery = True
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.formal = self.fixture.base / 'formal'
        self.make_formal()
        for name, value in (
                ('FORMAL_ROOT', str(self.formal)), ('FORMAL_COMMIT', self.fixture.pilot_commit),
                ('FORMAL_ROOT_PIN', self.pins(self.api.FORMAL_ROOT_PIN)),
                ('FORMAL_EVIDENCE_PIN', self.pins(self.api.FORMAL_EVIDENCE_PIN))):
            context = patch.object(self.api, name, value)
            context.start()
            self.addCleanup(context.stop)
        for name, value in (('ADAPTIVE_ROOT', str(self.fixture.new)),
                            ('ADAPTIVE_PIN', {n: self.api.legacy.sha((self.fixture.new / n).read_bytes())
                                              for n in self.api.ADAPTIVE_PIN})):
            context = patch.object(self.api, name, value)
            context.start()
            self.addCleanup(context.stop)
        context = patch.dict(os.environ, VFCL_EXPERIMENT_PROFILE='full-public-matrix')
        context.start()
        self.addCleanup(context.stop)

    def pins(self, names):
        return {name: self.api.legacy.sha((self.formal / name).read_bytes()) for name in names}

    def make_formal(self):
        f = self.fixture
        self.formal.mkdir(mode=0o700)
        keys = [f'cifar100:{m}:{s}' for m in ('finetune', 'lwf') for s in (42, 43, 44)]
        registry = json.loads((f.old / 'FORMAL_REGISTRY.json').read_bytes())
        registry['formal_cells'] = keys
        census = json.loads((f.old / 'COMPATIBILITY_CENSUS.json').read_bytes())
        census['records'] = []
        for key in keys:
            d, m, s = key.split(':')
            raw = {'status': 'RERUN_REQUIRED', 'reason': 'no_declared_candidate',
                   'spec': {'dataset': d, 'method': m, 'seed': int(s), 'explanation': False},
                   'protocol_sha256': fixtures.digest({'dataset': d, 'method': m}),
                   'source_sha256': '', 'artifact_sha256': {}, 'metrics': None,
                   'metric_formula_version': fixtures.FORMULA, 'trajectory_sha256': ''}
            census['records'].append({**raw, 'spec_key': key, 'admission_record_sha256': fixtures.digest(raw)})
        plan = json.loads((f.old / 'FORMAL_PLAN.json').read_bytes())
        plan.update(formal_cells=keys, missing_jobs=keys, census_sha256=fixtures.digest(census))
        missing = json.loads((f.old / 'MISSING_JOBS.json').read_bytes())
        missing.update(missing_jobs=keys, plan_sha256=fixtures.digest(plan))
        for name, value in (('FORMAL_REGISTRY.json', registry), ('COMPATIBILITY_CENSUS.json', census),
                            ('FORMAL_PLAN.json', plan), ('MISSING_JOBS.json', missing)):
            fixtures.install(self.formal / name, value)
        details = self.formal.stat()
        identity = json.loads((f.old / 'FORMAL_ROOT_IDENTITY.json').read_bytes())
        identity.update(root_dev=details.st_dev, root_inode=details.st_ino,
                        root_ctime_ns=details.st_ctime_ns, root_size=details.st_size,
                        plan_sha256=fixtures.digest(plan), missing_jobs_sha256=fixtures.digest(missing))
        identity_hash = fixtures.install(self.formal / 'FORMAL_ROOT_IDENTITY.json', identity)
        details = (self.formal / 'FORMAL_ROOT_IDENTITY.json').stat()
        owner = {'dev': details.st_dev, 'inode': self.formal.stat().st_ino,
                 'ctime_ns': details.st_ctime_ns, 'size': details.st_size, 'hash': identity_hash}
        for marker in ('FAILED_JOB', 'FORMAL_STOPPED'):
            shutil.copy2(f.old / marker, self.formal / marker)
        for key in keys:
            d, m, s = key.split(':')
            old_key = f'{d}:{m}:42'
            run = self.formal / 'runs' / quote(key, safe='')
            shutil.copytree(f.old / 'runs' / quote(old_key, safe=''), run)
            read = lambda name: json.loads((run / name).read_bytes())
            job = read('FORMAL_JOB_SPEC.json')
            command = job['command']
            for option, value in (('--seed', s), ('--exp_name', run.name),
                                  ('--results_dir', str(self.formal / 'runs'))):
                command[command.index(option) + 1] = value
            job.update(spec_key=key, spec={**job['spec'], 'seed': int(s)}, run_dir=str(run),
                       plan_sha256=fixtures.digest(plan), root_identity=owner,
                       command_sha256=fixtures.digest(command))
            job_hash = fixtures.install(run / 'FORMAL_JOB_SPEC.json', job)
            claim = read('CLAIM_OWNER.json')
            claim.update(job=key, root_identity=owner)
            claim_hash = fixtures.install(run / 'CLAIM_OWNER.json', claim)
            launch = read('LAUNCH_STARTED.json')
            launch.update(spec_key=key, root_identity=owner, plan_sha256=fixtures.digest(plan),
                          job_spec_sha256=job_hash, claim_sha256=claim_hash,
                          command_sha256=job['command_sha256'])
            launch_hash = fixtures.install(run / 'LAUNCH_STARTED.json', launch)
            resource = read('RESOURCE_EVIDENCE.json')
            resource.update(spec_key=key, plan_sha256=fixtures.digest(plan), job_spec_sha256=job_hash,
                            claim_sha256=claim_hash, launch_sha256=launch_hash,
                            command_sha256=job['command_sha256'])
            fixtures.install(run / 'RESOURCE_EVIDENCE.json', resource)
            record = json.loads((f.old / 'records' / (quote(old_key, safe='') + '.json')).read_bytes())
            record.update(spec_key=key, seed=int(s), plan_sha256=fixtures.digest(plan),
                          claim_sha256=claim_hash, launch_sha256=launch_hash,
                          command_sha256=job['command_sha256'])
            record['record_sha256'] = fixtures.digest({k: v for k, v in record.items() if k != 'record_sha256'})
            fixtures.install(self.formal / 'records' / (run.name + '.json'), record)
            prune = read('PRUNE_PLAN.json')
            prune.update(spec_key=key, record_sha256=record['record_sha256'])
            prune_hash = fixtures.install(run / 'PRUNE_PLAN.json', prune)
            receipt = read('PRUNED_EVIDENCE.json')
            receipt.update(spec_key=key, record_sha256=record['record_sha256'], plan_sha256=prune_hash)
            fixtures.install(run / 'PRUNED_EVIDENCE.json', receipt)

    def build(self):
        f = self.fixture
        with patch.object(self.api, 'FORMAL_SOURCE_FILES',
                          getattr(self, 'inventory', ('main.py',)), create=True):
            return self.api.build_reuse_bundle(f.old, f.new, self.formal, f.repo)

    def publish(self):
        f = self.fixture
        with patch.object(self.api, 'FORMAL_SOURCE_FILES',
                          getattr(self, 'inventory', ('main.py',)), create=True):
            return self.api.reconcile(f.old, f.new, self.formal, f.repo, f.out)

    def source_inventory_roots(self, missing=(), changed=None):
        """Generate Git-bound historical sources, then advance only current Git."""
        from adaptive_consolidation_audit import FORMAL_SOURCE_FILES
        f = self.fixture
        self.inventory = FORMAL_SOURCE_FILES
        f.repo = f.base / 'inventory-git'
        f.repo.mkdir()
        fixtures.git(f.repo, 'init', '-q')
        fixtures.git(f.repo, 'config', 'user.name', 'Fixture')
        fixtures.git(f.repo, 'config', 'user.email', 'fixture@example.invalid')
        for name in self.inventory:
            if name not in missing:
                path = f.repo / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# historical ' + name + '\n')
        f.base_commit = f.pilot_commit = f.recovery_commit = f.commit('historical sources')
        f.sources = {name: self.api.legacy.sha((f.repo / name).read_bytes())
                     for name in self.inventory if name not in missing}
        f.old, f.new = f.base / 'inventory-pilot', f.base / 'inventory-adaptive'
        f.make_root(f.old, False)
        f.make_root(f.new, True)
        self.formal = f.base / 'inventory-formal'
        self.make_formal()
        for module, values in (
                (self.api.legacy, {'PILOT_ROOT': str(f.old), 'PILOT_COMMIT': f.pilot_commit,
                                  'BASE_COMMIT': f.base_commit, 'PILOT_PIN': f.pin_old(),
                                  'PILOT_EVIDENCE_SHA256': f.pin_evidence()}),
                (self.api, {'FORMAL_ROOT': str(self.formal), 'FORMAL_COMMIT': f.pilot_commit,
                            'FORMAL_ROOT_PIN': self.pins(self.api.FORMAL_ROOT_PIN),
                            'FORMAL_EVIDENCE_PIN': self.pins(self.api.FORMAL_EVIDENCE_PIN),
                            'ADAPTIVE_ROOT': str(f.new),
                            'ADAPTIVE_PIN': {n: self.api.legacy.sha((f.new / n).read_bytes())
                                             for n in self.api.ADAPTIVE_PIN}})):
            for name, value in values.items():
                context = patch.object(module, name, value)
                context.start()
                self.addCleanup(context.stop)
        for name in missing:
            path = f.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# newly required ' + name + '\n')
        for name in ((changed,) if isinstance(changed, str) else (changed or ())):
            (f.repo / name).write_text('# incompatible current source\n')
        (f.repo / 'three_dataset_formal_driver.py').write_text('# unapproved operational history\n')
        f.commit('current inventory and unapproved operational change')

    def test_old_inventory_publishes_zero_reuse_before_unapproved_migration(self):
        import three_dataset_full_matrix_report as report
        missing = ('cl_methods/adagauss.py', 'cl_methods/afc.py', 'cl_methods/der_pp.py',
                   'cl_methods/er_ace.py', 'cl_methods/ewc.py', 'cl_methods/lwf_wa.py',
                   'cl_methods/proto_fedspace.py', 'cl_methods/target.py')
        self.source_inventory_roots(missing=missing, changed=(
            'adaptive_consolidation_audit.py', 'config.py', 'runner.py', 'vfl_trainer.py'))
        f = self.fixture
        roots = (f.old, f.new, self.formal)
        before = [fixtures.readonly_snapshot(root) for root in roots]
        try:
            bundle = self.publish()
        except ValueError as error:
            self.fail('all old-inventory candidates must be rejected before migration: ' + str(error))
        self.assertEqual([], bundle['admitted'])
        self.assertEqual([], bundle['ambiguous'])
        self.assertEqual(19, len(bundle['rejected']))
        expected_keys = {f'{d}:{m}:42' for d in fixtures.DATASETS for m in fixtures.METHODS}
        expected_keys.update(self.api.OPTIONAL_KEYS)
        self.assertEqual(expected_keys, {row['spec_key'] for row in bundle['rejected']})
        physical = [candidate for row in bundle['rejected'] for candidate in row['candidates']]
        self.assertEqual(21, len(physical))
        expected_reason = 'source-inventory-mismatch: missing=' + ','.join(missing) + '; extra='
        self.assertEqual({expected_reason}, {row['reason'] for row in bundle['rejected']})
        self.assertEqual({expected_reason}, {row['reason'] for row in physical})
        expected_records = {str(path): self.api.legacy.sha(path.read_bytes())
                            for root in roots for path in (root / 'records').iterdir()}
        self.assertEqual(expected_records, {row['origin_record_path']: row['origin_record_sha256']
                                            for row in physical})
        self.assertTrue(all(row['compatibility'] == 'REJECTED' for row in physical))
        self.assertTrue(all('migration_sha256' not in source for source in bundle['sources']))
        with patch.object(report.driver, '_source_commit', return_value=bundle['current_commit']):
            self.assertEqual(bundle, report.load_reuse_bundle(f.out / 'FULL_MATRIX_REUSE.json'))
            plan = report.driver.build_plan(report.reuse_census(bundle))
        self.assertEqual(plan['formal_cells'], plan['missing_jobs'])
        self.assertEqual(126, len(plan['missing_jobs']))
        self.assertEqual(bundle, self.build())
        self.assertEqual(before, [fixtures.readonly_snapshot(root) for root in roots])
        self.assertEqual(0, json.loads((f.out / 'REUSE_AUDIT_SUCCESS').read_bytes())['row_count'])

    def test_full_inventory_scientific_mismatch_needs_no_operational_approval(self):
        self.source_inventory_roots(changed='main.py')
        try:
            bundle = self.publish()
        except ValueError as error:
            self.fail('conclusive scientific mismatch must precede migration: ' + str(error))
        self.assertEqual([], bundle['admitted'])
        self.assertEqual(19, len(bundle['rejected']))
        self.assertEqual({'source-hash-mismatch: main.py'},
                         {row['reason'] for row in bundle['rejected']})

    def test_full_inventory_survivor_keeps_unapproved_migration_fatal(self):
        self.source_inventory_roots()
        with self.assertRaisesRegex(ValueError, 'operational migration history differs'):
            self.publish()
        self.assertFalse(self.fixture.out.exists())

    def test_migration_covered_source_mismatch_is_not_conclusive_rejection(self):
        self.source_inventory_roots(changed='runner.py')
        with self.assertRaisesRegex(ValueError, 'operational migration history differs'):
            self.publish()
        self.assertFalse(self.fixture.out.exists())

    def test_viable_formal_duplicate_prevents_zero_publication(self):
        self.source_inventory_roots(changed='main.py')
        key = 'cifar100:finetune:42'
        run = self.formal / 'runs' / quote(key, safe='')
        current_hash = self.api.legacy.sha((self.fixture.repo / 'main.py').read_bytes())
        self.fixture.alter(run / 'FORMAL_JOB_SPEC.json',
                           lambda job: job['source_sha256'].update({'main.py': current_hash}))
        self.change_record(key, lambda record: record['artifact_sha256'].update({'source:main.py': current_hash}))
        with patch.object(self.api, 'FORMAL_EVIDENCE_PIN', self.pins(self.api.FORMAL_EVIDENCE_PIN)):
            with self.assertRaisesRegex(ValueError, 'operational migration history differs'):
                self.publish()
        self.assertFalse(self.fixture.out.exists())

    def test_zero_reuse_rejects_dirty_current_worktree(self):
        self.source_inventory_roots(changed='main.py')
        (self.fixture.repo / 'main.py').write_text('# uncommitted source\n')
        with self.assertRaisesRegex(ValueError, 'worktree is not clean'):
            self.publish()
        self.assertFalse(self.fixture.out.exists())

    def test_zero_reuse_rechecks_physical_evidence_and_current_sources(self):
        self.source_inventory_roots(changed='main.py')
        original = os.link
        record = self.formal / 'records' / (quote('cifar100:lwf:42', safe='') + '.json')
        for index, target in enumerate((record, self.fixture.repo / 'main.py')):
            with self.subTest(target=target):
                self.fixture.out = self.fixture.base / f'zero-race-{index}'
                content, mode = target.read_bytes(), stat.S_IMODE(target.stat().st_mode)
                def mutate(src, dst, *args, **kwargs):
                    result = original(src, dst, *args, **kwargs)
                    if dst == 'FULL_MATRIX_REUSE_AUDIT.json':
                        target.chmod(0o600)
                        target.write_bytes(content + b'\n')
                        target.chmod(mode)
                    return result
                try:
                    with patch.object(os, 'link', side_effect=mutate):
                        with self.assertRaisesRegex(ValueError, 'consumed input changed'):
                            self.publish()
                    self.assertFalse((self.fixture.out / 'REUSE_AUDIT_SUCCESS').exists())
                finally:
                    target.chmod(0o600)
                    target.write_bytes(content)
                    target.chmod(mode)

    def test_old_25_file_candidate_is_rejected_and_planned_for_rerun_read_only(self):
        import three_dataset_formal_audit as audit
        import three_dataset_formal_driver as driver
        from test_three_dataset_formal_audit import RealProducerFixture

        historical_inventory = (
            'adaptive_consolidation_audit.py', 'adaptive_dual_branch_validation.py',
            'adaptive_head_consolidation.py', 'adaptive_tinyimagenet_heldout.py',
            'bic_calibration.py', 'calibration_split.py', 'cl_methods/__init__.py',
            'cl_methods/er.py', 'cl_methods/fedprotip_vfl.py',
            'cl_methods/finetune.py', 'cl_methods/gpm.py', 'cl_methods/lwf.py',
            'cl_methods/proto_evolve.py', 'config.py', 'data_utils.py',
            'determinism.py', 'head_consolidation.py', 'main.py', 'metrics.py',
            'models.py', 'runner.py', 'ul_methods/__init__.py',
            'ul_methods/retrain.py', 'vfl_trainer.py',
            'three_dataset_formal_runtime.py',
        )
        candidate = RealProducerFixture('finetune')
        self.addCleanup(candidate.close)
        declaration = copy.deepcopy(candidate.declaration)
        declaration['source_files'] = [
            entry for entry in declaration['source_files']
            if entry['logical_path'] in historical_inventory
        ]
        self.assertEqual(set(historical_inventory), {
            entry['logical_path'] for entry in declaration['source_files']})
        original_declaration = copy.deepcopy(declaration)
        before = fixtures.readonly_snapshot(candidate.root)
        key = driver.spec_key(candidate.spec)
        with patch.object(audit, '_AUTHORITATIVE_DATA', {
                candidate.spec.dataset: candidate.authoritative_data()}), \
                patch.object(audit, '_AUTHORITATIVE_MANIFEST', {
                    candidate.spec.dataset: candidate.authoritative_manifest()}):
            census = driver.build_census({key: declaration})
        rejected = next(row for row in census['records'] if row['spec_key'] == key)
        self.assertEqual('REJECTED', rejected['status'])
        self.assertEqual('source-inventory-mismatch', rejected['reason'])
        self.assertEqual('', rejected['source_sha256'])
        plan = driver.build_plan(census)
        self.assertIn(key, plan['missing_jobs'])
        self.assertEqual(plan['formal_cells'], plan['missing_jobs'])
        self.assertEqual(126, len(plan['missing_jobs']))
        self.assertEqual(original_declaration, declaration)
        self.assertEqual(before, fixtures.readonly_snapshot(candidate.root))

    def test_exact_nineteen_members_and_schema(self):
        bundle = self.build()
        self.assertEqual('full_public_matrix_reuse_v1', bundle['kind'])
        self.assertEqual('full-public-matrix', bundle['experiment_profile'])
        self.assertEqual(19, len(bundle['admitted']))
        self.assertEqual(19, len({r['spec_key'] for r in bundle['admitted']}))
        self.assertEqual(4, sum(r['seed'] in (43, 44) for r in bundle['admitted']))
        self.assertEqual([], bundle['ambiguous'])
        self.assertEqual([], bundle['rejected'])
        self.assertEqual({'kind', 'experiment_profile', 'registry_sha256', 'metric_formula_version',
                          'current_commit', 'sources', 'admitted', 'rejected', 'ambiguous'}, set(bundle))
        fields = set('spec_key dataset method seed metrics resource artifact_sha256 source_sha256 '
                     'trajectory_sha256 origin_root origin_record_path origin_record_sha256 '
                     'origin_source_commit compatibility migration_sha256'.split())
        for row in bundle['admitted']:
            self.assertEqual(fields, set(row))
            self.assertEqual('ADMITTED', row['compatibility'])
        self.assertEqual(bundle, self.build())

    def test_recovery_origin_remains_frozen_after_current_commit_advances(self):
        f = self.fixture
        (f.repo / 'three_dataset_full_matrix_reconcile.py').write_text('# reuse layer\n')
        head = f.commit('reuse only')
        bundle = self.build()
        self.assertEqual(head, bundle['current_commit'])
        self.assertEqual({f.recovery_commit}, {r['origin_source_commit'] for r in bundle['admitted']
                                             if r['method'] == 'adaptive'})

    def test_confirmed_rehashed_adaptive_metrics_still_rejected(self):
        f = self.fixture
        path = f.new / 'records' / (quote('isolet:adaptive:42', safe='') + '.json')
        record = json.loads(path.read_bytes())
        record['metrics']['aa_trajectory'][0] = 0.8
        record['record_sha256'] = fixtures.digest({k: v for k, v in record.items() if k != 'record_sha256'})
        fixtures.install(path, record)
        with self.assertRaisesRegex(ValueError, 'adaptive_pin_mismatch'):
            self.publish()
        self.assertFalse(f.out.exists())

    def test_optional_missing_preserves_fifteen_with_exact_reason(self):
        self.formal = self.fixture.base / 'unavailable'
        with patch.object(self.api, 'FORMAL_ROOT', str(self.formal)):
            bundle = self.build()
        self.assertEqual(15, len(bundle['admitted']))
        self.assertEqual({'formal_root_unavailable'}, {r['reason'] for r in bundle['rejected']})
        self.assertEqual(set(self.api.OPTIONAL_KEYS), {r['spec_key'] for r in bundle['rejected']})

    def test_optional_pinned_evidence_mutation_rejects_all_four(self):
        for name in (*self.api.FORMAL_ROOT_PIN, *self.api.FORMAL_EVIDENCE_PIN):
            with self.subTest(name=name):
                path = self.formal / name
                original = path.read_bytes()
                path.chmod(0o600)
                path.write_bytes(original + b' ')
                path.chmod(0o444)
                bundle = self.build()
                self.assertEqual(15, len(bundle['admitted']))
                self.assertEqual({'formal_pin_mismatch: ' + name}, {r['reason'] for r in bundle['rejected']})
                path.chmod(0o600)
                path.write_bytes(original)
                path.chmod(0o444)

    def test_duplicate_seed42_metric_inequality_rejects_all_four(self):
        key = 'cifar100:finetune:42'
        self.change_record(key, lambda r: r['metrics'].update(aa_final=0.9))
        with patch.object(self.api, 'FORMAL_EVIDENCE_PIN', self.pins(self.api.FORMAL_EVIDENCE_PIN)):
            bundle = self.build()
        self.assertEqual(15, len(bundle['admitted']))
        self.assertEqual({'duplicate_seed42_metrics_differ: ' + key}, {r['reason'] for r in bundle['rejected']})

    def change_record(self, key, change):
        run = self.formal / 'runs' / quote(key, safe='')
        path = self.formal / 'records' / (run.name + '.json')
        record = json.loads(path.read_bytes())
        change(record)
        record['record_sha256'] = fixtures.digest({k: v for k, v in record.items() if k != 'record_sha256'})
        fixtures.install(path, record)
        self.fixture.alter(run / 'RESOURCE_EVIDENCE.json', lambda r: r.update(resource=record['resource']))
        self.fixture.alter(run / 'PRUNE_PLAN.json', lambda r: r.update(record_sha256=record['record_sha256']))
        self.fixture.alter(run / 'PRUNED_EVIDENCE.json', lambda r: r.update(
            record_sha256=record['record_sha256'],
            plan_sha256=self.api.legacy.sha((run / 'PRUNE_PLAN.json').read_bytes())))

    def test_duplicate_resource_protocol_inequality_rejects_four(self):
        key = 'cifar100:lwf:42'
        self.change_record(key, lambda r: r['resource'].update(communication_bytes=1))
        with patch.object(self.api, 'FORMAL_EVIDENCE_PIN', self.pins(self.api.FORMAL_EVIDENCE_PIN)):
            bundle = self.build()
        self.assertEqual(15, len(bundle['admitted']))
        self.assertEqual({'duplicate_seed42_resource_protocol_differs: ' + key},
                         {r['reason'] for r in bundle['rejected']})

    def test_duplicate_runtime_observations_are_not_protocol(self):
        self.change_record('cifar100:lwf:42', lambda r: r['resource'].update(
            runtime_seconds=99.5, peak_gpu_memory_bytes=999))
        with patch.object(self.api, 'FORMAL_EVIDENCE_PIN', self.pins(self.api.FORMAL_EVIDENCE_PIN)):
            self.assertEqual(19, len(self.build()['admitted']))

    def test_only_two_frozen_entrypoint_prefixes_are_normalized(self):
        command = ['/python', '/home/c3080/YangXiaoXiang/VF-CL-worktrees/three-dataset-formal-comparison-3080/main.py',
                   '--seed', '43', '--results_dir', '/formal/runs', '--exp_name', 'cifar100%3Afinetune%3A43']
        pilot = list(command)
        pilot[1] = '/home/c3080/YangXiaoXiang/VF-CL-worktrees/formal-pipelined-audit-3080/main.py'
        self.assertEqual(self.api._command_family(command), self.api._command_family(pilot))
        pilot[1] = '/third/main.py'
        self.assertNotEqual(self.api._command_family(command), self.api._command_family(pilot))

    def test_origin_to_current_rejects_even_other_legacy_operational_edits(self):
        f = self.fixture
        (f.repo / 'prune_completed_runs.py').write_text('# later operational edit\n')
        f.commit('outside full reuse scope')
        with self.assertRaisesRegex(ValueError, 'origin-to-current Git scope|operational migration'):
            self.publish()
        self.assertFalse(f.out.exists())

    def operational_chain(self):
        f = self.fixture
        paths = ('three_dataset_formal_driver.py', 'three_dataset_full_matrix_report.py',
                 'three_dataset_resource_gate.py', 'run_three_dataset_formal_comparison.sh',
                 'prune_completed_runs.py')
        parent = f.recovery_commit
        for path in paths:
            (f.repo / path).write_bytes(b'# reviewed operational source\n')
        commit = f.commit('reviewed operational slice')
        chain = {path: [{'commit': commit, 'parent': parent,
                        'old_mode': '000000', 'new_mode': '100644',
                        'old_blob': '0' * 40,
                        'new_blob': fixtures.git(f.repo, 'rev-parse', commit + ':' + path),
                        'old_sha256': self.api.legacy.sha(b''),
                        'new_sha256': self.api.legacy.sha((f.repo / path).read_bytes())}]
                 for path in paths}
        path = paths[0]
        (f.repo / path).write_bytes(b'# next reviewed operational source\n')
        final = f.commit('next reviewed operational slice')
        chain[path].append({**chain[path][0], 'commit': final, 'parent': commit,
            'old_mode': '100644', 'old_blob': chain[path][0]['new_blob'],
            'old_sha256': chain[path][0]['new_sha256'],
            'new_blob': fixtures.git(f.repo, 'rev-parse', final + ':' + path),
            'new_sha256': self.api.legacy.sha((f.repo / path).read_bytes())})
        return chain

    def test_exact_operational_chain_admits_without_widening_legacy_api(self):
        chain = self.operational_chain()
        with patch.object(self.api, 'APPROVED_OPERATIONAL_MIGRATIONS', chain, create=True):
            bundle = self.build()
        self.assertEqual(19, len(bundle['admitted']))
        for source in bundle['sources']:
            self.assertEqual(chain, source['git_scope']['approved_operational_migration'])
        with self.assertRaisesRegex(ValueError, 'Git scope forbids source'):
            self.api.legacy.collect_seed42_rows(self.fixture.old, self.fixture.new,
                                               self.fixture.repo, self.fixture.repo)

    def test_operational_chain_pin_mutations_fail_before_publication(self):
        chain = self.operational_chain()
        path = 'three_dataset_formal_driver.py'
        for index, item in enumerate(chain[path]):
            for field in item:
                with self.subTest(index=index, field=field):
                    bad = copy.deepcopy(chain)
                    bad[path][index][field] = ('100755' if field.endswith('_mode')
                                             else 'f' * len(item[field]))
                    with patch.object(self.api, 'APPROVED_OPERATIONAL_MIGRATIONS', bad, create=True):
                        with self.assertRaisesRegex(ValueError, 'operational migration'):
                            self.publish()
                    self.assertFalse(self.fixture.out.exists())

    def test_operational_chain_rejects_unpinned_latest_driver_transition(self):
        chain = self.operational_chain()
        chain['three_dataset_formal_driver.py'].pop()
        with patch.object(self.api, 'APPROVED_OPERATIONAL_MIGRATIONS', chain):
            with self.assertRaisesRegex(ValueError, 'operational migration history differs'):
                self.build()
        self.assertFalse(self.fixture.out.exists())

    def test_operational_chain_rejects_reverted_edits_and_scientific_paths(self):
        chain = self.operational_chain()
        f = self.fixture
        path = f.repo / 'three_dataset_formal_driver.py'
        reviewed = path.read_bytes()
        path.write_bytes(b'# unreviewed intermediate change\n')
        f.commit('unreviewed')
        path.write_bytes(reviewed)
        f.commit('revert does not erase history')
        with patch.object(self.api, 'APPROVED_OPERATIONAL_MIGRATIONS', chain, create=True):
            with self.assertRaisesRegex(ValueError, 'operational migration history'):
                self.publish()
        for forbidden in ('runner.py', 'models.py', 'data_utils.py',
                          'three_dataset_formal_metrics.py', 'cl_methods/ewc.py'):
            with self.subTest(path=forbidden):
                bad = {**chain, forbidden: chain['three_dataset_formal_driver.py']}
                with patch.object(self.api, 'APPROVED_OPERATIONAL_MIGRATIONS', bad, create=True):
                    with self.assertRaisesRegex(ValueError, 'operational migration paths'):
                        self.publish()
        self.assertFalse(f.out.exists())

    def test_publication_does_not_overwrite_racing_target(self):
        original = os.link
        def race(src, dst, *args, **kwargs):
            if dst == 'FULL_MATRIX_REUSE.json':
                (self.fixture.out / dst).write_bytes(b'other owner')
            return original(src, dst, *args, **kwargs)
        with patch.object(os, 'link', side_effect=race):
            with self.assertRaises((ValueError, FileExistsError)):
                self.publish()
        self.assertEqual(b'other owner', (self.fixture.out / 'FULL_MATRIX_REUSE.json').read_bytes())
        self.assertFalse((self.fixture.out / 'REUSE_AUDIT_SUCCESS').exists())

    def test_confirmed_authority_mutations_fail_before_output(self):
        f = self.fixture
        for root, key in ((f.old, 'cifar100:finetune:42'), (f.new, 'cifar100:adaptive:42')):
            name = quote(key, safe='')
            paths = [(root / 'records' / (name + '.json'), field)
                     for field in ('record_sha256', 'method', 'seed', 'protocol_sha256', 'metrics')]
            paths += [(root / 'runs' / name / 'FORMAL_JOB_SPEC.json', 'source_sha256'),
                      (root / 'runs' / name / 'PRUNED_EVIDENCE.json', 'plan_sha256'),
                      (root / 'FORMAL_ROOT_IDENTITY.json', 'root_inode')]
            for path, field in paths:
                with self.subTest(root=root.name, field=field):
                    original = json.loads(path.read_bytes())
                    fixtures.install(path, {**original, field: 'tampered'})
                    with self.assertRaises((ValueError, OSError, TypeError)):
                        self.publish()
                    self.assertFalse(f.out.exists())
                    fixtures.install(path, original)

    def test_source_migration_tampering_fails_before_output(self):
        f = self.fixture
        (f.repo / 'runner.py').write_bytes(b'unapproved scientific change\n')
        f.commit('unapproved')
        with self.assertRaises(ValueError):
            self.publish()
        self.assertFalse(f.out.exists())

    def test_profile_must_be_explicit(self):
        with patch.dict(os.environ, VFCL_EXPERIMENT_PROFILE='seed42-pilot'):
            with self.assertRaisesRegex(ValueError, 'full-public-matrix'):
                self.publish()
        self.assertFalse(self.fixture.out.exists())

    def test_exclusive_publication_modes_hashes_and_input_immutability(self):
        f = self.fixture
        before = [fixtures.snapshot(root) for root in (f.old, f.new, self.formal)]
        bundle = self.publish()
        names = {'FULL_MATRIX_REUSE.json', 'FULL_MATRIX_REUSE_AUDIT.json', 'REUSE_AUDIT_SUCCESS'}
        self.assertEqual(names, {p.name for p in f.out.iterdir()})
        self.assertEqual(0o700, stat.S_IMODE(f.out.stat().st_mode))
        for path in f.out.iterdir():
            self.assertEqual(0o444, stat.S_IMODE(path.stat().st_mode))
        self.assertEqual(bundle, json.loads((f.out / 'FULL_MATRIX_REUSE.json').read_bytes()))
        success = json.loads((f.out / 'REUSE_AUDIT_SUCCESS').read_bytes())
        self.assertEqual(19, success['row_count'])
        self.assertEqual({n: self.api.legacy.sha((f.out / n).read_bytes())
                          for n in names - {'REUSE_AUDIT_SUCCESS'}}, success['artifact_sha256'])
        self.assertEqual(before, [fixtures.snapshot(root) for root in (f.old, f.new, self.formal)])
        with self.assertRaises((ValueError, FileExistsError)):
            self.publish()

    def test_publication_rechecks_input_and_installs_success_last(self):
        f = self.fixture
        original = os.link
        installed = []
        def mutate(src, dst, *args, **kwargs):
            result = original(src, dst, *args, **kwargs)
            installed.append(dst)
            if dst == 'FULL_MATRIX_REUSE_AUDIT.json':
                path = f.new / 'records' / (quote('isolet:adaptive:42', safe='') + '.json')
                f.alter(path, lambda v: v.update(seed=43))
            return result
        with patch.object(os, 'link', side_effect=mutate):
            with self.assertRaises(ValueError):
                self.publish()
        self.assertEqual(['FULL_MATRIX_REUSE.json', 'FULL_MATRIX_REUSE_AUDIT.json'], installed)
        self.assertFalse((f.out / 'REUSE_AUDIT_SUCCESS').exists())

    def test_existing_symlink_and_input_descendant_outputs_rejected(self):
        f = self.fixture
        for out in (f.old / 'output', f.new / 'output', self.formal / 'output'):
            f.out = out
            with self.assertRaises(ValueError):
                self.publish()
            self.assertFalse(out.exists())
        f.out = f.base / 'link'
        f.out.symlink_to(f.base, target_is_directory=True)
        with self.assertRaises((ValueError, OSError)):
            self.publish()

    def test_wrong_optional_path_cannot_publish_under_fixed_formal_root(self):
        fixed_formal = self.formal
        before = fixtures.snapshot(fixed_formal)
        self.formal = self.fixture.base / 'wrong-optional-formal'
        self.fixture.out = fixed_formal / 'forbidden-output'
        with self.assertRaisesRegex(ValueError, 'output cannot be inside an input root'):
            self.publish()
        self.assertFalse(self.fixture.out.exists())
        self.assertEqual(before, fixtures.snapshot(fixed_formal))

    def test_double_leading_slash_output_cannot_alias_fixed_formal_root(self):
        fixed_formal = self.formal
        before = fixtures.snapshot(fixed_formal)
        self.formal = self.fixture.base / 'wrong-optional-formal'
        self.fixture.out = '//' + str(fixed_formal / 'forbidden-alias-output').lstrip('/')
        with self.assertRaisesRegex(ValueError, 'output.*single leading slash'):
            self.publish()
        self.assertFalse((fixed_formal / 'forbidden-alias-output').exists())
        self.assertEqual(before, fixtures.snapshot(fixed_formal))

    def test_cli_preserves_raw_repeated_leading_slashes_for_rejection(self):
        f = self.fixture
        for prefix in ('//', '///'):
            with self.subTest(prefix=prefix):
                output = prefix + str(f.base / 'cli-alias-output').lstrip('/')
                error = io.StringIO()
                with redirect_stderr(error), patch.object(self.api.legacy, 'directory',
                        side_effect=AssertionError('filesystem opened before output validation')):
                    status = self.api.main(['--pilot-root', str(f.old), '--adaptive-root', str(f.new),
                        '--formal-root', str(self.formal), '--worktree', str(f.repo), '--output-root', output])
                self.assertEqual(1, status)
                self.assertIn('single leading slash', error.getvalue())
                self.assertFalse((f.base / 'cli-alias-output').exists())


if __name__ == '__main__':
    unittest.main()
