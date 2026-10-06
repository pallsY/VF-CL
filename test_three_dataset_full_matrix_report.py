"""Generated-only fixtures for the full-matrix authority/report boundary."""
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest import mock

import three_dataset_formal_driver as driver
import three_dataset_formal_registry as registry
from test_three_dataset_formal_driver import _completed_record


class FullMatrixReportTests(unittest.TestCase):
    def setUp(self):
        self.profile = mock.patch.dict(os.environ, {registry.PROFILE_ENV: registry.FULL_MATRIX_PROFILE})
        self.profile.start()
        self.addCleanup(self.profile.stop)
        self.temp = tempfile.TemporaryDirectory(prefix='generated_full_report_')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.bundle = self.make_bundle(19)

    def make_bundle(self, count):
        rows = []
        for spec in registry.formal_specs()[:count]:
            record = _completed_record(spec, float(spec.seed - 40))
            rows.append({
                **{name: record[name] for name in ('spec_key', 'dataset', 'method', 'seed', 'metrics', 'resource', 'trajectory_sha256')},
                'artifact_sha256': {'source:main.py': 'a' * 64, 'results': 'b' * 64},
                'source_sha256': {'main.py': 'a' * 64},
                'origin_root': '/generated/historical',
                'origin_record_path': '/generated/historical/records/' + registry.safe_spec_name(spec) + '.json',
                'origin_record_sha256': 'c' * 64, 'origin_source_commit': 'd' * 40,
                'compatibility': 'ADMITTED', 'migration_sha256': 'e' * 64,
            })
        return {'kind': 'full_public_matrix_reuse_v1', 'experiment_profile': registry.FULL_MATRIX_PROFILE,
                'registry_sha256': registry.registry_sha256(), 'metric_formula_version': driver.FORMULA_VERSION,
                'current_commit': driver._source_commit(), 'sources': [{'root': '/generated/historical', 'source_commit': 'd' * 40}],
                'admitted': rows,
                'rejected': [{'spec_key': driver.spec_key(s), 'origin_root': '/generated/historical',
                              'reason': 'generated candidate rejected'}
                             for s in registry.formal_specs()[15:19]] if count == 15 else [],
                'ambiguous': []}

    def publish(self, bundle=None, name='bundle'):
        bundle = self.bundle if bundle is None else bundle
        root = self.base / name
        root.mkdir()
        audit = {'kind': 'full_public_matrix_reuse_audit_v1', 'validation': 'ADMITTED',
                 'bundle_sha256': driver._digest(bundle), 'admitted_count': len(bundle['admitted']),
                 'rejected': bundle['rejected'], 'ambiguous': bundle['ambiguous']}
        for filename, value in [('FULL_MATRIX_REUSE.json', bundle), ('FULL_MATRIX_REUSE_AUDIT.json', audit)]:
            driver.install_json_exclusive(root / filename, value)
        marker = {'kind': 'full_public_matrix_reuse_audit_success', 'row_count': len(bundle['admitted']),
                  'artifact_sha256': {p.name: driver._sha256_bytes(p.read_bytes()) for p in root.iterdir()}}
        driver.install_json_exclusive(root / 'REUSE_AUDIT_SUCCESS', marker)
        return root / 'FULL_MATRIX_REUSE.json'

    def install(self, count=19):
        self.bundle = self.make_bundle(count)
        root = self.base / 'authority'
        census = driver._install_full_census(root, self.publish())
        plan = driver._install_plan(root)
        return root, census, plan

    def test_import_uses_actual_admitted_membership(self):
        self.assertTrue(hasattr(driver, '_install_full_census'), 'full census import is missing')
        for count in (19, 15):
            with self.subTest(count=count):
                census = driver._install_full_census(self.base / f'authority{count}', self.publish(self.make_bundle(count), f'bundle{count}'))
                self.assertEqual(126, len(census['records']))
                self.assertEqual(count, sum(r['status'] == 'REUSABLE' for r in census['records']))
                self.assertEqual(126 - count, len(driver.build_plan(census)['missing_jobs']))

    def test_import_rejects_changed_identity_membership_and_payload(self):
        self.assertTrue(hasattr(driver, '_install_full_census'), 'full census import is missing')
        mutations = [lambda b: b.update(experiment_profile='formal'),
                     lambda b: b.update(registry_sha256='f' * 64),
                     lambda b: b.update(current_commit='f' * 40),
                     lambda b: b['admitted'].append(copy.deepcopy(b['admitted'][0])),
                     lambda b: b['admitted'][0].update(method='prl', spec_key='cifar100:prl:42'),
                     lambda b: b['admitted'][0].update(dataset='unknown', spec_key='unknown:finetune:42')]
        for i, mutate in enumerate(mutations):
            bundle = copy.deepcopy(self.bundle)
            mutate(bundle)
            with self.subTest(i=i), self.assertRaises(ValueError):
                driver._install_full_census(self.base / f'bad{i}', self.publish(bundle, f'b{i}'))
        for field in ('metrics', 'resource', 'artifact_sha256'):
            path = self.publish(name=field)
            changed = copy.deepcopy(self.bundle)
            changed['admitted'][0][field] = {}
            path.chmod(0o644)
            path.write_bytes(driver._canonical_json(changed) + b'\n')
            path.chmod(0o444)
            with self.subTest(field=field), self.assertRaises(ValueError):
                driver._install_full_census(self.base / f'bad-{field}', path)

    def test_import_rejects_unsafe_or_noncanonical_input_and_absent_success(self):
        self.assertTrue(hasattr(driver, '_install_full_census'), 'full census import is missing')
        for mode in ('writable', 'symlink', 'noncanonical', 'absent'):
            path = self.publish(name=mode)
            if mode == 'writable':
                path.chmod(0o644)
            elif mode == 'symlink':
                original = path.with_name('original.json')
                path.rename(original)
                path.symlink_to(original)
            elif mode == 'noncanonical':
                path.chmod(0o644)
                path.write_text(json.dumps(self.bundle, indent=2))
                path.chmod(0o444)
            else:
                (path.parent / 'REUSE_AUDIT_SUCCESS').unlink()
            with self.subTest(mode=mode), self.assertRaises((ValueError, OSError)):
                driver._install_full_census(self.base / f'bad-{mode}', path)

    def test_root_binds_installed_bundle_and_cli(self):
        self.assertTrue(hasattr(driver, '_install_full_census'), 'full census import is missing')
        root, census, plan = self.install()
        identity = driver._read_installed(root, 'FORMAL_ROOT_IDENTITY.json')
        self.assertEqual(registry.FULL_MATRIX_PROFILE, identity['experiment_profile'])
        self.assertEqual(driver._sha256_bytes((root / 'FULL_MATRIX_REUSE.json').read_bytes()), identity['reuse_bundle_sha256'])
        self.assertEqual(driver._digest(self.bundle['admitted'][0]['source_sha256']), census['records'][0]['source_sha256'])
        args = driver._parser().parse_args(['census-full', '--root', str(root), '--reuse-bundle', '/bundle.json'])
        self.assertEqual('census-full', args.action)
        path = root / 'FULL_MATRIX_REUSE.json'
        changed = copy.deepcopy(self.bundle)
        changed['admitted'][0]['resource']['runtime_seconds'] += 1
        path.chmod(0o644)
        path.write_bytes(driver._canonical_json(changed) + b'\n')
        path.chmod(0o444)
        with self.assertRaises(ValueError):
            driver._load_installed_plan(root)

    def completed(self, root, plan):
        (root / 'records').mkdir()
        by_key = {driver.spec_key(s): s for s in registry.formal_specs()}
        for key in plan['missing_jobs']:
            spec = by_key[key]
            record = _completed_record(spec, float(spec.seed - 40), source_commit=driver._source_commit(), plan_sha256=driver._digest(plan))
            driver.install_json_exclusive(root / 'records' / (registry.safe_spec_name(spec) + '.json'), record)

    def test_combined_reports_have_exact_dimensions_statistics_order_and_provenance(self):
        self.assertTrue(hasattr(driver, '_install_full_census'), 'full census import is missing')
        import three_dataset_full_matrix_report as report
        root, census, plan = self.install()
        self.completed(root, plan)
        rows = report.combined_rows(root, plan, census, self.bundle)
        self.assertEqual(plan['formal_cells'], [r['spec_key'] for r in rows])
        self.assertEqual(['reused'] * 19 + ['new'] * 107, [r['origin'] for r in rows])
        self.assertEqual(['d' * 40] * 19, [r['origin_source_commit'] for r in rows[:19]])
        self.assertEqual([driver._source_commit()] * 107, [r['origin_source_commit'] for r in rows[19:]])
        outputs = report.render_full_matrix_tables(rows)
        parsed = {name: list(csv.DictReader(io.StringIO(data.decode()))) for name, data in outputs.items() if name.endswith('.csv')}
        self.assertEqual(126, len(parsed['FULL_MATRIX_PER_RUN.csv']))
        self.assertEqual(42, len(parsed['FULL_MATRIX_TABLE.csv']))
        self.assertEqual(14, len(parsed['FULL_MATRIX_MACRO.csv']))
        self.assertEqual(126, len(parsed['FULL_MATRIX_RESOURCE_PRIVACY.csv']))
        self.assertEqual(3.0, float(parsed['FULL_MATRIX_TABLE.csv'][0]['aa_final_mean']))
        self.assertEqual(statistics.stdev([2., 3., 4.]), float(parsed['FULL_MATRIX_TABLE.csv'][0]['aa_final_std']))
        self.assertEqual(list(registry.FULL_MATRIX_METHODS), [r['method'] for r in parsed['FULL_MATRIX_MACRO.csv']])
        self.assertEqual(3.0, float(parsed['FULL_MATRIX_MACRO.csv'][0]['aa_final_mean']))
        altered = copy.deepcopy(rows)
        for row in altered:
            offset = (0., 10., 50.)[registry.DATASETS.index(row['dataset'])]
            row['metrics'] = {k: [x + offset for x in value] if type(value) is list else value + offset
                              for k, value in row['metrics'].items()}
        macro = report.render_full_matrix_tables(altered)['FULL_MATRIX_MACRO.csv']
        self.assertEqual(23.0, float(next(csv.DictReader(io.StringIO(macro.decode())))['aa_final_mean']))
        self.assertEqual('FULL_MATRIX_AUDIT.json', list(outputs)[-1])
        (root / 'audit_queue').mkdir(mode=0o700)
        with mock.patch.object(driver, '_require_pipeline_drained', wraps=driver._require_pipeline_drained) as drained:
            hashes = driver.finalize_installed(root)
        drained.assert_called_once_with(root, ('formal',))
        self.assertEqual(set(outputs), set(hashes))
        audit = json.loads((root / 'tables/FULL_MATRIX_AUDIT.json').read_bytes())
        self.assertEqual(driver._digest(plan), audit['input_sha256']['plan'])
        for name, digest in hashes.items():
            self.assertEqual(digest, driver._sha256_bytes((root / 'tables' / name).read_bytes()))

    def test_corrected_method_resources_are_preserved_in_registry_order_and_audit(self):
        import three_dataset_full_matrix_report as report
        root, census, plan = self.install()
        self.completed(root, plan)
        rows = report.combined_rows(root, plan, census, self.bundle)
        baseline = report.render_full_matrix_tables(rows)
        corrected = copy.deepcopy(rows)
        sentinels = {
            'der_pp': {
                'added_parameters': 0, 'raw_examples_per_class': 11,
                'persistent_embeddings': 0,
                'replay_type': 'reservoir-raw-examples-and-logits',
                'privacy_label': 'persistent-raw-example-replay',
            },
            'er_ace': {
                'added_parameters': 0, 'raw_examples_per_class': 13,
                'persistent_embeddings': 0,
                'replay_type': 'reservoir-raw-examples',
                'privacy_label': 'persistent-raw-example-replay',
            },
            'target': {
                'added_parameters': 17, 'raw_examples_per_class': 0,
                'persistent_embeddings': 0, 'replay_type': 'synthetic-generator',
                'privacy_label': 'persistent-synthetic-generator-replay',
            },
            'adagauss': {
                'added_parameters': 19, 'raw_examples_per_class': 0,
                'persistent_embeddings': 23,
                'replay_type': 'class-gaussian-statistics',
                'privacy_label': 'persistent-derived-statistics-replay',
            },
            'proto_fedspace': {
                'added_parameters': 0, 'raw_examples_per_class': 0,
                'persistent_embeddings': 29,
                'replay_type': 'class-prototype-embeddings',
                'privacy_label': 'persistent-derived-embedding-replay',
            },
            'adaptive': {
                'added_parameters': 0, 'raw_examples_per_class': 31,
                'persistent_embeddings': 37,
                'replay_type': 'raw-examples-and-class-prototype-embeddings',
                'privacy_label': 'persistent-raw-and-derived-embedding-replay',
            },
        }
        selected = {}
        for row in corrected:
            if row['dataset'] == 'cifar100' and row['seed'] == 42 \
                    and row['method'] in sentinels:
                row['resource'].update(sentinels[row['method']])
                selected[row['spec_key']] = row

        expected_input = copy.deepcopy(corrected)
        expected_order = tuple(driver.spec_key(spec) for spec in registry.formal_specs())
        identity = ('spec_key', 'dataset', 'method', 'seed')
        provenance = ('origin', 'origin_root', 'origin_source_commit',
                      'origin_record_sha256', 'audit_status')
        resources = tuple(sorted(driver._RESOURCE_KEYS))
        expected_membership = [
            {name: row[name] for name in identity + provenance}
            for row in expected_input
        ]
        expected_digest = hashlib.sha256(json.dumps(
            expected_input, sort_keys=True, separators=(',', ':'),
            allow_nan=False).encode()).hexdigest()
        expected_selected = {}
        for key, source in copy.deepcopy(selected).items():
            expected = {name: str(source[name]) for name in identity + provenance}
            expected.update({
                name: (json.dumps(source['resource'][name], sort_keys=True,
                                  separators=(',', ':'), allow_nan=False)
                       if type(source['resource'][name]) is dict
                       else str(source['resource'][name]))
                for name in resources
            })
            expected_selected[key] = expected

        outputs = report.render_full_matrix_tables(corrected)
        self.assertEqual(expected_input, corrected, 'renderer mutated input rows')
        for name in ('FULL_MATRIX_PER_RUN.csv', 'FULL_MATRIX_TABLE.csv',
                     'FULL_MATRIX_MACRO.csv'):
            self.assertEqual(baseline[name], outputs[name])
        self.assertNotEqual(baseline['FULL_MATRIX_RESOURCE_PRIVACY.csv'],
                            outputs['FULL_MATRIX_RESOURCE_PRIVACY.csv'])

        resource_bytes = outputs['FULL_MATRIX_RESOURCE_PRIVACY.csv']
        reader = csv.DictReader(io.StringIO(resource_bytes.decode()))
        self.assertEqual(list(identity + resources + provenance), reader.fieldnames)
        resource_rows = list(reader)
        self.assertEqual(expected_order,
                         tuple(row['spec_key'] for row in resource_rows))
        actual_by_key = {row['spec_key']: row for row in resource_rows}
        self.assertEqual(6, len(expected_selected))
        for key, expected in expected_selected.items():
            with self.subTest(spec_key=key):
                self.assertEqual(expected, actual_by_key[key])

        audit = json.loads(outputs['FULL_MATRIX_AUDIT.json'])
        baseline_audit = json.loads(baseline['FULL_MATRIX_AUDIT.json'])
        self.assertEqual(expected_digest, audit['input_sha256']['rows'])
        self.assertEqual(expected_membership, audit['membership'])
        for name, data in outputs.items():
            if name.endswith('.csv'):
                self.assertEqual(hashlib.sha256(data).hexdigest(),
                                 audit['table_sha256'][name])
        for name in ('FULL_MATRIX_PER_RUN.csv', 'FULL_MATRIX_TABLE.csv',
                     'FULL_MATRIX_MACRO.csv'):
            self.assertEqual(baseline_audit['table_sha256'][name],
                             audit['table_sha256'][name])
        self.assertNotEqual(
            baseline_audit['table_sha256']['FULL_MATRIX_RESOURCE_PRIVACY.csv'],
            audit['table_sha256']['FULL_MATRIX_RESOURCE_PRIVACY.csv'])

    def test_corrected_resource_oracle_detects_renderer_input_mutation(self):
        import three_dataset_full_matrix_report as report
        original = report.render_full_matrix_tables
        calls = 0

        def mutate_corrected_row(rows):
            nonlocal calls
            calls += 1
            if calls == 2:
                row = next(row for row in rows
                           if row['spec_key'] == 'cifar100:der_pp:42')
                row['resource']['raw_examples_per_class'] = 999
            return original(rows)

        with mock.patch.object(report, 'render_full_matrix_tables',
                               side_effect=mutate_corrected_row):
            with self.assertRaisesRegex(AssertionError, 'renderer mutated input rows'):
                self.test_corrected_method_resources_are_preserved_in_registry_order_and_audit()
        self.assertEqual(2, calls)

    def test_resigned_source_map_mismatch_is_rejected(self):
        bundle = copy.deepcopy(self.bundle)
        bundle['admitted'][0]['source_sha256']['extra.py'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'source projection'):
            driver._install_full_census(self.base / 'bad-source', self.publish(bundle))

    def snapshot(self):
        return [(str(p.relative_to(self.base)), p.lstat().st_mode,
                 os.readlink(p) if p.is_symlink() else None if p.is_dir() else p.read_bytes())
                for p in sorted(self.base.rglob('*'))]

    def test_census_destination_protects_historical_and_bundle_roots_before_writes(self):
        import three_dataset_full_matrix_reconcile as reconcile
        import three_dataset_seed42_reconcile as legacy
        roots = [self.base / name for name in ('pilot-history', 'adaptive-history', 'formal-history', 'custom-history')]
        for root in roots:
            root.mkdir(mode=0o700)
            driver.install_json_exclusive(root / 'sentinel.json', {'untouched': True})
        bundle = copy.deepcopy(self.bundle)
        bundle['sources'][0]['root'] = str(roots[-1])
        for row in bundle['admitted']:
            row['origin_root'] = str(roots[-1])
            row['origin_record_path'] = str(roots[-1] / 'records' / Path(row['origin_record_path']).name)
        path = self.publish(bundle)
        before = self.snapshot()
        with mock.patch.object(legacy, 'PILOT_ROOT', str(roots[0])), \
                mock.patch.object(reconcile, 'ADAPTIVE_ROOT', str(roots[1])), \
                mock.patch.object(reconcile, 'FORMAL_ROOT', str(roots[2])):
            for protected in [*roots, path.parent]:
                for target in (protected, protected / 'new-authority'):
                    with self.subTest(target=target), mock.patch.object(driver, 'validate_formal_root', wraps=driver.validate_formal_root) as create:
                        with self.assertRaises((ValueError, OSError)):
                            driver._install_full_census(target, path)
                        create.assert_not_called()
                        self.assertEqual(before, self.snapshot())

    def test_census_existing_destination_requires_owned_private_empty_real_directory(self):
        path = self.publish()
        for case in ('nonempty', 'public', 'foreign-owner', 'symlink', 'symlink-parent'):
            target = self.base / case
            if case in ('symlink', 'symlink-parent'):
                real = self.base / (case + '-real')
                real.mkdir(mode=0o700)
                target.symlink_to(real, target_is_directory=True)
                if case == 'symlink-parent':
                    target = target / 'new-authority'
            else:
                target.mkdir(mode=0o700)
                if case == 'nonempty':
                    driver.install_json_exclusive(target / 'sentinel.json', {})
                if case == 'public':
                    target.chmod(0o755)
            before = self.snapshot()
            uid = os.getuid() + (1 if case == 'foreign-owner' else 0)
            with self.subTest(case=case), mock.patch.object(driver.os, 'getuid', return_value=uid):
                with self.assertRaises((ValueError, OSError)):
                    driver._install_full_census(target, path)
                self.assertEqual(before, self.snapshot())
        accepted = self.base / 'owned-empty'
        accepted.mkdir(mode=0o700)
        self.assertEqual(126, len(driver._install_full_census(accepted, path)['records']))

    def test_census_cli_rejects_raw_slash_aliases_before_bundle_access(self):
        import three_dataset_full_matrix_report as report
        for slashes in (2, 3):
            raw = '/' * slashes + str(self.base / 'alias').lstrip('/')
            before = self.snapshot()
            with self.subTest(raw=raw), mock.patch.object(report, 'load_reuse_bundle', side_effect=AssertionError('bundle access before root preflight')):
                with self.assertRaises(ValueError):
                    driver.main(['census-full', '--root', raw, '--reuse-bundle', '/unused.json'])
            self.assertEqual(before, self.snapshot())

    def test_finalization_installs_audit_last_and_rolls_back_owned_tables(self):
        root, census, plan = self.install(126)
        (root / 'audit_queue').mkdir(mode=0o700)
        installed = []
        original = driver._install_table_exclusive

        def fail_audit(parent, name, payload):
            installed.append(name)
            if name == 'FULL_MATRIX_AUDIT.json':
                raise OSError('generated install failure')
            return original(parent, name, payload)

        with mock.patch.object(driver, '_install_table_exclusive', side_effect=fail_audit):
            with self.assertRaisesRegex(OSError, 'generated install failure'):
                driver.finalize_installed(root)
        self.assertEqual('FULL_MATRIX_AUDIT.json', installed[-1])
        self.assertEqual(5, len(installed))
        self.assertEqual([], list((root / 'tables').iterdir()))

    def test_finalization_rejects_bundle_change_after_plan_validation(self):
        root, census, plan = self.install(126)
        (root / 'audit_queue').mkdir(mode=0o700)
        original = driver._load_installed_plan
        calls = 0

        def mutate_after_validation(*args, **kwargs):
            nonlocal calls
            result = original(*args, **kwargs)
            calls += 1
            if calls == 2:
                altered = copy.deepcopy(self.bundle)
                altered['admitted'][0]['resource']['runtime_seconds'] += 1.0
                path = root / 'FULL_MATRIX_REUSE.json'
                path.chmod(0o644)
                path.write_bytes(driver._canonical_json(altered) + b'\n')
                path.chmod(0o444)
            return result

        with mock.patch.object(driver, '_load_installed_plan', side_effect=mutate_after_validation):
            with self.assertRaisesRegex(ValueError, 'reuse authority'):
                driver.finalize_installed(root)
        self.assertFalse((root / 'tables').exists())

    def test_combined_rejects_missing_duplicate_overlap_and_render_membership(self):
        self.assertTrue(hasattr(driver, '_install_full_census'), 'full census import is missing')
        import three_dataset_full_matrix_report as report
        root, census, plan = self.install()
        self.completed(root, plan)
        rows = report.combined_rows(root, plan, census, self.bundle)
        for bad in (rows[:-1], rows + [rows[0]], [rows[0]] * 126):
            with self.assertRaises(ValueError):
                report.render_full_matrix_tables(bad)
        duplicate = copy.deepcopy(self.bundle)
        duplicate['admitted'].append(duplicate['admitted'][0])
        with self.assertRaises(ValueError):
            report.combined_rows(root, plan, census, duplicate)
        overlap = root / 'records' / (registry.safe_spec_name(registry.formal_specs()[0]) + '.json')
        driver.install_json_exclusive(overlap, {})
        with self.assertRaises(ValueError):
            report.combined_rows(root, plan, census, self.bundle)
        overlap.unlink()
        next((root / 'records').iterdir()).unlink()
        with self.assertRaises(ValueError):
            report.combined_rows(root, plan, census, self.bundle)

    def test_full_markers_are_profile_bound_and_reuse_does_not_block_drain(self):
        self.assertTrue(hasattr(driver, '_install_full_census'), 'full census import is missing')
        root, census, plan = self.install()
        self.completed(root, plan)
        self.assertTrue(driver._audit_drained(root, plan, 'formal', {}, None))
        for name in ('FULL_MATRIX_PHASE_SUCCESS', 'FULL_MATRIX_EXECUTION_SUCCESS'):
            self.assertIn(name, driver._MARKERS)
            with self.assertRaisesRegex(ValueError, 'audit queue'):
                driver.install_marker(root, name, {'kind': name.lower(), 'role': 'test', 'spec_key': '', 'exit_code': 0})
        for profile, names in ((registry.FULL_MATRIX_PROFILE, ('FORMAL_PHASE_SUCCESS', 'PILOT_PHASE_SUCCESS', 'RECOVERY_EXECUTION_SUCCESS')),
                               (registry.FORMAL_PROFILE, ('FULL_MATRIX_PHASE_SUCCESS', 'FULL_MATRIX_EXECUTION_SUCCESS'))):
            with mock.patch.dict(os.environ, {registry.PROFILE_ENV: profile}):
                for name in names:
                    with self.assertRaisesRegex(ValueError, 'profile'):
                        driver.install_marker(root, name, {'kind': name.lower(), 'role': 'test', 'spec_key': '', 'exit_code': 0})


if __name__ == '__main__':
    unittest.main()
