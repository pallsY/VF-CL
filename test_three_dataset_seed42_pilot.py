import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
REVIEWED_PYTHON = '/home/c3080/YangXiaoXiang/envs/vfcl/bin/python'


class Seed42PilotRegistryTest(unittest.TestCase):
    def test_wrapper_execs_generic_with_exact_profile_and_arguments(self):
        wrapper = ROOT / 'run_three_dataset_seed42_pilot.sh'
        self.assertTrue(wrapper.is_file(), 'dedicated pilot wrapper is missing')
        with tempfile.TemporaryDirectory(prefix='pilot_wrapper_test_') as temporary:
            path = Path(temporary)
            copied = path / wrapper.name
            shutil.copy2(wrapper, copied)
            generic = path / 'run_three_dataset_formal_comparison.sh'
            generic.write_text('#!/usr/bin/env bash\nexec ' + sys.executable + ''' -c '
import json, os, sys
print(json.dumps([os.getpid(), os.environ["VFCL_EXPERIMENT_PROFILE"], sys.argv[1:]]))
' "$@"
''')
            generic.chmod(0o755)
            environment = os.environ.copy()
            environment.pop('VFCL_EXPERIMENT_PROFILE', None)
            for args in (['/root with spaces'], ['--check', '/root with spaces'],
                         ['--smoke', '/root with spaces'], ['--unknown', '', 'tail']):
                with self.subTest(args=args):
                    process = subprocess.Popen(['bash', str(copied), *args],
                                               env=environment, stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE, text=True)
                    stdout, stderr = process.communicate(timeout=5)
                    self.assertEqual(0, process.returncode, stderr)
                    self.assertEqual([process.pid, 'seed42-pilot', args], json.loads(stdout))
            for profile in ('', 'formal', 'seed42-pilot', 'unknown'):
                with self.subTest(profile=profile):
                    completed = subprocess.run(['bash', str(copied), '--check', '/unused'],
                                               env={**environment, 'VFCL_EXPERIMENT_PROFILE': profile},
                                               capture_output=True, text=True, timeout=5)
                    self.assertEqual(64, completed.returncode)
                    self.assertEqual('', completed.stdout)
                    self.assertIn('external profile override is forbidden', completed.stderr)

    def _run_registry(self, profile=None, code=None):
        environment = os.environ.copy()
        environment['VFCL_PYTHON'] = REVIEWED_PYTHON
        if profile is None:
            environment.pop('VFCL_EXPERIMENT_PROFILE', None)
        else:
            environment['VFCL_EXPERIMENT_PROFILE'] = profile
        return subprocess.run(
            [sys.executable, '-B', '-c', code or '''
import json
import three_dataset_formal_registry as registry

specs = registry.formal_specs()
contracts = {
    registry.registered_spec_key(spec): registry.method_contract_for(spec)
    for spec in specs
}
print(json.dumps({
    'summary': (registry.experiment_profile(), *registry.profile_cardinality()),
    'keys': [registry.registered_spec_key(spec) for spec in specs],
    'contracts': contracts,
}, sort_keys=True))
'''],
            cwd=ROOT,
            capture_output=True,
            env=environment,
            text=True,
        )

    def _profile_data(self, profile=None):
        completed = self._run_registry(profile)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def test_profiles_are_isolated_in_fresh_interpreters(self):
        default = self._profile_data()
        pilot = self._profile_data('seed42-pilot')

        self.assertEqual(('formal', 81, 6), tuple(default['summary']))
        self.assertEqual(('seed42-pilot', 15, 0), tuple(pilot['summary']))
        self.assertEqual(
            {'finetune', 'lwf', 'er', 'afc', 'adaptive'},
            {key.split(':')[1] for key in pilot['keys']},
        )
        self.assertEqual([
            f'{dataset}:{method}:{seed}'
            for seed in (42,)
            for method in ('finetune', 'lwf', 'er', 'afc', 'adaptive')
            for dataset in ('cifar100', 'isolet', 'upmc_food101')
        ], pilot['keys'])
        contracts = pilot['contracts']
        self.assertEqual(300, contracts['cifar100:er:42']['er_per_class'])
        self.assertEqual(
            2.0, contracts['cifar100:afc:42']['afc_distill_weight'])
        self.assertEqual('afc', contracts['cifar100:afc:42']['cl_method'])
        self.assertEqual(
            'proto_evolve', contracts['cifar100:adaptive:42']['cl_method'])

    def test_unknown_profile_fails_during_import(self):
        completed = self._run_registry(
            'unrecognized-profile',
            'import three_dataset_formal_registry',
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('unsupported experiment profile', completed.stderr)

    def test_er_resource_measurements_use_checkpoint_counts_in_each_active_profile(self):
        for profile in (None, 'seed42-pilot'):
            with self.subTest(profile=profile):
                completed = self._run_registry(profile, '''
import json
from pathlib import Path
import tempfile
from unittest import mock
import torch
import three_dataset_formal_driver as driver
from test_three_dataset_formal_driver import _write_resource_checkpoint

with tempfile.TemporaryDirectory(prefix='pilot_resource_test_') as temporary:
    run = Path(temporary)
    (run / 'results.json').write_text(json.dumps({'comm_stats': []}))
    # Only hardware discovery is faked; no GPU or external process is used.
    with mock.patch.object(torch.cuda, 'is_available', return_value=True), \
         mock.patch.object(torch.cuda, 'device_count', return_value=1), \
         mock.patch.object(torch.cuda, 'get_device_name', return_value='test GPU'), \
         mock.patch.object(driver.subprocess, 'run',
                           return_value=mock.Mock(stdout='test-driver')):
        counts = {}
        for spec in driver.formal_specs():
            if spec.method == 'er' and spec.seed == 42:
                _write_resource_checkpoint(spec, run / 'checkpoints/formal_final.pt')
                measured = driver._resource_measurements(spec, run, 1.0, 0)
                assert measured['replay_type'] == 'raw-examples'
                assert measured['privacy_label'] == 'persistent-raw-example-replay'
                counts[spec.dataset] = measured['raw_examples_per_class']
print(json.dumps(counts))
''')
                self.assertEqual(0, completed.returncode, completed.stderr)
                self.assertEqual({dataset: 2 for dataset in (
                    'cifar100', 'isolet', 'upmc_food101')},
                    json.loads(completed.stdout))

    def test_pilot_success_markers_reject_roots_without_drained_authority(self):
        self._assert_pilot_driver('''
from test_three_dataset_formal_driver import FormalDriverTests
fixture = FormalDriverTests()
fixture.setUp()
try:
    no_queue, _ = fixture._installed(name='no-queue')
    incomplete, plan = fixture._installed(name='incomplete')
    (incomplete / 'records').mkdir()
    for spec in driver.formal_specs()[:-1]:
        driver.install_json_exclusive(
            incomplete / 'records' / f'{driver.safe_spec_name(spec)}.json',
            _completed_record(spec, .5, source_commit=driver._source_commit(),
                              plan_sha256=driver._digest(plan)))
    assert not driver.audit_phase_ready(incomplete, 'formal')
    empty = fixture.base / 'empty'
    empty.mkdir(mode=0o700)
    foreign = fixture.base / 'foreign'
    foreign.mkdir(mode=0o700)
    (foreign / 'foreign.json').write_text('{}')
    smoke = fixture.base / 'smoke'
    smoke.mkdir(mode=0o700)
    driver.plan_smoke(smoke)
    for root in (no_queue, incomplete, empty, foreign, smoke):
        for name in ('PILOT_PHASE_SUCCESS', 'PILOT_EXECUTION_SUCCESS'):
            payload = {'kind': name.lower(), 'role': 'launcher', 'spec_key': '', 'exit_code': 0}
            try:
                driver.install_marker(root, name, payload)
            except (ValueError, FileNotFoundError):
                pass
            else:
                raise AssertionError('accepted pilot success at ' + root.name)
            assert not (root / name).exists()
    assert not (no_queue / 'audit_queue').exists()
finally:
    fixture.doCleanups()
''')

    def test_pilot_success_markers_reject_wrong_profile_authority_with_and_without_queue(self):
        completed = self._run_registry(None, '''
import os
import subprocess
import sys
from test_three_dataset_formal_driver import FormalDriverTests
fixture = FormalDriverTests()
fixture.setUp()
try:
    root, _ = fixture._installed()
    code = """
from pathlib import Path
import sys
import three_dataset_formal_driver as driver
root = Path(sys.argv[1])
for name in ('PILOT_PHASE_SUCCESS', 'PILOT_EXECUTION_SUCCESS'):
    payload = {'kind': name.lower(), 'role': 'launcher', 'spec_key': '', 'exit_code': 0}
    try:
        driver.install_marker(root, name, payload)
    except ValueError:
        pass
    else:
        raise AssertionError('accepted wrong-profile authority')
    assert not (root / name).exists()
"""
    for queued in (False, True):
        if queued:
            (root / 'audit_queue').mkdir(mode=0o700)
        result = subprocess.run([sys.executable, '-B', '-c', code, str(root)],
                                env={**os.environ, 'VFCL_EXPERIMENT_PROFILE': 'seed42-pilot'},
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
finally:
    fixture.doCleanups()
''')
        self.assertEqual(0, completed.returncode, completed.stderr)

    def test_default_formal_success_markers_preserve_legacy_no_queue(self):
        completed = self._run_registry(None, '''
from pathlib import Path
import tempfile
import three_dataset_formal_driver as driver
with tempfile.TemporaryDirectory(prefix='formal_legacy_marker_test_') as temporary:
    root = Path(temporary)
    for name in ('FORMAL_PHASE_SUCCESS', 'EXPLANATION_PHASE_SUCCESS',
                 'FORMAL_EXECUTION_SUCCESS'):
        payload = {'kind': name.lower(), 'role': 'launcher', 'spec_key': '', 'exit_code': 0}
        assert driver.install_marker(root, name, payload) == payload
    assert not (root / 'audit_queue').exists()
''')
        self.assertEqual(0, completed.returncode, completed.stderr)

    def test_formal_success_markers_reject_pilot_authority_with_and_without_queue(self):
        for queued in (False, True):
            for marker in ('FORMAL_PHASE_SUCCESS', 'EXPLANATION_PHASE_SUCCESS',
                           'FORMAL_EXECUTION_SUCCESS'):
                with self.subTest(queued=queued, marker=marker):
                    completed = self._run_registry('seed42-pilot', '''
import os
import subprocess
import sys
from test_three_dataset_formal_driver import FormalDriverTests
fixture = FormalDriverTests()
fixture.setUp()
try:
    root, _ = fixture._installed()
    queued, marker = ''' + repr((queued, marker)) + '''
    if queued:
        (root / 'audit_queue').mkdir(mode=0o700)
    code = """
from pathlib import Path
import sys
import three_dataset_formal_driver as driver
root, name = Path(sys.argv[1]), sys.argv[2]
payload = {'kind': name.lower(), 'role': 'launcher', 'spec_key': '', 'exit_code': 0}
try:
    driver.install_marker(root, name, payload)
except ValueError:
    pass
else:
    raise AssertionError('accepted formal success in pilot authority')
assert not (root / name).exists()
"""
    result = subprocess.run([sys.executable, '-B', '-c', code, str(root), marker],
                            env={**os.environ, 'VFCL_EXPERIMENT_PROFILE': 'formal'},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (root / 'audit_queue').exists() == queued
finally:
    fixture.doCleanups()
''')
                    self.assertEqual(0, completed.returncode, completed.stderr)

    def test_formal_no_queue_authority_requires_completed_records_without_mutation(self):
        completed = self._run_registry(None, '''
import three_dataset_formal_driver as driver
from test_three_dataset_formal_driver import FormalDriverTests, _completed_record
fixture = FormalDriverTests()
fixture.setUp()
try:
    root, plan = fixture._installed()
    (root / 'records').mkdir()
    missing = driver.formal_specs()[-1]
    def install(spec):
        driver.install_json_exclusive(
            root / 'records' / f'{driver.safe_spec_name(spec)}.json',
            _completed_record(spec, .5, source_commit=driver._source_commit(),
                              plan_sha256=driver._digest(plan)))
    for spec in (*driver.formal_specs(), *driver.explanation_specs()):
        if spec != missing:
            install(spec)
    names = ('FORMAL_PHASE_SUCCESS', 'EXPLANATION_PHASE_SUCCESS',
             'FORMAL_EXECUTION_SUCCESS')
    for name in names:
        payload = {'kind': name.lower(), 'role': 'launcher', 'spec_key': '', 'exit_code': 0}
        try:
            driver.install_marker(root, name, payload)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted incomplete formal authority')
        assert not (root / name).exists()
        assert not (root / 'audit_queue').exists()
    install(missing)
    for name in names:
        payload = {'kind': name.lower(), 'role': 'launcher', 'spec_key': '', 'exit_code': 0}
        assert driver.install_marker(root, name, payload) == payload
        assert not (root / 'audit_queue').exists()
finally:
    fixture.doCleanups()
''')
        self.assertEqual(0, completed.returncode, completed.stderr)

    def test_formal_success_markers_reject_partial_authority_without_queue(self):
        completed = self._run_registry(None, '''
from pathlib import Path
import tempfile
import three_dataset_formal_driver as driver
with tempfile.TemporaryDirectory(prefix='partial_authority_marker_test_') as temporary:
    for authority_name in driver._FORMAL_OUTPUT_NAMES:
        root = Path(temporary) / authority_name
        root.mkdir(mode=0o700)
        driver.install_json_exclusive(root / authority_name, {})
        for name in ('FORMAL_PHASE_SUCCESS', 'EXPLANATION_PHASE_SUCCESS',
                     'FORMAL_EXECUTION_SUCCESS'):
            payload = {'kind': name.lower(), 'role': 'launcher', 'spec_key': '', 'exit_code': 0}
            try:
                driver.install_marker(root, name, payload)
            except (ValueError, FileNotFoundError):
                pass
            else:
                raise AssertionError('accepted partial authority ' + authority_name)
            assert not (root / name).exists()
            assert not (root / 'audit_queue').exists()
''')
        self.assertEqual(0, completed.returncode, completed.stderr)

    def _assert_pilot_driver(self, code):
        completed = self._run_registry('seed42-pilot', '''
import copy
import csv
import hashlib
import io
from pathlib import Path
import tempfile
from unittest import mock
import three_dataset_formal_driver as driver
from test_three_dataset_formal_driver import _completed_record

records = [_completed_record(spec, (index + 1) / 32)
           for index, spec in enumerate(driver.formal_specs())]
''' + code)
        self.assertEqual(0, completed.returncode, completed.stderr)

    def test_pilot_census_and_plan_cover_exactly_fifteen_cells(self):
        self._assert_pilot_driver('''
census = driver.build_census({})
assert len(census['records']) == 15
plan = driver.build_plan(census)
assert len(plan['formal_cells']) == 15
assert plan['missing_jobs'] == plan['formal_cells']
assert plan['explanation_cells'] == []
driver._validate_plan_shape(plan, census)
for field in ('formal_cells', 'missing_jobs', 'explanation_cells'):
    malformed = copy.deepcopy(plan)
    malformed[field].append(plan['formal_cells'][0])
    try:
        driver._validate_plan_shape(malformed, census)
    except ValueError:
        pass
    else:
        raise AssertionError('accepted malformed ' + field)
''')

    def test_pilot_summary_preserves_exact_values_and_validates_records(self):
        self._assert_pilot_driver('''
summary = driver.summarize_records(list(reversed(records)))
assert summary['kind'] == 'seed42_pilot_completed_summary'
assert summary['per_run_records'] == records
assert summary['pilot_rows'] == [{
    'dataset': record['dataset'], 'method': record['method'],
    'seed': record['seed'], 'aa_final': record['metrics']['aa_final'],
    'bwt': record['metrics']['bwt'],
    'taskil_final': record['metrics']['taskil_final'],
} for record in records]
assert len(summary['resource_rows']) == 15
assert summary['mechanism_rows'] == []
assert summary['formal_rows'] == []
variants = [records[:-1], records + [records[0]],
            records[:-1] + [records[0]]]
for field, value in (('seed', 43), ('registry_sha256', '0' * 64),
                     ('source_commit', '0' * 40), ('plan_sha256', '0' * 64)):
    changed = copy.deepcopy(records)
    changed[0][field] = value
    changed[0]['record_sha256'] = driver._digest({
        key: value for key, value in changed[0].items()
        if key != 'record_sha256'})
    variants.append(changed)
for variant in variants:
    try:
        driver.summarize_records(variant)
    except ValueError:
        pass
    else:
        raise AssertionError('accepted invalid pilot records')
''')

    def test_pilot_tables_have_single_seed_header_and_exclusive_install(self):
        self._assert_pilot_driver('''
tables = driver.render_tables(records)
assert tables == driver.render_tables(list(reversed(records)))
assert set(tables) == {
    'FORMAL_PER_RUN.csv', 'PILOT_TABLE.csv', 'RESOURCE_PRIVACY_TABLE.csv'}
rows = list(csv.reader(io.StringIO(tables['PILOT_TABLE.csv'].decode())))
assert rows[0] == ['dataset', 'method', 'seed', 'aa_final', 'bwt', 'taskil_final']
assert len(rows) == 16
for row, record in zip(rows[1:], records):
    assert row[:3] == [record['dataset'], record['method'], '42']
    assert row[3:] == [format(record['metrics'][key], '.6f')
                      for key in ('aa_final', 'bwt', 'taskil_final')]
with tempfile.TemporaryDirectory(prefix='pilot_tables_test_') as temporary:
    destination = Path(temporary) / 'tables'
    digests = driver.install_tables(records, destination)
    assert digests == {name: hashlib.sha256(payload).hexdigest()
                       for name, payload in tables.items()}
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == tables
    try:
        driver.install_tables(records, destination)
    except FileExistsError:
        pass
    else:
        raise AssertionError('overwrote installed tables')
''')

    def test_pilot_finalization_installs_only_the_three_pilot_tables(self):
        self._assert_pilot_driver('''
by_key = {record['spec_key']: record for record in records}
with tempfile.TemporaryDirectory(prefix='pilot_finalize_test_') as temporary:
    root = Path(temporary)
    with mock.patch.object(driver, '_require_pipeline_drained'), \
         mock.patch.object(driver, '_load_installed_plan', return_value=({}, {})), \
         mock.patch.object(driver, '_installed_completed_record',
                           side_effect=lambda root, spec, plan: by_key[driver.spec_key(spec)]):
        digests = driver.finalize_installed(root)
    assert set(digests) == {
        'FORMAL_PER_RUN.csv', 'PILOT_TABLE.csv', 'RESOURCE_PRIVACY_TABLE.csv'}
    assert {path.name for path in (root / 'tables').iterdir()} == set(digests)
''')

    def test_pilot_success_markers_are_immutable_and_require_only_formal_drain(self):
        self._assert_pilot_driver('''
from test_three_dataset_formal_driver import FormalDriverTests
fixture = FormalDriverTests()
fixture.setUp()
try:
    root, plan = fixture._installed()
    (root / 'records').mkdir()
    for spec in driver.formal_specs():
        driver.install_json_exclusive(
            root / 'records' / f'{driver.safe_spec_name(spec)}.json',
            _completed_record(spec, .5, source_commit=driver._source_commit(),
                              plan_sha256=driver._digest(plan)))
    assert plan['explanation_cells'] == []
    assert driver.audit_phase_ready(root, 'formal')
    for name in ('PILOT_PHASE_SUCCESS', 'PILOT_EXECUTION_SUCCESS'):
        payload = {'kind': name.lower(), 'role': 'launcher', 'spec_key': '', 'exit_code': 0}
        assert driver.install_marker(root, name, payload) == payload
        assert (root / name).stat().st_mode & 0o777 == 0o444
        try:
            driver.install_marker(root, name, payload)
        except FileExistsError:
            pass
        else:
            raise AssertionError('overwrote pilot marker')
finally:
    fixture.doCleanups()
''')

    def test_pilot_success_markers_reject_pending_and_active_audits(self):
        self._assert_pilot_driver('''
from test_three_dataset_formal_driver import FormalDriverTests
fixture = FormalDriverTests()
fixture.setUp()
try:
    root, plan = fixture._installed()
    key, producer, _ = fixture._pipeline_run(root, plan)
    driver.queue_audit(root, key, 0, producer)
    driver.release_gpu(root, 0, producer)
    auditor = fixture._owner(root, '', role='auditor')
    for active in (False, True):
        if active:
            driver.next_audit(root, 'formal', auditor)
            fixture._pipeline_record(root, plan, key)
            for spec in driver.formal_specs():
                if driver.spec_key(spec) != key:
                    driver.install_json_exclusive(
                        root / 'records' / f'{driver.safe_spec_name(spec)}.json',
                        _completed_record(spec, .5, source_commit=driver._source_commit(),
                                          plan_sha256=driver._digest(plan)))
        for name in ('PILOT_PHASE_SUCCESS', 'PILOT_EXECUTION_SUCCESS'):
            payload = {'kind': name.lower(), 'role': 'launcher', 'spec_key': '', 'exit_code': 0}
            try:
                driver.install_marker(root, name, payload)
            except ValueError:
                pass
            else:
                raise AssertionError('accepted undrained pilot success')
            assert not (root / name).exists()
    driver.complete_audit(root, key, auditor)
    for name in ('PILOT_PHASE_SUCCESS', 'PILOT_EXECUTION_SUCCESS'):
        payload = {'kind': name.lower(), 'role': 'launcher', 'spec_key': '', 'exit_code': 0}
        assert driver.install_marker(root, name, payload) == payload
finally:
    fixture.doCleanups()
''')

    def test_pilot_success_marker_namespace_on_drained_root(self):
        self._assert_pilot_driver('''
from unittest import TestResult
from test_three_dataset_formal_driver import FormalDriverTests
result = TestResult()
FormalDriverTests('test_success_markers_are_profile_bound_on_drained_root').run(result)
assert result.wasSuccessful(), result.failures + result.errors
''')
