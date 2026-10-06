"""Isolated contract tests for the three-cell adaptive recovery profile."""
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile
import unittest


PILOT_ROOT = Path(
    '/home/c3080/YangXiaoXiang/VF-CL/results/'
    'three_dataset_seed42_pilot_20260907_065121')
PILOT_SOURCE_COMMIT = 'f79062f20c54bb43849ecfc43682d643be23afe6'
RECOVERY_PROFILE = 'seed42-adaptive-recovery'
REVIEWED_PYTHON = '/home/c3080/YangXiaoXiang/envs/vfcl/bin/python'


class Seed42AdaptiveRecoveryTests(unittest.TestCase):
    def test_wrapper_execs_exact_profile_and_preserves_arguments(self):
        wrapper = Path(__file__).parent / 'run_three_dataset_adaptive_recovery.sh'
        self.assertTrue(wrapper.is_file(), 'dedicated recovery wrapper is missing')
        with tempfile.TemporaryDirectory(prefix='recovery_wrapper_') as temporary:
            copied = Path(temporary) / wrapper.name
            shutil.copy2(wrapper, copied)
            generic = copied.with_name('run_three_dataset_formal_comparison.sh')
            generic.write_text('#!/usr/bin/env bash\nexec ' + sys.executable + ''' -c '
import json, os, sys
print(json.dumps([os.getpid(), os.environ["VFCL_EXPERIMENT_PROFILE"], sys.argv[1:]]))
' "$@"
''')
            generic.chmod(0o755)
            env = os.environ.copy()
            env.pop('VFCL_EXPERIMENT_PROFILE', None)
            for args in (['/root with spaces'], ['--check', '/root with spaces'],
                         ['--smoke', '/root with spaces'], ['--unknown', '', 'tail']):
                with self.subTest(args=args):
                    process = subprocess.Popen(['bash', str(copied), *args], env=env,
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                               text=True)
                    stdout, stderr = process.communicate(timeout=5)
                    self.assertEqual(0, process.returncode, stderr)
                    self.assertEqual([process.pid, RECOVERY_PROFILE, args], json.loads(stdout))
            for profile in ('', 'formal', 'seed42-pilot', RECOVERY_PROFILE, 'unknown'):
                with self.subTest(profile=profile):
                    result = subprocess.run(['bash', str(copied), '--check', '/unused'],
                                            env={**env, 'VFCL_EXPERIMENT_PROFILE': profile},
                                            capture_output=True, text=True, timeout=5)
                    self.assertEqual(64, result.returncode)
                    self.assertEqual('', result.stdout)
                    self.assertIn('external profile override is forbidden', result.stderr)

    def test_profile_capacity_and_recovery_atomic_admission(self):
        for profile, expected in (('formal', 3), ('seed42-pilot', 3), (RECOVERY_PROFILE, 1)):
            result = self._run(profile, '''
import three_dataset_formal_driver as driver
assert hasattr(driver, 'pipeline_inflight_limit'), 'profile capacity policy is missing'
assert driver.pipeline_inflight_limit() == ''' + str(expected))
            self.assertEqual(0, result.returncode, result.stderr)
        result = self._run(RECOVERY_PROFILE, '''
import unittest
from test_three_dataset_formal_driver import FormalDriverTests
suite = unittest.TestSuite(FormalDriverTests(name) for name in (
    'test_recovery_atomic_claim_capacity_through_record_admission',
    'test_recovery_pending_audit_retains_slot_after_record_install',
    'test_recovery_queue_admission_counts_distinct_inflight_keys'))
assert unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
''')
        self.assertEqual(0, result.returncode, result.stderr)

    def _run(self, profile, code):
        environment = {**os.environ, 'VFCL_PYTHON': REVIEWED_PYTHON}
        if profile is None:
            environment.pop('VFCL_EXPERIMENT_PROFILE', None)
        else:
            environment['VFCL_EXPERIMENT_PROFILE'] = profile
        return subprocess.run(
            [sys.executable, '-B', '-c', code], cwd=Path(__file__).parent,
            env=environment, capture_output=True, text=True,
        )

    def test_recovery_generated_evidence_reaches_real_completed_record_admission(self):
        completed = self._run(RECOVERY_PROFILE, '''
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from unittest import mock

import three_dataset_formal_driver as driver
import three_dataset_formal_registry as formal_registry
from test_three_dataset_formal_audit import PRODUCER_SOURCES, RealProducerFixture

fixture = RealProducerFixture('adaptive')
try:
    controls = fixture.install_completed_controls()
    root = fixture.formal_root
    key = driver.spec_key(fixture.spec)
    producer = controls['owner']
    auditor = {**producer, 'job': '', 'worker_role': 'formal-auditor'}
    assert driver.claim_gpu(root, 0, producer)

    # The generated producer runs on CPU; only queue command reconstruction
    # adapts the deployment device. All admission validators remain real.
    def cpu_command(key, run):
        return formal_registry.command_for(
            driver.spec_for_key(key), 'cpu', str(run.parent), smoke=False)

    with ExitStack() as stack:
        stack.enter_context(mock.patch(
            'three_dataset_formal_audit._SOURCE_INVENTORY', PRODUCER_SOURCES))
        stack.enter_context(mock.patch(
            'three_dataset_formal_audit._AUTHORITATIVE_DATA',
            {fixture.spec.dataset: fixture.authoritative_data()}))
        stack.enter_context(mock.patch(
            'three_dataset_formal_audit._AUTHORITATIVE_MANIFEST',
            {fixture.spec.dataset: fixture.authoritative_manifest()}))
        stack.enter_context(mock.patch.object(
            formal_registry, '_deployment_paths', return_value=(
                fixture.root, fixture.source, Path(sys.executable))))
        with mock.patch.object(
                driver, 'command_for_run', side_effect=cpu_command) as command:
            handoff = driver.queue_audit(root, key, 0, producer)
        command.assert_called_once_with(key, fixture.run)
        driver.release_gpu(root, 0, producer)
        with mock.patch.object(
                driver, 'command_for_run', side_effect=cpu_command) as command:
            selected = driver.next_audit(root, 'formal', auditor)
        command.assert_called_once_with(key, fixture.run)
        assert selected == {name: handoff[name] for name in (
            'spec_key', 'run_dir', 'physical_gpu', 'seed')}

        record = driver.install_completed_record(
            fixture.formal_root, driver.spec_key(fixture.spec), fixture.run)
        record_path = root / 'records' / f'{driver.safe_spec_name(fixture.spec)}.json'
        assert json.loads(record_path.read_text()) == record
        assert list(record_path.parent.iterdir()) == [record_path]
        assert record['experiment_profile'] == driver.RECOVERY_PROFILE
        with mock.patch.object(
                driver, 'command_for_run', side_effect=cpu_command) as command:
            driver.complete_audit(root, key, auditor)
        command.assert_called_once_with(key, fixture.run)
        assert list((root / 'audit_queue').iterdir()) == []
        print('ADMISSION_EVIDENCE ' + json.dumps({
            'spec_key': key, 'experiment_profile': record['experiment_profile'],
            'record_sha256': record['record_sha256'],
            'record_count': 1, 'record_equals_installed': True,
            'selected_handoff': selected, 'audit_queue_empty': True,
        }, sort_keys=True))
finally:
    fixture.close()
''')
        self.assertEqual(0, completed.returncode, completed.stderr)
        print(next(line for line in completed.stdout.splitlines()
                   if line.startswith('ADMISSION_EVIDENCE ')))

    def _profile_data(self, profile):
        completed = self._run(profile, '''
import json
from pathlib import Path
import three_dataset_formal_registry as registry

pilot_root = Path(''' + repr(str(PILOT_ROOT)) + ''')
specs = registry.formal_specs()
rows = []
for spec in specs:
    if spec.method != 'adaptive' or spec.seed != 42:
        continue
    key = registry.registered_spec_key(spec)
    command = registry.command_for(spec, 'cuda:0', '/recovery/runs')
    pilot = json.loads((pilot_root / 'runs' /
        (registry.safe_spec_name(spec) + '/FORMAL_JOB_SPEC.json')).read_text())
    pilot_command = tuple(pilot['command'])
    current_parsed = registry.parsed_protocol(command, spec)
    pilot_parsed = registry.parsed_protocol(pilot_command, spec)
    method_names = set(registry.method_contract_for(spec))
    rows.append({
        'key': key,
        'source_commit': pilot['source_commit'],
        'current_command': list(command),
        'pilot_command': list(pilot_command),
        'current_parsed': current_parsed,
        'pilot_parsed': pilot_parsed,
        'method_contract': registry.method_contract_for(spec),
        'dataset_contract': {
            key: value for key, value in current_parsed['base_options'].items()
            if key not in method_names
        },
        'pilot_dataset_contract': {
            key: value for key, value in pilot_parsed['base_options'].items()
            if key not in method_names
        },
        'seed': spec.seed,
        'option_schema': registry.OPTION_SCHEMA,
        'current_options': sorted(registry._option_map(command, 3)),
        'pilot_options': sorted(registry._option_map(pilot_command, 3)),
    })
print(json.dumps({
    'summary': [registry.experiment_profile(), *registry.profile_cardinality()],
    'keys': [registry.registered_spec_key(spec) for spec in specs],
    'explanations': len(registry.explanation_specs()),
    'rows': rows,
}, sort_keys=True))
''')
        self.assertEqual(0, completed.returncode, completed.stderr)
        return json.loads(completed.stdout)

    @staticmethod
    def _normalized_command(command):
        command = list(command)
        command[command.index('--results_dir') + 1] = '<RESULTS_DIR>'
        return command

    def test_profiles_are_subprocess_isolated_and_recovery_matches_pilot_contract(self):
        default = self._profile_data(None)
        pilot = self._profile_data('seed42-pilot')
        recovery = self._profile_data(RECOVERY_PROFILE)

        self.assertEqual(('formal', 81, 6), tuple(default['summary']))
        self.assertEqual(('seed42-pilot', 15, 0), tuple(pilot['summary']))
        self.assertEqual(
            ('seed42-adaptive-recovery', 3, 0),
            tuple(recovery['summary']))
        self.assertEqual([
            'cifar100:adaptive:42',
            'isolet:adaptive:42',
            'upmc_food101:adaptive:42',
        ], recovery['keys'])
        self.assertEqual(0, recovery['explanations'])
        for row in recovery['rows']:
            with self.subTest(key=row['key']):
                self.assertEqual(PILOT_SOURCE_COMMIT, row['source_commit'])
                self.assertEqual(
                    self._normalized_command(row['pilot_command']),
                    self._normalized_command(row['current_command']))
                self.assertEqual(row['pilot_parsed'], row['current_parsed'])
                self.assertEqual(
                    row['pilot_dataset_contract'], row['dataset_contract'])
                self.assertEqual(42, row['seed'])
                self.assertEqual(row['pilot_options'], row['current_options'])
                self.assertTrue(
                    set(row['current_options']).issubset(row['option_schema']))

    def test_recovery_authority_reporting_and_cross_profile_rejection(self):
        completed = self._run(RECOVERY_PROFILE, '''
import copy
import csv
import io
import json
from pathlib import Path
import tempfile

import three_dataset_formal_driver as driver
from test_three_dataset_formal_driver import _completed_record

profile = driver.experiment_profile()
census = driver.build_census({})
plan = driver.build_plan(census)
assert census['experiment_profile'] == profile
assert plan['experiment_profile'] == profile
assert driver._registry_payload()['experiment_profile'] == profile

records = [_completed_record(spec, (index + 1) / 10)
           for index, spec in enumerate(driver.formal_specs())]
summary = driver.summarize_records(records)
assert summary['experiment_profile'] == profile
assert summary['kind'] == 'seed42_adaptive_recovery_completed_summary'
assert summary['formal_rows'] == []
assert summary['mechanism_rows'] == []
assert len(summary['recovery_rows']) == 3

foreign = copy.deepcopy(records)
foreign[0]['experiment_profile'] = driver.PILOT_PROFILE
foreign[0]['record_sha256'] = driver._digest({
    key: value for key, value in foreign[0].items()
    if key != 'record_sha256'})
try:
    driver.summarize_records(foreign)
except ValueError as error:
    assert 'profile' in str(error)
else:
    raise AssertionError('accepted a cross-profile completed record')

tables = driver.render_tables(records)
assert set(tables) == {
    'FORMAL_PER_RUN.csv', 'RECOVERY_TABLE.csv',
    'RESOURCE_PRIVACY_TABLE.csv'}
rows = list(csv.reader(io.StringIO(tables['RECOVERY_TABLE.csv'].decode())))
assert rows[0] == ['dataset', 'method', 'seed',
                   'aa_final', 'bwt', 'taskil_final']
assert [row[:3] for row in rows[1:]] == [
    ['cifar100', 'adaptive', '42'],
    ['isolet', 'adaptive', '42'],
    ['upmc_food101', 'adaptive', '42'],
]
assert all(len(row) == 6 for row in rows)

with tempfile.TemporaryDirectory(prefix='recovery_authority_') as temporary:
    root = driver.validate_formal_root(Path(temporary) / 'root')
    driver.install_json_exclusive(
        root / 'FORMAL_REGISTRY.json', driver._registry_payload())
    driver.install_json_exclusive(root / 'COMPATIBILITY_CENSUS.json', census)
    driver.install_json_exclusive(root / 'FORMAL_PLAN.json', plan)
    driver.install_json_exclusive(
        root / 'MISSING_JOBS.json', driver._missing_jobs_payload(plan, census))
    (root / 'claims').mkdir(mode=0o700)
    identity = driver.install_formal_root_identity(root, plan)
    frozen = json.loads((root / 'FORMAL_ROOT_IDENTITY.json').read_text())
    assert frozen['experiment_profile'] == profile
    assert identity == driver._root_identity(root)
print(json.dumps({'table': tables['RECOVERY_TABLE.csv'].decode()}))
''')
        self.assertEqual(0, completed.returncode, completed.stderr)
        table = json.loads(completed.stdout)['table']
        rows = list(csv.reader(io.StringIO(table)))
        self.assertEqual(4, len(rows))

    def test_recovery_markers_require_drained_three_record_authority(self):
        completed = self._run(RECOVERY_PROFILE, '''
from pathlib import Path
import tempfile

import three_dataset_formal_driver as driver
from test_three_dataset_formal_driver import FormalDriverTests, _completed_record

fixture = FormalDriverTests('runTest')
fixture.setUp()
try:
    root, plan = fixture._installed()
    records = root / 'records'
    records.mkdir()
    commit = driver._source_commit()
    for spec in driver.formal_specs():
        driver.install_json_exclusive(
            records / f'{driver.safe_spec_name(spec)}.json',
            _completed_record(spec, .5, source_commit=commit,
                              plan_sha256=driver._digest(plan)))
    assert len(driver.formal_specs()) == 3
    assert driver.explanation_specs() == ()
    assert driver.audit_phase_ready(root, 'formal')
    digests = driver.finalize_installed(root)
    assert 'RECOVERY_TABLE.csv' in digests

    recovery_names = ('RECOVERY_PHASE_SUCCESS', 'RECOVERY_EXECUTION_SUCCESS')
    rejected = ('FORMAL_PHASE_SUCCESS', 'EXPLANATION_PHASE_SUCCESS',
                'FORMAL_EXECUTION_SUCCESS',
                'PILOT_PHASE_SUCCESS', 'PILOT_EXECUTION_SUCCESS')
    for name in rejected:
        payload = {'kind': name.lower(), 'role': 'launcher',
                   'spec_key': '', 'exit_code': 0}
        try:
            driver.install_marker(root, name, payload)
        except ValueError as error:
            assert 'profile' in str(error)
        else:
            raise AssertionError('accepted foreign success marker ' + name)
        assert not (root / name).exists()
    for name in recovery_names:
        payload = {'kind': name.lower(), 'role': 'launcher',
                   'spec_key': '', 'exit_code': 0}
        assert driver.install_marker(root, name, payload) == payload
        assert (root / name).stat().st_mode & 0o777 == 0o444
finally:
    fixture.temporary.cleanup()
''')
        self.assertEqual(0, completed.returncode, completed.stderr)

    def test_preexisting_foreign_success_markers_poison_every_profile_authority(self):
        code = '''
from pathlib import Path

import three_dataset_formal_driver as driver
from test_three_dataset_formal_driver import FormalDriverTests, _completed_record

fixture = FormalDriverTests('runTest')
fixture.setUp()
try:
    active = driver.experiment_profile()
    allowed = {name for name, profile in driver._SUCCESS_MARKER_PROFILES.items()
               if profile == active}
    foreign = sorted(set(driver._SUCCESS_MARKER_PROFILES) - allowed)
    own_execution = {
        driver.FORMAL_PROFILE: 'FORMAL_EXECUTION_SUCCESS',
        driver.PILOT_PROFILE: 'PILOT_EXECUTION_SUCCESS',
        driver.RECOVERY_PROFILE: 'RECOVERY_EXECUTION_SUCCESS',
    }[active]
    for index, foreign_name in enumerate(foreign):
        root, plan = fixture._installed(name=f'{active}-{index}')
        records = root / 'records'
        records.mkdir()
        commit = driver._source_commit()
        for spec in (*driver.formal_specs(), *driver.explanation_specs()):
            driver.install_json_exclusive(
                records / f'{driver.safe_spec_name(spec)}.json',
                _completed_record(spec, .5, source_commit=commit,
                                  plan_sha256=driver._digest(plan)))
        payload = {
            'kind': foreign_name.lower(), 'role': 'foreign-launcher',
            'spec_key': '', 'exit_code': 0,
        }
        driver.install_json_exclusive(root / foreign_name, payload)

        for action in (
                lambda: driver.audit_phase_ready(root, 'formal'),
                lambda: driver.finalize_installed(root),
                lambda: driver.install_marker(root, own_execution, {
                    'kind': own_execution.lower(), 'role': 'launcher',
                    'spec_key': '', 'exit_code': 0,
                })):
            try:
                action()
            except ValueError as error:
                assert 'profile' in str(error)
            else:
                raise AssertionError(
                    f'{active} accepted pre-existing {foreign_name}')
        assert not (root / own_execution).exists()
        assert not (root / 'tables').exists()
finally:
    fixture.temporary.cleanup()
'''
        for profile in ('formal', 'seed42-pilot', RECOVERY_PROFILE):
            with self.subTest(profile=profile):
                completed = self._run(profile, code)
                self.assertEqual(0, completed.returncode, completed.stderr)


if __name__ == '__main__':
    unittest.main()
