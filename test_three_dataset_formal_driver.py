"""Fail-closed tests for the immutable formal census/plan/claim driver."""
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict, replace
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import random
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

import config as experiment_config
import three_dataset_formal_driver as driver
import three_dataset_formal_registry as registry
import three_dataset_seed42_reconcile as legacy
import runner
from three_dataset_cifar_continuation_reconcile import (
    build_bundle, publish_bundle,
)

from three_dataset_formal_audit import AdmissionRecord
from three_dataset_formal_metrics import FORMULA_VERSION, FormalMetrics
from three_dataset_formal_registry import (
    FormalSpec, explanation_specs, formal_specs, protocol_for,
    registry_sha256,
)


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode()


class FullMatrixBoundaryTests(unittest.TestCase):
    def test_only_full_preclaim_setup_failure_may_have_empty_spec(self):
        payload = {'kind': 'failed_setup', 'role': 'formal-worker-0',
                   'spec_key': '', 'exit_code': 1}
        with tempfile.TemporaryDirectory(prefix='generated_setup_marker_') as temporary:
            for profile in ('formal', 'seed42-pilot', 'seed42-adaptive-recovery', 'full-public-matrix'):
                with mock.patch.dict(os.environ, {'VFCL_EXPERIMENT_PROFILE': profile}):
                    for kind in ('failed_setup', 'failed_job', 'failed_audit', 'failed_retention'):
                        root = Path(temporary) / (profile + '-' + kind)
                        root.mkdir()
                        value = {**payload, 'kind': kind}
                        if profile == 'full-public-matrix' and kind == 'failed_setup':
                            self.assertEqual(value, driver.install_marker(root, 'FAILED_JOB', value))
                            self.assertTrue(driver._pipeline_terminal(root))
                            with self.assertRaises(FileExistsError):
                                driver.install_marker(root, 'FAILED_JOB', value)
                        else:
                            with self.assertRaises(ValueError):
                                driver.install_marker(root, 'FAILED_JOB', value)
                            self.assertFalse((root / 'FAILED_JOB').exists())
                            driver.install_json_exclusive(root / 'FAILED_JOB', value)
                            with self.assertRaises(ValueError):
                                driver._pipeline_terminal(root)

    def test_disk_status_parser_limits_requested_slots(self):
        for slots in ('1', '2'):
            args = driver._parser().parse_args([
                'disk-status', '--root', '/generated/root', '--requested-slots', slots])
            self.assertEqual(args.requested_slots, int(slots))
        with mock.patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
            driver._parser().parse_args([
                'disk-status', '--root', '/generated/root', '--requested-slots', '3'])

    def test_full_authority_names_are_forbidden_in_generated_smokes(self):
        self.assertTrue({'FULL_MATRIX_REUSE.json', 'FULL_MATRIX_PHASE_SUCCESS',
                         'FULL_MATRIX_EXECUTION_SUCCESS'} <= driver._SMOKE_FORMAL_NAMES)

    def test_full_profile_cannot_use_legacy_census_action(self):
        with mock.patch.dict(os.environ, {'VFCL_EXPERIMENT_PROFILE': 'full-public-matrix'}):
            with tempfile.TemporaryDirectory(prefix='generated_full_boundary_') as temporary:
                root = Path(temporary) / 'not-created'
                with self.assertRaisesRegex(ValueError, 'census-full'):
                    driver._install_census(root, {})
                self.assertFalse(root.exists())


class PilotMarkerSchemaTests(unittest.TestCase):
    def test_pilot_marker_payloads_fail_closed_before_installation(self):
        with tempfile.TemporaryDirectory(prefix='pilot_marker_schema_') as temporary:
            root = Path(temporary)
            for name in ('PILOT_PHASE_SUCCESS', 'PILOT_EXECUTION_SUCCESS'):
                self.assertIn(name, driver._MARKERS)
                payload = {'kind': name.lower(), 'role': 'launcher', 'spec_key': '', 'exit_code': 0}
                for malformed in ({**payload, 'kind': 'formal_phase_success'},
                                  {**payload, 'exit_code': 1},
                                  {**payload, 'exit_code': True},
                                  {**payload, 'spec_key': 'unexpected'},
                                  {**payload, 'extra': True}):
                    with self.subTest(name=name, payload=malformed), self.assertRaises(ValueError):
                        driver.install_marker(root, name, malformed)
                self.assertFalse((root / name).exists())


def _protocol_sha256(spec):
    value = json.loads(json.dumps(protocol_for(spec)))
    return hashlib.sha256(_canonical(value)).hexdigest()


def _admission(spec, status='REUSABLE', metrics_value=.5):
    reusable = status == 'REUSABLE'
    task_count = protocol_for(spec)['base_options']['num_tasks']
    metrics = (FormalMetrics(
        metrics_value, 0.0, metrics_value,
        (metrics_value,) * task_count,
        (metrics_value,) * task_count,
        (metrics_value,) * task_count,
    ) if reusable else None)
    return AdmissionRecord(
        status=status,
        reason='admitted' if reusable else 'declared_incompatible',
        spec=asdict(spec),
        protocol_sha256=_protocol_sha256(spec),
        source_sha256='a' * 64 if reusable else '',
        artifact_sha256={'results': 'b' * 64} if reusable else {},
        metrics=metrics,
        metric_formula_version=FORMULA_VERSION,
        trajectory_sha256='c' * 64 if reusable else '',
    )


def _completed_record(spec, metric, hardware=None, source_commit='d' * 40,
                      plan_sha256='e' * 64):
    task_count = protocol_for(spec)['base_options']['num_tasks']
    hardware = ({
        'gpu_name': 'test-gpu', 'gpu_count': 1, 'cuda': 'test-cuda',
        'torch': 'test-torch', 'driver': 'test-driver',
    } if hardware is None else hardware)
    resource = {
        'hardware_identity': hardware,
        'instrumentation': 'formal-resource-v1',
        'runtime_seconds': 1.25,
        'peak_gpu_memory_bytes': 1024,
        'checkpoint_size_bytes': 2048,
        'added_parameters': 0,
        'communication_bytes': 0,
        'replay_type': 'none',
        'raw_examples_per_class': 0,
        'persistent_embeddings': 0,
        'privacy_label': 'no-persistent-raw-or-embedding-replay',
    }
    record = {
        'kind': 'formal_completed_run',
        'spec_key': driver.spec_key(spec),
        'dataset': spec.dataset,
        'method': spec.method,
        'seed': spec.seed,
        'explanation': spec.explanation,
        'registry_sha256': registry_sha256(),
        'metric_formula_version': FORMULA_VERSION,
        'plan_sha256': plan_sha256,
        'source_commit': source_commit,
        'protocol_sha256': _protocol_sha256(spec),
        'trajectory_sha256': '1' * 64,
        'admission_record_sha256': '2' * 64,
        'artifact_sha256': {'results': '3' * 64},
        'metrics': {
            'aa_final': float(metric),
            'bwt': float(metric),
            'taskil_final': float(metric),
            'aa_trajectory': [float(metric)] * task_count,
            'class_final': [float(metric)] * task_count,
            'taskil_final_by_task': [float(metric)] * task_count,
        },
        'command_sha256': '4' * 64,
        'log_sha256': '5' * 64,
        'claim_sha256': '6' * 64,
        'launch_sha256': '7' * 64,
        'resource': resource,
    }
    record.update(driver._profile_binding())
    record['record_sha256'] = driver._digest(record)
    return record


def _write_resource_checkpoint(spec, path):
    """Install real method state for generated control-plane fixtures only."""
    with torch.random.fork_rng(devices=[]):
        _, checkpoint, _ = MethodResourceStateTests()._fixture(spec.method, spec=spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o644)
    torch.save(checkpoint, path)
    path.chmod(0o444)
    return driver._method_resource_state(spec, path)


def _all_completed_records():
    records = []
    for spec in formal_specs():
        records.append(_completed_record(
            spec, {42: .2, 43: .3, 44: .4}[spec.seed]))
    records.extend(_completed_record(spec, .5) for spec in explanation_specs())
    return records


class FormalDriverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix='formal_driver_test_')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)

    def test_continuation_audit_drains_only_the_twenty_four_new_cells(self):
        with mock.patch.dict(os.environ, {
                'VFCL_EXPERIMENT_PROFILE':
                'single-dataset-verified-continuation-v1',
                'VFCL_FORMAL_DATASET': 'cifar100'}):
            missing = {driver.spec_key(spec) for spec in formal_specs()
                       if spec.method not in {
                           'finetune', 'lwf', 'ewc', 'er', 'der_pp', 'er_ace'}}
            plan = {'missing_jobs': list(missing)}
            seen = []

            def completed(_root, spec, _plan):
                key = driver.spec_key(spec)
                seen.append(key)
                return {'spec_key': key} if key in missing else None

            with mock.patch.object(driver, '_installed_completed_record',
                                   side_effect=completed):
                self.assertTrue(driver._audit_drained(
                    self.base, plan, 'formal', {}, None))
                self.assertFalse(driver._audit_drained(
                    self.base, plan, 'formal', {
                        next(iter(missing)): {'phase': 'formal'}}, None))
                self.assertFalse(driver._audit_drained(
                    self.base, plan, 'formal', {}, {
                        'owner': {'phase': 'formal'}}))
            self.assertEqual(missing, set(seen))
            self.assertEqual(72, len(seen))
            with mock.patch.object(driver, '_installed_completed_record',
                                   return_value=None):
                self.assertFalse(driver._audit_drained(
                    self.base, plan, 'formal', {}, None))

    def test_continuation_success_markers_are_separate_from_dataset_markers(self):
        expected = {
            'DATASET_CONTINUATION_PHASE_SUCCESS':
                'dataset_continuation_phase_success',
            'DATASET_CONTINUATION_SUCCESS':
                'dataset_continuation_success',
        }
        for name, kind in expected.items():
            self.assertIn(name, driver._MARKERS)
            self.assertEqual({kind}, driver._MARKER_KINDS[name])
            self.assertEqual(driver.CONTINUATION_PROFILE,
                             driver._SUCCESS_MARKER_PROFILES[name])
            self.assertIn(name, driver._SMOKE_FORMAL_NAMES)
        self.assertEqual(driver.SINGLE_DATASET_PROFILE,
                         driver._SUCCESS_MARKER_PROFILES[
                             'DATASET_EXECUTION_SUCCESS'])
        root = self.base / 'continuation-markers'
        root.mkdir(mode=0o700)
        with mock.patch.dict(os.environ, {
                'VFCL_EXPERIMENT_PROFILE':
                'single-dataset-verified-continuation-v1',
                'VFCL_FORMAL_DATASET': 'cifar100'}):
            with self.assertRaisesRegex(ValueError, 'profile'):
                driver.install_marker(root, 'DATASET_EXECUTION_SUCCESS', {
                    'kind': 'dataset_execution_success', 'role': 'parent',
                    'spec_key': '', 'exit_code': 0})
            self.assertFalse((root / 'DATASET_EXECUTION_SUCCESS').exists())

    def test_continuation_finalize_installs_only_provenance_tables(self):
        import three_dataset_cifar_continuation_report as report
        import three_dataset_cifar_continuation_reconcile as reconcile

        root = self.base / 'continuation-finalize'
        root.mkdir(mode=0o700)
        census = {'kind': 'generated_continuation_census'}
        plan = {'missing_jobs': ['cifar100:gpm:42']}
        bundle = {'kind': 'generated_continuation_reuse'}
        driver.install_json_exclusive(root / 'COMPATIBILITY_CENSUS.json', census)
        tables = {
            'CIFAR_CONTINUATION_PER_RUN.csv': b'per-run\n',
            'CIFAR_CONTINUATION_TABLE.csv': b'summary\n',
            'CIFAR_CONTINUATION_RESOURCE.csv': b'resource\n',
            'CIFAR_CONTINUATION_AUDIT.json': _canonical({
                'kind': 'cifar_continuation_report_audit_v1',
                'table_sha256': {},
            }) + b'\n',
        }
        env = {'VFCL_EXPERIMENT_PROFILE':
               'single-dataset-verified-continuation-v1',
               'VFCL_FORMAL_DATASET': 'cifar100'}
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(driver, '_require_pipeline_drained') as drained, \
                mock.patch.object(driver, '_load_installed_plan',
                                  return_value=(plan, {})), \
                mock.patch.object(driver, '_read_root_identity_record',
                                  return_value=({
                                      'reuse_bundle_sha256': driver._sha256_bytes(
                                          driver._canonical_json(bundle) + b'\n'),
                                  }, b'', None)), \
                mock.patch.object(reconcile, 'load_bundle', return_value=bundle), \
                mock.patch.object(report, 'combined_rows', return_value=[{}]) as combine, \
                mock.patch.object(report, 'render_continuation_tables',
                                  return_value=tables):
            hashes = driver.finalize_installed(root)
            with self.assertRaises(FileExistsError):
                driver.finalize_installed(root)
        self.assertEqual(2, drained.call_count)
        drained.assert_called_with(root, ('formal',))
        self.assertEqual(2, combine.call_count)
        combine.assert_called_with(root, plan, census, bundle)
        self.assertEqual(set(tables), set(hashes))
        self.assertEqual(set(tables), {path.name for path in
                                     (root / 'tables').iterdir()})
        audit = json.loads((root / 'tables' /
                            'CIFAR_CONTINUATION_AUDIT.json').read_bytes())
        self.assertEqual(driver._digest(plan), audit['input_sha256']['plan'])
        self.assertEqual(driver._digest(census), audit['input_sha256']['census'])
        self.assertEqual(driver._sha256_bytes(driver._canonical_json(bundle)
                                              + b'\n'),
                         audit['input_sha256']['reuse_bundle'])

    def test_continuation_installs_exact_twenty_four_job_plan(self):
        old_env = {
            'VFCL_EXPERIMENT_PROFILE': 'single-dataset-full-matrix',
            'VFCL_FORMAL_DATASET': 'cifar100',
            'VFCL_PYTHON': '/home/c3080/YangXiaoXiang/envs/vfcl/bin/python',
        }
        new_env = {**old_env,
                   'VFCL_EXPERIMENT_PROFILE':
                   'single-dataset-verified-continuation-v1'}
        source_dir = self.base / 'source'
        origin = Path('/home/c3080/YangXiaoXiang/VF-CL/results/'
                      'formal-cifar3080-dual-20g-20260928-v8')
        worktree = Path(driver.__file__).resolve().parent
        real_git = legacy.git

        def clean_for_tdd(path, *args):
            if args == ('status', '--porcelain'):
                return ''
            return real_git(path, *args)

        with mock.patch.dict(os.environ, old_env):
            bundle = build_bundle(origin, worktree)
            with mock.patch.object(legacy, 'git', side_effect=clean_for_tdd):
                publish_bundle(bundle, source_dir)
        root = self.base / 'continuation-root'
        with mock.patch.dict(os.environ, new_env):
            driver._install_continuation_census(root, source_dir)
            plan = driver._install_plan(root)
            loaded, _manifest = driver._load_installed_plan(root)
            self.assertEqual(plan, loaded)
            self.assertEqual(len(plan['formal_cells']), 42)
            self.assertEqual(len(plan['missing_jobs']), 24)
            self.assertEqual(plan['missing_jobs'][0], 'cifar100:gpm:42')
            with mock.patch.dict(os.environ, {'VFCL_GPU_COUNT': '2'}):
                self.assertEqual(driver.pipeline_inflight_limit(), 2)
            claimed = driver.claim_next(
                plan, root / 'claims', 'formal', self._owner(root, ''),
                pipeline=False)
            self.assertEqual(claimed, 'cifar100:gpm:42')
            self.assertEqual({path.name for path in (root / 'reuse').iterdir()}, {
                'CIFAR_CONTINUATION_REUSE.json',
                'CIFAR_CONTINUATION_REUSE_AUDIT.json',
                'CIFAR_CONTINUATION_REUSE_SUCCESS',
            })
            identity = json.loads((root / 'FORMAL_ROOT_IDENTITY.json').read_text())
            self.assertEqual(identity['reuse_bundle_sha256'], hashlib.sha256(
                driver._canonical_json(bundle) + b'\n').hexdigest())
            target = root / 'reuse' / 'CIFAR_CONTINUATION_REUSE.json'
            target.chmod(0o644)
            target.write_bytes(target.read_bytes() + b' ')
            target.chmod(0o444)
            with self.assertRaises(ValueError):
                driver._load_installed_plan(root)

    def _census(self, reusable=(), metric=.5):
        reusable = set(reusable)
        declarations = {
            driver.spec_key(spec): {'marker': driver.spec_key(spec)}
            for spec in formal_specs() if spec in reusable
        }

        def audit(spec, declaration):
            self.assertIs(declarations[driver.spec_key(spec)], declaration)
            return _admission(spec, metrics_value=metric)

        with mock.patch.object(driver, 'audit_candidate', side_effect=audit):
            return driver.build_census(declarations)

    def test_single_dataset_plan_and_summary_are_exactly_42_cells(self):
        with mock.patch.dict(os.environ, {
                'VFCL_EXPERIMENT_PROFILE': 'single-dataset-full-matrix',
                'VFCL_FORMAL_DATASET': 'isolet'}):
            census = driver.build_census({})
            plan = driver.build_plan(census)
            self.assertEqual(42, len(plan['formal_cells']))
            self.assertEqual(42, len(plan['missing_jobs']))
            self.assertEqual([], plan['explanation_cells'])
            self.assertEqual('isolet', plan['formal_dataset'])
            first = formal_specs()[0]
            with mock.patch.object(driver, 'audit_candidate',
                                   return_value=_admission(first)):
                with self.assertRaisesRegex(ValueError, 'reuse'):
                    driver.build_census({driver.spec_key(first): {'old': True}})
            records = _all_completed_records()
            summary = driver.summarize_records(records)
            self.assertEqual(42, len(summary['per_run_records']))
            self.assertEqual(14, len(summary['formal_rows']))
            self.assertEqual('isolet', summary['formal_dataset'])
            self.assertAlmostEqual(.3, summary['formal_rows'][0]['aa_final_mean'])
            self.assertAlmostEqual(.1, summary['formal_rows'][0]['aa_final_std'])
            rendered = driver.render_tables(records)
            self.assertEqual({'FORMAL_PER_RUN.csv', 'FORMAL_TABLE.csv',
                              'RESOURCE_PRIVACY_TABLE.csv'}, set(rendered))
            with self.assertRaises(ValueError):
                driver.summarize_records(records[:-1])
            changed = copy.deepcopy(records)
            changed[0]['formal_dataset'] = 'upmc_food101'
            with self.assertRaises(ValueError):
                driver.summarize_records(changed)

    def test_single_dataset_success_markers_are_profile_specific(self):
        with mock.patch.dict(os.environ, {
                'VFCL_EXPERIMENT_PROFILE': 'single-dataset-full-matrix',
                'VFCL_FORMAL_DATASET': 'upmc_food101'}):
            self.assertIn('DATASET_PHASE_SUCCESS', driver._MARKERS)
            self.assertIn('DATASET_EXECUTION_SUCCESS', driver._MARKERS)
            self.assertEqual(driver.SINGLE_DATASET_PROFILE,
                             driver._SUCCESS_MARKER_PROFILES['DATASET_EXECUTION_SUCCESS'])

    def test_single_dataset_two_gpu_pipeline_has_two_reservations(self):
        with mock.patch.dict(os.environ, {
                'VFCL_EXPERIMENT_PROFILE': 'single-dataset-full-matrix',
                'VFCL_FORMAL_DATASET': 'cifar100', 'VFCL_GPU_COUNT': '2'}):
            root, plan = self._installed(name='scoped-two-gpu')
            owner = self._owner(root, '')
            self.assertEqual(2, driver.pipeline_inflight_limit())
            self.assertEqual(plan['missing_jobs'][0], driver.claim_next(
                plan, root / 'claims', 'formal', owner, pipeline=True))
            self.assertEqual(plan['missing_jobs'][1], driver.claim_next(
                plan, root / 'claims', 'formal', owner, pipeline=True))
            with self.assertRaises(driver.PipelineBackpressure):
                driver.claim_next(plan, root / 'claims', 'formal', owner, pipeline=True)

    def test_single_dataset_one_gpu_pipeline_remains_singleton(self):
        with mock.patch.dict(os.environ, {
                'VFCL_EXPERIMENT_PROFILE': 'single-dataset-full-matrix',
                'VFCL_FORMAL_DATASET': 'cifar100', 'VFCL_GPU_COUNT': '1'}):
            self.assertEqual(1, driver.pipeline_inflight_limit())

    def test_single_dataset_two_queued_jobs_keep_exact_audit_ownership(self):
        with mock.patch.dict(os.environ, {
                'VFCL_EXPERIMENT_PROFILE': 'single-dataset-full-matrix',
                'VFCL_FORMAL_DATASET': 'cifar100', 'VFCL_GPU_COUNT': '2',
                'VFCL_PYTHON': str(registry._REVIEWED_VFCL_PYTHON)}):
            root, plan = self._installed(name='scoped-two-audits')
            first, first_owner, _ = self._pipeline_run(root, plan, index=0, gpu=0)
            second, second_owner, _ = self._pipeline_run(root, plan, index=1, gpu=1)
            driver.queue_audit(root, first, 0, first_owner)
            driver.queue_audit(root, second, 1, second_owner)
            auditor = self._owner(root, '', role='retention-auditor')
            driver.release_gpu(root, 0, first_owner)
            self.assertEqual(first, driver.next_audit(
                root, 'formal', auditor)['spec_key'])
            self._pipeline_record(root, plan, first)
            self.assertEqual(first, driver.retention_audit_context(
                root, first, auditor)['handoff']['spec_key'])
            with self.assertRaises(driver.PipelineBackpressure):
                driver.claim_next(plan, root / 'claims', 'formal',
                                  self._owner(root, ''), pipeline=True)
            driver.complete_audit(root, first, auditor)
            self.assertEqual(plan['missing_jobs'][2], driver.claim_next(
                plan, root / 'claims', 'formal', self._owner(root, ''),
                pipeline=True))
            driver.release_gpu(root, 1, second_owner)
            self.assertEqual(second, driver.next_audit(
                root, 'formal', auditor)['spec_key'])

    def test_continuation_two_queued_jobs_keep_exact_audit_ownership(self):
        old_env = {
            'VFCL_EXPERIMENT_PROFILE': 'single-dataset-full-matrix',
            'VFCL_FORMAL_DATASET': 'cifar100',
            'VFCL_PYTHON': str(registry._REVIEWED_VFCL_PYTHON),
        }
        new_env = {
            **old_env,
            'VFCL_EXPERIMENT_PROFILE':
                'single-dataset-verified-continuation-v1',
            'VFCL_GPU_COUNT': '2',
        }
        source = self.base / 'continuation-reuse'
        origin = Path(
            '/home/c3080/YangXiaoXiang/VF-CL/results/'
            'formal-cifar3080-dual-20g-20260928-v8'
        )
        worktree = Path(driver.__file__).resolve().parent
        real_git = legacy.git

        def clean_for_tdd(path, *args):
            if args == ('status', '--porcelain'):
                return ''
            return real_git(path, *args)

        with mock.patch.dict(os.environ, old_env):
            bundle = build_bundle(origin, worktree)
            with mock.patch.object(legacy, 'git', side_effect=clean_for_tdd):
                publish_bundle(bundle, source)
        root = self.base / 'continuation-two-audits'
        with mock.patch.dict(os.environ, new_env):
            driver._install_continuation_census(root, source)
            plan = driver._install_plan(root)
            first, first_owner, _ = self._pipeline_run(
                root, plan, index=0, gpu=0)
            second, second_owner, _ = self._pipeline_run(
                root, plan, index=1, gpu=1)
            driver.queue_audit(root, first, 0, first_owner)
            driver.queue_audit(root, second, 1, second_owner)
            auditor = self._owner(root, '', role='retention-auditor')
            driver.release_gpu(root, 0, first_owner)
            self.assertEqual(first, driver.next_audit(
                root, 'formal', auditor)['spec_key'])
            self._pipeline_record(root, plan, first)
            self.assertEqual(first, driver.retention_audit_context(
                root, first, auditor)['handoff']['spec_key'])
            self.assertTrue((
                root / 'audit_queue' /
                f'{driver._claim_name(second)}.json'
            ).is_file())

    @staticmethod
    def _rewrite(path, value):
        path.chmod(0o644)
        path.write_bytes(_canonical(value) + b'\n')
        path.chmod(0o444)

    def _installed(self, census=None, name='formal'):
        census = driver.build_census({}) if census is None else census
        plan = driver.build_plan(census)
        root = driver.validate_formal_root(self.base / name)
        driver.install_json_exclusive(
            root / 'FORMAL_REGISTRY.json', driver._registry_payload())
        driver.install_json_exclusive(root / 'COMPATIBILITY_CENSUS.json', census)
        driver.install_json_exclusive(root / 'FORMAL_PLAN.json', plan)
        driver.install_json_exclusive(
            root / 'MISSING_JOBS.json',
            driver._missing_jobs_payload(plan, census))
        (root / 'claims').mkdir(mode=0o700)
        driver.install_formal_root_identity(root, plan)
        return root, plan

    def test_installed_plan_runs_completed_identity_preflight(self):
        root, plan = self._installed()
        with mock.patch.object(
                driver, '_validate_completed_plan_identity',
                wraps=driver._validate_completed_plan_identity) as validate:
            loaded, _manifest = driver._load_installed_plan(root)
        self.assertEqual(plan, loaded)
        validate.assert_called_once_with(plan)

    def _owner(self, root, job, token='token', pid=None, start=None,
               phase='formal', pgid=None, role='formal-worker'):
        pid = os.getpid() if pid is None else pid
        return {
            'kind': 'formal_job_claim',
            'job': job,
            'launcher_token': token,
            'worker_role': role,
            'phase': phase,
            'pid': pid,
            'pgid': os.getpgid(pid) if pgid is None else pgid,
            'process_start_time': (driver._process_start_time(pid)
                                   if start is None else start),
            'source_commit': driver._source_commit(),
            'root_identity': driver._root_identity(root),
        }

    @staticmethod
    def _claim_path(root, job):
        name = driver._claim_name(job) if hasattr(driver, '_claim_name') else job
        return root / 'claims' / name

    @staticmethod
    def _identity_for(root, plan, manifest):
        details = root.stat()
        return {
            'dev': details.st_dev,
            'inode': details.st_ino,
            'ctime_ns': details.st_ctime_ns,
            'size': details.st_size,
            'hash': driver._digest({
                'plan_sha256': driver._digest(plan),
                'missing_jobs_sha256': driver._digest(manifest),
            }),
        }

    def _pipeline_run(self, root, plan, index=0, producer=None, gpu=0):
        key = plan['missing_jobs'][index]
        owner = producer or self._owner(root, key, role=f'producer-{index}')
        owner = {**owner, 'job': key}
        self.assertEqual(key, driver.claim_next(
            plan, root / 'claims', 'formal', owner))
        self.assertTrue(driver.claim_gpu(root, gpu, owner))
        run = driver.prepare_run(root, key, owner)
        spec = driver.spec_for_key(key)
        for logical in driver.resource_artifact_names(spec):
            path = run / logical
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'synthetic CPU evidence\n')
            path.chmod(0o444)
        measurements = _completed_record(spec, .5)['resource'].copy()
        for name in ('instrumentation', 'checkpoint_size_bytes'):
            measurements.pop(name)
        measurements.update(_write_resource_checkpoint(
            spec, run / 'checkpoints/formal_final.pt'))
        driver.resource_record(root, key, run, measurements=measurements)
        return key, owner, run

    def _pipeline_record(self, root, plan, key):
        spec = driver.spec_for_key(key)
        run = root / 'runs' / driver.safe_spec_name(spec)
        evidence = driver._load_json_file(run / 'RESOURCE_EVIDENCE.json')
        record = _completed_record(
            spec, .5, source_commit=driver._source_commit(),
            plan_sha256=driver._digest(plan))
        for name in ('claim_sha256', 'launch_sha256', 'command_sha256',
                     'resource'):
            record[name] = evidence[name]
        record['log_sha256'] = evidence['artifact_sha256']['job.log']
        record['record_sha256'] = driver._digest({
            k: v for k, v in record.items() if k != 'record_sha256'})
        records = driver._ensure_directory(root, 'records')
        driver.install_json_exclusive(
            records / f'{driver.safe_spec_name(spec)}.json', record)
        return record

    def test_success_markers_are_profile_bound_on_drained_root(self):
        root, plan = self._installed()
        records = root / 'records'
        records.mkdir()
        commit = driver._source_commit()
        for spec in (*formal_specs(), *explanation_specs()):
            driver.install_json_exclusive(
                records / f'{driver.safe_spec_name(spec)}.json',
                _completed_record(spec, .5, source_commit=commit,
                                  plan_sha256=driver._digest(plan)))
        self.assertTrue(driver.audit_phase_ready(root, 'formal'))
        self.assertTrue(driver.audit_phase_ready(root, 'explanation'))
        self.assertEqual([], list((root / 'audit_queue').iterdir()))

        formal_names = ('FORMAL_PHASE_SUCCESS', 'EXPLANATION_PHASE_SUCCESS',
                        'FORMAL_EXECUTION_SUCCESS')
        pilot_names = ('PILOT_PHASE_SUCCESS', 'PILOT_EXECUTION_SUCCESS')
        recovery_names = (
            'RECOVERY_PHASE_SUCCESS', 'RECOVERY_EXECUTION_SUCCESS')
        by_profile = {
            driver.FORMAL_PROFILE: formal_names,
            driver.PILOT_PROFILE: pilot_names,
            driver.RECOVERY_PROFILE: recovery_names,
        }
        allowed = by_profile[driver.experiment_profile()]
        rejected = tuple(
            name for profile, names in by_profile.items()
            if profile != driver.experiment_profile() for name in names)
        for name in rejected:
            with self.subTest(rejected=name):
                payload = {'kind': name.lower(), 'role': 'launcher',
                           'spec_key': '', 'exit_code': 0}
                with self.assertRaisesRegex(ValueError, 'profile'):
                    driver.install_marker(root, name, payload)
                self.assertFalse((root / name).exists())
        for name in allowed:
            with self.subTest(allowed=name):
                payload = {'kind': name.lower(), 'role': 'launcher',
                           'spec_key': '', 'exit_code': 0}
                self.assertEqual(payload, driver.install_marker(root, name, payload))
                self.assertEqual(0o444, (root / name).stat().st_mode & 0o777)
        for name, kind, key in (
                ('FAILED_JOB', 'failed_job', driver.spec_key(formal_specs()[0])),
                ('FORMAL_STOPPED', 'formal_stopped', '')):
            payload = {'kind': kind, 'role': 'launcher', 'spec_key': key, 'exit_code': 7}
            self.assertEqual(payload, driver.install_marker(root, name, payload))
            self.assertEqual(0o444, (root / name).stat().st_mode & 0o777)

    @unittest.skipUnless(driver.experiment_profile() == driver.RECOVERY_PROFILE,
                         'recovery subprocess contract')
    def test_recovery_atomic_claim_capacity_through_record_admission(self):
        root, plan = self._installed()
        owner = self._owner(root, '')
        def reserve(_):
            try:
                return driver.claim_next(plan, root / 'claims', 'formal', owner, pipeline=True)
            except driver.PipelineBackpressure:
                return 'full'
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(reserve, range(3)))
        self.assertEqual([plan['missing_jobs'][0]], [key for key in results if key != 'full'])
        self.assertEqual(2, results.count('full'))

    @unittest.skipUnless(driver.experiment_profile() == driver.RECOVERY_PROFILE,
                         'recovery subprocess contract')
    def test_recovery_pending_audit_retains_slot_after_record_install(self):
        root, plan = self._installed(name='admission')
        key, producer, _ = self._pipeline_run(root, plan)
        owner = self._owner(root, '')
        auditor = self._owner(root, '', role='auditor')
        driver.queue_audit(root, key, 0, producer)  # same key at capacity is a handoff
        with self.assertRaises(driver.PipelineBackpressure):
            driver.claim_next(plan, root / 'claims', 'formal', owner, pipeline=True)
        driver.release_gpu(root, 0, producer)
        self.assertIsNotNone(driver.next_audit(root, 'formal', auditor))
        self._pipeline_record(root, plan, key)
        with self.assertRaises(driver.PipelineBackpressure):
            driver.claim_next(plan, root / 'claims', 'formal', owner, pipeline=True)
        driver.complete_audit(root, key, auditor)
        self.assertEqual(plan['missing_jobs'][1], driver.claim_next(
            plan, root / 'claims', 'formal', owner, pipeline=True))

    @unittest.skipUnless(driver.experiment_profile() == driver.RECOVERY_PROFILE,
                         'recovery retention contract')
    def test_retention_context_requires_exact_active_completed_audit(self):
        root, plan = self._installed(name='retention-context')
        key, producer, run = self._pipeline_run(root, plan)
        auditor = self._owner(root, '', role='retention-auditor')
        driver.queue_audit(root, key, 0, producer)
        driver.release_gpu(root, 0, producer)
        self.assertEqual(key, driver.next_audit(
            root, 'formal', auditor)['spec_key'])
        record = self._pipeline_record(root, plan, key)
        resource = driver._load_json_file(run / 'RESOURCE_EVIDENCE.json')
        before = self._pipeline_snapshot(root)
        context = driver.retention_audit_context(root, key, auditor)
        self.assertEqual({
            'plan', 'handoff', 'record', 'resource', 'run_dir', 'num_tasks'},
            set(context))
        self.assertEqual(plan, context['plan'])
        self.assertEqual(record, context['record'])
        self.assertEqual(resource, context['resource'])
        self.assertEqual(str(run), context['run_dir'])
        self.assertEqual(10, context['num_tasks'])
        self.assertEqual(before, self._pipeline_snapshot(root))
        for changed_key, changed_owner in (
                (plan['missing_jobs'][1], auditor),
                (key, {**auditor, 'launcher_token': 'foreign'})):
            with self.assertRaises(ValueError):
                driver.retention_audit_context(root, changed_key, changed_owner)
            self.assertEqual(before, self._pipeline_snapshot(root))

    @unittest.skipUnless(driver.experiment_profile() == driver.RECOVERY_PROFILE,
                         'recovery retention contract')
    def test_retention_context_rejects_second_valid_queued_handoff(self):
        root, plan = self._installed(name='retention-unrelated-queue')
        first, first_owner, _ = self._pipeline_run(root, plan)
        driver.queue_audit(root, first, 0, first_owner)
        driver.release_gpu(root, 0, first_owner)
        auditor = self._owner(root, '', role='retention-auditor')
        self.assertEqual(first, driver.next_audit(
            root, 'formal', auditor)['spec_key'])
        self._pipeline_record(root, plan, first)

        second, second_owner, _ = self._pipeline_run(root, plan, index=1, gpu=1)
        unrelated = driver._audit_handoff(root, plan, second, 1)
        driver.release_gpu(root, 1, second_owner)
        driver.install_json_exclusive(
            root / 'audit_queue' / f'{driver._claim_name(second)}.json',
            unrelated)
        before = self._pipeline_snapshot(root)

        with self.assertRaisesRegex(ValueError, 'queue'):
            driver.retention_audit_context(root, first, auditor)
        self.assertEqual(before, self._pipeline_snapshot(root))

    @unittest.skipUnless(driver.experiment_profile() == driver.RECOVERY_PROFILE,
                         'recovery subprocess contract')
    def test_recovery_queue_admission_counts_distinct_inflight_keys(self):
        for queued_first in (False, True):
            with self.subTest(queued_first=queued_first):
                root, plan = self._installed(name=f'queue-capacity-{queued_first}')
                first, producer, _ = self._pipeline_run(root, plan)
                if queued_first:
                    driver.queue_audit(root, first, 0, producer)
                second, second_owner, _ = self._pipeline_run(root, plan, index=1, gpu=1)
                driver.audit_phase_ready(root, 'formal')  # initialize existing control directory
                before = self._pipeline_snapshot(root)
                with self.assertRaisesRegex(ValueError, 'capacity'):
                    driver.queue_audit(root, second, 1, second_owner)
                self.assertEqual(before, self._pipeline_snapshot(root))

    def test_pipeline_capacity_three_atomic_and_legacy_unchanged(self):
        self.assertTrue(hasattr(driver, 'PipelineBackpressure'),
                        'pipeline capacity exception is missing')
        root, plan = self._installed()
        owner = self._owner(root, '')
        def reserve(_):
            try:
                return driver.claim_next(
                    plan, root / 'claims', 'formal', owner, pipeline=True)
            except driver.PipelineBackpressure:
                return 'full'
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(reserve, range(5)))
        self.assertEqual(3, len(set(results) - {'full'}))
        self.assertEqual(2, results.count('full'))
        self.assertEqual(3, len(list((root / 'claims').iterdir())))
        self.assertEqual(plan['missing_jobs'][3], driver.claim_next(
            plan, root / 'claims', 'formal', owner))
        with mock.patch('sys.stdout', io.StringIO()) as output:
            status = driver.main([
                'claim', '--root', str(root), '--phase', 'formal',
                '--owner-json', _canonical(owner).decode(), '--pipeline'])
        self.assertEqual(75, status)
        self.assertEqual('', output.getvalue())

    def test_pipeline_queue_release_singleton_completion_and_drain(self):
        self.assertTrue(hasattr(driver, 'queue_audit'),
                        'durable audit handoff API is missing')
        root, plan = self._installed()
        key, producer, run = self._pipeline_run(root, plan)
        before = {name: (run / name).read_bytes() for name in (
            'CLAIM_OWNER.json', 'FORMAL_JOB_SPEC.json', 'LAUNCH_STARTED.json',
            'RESOURCE_EVIDENCE.json')}
        auditor = self._owner(root, '', role='auditor')
        driver.queue_audit(root, key, 0, producer)
        self.assertTrue((root / 'gpu_claims' / 'gpu-0').is_dir())
        self.assertIsNone(driver.next_audit(root, 'formal', auditor))
        self.assertFalse(driver.audit_phase_ready(root, 'formal'))
        with self.assertRaises((ValueError, FileExistsError)):
            driver.queue_audit(root, key, 0, producer)
        driver.release_gpu(root, 0, producer)
        key2, producer2, _ = self._pipeline_run(
            root, plan, index=1, producer=producer)
        selected = driver.next_audit(root, 'formal', auditor)
        self.assertEqual({
            'spec_key': key, 'run_dir': str(run), 'physical_gpu': 0,
            'seed': driver.spec_for_key(key).seed}, selected)
        self.assertEqual(int, type(selected['physical_gpu']))
        self.assertEqual(int, type(selected['seed']))
        self.assertIsNone(driver.next_audit(root, 'formal', auditor))
        with self.assertRaises(ValueError):
            driver.complete_audit(root, key, auditor)
        self._pipeline_record(root, plan, key)
        with self.assertRaises(ValueError):
            driver.complete_audit(
                root, key, {**auditor, 'launcher_token': 'foreign'})
        driver.complete_audit(root, key, auditor)
        self.assertEqual([], list((root / 'audit_queue').iterdir()))
        for name, content in before.items():
            self.assertEqual(content, (run / name).read_bytes())
        self.assertTrue((root / 'gpu_claims' / 'gpu-0').exists())
        with self.assertRaises((ValueError, FileNotFoundError)):
            driver.queue_audit(root, key, 0, producer)
        with self.assertRaises((ValueError, FileNotFoundError)):
            driver.complete_audit(root, key, auditor)
        self.assertFalse(driver.audit_phase_ready(root, 'formal'))
        driver.release_gpu(root, 0, producer2)

    def test_pipeline_exhausted_precedes_capacity_and_completed_frees_slot(self):
        root, plan = self._installed(self._census(formal_specs()[:-3]))
        owner = self._owner(root, '')
        for _ in range(3):
            self.assertIsNotNone(driver.claim_next(
                plan, root / 'claims', 'formal', owner, pipeline=True))
        self.assertIsNone(driver.claim_next(
            plan, root / 'claims', 'formal', owner, pipeline=True))
        with mock.patch('sys.stdout', io.StringIO()) as output:
            self.assertEqual(0, driver.main([
                'claim', '--root', str(root), '--phase', 'formal',
                '--owner-json', _canonical(owner).decode(), '--pipeline']))
        self.assertEqual('\n', output.getvalue())
        root2, plan2 = self._installed(name='capacity-completed')
        key, producer, _ = self._pipeline_run(root2, plan2)
        template = self._owner(root2, '')
        for _ in range(2):
            driver.claim_next(plan2, root2 / 'claims', 'formal',
                              template, pipeline=True)
        with self.assertRaises(driver.PipelineBackpressure):
            driver.claim_next(plan2, root2 / 'claims', 'formal',
                              template, pipeline=True)
        self._pipeline_record(root2, plan2, key)
        self.assertEqual(plan2['missing_jobs'][3], driver.claim_next(
            plan2, root2 / 'claims', 'formal', template, pipeline=True))
        driver.release_gpu(root2, 0, producer)

    def test_pipeline_handoff_rejects_tampered_authorities_and_symlinks(self):
        root, plan = self._installed()
        key, producer, run = self._pipeline_run(root, plan)
        auditor = self._owner(root, '', role='auditor')
        with self.assertRaises(ValueError):
            driver.queue_audit(root, key, 1, producer)
        with self.assertRaises(ValueError):
            driver.queue_audit(
                root, key, 0, {**producer, 'launcher_token': 'foreign'})
        driver.queue_audit(root, key, 0, producer)
        path = root / 'audit_queue' / f'{driver._claim_name(key)}.json'
        original = driver._load_json_file(path)
        for name, value in (
                ('spec_key', plan['missing_jobs'][1]), ('phase', 'explanation'),
                ('source_commit', '0' * 40), ('plan_sha256', '0' * 64),
                ('root_identity', {**original['root_identity'], 'hash': '0' * 64}),
                ('seed', True), ('physical_gpu', True), ('physical_gpu', 1),
                ('resource_sha256', '0' * 64), ('started_sha256', '0' * 64),
                ('owner', {**producer, 'launcher_token': 'foreign'}),
                ('run_dir', str(self.base)), ('extra', 'unregistered')):
            with self.subTest(name=name, value=value):
                self._rewrite(path, {**original, name: value})
                with self.assertRaises(ValueError):
                    driver.next_audit(root, 'formal', auditor)
                with self.assertRaises(ValueError):
                    driver.audit_phase_ready(root, 'formal')
        self._rewrite(path, original)
        resource_path = run / 'RESOURCE_EVIDENCE.json'
        resource = driver._load_json_file(resource_path)
        self._rewrite(resource_path, {
            **resource, 'resource': {**resource['resource'], 'runtime_seconds': 2.0}})
        with self.assertRaises(ValueError):
            driver.next_audit(root, 'formal', auditor)
        self._rewrite(resource_path, resource)
        for target in ('file', 'directory'):
            saved = path.read_bytes()
            path.unlink()
            path.symlink_to(resource_path if target == 'file' else run)
            with self.assertRaises((ValueError, OSError)):
                driver.next_audit(root, 'formal', auditor)
            path.unlink()
            path.write_bytes(saved)
            path.chmod(0o444)
        unknown = root / 'audit_queue' / 'unknown.json'
        driver.install_json_exclusive(unknown, {})
        with self.assertRaises(ValueError):
            driver.next_audit(root, 'formal', auditor)
        unknown.unlink()
        driver.release_gpu(root, 0, producer)

    def test_pipeline_requires_started_and_complete_resource_metadata(self):
        root, plan = self._installed()
        key = plan['missing_jobs'][0]
        producer = self._owner(root, key)
        driver.claim_next(plan, root / 'claims', 'formal', producer)
        driver.claim_gpu(root, 0, producer)
        with self.assertRaises((ValueError, FileNotFoundError)):
            driver.queue_audit(root, key, 0, producer)
        run = driver.prepare_run(root, key, producer)
        with self.assertRaises(ValueError):
            driver.queue_audit(root, key, 0, producer)
        driver.install_json_exclusive(run / 'RESOURCE_EVIDENCE.json', {})
        with self.assertRaises(ValueError):
            driver.queue_audit(root, key, 0, producer)
        self.assertEqual([], list((root / 'audit_queue').iterdir()))
        driver.release_gpu(root, 0, producer)

    def test_pipeline_singleton_race_fifo_and_exact_gpu_release(self):
        root, plan = self._installed()
        jobs = [self._pipeline_run(root, plan, index=i, gpu=i)
                for i in range(2)]
        for gpu in (1, 0):
            key, producer, _ = jobs[gpu]
            driver.queue_audit(root, key, gpu, producer)
        auditor = self._owner(root, '', role='auditor')
        # The second ready job can pass a first job still holding its GPU.
        driver.release_gpu(root, 1, jobs[1][1])
        with ThreadPoolExecutor(max_workers=2) as executor:
            selected = list(executor.map(
                lambda _: driver.next_audit(root, 'formal', auditor), range(2)))
        self.assertEqual(1, selected.count(None))
        self.assertEqual(jobs[1][0], next(x for x in selected if x)['spec_key'])
        self._pipeline_record(root, plan, jobs[1][0])
        driver.complete_audit(root, jobs[1][0], auditor)
        driver.release_gpu(root, 0, jobs[0][1])
        self.assertEqual(jobs[0][0],
                         driver.next_audit(root, 'formal', auditor)['spec_key'])

    def test_pipeline_original_producer_can_exit_and_dead_auditor_can_cancel(self):
        root, plan = self._installed()
        process = subprocess.Popen([sys.executable, '-c',
                                    'import time; time.sleep(120)'])
        self.addCleanup(lambda: process.poll() is None and process.kill())
        producer = self._owner(root, '', role='child-producer', pid=process.pid)
        key, producer, run = self._pipeline_run(root, plan, producer=producer)
        before = (run / 'CLAIM_OWNER.json').read_bytes()
        driver.queue_audit(root, key, 0, producer)
        driver.release_gpu(root, 0, producer)
        process.terminate()
        process.wait(timeout=5)
        audit_process = subprocess.Popen([sys.executable, '-c',
                                          'import time; time.sleep(120)'])
        self.addCleanup(lambda: audit_process.poll() is None and audit_process.kill())
        auditor = self._owner(root, '', role='child-auditor', pid=audit_process.pid)
        driver.next_audit(root, 'formal', auditor)
        with self.assertRaises(ValueError):
            driver.cancel_audit(root, auditor)
        audit_process.terminate()
        audit_process.wait(timeout=5)
        with self.assertRaises(ValueError):
            driver.next_audit(root, 'formal', self._owner(root, '', role='next'))
        driver.install_marker(root, 'FAILED_JOB', {
            'kind': 'failed_audit', 'role': 'auditor', 'spec_key': key, 'exit_code': 1})
        with self.assertRaises(ValueError):
            driver.cancel_audit(root, {**auditor, 'launcher_token': 'foreign'})
        driver.cancel_audit(root, auditor)
        driver.cancel_audit(root, auditor)
        self.assertFalse((root / 'audit_queue' / 'active.json').exists())
        self.assertTrue((root / 'audit_queue' / f'{driver._claim_name(key)}.json').exists())
        self.assertEqual(before, (run / 'CLAIM_OWNER.json').read_bytes())
        for operation in (
                lambda: driver.next_audit(root, 'formal', self._owner(root, '')),
                lambda: driver.claim_next(plan, root / 'claims', 'formal',
                                          self._owner(root, ''), pipeline=True),
                lambda: driver.queue_audit(root, key, 0, self._owner(root, key)),
                lambda: driver.audit_phase_ready(root, 'formal')):
            with self.assertRaises(ValueError):
                operation()

    def test_pipeline_phase_and_final_success_require_drained_queue(self):
        root, plan = self._installed()
        key, producer, _ = self._pipeline_run(root, plan)
        auditor = self._owner(root, '', role='auditor')
        driver.queue_audit(root, key, 0, producer)
        driver.release_gpu(root, 0, producer)
        driver.next_audit(root, 'formal', auditor)
        self._pipeline_record(root, plan, key)
        for spec in (*formal_specs(), *explanation_specs()):
            if driver.spec_key(spec) == key:
                continue
            record = _completed_record(
                spec, .5, source_commit=driver._source_commit(),
                plan_sha256=driver._digest(plan))
            driver.install_json_exclusive(
                root / 'records' / f'{driver.safe_spec_name(spec)}.json', record)
        self.assertFalse(driver.audit_phase_ready(root, 'formal'))
        with self.assertRaises(ValueError):
            driver.claim_next(plan, root / 'claims', 'explanation',
                              self._owner(root, '', phase='explanation'),
                              pipeline=True)
        with self.assertRaises(ValueError):
            driver.finalize_installed(root)
        with self.assertRaises(ValueError):
            driver.install_marker(root, 'FORMAL_PHASE_SUCCESS', {
                'kind': 'formal_phase_success', 'role': 'parent',
                'spec_key': '', 'exit_code': 0})
        driver.complete_audit(root, key, auditor)
        self.assertTrue(driver.audit_phase_ready(root, 'formal'))
        self.assertTrue(driver.audit_phase_ready(root, 'explanation'))
        self.assertEqual(plan['explanation_cells'][0], driver.claim_next(
            plan, root / 'claims', 'explanation',
            self._owner(root, '', phase='explanation'), pipeline=True))

    def test_pipeline_active_and_requester_validation_fail_closed(self):
        root, plan = self._installed()
        key, producer, _ = self._pipeline_run(root, plan)
        auditor = self._owner(root, '', role='auditor')
        driver.queue_audit(root, key, 0, producer)
        driver.release_gpu(root, 0, producer)
        other_root, _ = self._installed(name='other-root')
        for changed in (
                {**auditor, 'job': key},
                {**auditor, 'phase': 'explanation'},
                {**auditor, 'root_identity': driver._root_identity(other_root)},
                {**auditor, 'source_commit': '0' * 40},
                {**auditor, 'process_start_time': 'wrong'},
                {**producer, 'job': ''}):
            with self.assertRaises(ValueError):
                driver.next_audit(root, 'formal', changed)
        driver.next_audit(root, 'formal', auditor)
        path = root / 'audit_queue' / 'active.json'
        active = driver._load_json_file(path)
        for changed in (
                {}, {**active, 'spec_key': plan['missing_jobs'][1]},
                {**active, 'handoff_sha256': '0' * 64},
                {**active, 'owner': {**auditor, 'process_start_time': 'wrong'}}):
            self._rewrite(path, changed)
            with self.assertRaises(ValueError):
                driver.next_audit(root, 'formal', auditor)
        self._rewrite(path, active)
        path.chmod(0o644)
        with self.assertRaises(ValueError):
            driver.next_audit(root, 'formal', auditor)
        path.chmod(0o444)
        driver.install_json_exclusive(root / 'FORMAL_STOPPED', {})
        with self.assertRaises(ValueError):
            driver.cancel_audit(root, auditor)
        self.assertTrue(path.exists())
        self._rewrite(root / 'FORMAL_STOPPED', {
            'kind': 'formal_stopped', 'role': 'parent',
            'spec_key': '', 'exit_code': 130})
        driver.cancel_audit(root, auditor)
        self.assertFalse(path.exists())

    def test_pipeline_queue_directory_symlink_rejects_without_touching_target(self):
        root, plan = self._installed()
        target = self.base / 'foreign-queue'
        target.mkdir()
        (root / 'audit_queue').symlink_to(target, target_is_directory=True)
        with self.assertRaises(ValueError):
            driver.claim_next(plan, root / 'claims', 'formal',
                              self._owner(root, ''), pipeline=True)
        with self.assertRaises(ValueError):
            driver.next_audit(root, 'formal', self._owner(root, ''))
        self.assertEqual([], list(target.iterdir()))

    def test_pipeline_cli_queue_contract(self):
        root, plan = self._installed()
        key, producer, run = self._pipeline_run(root, plan)
        auditor = self._owner(root, '', role='auditor')
        def cli(action, *options, owner=None):
            argv = [action, '--root', str(root), *options]
            if owner is not None:
                argv += ['--owner-json', _canonical(owner).decode()]
            with mock.patch('sys.stdout', io.StringIO()) as output:
                self.assertEqual(0, driver.main(argv))
            return json.loads(output.getvalue())
        cli('queue-audit', '--spec', key, '--physical-gpu', '0', owner=producer)
        self.assertIsNone(cli('next-audit', '--phase', 'formal', owner=auditor))
        driver.release_gpu(root, 0, producer)
        self.assertEqual({
            'spec_key': key, 'run_dir': str(run), 'physical_gpu': 0,
            'seed': driver.spec_for_key(key).seed},
            cli('next-audit', '--phase', 'formal', owner=auditor))
        self._pipeline_record(root, plan, key)
        self.assertEqual({'completed_audit': key},
                         cli('complete-audit', '--spec', key, owner=auditor))
        self.assertIs(False, cli('audit-phase-ready', '--phase', 'formal'))
        self.assertEqual({'cancelled_audit': True},
                         cli('cancel-audit', owner=auditor))

    def test_pipeline_cancel_rejects_nonobject_owner_without_dispatch(self):
        root, _ = self._installed()
        for invalid in (None, [], 'owner', {}):
            with self.subTest(owner=invalid), self.assertRaises(ValueError):
                driver.cancel_audit(root, invalid)

    def _pipeline_terminal_interleaving(self, action, marker_name):
        root, plan = self._installed(name=f'race-{action}-{marker_name}')
        key = plan['missing_jobs'][0]
        producer = None
        if action != 'claim':
            key, producer, _ = self._pipeline_run(root, plan)
        if action == 'next':
            driver.queue_audit(root, key, 0, producer)
            driver.release_gpu(root, 0, producer)
        else:
            self.assertFalse((root / 'audit_queue').exists())
        template = self._owner(root, '', role='auditor')
        payload = {
            'kind': ('failed_audit' if marker_name == 'FAILED_JOB'
                     else 'formal_stopped'),
            'role': 'parent', 'spec_key': key if marker_name == 'FAILED_JOB' else '',
            'exit_code': 1,
        }
        dispatch_checked = threading.Event()
        marker_attempted = threading.Event()
        marker_blocked = threading.Event()
        commits = []
        marker_thread = []
        real_running = driver._pipeline_running
        real_flock = driver.fcntl.flock
        real_install = driver.install_json_exclusive

        def running(path):
            real_running(path)
            dispatch_checked.set()
            self.assertTrue(marker_attempted.wait(20), 'marker did not attempt commit')

        def flock(fd, operation):
            if marker_thread == [threading.get_ident()] and operation == driver.fcntl.LOCK_EX:
                try:
                    real_flock(fd, operation | driver.fcntl.LOCK_NB)
                except BlockingIOError:
                    marker_blocked.set()
                    marker_attempted.set()
                    return real_flock(fd, operation)
                return None
            return real_flock(fd, operation)

        def install(path, value):
            result = real_install(path, value)
            if path == root / marker_name:
                commits.append('terminal')
                marker_attempted.set()
            elif ((action == 'claim' and path.name == 'owner.json')
                  or (action == 'queue' and path.parent.name == 'audit_queue')
                  or (action == 'next' and path.name == 'active.json')):
                commits.append('dispatch')
            return result

        def mark():
            marker_thread.append(threading.get_ident())
            return driver.install_marker(root, marker_name, payload)

        operations = {
            'claim': lambda: driver.claim_next(
                plan, root / 'claims', 'formal', template, pipeline=True),
            'queue': lambda: driver.queue_audit(root, key, 0, producer),
            'next': lambda: driver.next_audit(root, 'formal', template),
        }
        with mock.patch.object(driver, '_pipeline_running', side_effect=running), \
                mock.patch.object(driver.fcntl, 'flock', side_effect=flock), \
                mock.patch.object(driver, 'install_json_exclusive', side_effect=install), \
                ThreadPoolExecutor(max_workers=2) as executor:
            dispatch = executor.submit(operations[action])
            self.assertTrue(dispatch_checked.wait(20), 'dispatch never checked terminal state')
            terminal = executor.submit(mark)
            dispatch.result(timeout=30)
            terminal.result(timeout=30)
        self.assertEqual(['dispatch', 'terminal'], commits)
        self.assertTrue(marker_blocked.is_set())
        self.assertTrue(driver._pipeline_terminal(root))
        with self.assertRaises(ValueError):
            operations[action]()
        if action == 'queue':
            driver.release_gpu(root, 0, producer)

    def test_pipeline_terminal_commit_serializes_initial_claim(self):
        for name in ('FAILED_JOB', 'FORMAL_STOPPED'):
            with self.subTest(marker=name):
                self._pipeline_terminal_interleaving('claim', name)

    def test_pipeline_terminal_commit_serializes_queue_handoff(self):
        for name in ('FAILED_JOB', 'FORMAL_STOPPED'):
            with self.subTest(marker=name):
                self._pipeline_terminal_interleaving('queue', name)

    def test_pipeline_terminal_commit_serializes_next_audit(self):
        for name in ('FAILED_JOB', 'FORMAL_STOPPED'):
            with self.subTest(marker=name):
                self._pipeline_terminal_interleaving('next', name)

    def test_pipeline_terminal_marker_preserves_legacy_and_rejects_unsafe_claims(self):
        payload = {'kind': 'formal_stopped', 'role': 'parent',
                   'spec_key': '', 'exit_code': 1}
        legacy = driver.validate_formal_root(self.base / 'legacy-marker-only')
        driver.install_marker(legacy, 'FORMAL_STOPPED', payload)
        self.assertEqual({'FORMAL_STOPPED'}, {p.name for p in legacy.iterdir()})
        for kind in ('symlink', 'file'):
            root = driver.validate_formal_root(self.base / f'unsafe-marker-{kind}')
            target = self.base / f'foreign-{kind}'
            target.mkdir()
            if kind == 'symlink':
                (root / 'claims').symlink_to(target, target_is_directory=True)
            else:
                (root / 'claims').write_text('foreign')
            with self.assertRaises(ValueError):
                driver.install_marker(root, 'FORMAL_STOPPED', payload)
            self.assertFalse((root / 'FORMAL_STOPPED').exists())
            self.assertEqual([], list(target.iterdir()))

    @staticmethod
    def _pipeline_snapshot(root):
        return {str(path.relative_to(root)): (
            stat.S_IMODE(path.lstat().st_mode),
            path.read_bytes() if path.is_file() else None)
            for path in root.rglob('*')}

    def test_pipeline_launcher_nonce_rejects_foreign_pending_auditor(self):
        root, plan = self._installed()
        key, producer, _ = self._pipeline_run(root, plan)
        driver.queue_audit(root, key, 0, producer)
        foreign = self._owner(root, '', token='other-launch', role='auditor')
        for released in (False, True):
            if released:
                driver.release_gpu(root, 0, producer)
            before = self._pipeline_snapshot(root)
            with self.subTest(gpu_released=released), self.assertRaises(ValueError):
                driver.next_audit(root, 'formal', foreign)
            self.assertEqual(before, self._pipeline_snapshot(root))

    def test_pipeline_launcher_nonce_rejects_foreign_active_auditor(self):
        root, plan = self._installed()
        key, producer, _ = self._pipeline_run(root, plan)
        driver.queue_audit(root, key, 0, producer)
        driver.release_gpu(root, 0, producer)
        auditor = self._owner(root, '', role='auditor')
        driver.next_audit(root, 'formal', auditor)
        before = self._pipeline_snapshot(root)
        with self.assertRaises(ValueError):
            driver.next_audit(
                root, 'formal', {**auditor, 'launcher_token': 'other-launch'})
        self.assertEqual(before, self._pipeline_snapshot(root))

    def test_pipeline_launcher_nonce_rejects_cross_launcher_active_binding(self):
        root, plan = self._installed()
        key, producer, _ = self._pipeline_run(root, plan)
        driver.queue_audit(root, key, 0, producer)
        driver.release_gpu(root, 0, producer)
        auditor = self._owner(root, '', role='auditor')
        driver.next_audit(root, 'formal', auditor)
        path = root / 'audit_queue' / 'active.json'
        active = driver._load_json_file(path)
        self._rewrite(path, {
            **active, 'owner': {**auditor, 'launcher_token': 'other-launch'}})
        before = self._pipeline_snapshot(root)
        for operation in (
                lambda: driver.audit_phase_ready(root, 'formal'),
                lambda: driver.next_audit(root, 'formal', auditor),
                lambda: driver.cancel_audit(root, auditor)):
            with self.assertRaises(ValueError):
                operation()
            self.assertEqual(before, self._pipeline_snapshot(root))

    def test_pipeline_launcher_nonce_rejects_old_claims_before_none_or_capacity(self):
        for count, exact_job in ((1, False), (3, False), (1, True)):
            with self.subTest(claim_count=count, exact_job=exact_job):
                root, plan = self._installed(name=f'nonce-{count}-{exact_job}')
                old = self._owner(root, '', token='old-launch')
                for _ in range(count):
                    driver.claim_next(plan, root / 'claims', 'formal', old)
                requested = self._owner(
                    root, plan['missing_jobs'][0] if exact_job else '',
                    token='new-launch')
                before = self._pipeline_snapshot(root)
                with self.assertRaises(ValueError):
                    driver.claim_next(plan, root / 'claims', 'formal',
                                      requested, pipeline=True)
                self.assertEqual(before, self._pipeline_snapshot(root))
                with self.assertRaises(ValueError):
                    driver.next_audit(
                        root, 'formal', {**requested, 'job': '', 'worker_role': 'auditor'})
                self.assertEqual(before, self._pipeline_snapshot(root))
                # Default/legacy claims retain their previous caller contract.
                expected = None if exact_job else plan['missing_jobs'][count]
                self.assertEqual(expected, driver.claim_next(
                    plan, root / 'claims', 'formal', requested))

    def test_pipeline_launcher_nonce_rejects_old_claims_in_other_phase(self):
        for requested_phase in ('formal', 'explanation'):
            with self.subTest(phase=requested_phase):
                root, plan = self._installed(name=f'nonce-cross-{requested_phase}')
                old_phase = 'explanation' if requested_phase == 'formal' else 'formal'
                old_key = (plan['explanation_cells'][0] if old_phase == 'explanation'
                           else plan['missing_jobs'][0])
                claim = self._claim_path(root, old_key)
                claim.mkdir()
                driver.install_json_exclusive(claim / 'owner.json', self._owner(
                    root, old_key, token='old-launch', phase=old_phase))
                if requested_phase == 'explanation':
                    records = driver._ensure_directory(root, 'records')
                    commit = driver._source_commit()
                    for spec in formal_specs():
                        record = _completed_record(
                            spec, .5, source_commit=commit,
                            plan_sha256=driver._digest(plan))
                        driver.install_json_exclusive(
                            records / f'{driver.safe_spec_name(spec)}.json', record)
                requested = self._owner(
                    root, '', token='new-launch', phase=requested_phase)
                before = self._pipeline_snapshot(root)
                with self.assertRaises(ValueError):
                    driver.claim_next(plan, root / 'claims', requested_phase,
                                      requested, pipeline=True)
                self.assertEqual(before, self._pipeline_snapshot(root))
                with self.assertRaises(ValueError):
                    driver.next_audit(root, requested_phase, requested)
                self.assertEqual(before, self._pipeline_snapshot(root))

    def test_pipeline_complete_interruption_leaves_no_orphan_active(self):
        root, plan = self._installed()
        key, producer, _ = self._pipeline_run(root, plan)
        driver.queue_audit(root, key, 0, producer)
        driver.release_gpu(root, 0, producer)
        auditor = self._owner(root, '', role='auditor')
        driver.next_audit(root, 'formal', auditor)
        self._pipeline_record(root, plan, key)
        queue = root / 'audit_queue'
        handoff = queue / f'{driver._claim_name(key)}.json'
        record = root / 'records' / f'{driver._claim_name(key)}.json'
        before = {path: path.read_bytes() for path in (handoff, record)}
        queue_identity = (queue.stat().st_dev, queue.stat().st_ino)
        real_fsync = driver.os.fsync
        interrupted = []

        def fsync(fd):
            real_fsync(fd)
            details = os.fstat(fd)
            if (details.st_dev, details.st_ino) == queue_identity and not interrupted:
                interrupted.append(True)
                raise RuntimeError('controlled interruption after first queue fsync')

        with mock.patch.object(driver.os, 'fsync', side_effect=fsync):
            with self.assertRaisesRegex(RuntimeError, 'controlled interruption'):
                driver.complete_audit(root, key, auditor)
        self.assertEqual([True], interrupted)
        self.assertFalse((queue / 'active.json').exists())
        for path, content in before.items():
            self.assertEqual(content, path.read_bytes())
        with self.assertRaises(ValueError):
            driver.next_audit(root, 'formal', auditor)
        self.assertFalse(driver.audit_phase_ready(root, 'formal'))
        driver.install_marker(root, 'FORMAL_STOPPED', {
            'kind': 'formal_stopped', 'role': 'parent',
            'spec_key': '', 'exit_code': 1})
        driver.cancel_audit(root, auditor)
        self.assertFalse((queue / 'active.json').exists())
        for path, content in before.items():
            self.assertEqual(content, path.read_bytes())

    def _pipeline_dead_gpu(self, name):
        root, plan = self._installed(name=name)
        process = subprocess.Popen([sys.executable, '-c',
                                    'import time; time.sleep(120)'])
        self.addCleanup(lambda: process.poll() is None and process.kill())
        owner = self._owner(root, '', role='producer', pid=process.pid)
        key, owner, _ = self._pipeline_run(root, plan, producer=owner)
        driver._ensure_directory(root, 'audit_queue')
        process.kill()
        process.wait(timeout=5)
        return root, plan, key, owner

    def test_pipeline_gpu_cleanup_dead_owner_requires_valid_terminal(self):
        for marker in ('FAILED_JOB', 'FORMAL_STOPPED'):
            with self.subTest(marker=marker):
                root, plan, key, owner = self._pipeline_dead_gpu(f'dead-gpu-{marker}')
                before = self._pipeline_snapshot(root)
                for operation in (
                        lambda: driver.release_gpu(root, 0, owner),
                        lambda: driver.claim_gpu(root, 1, owner),
                        lambda: driver.queue_audit(root, key, 0, owner)):
                    with self.assertRaises(ValueError):
                        operation()
                    self.assertEqual(before, self._pipeline_snapshot(root))
                driver.install_json_exclusive(root / marker, {})
                before = self._pipeline_snapshot(root)
                with self.assertRaises(ValueError):
                    driver.release_gpu(root, 0, owner)
                self.assertEqual(before, self._pipeline_snapshot(root))
                self._rewrite(root / marker, {
                    'kind': 'failed_audit' if marker == 'FAILED_JOB' else 'formal_stopped',
                    'role': 'parent', 'spec_key': key if marker == 'FAILED_JOB' else '',
                    'exit_code': 1})
                before = self._pipeline_snapshot(root)
                driver.release_gpu(root, 0, owner)
                self.assertEqual({
                    name: value for name, value in before.items()
                    if name != 'gpu_claims/gpu-0' and not name.startswith('gpu_claims/gpu-0/')
                }, self._pipeline_snapshot(root))
                with self.assertRaises(ValueError):
                    driver.claim_gpu(root, 0, owner)

    def test_pipeline_gpu_cleanup_rejects_foreign_identity_and_reused_slot(self):
        root, plan, key, owner = self._pipeline_dead_gpu('dead-gpu-identity')
        driver.install_marker(root, 'FAILED_JOB', {
            'kind': 'failed_audit', 'role': 'parent', 'spec_key': key, 'exit_code': 1})
        other_root, _ = self._installed(name='dead-gpu-other-root')
        changes = (
            ('launcher_token', 'foreign-launch'),
            ('process_start_time', owner['process_start_time'] + '1'),
            ('pid', owner['pid'] + 1), ('pgid', owner['pgid'] + 1),
            ('job', plan['missing_jobs'][1]), ('source_commit', '0' * 40),
            ('root_identity', driver._root_identity(other_root)))
        for field, value in changes:
            before = self._pipeline_snapshot(root)
            with self.subTest(field=field), self.assertRaises(ValueError):
                driver.release_gpu(root, 0, {**owner, field: value})
            self.assertEqual(before, self._pipeline_snapshot(root))
        owner_path = root / 'gpu_claims' / 'gpu-0' / 'owner.json'
        original = driver._load_json_file(owner_path)
        # Even matching GPU JSON cannot authorize a different root/source.
        for field, value in changes[-2:]:
            changed = {**owner, field: value}
            self._rewrite(owner_path, {
                **changed, 'kind': 'formal_gpu_claim', 'physical_gpu': 0})
            before = self._pipeline_snapshot(root)
            with self.assertRaises(ValueError):
                driver.release_gpu(root, 0, changed)
            self.assertEqual(before, self._pipeline_snapshot(root))
        self._rewrite(owner_path, original)
        (root / 'gpu_claims' / 'gpu-0').rename(root / 'retained-old-gpu-claim')
        replacement = self._owner(root, plan['missing_jobs'][1],
                                  token='replacement-launch', role='replacement')
        self.assertTrue(driver.claim_gpu(root, 0, replacement))
        before = self._pipeline_snapshot(root)
        with self.assertRaises(ValueError):
            driver.release_gpu(root, 0, owner)
        self.assertEqual(before, self._pipeline_snapshot(root))
        driver.release_gpu(root, 0, replacement)

    def test_pipeline_gpu_cleanup_does_not_treat_live_identity_mismatch_as_dead(self):
        root, plan = self._installed()
        owner = self._owner(root, plan['missing_jobs'][0])
        self.assertTrue(driver.claim_gpu(root, 0, owner))
        driver.release_gpu(root, 0, owner)
        self.assertFalse((root / 'gpu_claims' / 'gpu-0').exists())
        self.assertTrue(driver.claim_gpu(root, 0, owner))
        driver.install_marker(root, 'FORMAL_STOPPED', {
            'kind': 'formal_stopped', 'role': 'parent', 'spec_key': '', 'exit_code': 1})
        owner_path = root / 'gpu_claims' / 'gpu-0' / 'owner.json'
        original = driver._load_json_file(owner_path)
        changed = {**owner, 'pgid': owner['pgid'] + 1}
        self._rewrite(owner_path, {
            **changed, 'kind': 'formal_gpu_claim', 'physical_gpu': 0})
        before = self._pipeline_snapshot(root)
        with self.assertRaises(ValueError):
            driver.release_gpu(root, 0, changed)
        self.assertEqual(before, self._pipeline_snapshot(root))
        self._rewrite(owner_path, original)
        driver.release_gpu(root, 0, owner)

    def test_exact_81_and_6_plan_has_79_missing_for_two_reusable(self):
        specs = formal_specs()
        declarations = {
            driver.spec_key(specs[0]): {'marker': 'first'},
            driver.spec_key(specs[-1]): {'marker': 'last'},
        }

        def audit(spec, declaration):
            self.assertIs(declarations[driver.spec_key(spec)], declaration)
            return _admission(spec)

        with mock.patch.object(driver, 'audit_candidate', side_effect=audit):
            census = driver.build_census(declarations)
        plan = driver.build_plan(census)
        formal = [driver.spec_key(spec) for spec in specs]
        explanations = [driver.spec_key(spec) for spec in explanation_specs()]
        self.assertEqual(81, len(census['records']))
        self.assertEqual(formal, plan['formal_cells'])
        self.assertEqual(explanations, plan['explanation_cells'])
        self.assertEqual(79, len(plan['missing_jobs']))
        self.assertNotIn(formal[0], plan['missing_jobs'])
        self.assertNotIn(formal[-1], plan['missing_jobs'])
        self.assertEqual(
            {'kind', 'registry_sha256', 'metric_formula_version', 'records'},
            set(census),
        )
        self.assertEqual({
            'kind', 'registry_sha256', 'metric_formula_version',
            'formal_cells', 'missing_jobs', 'explanation_cells',
            'census_sha256',
        }, set(plan))

    def test_metric_changes_do_not_change_missing_membership(self):
        reusable = formal_specs()[:2]
        first = driver.build_plan(self._census(reusable, metric=.25))
        second = driver.build_plan(self._census(reusable, metric=.75))
        self.assertNotEqual(first['census_sha256'], second['census_sha256'])
        self.assertEqual(first['missing_jobs'], second['missing_jobs'])

    def test_zero_declarations_is_deterministic_without_discovery(self):
        with mock.patch.object(driver, 'audit_candidate') as audit, \
                mock.patch.object(Path, 'rglob', side_effect=AssertionError), \
                mock.patch.object(Path, 'glob', side_effect=AssertionError), \
                mock.patch.object(os, 'walk', side_effect=AssertionError):
            first = driver.build_census({})
            second = driver.build_census({})
        audit.assert_not_called()
        self.assertEqual(_canonical(first), _canonical(second))
        self.assertEqual(81, len(first['records']))
        self.assertTrue(all(
            record['status'] == 'RERUN_REQUIRED'
            and record['reason'] == 'no_declared_candidate'
            for record in first['records']
        ))

    def test_declared_temporary_candidate_routes_through_real_auditor(self):
        import three_dataset_formal_audit as audit_module
        from test_three_dataset_formal_audit import (
            PRODUCER_SOURCES, RealProducerFixture,
        )

        fixture = RealProducerFixture('finetune')
        self.addCleanup(fixture.close)
        key = driver.spec_key(fixture.spec)
        with mock.patch.object(
                audit_module, '_SOURCE_INVENTORY', PRODUCER_SOURCES), \
                mock.patch.object(
                    audit_module, '_AUTHORITATIVE_DATA',
                    {fixture.spec.dataset: fixture.authoritative_data()}), \
                mock.patch.object(
                    audit_module, '_AUTHORITATIVE_MANIFEST',
                    {fixture.spec.dataset: fixture.authoritative_manifest()}):
            census = driver.build_census({key: fixture.declaration})
        record = next(item for item in census['records']
                      if item['spec_key'] == key)
        self.assertEqual('REUSABLE', record['status'])
        self.assertEqual('admitted', record['reason'])
        self.assertEqual(FORMULA_VERSION, record['metric_formula_version'])

    def test_declaration_and_audit_schemas_fail_closed(self):
        formal_key = driver.spec_key(formal_specs()[0])
        invalid = (
            [],
            {driver.spec_key(explanation_specs()[0]): {}},
            {'tinyimagenet:finetune:42': {}},
            {formal_key: []},
        )
        for value in invalid:
            with self.subTest(value=repr(value)):
                with self.assertRaises((TypeError, ValueError)):
                    driver.build_census(value)
        with mock.patch.object(driver, 'audit_candidate', return_value={}):
            with self.assertRaises(ValueError):
                driver.build_census({formal_key: {}})
        with mock.patch.object(
                driver, 'audit_candidate',
                return_value=replace(_admission(formal_specs()[0]),
                                     metric_formula_version='tampered')):
            with self.assertRaises(ValueError):
                driver.build_census({formal_key: {}})

    def test_reusable_metrics_require_exact_finite_formula_projection(self):
        spec = formal_specs()[0]
        key = driver.spec_key(spec)
        valid = _admission(spec)
        count = protocol_for(spec)['base_options']['num_tasks']
        malformed = (
            replace(valid, metrics={}),
            replace(valid, metrics=replace(
                valid.metrics, class_final=(.5,) * (count - 1))),
            replace(valid, metrics=replace(
                valid.metrics, aa_trajectory=(.5,) * (count - 1))),
            replace(valid, metrics=replace(valid.metrics, aa_final=1)),
            replace(valid, metrics=replace(valid.metrics, bwt=math.nan)),
            replace(valid, metrics=replace(
                valid.metrics, aa_final=.6,
                class_final=(.5,) * count)),
            replace(valid, metrics=replace(
                valid.metrics, taskil_final=.6,
                taskil_final_by_task=(.5,) * count)),
            replace(valid, metrics=replace(
                valid.metrics,
                aa_trajectory=(.5,) * (count - 1) + (.6,))),
        )
        for record in malformed:
            with self.subTest(metrics=record.metrics):
                with mock.patch.object(
                        driver, 'audit_candidate', return_value=record):
                    with self.assertRaises(ValueError):
                        driver.build_census({key: {}})

    def test_installed_reusable_metrics_are_revalidated_not_self_attested(self):
        spec = formal_specs()[0]
        census = self._census((spec,))
        record = census['records'][0]
        record['metrics'] = {}
        raw = {key: record[key] for key in driver._ADMISSION_KEYS}
        record['admission_record_sha256'] = driver._digest(raw)
        with self.assertRaises(ValueError):
            driver.build_plan(census)

    def test_duplicate_unknown_missing_and_provenance_tampering_reject(self):
        census = driver.build_census({})
        variants = []
        duplicate = copy.deepcopy(census)
        duplicate['records'][-1] = copy.deepcopy(duplicate['records'][0])
        variants.append(duplicate)
        missing = copy.deepcopy(census)
        missing['records'].pop()
        variants.append(missing)
        unknown = copy.deepcopy(census)
        unknown['records'][0]['spec_key'] = 'tinyimagenet:finetune:42'
        variants.append(unknown)
        registry = copy.deepcopy(census)
        registry['registry_sha256'] = '0' * 64
        variants.append(registry)
        formula = copy.deepcopy(census)
        formula['metric_formula_version'] = 'tampered'
        variants.append(formula)
        admission = copy.deepcopy(census)
        admission['records'][0]['reason'] = 'tampered'
        variants.append(admission)
        for value in variants:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    driver.build_plan(value)

    def test_repeated_rebuilds_are_byte_identical(self):
        first = driver.build_census({})
        second = driver.build_census({})
        self.assertEqual(_canonical(first), _canonical(second))
        self.assertEqual(
            _canonical(driver.build_plan(first)),
            _canonical(driver.build_plan(second)),
        )

    def test_exact_81_plus_6_summary_uses_sample_std_and_registry_order(self):
        records = _all_completed_records()
        summary = driver.summarize_records(records)
        self.assertEqual('formal_completed_summary', summary['kind'])
        self.assertNotIn('pilot_rows', summary)
        self.assertEqual(87, len(summary['per_run_records']))
        self.assertEqual(27, len(summary['formal_rows']))
        self.assertEqual(27, len(summary['resource_rows']))
        self.assertEqual(18, len(summary['mechanism_rows']))
        first = summary['formal_rows'][0]
        self.assertEqual(
            (formal_specs()[0].dataset, formal_specs()[0].method),
            (first['dataset'], first['method']),
        )
        for name in ('aa_final', 'bwt', 'taskil_final'):
            self.assertAlmostEqual(.3, first[f'{name}_mean'])
            self.assertAlmostEqual(.1, first[f'{name}_std'])
        self.assertEqual(
            [driver.spec_key(spec)
             for spec in (*formal_specs(), *explanation_specs())],
            [record['spec_key'] for record in summary['per_run_records']],
        )

    def test_summary_rejects_membership_identity_resource_and_metric_tamper(self):
        records = _all_completed_records()
        variants = []
        variants.append(('missing-seed', records[:-7] + records[-6:]))
        variants.append(('duplicate', records + [copy.deepcopy(records[0])]))
        partial = records[:-1]
        variants.append(('partial-explanation', partial))
        for name, field, value in (
                ('registry', 'registry_sha256', '0' * 64),
                ('formula', 'metric_formula_version', 'tampered'),
                ('source', 'source_commit', '0' * 40),
                ('plan', 'plan_sha256', '0' * 64),
                ('protocol', 'protocol_sha256', '0' * 64)):
            changed = copy.deepcopy(records)
            changed[0][field] = value
            changed[0]['record_sha256'] = driver._digest({
                key: value for key, value in changed[0].items()
                if key != 'record_sha256'
            })
            variants.append((name, changed))
        unknown = copy.deepcopy(records)
        unknown[0]['method'] = 'unknown'
        unknown[0]['record_sha256'] = driver._digest({
            key: value for key, value in unknown[0].items()
            if key != 'record_sha256'
        })
        variants.append(('unknown', unknown))
        mixed_hardware = copy.deepcopy(records)
        mixed_hardware[3]['resource']['hardware_identity']['gpu_name'] = 'other'
        mixed_hardware[3]['record_sha256'] = driver._digest({
            key: value for key, value in mixed_hardware[3].items()
            if key != 'record_sha256'
        })
        variants.append(('mixed-hardware', mixed_hardware))
        mixed_instrumentation = copy.deepcopy(records)
        mixed_instrumentation[3]['resource']['instrumentation'] = 'other'
        mixed_instrumentation[3]['record_sha256'] = driver._digest({
            key: value for key, value in mixed_instrumentation[3].items()
            if key != 'record_sha256'
        })
        variants.append(('mixed-instrumentation', mixed_instrumentation))
        malformed_resource = copy.deepcopy(records)
        malformed_resource[0]['resource']['peak_gpu_memory_bytes'] = True
        malformed_resource[0]['record_sha256'] = '8' * 64
        variants.append(('malformed-resource', malformed_resource))
        nonfinite = copy.deepcopy(records)
        nonfinite[0]['metrics']['aa_final'] = math.inf
        nonfinite[0]['record_sha256'] = '8' * 64
        variants.append(('nonfinite', nonfinite))
        malformed_hash = copy.deepcopy(records)
        malformed_hash[0]['command_sha256'] = 'not-a-hash'
        malformed_hash[0]['record_sha256'] = '8' * 64
        variants.append(('malformed-hash', malformed_hash))
        for name, value in variants:
            with self.subTest(name=name):
                with self.assertRaises((TypeError, ValueError)):
                    driver.summarize_records(value)

    def test_rendered_tables_are_deterministic_exact_csv_and_metric_independent(self):
        records = _all_completed_records()
        first = driver.render_tables(records)
        second = driver.render_tables(list(reversed(records)))
        self.assertEqual(first, second)
        self.assertEqual({
            'FORMAL_PER_RUN.csv', 'FORMAL_TABLE.csv',
            'RESOURCE_PRIVACY_TABLE.csv', 'MECHANISM_TABLE.csv',
        }, set(first))
        expected_rows = {
            'FORMAL_PER_RUN.csv': 88,
            'FORMAL_TABLE.csv': 28,
            'RESOURCE_PRIVACY_TABLE.csv': 28,
            'MECHANISM_TABLE.csv': 19,
        }
        for name, count in expected_rows.items():
            with self.subTest(name=name):
                payload = first[name]
                self.assertEqual(payload, payload.decode('utf-8').encode('utf-8'))
                self.assertTrue(payload.endswith(b'\n'))
                self.assertNotIn(b'\r', payload)
                self.assertEqual(count, len(list(csv.reader(
                    io.StringIO(payload.decode('utf-8'))))))
        formal_rows = list(csv.DictReader(io.StringIO(
            first['FORMAL_TABLE.csv'].decode('utf-8'))))
        self.assertEqual('0.300000', formal_rows[0]['aa_final_mean'])
        self.assertEqual('0.100000', formal_rows[0]['aa_final_std'])
        self.assertEqual(
            [(spec.dataset, spec.method)
             for spec in formal_specs()[::3]],
            [(row['dataset'], row['method']) for row in formal_rows],
        )
        changed = copy.deepcopy(records)
        for record in changed:
            record['metrics']['bwt'] += .01
            record['record_sha256'] = driver._digest({
                key: value for key, value in record.items()
                if key != 'record_sha256'
            })
        changed_rows = list(csv.DictReader(io.StringIO(
            driver.render_tables(changed)['FORMAL_TABLE.csv'].decode('utf-8'))))
        self.assertEqual(
            [(row['dataset'], row['method']) for row in formal_rows],
            [(row['dataset'], row['method']) for row in changed_rows],
        )
        self.assertEqual(
            (['formal'] * 4 + ['explanation'] * 2) * 3,
            [row['explanation_scope'] for row in csv.DictReader(io.StringIO(
                first['MECHANISM_TABLE.csv'].decode('utf-8')))],
        )

    def test_install_tables_is_exclusive_atomic_and_never_partial(self):
        records = _all_completed_records()
        destination = self.base / 'tables'
        destination.mkdir()
        rendered = driver.render_tables(records)
        manifest = driver.install_tables(records, destination)
        self.assertEqual(
            {name: hashlib.sha256(payload).hexdigest()
             for name, payload in rendered.items()},
            manifest,
        )
        before = {name: (destination / name).read_bytes() for name in rendered}
        for name, payload in before.items():
            self.assertEqual(rendered[name], payload)
            self.assertEqual(0o444, stat.S_IMODE((destination / name).stat().st_mode))
        with self.assertRaises(FileExistsError):
            driver.install_tables(records, destination)
        self.assertEqual(
            before,
            {name: (destination / name).read_bytes() for name in rendered},
        )

        partial = self.base / 'partial-tables'
        partial.mkdir()
        existing = partial / 'FORMAL_TABLE.csv'
        existing.write_bytes(b'existing\n')
        with self.assertRaises(FileExistsError):
            driver.install_tables(records, partial)
        self.assertEqual({'FORMAL_TABLE.csv'}, {path.name for path in partial.iterdir()})
        self.assertEqual(b'existing\n', existing.read_bytes())

    def test_root_symlink_escape_existing_output_and_inode_swap_reject(self):
        real = self.base / 'real'
        real.mkdir()
        alias = self.base / 'alias'
        alias.symlink_to(real, target_is_directory=True)
        with self.assertRaises(ValueError):
            driver.validate_formal_root(alias / 'formal')
        direct = driver.validate_formal_root(self.base / 'direct-root')
        direct_alias = self.base / 'direct-root-alias'
        direct_alias.symlink_to(direct, target_is_directory=True)
        with self.assertRaises(ValueError):
            driver.validate_formal_root(direct_alias)
        with self.assertRaises(ValueError):
            driver.validate_formal_root(self.base / 'real' / '..' / 'escape')
        regular = self.base / 'regular'
        regular.write_text('not a directory')
        with self.assertRaises(ValueError):
            driver.validate_formal_root(regular)

        root = driver.validate_formal_root(self.base / 'formal')
        destination = root / 'payload.json'
        driver.install_json_exclusive(destination, {'value': 1})
        expected = destination.read_bytes()
        with self.assertRaises(FileExistsError):
            driver.install_json_exclusive(destination, {'value': 2})
        self.assertEqual(expected, destination.read_bytes())

        swapped = root / 'swapped.json'
        replacement = b'{"replacement":true}\n'
        original = driver._read_descriptor

        def swap_after_read(descriptor):
            result = original(descriptor)
            swapped.unlink()
            swapped.write_bytes(replacement)
            swapped.chmod(0o444)
            return result

        with mock.patch.object(
                driver, '_read_descriptor', side_effect=swap_after_read):
            with self.assertRaises(RuntimeError):
                driver.install_json_exclusive(swapped, {'value': 3})
        self.assertEqual(replacement, swapped.read_bytes())

    def test_installed_files_are_canonical_read_only_and_hash_bound(self):
        root, plan = self._installed()
        for name in (
                'FORMAL_REGISTRY.json', 'COMPATIBILITY_CENSUS.json',
                'FORMAL_PLAN.json', 'MISSING_JOBS.json',
                'FORMAL_ROOT_IDENTITY.json'):
            path = root / name
            value = json.loads(path.read_text())
            self.assertEqual(_canonical(value) + b'\n', path.read_bytes())
            self.assertEqual(0o444, stat.S_IMODE(path.stat().st_mode))
        manifest = root / 'MISSING_JOBS.json'
        manifest.chmod(0o644)
        value = json.loads(manifest.read_text())
        value['plan_sha256'] = '0' * 64
        manifest.write_bytes(_canonical(value) + b'\n')
        manifest.chmod(0o444)
        with self.assertRaises(ValueError):
            driver._installed_jobs(root)
        self.assertEqual(81, len(plan['missing_jobs']))

    def test_installed_bundle_rejects_self_hashed_fabricated_authority(self):
        root, plan = self._installed(name='fabricated')
        registry = json.loads((root / 'FORMAL_REGISTRY.json').read_text())
        census = json.loads((root / 'COMPATIBILITY_CENSUS.json').read_text())
        registry['registry_sha256'] = '0' * 64
        plan['registry_sha256'] = '0' * 64
        plan['census_sha256'] = '0' * 64
        plan['missing_jobs'] = plan['missing_jobs'][1:]
        manifest = {
            'kind': 'formal_missing_jobs',
            'registry_sha256': plan['registry_sha256'],
            'metric_formula_version': plan['metric_formula_version'],
            'missing_jobs': list(plan['missing_jobs']),
            'plan_sha256': driver._digest(plan),
        }
        self._rewrite(root / 'FORMAL_REGISTRY.json', registry)
        self._rewrite(root / 'FORMAL_PLAN.json', plan)
        self._rewrite(root / 'MISSING_JOBS.json', manifest)
        self.assertEqual(81, len(census['records']))
        with self.assertRaises(ValueError):
            driver._installed_jobs(root)

    def test_installed_bundle_requires_exact_ordered_registry_membership(self):
        root, plan = self._installed(name='membership')
        plan['formal_cells'][0], plan['formal_cells'][1] = (
            plan['formal_cells'][1], plan['formal_cells'][0])
        plan['missing_jobs'] = list(plan['formal_cells'])
        plan['explanation_cells'].reverse()
        manifest = {
            'kind': 'formal_missing_jobs',
            'registry_sha256': plan['registry_sha256'],
            'metric_formula_version': plan['metric_formula_version'],
            'missing_jobs': list(plan['missing_jobs']),
            'plan_sha256': driver._digest(plan),
        }
        self._rewrite(root / 'FORMAL_PLAN.json', plan)
        self._rewrite(root / 'MISSING_JOBS.json', manifest)
        with self.assertRaises(ValueError):
            driver._installed_jobs(root)

    def test_jobs_cross_binds_all_four_installed_authorities(self):
        root, plan = self._installed()
        self.assertEqual(plan['missing_jobs'], driver._installed_jobs(root))
        census = root / 'COMPATIBILITY_CENSUS.json'
        value = json.loads(census.read_text())
        value['records'][0]['reason'] = 'tampered'
        self._rewrite(census, value)
        with self.assertRaises(ValueError):
            driver._installed_jobs(root)

    def test_malicious_job_paths_never_create_outside_claims(self):
        for index, job in enumerate((
                '../../traversal-escape', str(self.base / 'absolute-escape'))):
            with self.subTest(job=job):
                root, plan = self._installed(name=f'unsafe-{index}')
                owner = self._owner(root, plan['missing_jobs'][0])
                plan['formal_cells'][0] = job
                plan['missing_jobs'] = [job]
                manifest = {
                    'kind': 'formal_missing_jobs',
                    'registry_sha256': plan['registry_sha256'],
                    'metric_formula_version': plan['metric_formula_version'],
                    'missing_jobs': [job],
                    'plan_sha256': driver._digest(plan),
                }
                self._rewrite(root / 'FORMAL_PLAN.json', plan)
                self._rewrite(root / 'MISSING_JOBS.json', manifest)
                outside = ((root / 'claims' / job).resolve()
                           if not Path(job).is_absolute() else Path(job))
                owner['job'] = job
                owner['root_identity'] = self._identity_for(
                    root, plan, manifest)
                with self.assertRaises(ValueError):
                    driver.claim_next(plan, root / 'claims', 'formal', owner)
                self.assertFalse(outside.exists())

    def test_two_owners_cannot_claim_one_job_and_exact_owner_releases(self):
        specs = formal_specs()
        census = self._census(specs[:-1])
        root, plan = self._installed(census)
        job = plan['missing_jobs'][0]
        owners = [self._owner(root, job, token) for token in ('one', 'two')]
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(
                lambda owner: driver.claim_next(
                    plan, root / 'claims', 'formal', owner), owners))
        self.assertEqual(1, results.count(job))
        self.assertEqual(1, results.count(None))
        installed = json.loads(
            (self._claim_path(root, job) / 'owner.json').read_text())
        winner = next(owner for owner in owners if owner == installed)
        loser = next(owner for owner in owners if owner != installed)
        with self.assertRaises(ValueError):
            driver.release_prelaunch_claim(self._claim_path(root, job), loser)
        driver.release_prelaunch_claim(self._claim_path(root, job), winner)
        self.assertFalse(self._claim_path(root, job).exists())

    def test_started_and_changed_owner_claims_remain_evidence(self):
        specs = formal_specs()
        root, plan = self._installed(self._census(specs[:-1]))
        job = plan['missing_jobs'][0]
        owner = self._owner(root, job)
        self.assertEqual(
            job, driver.claim_next(plan, root / 'claims', 'formal', owner))
        claim = self._claim_path(root, job)
        for name, value in (
                ('pid', owner['pid'] + 1),
                ('pgid', owner['pgid'] + 1),
                ('phase', 'explanation'),
                ('process_start_time', owner['process_start_time'] + 'x'),
                ('source_commit', '0' * 40),
                ('root_identity', {**owner['root_identity'], 'hash': '0' * 64}),
        ):
            changed = copy.deepcopy(owner)
            changed[name] = value
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    driver.release_prelaunch_claim(claim, changed)
                self.assertTrue(claim.exists())
        driver.install_json_exclusive(claim / 'started.json', {'started': True})
        with self.assertRaises(ValueError):
            driver.release_prelaunch_claim(claim, owner)
        self.assertTrue(claim.exists())

    def test_incomplete_malformed_and_symlink_claims_are_never_stolen(self):
        specs = formal_specs()
        root, plan = self._installed(self._census(specs[:-1]))
        job = plan['missing_jobs'][0]
        claim = self._claim_path(root, job)
        claim.mkdir()
        owner = self._owner(root, job, 'other')
        with self.assertRaises(ValueError):
            driver.claim_next(plan, root / 'claims', 'formal', owner)
        self.assertTrue(claim.exists())
        claim.rmdir()

        target = self.base / 'claim-target'
        target.mkdir()
        claim.symlink_to(target, target_is_directory=True)
        with self.assertRaises(ValueError):
            driver.claim_next(plan, root / 'claims', 'formal', owner)
        self.assertTrue(claim.is_symlink())

    def test_dead_requesters_reject_but_live_and_stale_claims_are_preserved(self):
        specs = formal_specs()
        root, plan = self._installed(self._census(specs[:-1]), 'liveness')
        job = plan['missing_jobs'][0]
        process = subprocess.Popen([
            sys.executable, '-c', 'import time; time.sleep(2)',
        ])
        self.addCleanup(lambda: process.poll() is None and process.kill())
        start = driver._process_start_time(process.pid)
        stale_owner = self._owner(
            root, job, token='stale', pid=process.pid, start=start)
        self.assertEqual(
            job, driver.claim_next(
                plan, root / 'claims', 'formal', stale_owner))
        process.terminate()
        process.wait(timeout=5)
        live_owner = self._owner(root, job, token='live')
        self.assertIsNone(driver.claim_next(
            plan, root / 'claims', 'formal', live_owner))
        self.assertEqual(
            stale_owner,
            json.loads((self._claim_path(root, job) / 'owner.json').read_text()),
        )

        driver.release_prelaunch_claim(self._claim_path(root, job), stale_owner)
        dead_requester = copy.deepcopy(stale_owner)
        with self.assertRaises(ValueError):
            driver.claim_next(
                plan, root / 'claims', 'formal', dead_requester)
        self.assertFalse(self._claim_path(root, job).exists())

    def test_cli_control_plane_actions_never_launch_or_train(self):
        root = self.base / 'cli-formal'
        declarations = self.base / 'declarations.json'
        declarations.write_text('{}')
        stdout = io.StringIO()
        source_commit = driver._source_commit()
        with mock.patch('sys.stdout', stdout), \
                mock.patch.object(driver, 'audit_candidate') as audit, \
                mock.patch.object(
                    driver, '_source_commit', return_value=source_commit), \
                mock.patch.object(driver.os, 'system') as system, \
                mock.patch.object(driver.subprocess, 'Popen') as popen:
            with self.assertRaises(ValueError):
                driver.main(['check', '--root', str(root)])
            self.assertFalse(root.exists())
            self.assertEqual(0, driver.main([
                'census', '--root', str(root),
                '--declarations', str(declarations),
            ]))
            self.assertEqual(0, driver.main(['plan', '--root', str(root)]))
            self.assertEqual(0, driver.main(['check', '--root', str(root)]))
            self.assertEqual(0, driver.main(['jobs', '--root', str(root)]))
            plan = json.loads((root / 'FORMAL_PLAN.json').read_text())
            owner = self._owner(root, plan['missing_jobs'][0])
            self.assertEqual(0, driver.main([
                'claim', '--root', str(root), '--phase', 'formal',
                '--owner-json', _canonical(owner).decode(),
            ]))
        audit.assert_not_called()
        system.assert_not_called()
        popen.assert_not_called()
        self.assertEqual({
            'FORMAL_REGISTRY.json', 'COMPATIBILITY_CENSUS.json',
            'FORMAL_PLAN.json', 'MISSING_JOBS.json',
            'FORMAL_ROOT_IDENTITY.json', 'claims',
        }, {path.name for path in root.iterdir()})
        self.assertTrue(self._claim_path(root, plan['missing_jobs'][0]).is_dir())
        records_path = self.base / 'records.json'
        records_path.write_text('[]')
        tables = self.base / 'cli-tables'
        completed = _completed_record(formal_specs()[0], .5)
        with mock.patch('sys.stdout', io.StringIO()), mock.patch.object(
                driver, 'completed_run_record', return_value=completed) as audit_run, \
                mock.patch.object(
                    driver, '_source_commit', return_value=source_commit), \
                mock.patch.object(
                    driver, 'install_tables', return_value={'table': '8' * 64}
                ) as summarize, mock.patch.object(
                    driver.os, 'system') as system, mock.patch.object(
                    driver.subprocess, 'Popen') as popen:
            self.assertEqual(0, driver.main([
                'audit-run', '--root', str(root),
                '--spec-key', driver.spec_key(formal_specs()[0]),
                '--run-dir', str(self.base / 'completed-run'),
            ]))
            self.assertEqual(0, driver.main([
                'summarize', '--records', str(records_path),
                '--destination', str(tables),
            ]))
        audit_run.assert_called_once()
        summarize.assert_called_once_with([], tables)
        system.assert_not_called()
        popen.assert_not_called()

    def test_strict_json_rejects_duplicates_and_nonfinite_numbers(self):
        duplicate = self.base / 'duplicate.json'
        duplicate.write_text('{"a":1,"a":2}')
        nonfinite = self.base / 'nonfinite.json'
        nonfinite.write_text('{"a":NaN}')
        for path in (duplicate, nonfinite):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    driver._load_json_file(path)

    @mock.patch.dict(os.environ, {'VFCL_EXPERIMENT_PROFILE': 'full-public-matrix'})
    @mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_full_cifar_smoke_passes_real_runner_bic_preflight(self, _stdout):
        from data_utils import TaskManager
        for method in driver.formal_registry.FULL_MATRIX_METHODS:
            with self.subTest(method=method):
                command = driver._smoke_command_for(
                    self.base, 'cifar100', method, f'cifar100-{method}')
                flags = driver._smoke_command_options(command)
                tokens = [token for pair in flags.items() for token in pair]
                with mock.patch('sys.argv', ['main.py', *tokens]):
                    args = experiment_config.get_config()
                protocol = runner._checkpoint_protocol(args)
                cache = runner._formal_bic_cache_plan(args)
                self.assertEqual(10, protocol['num_tasks'])
                self.assertEqual(20, args.num_classes)
                self.assertEqual(
                    [('CIL', i, [2 * i, 2 * i + 1]) for i in range(10)],
                    [(event['type'], event['task_id'], event['new_classes'])
                     for event in TaskManager(args).get_timeline()])
                self.assertIs(protocol['bic_enabled'], True)
                self.assertEqual('joint_each_stage', protocol['bic_fit_mode'])
                self.assertEqual(25, protocol['bic_per_class'])
                self.assertEqual(25, protocol['lambda_validation_per_class'])
                self.assertIs(protocol['lambda_validation_enabled'], True)
                self.assertIs(protocol['formal_deferred_evaluation'], True)
                self.assertEqual(
                    ('validation', 'final_validation_pre_install')
                    if method == 'adaptive' else
                    ('calibration', 'final_bic_calibration_post_freeze'), cache)

    @mock.patch.dict(os.environ, {'VFCL_EXPERIMENT_PROFILE': 'full-public-matrix'})
    def test_full_generated_er_ace_commands_keep_frozen_replay_options(self):
        for dataset in driver.formal_registry.DATASETS:
            with self.subTest(dataset=dataset):
                command = driver._smoke_command_for(
                    self.base, dataset, 'er_ace', f'{dataset}-er-ace')
                flags = driver._smoke_command_options(command)
                self.assertEqual('0', flags['--er_ace_buffer_size'])
                self.assertEqual('64', flags['--er_ace_batch'])
                with mock.patch('sys.argv', ['main.py', *[
                        token for pair in flags.items() for token in pair]]):
                    args = experiment_config.get_config()
                from cl_methods import get_cl_method
                method = get_cl_method('er_ace', None, args)
                self.assertEqual(20 * args.num_classes, method.buffer_size)
                protocol = runner._checkpoint_protocol(args)
                self.assertEqual(0, protocol.get('er_ace_buffer_size'))
                self.assertEqual(64, protocol.get('er_ace_batch'))
                self.assertEqual(args.num_classes, protocol.get('num_classes'))
                for name in ('num_classes', 'er_ace_buffer_size', 'er_ace_batch'):
                    self.assertIs(type(protocol[name]), int)
                args.er_ace_buffer_size, args.er_ace_batch = 7, 3
                explicit = runner._checkpoint_protocol(args)
                self.assertEqual((7, 3), (explicit['er_ace_buffer_size'],
                                         explicit['er_ace_batch']))
                for other in driver.formal_registry.FULL_MATRIX_METHODS:
                    if other == 'er_ace':
                        continue
                    args.cl_method = other
                    inherited = runner._checkpoint_protocol(args)
                    self.assertFalse({'num_classes', 'er_ace_buffer_size',
                                      'er_ace_batch'} & inherited.keys())

    @mock.patch.dict(os.environ, {'VFCL_EXPERIMENT_PROFILE': 'full-public-matrix'})
    def test_full_smoke_plan_has_exact_42_pairs_and_only_budget_overrides(self):
        root = self.base / 'full-generated-smoke'
        root.mkdir()
        plan = driver.plan_smoke(root)
        registry = driver.formal_registry
        self.assertEqual(42, len(plan['jobs']))
        self.assertEqual(42, len({(j['dataset'], j['method']) for j in plan['jobs']}))
        self.assertEqual({42}, {j['seed'] for j in plan['jobs']})
        self.assertEqual([(d, m) for d in registry.DATASETS
                          for m in registry.FULL_MATRIX_METHODS],
                         [(j['dataset'], j['method']) for j in plan['jobs']])
        overrides = {'num_classes': '4', 'num_tasks': '2',
                     'custom_tasks': '0,1|2,3', 'classes_per_task': '2',
                     'epochs_per_task': '1', 'batch_size': '8',
                     'bic_steps': '1', 'head_consolidation_steps': '1',
                     'adagauss_adapter_epochs': '1'}
        for job in plan['jobs']:
            spec = FormalSpec(job['dataset'], job['method'], 42)
            formal = driver._smoke_command_options(registry.command_for(
                spec, 'cuda:0', str(root / 'runs')))
            expected = dict(formal)
            for name, value in overrides.items():
                if '--' + name in expected or name == 'custom_tasks':
                    expected['--' + name] = value
            if job['dataset'] == 'cifar100':
                expected.update({
                    '--num_classes': '20', '--num_tasks': '10',
                    '--custom_tasks': '|'.join(f'{i},{i + 1}' for i in range(0, 20, 2)),
                })
            expected['--exp_name'] = job['job_id']
            expected['--data_path'] = str(root / 'fixtures' / job['fixture'])
            if '--vector_npz' in expected:
                expected['--vector_npz'] = str(root / 'fixtures/vector/isolet_vfl.npz')
            self.assertEqual(expected, driver._smoke_command_options(job['command']))
            self.assertEqual('1', expected['--lambda_validation_enabled'])
            self.assertEqual('1' if job['dataset'] == 'cifar100' else '0',
                             expected['--bic_enabled'])
            self.assertIs(job['generated_only'], True)
            self.assertIs(job['scientific_gate'], False)
            self.assertTrue((root / job['run_dir']).is_relative_to(root / 'runs'))
            self.assertNotIn(job['method'], {'lwf_fim', 'proto_evolve_radapt',
                                           'proto_aug', 'prl'})
            with self.assertRaises(ValueError):
                registry.parsed_protocol(job['command'], spec)
        self.assertEqual(66, plan['fixtures']['vector']['train_per_class'])
        with np.load(root / plan['fixtures']['vector']['files']['npz']['path']) as fixture:
            self.assertEqual([66] * 4, np.bincount(
                fixture['y'][fixture['train_idx']]).tolist())
        self.assertEqual(plan, driver._validate_smoke_plan(root))

    @mock.patch.dict(os.environ, {'VFCL_EXPERIMENT_PROFILE': 'full-public-matrix'})
    @mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_full_smoke_generated_holdouts_preserve_party_counts_and_disjointness(self, _stdout):
        from data_utils import VFLDataset
        import config as experiment_config
        root = self.base / 'full-holdouts'
        root.mkdir()
        plan = driver.plan_smoke(root)
        for job in plan['jobs']:
            flags = driver._smoke_command_options(job['command'])
            args_tokens = [token for pair in flags.items() for token in pair]
            with mock.patch('sys.argv', ['main.py', *args_tokens]):
                args = experiment_config.get_config()
            for option, token in flags.items():
                kind = driver.formal_registry.OPTION_SCHEMA[option[2:]]
                normalized = (token if kind == 'tasks' else
                              driver.formal_registry._normalize_value(kind, token))
                if option == '--unlearn_after_tasks':
                    normalized = [normalized]
                elif option == '--unlearn_classes':
                    normalized = [[normalized]]
                self.assertEqual(normalized, getattr(args, option[2:]), option)
            expected = driver.formal_registry.protocol_for(
                FormalSpec(job['dataset'], job['method'], 42))['base_options']
            self.assertEqual(expected['num_parties'], args.num_parties)
            # All 42 commands pass the real parser; one dataset build per fixture use.
            if job['method'] != 'adaptive':
                continue
            if job['dataset'] == 'cifar100':
                with driver._generated_cifar_runtime(args.data_path):
                    dataset = VFLDataset(args)
            else:
                dataset = VFLDataset(args)
                self.assertFalse(expected['bic_enabled'])
                self.assertEqual(args.num_parties, len(args.party_col_ranges))
            calibration = dataset.calibration_indices
            validation = dataset.validation_indices
            self.assertFalse(calibration & validation)
            self.assertEqual(args.num_classes * expected['lambda_validation_per_class'], len(validation))
            self.assertEqual(args.num_classes * expected.get('bic_per_class', 0)
                             if expected['bic_enabled'] else 0, len(calibration))
            train = set(range(len(dataset.trainset))) - calibration - validation
            labels = np.asarray(dataset.trainset.targets)
            self.assertGreaterEqual(min(np.bincount(labels[sorted(train)])), 2)
            if job['dataset'] == 'cifar100':
                self.assertEqual([500] * 20, np.bincount(labels).tolist())
                self.assertEqual([25] * 20, np.bincount(labels[sorted(calibration)]).tolist())
                self.assertEqual([25] * 20, np.bincount(labels[sorted(validation)]).tolist())
                self.assertEqual([450] * 20, np.bincount(labels[sorted(train)]).tolist())
                self.assertEqual([100] * 20, np.bincount(dataset.testset.targets).tolist())
                self.assertFalse(train & calibration)
                self.assertFalse(train & validation)
                with mock.patch.object(dataset, '_loader') as loader:
                    dataset.get_train_loader(list(range(20)))
                    self.assertEqual(train, set(loader.call_args.args[1]))
                self.assertTrue(dataset.selection_audit()['passed'])
                overlap = next(iter(calibration))
                validation.add(overlap)
                self.assertFalse(dataset.selection_audit()['passed'])
                self.assertFalse(dataset.calibration_audit()['passed'])
                validation.remove(overlap)

    @mock.patch.dict(os.environ, {'VFCL_EXPERIMENT_PROFILE': 'full-public-matrix'})
    def test_full_cifar_plan_rejects_changed_task_order_shape_and_disabled_bic(self):
        root = self.base / 'full-task-membership'
        root.mkdir()
        plan = driver.plan_smoke(root)
        manifest = json.loads((root / 'fixtures/image/fixture_manifest.json').read_text())
        tasks = [[i, i + 1] for i in range(0, 20, 2)]
        self.assertEqual(list(range(20)), manifest['classes'])
        self.assertEqual(tasks, manifest['task_classes'])
        self.assertEqual(500, manifest['train_per_class'])
        self.assertEqual(100, manifest['test_per_class'])
        for option, value in (
                ('--num_tasks', '9'), ('--num_tasks', '11'),
                ('--num_classes', '4'), ('--bic_enabled', '0'),
                ('--custom_tasks', '0,1|2,3'),
                ('--custom_tasks', '|'.join(','.join(map(str, t)) for t in tasks[::-1]))):
            with self.subTest(option=option, value=value):
                candidate = copy.deepcopy(plan)
                command = candidate['jobs'][0]['command']
                command[command.index(option) + 1] = value
                candidate['jobs'][0]['command_sha256'] = driver._digest(command)
                self._rewrite(root / 'SMOKE_PLAN.json', candidate)
                with self.assertRaisesRegex(ValueError, 'command differs'):
                    driver._validate_smoke_plan(root)
        self._rewrite(root / 'SMOKE_PLAN.json', plan)

    def test_smoke_vector_and_legacy_command_fixture_hashes_are_unchanged(self):
        baselines = {
            'formal': ('18c1536153d450d0efebf0134229043176f8c691cf1678ffcadbca3de8fe5fc9',
                       '3d5398546a8e765d155a38bcd6ab95a2bfc71a3380bdad3ccebd5984a7c30af4'),
            'seed42-pilot': ('3cb8442fb58aff6a529baa7bc2a01dee23b9a66f219d6e4fcac3307632f49da1',
                             '3d5398546a8e765d155a38bcd6ab95a2bfc71a3380bdad3ccebd5984a7c30af4'),
            'seed42-adaptive-recovery': ('18c1536153d450d0efebf0134229043176f8c691cf1678ffcadbca3de8fe5fc9',
                                         '3d5398546a8e765d155a38bcd6ab95a2bfc71a3380bdad3ccebd5984a7c30af4'),
            'full-public-matrix': ('e2765a127b3f2d7c93e84ddb8ef785ecb6ab1edee62fa1b34f618ba96dbd44b9',
                                   '6f71c10529f7afa0a3d12aaae1f763706d2bc0cf6e95b766dbfa61448eaf880f'),
        }
        # Fixed deployment/root names remove environment paths, not command bytes.
        for profile, (jobs_hash, fixtures_hash) in baselines.items():
            with self.subTest(profile=profile), mock.patch.dict(
                    os.environ, {'VFCL_EXPERIMENT_PROFILE': profile}), mock.patch.object(
                    driver.formal_registry, '_deployment_paths', return_value=(
                        Path('/deployment'), Path('/worktree'), Path('/python'))):
                jobs = driver._smoke_plan_payload(Path('/generated/smoke'), {})['jobs']
                root = self.base / profile
                root.mkdir()
                (root / 'fixtures').mkdir()
                fixtures = {'vector': driver._generated_vector_fixture(root)}
                if profile == driver.FULL_MATRIX_PROFILE:
                    jobs = [job for job in jobs if job['data_shape'] == 'vector']
                    # Only the frozen ER-ACE replay options change this golden.
                    old_jobs = copy.deepcopy(jobs)
                    for job in old_jobs:
                        if job['method'] == 'er_ace':
                            for name in ('--er_ace_buffer_size', '--er_ace_batch'):
                                index = job['command'].index(name)
                                del job['command'][index:index + 2]
                            job['command_sha256'] = driver._digest(job['command'])
                    self.assertEqual(
                        '6dacd25c57e7e8535a1182e199d12c4f38260ea56a51a65d75df73007fe5dc17',
                        driver._digest(old_jobs))
                else:
                    fixtures['image'] = driver._generated_image_fixture(root)
                self.assertEqual(jobs_hash, driver._digest(jobs))
                self.assertEqual(fixtures_hash, driver._digest(fixtures))

    @mock.patch.dict(os.environ, {'VFCL_EXPERIMENT_PROFILE': 'full-public-matrix'})
    def test_full_smoke_vector_remaining_counts_are_exact_in_plan_and_fixture(self):
        root = self.base / 'full-vector-counts'
        root.mkdir()
        plan = driver.plan_smoke(root)
        fixture = plan['fixtures']['vector']
        path = root / fixture['files']['metadata']['path']
        metadata = json.loads(path.read_text())
        expected = {'isolet': 26, 'upmc_food101': 2}
        for value in (fixture, metadata):
            self.assertNotIn('remaining_train_per_class', value)
            self.assertEqual(expected, value['remaining_train_per_class_by_dataset'])
            self.assertEqual(2, value['minimum_remaining_train_per_class'])
        for target in ('plan', 'metadata'):
            candidate = copy.deepcopy(plan)
            changed = (candidate['fixtures']['vector'] if target == 'plan'
                       else copy.deepcopy(metadata))
            changed['remaining_train_per_class_by_dataset']['isolet'] = 2
            if target == 'metadata':
                self._rewrite(path, changed)
                entry = candidate['fixtures']['vector']['files']['metadata']
                entry['size'] = path.stat().st_size
                entry['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            self._rewrite(root / 'SMOKE_PLAN.json', candidate)
            with self.assertRaisesRegex(ValueError, 'vector fixture counts differ'):
                driver._validate_smoke_plan(root)
            self._rewrite(path, metadata)
            self._rewrite(root / 'SMOKE_PLAN.json', plan)
        self.assertEqual(plan, driver._validate_smoke_plan(root))

    @mock.patch.dict(os.environ, {'VFCL_EXPERIMENT_PROFILE': 'full-public-matrix'})
    def test_full_smoke_rejects_every_scientific_marker(self):
        root = self.base / 'full-smoke-markers'
        root.mkdir()
        driver.plan_smoke(root)
        for name in driver._SMOKE_FORMAL_NAMES:
            with self.subTest(marker=name):
                path = root / name
                path.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'forbidden marker'):
                    driver.smoke_check(root)
                with self.assertRaisesRegex(ValueError, 'marker is forbidden'):
                    driver.audit_smoke(root)
                path.unlink()

    def test_legacy_smoke_profiles_keep_identical_four_job_payloads(self):
        root = self.base / 'legacy-smoke'
        payloads = []
        for profile in ('formal', 'seed42-pilot', 'seed42-adaptive-recovery'):
            with mock.patch.dict(os.environ, {'VFCL_EXPERIMENT_PROFILE': profile}):
                payload = driver._smoke_plan_payload(root, {})
                self.assertEqual(4, len(payload['jobs']))
                self.assertTrue(all('seed' not in job for job in payload['jobs']))
                payloads.append(payload)
        # ER's existing pilot buffer differs by design; all other legacy jobs match.
        self.assertEqual(payloads[0]['jobs'][0], payloads[1]['jobs'][0])
        self.assertEqual(payloads[0], payloads[2])

    def test_generated_smoke_plan_has_exact_four_shapes_and_local_fixtures(self):
        root = self.base / 'generated-smoke'
        root.mkdir()
        plan = driver.plan_smoke(root)

        self.assertEqual('three_dataset_generated_smoke_plan', plan['kind'])
        self.assertEqual(2, plan['schema_version'])
        self.assertIs(plan['generated_only'], True)
        self.assertIs(plan['scientific_gate'], False)
        self.assertEqual(driver._source_branch(), plan['source_branch'])
        self.assertEqual(driver._source_hashes(), plan['source_sha256'])
        self.assertEqual([
            ('image', 'replay_free'),
            ('vector', 'raw_replay'),
            ('vector', 'fixed_endpoint'),
            ('vector', 'adaptive'),
        ], [(job['data_shape'], job['method_shape']) for job in plan['jobs']])
        self.assertEqual(4, len(plan['jobs']))
        self.assertEqual(4, len({job['job_id'] for job in plan['jobs']}))
        self.assertEqual([0, 1, 0, 1], [
            job['preferred_gpu'] for job in plan['jobs']])
        self.assertTrue(all('gpu_index' not in job for job in plan['jobs']))
        for job in plan['jobs']:
            self.assertIs(job['generated_only'], True)
            flags = driver._smoke_command_options(job['command'])
            self.assertEqual('4', flags['--num_classes'])
            self.assertEqual('2', flags['--num_tasks'])
            self.assertEqual('0,1|2,3', flags['--custom_tasks'])
            self.assertEqual('1', flags['--epochs_per_task'])
            self.assertEqual('4', flags['--num_parties'])
            self.assertEqual('1', flags['--deterministic'])
            self.assertEqual('2', flags['--num_workers'])
            self.assertEqual('cuda:0', flags['--device'])
            self.assertEqual('0', flags['--bic_enabled'])
            if job['data_shape'] == 'image':
                self.assertEqual('0', flags['--lambda_validation_enabled'])
            elif job['method_shape'] in {'fixed_endpoint', 'adaptive'}:
                self.assertEqual('1', flags['--lambda_validation_enabled'])
            for option in ('--data_path', '--vector_npz', '--results_dir'):
                if option in flags:
                    self.assertTrue(
                        Path(flags[option]).is_relative_to(root),
                        (job['job_id'], option, flags[option]),
                    )
            flattened = '\n'.join(job['command']).lower()
            self.assertNotIn('tiny', flattened)
            self.assertNotIn('/data/isolet', flattened)
            self.assertNotIn('/data/cifar', flattened)

        image = root / plan['fixtures']['image']['files']['train']['path']
        with image.open('rb') as stream:
            payload = pickle.load(stream, encoding='latin1')
        labels = np.asarray(payload['fine_labels'])
        rows = np.asarray(payload['data'])
        self.assertEqual((208, 3072), rows.shape)
        self.assertEqual([52] * 4, np.bincount(labels, minlength=4).tolist())
        self.assertEqual(208, len({row.tobytes() for row in rows}))

        vector = root / plan['fixtures']['vector']['files']['npz']['path']
        with np.load(vector, allow_pickle=False) as payload:
            ranges = list(zip(payload['range_lo'].tolist(),
                              payload['range_hi'].tolist()))
            self.assertEqual([(0, 4), (4, 9), (9, 15), (15, 22)], ranges)
            self.assertEqual(['view_0', 'view_1', 'view_2', 'view_3'],
                             payload['view_names'].tolist())
            self.assertEqual([42] * 4, np.bincount(
                payload['y'][payload['train_idx']], minlength=4).tolist())
            self.assertEqual(176, len(set(payload['sample_ids'].tolist())))
            self.assertEqual(
                176, len({row.tobytes() for row in payload['X']}))

        self.assertEqual(plan, driver._validate_smoke_plan(root))
        self.assertEqual(
            _canonical(plan) + b'\n', (root / 'SMOKE_PLAN.json').read_bytes())
        expected_files = {
            root / entry['path']
            for fixture in plan['fixtures'].values()
            for entry in fixture['files'].values()
        }
        expected_files.add(root / 'SMOKE_PLAN.json')
        for path in expected_files:
            self.assertEqual(0o444, stat.S_IMODE(path.stat().st_mode), path)
        with self.assertRaises((FileExistsError, ValueError)):
            driver.plan_smoke(root)

    def test_smoke_gpu_policy_binds_preference_and_actual(self):
        actual = [1, 0, 1, 0]
        root = self.base / 'dynamic-smoke-gpus'
        root.mkdir()
        plan = driver.plan_smoke(root)
        for job, physical_gpu in zip(plan['jobs'], actual):
            launch = driver.smoke_begin(root, job['job_id'], physical_gpu)
            self.assertEqual(job['preferred_gpu'], launch['preferred_gpu'])
            self.assertEqual(physical_gpu, launch['physical_gpu'])
        for invalid in (None, True, False, -1, 2, 0.0, '0'):
            other = self.base / f'invalid-dynamic-gpu-{str(invalid)}'
            other.mkdir()
            invalid_plan = driver.plan_smoke(other)
            with self.assertRaisesRegex(ValueError, 'physical GPU'):
                driver.smoke_begin(
                    other, invalid_plan['jobs'][0]['job_id'], invalid)

    def test_smoke_record_rejects_tampered_launch_gpu_evidence(self):
        missing = object()
        cases = (
            ('preferred-missing', 'preferred_gpu', missing),
            ('preferred-bool', 'preferred_gpu', False),
            ('preferred-string', 'preferred_gpu', '0'),
            ('preferred-float', 'preferred_gpu', 0.0),
            ('preferred-range', 'preferred_gpu', 2),
            ('preferred-mismatch', 'preferred_gpu', 1),
            ('physical-missing', 'physical_gpu', missing),
            ('physical-bool', 'physical_gpu', True),
            ('physical-string', 'physical_gpu', '1'),
            ('physical-float', 'physical_gpu', 1.0),
            ('physical-range', 'physical_gpu', 2),
        )
        for name, field, replacement in cases:
            with self.subTest(name=name):
                root, plan = self._complete_generated_smoke(
                    f'tampered-launch-{name}')
                job = plan['jobs'][0]
                control = root / 'control' / f"{job['job_id']}.json"
                launch = json.loads(control.read_text())
                if replacement is missing:
                    launch.pop(field)
                else:
                    launch[field] = replacement
                control.chmod(0o644)
                control.write_bytes(_canonical(launch) + b'\n')
                control.chmod(0o444)
                with self.assertRaisesRegex(ValueError, 'launch evidence'):
                    driver.install_smoke_record(
                        root, job['job_id'], root / job['run_dir'])

    def test_generated_vector_smoke_uses_formal_isolet_adaptive_contract(self):
        root = self.base / 'generated-adaptive-contract'
        root.mkdir()
        plan = driver.plan_smoke(root)

        files = plan['fixtures']['vector']['files']
        with self.subTest(field='npz'):
            self.assertEqual(
                'fixtures/vector/isolet_vfl.npz', files['npz']['path'])
        with self.subTest(field='metadata'):
            self.assertEqual(
                'fixtures/vector/isolet_vfl.metadata.json',
                files['metadata']['path'])

        jobs = {
            job['method_shape']: job for job in plan['jobs']
            if job['method_shape'] in {'fixed_endpoint', 'adaptive'}
        }
        self.assertEqual({'fixed_endpoint', 'adaptive'}, set(jobs))
        for method_shape, job in jobs.items():
            with self.subTest(method_shape=method_shape):
                flags = driver._smoke_command_options(job['command'])
                self.assertEqual('isolet_vfl.npz', Path(
                    flags['--vector_npz']).name)
                self.assertEqual('20260809',
                                 flags['--lambda_validation_split_seed'])
                self.assertEqual('500', flags['--head_full_steps'])
                self.assertEqual('600', flags['--head_bias_steps'])
                self.assertEqual(
                    '80', flags['--head_gate_solver_max_iterations'])
                options = {
                    option[2:]: value for option, value in flags.items()
                }
                args = SimpleNamespace(
                    **driver.formal_registry._normalize_options(options))
                experiment_config.validate_adaptive_head_consolidation(args)

    @unittest.skipUnless(os.name == 'posix', 'symlink containment is POSIX-only')
    def test_generated_smoke_fixture_rejects_tamper_extra_and_symlink_escape(self):
        root = self.base / 'smoke-containment'
        root.mkdir()
        plan = driver.plan_smoke(root)
        vector = root / plan['fixtures']['vector']['files']['npz']['path']
        vector.chmod(0o644)
        vector.write_bytes(vector.read_bytes() + b'tamper')
        with self.assertRaises(ValueError):
            driver._validate_smoke_plan(root)

        other = self.base / 'smoke-extra'
        other.mkdir()
        driver.plan_smoke(other)
        extra = other / 'fixtures' / 'vector' / 'extra.bin'
        extra.write_bytes(b'extra')
        with self.assertRaises(ValueError):
            driver._validate_smoke_plan(other)

        target = self.base / 'real-smoke-root'
        target.mkdir()
        alias = self.base / 'smoke-root-link'
        alias.symlink_to(target, target_is_directory=True)
        with self.assertRaises(ValueError):
            driver.plan_smoke(alias)

    def test_resume_checkpoint_tree_copy_is_exact_read_only_and_rejects_unsafe(self):
        source = self.base / 'source-checkpoints'
        source.mkdir()
        (source / 'event_0_CIL.pt').write_bytes(b'first-checkpoint')
        (source / 'event_1_CIL.pt').write_bytes(b'latest-checkpoint')
        before = {
            path.name: (path.stat().st_ino, path.stat().st_mtime_ns,
                        path.stat().st_ctime_ns, path.read_bytes())
            for path in source.iterdir()
        }
        target = self.base / 'scratch-checkpoints'
        snapshot = driver._copy_resume_checkpoint_tree(source, target)
        self.assertEqual(set(before), set(snapshot))
        self.assertEqual(before, {
            path.name: (path.stat().st_ino, path.stat().st_mtime_ns,
                        path.stat().st_ctime_ns, path.read_bytes())
            for path in source.iterdir()
        })
        self.assertEqual(
            {name: value[3] for name, value in before.items()},
            {path.name: path.read_bytes() for path in target.iterdir()},
        )
        self.assertTrue(all(
            stat.S_IMODE(path.stat().st_mode) == 0o444
            for path in target.iterdir()
        ))
        with self.assertRaises(ValueError):
            driver._copy_resume_checkpoint_tree(source, target)

        empty = self.base / 'empty-checkpoints'
        empty.mkdir()
        with self.assertRaises(ValueError):
            driver._copy_resume_checkpoint_tree(
                empty, self.base / 'empty-copy')
        if os.name == 'posix':
            unsafe = self.base / 'unsafe-checkpoints'
            unsafe.mkdir()
            (unsafe / 'target.pt').write_bytes(b'x')
            (unsafe / 'resume_latest.pt').symlink_to(unsafe / 'target.pt')
            with self.assertRaises(ValueError):
                driver._copy_resume_checkpoint_tree(
                    unsafe, self.base / 'unsafe-copy')

    def _resume_run(self, name, final=True):
        run = self.base / name
        run.mkdir()
        args = SimpleNamespace(
            output_dir=str(run), resume_run_dir=str(run),
            save_task_checkpoints=3, num_tasks=2, num_parties=4,
            seed=42, data='tabvfl', cl_method='finetune',
            head_consolidation_enabled=False,
            head_consolidation_mode=None,
        )
        (run / 'config.json').write_text(json.dumps(vars(args)))

        class State:
            def __init__(self, value=0):
                self.value = value

            def get_state(self):
                return {'value': self.value}

            def load_state(self, state):
                self.value = state['value']

        tracker = driver.MetricsTracker() if hasattr(driver, 'MetricsTracker') \
            else __import__('metrics').MetricsTracker()
        random.seed(123); np.random.seed(123); torch.manual_seed(123)
        runner._save_cil_checkpoint(
            State(11), State(12), args, 'event_0_CIL', 0, [0, 1],
            {0: [0, 1]}, tracker_state=tracker.to_dict(),
        )
        if final:
            runner._save_cil_checkpoint(
                State(21), State(22), args, 'event_1_CIL', 1, [2, 3],
                {0: [0, 1], 1: [2, 3]}, tracker_state=tracker.to_dict(),
            )
        return run, State

    def test_strict_resume_probe_calls_real_loader_at_exact_latest_boundary(self):
        run, State = self._resume_run('resume-positive')
        calls = []

        class Dataset:
            def __init__(self, args):
                calls.append('dataset')
                args.party_col_ranges = [(0, 4), (4, 9), (9, 15), (15, 22)]

            def get_test_loader(self, *args, **kwargs):
                raise AssertionError('resume probe must not open the test loader')

        class Tasks:
            def __init__(self, args):
                self.seen = []

            def get_timeline(self):
                return [
                    {'type': 'CIL', 'task_id': 0, 'new_classes': [0, 1]},
                    {'type': 'CIL', 'task_id': 1, 'new_classes': [2, 3]},
                ]

            def advance_task(self, task_id):
                self.seen.append(task_id)

            def apply_unlearn(self, classes):
                raise AssertionError('smoke timeline contains no unlearning')

        def models(args):
            self.assertEqual('dataset', calls[-1])
            self.assertEqual(4, len(args.party_col_ranges))
            calls.append('models')
            return [], object()

        before = driver._checkpoint_tree_snapshot(run / 'checkpoints')
        probe_paths_before = set(Path('/tmp').glob('formal_resume_probe_*'))
        with mock.patch('data_utils.VFLDataset', Dataset), \
                mock.patch('data_utils.TaskManager', Tasks), \
                mock.patch('models.build_models', side_effect=models), \
                mock.patch('vfl_trainer.VFLTrainer', return_value=State()), \
                mock.patch('cl_methods.get_cl_method', return_value=State()):
            result = driver._strict_resume_probe_main(run)
        self.assertEqual('RESUME_PROBE_SUCCESS', result['status'])
        self.assertEqual(2, result['start_event_idx'])
        self.assertEqual({'0': [0, 1], '1': [2, 3]}, result['seen_tasks'])
        self.assertEqual(before, driver._checkpoint_tree_snapshot(
            run / 'checkpoints'))
        self.assertEqual(probe_paths_before,
                         set(Path('/tmp').glob('formal_resume_probe_*')))

    def test_strict_resume_probe_rejects_stale_missing_tampered_and_symlink(self):
        class State:
            def __init__(self, value=0):
                self.value = value

            def get_state(self):
                return {'value': self.value}

            def load_state(self, state):
                self.value = state['value']

        class Dataset:
            def __init__(self, args):
                args.party_col_ranges = [(0, 1)] * 4

        class Tasks:
            def __init__(self, args):
                self.seen = []

            def get_timeline(self):
                return [
                    {'type': 'CIL', 'task_id': 0, 'new_classes': [0, 1]},
                    {'type': 'CIL', 'task_id': 1, 'new_classes': [2, 3]},
                ]

            def advance_task(self, task_id):
                self.seen.append(task_id)

            def apply_unlearn(self, classes):
                raise AssertionError

        def probe(run):
            with mock.patch('data_utils.VFLDataset', Dataset), \
                    mock.patch('data_utils.TaskManager', Tasks), \
                    mock.patch('models.build_models', return_value=([], object())), \
                    mock.patch('vfl_trainer.VFLTrainer', return_value=State()), \
                    mock.patch('cl_methods.get_cl_method', return_value=State()):
                return driver._strict_resume_probe_main(run)

        stale, _ = self._resume_run('resume-stale', final=False)
        with self.assertRaises(ValueError):
            probe(stale)

        missing = self.base / 'resume-missing'
        missing.mkdir()
        (missing / 'config.json').write_text('{}')
        with self.assertRaises(ValueError):
            driver._strict_resume_probe_main(missing)

        tampered, _ = self._resume_run('resume-tampered')
        checkpoint = tampered / 'checkpoints' / 'event_1_CIL.pt'
        checkpoint.write_bytes(b'torn')
        with self.assertRaises((RuntimeError, ValueError)):
            probe(tampered)

        if os.name == 'posix':
            linked, _ = self._resume_run('resume-linked')
            checkpoint = linked / 'checkpoints' / 'event_1_CIL.pt'
            real = linked / 'checkpoints' / 'real.pt'
            checkpoint.rename(real)
            checkpoint.symlink_to(real)
            with self.assertRaises(ValueError):
                probe(linked)

    def test_resume_probe_pins_child_to_exact_physical_gpu_and_rejects_invalid(self):
        run = self.base / 'resume-probe-child'
        run.mkdir()
        success = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout='{"status":"RESUME_PROBE_SUCCESS"}\n')
        with mock.patch.object(
                driver.subprocess, 'run', return_value=success) as run_child, \
                mock.patch.dict(
                    driver.os.environ, {'VFCL_PROBE_SENTINEL': 'preserved'},
                    clear=False):
            result = driver.resume_probe(run, 1)
            self.assertEqual('RESUME_PROBE_SUCCESS', result['status'])
            args, kwargs = run_child.call_args
            self.assertEqual(sys.executable, args[0][0])
            self.assertEqual(str(run), args[0][-1])
            self.assertEqual('1', kwargs['env']['CUDA_VISIBLE_DEVICES'])
            self.assertEqual('preserved', kwargs['env']['VFCL_PROBE_SENTINEL'])
            self.assertIsNot(kwargs['env'], driver.os.environ)
            run_child.reset_mock()
            for value in (None, True, False, 0.0, 1.0, '0', '1', -1, 2):
                with self.assertRaisesRegex(ValueError, 'physical GPU'):
                    driver.resume_probe(run, value)
            run_child.assert_not_called()

    def test_smoke_cli_entries_never_install_formal_authority_or_launch(self):
        root = self.base / 'smoke-cli'
        root.mkdir()
        with mock.patch.object(driver, '_source_commit', return_value='c' * 40), \
                mock.patch.object(
                    driver, '_source_branch', return_value='reviewed-branch'), \
                mock.patch.object(
                    driver.formal_registry, '_deployment_paths',
                    return_value=(self.base, Path(driver.__file__).parent,
                                  Path(sys.executable))), \
                mock.patch.object(
                    driver, 'resume_probe', return_value={
                        'status': 'RESUME_PROBE_SUCCESS'}) as resume_probe, \
                mock.patch.object(driver.subprocess, 'Popen') as popen, \
                mock.patch.object(driver.os, 'system') as system, \
                mock.patch('sys.stdout', io.StringIO()):
            self.assertEqual(0, driver.main([
                'smoke-plan', '--root', str(root)]))
            self.assertEqual(0, driver.main([
                'resume-probe', '--run-dir', str(self.base / 'missing-run'),
                '--physical-gpu', '0']))
        resume_probe.assert_called_once_with(
            (self.base / 'missing-run').resolve(), 0)
        popen.assert_not_called()
        system.assert_not_called()
        self.assertFalse(any((root / name).exists() for name in (
            'FORMAL_REGISTRY.json', 'COMPATIBILITY_CENSUS.json',
            'FORMAL_PLAN.json', 'MISSING_JOBS.json',
            'FORMAL_EXECUTION_SUCCESS',
        )))

    def _complete_generated_smoke(self, name='complete-generated-smoke'):
        root = self.base / name
        root.mkdir()
        plan = driver.plan_smoke(root)
        actual = [1 - job['preferred_gpu'] for job in plan['jobs']]
        for job, physical_gpu in zip(plan['jobs'], actual):
            job_id = job['job_id']
            run = root / job['run_dir']
            run.mkdir()
            options = driver._smoke_command_options(job['command'])
            config = {
                'num_classes': 4, 'num_tasks': 2,
                'custom_tasks': '0,1|2,3', 'num_parties': 4,
                'epochs_per_task': 1, 'formal_deferred_evaluation': True,
                'save_task_checkpoints': 3, 'seed': 42,
                'device': 'cuda:0', 'output_dir': str(run),
                'data_path': options['--data_path'],
                'bic_enabled': int(options['--bic_enabled']),
                'lambda_validation_enabled': int(
                    options['--lambda_validation_enabled']),
            }
            if driver.experiment_profile() == driver.FULL_MATRIX_PROFILE:
                tokens = [token for pair in options.items() for token in pair]
                with mock.patch('sys.argv', ['main.py', *tokens]), \
                        mock.patch('sys.stdout', io.StringIO()):
                    config = vars(experiment_config.get_config())
            if '--vector_npz' in options:
                config['vector_npz'] = options['--vector_npz']
            (run / 'config.json').write_text(json.dumps(config))
            (run / 'results.json').write_text(json.dumps({
                'generated_only': True, 'job_id': job_id,
            }))
            access = {
                'event': 'first_iteration',
                'loader_key': repr(('test', (0, 1, 2, 3))),
                'split': 'test', 'phase': 'final_test_post_install',
                'event_idx': config['num_tasks'] - 1,
                'task_id': config['num_tasks'] - 1,
                'timeline_step': f"event_{config['num_tasks'] - 1}_CIL",
                'classes': list(range(config['num_classes'])),
            }
            (run / 'data_flow_audit.jsonl').write_text(
                json.dumps(access) + '\n')
            required = [
                *(f'checkpoints/event_{i}_CIL.pt' for i in range(config['num_tasks'])),
                'checkpoints/formal_final.pt',
                *(f'formal_snapshots/event_{i}_CIL.pt' for i in range(config['num_tasks'])),
                'FORMAL_STATE_FROZEN.json',
                'FORMAL_EVALUATION_PENDING.json',
                'FORMAL_EVALUATION_CONSUMING.json',
                'FORMAL_EVALUATION_COMPLETE.json',
                'FORMAL_EVALUATION_SEALED.json',
                'FORMAL_EVALUATION_PUBLISHING.json',
                'FORMAL_EVALUATION_PUBLISHED.json',
                'formal_access/test.consumed.json',
            ]
            validation_access = job['method_shape'] in {'fixed_endpoint', 'adaptive'}
            if driver.experiment_profile() == driver.FULL_MATRIX_PROFILE:
                validation_access = driver.formal_registry.validation_access_for(
                    FormalSpec(job['dataset'], job['method'], 42))
            if validation_access:
                required.append('formal_access/validation.consumed.json')
            if config['bic_enabled'] and not validation_access:
                required.append('formal_access/calibration.consumed.json')
            for logical in required:
                path = run / logical
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(
                    b'checkpoint' if path.suffix == '.pt' else b'{}\n')
            log = root / 'logs' / f'{job_id}.log'
            log.write_text(f'{job_id} complete\n')
            log.chmod(0o444)
            driver.smoke_begin(root, job_id, physical_gpu)
        producer = mock.patch.object(
            driver, '_strict_smoke_producer_audit', create=True,
            side_effect=lambda run, job: {
                'status': 'FORMAL_SMOKE_PRODUCER_VERIFIED',
                'transaction_sha256': hashlib.sha256(
                    str(run).encode()).hexdigest(),
                'published': True,
                'source_commit': plan['source_commit'],
                'source_sha256': plan['source_sha256'],
            },
        )
        resume = mock.patch.object(
            driver, 'resume_probe', return_value={
                'status': 'RESUME_PROBE_SUCCESS',
                'start_event_idx': 2,
                'source_unchanged': True,
                'test_loader_opened': False,
            },
        )
        with producer, resume as resume:
            for job in plan['jobs']:
                driver.install_smoke_record(
                    root, job['job_id'], root / job['run_dir'])
        self.assertEqual([
            mock.call(root / job['run_dir'], physical_gpu)
            for job, physical_gpu in zip(plan['jobs'], actual)
        ], resume.call_args_list)
        for job, physical_gpu in zip(plan['jobs'], actual):
            launch = driver._load_json_document(
                root / 'control' / f"{job['job_id']}.json",
                require_object=True)
            record = driver._load_json_document(
                root / 'records' / f"{job['job_id']}.json",
                require_object=True)
            self.assertEqual(job['preferred_gpu'], launch['preferred_gpu'])
            self.assertEqual(physical_gpu, launch['physical_gpu'])
            self.assertEqual(job['preferred_gpu'], record['preferred_gpu'])
            self.assertEqual(physical_gpu, record['physical_gpu'])
        return root, plan

    def _audit_complete_generated_smoke(self, root):
        plan = driver._validate_smoke_plan(root)
        with mock.patch.object(
                driver, '_strict_smoke_producer_audit', create=True,
                side_effect=lambda run, job: {
                    'status': 'FORMAL_SMOKE_PRODUCER_VERIFIED',
                    'transaction_sha256': hashlib.sha256(
                    str(run).encode()).hexdigest(),
                    'published': True,
                    'source_commit': plan['source_commit'],
                    'source_sha256': plan['source_sha256'],
                }), mock.patch.object(
                    driver, 'resume_probe', return_value={
                        'status': 'RESUME_PROBE_SUCCESS',
                        'start_event_idx': 2,
                        'source_unchanged': True,
                    'test_loader_opened': False,
                }):
            return driver.audit_smoke(root)

    def _reinstall_generated_smoke_record(self, root, plan, job):
        (root / 'records' / f"{job['job_id']}.json").unlink(missing_ok=True)
        with mock.patch.object(
                driver, '_strict_smoke_producer_audit', create=True,
                side_effect=lambda run, _job: {
                    'status': 'FORMAL_SMOKE_PRODUCER_VERIFIED',
                    'transaction_sha256': hashlib.sha256(
                        str(run).encode()).hexdigest(),
                    'published': True,
                    'source_commit': plan['source_commit'],
                    'source_sha256': plan['source_sha256'],
                }), mock.patch.object(
                    driver, 'resume_probe', return_value={
                        'status': 'RESUME_PROBE_SUCCESS',
                        'start_event_idx': 2,
                        'source_unchanged': True,
                        'test_loader_opened': False,
                    }):
            driver.install_smoke_record(
                root, job['job_id'], root / job['run_dir'])

    @mock.patch.dict(os.environ, {'VFCL_EXPERIMENT_PROFILE': 'full-public-matrix'})
    def test_full_cifar_records_require_exact_ten_events_snapshots_and_final_access(self):
        root, plan = self._complete_generated_smoke('full-event-membership')
        job = plan['jobs'][0]
        run = root / job['run_dir']
        record = json.loads((root / 'records' / f"{job['job_id']}.json").read_text())
        self.assertEqual(10, record['train_only_stage_snapshots'])
        expected = {f'{folder}/event_{i}_CIL.pt'
                    for folder in ('checkpoints', 'formal_snapshots') for i in range(10)}
        expected.add('checkpoints/formal_final.pt')
        self.assertEqual(expected, {name for name in record['artifact_sha256']
                                    if name.endswith('.pt')})
        # save_task_checkpoints=3 + force=True retains the runner's rolling copy.
        (run / 'checkpoints/resume_latest.pt').write_bytes(
            (run / 'checkpoints/event_9_CIL.pt').read_bytes())
        self._reinstall_generated_smoke_record(root, plan, job)
        for name in sorted(expected):
            with self.subTest(missing=name):
                path = run / name
                content = path.read_bytes()
                path.unlink()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    driver._smoke_run_record(root, job['job_id'], run)
                path.write_bytes(content)
        for folder in ('checkpoints', 'formal_snapshots'):
            path = run / folder / 'event_10_CIL.pt'
            path.write_bytes(b'extra')
            with self.subTest(extra=folder), self.assertRaisesRegex(ValueError, 'membership differs'):
                driver._smoke_run_record(root, job['job_id'], run)
            path.unlink()
        flow = run / 'data_flow_audit.jsonl'
        access = json.loads(flow.read_text())
        for field, value in (('event_idx', 1), ('task_id', 8),
                             ('timeline_step', 'event_8_CIL'),
                             ('classes', list(reversed(range(20))))):
            flow.write_text(json.dumps({**access, field: value}) + '\n')
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'post-freeze test access'):
                driver._smoke_run_record(root, job['job_id'], run)
        flow.write_text(json.dumps(access) + '\n')

    @mock.patch.dict(os.environ, {'VFCL_EXPERIMENT_PROFILE': 'full-public-matrix'})
    def test_full_adaptive_smoke_uses_validation_without_calibration_access(self):
        root, plan = self._complete_generated_smoke(
            'full-adaptive-formal-access')
        job = next(job for job in plan['jobs']
                   if job['dataset'] == 'cifar100'
                   and job['method'] == 'adaptive')
        run = root / job['run_dir']

        self._reinstall_generated_smoke_record(root, plan, job)

        record = json.loads(
            (root / 'records' / f"{job['job_id']}.json").read_text())
        self.assertIn('formal_access/validation.consumed.json',
                      record['artifact_sha256'])
        self.assertNotIn('formal_access/calibration.consumed.json',
                         record['artifact_sha256'])

    @mock.patch.dict(os.environ, {'VFCL_EXPERIMENT_PROFILE': 'full-public-matrix'})
    def test_full_adaptive_smoke_rejects_extra_calibration_access(self):
        root, plan = self._complete_generated_smoke(
            'full-adaptive-extra-calibration')
        job = next(job for job in plan['jobs']
                   if job['dataset'] == 'cifar100'
                   and job['method'] == 'adaptive')
        calibration = (root / job['run_dir']
                       / 'formal_access/calibration.consumed.json')
        calibration.write_text('{}\n')

        with self.assertRaisesRegex(ValueError, 'unexpected calibration'):
            self._reinstall_generated_smoke_record(
                root, plan, job)

    def test_legacy_smoke_keeps_unrelated_formal_access_files(self):
        root, plan = self._complete_generated_smoke(
            'legacy-unrelated-formal-access')
        job = plan['jobs'][0]
        extra = (root / job['run_dir'] / 'formal_access/legacy-note.json')
        extra.write_text('{}\n')

        self._reinstall_generated_smoke_record(root, plan, job)

        record = json.loads(
            (root / 'records' / f"{job['job_id']}.json").read_text())
        self.assertNotIn('formal_access/legacy-note.json', record['artifact_sha256'])

    @mock.patch.dict(os.environ, {'VFCL_EXPERIMENT_PROFILE': 'full-public-matrix'})
    def test_full_smoke_audit_requires_42_exact_safe_generated_records(self):
        root, plan = self._complete_generated_smoke('full-audit')
        report = self._audit_complete_generated_smoke(root)
        self.assertEqual(42, report['job_count'])
        self.assertEqual(42, len(report['job_records']))
        self.assertIs(report['scientific_gate'], False)
        for name in driver._SMOKE_FORMAL_NAMES:
            self.assertFalse((root / name).exists())
        (root / 'SMOKE_AUDIT.json').unlink()
        (root / 'SMOKE_EXECUTION_SUCCESS').unlink()
        last = plan['jobs'][-1]
        record = root / 'records' / (last['job_id'] + '.json')
        content = record.read_bytes()
        record.unlink()
        with self.assertRaises(ValueError):
            self._audit_complete_generated_smoke(root)
        record.write_bytes(content)
        record.chmod(0o444)
        first = plan['jobs'][0]
        first_run = root / first['run_dir']
        first_config_path = first_run / 'config.json'
        first_config = json.loads(first_config_path.read_text())
        for field, value in (
                ('unlearn_after_tasks', 99), ('unlearn_after_tasks', [99, 99]),
                ('unlearn_after_tasks', [[99]]), ('unlearn_after_tasks', [99.0]),
                ('unlearn_classes', [0]), ('unlearn_classes', [[0, 1]]),
                ('unlearn_classes', [[[0]]]), ('unlearn_classes', [[False]])):
            with self.subTest(field=field, value=value):
                first_config_path.write_text(json.dumps({**first_config, field: value}))
                with self.assertRaisesRegex(ValueError, 'runtime config differs'):
                    driver._smoke_run_record(root, first['job_id'], first_run)
        first_config_path.write_text(json.dumps(first_config))
        config_path = root / last['run_dir'] / 'config.json'
        config = json.loads(config_path.read_text())
        config['head_full_steps'] = 1
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'runtime config differs'):
            self._audit_complete_generated_smoke(root)
        config['head_full_steps'] = 500
        config['num_parties'] = 4
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'runtime config differs'):
            self._audit_complete_generated_smoke(root)
        config['num_parties'] = 2
        config_path.write_text(json.dumps(config))
        checkpoint = root / last['run_dir'] / 'checkpoints/formal_final.pt'
        checkpoint.unlink()
        checkpoint.symlink_to(self.base / 'outside.pt')
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            self._audit_complete_generated_smoke(root)

    def test_real_smoke_audit_accepts_complete_generated_plan_with_tiny_source_name(self):
        root, plan = self._complete_generated_smoke()
        self.assertIn('adaptive_tinyimagenet_heldout.py', plan['source_sha256'])
        report = self._audit_complete_generated_smoke(root)
        self.assertEqual('SMOKE_EXECUTION_SUCCESS', report['status'])
        self.assertEqual(4, report['job_count'])
        self.assertEqual({
            'claim_entries': 0, 'gpu_claim_entries': 0,
            'live_processes': 0,
        }, report['runtime_state'])
        self.assertTrue((root / 'SMOKE_EXECUTION_SUCCESS').is_file())
        self.assertFalse((root / 'FORMAL_EXECUTION_SUCCESS').exists())

    def test_smoke_check_rejects_each_nonformal_success_marker(self):
        for name in driver._SUCCESS_MARKER_PROFILES:
            with self.subTest(marker=name):
                root = self.base / name
                root.mkdir()
                driver.plan_smoke(root)
                (root / name).write_text('{}')
                with self.assertRaisesRegex(ValueError, 'forbidden marker'):
                    driver.smoke_check(root)

    def test_smoke_audit_rejects_each_nonformal_success_marker(self):
        for name in driver._SUCCESS_MARKER_PROFILES:
            with self.subTest(marker=name):
                root, _ = self._complete_generated_smoke(name)
                (root / name).write_text('{}')
                with self.assertRaisesRegex(ValueError, 'forbidden in smoke audit'):
                    self._audit_complete_generated_smoke(root)

    def test_smoke_audit_rederives_runs_and_rejects_forged_empty_tampered_extra(self):
        empty = self.base / 'empty-smoke-runs'
        empty.mkdir()
        driver.plan_smoke(empty)
        with self.assertRaises(ValueError):
            driver.audit_smoke(empty)

        forged, _ = self._complete_generated_smoke('forged-smoke-runs')
        shutil.rmtree(forged / 'runs')
        (forged / 'runs').mkdir()
        with self.assertRaises(ValueError):
            self._audit_complete_generated_smoke(forged)

        tampered, plan = self._complete_generated_smoke('tampered-smoke-runs')
        result = tampered / plan['jobs'][0]['run_dir'] / 'results.json'
        result.write_text('{"tampered":true}\n')
        with self.assertRaises(ValueError):
            self._audit_complete_generated_smoke(tampered)

        extra, _ = self._complete_generated_smoke('extra-smoke-runs')
        (extra / 'runs' / 'extra-run').mkdir()
        with self.assertRaises(ValueError):
            self._audit_complete_generated_smoke(extra)

        unexpected, plan = self._complete_generated_smoke(
            'unexpected-image-calibration')
        image_run = unexpected / plan['jobs'][0]['run_dir']
        calibration = image_run / 'formal_access/calibration.consumed.json'
        calibration.parent.mkdir(parents=True, exist_ok=True)
        calibration.write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'unexpected calibration'):
            self._audit_complete_generated_smoke(unexpected)

    def test_smoke_audit_binds_bic_and_validation_runtime_config(self):
        for field in ('bic_enabled', 'lambda_validation_enabled'):
            with self.subTest(missing_image_field=field):
                root, plan = self._complete_generated_smoke(
                    f'missing-image-{field}')
                config_path = root / plan['jobs'][0]['run_dir'] / 'config.json'
                config = json.loads(config_path.read_text())
                del config[field]
                config_path.write_text(json.dumps(config))
                with self.assertRaisesRegex(
                        ValueError, 'generated smoke runtime config differs'):
                    self._reinstall_generated_smoke_record(root, plan,
                                                           plan['jobs'][0])

        for field in ('bic_enabled', 'lambda_validation_enabled'):
            with self.subTest(mismatched_image_field=field):
                root, plan = self._complete_generated_smoke(
                    f'mismatched-image-{field}')
                config_path = root / plan['jobs'][0]['run_dir'] / 'config.json'
                config = json.loads(config_path.read_text())
                config[field] = 1
                config_path.write_text(json.dumps(config))
                with self.assertRaisesRegex(
                        ValueError, 'generated smoke runtime config differs'):
                    self._reinstall_generated_smoke_record(root, plan,
                                                           plan['jobs'][0])

        for job in self._complete_generated_smoke(
                'mismatched-vector-validation')[1]['jobs']:
            if job['method_shape'] not in {'fixed_endpoint', 'adaptive'}:
                continue
            with self.subTest(mismatched_vector_job=job['job_id']):
                root, plan = self._complete_generated_smoke(
                    f'mismatched-{job["job_id"]}-validation')
                selected = next(candidate for candidate in plan['jobs']
                                if candidate['job_id'] == job['job_id'])
                config_path = root / selected['run_dir'] / 'config.json'
                config = json.loads(config_path.read_text())
                config['lambda_validation_enabled'] = 0
                config_path.write_text(json.dumps(config))
                with self.assertRaisesRegex(
                        ValueError, 'generated smoke runtime config differs'):
                    self._reinstall_generated_smoke_record(root, plan,
                                                           selected)

        for job_id in ('image-replay-free', 'vector-adaptive'):
            root, plan = self._complete_generated_smoke(
                f'noninteger-{job_id}')
            selected = next(job for job in plan['jobs']
                            if job['job_id'] == job_id)
            config_path = root / selected['run_dir'] / 'config.json'
            baseline = json.loads(config_path.read_text())
            for field in ('bic_enabled', 'lambda_validation_enabled'):
                expected = int(driver._smoke_command_options(
                    selected['command'])[f'--{field}'])
                for value in (bool(expected), float(expected)):
                    with self.subTest(job=job_id, field=field,
                                      representation=type(value).__name__):
                        config_path.write_text(json.dumps(baseline))
                        config = json.loads(config_path.read_text())
                        config[field] = value
                        config_path.write_text(json.dumps(config))
                        with self.assertRaisesRegex(
                                ValueError,
                                'generated smoke runtime config differs'):
                            self._reinstall_generated_smoke_record(
                                root, plan, selected)

    @unittest.skipUnless(os.name == 'posix', 'active/symlink audit is POSIX-only')
    def test_smoke_audit_rejects_active_claim_gpu_process_and_symlink(self):
        for unsafe_name in ('claims', 'gpu_claims'):
            root, _ = self._complete_generated_smoke(f'unsafe-{unsafe_name}')
            (root / unsafe_name).mkdir()
            with self.assertRaises(ValueError):
                self._audit_complete_generated_smoke(root)

        linked, plan = self._complete_generated_smoke('symlink-smoke-run')
        result = linked / plan['jobs'][0]['run_dir'] / 'results.json'
        target = linked / 'outside-results.json'
        result.rename(target)
        result.symlink_to(target)
        with self.assertRaises(ValueError):
            self._audit_complete_generated_smoke(linked)

        active, _ = self._complete_generated_smoke('active-smoke-run')
        process = subprocess.Popen([
            sys.executable, '-c', 'import time; time.sleep(30)', str(active),
        ])
        self.addCleanup(lambda: process.poll() is None and process.kill())
        try:
            with self.assertRaises(ValueError):
                self._audit_complete_generated_smoke(active)
        finally:
            process.kill()
            process.wait(timeout=5)

    def test_smoke_plan_rejects_forged_current_authority_and_binds_controls(self):
        root = self.base / 'authority-smoke'
        root.mkdir()
        plan = driver.plan_smoke(root)
        controls = plan.get('control_sha256')
        self.assertEqual({
            'three_dataset_formal_driver.py',
            'run_three_dataset_formal_comparison.sh',
        }, set(controls or {}))
        for name, mutate in (
                ('commit', lambda value: value.__setitem__(
                    'source_commit', '0' * 40)),
                ('branch', lambda value: value.__setitem__(
                    'source_branch', 'forged-branch')),
                ('source', lambda value: value['source_sha256'].__setitem__(
                    next(iter(value['source_sha256'])), '0' * 64)),
                ('driver', lambda value: value['control_sha256'].__setitem__(
                    'three_dataset_formal_driver.py', '0' * 64)),
                ('launcher', lambda value: value['control_sha256'].__setitem__(
                    'run_three_dataset_formal_comparison.sh', '0' * 64))):
            with self.subTest(name=name):
                candidate = copy.deepcopy(plan)
                mutate(candidate)
                self._rewrite(root / 'SMOKE_PLAN.json', candidate)
                with self.assertRaises(ValueError):
                    driver._validate_smoke_plan(root)
                self._rewrite(root / 'SMOKE_PLAN.json', plan)

    def test_strict_smoke_producer_audit_uses_real_formal_publication_validator(self):
        self._check_strict_smoke_producer()

    @mock.patch.dict(os.environ, {'VFCL_EXPERIMENT_PROFILE': 'full-public-matrix'})
    def test_full_smoke_strict_producer_preserves_upmc_two_party_protocol(self):
        self._check_strict_smoke_producer(num_parties=2, dataset='upmc_food101')

    def _check_strict_smoke_producer(self, num_parties=4, dataset='isolet'):
        audit = getattr(driver, '_strict_smoke_producer_audit', None)
        self.assertTrue(callable(audit), 'strict formal smoke producer audit is missing')
        from test_adaptive_deferred_evaluation import _FormalStateTrainer
        from adaptive_consolidation_audit import (
            evaluate_formal_deferred_trajectory,
            prepare_formal_deferred_evaluation,
            save_deferred_cil_snapshot,
        )
        from cl_methods import get_cl_method
        from metrics import MetricsTracker, cache_formal_batches
        root = self.base / 'strict-producer'
        root.mkdir()

        class FourPartyTrainer(_FormalStateTrainer):
            def get_state(self):
                return {
                    'bottoms': [{} for _ in range(num_parties)],
                    'top_model': copy.deepcopy(self.state),
                }

            def load_state(self, state):
                self.state = copy.deepcopy(state['top_model'])

        args = SimpleNamespace(
            output_dir=str(root), save_task_checkpoints=0,
            formal_deferred_evaluation=True, device='cpu',
            seed=42, data='tabvfl', cl_method='er', num_tasks=2,
            num_parties=num_parties, head_consolidation_enabled=0,
            head_consolidation_mode='full_classifier', bic_enabled=0,
            batch_size=2, num_classes=4, sanitize_cl_state=1,
            party_kd_enabled=0, party_kd_mode='uniform',
            party_proto_enabled=0, dep_tracking_enabled=0,
        )
        task_classes = {0: [0, 1], 1: [2, 3]}
        tracker = MetricsTracker()
        paths = []
        states = [
            {'correct': {0: True, 1: True}, 'fallback': 0},
            {'correct': {0: True, 1: False, 2: True, 3: True}, 'fallback': 0},
        ]
        for event_idx in range(2):
            step = f'event_{event_idx}_CIL'
            comm = {'comm_rounds': event_idx, 'megabytes_transmitted': 0.0}
            tracker.record_timing(step, 0.25)
            tracker.record_comm(step, comm)
            tracker.record_step({
                'event_idx': event_idx, 'type': 'CIL',
                'task_id': event_idx,
                'new_classes': task_classes[event_idx],
                'evaluation_deferred': True, 'train_time': 0.25,
                'comm': comm,
            })
            trainer = FourPartyTrainer(states[event_idx])
            method = get_cl_method('er', trainer, args)
            method.buffer.add_batch(
                torch.tensor([[float(event_idx)]]),
                torch.tensor([event_idx]),
            )
            seen = {key: task_classes[key]
                    for key in range(event_idx + 1)}
            runner._save_cil_checkpoint(
                trainer, method, args, step, event_idx,
                task_classes[event_idx], seen,
                tracker_state=tracker.to_dict(), force=True,
            )
            paths.append(Path(save_deferred_cil_snapshot(
                trainer, method, args, event_idx, event_idx, seen,
                protocol_kind='formal',
            )))
        formal_final = root / 'checkpoints' / 'formal_final.pt'
        runner._atomic_torch_save(torch.load(
            root / 'checkpoints' / 'event_1_CIL.pt',
            map_location='cpu', weights_only=True), formal_final)
        labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        cache = cache_formal_batches([
            (torch.arange(4, dtype=torch.float32).view(4, 1), labels[:4]),
            (torch.arange(4, 8, dtype=torch.float32).view(4, 1), labels[4:]),
        ])
        prepare_formal_deferred_evaluation(
            args=args, snapshot_paths=paths,
            final_checkpoint=formal_final, task_classes=task_classes,
            output_dir=root,
        )
        with mock.patch(
                'adaptive_consolidation_audit._fresh_trainer',
                side_effect=lambda _payload, _args: FourPartyTrainer()):
            result = evaluate_formal_deferred_trajectory(
                args=args, snapshot_paths=paths,
                final_checkpoint=formal_final,
                task_classes=task_classes, cached_test_batches=cache,
                output_dir=root,
            )
        final = {**result, 'config': {'cl_method': 'er'}}
        with mock.patch(
                'adaptive_consolidation_audit._fresh_trainer',
                side_effect=lambda _payload, _args: FourPartyTrainer()):
            runner._publish_formal_deferred_result(
                args=args, final_checkpoint=formal_final,
                tracker_state=result, final_result=final,
            )
        job = {
            'dataset': dataset, 'method': 'er',
            'data_shape': 'vector', 'method_shape': 'raw_replay',
            'command': driver._smoke_command_for(self.base, dataset, 'er', 'producer'),
        }
        proof = audit(root, job)
        self.assertEqual('FORMAL_SMOKE_PRODUCER_VERIFIED', proof['status'])
        self.assertTrue(proof['published'])
        if num_parties == 2:
            command = list(job['command'])
            command[command.index('--num_parties') + 1] = '4'
            with self.assertRaisesRegex(ValueError, 'producer protocol differs'):
                audit(root, {**job, 'command': command})


class GeneratedERACECompatibilityTests(unittest.TestCase):
    @mock.patch.dict(os.environ, {
        'VFCL_EXPERIMENT_PROFILE': 'full-public-matrix',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 'PYTHONHASHSEED': '42'})
    @mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_generated_er_ace_trains_publishes_and_resumes_only_with_replay(self, output):
        import adaptive_consolidation_audit as deferred
        import models
        from cl_methods.er_ace import ERAccCL
        import runpy

        build_models = models.build_models

        def cpu_models(args):
            args.device = 'cpu'
            return build_models(args)

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory(prefix='generated_er_ace_') as temporary:
            root = Path(temporary)
            with mock.patch.object(driver.formal_registry, '_deployment_paths',
                    return_value=(root, Path(driver.__file__).parent, Path(sys.executable))):
                plan = driver.plan_smoke(root)
                job = driver._smoke_job(plan, 'isolet-er-ace')
                with mock.patch('sys.argv', list(job['command'][1:])), \
                        mock.patch.object(runner, 'build_models', cpu_models), \
                        mock.patch.object(deferred, 'build_models', cpu_models), \
                        mock.patch('torch.cuda.is_available', return_value=False):
                    runpy.run_path(job['command'][1], run_name='__main__')
                run = root / job['run_dir']
                checkpoints = sorted((run / 'checkpoints').glob('event_*_CIL.pt'))
                self.assertEqual(2, len(checkpoints))
                for path in [*checkpoints, run / 'checkpoints/formal_final.pt']:
                    payload = runner._decode_checkpoint_value(torch.load(
                        path, map_location='cpu', weights_only=True))
                    state = payload['cl_state']
                    self.assertEqual('er_ace', payload['protocol']['cl_method'])
                    self.assertEqual(80, state['buffer_size'])
                    self.assertGreater(state['num_seen'], 0)
                    self.assertEqual(min(80, state['num_seen']), len(state['labels']))
                    self.assertEqual(len(state['labels']), len(state['examples']))
                driver.smoke_begin(root, job['job_id'], 0)
                driver._install_bytes_exclusive(
                    root / 'logs' / f"{job['job_id']}.log", output.getvalue().encode())
                with mock.patch('models.build_models', cpu_models), \
                        mock.patch('torch.cuda.is_available', return_value=False), \
                        mock.patch.object(driver, 'resume_probe', side_effect=lambda path, gpu:
                                          driver._strict_resume_probe_main(path)):
                    record = driver.install_smoke_record(root, job['job_id'], run)
                self.assertEqual('isolet-er-ace', record['job_id'])
                self.assertTrue(record['passed'])
                self.assertTrue(record['generated_only'])
                self.assertFalse(record['scientific_gate'])
                self.assertEqual(2, record['resume_probe']['start_event_idx'])
                self.assertTrue(record['resume_probe']['source_unchanged'])
                self.assertFalse(record['resume_probe']['test_loader_opened'])
                accesses = [json.loads(line) for line in
                            (run / 'data_flow_audit.jsonl').read_text().splitlines()]
                self.assertEqual(['final_test_post_install'], [entry['phase']
                    for entry in accesses if entry.get('split') == 'test'])
                self.assertEqual(['test.consumed.json'], sorted(
                    path.name for path in (run / 'formal_access').iterdir()))

                # Fresh/initial empty replay is valid, but cannot represent a
                # completed ER-ACE stage at the formal publication boundary.
                snapshot = torch.load(run / 'formal_snapshots/event_1_CIL.pt',
                                      map_location='cpu', weights_only=True)
                empty = ERAccCL(None, SimpleNamespace(
                    num_classes=4, er_ace_buffer_size=0, er_ace_batch=64)).get_state()
                snapshot['cl_state'] = runner._encode_checkpoint_value(empty)
                args = SimpleNamespace(**json.loads((run / 'config.json').read_text()))
                from data_utils import VFLDataset
                VFLDataset(args)  # Restore generated vector party-column widths.
                with mock.patch.object(deferred, 'build_models', cpu_models), \
                        self.assertRaisesRegex(ValueError, 'ER-ACE.*nonempty'):
                    deferred._fresh_formal_state(snapshot, args)
                final = torch.load(run / 'checkpoints/formal_final.pt',
                                   map_location='cpu', weights_only=True)
                for invalid in ({}, empty, {**empty, 'num_seen': 1},
                                {**empty, 'buffer_size': 81},
                                {**empty, 'extra': True}):
                    with self.subTest(fields=tuple(invalid)):
                        bad_final = {**final, 'cl_state': runner._encode_checkpoint_value(invalid)}
                        # Substitute the decoded checkpoint after file checks to
                        # isolate semantic admission from integrity rejection.
                        with mock.patch.object(deferred, '_safe_torch_load',
                                               return_value=bad_final), \
                                self.assertRaisesRegex(ValueError, 'ER-ACE'):
                            driver._strict_smoke_producer_audit(run, job)

                # Resume must reject an empty latest candidate on isolated probes
                # and recover the older real checkpoint before touching live state.
                from bic_calibration import TaskAffineCalibrator
                from data_utils import TaskManager
                from metrics import MetricsTracker
                from vfl_trainer import VFLTrainer
                source_before = driver._checkpoint_tree_snapshot(run / 'checkpoints')
                with tempfile.TemporaryDirectory(prefix='generated_er_ace_resume_') as resume_dir, \
                        mock.patch('torch.cuda.is_available', return_value=False):
                    scratch = Path(resume_dir)
                    shutil.copytree(run / 'checkpoints', scratch / 'checkpoints')
                    args.output_dir = args.resume_run_dir = str(scratch)
                    trainer = VFLTrainer(*cpu_models(args), args)
                    method = ERAccCL(trainer, args)
                    tracker, task_manager = MetricsTracker(), TaskManager(args)
                    calibrator = TaskAffineCalibrator()
                    latest_path = scratch / 'checkpoints/event_1_CIL.pt'
                    latest = torch.load(latest_path, map_location='cpu', weights_only=True)
                    latest['cl_state'] = runner._encode_checkpoint_value(empty)
                    runner._atomic_torch_save(latest, latest_path)
                    rolling = scratch / 'checkpoints/resume_latest.pt'
                    if rolling.is_file():
                        self.assertEqual(1, torch.load(
                            rolling, map_location='cpu', weights_only=True)['event_idx'])
                        runner._atomic_torch_save(latest, rolling)
                    # The initial empty reservoir is structurally valid, so the
                    # completed/formal candidate preflight must add this guard.
                    runner._validate_resume_checkpoint_payload(latest, args)

                    def live_state():
                        return copy.deepcopy({
                            'trainer': trainer.get_state(), 'method': method.get_state(),
                            'tracker': tracker.to_dict(),
                            'task': {key: value for key, value in vars(task_manager).items()
                                     if key != 'args'},
                            'rng': runner._capture_rng_state(),
                        })

                    before = live_state()
                    with self.assertRaisesRegex(ValueError, 'formal ER-ACE requires nonempty replay state'):
                        runner._validate_resume_candidate_loadability(
                            latest, args, trainer, method, task_manager, calibrator, tracker)
                    after = live_state()
                    for key in before:
                        self.assertTrue(runner._checkpoint_values_equal(before[key], after[key]), key)
                    older = runner._decode_checkpoint_value(torch.load(
                        scratch / 'checkpoints/event_0_CIL.pt', map_location='cpu', weights_only=True))
                    self.assertGreater(len(older['cl_state']['labels']), 0)
                    start, seen, history = runner._load_resume_checkpoint(
                        args, trainer, method, task_manager, tracker, calibrator)
                    self.assertEqual((1, {0: [0, 1]}, []), (start, seen, history))
                    self.assertEqual([0, 1], task_manager.get_all_seen_classes())
                    for actual, expected in (
                            (trainer.get_state(), older['trainer_state']),
                            (method.get_state(), older['cl_state']),
                            (tracker.to_dict(), older['tracker_state']),
                            (runner._capture_rng_state(), older['rng_state'])):
                        self.assertTrue(runner._checkpoint_values_equal(actual, expected))
                self.assertEqual(source_before, driver._checkpoint_tree_snapshot(run / 'checkpoints'))


class GeneratedCifarManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='generated_cifar_adapter_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for patch in (
                mock.patch.dict(os.environ, {
                    'VFCL_EXPERIMENT_PROFILE': 'full-public-matrix',
                    'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '1',
                    'MKL_NUM_THREADS': '1', 'PYTHONHASHSEED': '42'}),
                mock.patch.object(driver.formal_registry, '_deployment_paths',
                                  return_value=(self.root, Path(driver.__file__).parent,
                                                Path(sys.executable)))):
            patch.start()
            self.addCleanup(patch.stop)
        self.plan = driver.plan_smoke(self.root)
        self.job = driver._smoke_job(self.plan, 'cifar100-finetune')
        self.fixture = self.root / 'fixtures/image'
        self.manifest = json.loads((self.fixture / 'fixture_manifest.json').read_text())
        self.argv = ['-c', *self.job['command'][3:]]

    def context(self, root=None, job=None, manifest=None):
        return driver._generated_cifar_manifest_expectation(
            self.root if root is None else root,
            self.job if job is None else job,
            self.manifest if manifest is None else manifest)

    def test_full_fixture_has_formal_cifar_cardinality_without_changing_tasks(self):
        self.assertEqual(500, self.manifest['train_per_class'])
        self.assertEqual(100, self.manifest['test_per_class'])
        self.assertEqual(450, self.plan['fixtures']['image']['remaining_train_per_class'])
        self.assertEqual([[i, i + 1] for i in range(0, 20, 2)], self.manifest['task_classes'])

    def rewrite_fixture(self, name, payload):
        path = self.root / self.plan['fixtures']['image']['files'][name]['path']
        path.chmod(0o644)
        path.write_bytes(payload)
        path.chmod(0o444)
        record = self.plan['fixtures']['image']['files'][name]
        record.update(sha256=driver._sha256_bytes(payload), size=len(payload))
        FormalDriverTests._rewrite(self.root / 'SMOKE_PLAN.json', self.plan)

    def test_context_installs_verified_expectation_and_restores_exact_object(self):
        import three_dataset_formal_audit as audit
        original = audit._AUTHORITATIVE_MANIFEST['cifar100']
        vector = audit._AUTHORITATIVE_MANIFEST['isolet']
        for failure in (None, RuntimeError('body failure'), SystemExit(19)):
            with self.subTest(failure=failure):
                try:
                    with self.context() as validation:
                        expected = audit._AUTHORITATIVE_MANIFEST['cifar100']
                        self.assertIsNot(original, expected)
                        self.assertEqual(validation['sha256'], expected['sha256'])
                        self.assertIs(vector, audit._AUTHORITATIVE_MANIFEST['isolet'])
                        self.assertEqual(500, len(set(validation['ordered_indices'])))
                        self.assertEqual(driver._digest(validation['ordered_indices']),
                                         validation['sha256'])
                        if failure:
                            raise failure
                except BaseException as error:
                    if failure is None or error is not failure:
                        raise
                self.assertIs(original, audit._AUTHORITATIVE_MANIFEST['cifar100'])

    def test_context_rejects_unbound_plan_job_and_noncanonical_root(self):
        import three_dataset_formal_audit as audit
        original = audit._AUTHORITATIVE_MANIFEST['cifar100']
        bad_job = {**self.job, 'command_sha256': '0' * 64}
        for candidate in (bad_job, self.plan['jobs'][-1]):
            with self.subTest(job=candidate['job_id']), self.assertRaises(ValueError):
                with self.context(job=candidate):
                    self.fail('unbound job entered the context')
        for root in (str(self.root) + '/.', str(self.root) + '//',
                     str(self.root / 'child' / '..')):
            with self.subTest(root=root), self.assertRaises(ValueError):
                with self.context(root=root):
                    self.fail('noncanonical root entered the context')
        path = self.root / 'SMOKE_PLAN.json'
        for value in ({**self.plan, 'source_commit': '0' * 40},
                      {**self.plan, 'kind': 'formal_plan'}):
            FormalDriverTests._rewrite(path, value)
            with self.assertRaises(ValueError), self.context():
                self.fail('invalid plan entered the context')
        path.chmod(0o644)
        path.write_text(json.dumps(self.plan, indent=2))
        path.chmod(0o444)
        with self.assertRaisesRegex(ValueError, 'canonical'), self.context():
            self.fail('noncanonical plan entered the context')
        self.assertIs(original, audit._AUTHORITATIVE_MANIFEST['cifar100'])

    def test_context_rejects_manifest_metadata_and_file_identity_changes(self):
        path = self.fixture / 'fixture_manifest.json'
        original_bytes = path.read_bytes()
        changes = (
            ('classes', list(range(19))), ('task_classes', [[0, 1], [2, 3]]),
            ('task_classes', self.manifest['task_classes'][::-1]),
            ('train_per_class', 52), ('train_per_class', 499), ('test_per_class', 2),
            ('schema_version', True), ('generated_only', False))
        for key, value in changes:
            candidate = {**self.manifest, key: value}
            self.rewrite_fixture('manifest', _canonical(candidate) + b'\n')
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                with self.context(manifest=candidate):
                    self.fail('invalid fixture metadata entered the context')
        self.rewrite_fixture('manifest', original_bytes)
        for kind in ('symlink', 'hardlink'):
            with self.subTest(kind=kind):
                target = self.root / f'{kind}-manifest.json'
                path.rename(target)
                if kind == 'symlink':
                    path.symlink_to(target)
                else:
                    os.link(target, path)
                try:
                    with self.assertRaises(ValueError), self.context():
                        self.fail('linked manifest entered the context')
                finally:
                    path.unlink()
                    target.rename(path)
        candidate = copy.deepcopy(self.manifest)
        candidate['files']['train']['sha256'] = '0' * 64
        with self.assertRaises(ValueError), self.context(manifest=candidate):
            self.fail('unbound manifest entered the context')

    def test_context_rejects_nonprivate_root_permissions(self):
        self.root.chmod(0o755)
        try:
            with self.assertRaisesRegex(ValueError, 'root'), self.context():
                self.fail('nonprivate root entered the context')
        finally:
            self.root.chmod(0o700)

    @mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_full_self_consistent_double_slash_plan_rejects_before_any_io(self, _stdout):
        import three_dataset_formal_audit as audit
        alias_root = Path('/' + str(self.root))
        plan = driver._smoke_plan_payload(alias_root, self.plan['fixtures'])
        FormalDriverTests._rewrite(self.root / 'SMOKE_PLAN.json', plan)
        job = driver._smoke_job(plan, 'cifar100-finetune')
        fixture = alias_root / 'fixtures/image'
        run = alias_root / job['run_dir']
        argv = ['-c', *job['command'][3:]]
        # The entire installed plan, generated command, fixture and run/config
        # agree on //tmp/...; rejection must not rely on a mismatched plan.
        self.assertEqual(plan, driver._validate_smoke_plan(alias_root))
        with mock.patch('sys.argv', [job['command'][4], *argv[3:]]):
            args = experiment_config.get_config()
        self.assertEqual(str(run), args.output_dir)
        self.assertEqual(str(fixture), args.data_path)

        def snapshot():
            return {
                str(path.relative_to(self.root)): (
                    driver._identity(path.stat()),
                    driver._sha256_bytes(path.read_bytes()) if path.is_file() else None)
                for path in (self.root, *self.root.rglob('*'))}

        before = snapshot()
        original = audit._AUTHORITATIVE_MANIFEST['cifar100']
        context = driver._generated_cifar_manifest_expectation
        for caller in ('context', 'entry', 'producer'):
            with self.subTest(caller=caller), mock.patch('sys.argv', argv), \
                    mock.patch.object(driver, '_PinnedRoot', wraps=driver._PinnedRoot) as opened, \
                    mock.patch.object(driver, '_validate_smoke_plan',
                                      wraps=driver._validate_smoke_plan) as read_plan, \
                    mock.patch.object(driver, '_generated_cifar_manifest_expectation',
                                      wraps=context) as installed, \
                    mock.patch.object(driver.runpy, 'run_path') as main:
                with self.assertRaisesRegex(ValueError, 'leading slash'):
                    if caller == 'context':
                        with context(alias_root, job, self.manifest):
                            self.fail('double-slash root installed the context')
                    elif caller == 'entry':
                        driver._generated_image_entry(str(fixture), job['command'][4])
                    else:
                        driver._strict_smoke_producer_audit(run, job)
                opened.assert_not_called()
                read_plan.assert_not_called()
                installed.assert_not_called()
                main.assert_not_called()
                self.assertEqual(before, snapshot())
                self.assertIs(original, audit._AUTHORITATIVE_MANIFEST['cifar100'])

    def test_context_rejects_rehashed_pickle_shape_and_label_order(self):
        payload = (self.fixture / 'cifar-100-python/train').read_bytes()
        for key, changed in (
                ('data', np.zeros((10000, 3071), dtype=np.uint8)),
                ('fine_labels', [label for label in range(20) for _ in range(500)][::-1])):
            candidate = pickle.loads(payload)
            candidate[key] = changed
            encoded = driver._pickle_bytes(candidate)
            self.rewrite_fixture('train', encoded)
            manifest = copy.deepcopy(self.manifest)
            manifest['files']['train'].update(
                sha256=driver._sha256_bytes(encoded), md5=driver._md5_bytes(encoded),
                size=len(encoded))
            self.rewrite_fixture('manifest', _canonical(manifest) + b'\n')
            with self.subTest(key=key), self.assertRaises(ValueError):
                with self.context(manifest=manifest):
                    self.fail('changed fixture shape entered the context')

    def test_context_rejects_invalid_rebuilt_split_before_changing_authority(self):
        import calibration_split
        import three_dataset_formal_audit as audit
        build = calibration_split.build_manifest
        original = audit._AUTHORITATIVE_MANIFEST['cifar100']
        for change in ('hash', 'overlap', 'count', 'class'):
            def changed(*args, **kwargs):
                value = build(*args, **kwargs)
                if kwargs.get('excluded_indices') is not None:
                    if change == 'hash':
                        value['sha256'] = '0' * 64
                    elif change == 'overlap':
                        value['by_class']['0'][0] = min(kwargs['excluded_indices'])
                    elif change == 'count':
                        value['by_class']['0'].pop()
                    else:
                        value['by_class']['19'] = value['by_class'].pop('18')
                    if change != 'hash':
                        value['ordered_indices'] = [i for k in sorted(value['by_class'], key=int)
                                                    for i in value['by_class'][k]]
                        value['sha256'] = driver._digest(value['ordered_indices'])
                return value
            with self.subTest(change=change), mock.patch.object(
                    calibration_split, 'build_manifest', side_effect=changed):
                with self.assertRaises(ValueError), self.context():
                    self.fail('invalid split entered the context')
                self.assertIs(original, audit._AUTHORITATIVE_MANIFEST['cifar100'])

    def test_context_rejects_pickle_globals_object_dtype_extra_keys_and_trailing_bytes(self):
        payload = (self.fixture / 'cifar-100-python/train').read_bytes()
        extra = pickle.loads(payload)
        extra['extra'] = True
        objects = pickle.loads(payload)
        objects['data'] = np.zeros((1, 1), dtype=object)
        candidates = (
            b"cos\nsystem\n(S'false'\ntR.",
            driver._pickle_bytes(extra), driver._pickle_bytes(objects), payload + b'extra')
        for encoded in candidates:
            self.rewrite_fixture('train', encoded)
            manifest = copy.deepcopy(self.manifest)
            manifest['files']['train'].update(
                sha256=driver._sha256_bytes(encoded), md5=driver._md5_bytes(encoded),
                size=len(encoded))
            self.rewrite_fixture('manifest', _canonical(manifest) + b'\n')
            with self.subTest(size=len(encoded)), mock.patch('os.system') as execute:
                with self.assertRaises(ValueError), self.context(manifest=manifest):
                    self.fail('unsafe pickle entered the context')
                execute.assert_not_called()

    def test_entry_rejects_unbound_direct_and_aliased_commands_before_main(self):
        commands = [
            [self.job['command'][4], *self.argv[3:]],
            [*self.argv, '--manifest_override', '1'],
            [self.argv[0], str(self.fixture) + '/.', *self.argv[2:]],
            [*self.argv[:2], str(Path(self.job['command'][4]).parent) + '/./main.py',
             *self.argv[3:]],
        ]
        changed = list(self.argv)
        changed[changed.index('--num_tasks') + 1] = '9'
        commands.append(changed)
        for argv in commands:
            with self.subTest(argv=argv[:3]), mock.patch('sys.argv', argv), \
                    mock.patch.object(driver.runpy, 'run_path') as run:
                with self.assertRaises(ValueError):
                    driver._generated_image_entry(argv[1], argv[2])
                run.assert_not_called()
        self.assertEqual([], list((self.root / 'runs').iterdir()))

    def test_nonfull_context_rejects_and_legacy_entry_never_adapts(self):
        for profile in ('formal', 'seed42-pilot', 'seed42-adaptive-recovery'):
            with self.subTest(profile=profile), mock.patch.dict(
                    os.environ, {'VFCL_EXPERIMENT_PROFILE': profile}):
                with self.assertRaises(ValueError), self.context():
                    self.fail('non-full context was accepted')
                for prefix in ('', '/'):
                    fixture = prefix + str(self.fixture)
                    main_path = prefix + self.job['command'][4]
                    with mock.patch.object(driver, '_generated_cifar_manifest_expectation',
                                           side_effect=AssertionError('adapter called')), \
                            mock.patch.object(driver.runpy, 'run_path') as run, \
                            mock.patch('sys.argv', ['-c', fixture, main_path, *self.argv[3:]]):
                        driver._generated_image_entry(fixture, main_path)
                        run.assert_called_once()

    def test_both_smoke_callers_restore_authority_when_the_body_raises(self):
        import adaptive_consolidation_audit as deferred
        import three_dataset_formal_audit as audit
        original = audit._AUTHORITATIVE_MANIFEST['cifar100']

        def fail_in_context(*_args, **_kwargs):
            self.assertIsNot(original, audit._AUTHORITATIVE_MANIFEST['cifar100'])
            raise RuntimeError('verified body failed')

        with mock.patch('sys.argv', self.argv), \
                mock.patch.object(driver.runpy, 'run_path', side_effect=fail_in_context):
            with self.assertRaisesRegex(RuntimeError, 'verified body failed'):
                driver._generated_image_entry(str(self.fixture), self.job['command'][4])
        self.assertIs(original, audit._AUTHORITATIVE_MANIFEST['cifar100'])
        with mock.patch('sys.argv', [self.job['command'][4], *self.argv[3:]]):
            args = experiment_config.get_config()
        run = Path(args.output_dir)
        for name in ('FORMAL_EVALUATION_PUBLISHING.json', 'FORMAL_EVALUATION_PUBLISHED.json'):
            (run / name).write_text('{}')
        with mock.patch.object(deferred, '_load_sealed_complete_artifact',
                               side_effect=fail_in_context):
            with self.assertRaisesRegex(RuntimeError, 'verified body failed'):
                driver._strict_smoke_producer_audit(run, self.job)
        self.assertIs(original, audit._AUTHORITATIVE_MANIFEST['cifar100'])

    def test_producer_rejects_unbound_root_job_and_config_before_context(self):
        with mock.patch('sys.argv', [self.job['command'][4], *self.argv[3:]]):
            args = experiment_config.get_config()
        run = Path(args.output_dir)
        config_path = run / 'config.json'
        config = json.loads(config_path.read_text())
        for candidate in ({**config, 'output_dir': str(self.root)},
                          {**config, 'data_path': '/official/cifar100'},
                          {**config, 'num_classes': 100},
                          {**config, 'bic_enabled': 0}):
            config_path.write_text(json.dumps(candidate))
            with mock.patch.object(driver, '_generated_cifar_manifest_expectation') as context:
                with self.assertRaises(ValueError):
                    driver._strict_smoke_producer_audit(run, self.job)
                context.assert_not_called()
        config_path.write_text(json.dumps(config))
        for candidate_run, candidate_job in (
                (str(run) + '/.', self.job),
                (run, {**self.job, 'command_sha256': '0' * 64}),
                (self.root / 'formal-run', self.job)):
            with mock.patch.object(driver, '_generated_cifar_manifest_expectation') as context:
                with self.assertRaises(ValueError):
                    driver._strict_smoke_producer_audit(candidate_run, candidate_job)
                context.assert_not_called()
        (self.root / 'SMOKE_PLAN.json').unlink()
        with mock.patch.object(driver, '_generated_cifar_manifest_expectation') as context:
            with self.assertRaises(ValueError):
                driver._strict_smoke_producer_audit(run, self.job)
            context.assert_not_called()

    @mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_full_generated_finetune_reaches_real_deferred_publication(self, output):
        import adaptive_consolidation_audit as deferred
        import three_dataset_formal_audit as audit
        from models import TopModel

        def cpu_models(args):
            args.device = 'cpu'
            bottoms = [torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                       torch.nn.Flatten(), torch.nn.Linear(3, 4))
                       for _ in range(args.num_parties)]
            return bottoms, TopModel(
                4 * (args.num_parties if args.aggregation == 'concat' else 1),
                args.num_classes)

        original_parser = experiment_config.get_config

        def cpu_config():
            args = original_parser()
            args.device = 'cpu'
            return args

        original = audit._AUTHORITATIVE_MANIFEST['cifar100']
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        error = None
        # Only the CPU compute scale changes. Dataset, Finetune, all ten runner
        # events, BiC, deferred evaluation, validators and publication are real.
        with mock.patch('sys.argv', self.argv), \
                mock.patch.object(experiment_config, 'get_config', cpu_config), \
                mock.patch.object(runner, 'build_models', cpu_models), \
                mock.patch.object(deferred, 'build_models', cpu_models), \
                mock.patch('torch.cuda.is_available', return_value=False):
            try:
                driver._generated_image_entry(str(self.fixture), self.job['command'][4])
            except ValueError as caught:
                error = caught
        run = self.root / self.job['run_dir']
        self.assertEqual(10, len(list((run / 'formal_snapshots').glob('event_*_CIL.pt'))),
                         (error, output.getvalue()))
        self.assertEqual(10, output.getvalue().count('-> evaluation deferred'))
        self.assertIs(original, audit._AUTHORITATIVE_MANIFEST['cifar100'])
        if error is not None:
            self.assertEqual([], list((self.root / 'records').iterdir()))
        self.assertIsNone(error, f'real ten-stage deferred evaluation failed: '
                          f'{error}; cause: {error.__cause__ if error else None}')
        self.assertTrue((run / 'FORMAL_EVALUATION_PUBLISHED.json').is_file())
        results = json.loads((run / 'results.json').read_text())
        self.assertEqual(10, len(results['step_results']))
        self.assertTrue(results['selection_audit']['passed'])
        self.assertEqual(9000, results['selection_audit']['training_count'])
        self.assertEqual(500, results['selection_audit']['calibration_count'])
        self.assertEqual(500, results['selection_audit']['validation_count'])
        with self.context() as expected:
            self.assertEqual(expected, json.loads(
                (run / 'validation/validation_manifest.json').read_text()))
        self.assertIs(original, audit._AUTHORITATIVE_MANIFEST['cifar100'])
        complete = json.loads((run / 'FORMAL_EVALUATION_COMPLETE.json').read_text())
        self.assertEqual(2000, complete['cache_identity']['test']['sample_count'])
        proof = driver._strict_smoke_producer_audit(run, self.job)
        self.assertTrue(proof['published'])
        self.assertIs(original, audit._AUTHORITATIVE_MANIFEST['cifar100'])
        driver.smoke_begin(self.root, self.job['job_id'], 0)
        driver._install_bytes_exclusive(
            self.root / 'logs' / f"{self.job['job_id']}.log", output.getvalue().encode())
        # Keep the real resume checker; replace only its GPU child transport.
        with mock.patch('models.build_models', cpu_models), \
                mock.patch('torch.cuda.is_available', return_value=False), \
                mock.patch.object(driver, 'resume_probe', side_effect=lambda path, gpu:
                                  driver._strict_resume_probe_main(path)):
            record = driver.install_smoke_record(self.root, self.job['job_id'], run)
        self.assertTrue(record['passed'])
        self.assertTrue(record['generated_only'])
        self.assertFalse(record['scientific_gate'])
        self.assertEqual(10, record['resume_probe']['start_event_idx'])
        self.assertTrue(record['resume_probe']['source_unchanged'])
        self.assertFalse(record['resume_probe']['test_loader_opened'])
        self.assertEqual(1, len(list((self.root / 'records').iterdir())))
        self.assertIs(original, audit._AUTHORITATIVE_MANIFEST['cifar100'])


class MethodResourceStateTests(unittest.TestCase):
    """Resource semantics use real method get_state contracts, without training."""

    def setUp(self):
        self.profile = mock.patch.dict(
            os.environ, {'VFCL_EXPERIMENT_PROFILE': 'full-public-matrix'})
        self.profile.start()
        self.addCleanup(self.profile.stop)
        self.temporary = tempfile.TemporaryDirectory(prefix='method_resource_')
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'formal_final.pt'
        self.rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        self.addCleanup(random.setstate, self.rng[0])
        self.addCleanup(np.random.set_state, self.rng[1])
        self.addCleanup(torch.set_rng_state, self.rng[2])

    def _fixture(self, method, dim=3, spec=None):
        from cl_methods import get_cl_method
        from cl_methods.adagauss import _Adapter
        from cl_methods.target import ConditionalGenerator
        from models import TopModel

        full_class_store = spec is not None
        spec = spec or FormalSpec('isolet', method, 42)
        protocol = protocol_for(spec)
        args = SimpleNamespace(**{
            **protocol['base_options'], **protocol['method_contract'],
            'device': 'cpu', 'seed': 42,
        })
        top_dim = (dim * args.num_parties
                   if method == 'fedprotip_vfl' and args.aggregation == 'concat' else dim)
        trainer = SimpleNamespace(
            top_model=TopModel(top_dim, args.num_classes, cosine=False),
            bottoms=[torch.nn.Linear(2, dim) for _ in range(args.num_parties)],
            evaluate=lambda *args, **kwargs: None,
        )
        source = get_cl_method(args.cl_method, trainer, args)
        samples = torch.arange(2 * args.num_classes * 4, dtype=torch.float32).reshape(-1, 4)
        labels = torch.arange(args.num_classes).repeat_interleave(2)
        added = 0
        if method in {'er', 'er_ace'}:
            source.buffer.add_batch(samples, labels)
        elif method == 'der_pp':
            for x, y in zip(samples, labels):
                source.buffer.add(x, y, torch.ones(args.num_classes))
        elif method == 'proto_fedspace':
            source.protos = {c: torch.ones(dim) for c in (0, 1, 2)}
            source.radius = np.float64(0.5)
        elif args.cl_method == 'proto_evolve':
            prototype_count = args.num_classes if full_class_store else 3
            source.global_protos = {
                c: {'mean': torch.ones(dim), 'std': torch.ones(dim)}
                for c in range(prototype_count)}
            source.prev_protos = copy.deepcopy(source.global_protos)
            if args.head_consolidation_enabled:
                source.head_raw_replay = {
                    c: chunk for c, chunk in enumerate(samples.chunk(prototype_count))}
        elif method == 'adagauss':
            source.gaussians = {
                c: {'mean': torch.ones(dim), 'cov': torch.eye(dim)}
                for c in (0, 1, 2)}
            source.adapters = [_Adapter(dim), _Adapter(dim)]
            added = sum(p.numel() for a in source.adapters for p in a.parameters())
        elif method == 'target':
            source.task_classes = {0: [0, 1], 1: [2]}
            source.generators = {
                t: ConditionalGenerator(len(classes), dim, noise_dim=5, hidden=7)
                for t, classes in source.task_classes.items()}
            source.forgotten = {1}
            source.fim_masks = [{'weight': True} for _ in trainer.bottoms]
            added = sum(p.numel() for g in source.generators.values()
                        for p in g.parameters())
        elif method == 'ewc':
            source.fisher = [{'weight': torch.ones(dim, 2)} for _ in trainer.bottoms]
            source.old_params = [{'weight': torch.zeros(dim, 2)} for _ in trainer.bottoms]
        elif method in {'gpm', 'fedprotip_vfl'}:
            source.feature_lists = [{'weight': torch.eye(2)} for _ in trainer.bottoms]
            source.head_basis = torch.eye(top_dim)
            if method == 'fedprotip_vfl':
                source.task_classes = {0: [0, 1], 1: [2]}
                source.task_means = [{0: torch.zeros(dim), 1: torch.ones(dim)}
                                     for _ in trainer.bottoms]
                source.task_bases = [{0: torch.eye(dim), 1: torch.eye(dim)}
                                     for _ in trainer.bottoms]
        elif method == 'afc':
            source.importance = torch.ones(dim)
            source._n_seen_classes = args.num_classes
        checkpoint = {
            'schema_version': 4,
            'protocol': {'cl_method': args.cl_method, 'num_parties': args.num_parties},
            'trainer_state': {
                'bottoms': [bottom.state_dict() for bottom in trainer.bottoms],
                'top_model': trainer.top_model.state_dict(),
            },
            'cl_state': runner._encode_checkpoint_value(source.get_state()),
        }
        return spec, checkpoint, added

    def _measure(self, spec, checkpoint):
        torch.save(checkpoint, self.path)
        self.assertTrue(callable(getattr(driver, '_method_resource_state', None)),
                        'checkpoint-derived method resource classifier is missing')
        return driver._method_resource_state(spec, self.path)

    def test_all_fourteen_methods_use_actual_persisted_state(self):
        none = ('none', 0, 0, 'no-persistent-raw-or-embedding-replay')
        expected = dict.fromkeys(driver.formal_registry.FULL_MATRIX_METHODS, none)
        expected.update({
            'er': ('raw-examples', 2, 0, 'persistent-raw-example-replay'),
            'der_pp': ('reservoir-raw-examples-and-logits', 2, 0,
                       'persistent-raw-example-replay'),
            'er_ace': ('reservoir-raw-examples', 2, 0, 'persistent-raw-example-replay'),
            'proto_fedspace': ('class-prototype-embeddings', 0, 3,
                               'persistent-derived-embedding-replay'),
            'adaptive': ('raw-examples-and-class-prototype-embeddings', 2, 3,
                         'persistent-raw-and-derived-embedding-replay'),
            'adagauss': ('class-gaussian-statistics', 0, 3,
                         'persistent-derived-statistics-replay'),
            'target': ('synthetic-generator', 0, 0, 'persistent-synthetic-generator-replay'),
        })
        self.assertEqual(14, len(expected))
        for method, values in expected.items():
            with self.subTest(method=method):
                spec, checkpoint, added = self._fixture(method)
                replay, raw, embeddings, privacy = values
                self.assertEqual({
                    'replay_type': replay, 'raw_examples_per_class': raw,
                    'persistent_embeddings': embeddings, 'added_parameters': added,
                    'privacy_label': privacy,
                }, self._measure(spec, checkpoint))

    def test_added_parameters_count_every_persisted_network_at_actual_width(self):
        for method in ('adagauss', 'target'):
            for dim in (2, 5):
                with self.subTest(method=method, dim=dim):
                    spec, checkpoint, added = self._fixture(method, dim)
                    self.assertGreater(added, 0)
                    self.assertEqual(added, self._measure(spec, checkpoint)['added_parameters'])

    def test_pipeline_fixture_respects_cifar100_adaptive_raw_replay_capacity(self):
        spec = FormalSpec('cifar100', 'adaptive', 42)
        _, checkpoint, _ = self._fixture(spec.method, spec=spec)
        resource = self._measure(spec, checkpoint)
        self.assertEqual('raw-examples-and-class-prototype-embeddings', resource['replay_type'])
        self.assertEqual(2, resource['raw_examples_per_class'])
        self.assertEqual(100, resource['persistent_embeddings'])

    def test_adaptive_rejects_unsupported_method_and_top_versions(self):
        from adaptive_head_consolidation import ADAPTIVE_METHOD_VERSION

        for field in ('adaptive_method_version', 'adaptive_top_version'):
            for version in (-1, ADAPTIVE_METHOD_VERSION + 1):
                spec, checkpoint, _ = self._fixture('adaptive')
                checkpoint['cl_state'][field] = version
                with self.subTest(field=field, version=version), self.assertRaises(ValueError):
                    self._measure(spec, checkpoint)

    def test_adagauss_rejects_indefinite_or_asymmetric_covariance(self):
        for covariance in (torch.tensor([[1., 2., 0.], [2., 1., 0.], [0., 0., 1.]]),
                           torch.tensor([[1., 1., 0.], [0., 1., 0.], [0., 0., 1.]])):
            spec, checkpoint, _ = self._fixture('adagauss')
            checkpoint['cl_state']['gaussians'][0]['cov'] = covariance
            with self.subTest(covariance=covariance), self.assertRaises(ValueError):
                self._measure(spec, checkpoint)

    def test_adagauss_accepts_semidefinite_covariance_with_consumer_jitter(self):
        spec, checkpoint, _ = self._fixture('adagauss')
        checkpoint['cl_state']['gaussians'][0]['cov'] = torch.zeros(3, 3)
        self.assertEqual(3, self._measure(spec, checkpoint)['persistent_embeddings'])

    def test_proto_fedspace_requires_supported_model_dtype_and_embedding_width(self):
        for dtype, dim in ((torch.float64, 3), (torch.float16, 3), (torch.float32, 4)):
            spec, checkpoint, _ = self._fixture('proto_fedspace')
            checkpoint['cl_state']['protos'] = {c: torch.ones(dim, dtype=dtype)
                                                for c in (0, 1, 2)}
            with self.subTest(dtype=dtype, dim=dim), self.assertRaises(ValueError):
                self._measure(spec, checkpoint)
        spec, checkpoint, _ = self._fixture('proto_fedspace')
        checkpoint['trainer_state']['top_model']['classifier.weight'] = torch.ones(26, 3).double()
        checkpoint['cl_state']['protos'] = {c: torch.ones(3).double() for c in (0, 1, 2)}
        with self.assertRaises(ValueError):
            self._measure(spec, checkpoint)

    def test_ewc_final_state_requires_nonempty_positive_trace_per_party(self):
        for mutation in ('all_empty', 'one_empty', 'zero_trace', 'empty_anchors'):
            spec, checkpoint, _ = self._fixture('ewc')
            state = checkpoint['cl_state']
            if mutation == 'all_empty':
                state['fisher'] = [{} for _ in state['fisher']]
                state['old_params'] = [{} for _ in state['old_params']]
            elif mutation == 'one_empty':
                state['fisher'][0] = state['old_params'][0] = {}
            elif mutation == 'zero_trace':
                state['fisher'][0] = {k: torch.zeros_like(v) for k, v in state['fisher'][0].items()}
            else:
                state['old_params'][0] = {}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self._measure(spec, checkpoint)

    def test_afc_final_importance_requires_model_compatible_nonempty_vector(self):
        for field, value in (('importance', None), ('importance', torch.empty(0)),
                             ('importance', torch.ones(4)),
                             ('importance', torch.ones(3).double()), ('n_seen', 0)):
            spec, checkpoint, _ = self._fixture('afc')
            checkpoint['cl_state'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self._measure(spec, checkpoint)

    def test_dimension_checks_require_supported_checkpoint_model_evidence(self):
        for method in ('proto_fedspace', 'afc', 'fedprotip_vfl'):
            for bad_trainer in (None, {}, {'top_model': None},
                                {'top_model': {'classifier.weight': torch.ones(25, 3)}}):
                spec, checkpoint, _ = self._fixture(method)
                checkpoint['trainer_state'] = bad_trainer
                with self.subTest(method=method, trainer=bad_trainer), self.assertRaises(ValueError):
                    self._measure(spec, checkpoint)

    def test_fedprotip_reference_basis_must_fit_persisted_model_dimension(self):
        for mutation in ('overwide', 'task_width_changed', 'party_width_changed'):
            spec, checkpoint, _ = self._fixture('fedprotip_vfl')
            state = checkpoint['cl_state']
            if mutation == 'overwide':
                state['task_bases'][0][0] = torch.ones(3, 4)
            else:
                tasks = (0,) if mutation == 'task_width_changed' else (0, 1)
                for tid in tasks:
                    state['task_bases'][0][tid] = torch.eye(4)
                    state['task_means'][0][tid] = torch.ones(4)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self._measure(spec, checkpoint)

    def test_fedprotip_rejects_dtype_mismatch_that_breaks_relevance_consumer(self):
        from cl_methods.fedprotip_vfl import normalized_relevance

        spec, checkpoint, _ = self._fixture('fedprotip_vfl')
        state = checkpoint['cl_state']
        state['task_bases'][0][0] = state['task_bases'][0][0].double()
        with self.assertRaisesRegex(RuntimeError, 'dtype|scalar type'):
            normalized_relevance(torch.ones(2, 3), state['task_means'][0][0],
                                 state['task_bases'][0][0])
        with self.assertRaises(ValueError):
            self._measure(spec, checkpoint)

    def test_fedprotip_reference_dtype_checks_preserve_checkpoint_and_rng(self):
        for mutation in ('valid_float32', 'mean_only', 'basis_only', 'one_task_pair',
                         'one_party', 'all_references', 'references_and_model'):
            spec, checkpoint, _ = self._fixture('fedprotip_vfl')
            self.assertEqual('concat', protocol_for(spec)['base_options']['aggregation'])
            state = checkpoint['cl_state']
            for key in ('task_means', 'task_bases'):
                for party_id, party in enumerate(state[key]):
                    for tid, value in party.items():
                        convert = (
                            mutation in ('all_references', 'references_and_model')
                            or (party_id == 0 and mutation == 'one_party')
                            or (party_id == 0 and tid == 0 and (
                                mutation == 'one_task_pair'
                                or (mutation == 'mean_only' and key == 'task_means')
                                or (mutation == 'basis_only' and key == 'task_bases'))))
                        if convert:
                            party[tid] = value.double()
            if mutation == 'references_and_model':
                model = checkpoint['trainer_state']['top_model']
                model['classifier.weight'] = model['classifier.weight'].double()
            torch.save(checkpoint, self.path)
            before = self.path.read_bytes()
            original = copy.deepcopy(checkpoint)
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            with self.subTest(mutation=mutation):
                if mutation == 'valid_float32':
                    self.assertEqual('none', driver._method_resource_state(spec, self.path)['replay_type'])
                else:
                    with self.assertRaises(ValueError):
                        driver._method_resource_state(spec, self.path)
                self.assertEqual(before, self.path.read_bytes())
                self.assertTrue(runner._checkpoint_values_equal(original, checkpoint))
                self.assertTrue(runner._checkpoint_values_equal(
                    rng, (random.getstate(), np.random.get_state(), torch.get_rng_state())))

    def test_real_zero_energy_after_task_preserves_legal_gpm_and_afc_state(self):
        from cl_methods import get_cl_method
        from models import TopModel
        from vfl_trainer import VFLTrainer

        for name in ('gpm', 'afc', 'ewc'):
            spec, checkpoint, _ = self._fixture(name)
            protocol = protocol_for(spec)
            count = protocol['base_options']['num_parties']
            args = SimpleNamespace(**{
                **protocol['method_contract'], 'device': 'cpu', 'aggregation': 'sum',
                'num_parties': count, 'num_classes': 26, 'data': 'synthvfl',
                'party_col_ranges': [(2 * i, 2 * i + 2) for i in range(count)],
            })
            trainer = VFLTrainer([torch.nn.Linear(2, 3) for _ in range(count)],
                                 TopModel(3, 26, cosine=False), args)
            for model in [*trainer.bottoms, trainer.top_model]:
                for parameter in model.parameters():
                    torch.nn.init.zeros_(parameter)
            method = get_cl_method(name, trainer, args)
            method.before_task(0, [0, 1], [0, 1])
            batches = [(torch.zeros(2, 2 * count), torch.tensor([0, 1]))]
            with self.subTest(method=name):
                if name == 'ewc':
                    with self.assertRaisesRegex(FloatingPointError, 'trace.*positive'):
                        method.after_task(batches, 0)
                    continue
                method.after_task(batches, 0)
                state = method.get_state()
                if name == 'gpm':
                    self.assertEqual([{} for _ in range(count)], state['feature_lists'])
                    self.assertIsNone(state['head_basis'])
                else:
                    torch.testing.assert_close(state['importance'], torch.zeros(3))
                checkpoint['cl_state'] = runner._encode_checkpoint_value(state)
                checkpoint['trainer_state'] = trainer.get_state()
                result = self._measure(spec, checkpoint)
                self.assertEqual('none', result['replay_type'])
                self.assertEqual(0, result['persistent_embeddings'])
                self.assertEqual(0, result['added_parameters'])

    def test_prototype_only_state_remains_distinct_from_adaptive_raw_replay(self):
        for profile, method in (('full-public-matrix', 'adaptive'),
                                ('formal', 'no_consolidation')):
            with self.subTest(method=method), mock.patch.dict(
                    os.environ, {'VFCL_EXPERIMENT_PROFILE': profile}):
                spec, checkpoint, _ = self._fixture(method)
                checkpoint['cl_state']['head_raw_replay'] = {}
                self.assertEqual({
                    'replay_type': 'class-prototype-embeddings',
                    'raw_examples_per_class': 0, 'persistent_embeddings': 3,
                    'added_parameters': 0,
                    'privacy_label': 'persistent-derived-embedding-replay',
                }, self._measure(spec, checkpoint))

    def test_persisted_network_tensors_reject_nonfinite_shape_and_key_corruption(self):
        for method in ('adagauss', 'target'):
            spec, checkpoint, _ = self._fixture(method)
            for mutation in ('nan', 'shape', 'extra', 'dtype'):
                payload = copy.deepcopy(checkpoint)
                state = payload['cl_state']
                network = (state['adapters'][0] if method == 'adagauss'
                           else state['generators'][0]['state_dict'])
                if mutation == 'nan':
                    network['net.0.weight'][0, 0] = float('nan')
                elif mutation == 'shape':
                    network['net.0.weight'] = torch.ones(1, 1)
                elif mutation == 'extra':
                    network['running_buffer'] = torch.ones(3)
                else:
                    network['net.0.weight'] = network['net.0.weight'].double()
                with self.subTest(method=method, mutation=mutation), self.assertRaises(ValueError):
                    self._measure(spec, payload)

    def test_formal_replay_bearing_methods_reject_empty_stores(self):
        for method, fields in (
                ('er', {'data': {}, 'seen_count': {}}),
                ('der_pp', {'examples': None, 'labels': torch.empty(0, dtype=torch.long),
                            'logits': None, 'num_seen': 0}),
                ('er_ace', {'examples': None, 'labels': torch.empty(0, dtype=torch.long),
                            'num_seen': 0}),
                ('proto_fedspace', {'protos': {}}),
                ('adaptive', {'global_protos': {}, 'prev_protos': {}, 'head_raw_replay': {}}),
                ('adagauss', {'gaussians': {}, 'adapters': []}),
                ('target', {'task_classes': {}, 'generators': {}, 'forgotten': []})):
            spec, checkpoint, _ = self._fixture(method)
            checkpoint['cl_state'].update(fields)
            with self.subTest(method=method), self.assertRaises(ValueError):
                self._measure(spec, checkpoint)

    def test_prototype_caches_require_compatible_membership_width_and_dtype(self):
        for field, value in (
                ('prev_protos', {3: {'mean': torch.ones(3), 'std': torch.ones(3)}}),
                ('head_raw_replay', {3: torch.ones(13, 4), 4: torch.ones(13, 4)}),
                ('global_protos', {0: {'mean': torch.ones(3), 'std': torch.ones(3).double()}})):
            spec, checkpoint, _ = self._fixture('adaptive')
            checkpoint['cl_state'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self._measure(spec, checkpoint)

    def test_checkpoint_numpy_encoding_is_decoded_before_state_validation(self):
        spec, checkpoint, _ = self._fixture('proto_fedspace')
        checkpoint['cl_state']['radius'] = runner._encode_checkpoint_value(np.float64(0.5))
        self.assertEqual(3, self._measure(spec, checkpoint)['persistent_embeddings'])
        checkpoint['cl_state']['radius']['__vfcl_numpy_scalar_v1__'] = False
        with self.assertRaises(ValueError):
            self._measure(spec, checkpoint)

    def test_loader_is_cpu_weights_only_and_does_not_change_rng_or_checkpoint(self):
        for method in driver.formal_registry.FULL_MATRIX_METHODS:
            spec, checkpoint, _ = self._fixture(method)
            torch.save(checkpoint, self.path)
            before = self.path.read_bytes()
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            self.assertTrue(callable(getattr(driver, '_method_resource_state', None)))
            with self.subTest(method=method), mock.patch('torch.load', wraps=torch.load) as load:
                driver._method_resource_state(spec, self.path)
                load.assert_called_once_with(self.path, map_location='cpu', weights_only=True)
                self.assertEqual(before, self.path.read_bytes())
                self.assertTrue(runner._checkpoint_values_equal(
                    rng, (random.getstate(), np.random.get_state(), torch.get_rng_state())))

    def test_checkpoint_schema_protocol_and_method_incompatible_state_fail_closed(self):
        for method in driver.formal_registry.FULL_MATRIX_METHODS:
            spec, checkpoint, _ = self._fixture(method)
            other_state = ({'protos': {0: torch.ones(3)}, 'radius': 1.0}
                           if method in ('finetune', 'lwf', 'lwf_wa') else {})
            invalid = [
                {**checkpoint, 'schema_version': 3},
                {**checkpoint, 'schema_version': 4.0},
                {**checkpoint, 'protocol': {'cl_method': 'unknown'}},
                {**checkpoint, 'protocol': None},
                {**checkpoint, 'cl_state': []},
                {**checkpoint, 'cl_state': other_state},
                {**checkpoint, 'cl_state': {**checkpoint['cl_state'], 'unexpected': 0}},
            ]
            for index, payload in enumerate(invalid):
                with self.subTest(method=method, mutation=index), self.assertRaises(ValueError):
                    self._measure(spec, payload)

    def test_resource_bearing_states_reject_malformed_or_nonfinite_values(self):
        mutations = {
            'er': [('per_class_size', True), ('data', {0: torch.ones(1, 4)}),
                   ('seen_count', {}), ('data', {0: torch.tensor([[float('nan')]])})],
            'der_pp': [('logits', None), ('logits', torch.ones(52, 3)),
                       ('logits', torch.full((52, 26), float('inf'))),
                       ('num_seen', True), ('labels', torch.zeros(52))],
            'er_ace': [('num_seen', -1), ('examples', torch.ones(1, 4)),
                       ('labels', torch.full((52,), 26, dtype=torch.long))],
            'proto_fedspace': [('radius', float('nan')), ('protos', {26: torch.ones(3)}),
                               ('protos', {0: torch.ones(2, 3)})],
            'adaptive': [('global_protos', {0: {'mean': torch.ones(3)}}),
                         ('head_raw_replay', {0: torch.ones(1, 4)}),
                         ('adaptive_audit_bundle', {float('nan'): 0}),
                         ('prev_protos', {0: {'mean': torch.ones(3), 'std': -torch.ones(3)}})],
            'adagauss': [('gaussians', {0: {'mean': torch.ones(3), 'cov': torch.eye(2)}}),
                         ('adapters', [{'net.0.weight': torch.ones(6, 3)}])],
            'target': [('task_classes', {}), ('generators', {0: {}}),
                       ('fim_masks', [{'weight': 1}])],
            'ewc': [('fisher', []), ('fisher', [{'weight': float('inf')}])],
            'gpm': [('head_basis', torch.ones(3)), ('threshold', float('nan'))],
            'fedprotip_vfl': [('task_means', []), ('max_batches', True)],
            'afc': [('importance', torch.ones(2, 2)), ('n_seen', True)],
        }
        for method, changes in mutations.items():
            spec, checkpoint, _ = self._fixture(method)
            for field, value in changes:
                with self.subTest(method=method, field=field), self.assertRaises(ValueError):
                    self._measure(spec, {**checkpoint, 'cl_state': {
                        **checkpoint['cl_state'], field: value}})

    def test_raw_reservoir_counts_are_exact_integers_not_capacity_or_rounding(self):
        for method in ('der_pp', 'er_ace'):
            spec, checkpoint, _ = self._fixture(method)
            state = checkpoint['cl_state']
            state['labels'].fill_(0)  # Flat reservoirs need not be class-balanced.
            self.assertEqual(2, self._measure(spec, checkpoint)['raw_examples_per_class'])
            for key in ('labels', 'examples', 'logits'):
                if key in state:
                    state[key] = state[key][:-1]
            state['num_seen'] -= 1
            with self.subTest(method=method), self.assertRaises(ValueError):
                self._measure(spec, checkpoint)


if __name__ == '__main__':
    unittest.main()
