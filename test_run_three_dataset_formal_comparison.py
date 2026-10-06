import ast
import copy
import ctypes
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import urllib.parse
from unittest import mock

import three_dataset_formal_driver as driver
from test_three_dataset_formal_driver import _completed_record, _write_resource_checkpoint
from three_dataset_formal_registry import (
    command_for, explanation_specs, formal_specs, safe_spec_name,
)


SCRIPT = Path(__file__).with_name('run_three_dataset_formal_comparison.sh')
PYTHON = '/home/c3080/YangXiaoXiang/envs/vfcl/bin/python'
HEAD = '6f5f6dba06ce52c3d16819ca84384e2b288f9461'
BRANCH = 'codex/three-dataset-seed42-pilot'
RECOVERY_BRANCH = 'codex/adaptive-bic-corpus-identity-fix'
FULL_MATRIX_BRANCH = 'codex/full-matrix-scientific-correctness-fix'
DATASET_BRANCH = 'codex/dual-gpu-dataset-formal'
CONTINUATION_BRANCH = 'codex/gpm-audit-low-memory'


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False,
    )


class DriverLauncherControlTests(unittest.TestCase):
    def test_generated_manifest_adapter_has_only_two_private_smoke_callers(self):
        tree = ast.parse(Path(driver.__file__).read_text())
        callers = []
        for function in tree.body:
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(function):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                            and node.func.id == '_generated_cifar_manifest_expectation'):
                        callers.append(function.name)
        self.assertEqual(['_generated_image_entry', '_strict_smoke_producer_audit'], callers)
        for spec in formal_specs():
            self.assertNotIn('_generated_image_entry', '\n'.join(
                command_for(spec, 'cuda:0', '/formal/results')))
        self.assertNotIn('manifest_expectation', SCRIPT.read_text())

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='formal_launcher_driver_')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = driver.validate_formal_root(self.base / 'formal')
        census = driver.build_census({})
        self.plan = driver.build_plan(census)
        for name, value in (
                ('FORMAL_REGISTRY.json', driver._registry_payload()),
                ('COMPATIBILITY_CENSUS.json', census),
                ('FORMAL_PLAN.json', self.plan),
                ('MISSING_JOBS.json',
                 driver._missing_jobs_payload(self.plan, census))):
            driver.install_json_exclusive(self.root / name, value)
        (self.root / 'claims').mkdir(mode=0o700)
        driver.install_formal_root_identity(self.root, self.plan)
        self.commit = driver._source_commit()

    def owner(self, phase='formal', role='worker-0'):
        return driver.owner_for(
            self.root, phase, 'launcher-token', role,
            os.getpid(), os.getpgid(os.getpid()),
        )

    def install_records(self, specs):
        records = self.root / 'records'
        records.mkdir(exist_ok=True)
        for spec in specs:
            record = _completed_record(
                spec, .5, source_commit=self.commit,
                plan_sha256=driver._digest(self.plan),
            )
            record['record_sha256'] = driver._digest({
                key: value for key, value in record.items()
                if key != 'record_sha256'
            })
            driver.install_json_exclusive(
                records / f'{safe_spec_name(spec)}.json', record)

    def test_phase_claims_use_only_installed_sets_and_cross_phase_rejects(self):
        formal_owner = self.owner('formal')
        first = driver.claim_next(
            self.plan, self.root / 'claims', 'formal', formal_owner)
        self.assertIn(first, self.plan['missing_jobs'])
        with self.assertRaises(ValueError):
            driver.claim_next(
                self.plan, self.root / 'claims', 'explanation', formal_owner)

        explanation_owner = self.owner('explanation', 'worker-1')
        with self.assertRaises(ValueError):
            driver.claim_next(
                self.plan, self.root / 'claims', 'explanation',
                explanation_owner)
        self.install_records(formal_specs())
        self.assertTrue(driver.phase_ready(self.root, 'explanation'))
        explanation = driver.claim_next(
            self.plan, self.root / 'claims', 'explanation',
            explanation_owner)
        expected = {driver.spec_key(spec) for spec in explanation_specs()}
        self.assertIn(explanation, expected)
        wrong = copy.deepcopy(explanation_owner)
        wrong['job'] = self.plan['missing_jobs'][1]
        with self.assertRaises(ValueError):
            driver.claim_next(
                self.plan, self.root / 'claims', 'explanation', wrong)

    def test_owner_started_and_run_controls_bind_explicit_phase_and_pgid(self):
        template = self.owner()
        key = driver.claim_next(
            self.plan, self.root / 'claims', 'formal', template)
        installed_owner = driver.installed_claim_owner(self.root, key)
        run_dir = driver.prepare_run(self.root, key, installed_owner)
        self.assertEqual(
            self.root / 'runs' / safe_spec_name(driver.spec_for_key(key, 'formal')),
            run_dir,
        )
        controls = {
            path.name: json.loads(path.read_text())
            for path in run_dir.iterdir() if path.suffix == '.json'
        }
        self.assertEqual(
            {'FORMAL_JOB_SPEC.json', 'CLAIM_OWNER.json',
             'LAUNCH_STARTED.json'}, set(controls))
        claim = controls['CLAIM_OWNER.json']
        launch = controls['LAUNCH_STARTED.json']
        installed_started = json.loads((
            self.root / 'claims' / safe_spec_name(
                driver.spec_for_key(key, 'formal')) / 'started.json'
        ).read_text())
        for field in ('phase', 'pgid'):
            self.assertEqual(installed_owner[field], claim[field])
            self.assertEqual(installed_owner[field], launch[field])
            self.assertEqual(installed_owner[field], installed_started[field])
        with self.assertRaises(FileExistsError):
            driver.prepare_run(self.root, key, installed_owner)

    def test_command_is_exact_logical_cuda_zero_and_run_name_is_injective(self):
        seen = set()
        for spec in (*formal_specs(), *explanation_specs()):
            run_dir = self.root / 'runs' / safe_spec_name(spec)
            command = driver.command_for_run(driver.spec_key(spec), run_dir)
            self.assertEqual(
                command_for(spec, 'cuda:0', str(self.root / 'runs')),
                command)
            self.assertEqual('cuda:0', command[command.index('--device') + 1])
            self.assertEqual(run_dir.name, command[command.index('--exp_name') + 1])
            self.assertNotIn(run_dir.name, seen)
            seen.add(run_dir.name)
        with self.assertRaises(ValueError):
            driver.command_for_run('../escape', self.root / 'runs' / 'escape')

    def test_phase_ready_requires_all_exact_valid_formal_records(self):
        self.install_records(formal_specs()[:-1])
        self.assertFalse(driver.phase_ready(self.root, 'explanation'))
        self.install_records(formal_specs()[-1:])
        self.assertTrue(driver.phase_ready(self.root, 'explanation'))
        path = self.root / 'records' / f'{safe_spec_name(formal_specs()[0])}.json'
        value = json.loads(path.read_text())
        value['phase'] = 'tampered'
        path.chmod(0o644)
        path.write_text(canonical(value) + '\n')
        path.chmod(0o444)
        with self.assertRaises(ValueError):
            driver.phase_ready(self.root, 'explanation')

    def test_resource_record_uses_explicit_fake_measurements_and_exact_artifacts(self):
        owner = self.owner()
        key = driver.claim_next(
            self.plan, self.root / 'claims', 'formal', owner)
        owner = driver.installed_claim_owner(self.root, key)
        run_dir = driver.prepare_run(self.root, key, owner)
        required = driver.resource_artifact_names(
            driver.spec_for_key(key, 'formal'))
        for logical in required:
            path = run_dir / logical
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'x')
            if logical == 'job.log':
                path.chmod(0o444)
        measurements = {
            'hardware_identity': {
                'gpu_name': 'fake-gpu', 'gpu_count': 1,
                'cuda': 'fake-cuda', 'torch': 'fake-torch',
                'driver': 'fake-driver',
            },
            'runtime_seconds': 1.5,
            'peak_gpu_memory_bytes': 4096,
            'added_parameters': 0,
            'communication_bytes': 0,
            'replay_type': 'none',
            'raw_examples_per_class': 0,
            'persistent_embeddings': 0,
            'privacy_label': 'no-persistent-raw-or-embedding-replay',
        }
        measurements.update(_write_resource_checkpoint(
            driver.spec_for_key(key), run_dir / 'checkpoints/formal_final.pt'))
        evidence = driver.resource_record(
            self.root, key, run_dir, measurements=measurements)
        self.assertEqual('formal_resource_evidence', evidence['kind'])
        self.assertEqual(
            (run_dir / 'checkpoints/formal_final.pt').stat().st_size,
            evidence['resource']['checkpoint_size_bytes'])
        self.assertEqual(
            set(required), set(evidence['artifact_sha256']))
        with self.assertRaises(FileExistsError):
            driver.resource_record(
                self.root, key, run_dir, measurements=measurements)

    def test_owner_phase_pgid_and_safe_name_tamper_reject(self):
        owner = self.owner()
        for field, value in (
                ('phase', 'explanation'), ('pgid', owner['pgid'] + 1),
                ('job', '../escape')):
            tampered = copy.deepcopy(owner)
            tampered[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                driver.claim_next(
                    self.plan, self.root / 'claims', 'formal', tampered)

    def test_resource_hardware_accepts_uniform_driver_identity_from_two_physical_gpus(self):
        run_dir = self.base / 'measurement-run'
        run_dir.mkdir()
        (run_dir / 'results.json').write_text(canonical({
            'comm_stats': [{'megabytes_transmitted': 1.5}],
        }))
        _write_resource_checkpoint(formal_specs()[0], run_dir / 'checkpoints/formal_final.pt')
        completed = mock.Mock(stdout='550.90\n550.90\n')
        with mock.patch('torch.cuda.is_available', return_value=True), \
                mock.patch('torch.cuda.device_count', return_value=1), \
                mock.patch('torch.cuda.get_device_name', return_value='fake'), \
                mock.patch.object(driver.subprocess, 'run', return_value=completed):
            measured = driver._resource_measurements(
                formal_specs()[0], run_dir, 1.0, 4096)
        self.assertEqual('550.90', measured['hardware_identity']['driver'])
        self.assertEqual(1572864, measured['communication_bytes'])

    def test_root_identity_survives_child_evidence_but_replacement_and_token_tamper_reject(self):
        before = driver._root_identity(self.root)
        (self.root / 'runs').mkdir()
        (self.root / 'runs' / 'child').mkdir()
        (self.root / 'FORMAL_PHASE_SUCCESS').write_text('{}\n')
        self.assertEqual(before, driver._root_identity(self.root))
        owner = self.owner()

        replacement = self.base / 'replacement'
        shutil.copytree(self.root, replacement)
        with self.assertRaises(ValueError):
            driver._root_identity(replacement)

        identity = self.root / 'FORMAL_ROOT_IDENTITY.json'
        value = json.loads(identity.read_text())
        value['token'] = '0' * 64
        identity.chmod(0o644)
        identity.write_text(canonical(value) + '\n')
        identity.chmod(0o444)
        self.assertNotEqual(before, driver._root_identity(self.root))
        with self.assertRaises(ValueError):
            driver.claim_next(
                self.plan, self.root / 'claims', 'formal', owner)

    def test_marker_installation_requires_exact_kind_schema_and_no_overwrite(self):
        payload = {
            'kind': 'formal_phase_success', 'role': 'launcher',
            'spec_key': '', 'exit_code': 0,
        }
        with self.assertRaisesRegex(ValueError, 'not drained'):
            driver.install_marker(self.root, 'FORMAL_PHASE_SUCCESS', payload)
        self.assertFalse((self.root / 'FORMAL_PHASE_SUCCESS').exists())
        self.assertFalse((self.root / 'audit_queue').exists())
        legacy = self.base / 'legacy-marker-only'
        legacy.mkdir(mode=0o700)
        self.assertEqual([], list(legacy.iterdir()))
        self.assertEqual(
            payload,
            driver.install_marker(legacy, 'FORMAL_PHASE_SUCCESS', payload))
        with self.assertRaises(FileExistsError):
            driver.install_marker(
                legacy, 'FORMAL_PHASE_SUCCESS', payload)
        failure = {
            'kind': 'failed_setup', 'role': 'formal-worker-0',
            'spec_key': driver.spec_key(formal_specs()[0]), 'exit_code': 96,
        }
        self.assertEqual(
            failure, driver.install_marker(legacy, 'FAILED_JOB', failure))
        for name, malformed in (
                ('EXPLANATION_PHASE_SUCCESS', {**payload, 'extra': True}),
                ('FORMAL_EXECUTION_SUCCESS', {
                    **payload, 'kind': 'wrong'}),
                ('FAILED_JOB', {
                    **payload, 'kind': 'failed_job', 'exit_code': 0}),
                ('FORMAL_STOPPED', {
                    **payload, 'kind': 'formal_stopped', 'role': ''})):
            with self.subTest(name=name), self.assertRaises(ValueError):
                driver.install_marker(legacy, name, malformed)
        self.assertEqual(
            {'FORMAL_PHASE_SUCCESS', 'FAILED_JOB'},
            {path.name for path in legacy.iterdir()})

    def test_failed_retention_is_an_exact_failed_job_marker_only(self):
        key = driver.spec_key(formal_specs()[0])
        payload = {
            'kind': 'failed_retention', 'role': 'formal-worker-2',
            'spec_key': key, 'exit_code': 41,
        }
        roots = [self.base / f'retention-marker-{index}' for index in range(5)]
        for root in roots:
            root.mkdir(mode=0o700)
        self.assertEqual(
            payload, driver.install_marker(roots[0], 'FAILED_JOB', payload))
        self.assertEqual(
            canonical(payload) + '\n', (roots[0] / 'FAILED_JOB').read_text())
        for index, (name, malformed) in enumerate((
                ('FORMAL_STOPPED', payload),
                ('FAILED_JOB', {**payload, 'kind': 'failed_retention_typo'}),
                ('FAILED_JOB', {**payload, 'extra': True}),
                ('FAILED_JOB', {**payload, 'spec_key': ''})), start=1):
            with self.subTest(name=name, payload=malformed), \
                    self.assertRaises(ValueError):
                driver.install_marker(roots[index], name, malformed)
            self.assertEqual([], list(roots[index].iterdir()))


@unittest.skipUnless(os.name == 'posix', 'launcher is a POSIX Bash program')
class FakeLauncherTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='formal_launcher_fake_')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.tmpdir = self.base / 'tmp'
        self.tmpdir.mkdir()
        self.initial_child_gates = set(
            self.tmpdir.glob('formal-child-gates.*'))
        self.worktree = self.base / 'worktree'
        self.worktree.mkdir()
        self.launcher = self.worktree / SCRIPT.name
        shutil.copy2(SCRIPT, self.launcher)
        self.launcher.chmod(0o755)
        self.root = self.base / 'formal'
        self.root.mkdir()
        for name in ('FORMAL_REGISTRY.json', 'COMPATIBILITY_CENSUS.json',
                     'FORMAL_PLAN.json', 'MISSING_JOBS.json'):
            (self.root / name).write_text('{}\n')
        self.fakebin = self.base / 'fakebin'
        self.fakebin.mkdir()
        self.driver_log = self.base / 'driver.log'
        self.worker_log = self.base / 'worker.log'
        self._write_executable(self.fakebin / 'git', r'''#!/bin/bash
set -eu
case " $* " in
  *" branch --show-current "*) printf '%s\n' "${FAKE_GIT_BRANCH}" ;;
  *" rev-parse --git-common-dir "*) printf '%s\n' "${FAKE_GIT_COMMON}" ;;
  *" rev-parse HEAD "*) printf '%s\n' "${FAKE_GIT_HEAD}" ;;
  *" status --porcelain=v1 "*) test -z "${FAKE_GIT_DIRTY:-}" || printf ' M dirty\n' ;;
  *" ls-files --error-unmatch -- prune_completed_runs.py "*) printf '%s\n' prune_completed_runs.py ;;
  *) exit 91 ;;
esac
''')
        self._write_executable(self.fakebin / 'df', r'''#!/bin/bash
available=${FAKE_DISK_KB:-999999999}
[[ -z "${FAKE_DISK_FILE:-}" ]] || read -r available < "$FAKE_DISK_FILE"
[[ -z "${FAKE_DISK_LOG:-}" ]] || printf '%s\n' "$available" >> "$FAKE_DISK_LOG"
printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n'
printf 'fake 999999999 1 %s 1%% /\n' "$available"
''')
        self._write_executable(self.fakebin / 'awk', r'''#!/home/c3080/YangXiaoXiang/envs/vfcl/bin/python
import json
import os
from pathlib import Path
import sys
import time

if sys.argv[-1] != '/proc/meminfo':
    os.execv('/usr/bin/awk', ['awk', *sys.argv[1:]])
pid = os.getppid()
worker = 'launcher'
while pid > 1:
    parts = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if b'--internal-worker' in parts:
        worker = parts[parts.index(b'--internal-worker') + 3].decode()
        break
    pid = int(Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[1])
available = os.environ.get('FAKE_MEMAVAILABLE_KB', '41943040')
path = os.environ.get('FAKE_MEMAVAILABLE_FILE')
if path:
    values = json.loads(Path(path).read_text())
    available = values.get(worker, values.get('default', available))
with open(os.environ['FAKE_MEMORY_LOG'], 'a') as stream:
    stream.write(f'{pid}|{worker}|{available}\n')
release = os.environ.get('FAKE_MEMORY_READY_RELEASE')
read_number = sum(row.split('|')[1] == worker for row in
                  Path(os.environ['FAKE_MEMORY_LOG']).read_text().splitlines())
block_number = os.environ.get('FAKE_MEMORY_BLOCK_READ_NUMBER')
if available == '41943040' and release and (not block_number or str(read_number) == block_number):
    while not Path(release).exists():
        time.sleep(.02)
if available == 'unavailable':
    raise SystemExit(1)
print(available)
''')
        self._write_executable(self.fakebin / 'mktemp', r'''#!/bin/bash
printf '%s\n' "$*" >> "$FAKE_MKTEMP_LOG"
if [[ -n "${FAKE_PEAK_MKTEMP_FAIL:-}" && "$*" == *"formal-peak."* ]]; then
  exit 96
fi
exec /usr/bin/mktemp "$@"
''')
        self._write_executable(self.fakebin / 'setsid', r'''#!/bin/bash
if [[ -n "${FAKE_SETSID_DELAY:-}" && " $* " == *" --internal-worker "* ]]; then
  printf '%s\n' "$$" >> "$FAKE_SETSID_READY"
  sleep "$FAKE_SETSID_DELAY"
fi
if [[ -n "${FAKE_PARENT_START_CORRUPT:-}" && " $* " == *" --internal-worker "* ]]; then
  export FORMAL_PARENT_START=0
fi
if [[ " $* " != *" --internal-worker "* ]]; then
  printf '%s\n' "$$" >> "$FAKE_DORMANT_PID_LOG"
fi
args=("$@")
if [[ -n "${FAKE_CHILD_TOKEN_CORRUPT:-}" &&
      ${args[3]:-} == formal-child-wrapper ]]; then
  args[5]='corrupted-child-token'
fi
exec /usr/bin/setsid "${args[@]}"
''')
        self._write_executable(self.fakebin / 'ps', r'''#!/bin/bash
target=
[[ -s "${FAKE_DORMANT_PID_LOG:-}" ]] && target=$(tail -n 1 "$FAKE_DORMANT_PID_LOG")
if [[ -n "$target" && ${1:-} == -o && ${2:-} == pgid= &&
      ${3:-} == -p && ${4:-} == "$target" ]]; then
  if [[ -n "${FAKE_CHILD_PGID_DELAY:-}" &&
        ! -e "${FAKE_CHILD_PGID_DELAY_DONE}" ]]; then
    : > "$FAKE_CHILD_PGID_DELAY_DONE"
    printf '%s\n' "$target" > "$FAKE_CHILD_REGISTER_READY"
    sleep "$FAKE_CHILD_PGID_DELAY"
  fi
  [[ -z "${FAKE_CHILD_PGID_FAIL:-}" ]] || exit 97
  if [[ -n "${FAKE_CHILD_PGID_WRONG:-}" ]]; then
    printf '%s\n' "$((target + 1))"
    exit 0
  fi
fi
exec /usr/bin/ps "$@"
''')
        self._write_executable(self.fakebin / 'mv', r'''#!/bin/bash
if [[ -n "${FAKE_CHILD_GATE_DELAY:-}" && " $* " == *"formal-child-gates."* ]]; then
  printf 'ready\n' > "$FAKE_CHILD_GATE_READY"
  sleep "$FAKE_CHILD_GATE_DELAY"
fi
exec /usr/bin/mv "$@"
''')
        self._write_executable(self.fakebin / 'nvidia-smi', r'''#!/bin/bash
case " $* " in
  *"--query-gpu="*)
    if [[ -n "${FAKE_GPU_QUERY_RESOURCE_FILE:-}" && ! -e "$FAKE_GPU_QUERY_DROPPED" ]]; then
      printf '%s\n' "$FAKE_GPU_QUERY_RESOURCE_VALUE" > "$FAKE_GPU_QUERY_RESOURCE_FILE"
      : > "$FAKE_GPU_QUERY_DROPPED"
    fi
    [[ -z "${FAKE_GPU_QUERY_LOG:-}" ]] || printf 'gpu\n' >> "$FAKE_GPU_QUERY_LOG"
    [[ -z "${FAKE_GPU_FAIL:-}" ]] || exit 94
    call=0
    if [[ -n "${FAKE_GPU_CALLS_FILE:-}" ]]; then
      [[ ! -s "$FAKE_GPU_CALLS_FILE" ]] || read -r call < "$FAKE_GPU_CALLS_FILE"
      call=$((call + 1))
      printf '%s\n' "$call" > "$FAKE_GPU_CALLS_FILE"
    fi
    if [[ ",${FAKE_GPU_FAIL_CALLS:-}," == *",$call,"* ]]; then
      exit 94
    fi
    if [[ ",${FAKE_GPU_MALFORMED_CALLS:-}," == *",$call,"* ]]; then
      printf '0, GPU-0, invalid, 0\n1, GPU-1, 2000, 0\n'
    elif [[ ",${FAKE_GPU0_UUID_CHANGE_CALLS:-}," == *",$call,"* ]]; then
      printf '0, GPU-0-CHANGED, 8000, 0\n1, GPU-1, 2000, 0\n'
    elif [[ ",${FAKE_GPU0_LOW_MEMORY_CALLS:-}," == *",$call,"* ]]; then
      printf '0, GPU-0, 2000, 0\n1, GPU-1, 2000, 0\n'
    else
      if [[ -n "${FAKE_GPU0_LOW_MEMORY_FILE:-}" &&
            -e "$FAKE_GPU0_LOW_MEMORY_FILE" ]]; then
        printf '0, GPU-0, 2000, 0\n1, GPU-1, 8000, 0\n'
      else
        printf '%b\n' "${FAKE_GPU_ROWS:-0, GPU-0, 8000, 0\\n1, GPU-1, 8000, 0}"
      fi
    fi ;;
  *"--query-compute-apps=gpu_uuid,pid,used_gpu_memory"*)
    [[ -z "${FAKE_MONITOR_FAIL:-}" ]] || exit 93
    if [[ -s "${FAKE_JOB_PID_FILE:-}" ]]; then
      printf '%s, %s, 123\n' "${FAKE_MONITOR_UUID:-GPU-0}" "$(<"$FAKE_JOB_PID_FILE")"
    fi ;;
  *"--query-compute-apps="*)
    [[ -z "${FAKE_COMPUTE_FAIL:-}" ]] || exit 94
    if [[ -n "${FAKE_COMPUTE_FAIL_FILE:-}" &&
          -e "$FAKE_COMPUTE_FAIL_FILE" ]]; then
      exit 94
    fi
    [[ -z "${FAKE_COMPUTE_SLEEP:-}" ]] || sleep "$FAKE_COMPUTE_SLEEP"
    call=0
    if [[ -n "${FAKE_COMPUTE_CALLS_FILE:-}" ]]; then
      [[ ! -s "$FAKE_COMPUTE_CALLS_FILE" ]] ||
        read -r call < "$FAKE_COMPUTE_CALLS_FILE"
      call=$((call + 1))
      printf '%s\n' "$call" > "$FAKE_COMPUTE_CALLS_FILE"
    fi
    if [[ ",${FAKE_GPU0_BUSY_CALLS:-}," == *",$call,"* ]]; then
      printf 'GPU-0, 900001, python\n'
    elif [[ -n "${FAKE_COMPUTE_BUSY_FILE:-}" &&
          -e "$FAKE_COMPUTE_BUSY_FILE" ]]; then
      printf 'GPU-0, 900001, python\nGPU-1, 900002, python\n'
    elif [[ -n "${FAKE_GPU0_BUSY_FILE:-}" &&
            -e "$FAKE_GPU0_BUSY_FILE" ]]; then
      printf 'GPU-0, 900001, python\n'
    elif [[ -n "${FAKE_COMPUTE_ROWS:-}" ]]; then
      printf '%b\n' "$FAKE_COMPUTE_ROWS"
    fi
    [[ -z "${FAKE_GPU_QUERY_LOG:-}" ]] || printf 'compute\n' >> "$FAKE_GPU_QUERY_LOG" ;;
  *) exit 92 ;;
esac
''')
        self.worker = self.base / 'fake-worker.sh'
        self._write_executable(self.worker, r'''#!/bin/bash
set -u
spec=$1
run_dir=$2
shift 2
printf '%s|%s\n' "$$" "$spec" >> "$FAKE_CHILD_PID_LOG"
printf '%s|%s|%s|%s|%s|%s\n' "$spec" "${CUDA_VISIBLE_DEVICES:-}" "$*" \
  "${CUBLAS_WORKSPACE_CONFIG:-}" "${OMP_NUM_THREADS:-}" "${PYTHONHASHSEED:-}" >> "$FAKE_WORKER_LOG"
printf '{"kind":"training-start","key":"%s"}\n' "$spec" >> "$FAKE_PIPELINE_EVENTS"
if [[ -n "${FAKE_REQUIRE_HASH_SEED_MATCH:-}" ]]; then
  expected_hash_seed=${spec##*:}
  if [[ ! $expected_hash_seed =~ ^(42|43|44)$ ||
        ${PYTHONHASHSEED:-} != "$expected_hash_seed" ]]; then
    exit 87
  fi
fi
if [[ "$spec" == "${FAKE_BUSY_AFTER_JOB:-never}" ]]; then
  rm -f -- "$FAKE_GPU0_BUSY_FILE"
  : > "$FAKE_COMPUTE_BUSY_FILE"
fi
if [[ "$spec" == "${FAKE_MONITOR_JOB:-never}" ]]; then
  printf '%s\n' "$$" > "$FAKE_JOB_PID_FILE"
  sleep 0.3
fi
if [[ "$spec" == "${FAKE_IGNORE_TERM_JOB:-never}" ]]; then
  printf '%s\n' "$spec" >> "$FAKE_READY_LOG"
  trap '' TERM
  while :; do sleep 0.1; done
fi
if [[ "$spec" == "${FAKE_TRAIN_GRANDCHILD:-never}" ]]; then
  "$VFCL_PYTHON" -c 'import time; time.sleep(120)' &
  printf '{"kind":"training-grandchild","pid":%s}\n' "$!" >> "$FAKE_PIPELINE_EVENTS"
fi
if [[ "$spec" == "${FAKE_SLOW_JOB:-never}" ]]; then
  printf '%s\n' "$spec" >> "$FAKE_READY_LOG"
  trap 'printf "%s\n" "$spec" >> "$FAKE_KILLED_LOG"; exit 143' TERM INT HUP
  while :; do sleep 0.1; done
fi
gpu_lock=
if [[ -n "${FAKE_GPU_LOCK_ROOT:-}" && -d "$FAKE_GPU_LOCK_ROOT" ]]; then
  gpu_lock="$FAKE_GPU_LOCK_ROOT/gpu-${CUDA_VISIBLE_DEVICES:-missing}"
  if ! mkdir "$gpu_lock" 2>/dev/null; then
    printf '%s|%s\n' "$spec" "${CUDA_VISIBLE_DEVICES:-}" >> "$FAKE_GPU_OVERLAP_LOG"
    exit 98
  fi
  trap 'rmdir -- "$gpu_lock" 2>/dev/null || true' EXIT
  sleep "${FAKE_WORKER_DELAY:-0.2}"
fi
if [[ "$spec" == "${FAKE_FAIL_JOB:-never}" ]]; then
  exit "${FAKE_FAIL_CODE:-7}"
fi
if [[ -n "${FAKE_WAIT_FIRST_AUDIT:-}" && "$spec" != "$FAKE_AUDIT_BLOCK" ]]; then
  while [[ ! -e "$FAKE_AUDIT_READY" ]]; do sleep 0.02; done
fi
printf 'worker complete\n'
printf '%s\n' "$spec" >> "$FAKE_FINISHED_LOG"
if [[ "$spec" == "${FAKE_SWITCH_GPU_AFTER_JOB:-never}" ]]; then
  : > "$FAKE_GPU0_LOW_MEMORY_FILE"
fi
''')
        (self.worktree / 'three_dataset_formal_driver.py').write_text(
            self._fake_driver_source())
        (self.worktree / 'prune_completed_runs.py').write_text(
            self._fake_retention_source())
        self.env = os.environ.copy()
        self.env.pop('VFCL_EXPERIMENT_PROFILE', None)
        common = Path(subprocess.check_output(
            ['git', '-C', str(SCRIPT.parent), 'rev-parse', '--git-common-dir'],
            text=True,
        ).strip())
        if not common.is_absolute():
            common = (SCRIPT.parent / common).resolve()
        self.env.update({
            'PATH': f'{self.fakebin}:{self.env["PATH"]}',
            'TMPDIR': str(self.tmpdir),
            'VFCL_PYTHON': PYTHON,
            'FAKE_GIT_BRANCH': BRANCH,
            'FAKE_GIT_HEAD': HEAD,
            'FAKE_GIT_COMMON': str(common),
            'FAKE_DRIVER_LOG': str(self.driver_log),
            'FAKE_RETENTION_LOG': str(self.base / 'retention.log'),
            'FAKE_RETENTION_READY': str(self.base / 'retention-ready'),
            'FAKE_RETENTION_RELEASE': str(self.base / 'retention-release'),
            'FAKE_MEMORY_LOG': str(self.base / 'memory.log'),
            'FAKE_WORKER': str(self.worker),
            'FAKE_WORKER_LOG': str(self.worker_log),
            'FAKE_FINISHED_LOG': str(self.base / 'finished.log'),
            'FAKE_PIPELINE_EVENTS': str(self.base / 'pipeline-events.jsonl'),
            'FAKE_AUDIT_READY': str(self.base / 'audit-ready'),
            'FAKE_AUDIT_RELEASE': str(self.base / 'audit-release'),
            'FAKE_READY_LOG': str(self.base / 'ready.log'),
            'FAKE_KILLED_LOG': str(self.base / 'killed.log'),
            'FAKE_JOB_PID_FILE': str(self.base / 'job.pid'),
            'FAKE_CHILD_PID_LOG': str(self.base / 'child-pids.log'),
            'FAKE_DORMANT_PID_LOG': str(self.base / 'dormant-pids.log'),
            'FAKE_CHILD_REGISTER_READY': str(self.base / 'child-register-ready'),
            'FAKE_CHILD_PGID_DELAY_DONE': str(self.base / 'child-pgid-delay-done'),
            'FAKE_CHILD_GATE_READY': str(self.base / 'child-gate-ready'),
            'FAKE_GPU_RELEASE_READY': str(self.base / 'gpu-release-ready'),
            'FAKE_SETSID_READY': str(self.base / 'setsid-ready.log'),
            'FAKE_OWNER_READY': str(self.base / 'owner-ready.log'),
            'FAKE_PLAN_BAD_FILE': str(self.base / 'plan-bad'),
            'FAKE_MKTEMP_LOG': str(self.base / 'mktemp.log'),
            'FAKE_SMOKE_JOBS_READY': str(self.base / 'smoke-jobs-ready'),
            'FAKE_SMOKE_JOBS_PID_FILE': str(self.base / 'smoke-jobs.pid'),
            'FAKE_COMPUTE_BUSY_FILE': str(self.base / 'compute-busy'),
            'FAKE_COMPUTE_CALLS_FILE': str(self.base / 'compute-calls'),
            'FAKE_COMPUTE_FAIL_FILE': str(self.base / 'compute-fail'),
            'FAKE_GPU_CALLS_FILE': str(self.base / 'gpu-calls'),
            'FAKE_GPU0_BUSY_FILE': str(self.base / 'gpu0-busy'),
            'FAKE_GPU0_LOW_MEMORY_FILE': str(self.base / 'gpu0-low-memory'),
            'FAKE_GPU_QUERY_LOG': str(self.base / 'gpu-query.log'),
            'FAKE_GPU_LOCK_ROOT': str(self.base / 'gpu-locks'),
            'FAKE_GPU_OVERLAP_LOG': str(self.base / 'gpu-overlap.log'),
            'FAKE_POSTPROCESS_ENV_LOG': str(self.base / 'postprocess-env.log'),
            'FAKE_FORMAL_TOTAL': '0',
            'FAKE_EXPLANATION_TOTAL': '0',
        })
        self.addCleanup(self._remove_new_child_gates)
        self.addCleanup(self._stop_logged_fake_children)
        self.addCleanup(self._stop_logged_dormant_children)

    @staticmethod
    def _write_executable(path, content):
        path.write_text(textwrap.dedent(content).lstrip())
        path.chmod(0o755)

    @staticmethod
    def _stop_owned_process_group(process):
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=5)

    @staticmethod
    def _stop_process(process):
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)

    def _live_logged_fake_children(self):
        path = self.base / 'child-pids.log'
        live = []
        if not path.exists():
            return live
        for line in path.read_text().splitlines():
            raw_pid, _spec = line.split('|', 1)
            pid = int(raw_pid)
            try:
                command = Path(f'/proc/{pid}/cmdline').read_bytes()
            except OSError:
                continue
            if str(self.worker).encode() in command:
                live.append(pid)
        return live

    def _stop_logged_fake_children(self):
        for pid in self._live_logged_fake_children():
            try:
                pgid = os.getpgid(pid)
                if pgid != os.getpgrp():
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.time() + 3
        while time.time() < deadline and self._live_logged_fake_children():
            time.sleep(.02)

    def _live_logged_dormant_children(self):
        path = self.base / 'dormant-pids.log'
        live = []
        if not path.exists():
            return live
        for raw_pid in path.read_text().splitlines():
            pid = int(raw_pid)
            try:
                command = Path(f'/proc/{pid}/cmdline').read_bytes()
            except OSError:
                continue
            if (b'formal-child-wrapper' in command
                    or b'formal-child-gates.' in command
                    or str(self.worktree / 'three_dataset_formal_driver.py').encode() in command
                    or str(self.worker).encode() in command):
                live.append(pid)
        return live

    def _stop_logged_dormant_children(self):
        for pid in self._live_logged_dormant_children():
            try:
                pgid = os.getpgid(pid)
                if pgid != os.getpgrp():
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.time() + 3
        while time.time() < deadline and self._live_logged_dormant_children():
            time.sleep(.02)

    def _new_child_gates(self):
        return set(self.tmpdir.glob('formal-child-gates.*')).difference(
            self.initial_child_gates)

    def _remove_new_child_gates(self):
        for path in self._new_child_gates():
            if (path.parent == self.tmpdir
                    and path.name.startswith('formal-child-gates.')
                    and path.is_dir() and not path.is_symlink()):
                for name in ('gate', '.gate'):
                    (path / name).unlink(missing_ok=True)
                try:
                    path.rmdir()
                except OSError:
                    pass

    def _fake_driver_source(self):
        return textwrap.dedent(r'''
            import argparse
            import fcntl
            import hashlib
            import json
            import os
            from pathlib import Path
            import signal
            import sys
            import time
            from urllib.parse import quote

            def log():
                with open(os.environ['FAKE_DRIVER_LOG'], 'a') as stream:
                    stream.write(' '.join(sys.argv[1:]) + '\n')

            def canonical(value):
                return json.dumps(value, sort_keys=True, separators=(',', ':'))

            def value(name):
                index = sys.argv.index(name)
                return sys.argv[index + 1]

            def safe(key):
                return quote(key, safe='')

            def install(path, payload):
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('x') as stream:
                    stream.write(canonical(payload) + '\n')

            def event(kind, **fields):
                with open(os.environ['FAKE_PIPELINE_EVENTS'], 'a') as stream:
                    stream.write(canonical({'kind': kind, **fields}) + '\n')

            def phase_ready(phase):
                records = list((root / 'records').glob('*.json'))
                completed = sum(json.loads(p.read_text())['phase'] == phase
                                for p in records)
                pending = [p for p in (root / 'audit_queue').glob('*.json')
                           if spec_phase(json.loads(p.read_text())['spec_key']) == phase]
                return completed == int(os.environ['FAKE_' + phase.upper() + '_TOTAL']) and not pending

            def spec_phase(key):
                return json.loads((root / 'claims' / safe(key) / 'owner.json').read_text())['phase']

            def disk_safe():
                path = os.environ.get('FAKE_DISK_SLOTS_FILE')
                slots = int(Path(path).read_text()) if path else 3
                outstanding = sum(not (root / 'records' / (p.name + '.json')).exists()
                                  or (root / 'audit_queue' / (p.name + '.json')).exists()
                                  for p in (root / 'claims').glob('*') if p.is_dir())
                if os.environ.get('FAKE_DISK_REQUIRE_LIVE_OWNER'):
                    for claim in (root / 'claims').glob('*'):
                        if (root / 'records' / (claim.name + '.json')).exists() and not (
                                root / 'audit_queue' / (claim.name + '.json')).exists():
                            continue
                        owner = json.loads((claim / 'owner.json').read_text())
                        state = Path(f"/proc/{owner['pid']}/stat")
                        if not state.exists() or state.read_text().rsplit(') ', 1)[1].split()[0] == 'Z':
                            raise SystemExit('unsettled reservation owner exited')
                return outstanding < slots

            def validate_postprocess_environment(action, key, root):
                if not os.environ.get('FAKE_REQUIRE_POSTPROCESS_GPU_MATCH'):
                    return
                owners = []
                if action == 'audit-run':
                    active = json.loads((root / 'audit_queue' / 'active.json').read_text())
                    if active['spec_key'] != key:
                        raise SystemExit(18)
                    owners.append(active)
                gpu_claims = root / 'gpu_claims'
                if action == 'resource-record' and gpu_claims.is_dir():
                    for path in gpu_claims.glob('gpu-*/owner.json'):
                        owner = json.loads(path.read_text())
                        if (owner.get('kind') == 'formal_gpu_claim'
                                and owner.get('job') == key):
                            owners.append(owner)
                visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                hash_seed = os.environ.get('PYTHONHASHSEED', '')
                with Path(os.environ['FAKE_POSTPROCESS_ENV_LOG']).open('a') as stream:
                    stream.write(f'{action}|{key}|{visible}|{hash_seed}\n')
                if (len(owners) != 1 or visible != str(owners[0]['physical_gpu'])
                        or hash_seed != key.rsplit(':', 1)[-1]):
                    raise SystemExit(18)

            log()
            action = sys.argv[1]
            root = Path(value('--root')) if '--root' in sys.argv else None
            if action in ('claim', 'release-claim', 'queue-audit', 'next-audit', 'complete-audit',
                          'cancel-audit', 'audit-phase-ready', 'mark'):
                pipeline_lock = (root / '.pipeline-lock').open('a')
                fcntl.flock(pipeline_lock, fcntl.LOCK_EX)
            if action in ('claim', 'queue-audit', 'next-audit', 'audit-phase-ready'):
                if any((root / name).exists() for name in ('FAILED_JOB', 'FORMAL_STOPPED')):
                    raise SystemExit(19)
            smoke_jobs = [
                'image-replay-free', 'vector-raw-replay',
                'vector-fixed-endpoint', 'vector-adaptive',
            ]
            if os.environ.get('VFCL_EXPERIMENT_PROFILE') == 'full-public-matrix':
                smoke_jobs = [f'{dataset}-{method}'.replace('_', '-')
                              for dataset in ('cifar100', 'isolet', 'upmc_food101')
                              for method in ('finetune', 'lwf', 'ewc', 'er', 'der_pp',
                                  'er_ace', 'gpm', 'fedprotip_vfl', 'target', 'afc',
                                  'lwf_wa', 'adagauss', 'proto_fedspace', 'adaptive')]
            if action == 'smoke-plan':
                if any(root.iterdir()):
                    raise SystemExit('smoke root is not empty')
                for name in ('fixtures', 'runs', 'logs', 'records', 'control'):
                    (root / name).mkdir()
                install(root / 'SMOKE_PLAN.json', {
                    'kind': 'three_dataset_generated_smoke_plan',
                    'generated_only': True, 'scientific_gate': False,
                    'jobs': [
                        {'job_id': job, 'generated_only': True,
                         'preferred_gpu': index % 2}
                        for index, job in enumerate(smoke_jobs)
                    ],
                })
                print((root / 'SMOKE_PLAN.json').read_text(), end='')
            elif action == 'smoke-check':
                if (os.environ.get('FAKE_PLAN_BAD_FILE')
                        and Path(os.environ['FAKE_PLAN_BAD_FILE']).exists()):
                    raise SystemExit('smoke plan rejected')
                plan = json.loads((root / 'SMOKE_PLAN.json').read_text())
                if ([job['job_id'] for job in plan['jobs']] != smoke_jobs
                        or [job['preferred_gpu'] for job in plan['jobs']]
                        != [index % 2 for index in range(len(smoke_jobs))]):
                    raise SystemExit('smoke plan differs')
                print('{}')
            elif action == 'smoke-jobs':
                if os.environ.get('FAKE_SMOKE_JOBS_PID_FILE'):
                    Path(os.environ['FAKE_SMOKE_JOBS_PID_FILE']).write_text(
                        str(os.getpid()) + '\n')
                if os.environ.get('FAKE_SMOKE_JOBS_READY'):
                    Path(os.environ['FAKE_SMOKE_JOBS_READY']).write_text('ready\n')
                if os.environ.get('FAKE_SMOKE_JOBS_DELAY'):
                    time.sleep(float(os.environ['FAKE_SMOKE_JOBS_DELAY']))
                if os.environ.get('FAKE_SMOKE_JOBS_FAIL'):
                    raise SystemExit(int(os.environ['FAKE_SMOKE_JOBS_FAIL']))
                rows = (os.environ.get('FAKE_SMOKE_JOBS_ROWS', '').split(',')
                        if 'FAKE_SMOKE_JOBS_ROWS' in os.environ else smoke_jobs)
                for job in rows:
                    print(job)
            elif action == 'smoke-command':
                job = value('--job'); run = value('--run-dir')
                if job not in smoke_jobs:
                    raise SystemExit('unknown smoke job')
                tokens = [os.environ['FAKE_WORKER'], job, run,
                          '--device', 'cuda:0']
                sys.stdout.buffer.write(
                    b'\0'.join(token.encode() for token in tokens) + b'\0')
            elif action == 'smoke-begin':
                job = value('--job')
                install(root / 'control' / f'{job}.json', {
                    'kind': 'generated_smoke_launch', 'job_id': job,
                    'preferred_gpu': smoke_jobs.index(job) % 2,
                    'physical_gpu': int(value('--physical-gpu')),
                })
                print('{}')
            elif action == 'smoke-record':
                job = value('--job'); run = Path(value('--run-dir'))
                run.mkdir(parents=True, exist_ok=True)
                (run / 'config.json').write_text('{}\n')
                (run / 'results.json').write_text('{}\n')
                install(root / 'records' / f'{job}.json', {
                    'kind': 'generated_smoke_completed', 'job_id': job,
                    'generated_only': True, 'resume_probe': True,
                })
                print('{}')
            elif action == 'audit-smoke':
                records = sorted(path.stem for path in (root / 'records').glob('*.json'))
                if records != sorted(smoke_jobs):
                    raise SystemExit('incomplete smoke records')
                install(root / 'SMOKE_AUDIT.json', {
                    'status': 'SMOKE_EXECUTION_SUCCESS',
                    'generated_only': True, 'scientific_gate': False,
                })
                install(root / 'SMOKE_EXECUTION_SUCCESS', {
                    'status': 'SMOKE_EXECUTION_SUCCESS',
                    'scientific_gate': False,
                })
                print('{}')
            elif action == 'smoke-mark':
                install(root / value('--name'), {
                    'kind': value('--kind'),
                    'exit_code': int(value('--exit-code')),
                    'scientific_gate': False,
                })
                print('{}')
            elif action == 'check':
                if os.environ.get('VFCL_EXPERIMENT_PROFILE') == 'full-public-matrix':
                    plan = json.loads((root / 'FORMAL_PLAN.json').read_text())
                    if len(plan.get('formal_jobs', [])) != 126 or plan.get('explanation_jobs') != []:
                        raise SystemExit('full matrix plan rejected')
                if os.environ.get('VFCL_EXPERIMENT_PROFILE') == 'single-dataset-full-matrix':
                    plan = json.loads((root / 'FORMAL_PLAN.json').read_text())
                    dataset = os.environ.get('VFCL_FORMAL_DATASET')
                    if (dataset not in ('cifar100', 'isolet', 'upmc_food101')
                            or len(plan.get('formal_jobs', [])) != 42
                            or plan.get('explanation_jobs') != []
                            or any(not key.startswith(dataset + ':')
                                   for key in plan['formal_jobs'])):
                        raise SystemExit('single dataset plan rejected')
                if (os.environ.get('FAKE_PLAN_BAD')
                        or (os.environ.get('FAKE_PLAN_BAD_FILE')
                            and Path(os.environ['FAKE_PLAN_BAD_FILE']).exists())):
                    raise SystemExit('installed plan rejected')
                for name in ('FORMAL_REGISTRY.json', 'COMPATIBILITY_CENSUS.json',
                             'FORMAL_PLAN.json', 'MISSING_JOBS.json'):
                    if not (root / name).is_file():
                        raise SystemExit('installed bundle missing')
                print('{}')
            elif action == 'disk-status':
                requested_slots = int(value('--requested-slots'))
                assert requested_slots in (1, 2)
                with (root / 'FAKE_FORMAL_QUEUE').open('r+') as stream:
                    fcntl.flock(stream, fcntl.LOCK_EX)
                    safe_to_claim = disk_safe()
                event('disk-status', safe=safe_to_claim, requested_slots=requested_slots)
                if os.environ.get('FAKE_DISK_STATUS_EXIT'):
                    raise SystemExit(23)
                if 'FAKE_DISK_STATUS_OUTPUT' in os.environ:
                    print(os.environ['FAKE_DISK_STATUS_OUTPUT'])
                else:
                    print(canonical({'kind': 'full_matrix_disk_status_v1',
                        'available_bytes': 100 * 1024 ** 3, 'safety_bytes': 30 * 1024 ** 3,
                        'active_reservation_bytes': 0, 'requested_reservation_bytes': 0,
                        'predicted_retained_remainder_bytes': 0,
                        'requested_slots': requested_slots,
                        'safe': safe_to_claim, 'plan_sha256': 'a' * 64}))
            elif action == 'owner':
                delay = os.environ.get('FAKE_OWNER_DELAY')
                delay_phase = os.environ.get('FAKE_OWNER_DELAY_PHASE')
                if delay and (not delay_phase or delay_phase == value('--phase')):
                    Path(os.environ['FAKE_OWNER_READY']).write_text('ready\n')
                    time.sleep(float(delay))
                pid = int(value('--pid'))
                owner = {
                    'kind': 'formal_job_claim', 'job': '',
                    'launcher_token': value('--launcher-token'),
                    'worker_role': value('--worker-role'),
                    'phase': value('--phase'), 'pid': pid,
                    'pgid': int(value('--pgid')),
                    'process_start_time': Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19],
                    'source_commit': os.environ['FAKE_GIT_HEAD'],
                    'root_identity': {'dev': 1, 'inode': 2, 'ctime_ns': 3,
                                      'size': 4, 'hash': 'a' * 64},
                }
                if os.environ.get('FAKE_MUTATE_PLAN_ON_OWNER'):
                    Path(os.environ['FAKE_PLAN_BAD_FILE']).write_text('bad\n')
                print(canonical(owner))
            elif action == 'claim':
                if os.environ.get('FAKE_CLAIM_ERROR'):
                    raise SystemExit(int(os.environ['FAKE_CLAIM_ERROR']))
                requested_owner = json.loads(value('--owner-json'))
                fail_file = os.environ.get('FAKE_CLAIM_FAIL_FILE')
                fail_worker = os.environ.get('FAKE_CLAIM_FAIL_WORKER')
                if (fail_file and Path(fail_file).exists()
                        and requested_owner['worker_role'] == fail_worker):
                    raise SystemExit(31)
                required_worker = os.environ.get('FAKE_REQUIRED_CLAIM_WORKER')
                if required_worker and requested_owner['worker_role'] != required_worker:
                    print('')
                    raise SystemExit(0)
                phase = value('--phase')
                queue = root / f'FAKE_{phase.upper()}_QUEUE'
                queue.touch(exist_ok=True)
                with queue.open('r+') as stream:
                    fcntl.flock(stream, fcntl.LOCK_EX)
                    if os.environ.get('VFCL_EXPERIMENT_PROFILE') == 'full-public-matrix' and not disk_safe():
                        raise SystemExit(75)
                    rows = [row for row in stream.read().splitlines() if row]
                    if not rows:
                        event('claim-empty', worker=requested_owner['worker_role'])
                        if os.environ.get('FAKE_MUTATE_PLAN_ON_EMPTY_CLAIM'):
                            Path(os.environ['FAKE_PLAN_BAD_FILE']).write_text('bad\n')
                        print('')
                    else:
                        recovery = os.environ.get('VFCL_EXPERIMENT_PROFILE') == 'seed42-adaptive-recovery'
                        scoped = os.environ.get('VFCL_EXPERIMENT_PROFILE') in (
                            'single-dataset-full-matrix',
                            'single-dataset-verified-continuation-v1')
                        outstanding = sum(not (root / 'records' / (p.name + '.json')).exists()
                                          or ((recovery or scoped) and
                                              (root / 'audit_queue' / (p.name + '.json')).exists())
                                          for p in (root / 'claims').glob('*') if p.is_dir())
                        limit = (1 if recovery else int(os.environ.get('VFCL_GPU_COUNT', '1'))
                                 if scoped else 3)
                        if '--pipeline' in sys.argv and outstanding >= limit:
                            event('backpressure', outstanding=outstanding)
                            raise SystemExit(75)
                        key = rows.pop(0)
                        stream.seek(0); stream.truncate()
                        stream.write(''.join(row + '\n' for row in rows)); stream.flush()
                        claim = root / 'claims' / safe(key)
                        claim.mkdir(parents=True, exist_ok=False)
                        owner = json.loads(value('--owner-json')); owner['job'] = key
                        install(claim / 'owner.json', owner)
                        event('claim', key=key, outstanding=outstanding + 1)
                        print(key)
            elif action == 'claim-owner':
                print((root / 'claims' / safe(value('--spec')) / 'owner.json').read_text(), end='')
            elif action == 'prepare-run':
                key = value('--spec')
                if os.environ.get('FAKE_PREPARE_FAIL') == key:
                    raise SystemExit(14)
                run = root / 'runs' / safe(key)
                run.mkdir(parents=True, exist_ok=False)
                owner = json.loads((root / 'claims' / safe(key) / 'owner.json').read_text())
                for name, payload in (
                        ('FORMAL_JOB_SPEC.json', {'kind': 'formal_job_spec', 'spec_key': key}),
                        ('CLAIM_OWNER.json', owner),
                        ('LAUNCH_STARTED.json', {'kind': 'formal_launch_started',
                                                 'spec_key': key, 'phase': owner['phase'],
                                                 'pgid': owner['pgid']})):
                    install(run / name, payload)
                install(root / 'claims' / safe(key) / 'started.json',
                        {'kind': 'formal_job_started', 'phase': owner['phase'],
                         'pgid': owner['pgid']})
                print(run)
            elif action == 'release-claim':
                key = value('--spec')
                claim = root / 'claims' / safe(key)
                if (claim / 'started.json').exists():
                    raise SystemExit(12)
                owner = json.loads((claim / 'owner.json').read_text())
                assert owner == json.loads(value('--owner-json'))
                queue = root / f"FAKE_{owner['phase'].upper()}_QUEUE"
                queue.write_text(key + '\n' + queue.read_text())
                shutil = __import__('shutil'); shutil.rmtree(claim)
                event('release-claim', key=key)
                print('{}')
            elif action == 'command':
                key = value('--spec'); run = value('--run-dir')
                if Path(run).name != safe(key):
                    raise SystemExit(20)
                seed = key.rsplit(':', 1)[-1]
                seed_mode = os.environ.get('FAKE_COMMAND_SEED_MODE', 'valid')
                seed_tokens = ['--seed', seed]
                if seed_mode == 'missing':
                    seed_tokens = []
                elif seed_mode == 'duplicate':
                    seed_tokens = ['--seed', seed, '--seed', seed]
                elif seed_mode == 'valueless':
                    seed_tokens = ['--seed']
                elif seed_mode == 'noninteger':
                    seed_tokens = ['--seed', 'not-an-integer']
                elif seed_mode == 'noncanonical':
                    seed_tokens = ['--seed', f'{int(seed):03d}']
                elif seed_mode == 'mismatch':
                    seed_tokens = ['--seed', '44' if seed != '44' else '43']
                elif seed_mode != 'valid':
                    raise SystemExit('unknown fake command seed mode')
                tokens = [
                    os.environ['FAKE_WORKER'], key, run, '--device', 'cuda:0',
                    *seed_tokens,
                ]
                sys.stdout.buffer.write(b'\0'.join(token.encode() for token in tokens) + b'\0')
            elif action == 'resource-record':
                key = value('--spec'); run = Path(value('--run-dir'))
                validate_postprocess_environment('resource-record', key, root)
                if os.environ.get('FAKE_RESOURCE_FAIL') == key:
                    raise SystemExit(8)
                install(run / 'RESOURCE_EVIDENCE.json',
                        {'kind': 'formal_resource_evidence', 'spec_key': key})
                print('{}')
            elif action == 'audit-run':
                key = value('--spec-key')
                validate_postprocess_environment('audit-run', key, root)
                if not (Path(value('--run-dir')) / 'RESOURCE_EVIDENCE.json').is_file():
                    raise SystemExit(22)
                event('audit-start', key=key, pid=os.getpid(), parent=os.getppid())
                if os.environ.get('FAKE_AUDIT_BLOCK') == key:
                    if os.environ.get('FAKE_AUDIT_GRANDCHILD'):
                        import subprocess
                        command = 'import time; time.sleep(120)'
                        arguments = []
                        if os.environ.get('FAKE_AUDIT_GRANDCHILD_TERM_MARK'):
                            command = (
                                'import pathlib, signal, sys, time\n'
                                'ready, term = map(pathlib.Path, sys.argv[1:])\n'
                                'signal.signal(signal.SIGTERM, lambda *_: term.touch())\n'
                                'ready.touch()\n'
                                'while True: time.sleep(.05)\n')
                            arguments = [os.environ['FAKE_AUDIT_GRANDCHILD_READY'],
                                         os.environ['FAKE_AUDIT_GRANDCHILD_TERM_MARK']]
                        child = subprocess.Popen(
                            [sys.executable, '-c', command, *arguments])
                        event('audit-grandchild', pid=child.pid)
                    if os.environ.get('FAKE_AUDIT_FOREIGN_CHILD'):
                        import subprocess
                        child = subprocess.Popen([sys.executable, '-c',
                                                  'import time; time.sleep(120)'],
                                                 env={**os.environ, 'FORMAL_CHILD_TOKEN': 'foreign-token'},
                                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        event('foreign-grandchild', pid=child.pid)
                    Path(os.environ['FAKE_AUDIT_READY']).write_text(str(os.getpid()))
                    while not Path(os.environ['FAKE_AUDIT_RELEASE']).exists():
                        time.sleep(.02)
                if os.environ.get('FAKE_AUDIT_FAIL') == key:
                    print('fake audit rejected', file=sys.stderr)
                    raise SystemExit(9)
                phase = spec_phase(key)
                install(root / 'records' / f'{safe(key)}.json',
                        {'kind': 'formal_completed_run', 'spec_key': key,
                         'phase': phase})
                event('record-installed', key=key)
                if os.environ.get('FAKE_POST_AUDIT_RUN_BLOCK') == key:
                    Path(os.environ['FAKE_POST_AUDIT_RUN_READY']).touch()
                    while not Path(os.environ['FAKE_POST_AUDIT_RUN_RELEASE']).exists():
                        time.sleep(.02)
                if os.environ.get('FAKE_RECORD_BLOCK') == key:
                    Path(os.environ['FAKE_RECORD_READY']).touch()
                    while not Path(os.environ['FAKE_RECORD_RELEASE']).exists():
                        time.sleep(.02)
                print('{}')
            elif action == 'queue-audit':
                key = value('--spec')
                if os.environ.get('FAKE_QUEUE_FAIL') == key:
                    raise SystemExit(21)
                run = root / 'runs' / safe(key)
                if not (run / 'RESOURCE_EVIDENCE.json').is_file():
                    raise SystemExit(22)
                payload = {'spec_key': key, 'run_dir': str(run),
                           'physical_gpu': int(value('--physical-gpu')),
                           'seed': int(key.rsplit(':', 1)[-1])}
                install(root / 'audit_queue' / (safe(key) + '.json'), payload)
                event('queue', key=key)
                print(canonical(payload))
            elif action == 'next-audit':
                owner = json.loads(value('--owner-json'))
                assert owner['job'] == '' and owner['worker_role'].endswith('-2')
                active = root / 'audit_queue' / 'active.json'
                payload = None
                if not active.exists():
                    for path in sorted((root / 'audit_queue').glob('*.json')):
                        candidate = json.loads(path.read_text())
                        if spec_phase(candidate['spec_key']) != value('--phase'):
                            continue
                        gpu_owner = root / 'gpu_claims' / f"gpu-{candidate['physical_gpu']}" / 'owner.json'
                        if gpu_owner.exists() and json.loads(gpu_owner.read_text()).get('job') == candidate['spec_key']:
                            continue
                        payload = candidate
                        install(active, payload)
                        install(root / 'audit-owner.json', owner)
                        break
                if payload and os.environ.get('FAKE_POST_NEXT_AUDIT_BLOCK'):
                    Path(os.environ['FAKE_POST_NEXT_AUDIT_READY']).touch()
                    while not Path(os.environ['FAKE_POST_NEXT_AUDIT_RELEASE']).exists():
                        time.sleep(.02)
                if payload and os.environ.get('FAKE_AUDIT_PAYLOAD'):
                    payload.update(json.loads(os.environ['FAKE_AUDIT_PAYLOAD']))
                if payload and os.environ.get('FAKE_REMOVE_RESOURCE'):
                    (Path(payload['run_dir']) / 'RESOURCE_EVIDENCE.json').unlink()
                print(canonical(payload))
            elif action in ('complete-audit', 'cancel-audit'):
                active = root / 'audit_queue' / 'active.json'
                if active.exists():
                    assert json.loads(value('--owner-json')) == json.loads((root / 'audit-owner.json').read_text())
                    payload = json.loads(active.read_text())
                    if action == 'complete-audit':
                        assert payload['spec_key'] == value('--spec')
                        assert (root / 'records' / (safe(value('--spec')) + '.json')).is_file()
                        (root / 'audit_queue' / (safe(value('--spec')) + '.json')).unlink()
                        event('complete', key=value('--spec'))
                    else:
                        assert any((root / name).exists() for name in ('FAILED_JOB', 'FORMAL_STOPPED'))
                    active.unlink()
                    (root / 'audit-owner.json').unlink()
                print('{}')
            elif action == 'audit-phase-ready':
                print(canonical(phase_ready(value('--phase'))))
            elif action == 'phase-ready':
                records = list((root / 'records').glob('*.json')) if (root / 'records').is_dir() else []
                formal = sum(json.loads(path.read_text()).get('phase') == 'formal'
                             for path in records)
                if formal != int(os.environ['FAKE_FORMAL_TOTAL']):
                    raise SystemExit(10)
                print('true')
            elif action == 'gpu-claim':
                gpu = int(value('--physical-gpu'))
                claim = root / 'gpu_claims' / f'gpu-{gpu}'
                try:
                    claim.mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    raise SystemExit(4)
                owner = json.loads(value('--owner-json'))
                owner['kind'] = 'formal_gpu_claim'
                owner['physical_gpu'] = gpu
                install(claim / 'owner.json', owner)
                drop = os.environ.get('FAKE_MEMORY_DROP_AFTER_GPU')
                if drop and not Path(drop).exists():
                    Path(drop).touch()
                    Path(os.environ['FAKE_MEMAVAILABLE_FILE']).write_text(
                        canonical({'default': '41943039'}))
                if os.environ.get('FAKE_GPU_CLAIM_INSTALL_THEN_FAIL'):
                    raise SystemExit(
                        int(os.environ['FAKE_GPU_CLAIM_INSTALL_THEN_FAIL']))
                print('true')
            elif action == 'gpu-release':
                gpu = int(value('--physical-gpu'))
                requested = json.loads(value('--owner-json'))
                owner_stat = Path(f"/proc/{requested['pid']}/stat")
                owner_alive = owner_stat.exists() and owner_stat.read_text().rsplit(')', 1)[1].split()[0] != 'Z'
                if not owner_alive:
                    assert any((root / name).exists() for name in ('FAILED_JOB', 'FORMAL_STOPPED'))
                    assert requested['source_commit'] == os.environ['FAKE_GIT_HEAD']
                    assert requested['root_identity'] == {'dev': 1, 'inode': 2, 'ctime_ns': 3,
                                                           'size': 4, 'hash': 'a' * 64}
                if os.environ.get('FAKE_GPU_RELEASE_DELAY') and owner_alive:
                    Path(os.environ['FAKE_GPU_RELEASE_READY']).write_text('ready\n')
                    time.sleep(float(os.environ['FAKE_GPU_RELEASE_DELAY']))
                fail_once = os.environ.get('FAKE_GPU_RELEASE_FAIL_ONCE_FILE')
                if fail_once and not Path(fail_once).exists():
                    Path(fail_once).write_text('failed\n')
                    raise SystemExit(14)
                claim = root / 'gpu_claims' / f'gpu-{gpu}'
                installed = json.loads((claim / 'owner.json').read_text())
                expected = json.loads(value('--owner-json'))
                expected['kind'] = 'formal_gpu_claim'
                expected['physical_gpu'] = gpu
                if installed != expected:
                    raise SystemExit(13)
                (claim / 'owner.json').unlink(); claim.rmdir()
                print('{}')
            elif action == 'mark':
                marker = json.loads(value('--payload-json'))
                if (marker.get('kind') == 'failed_retention'
                        and os.environ.get('FAKE_REAL_MARK_DRIVER')):
                    import importlib.util
                    real_driver = Path(
                        os.environ['FAKE_REAL_MARK_DRIVER']).resolve()
                    real_parent = real_driver.parent
                    fake_parent = Path(__file__).resolve().parent
                    assert real_parent != fake_parent
                    assert Path('/tmp') != real_parent
                    assert Path('/tmp') not in real_parent.parents
                    sys.path.insert(0, str(real_parent))
                    assert Path(sys.path[0]).resolve() == real_parent
                    spec = importlib.util.spec_from_file_location(
                        'reviewed_real_driver', real_driver)
                    reviewed = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(reviewed)
                    assert Path(reviewed.__file__).resolve() == real_driver
                    assert (Path(reviewed.formal_registry.__file__).resolve().parent
                            == real_parent)
                    reviewed.install_marker(
                        Path(os.environ['FAKE_REAL_MARK_ROOT']),
                        value('--name'), marker)
                if (marker.get('kind') == 'failed_retention'
                        and os.environ.get('FAKE_RETENTION_MARK_RELEASE')):
                    Path(os.environ['FAKE_RETENTION_MARK_RELEASE']).touch()
                install(root / value('--name'), marker)
                if value('--name') == 'FAILED_JOB' and os.environ.get('FAKE_CLEANUP_MARK_BLOCK'):
                    Path(os.environ['FAKE_AUDIT_READY']).touch()
                    while not Path(os.environ['FAKE_AUDIT_RELEASE']).exists():
                        time.sleep(.02)
                print('{}')
            elif action == 'finalize':
                assert phase_ready('formal') and phase_ready('explanation')
                records = list((root / 'records').glob('*.json')) if (root / 'records').is_dir() else []
                phases = [json.loads(path.read_text()).get('phase') for path in records]
                if (phases.count('formal') != int(os.environ['FAKE_FORMAL_TOTAL'])
                        or phases.count('explanation') != int(os.environ['FAKE_EXPLANATION_TOTAL'])):
                    raise SystemExit(11)
                tables = root / 'tables'; tables.mkdir()
                table = ('PILOT_TABLE.csv' if os.environ.get('VFCL_EXPERIMENT_PROFILE')
                         == 'seed42-pilot' else 'FORMAL_TABLE.csv')
                if os.environ.get('VFCL_EXPERIMENT_PROFILE') == 'seed42-adaptive-recovery':
                    table = 'RECOVERY_TABLE.csv'
                if os.environ.get('VFCL_EXPERIMENT_PROFILE') == 'full-public-matrix':
                    reuse = json.loads((root / 'FULL_MATRIX_REUSE.json').read_text())
                    assert len(records) + len(reuse) == 126
                    assert not list((root / 'gpu_claims').glob('gpu-*'))
                    assert not list(Path(os.environ['TMPDIR']).glob('formal-*-gates.*'))
                    table = 'FULL_MATRIX_TABLE.csv'
                    event('finalize', rows=len(records) + len(reuse))
                if os.environ.get('VFCL_EXPERIMENT_PROFILE') == 'single-dataset-verified-continuation-v1':
                    assert len(records) == 24
                    table = 'CIFAR_CONTINUATION_TABLE.csv'
                    event('finalize', rows=42)
                (tables / table).write_text('ok\n')
                print('{}')
            else:
                raise SystemExit(90)
        ''').lstrip()

    @staticmethod
    def _fake_retention_source():
        return textwrap.dedent(r'''
            import json
            import os
            from pathlib import Path
            import sys
            import time

            def value(name):
                index = sys.argv.index(name)
                return sys.argv[index + 1]

            root = Path(value('--root'))
            key = value('--spec-key')
            owner = json.loads(value('--owner-json'))
            mode = 'dry-run' if '--dry-run' in sys.argv else 'apply'
            assert set(sys.argv[1:]) >= {
                '--root', str(root), '--worktree', str(Path(__file__).resolve().parent),
                '--expected-head', os.environ['FAKE_GIT_HEAD'], '--spec-key', key,
                '--owner-json', value('--owner-json'), '--' + mode,
            }
            assert json.loads((root / 'audit-owner.json').read_text()) == owner
            active = json.loads((root / 'audit_queue/active.json').read_text())
            assert active['spec_key'] == key
            with Path(os.environ['FAKE_RETENTION_LOG']).open('a') as stream:
                stream.write(' '.join(sys.argv[1:]) + '\n')
            with Path(os.environ['FAKE_PIPELINE_EVENTS']).open('a') as stream:
                stream.write(json.dumps({'kind': 'retention-' + mode, 'key': key},
                                        sort_keys=True, separators=(',', ':')) + '\n')
            failure = os.environ.get('FAKE_RETENTION_FAIL')
            if mode == 'dry-run':
                if failure == mode:
                    raise SystemExit(31)
                print('{"applied":false}')
                raise SystemExit(0)
            run = Path(active['run_dir'])
            plan = run / 'PRUNE_PLAN.json'
            plan.write_text('{"kind":"formal_retention_plan"}\n')
            quarantine = run / 'retention-quarantine'
            quarantine.mkdir(mode=0o700)
            tombstone = quarantine / 'event_0_CIL.pt'
            tombstone.write_bytes(b'')
            (run / 'PRUNED_EVIDENCE.json').write_text(
                '{"kind":"formal_pruned_evidence","tombstones":1}\n')
            ready = Path(os.environ['FAKE_RETENTION_READY'])
            if os.environ.get('FAKE_RETENTION_BLOCK'):
                ready.write_text('ready\n')
                release = Path(os.environ['FAKE_RETENTION_RELEASE'])
                while not release.exists():
                    time.sleep(.02)
            if failure == mode:
                raise SystemExit(32)
            print('{"applied":true}')
        ''').lstrip()

    def queue(self, phase, *jobs):
        (self.root / f'FAKE_{phase.upper()}_QUEUE').write_text(
            ''.join(job + '\n' for job in jobs))
        self.env[f'FAKE_{phase.upper()}_TOTAL'] = str(len(jobs))

    def run_launcher(self, *args, env=None, timeout=20):
        return subprocess.run(
            ['bash', str(self.launcher), *map(str, args)],
            text=True, capture_output=True, timeout=timeout,
            env=self.env if env is None else env,
        )

    def new_smoke_root(self, name='generated-smoke'):
        root = self.base / name
        root.mkdir()
        return root

    @staticmethod
    def snapshot(path):
        return sorted(
            (entry.relative_to(path).as_posix(), entry.is_dir(),
             None if entry.is_dir() else entry.read_bytes())
            for entry in path.rglob('*')
        )

    def test_static_launcher_has_one_authority_and_no_manual_membership(self):
        source = self.launcher.read_text()
        self.assertNotIn('/home/chase', source)
        self.assertNotIn('tiny-imagenet', source.lower())
        self.assertNotIn('tee ', source)
        for forbidden in ('cifar100', 'isolet', 'upmc_food101',
                          'finetune', 'fixed_half', 'sample_mean_nll'):
            self.assertNotIn(forbidden, source)
        self.assertIn('SMOKE_MODE', source)
        self.assertIn('three_dataset_formal_driver.py', source)

    def test_smoke_child_registration_stops_after_process_disappears(self):
        source = self.launcher.read_text()
        begin = '    SMOKE_CHILD_PGID=$SMOKE_CHILD_PID\n    start=; pgid=\n'
        end = '    "$REVIEWED_PYTHON" "$DRIVER" smoke-record --root "$FORMAL_ROOT" '
        self.assertEqual(1, source.count(begin))
        self.assertEqual(1, source.count(end))
        first, last = source.index(begin), source.index(end)
        self.assertLess(first, last)
        control_flow = source[first:last]
        self.assertEqual(1, source.count(control_flow))
        harness = self.base / 'smoke-child-registration.sh'
        harness.write_text(textwrap.dedent(r'''
            set -uo pipefail
            WAIT_STATUS=$2; ROUND=0
            exec 3>"$3"
            IFS=, read -r -a START_VALUES <<< "$4"
            IFS=, read -r -a PGID_VALUES <<< "$5"
            IFS=, read -r -a STATE_VALUES <<< "$6"
            EXISTS_STATUS=$7; OWNED_STATUS=$8
            SMOKE_CHILD_PID=424242; SMOKE_CHILD_TOKEN=test-token
            PENDING_SMOKE_SIGNAL=0; log=unused
            # Trace survives the command-substitution subshells.
            proc_start() {
              printf 'start %s\n' "$1" >&3
              local value=${START_VALUES[ROUND]:-${START_VALUES[-1]}}
              [[ $value != absent ]] || return 1
              printf '%s\n' "$value"
            }
            proc_pgid() {
              printf 'pgid %s\n' "$1" >&3
              local value=${PGID_VALUES[ROUND]:-${PGID_VALUES[-1]}}
              [[ $value != absent ]] || return 1
              printf '%s\n' "$value"
            }
            proc_state() {
              printf 'state %s\n' "$1" >&3
              local value=${STATE_VALUES[ROUND]:-${STATE_VALUES[-1]}}
              [[ $value != absent ]] || return 1
              printf '%s\n' "$value"
            }
            sleep() {
              printf 'sleep %s\n' "$*" >&3
              ROUND=$((ROUND + 1))
            }
            wait() { printf 'wait %s\n' "$1" >&3; return "$WAIT_STATUS"; }
            smoke_child_owned() {
              printf 'owned %s\n' "$SMOKE_CHILD_PID" >&3
              return "$OWNED_STATUS"
            }
            smoke_stop_child() { printf 'stop-child %s\n' "$SMOKE_CHILD_PID" >&3; }
            smoke_mark_failure() { printf 'failure %s\n' "$1" >&3; }
            smoke_signal() { printf 'dispatch %s\n' "$1" >&3; }
            kill() {
              if [[ $# -eq 2 && $1 == -0 && $2 == "$SMOKE_CHILD_PID" ]]; then
                printf 'probe %s\n' "$2" >&3
                return "$EXISTS_STATUS"
              fi
              printf 'signal %s\n' "$*" >&3
            }
            chmod() { :; }
            trap() { :; }
            run_registration() {
        ''') + control_flow + textwrap.dedent(r'''
              printf 'smoke-record\n' >&3
              return 0
            }
            run_registration
            exit $?
        '''))
        completed_probe = 'start pgid state probe state probe wait '
        registered_exit = 'start pgid owned state '
        transient = 'start pgid state probe sleep start pgid owned wait smoke-record'
        live_polls = 'start pgid state sleep ' * 100 + 'state stop-child failure'
        unreadable_polls = 'start pgid state probe sleep ' * 100 + 'state probe '
        cases = (
            # name, per-round start/PGID/state, existence/ownership status,
            # wait status, expected return, exact observation/side-effect order.
            ('missing', 'absent', 'absent', 'absent', 1, 1, 7, 7,
             completed_probe + 'failure'),
            ('stale-success', '123', 'absent', 'absent', 1, 1, 0, 0,
             completed_probe + 'smoke-record'),
            ('stale-failure', '123', 'absent', 'absent', 1, 1, 17, 17,
             completed_probe + 'failure'),
            ('stale-zombie', '123', 'absent', 'Z', 0, 1, 7, 7,
             'start pgid state state wait failure'),
            ('transient-missing-start', 'absent,123', 'absent,424242', 'absent,S',
             0, 0, 0, 0, transient),
            ('transient-stale-start', '123', '424243,424242', 'absent,S',
             0, 0, 0, 0, transient),
            ('owned-then-missing-success', '123', '424242', 'absent', 1, 1, 0, 0,
             registered_exit + 'probe wait smoke-record'),
            ('owned-then-missing-failure', '123', '424242', 'absent', 1, 1, 17, 17,
             registered_exit + 'probe wait failure'),
            ('owned-then-zombie-success', '123', '424242', 'Z', 0, 1, 0, 0,
             registered_exit + 'wait smoke-record'),
            ('owned-then-zombie-failure', '123', '424242', 'Z', 0, 1, 17, 17,
             registered_exit + 'wait failure'),
            ('wrong-pgid', '123', '424243', 'S', 0, 1, 0, 1, live_polls),
            ('bad-token', '123', '424242', 'S', 0, 1, 0, 1,
             registered_exit + 'stop-child failure'),
            ('unreadable-wrong-pgid', '123', '424243', 'absent', 0, 1, 0, 1,
             unreadable_polls + 'stop-child failure'),
            ('unreadable-bad-token', '123', '424242', 'absent', 0, 1, 0, 1,
             registered_exit + 'probe stop-child failure'),
            ('unreadable-through-cap', 'absent', 'absent', 'absent', 0, 1, 0, 1,
             unreadable_polls + 'signal wait failure'),
        )
        pid_events = {'start', 'pgid', 'state', 'probe', 'owned', 'wait', 'stop-child'}
        for (case, starts, pgids, states, exists, owned,
             wait_status, status, events) in cases:
            with self.subTest(case=case):
                trace = self.base / f'{case}-registration.trace'
                completed = subprocess.run(
                    ['bash', str(harness), case, str(wait_status), str(trace),
                     starts, pgids, states, str(exists), str(owned)],
                    env=self.env, text=True, capture_output=True, timeout=20)
                self.assertEqual(status, completed.returncode, completed.stderr)
                rows = trace.read_text().splitlines()
                expected = []
                for event in events.split():
                    if event in pid_events:
                        expected.append(f'{event} 424242')
                    else:
                        expected.append({'sleep': 'sleep 0.01',
                                         'failure': f'failure {status}',
                                         'signal': 'signal -TERM 424242',
                                         'smoke-record': 'smoke-record'}[event])
                self.assertEqual(expected, rows)

    def test_full_smoke_runs_exact_42_generated_jobs_with_gpu_reassignment(self):
        self.env.update(VFCL_EXPERIMENT_PROFILE='full-public-matrix',
                        FAKE_GIT_BRANCH=FULL_MATRIX_BRANCH,
                        FAKE_SWITCH_GPU_AFTER_JOB='cifar100-finetune')
        root = self.new_smoke_root('full-generated')
        completed = self.run_launcher('--smoke', root, timeout=90)
        self.assertEqual(0, completed.returncode, completed.stderr)
        rows = [line.split('|') for line in self.worker_log.read_text().splitlines()]
        expected = [f'{dataset}-{method}'.replace('_', '-')
                    for dataset in driver.formal_registry.DATASETS
                    for method in driver.formal_registry.FULL_MATRIX_METHODS]
        self.assertEqual(expected, [row[0] for row in rows])
        self.assertEqual(['0'] + ['1'] * 41, [row[1] for row in rows])
        self.assertEqual(42, len(list((root / 'records').glob('*.json'))))
        self.assertTrue((root / 'SMOKE_EXECUTION_SUCCESS').exists())
        self.assertFalse(any((root / name).exists() for name in driver._SMOKE_FORMAL_NAMES))
        actions = [row.split()[0] for row in self.driver_log.read_text().splitlines()]
        self.assertEqual(42, actions.count('smoke-record'))
        self.assertFalse(set(actions) & {'claim', 'disk-status', 'finalize', 'mark'})
        self.assertEqual([], self._live_logged_fake_children())
        self.assertEqual([], self._live_logged_dormant_children())

    def test_full_smoke_accepts_single_gpu_and_runs_every_job_on_zero(self):
        self.env.update(
            VFCL_EXPERIMENT_PROFILE='full-public-matrix',
            FAKE_GIT_BRANCH=FULL_MATRIX_BRANCH,
            FAKE_GPU_ROWS='0, GPU-0, 8000, 0',
            VFCL_GPU_COUNT='1',
        )
        root = self.new_smoke_root('full-generated-single-gpu')
        completed = self.run_launcher('--smoke', root, timeout=90)
        self.assertEqual(0, completed.returncode, completed.stderr)
        rows = [line.split('|')
                for line in self.worker_log.read_text().splitlines()]
        self.assertEqual(42, len(rows))
        self.assertEqual(['0'] * 42, [row[1] for row in rows])
        self.assertTrue((root / 'SMOKE_EXECUTION_SUCCESS').exists())
        self.assertFalse((root / 'SMOKE_FAILED').exists())
        self.assertEqual([], self._live_logged_fake_children())
        self.assertEqual([], self._live_logged_dormant_children())

    @mock.patch.dict(os.environ, {'VFCL_EXPERIMENT_PROFILE': 'full-public-matrix'})
    def test_full_smoke_real_command_handoff_keeps_ten_task_cifar_bic(self):
        # Exercise the real command boundary; the shell's fake worker never trains.
        root = self.new_smoke_root('full-command-handoff')
        plan = driver.plan_smoke(root)
        for job in plan['jobs']:
            with self.subTest(job=job['job_id']), mock.patch(
                    'sys.stdout', new=io.TextIOWrapper(
                        io.BytesIO(), encoding='utf-8')) as output:
                self.assertEqual(0, driver.main([
                    'smoke-command', '--root', str(root), '--job', job['job_id'],
                    '--run-dir', str(root / job['run_dir']), '--format', 'nul']))
                output.flush()
                tokens = output.buffer.getvalue().rstrip(b'\0').decode().split('\0')
                self.assertEqual(job['command'], tokens)
                flags = driver._smoke_command_options(tokens)
                image = job['dataset'] == 'cifar100'
                self.assertEqual('10' if image else '2', flags['--num_tasks'])
                self.assertEqual('20' if image else '4', flags['--num_classes'])
                self.assertEqual('1' if image else '0', flags['--bic_enabled'])
                replay = {name: flags[name] for name in (
                    '--er_ace_buffer_size', '--er_ace_batch') if name in flags}
                self.assertEqual(
                    {'--er_ace_buffer_size': '0', '--er_ace_batch': '64'}
                    if job['method'] == 'er_ace' else {}, replay)

    def test_full_smoke_rejects_four_or_41_job_enumeration_before_worker(self):
        self.env.update(VFCL_EXPERIMENT_PROFILE='full-public-matrix',
                        FAKE_GIT_BRANCH=FULL_MATRIX_BRANCH)
        for count in (4, 41):
            root = self.new_smoke_root(f'wrong-full-{count}')
            result = self.run_launcher('--smoke', root, env={**self.env,
                'FAKE_SMOKE_JOBS_ROWS': ','.join(f'job-{i}' for i in range(count))})
            self.assertNotEqual(0, result.returncode)
            self.assertIn('42', result.stderr)
            self.assertFalse(self.worker_log.exists())
            self.assertTrue((root / 'SMOKE_FAILED').exists())

    def test_full_smoke_preflight_accepts_only_scientific_correctness_branch(self):
        library = self.worktree / 'launcher-library.sh'
        library.write_text(self.launcher.read_text().rsplit('\nmain "$@"', 1)[0])
        branches = (RECOVERY_BRANCH, BRANCH, 'codex/full-public-matrix',
                    'codex/arbitrary', FULL_MATRIX_BRANCH + '-ad-hoc', '',
                    FULL_MATRIX_BRANCH)
        for index, branch in enumerate(branches):
            with self.subTest(branch=branch):
                root = self.new_smoke_root(f'branch-preflight-{index}')
                driver_log = self.base / f'branch-preflight-{index}.log'
                completed = subprocess.run(
                    ['bash', '-c', 'source "$1"; smoke_preflight "$2"',
                     'test', str(library), str(root)],
                    env={**self.env, 'VFCL_EXPERIMENT_PROFILE': 'full-public-matrix',
                         'FAKE_GIT_BRANCH': branch, 'FAKE_DRIVER_LOG': str(driver_log)},
                    capture_output=True, text=True, timeout=20)
                if branch == FULL_MATRIX_BRANCH:
                    self.assertEqual(0, completed.returncode, completed.stderr)
                    actions = [row.split()[0] for row in
                               driver_log.read_text().splitlines()]
                    self.assertEqual(['smoke-plan', 'smoke-check'], actions)
                else:
                    self.assertNotEqual(0, completed.returncode)
                    self.assertIn('implementation branch differs', completed.stderr)
                    self.assertEqual([], list(root.iterdir()))
                    self.assertFalse(driver_log.exists())
                self.assertFalse(self.worker_log.exists())

    def test_reviewed_branch_profile_mapping_remains_exact(self):
        library = self.worktree / 'launcher-library.sh'
        library.write_text(self.launcher.read_text().rsplit('\nmain "$@"', 1)[0])
        for profile, expected in ((None, BRANCH), ('formal', BRANCH),
                ('seed42-pilot', BRANCH), ('seed42-adaptive-recovery', RECOVERY_BRANCH),
                ('full-public-matrix', FULL_MATRIX_BRANCH),
                ('single-dataset-full-matrix', DATASET_BRANCH),
                ('single-dataset-verified-continuation-v1',
                 CONTINUATION_BRANCH)):
            with self.subTest(profile=profile):
                env = {key: value for key, value in self.env.items()
                       if key != 'VFCL_EXPERIMENT_PROFILE'}
                if profile is not None:
                    env['VFCL_EXPERIMENT_PROFILE'] = profile
                completed = subprocess.run(
                    ['bash', '-c', 'source "$1"; reviewed_branch', 'test', str(library)],
                    env=env, capture_output=True, text=True, timeout=20)
                self.assertEqual(0, completed.returncode, completed.stderr)
                self.assertEqual(expected + '\n', completed.stdout)

    def test_copied_smoke_success_runs_exact_four_with_gpu_mapping_and_isolation(self):
        root = self.new_smoke_root()
        completed = self.run_launcher('--smoke', root)
        self.assertEqual(0, completed.returncode, completed.stderr)
        rows = [line.split('|') for line in self.worker_log.read_text().splitlines()]
        self.assertEqual([
            'image-replay-free', 'vector-raw-replay',
            'vector-fixed-endpoint', 'vector-adaptive',
        ], [row[0] for row in rows])
        self.assertEqual(['0', '1', '0', '1'], [row[1] for row in rows])
        self.assertTrue((root / 'SMOKE_AUDIT.json').is_file())
        self.assertTrue((root / 'SMOKE_EXECUTION_SUCCESS').is_file())
        self.assertFalse(any((root / name).exists() for name in (
            'FORMAL_REGISTRY.json', 'COMPATIBILITY_CENSUS.json',
            'FORMAL_PLAN.json', 'MISSING_JOBS.json',
            'FORMAL_PHASE_SUCCESS', 'EXPLANATION_PHASE_SUCCESS',
            'FORMAL_EXECUTION_SUCCESS',
        )))
        actions = [line.split()[0] for line in self.driver_log.read_text().splitlines()]
        self.assertNotIn('census', actions)
        self.assertNotIn('plan', actions)
        self.assertNotIn('finalize', actions)
        self.assertEqual([], self._live_logged_fake_children())
        self.assertEqual([], self._live_logged_dormant_children())

    def _launch_until_second_smoke_job_wait(self, root):
        env = self.env.copy()
        gpu0_busy = Path(env['FAKE_GPU0_BUSY_FILE'])
        both_busy = Path(env['FAKE_COMPUTE_BUSY_FILE'])
        query_log = Path(env['FAKE_GPU_QUERY_LOG'])
        gpu0_busy.touch()
        env['FAKE_BUSY_AFTER_JOB'] = 'image-replay-free'
        process = subprocess.Popen(
            ['bash', str(self.launcher), '--smoke', str(root)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, start_new_session=True,
        )
        self.addCleanup(self._stop_process, process)
        deadline = time.time() + 8
        while time.time() < deadline:
            query_count = (query_log.read_text().splitlines().count('compute')
                           if query_log.exists() else 0)
            if both_busy.exists() and query_count >= 5:
                break
            if process.poll() is not None:
                break
            time.sleep(.02)
        self.assertTrue(both_busy.exists(), process.poll())
        self.assertGreaterEqual(query_count, 5)
        self.assertIsNone(process.poll())
        self.assertEqual(
            ['image-replay-free'],
            [row.split('|')[0]
             for row in self.worker_log.read_text().splitlines()],
        )
        self.assertEqual(
            ['image-replay-free.json'],
            sorted(path.name for path in (root / 'control').iterdir()),
        )
        self.assertEqual(
            ['image-replay-free.log'],
            sorted(path.name for path in (root / 'logs').iterdir()),
        )
        self.assertEqual(
            ['image-replay-free.json'],
            sorted(path.name for path in (root / 'records').iterdir()),
        )
        self.assertEqual([], list(self.tmpdir.glob('formal-smoke-*')))
        self.assertEqual(set(), self._new_child_gates())
        return process, env

    def _assert_mid_wait_failure(self, root, process, expected):
        stdout, stderr = process.communicate(timeout=15)
        self.assertEqual(expected, process.returncode, (stdout, stderr))
        self.assertEqual({
            'kind': 'failed', 'exit_code': expected,
            'scientific_gate': False,
        }, json.loads((root / 'SMOKE_FAILED').read_text()))
        self.assertEqual({
            'kind': 'stopped', 'exit_code': expected,
            'scientific_gate': False,
        }, json.loads((root / 'SMOKE_STOPPED').read_text()))
        self.assertEqual(
            ['image-replay-free.json'],
            sorted(path.name for path in (root / 'control').iterdir()),
        )
        self.assertEqual(
            ['image-replay-free.log'],
            sorted(path.name for path in (root / 'logs').iterdir()),
        )
        self.assertEqual(
            ['image-replay-free.json'],
            sorted(path.name for path in (root / 'records').iterdir()),
        )
        self.assertFalse(any((root / name).exists() for name in (
            'SMOKE_AUDIT.json', 'SMOKE_EXECUTION_SUCCESS',
            'FORMAL_PHASE_SUCCESS', 'EXPLANATION_PHASE_SUCCESS',
            'FORMAL_EXECUTION_SUCCESS',
        )))
        self.assertEqual([], self._live_logged_fake_children())
        self.assertEqual([], self._live_logged_dormant_children())
        self.assertEqual([], list(self.tmpdir.glob('formal-smoke-*')))
        self.assertEqual(set(), self._new_child_gates())

    def test_copied_smoke_final_recheck_busy_reselects_before_launch(self):
        root = self.new_smoke_root('generated-smoke-final-recheck')
        env = self.env.copy()
        env.update({
            'FAKE_GPU0_BUSY_CALLS': '2,3,4,5',
            'FAKE_FAIL_JOB': 'image-replay-free',
            'FAKE_FAIL_CODE': '17',
        })
        completed = self.run_launcher('--smoke', root, env=env)
        self.assertEqual(17, completed.returncode, completed.stderr)
        self.assertEqual('5\n', Path(env['FAKE_COMPUTE_CALLS_FILE']).read_text())
        rows = [line.split('|')
                for line in self.worker_log.read_text().splitlines()]
        self.assertEqual([('image-replay-free', '1')], [
            (row[0], row[1]) for row in rows])
        controls = list((root / 'control').iterdir())
        logs = list((root / 'logs').iterdir())
        self.assertEqual(['image-replay-free.json'], [
            path.name for path in controls])
        self.assertEqual(['image-replay-free.log'], [path.name for path in logs])
        launch = json.loads(controls[0].read_text())
        self.assertEqual((0, 1),
                         (launch['preferred_gpu'], launch['physical_gpu']))
        self.assertEqual([], list((root / 'records').iterdir()))
        self.assertEqual([], self._live_logged_fake_children())

    def test_copied_smoke_mid_wait_authority_failure_preserves_completed(self):
        root = self.new_smoke_root('generated-smoke-wait-authority-failure')
        process, env = self._launch_until_second_smoke_job_wait(root)
        Path(env['FAKE_PLAN_BAD_FILE']).touch()
        self._assert_mid_wait_failure(root, process, 1)

    def test_copied_smoke_mid_wait_query_failure_preserves_completed(self):
        root = self.new_smoke_root('generated-smoke-wait-query-failure')
        process, env = self._launch_until_second_smoke_job_wait(root)
        Path(env['FAKE_COMPUTE_FAIL_FILE']).touch()
        self._assert_mid_wait_failure(root, process, 95)

    def test_copied_smoke_falls_back_and_waits_without_rerunning_completed(self):
        root = self.new_smoke_root('generated-smoke-gpu-wait')
        env = self.env.copy()
        gpu0_busy = Path(env['FAKE_GPU0_BUSY_FILE'])
        both_busy = Path(env['FAKE_COMPUTE_BUSY_FILE'])
        query_log = Path(env['FAKE_GPU_QUERY_LOG'])
        gpu0_busy.touch()
        env['FAKE_BUSY_AFTER_JOB'] = 'image-replay-free'
        process = subprocess.Popen(
            ['bash', str(self.launcher), '--smoke', str(root)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, start_new_session=True,
        )
        self.addCleanup(self._stop_process, process)
        deadline = time.time() + 8
        while time.time() < deadline:
            query_count = (query_log.read_text().splitlines().count('compute')
                           if query_log.exists() else 0)
            if both_busy.exists() and query_count >= 5:
                break
            if process.poll() is not None:
                break
            time.sleep(.02)
        self.assertTrue(
            both_busy.exists(),
            f'launcher exited before GPU fallback: {process.poll()}',
        )
        self.assertGreaterEqual(query_count, 5)
        self.assertIsNone(process.poll())
        rows = self.worker_log.read_text().splitlines()
        self.assertEqual(['image-replay-free'], [row.split('|')[0] for row in rows])
        self.assertTrue((root / 'records' / 'image-replay-free.json').is_file())
        self.assertFalse(
            (root / 'control' / 'vector-raw-replay.json').exists())
        self.assertFalse(any((root / name).exists() for name in (
            'SMOKE_FAILED', 'SMOKE_STOPPED', 'SMOKE_AUDIT.json',
            'SMOKE_EXECUTION_SUCCESS',
        )))
        both_busy.unlink()
        stdout, stderr = process.communicate(timeout=20)
        self.assertEqual(0, process.returncode, (stdout, stderr))
        rows = [line.split('|') for line in self.worker_log.read_text().splitlines()]
        self.assertEqual([
            'image-replay-free', 'vector-raw-replay',
            'vector-fixed-endpoint', 'vector-adaptive',
        ], [row[0] for row in rows])
        plan = json.loads((root / 'SMOKE_PLAN.json').read_text())
        self.assertEqual([0, 1, 0, 1], [
            job['preferred_gpu'] for job in plan['jobs']])
        controls = [json.loads((root / 'control' / f'{job}.json').read_text())
                    for job in [
                        'image-replay-free', 'vector-raw-replay',
                        'vector-fixed-endpoint', 'vector-adaptive',
                    ]]
        self.assertEqual([0, 1, 0, 1], [
            control['preferred_gpu'] for control in controls])
        self.assertEqual(['1', '1', '0', '1'], [row[1] for row in rows])
        self.assertEqual([1, 1, 0, 1], [
            control['physical_gpu'] for control in controls])

    def test_copied_smoke_signal_while_waiting_marks_and_cleans(self):
        root = self.new_smoke_root('generated-smoke-gpu-wait-signal')
        env = self.env.copy()
        both_busy = Path(env['FAKE_COMPUTE_BUSY_FILE'])
        query_log = Path(env['FAKE_GPU_QUERY_LOG'])
        both_busy.touch()
        initial_commands = set(self.tmpdir.glob('formal-smoke-command.*'))
        initial_jobs = set(self.tmpdir.glob('formal-smoke-jobs.*'))
        foreign = subprocess.Popen(['/bin/sleep', '30'])
        self.addCleanup(self._stop_process, foreign)
        process = subprocess.Popen(
            ['bash', str(self.launcher), '--smoke', str(root)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, start_new_session=True,
        )
        self.addCleanup(self._stop_process, process)
        deadline = time.time() + 8
        while time.time() < deadline:
            query_count = (query_log.read_text().splitlines().count('compute')
                           if query_log.exists() else 0)
            if query_count >= 2:
                break
            if process.poll() is not None:
                break
            time.sleep(.02)
        self.assertGreaterEqual(
            query_count, 2,
            f'launcher exited before waiting: {process.poll()}',
        )
        self.assertIsNone(process.poll())
        self.assertFalse(self.worker_log.exists())
        self.assertEqual([], list((root / 'control').iterdir()))
        self.assertEqual([], list((root / 'logs').iterdir()))
        self.assertEqual([], list((root / 'records').iterdir()))
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
        self.assertEqual(143, process.returncode, (stdout, stderr))
        failed = json.loads((root / 'SMOKE_FAILED').read_text())
        stopped = json.loads((root / 'SMOKE_STOPPED').read_text())
        self.assertEqual(('failed', 143),
                         (failed['kind'], failed['exit_code']))
        self.assertEqual(('stopped', 143),
                         (stopped['kind'], stopped['exit_code']))
        self.assertFalse(any((root / name).exists() for name in (
            'SMOKE_AUDIT.json', 'SMOKE_EXECUTION_SUCCESS',
            'FORMAL_PHASE_SUCCESS', 'EXPLANATION_PHASE_SUCCESS',
            'FORMAL_EXECUTION_SUCCESS',
        )))
        self.assertEqual([], self._live_logged_fake_children())
        self.assertEqual([], self._live_logged_dormant_children())
        self.assertIsNone(foreign.poll())
        self.assertEqual(initial_commands,
                         set(self.tmpdir.glob('formal-smoke-command.*')))
        self.assertEqual(initial_jobs,
                         set(self.tmpdir.glob('formal-smoke-jobs.*')))
        self.assertEqual(set(), self._new_child_gates())

    def test_copied_smoke_first_failure_stops_remaining_jobs_and_marks_nonformal(self):
        root = self.new_smoke_root('generated-smoke-failure')
        env = self.env.copy()
        env.update({
            'FAKE_FAIL_JOB': 'vector-raw-replay',
            'FAKE_FAIL_CODE': '17',
        })
        completed = self.run_launcher('--smoke', root, env=env)
        self.assertEqual(17, completed.returncode, completed.stderr)
        rows = [line.split('|', 1)[0]
                for line in self.worker_log.read_text().splitlines()]
        self.assertEqual(['image-replay-free', 'vector-raw-replay'], rows)
        self.assertTrue((root / 'SMOKE_FAILED').is_file())
        self.assertTrue((root / 'SMOKE_STOPPED').is_file())
        self.assertFalse((root / 'SMOKE_EXECUTION_SUCCESS').exists())
        self.assertFalse((root / 'FORMAL_EXECUTION_SUCCESS').exists())
        self.assertEqual([], self._live_logged_fake_children())
        self.assertEqual([], self._live_logged_dormant_children())

    def test_copied_smoke_signal_reaps_owned_child_and_never_falls_into_formal(self):
        root = self.new_smoke_root('generated-smoke-signal')
        env = self.env.copy()
        env['FAKE_SLOW_JOB'] = 'image-replay-free'
        initial_commands = set(self.tmpdir.glob('formal-smoke-command.*'))
        initial_jobs = set(self.tmpdir.glob('formal-smoke-jobs.*'))
        foreign = subprocess.Popen(['/bin/sleep', '30'])
        self.addCleanup(self._stop_process, foreign)
        process = subprocess.Popen(
            ['bash', str(self.launcher), '--smoke', str(root)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, start_new_session=True,
        )
        self.addCleanup(self._stop_process, process)
        ready = Path(self.env['FAKE_READY_LOG'])
        deadline = time.time() + 8
        while time.time() < deadline and not ready.exists():
            time.sleep(.02)
        self.assertTrue(ready.exists(), 'smoke child never reached its ready point')
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
        self.assertEqual(143, process.returncode, (stdout, stderr))
        self.assertTrue((root / 'SMOKE_FAILED').is_file())
        self.assertTrue((root / 'SMOKE_STOPPED').is_file())
        failed = json.loads((root / 'SMOKE_FAILED').read_text())
        stopped = json.loads((root / 'SMOKE_STOPPED').read_text())
        self.assertEqual(('failed', 143),
                         (failed['kind'], failed['exit_code']))
        self.assertEqual(('stopped', 143),
                         (stopped['kind'], stopped['exit_code']))
        self.assertFalse((root / 'SMOKE_EXECUTION_SUCCESS').exists())
        self.assertFalse((root / 'FORMAL_EXECUTION_SUCCESS').exists())
        self.assertEqual([], self._live_logged_fake_children())
        self.assertEqual([], self._live_logged_dormant_children())
        self.assertIsNone(foreign.poll())
        self.assertEqual(initial_commands,
                         set(self.tmpdir.glob('formal-smoke-command.*')))
        self.assertEqual(initial_jobs,
                         set(self.tmpdir.glob('formal-smoke-jobs.*')))
        self.assertEqual(set(), self._new_child_gates())

    def test_copied_smoke_signal_preserves_existing_failure_evidence(self):
        root = self.new_smoke_root('generated-smoke-signal-existing-failure')
        env = self.env.copy()
        env['FAKE_SLOW_JOB'] = 'image-replay-free'
        process = subprocess.Popen(
            ['bash', str(self.launcher), '--smoke', str(root)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, start_new_session=True,
        )
        self.addCleanup(self._stop_process, process)
        ready = Path(self.env['FAKE_READY_LOG'])
        deadline = time.time() + 8
        while time.time() < deadline and not ready.exists():
            time.sleep(.02)
        self.assertTrue(ready.exists(), 'smoke child never reached its ready point')
        installed = subprocess.run([
            PYTHON, str(self.worktree / 'three_dataset_formal_driver.py'),
            'smoke-mark', '--root', str(root), '--name', 'SMOKE_FAILED',
            '--kind', 'failed', '--exit-code', '17',
        ], text=True, capture_output=True, env=env)
        self.assertEqual(0, installed.returncode, installed.stderr)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
        self.assertEqual(143, process.returncode, (stdout, stderr))
        failed = json.loads((root / 'SMOKE_FAILED').read_text())
        stopped = json.loads((root / 'SMOKE_STOPPED').read_text())
        self.assertEqual(('failed', 17),
                         (failed['kind'], failed['exit_code']))
        self.assertEqual(('stopped', 143),
                         (stopped['kind'], stopped['exit_code']))
        self.assertFalse((root / 'SMOKE_EXECUTION_SUCCESS').exists())
        self.assertFalse((root / 'FORMAL_EXECUTION_SUCCESS').exists())
        self.assertEqual([], self._live_logged_fake_children())

    def test_copied_smoke_job_enumeration_fails_closed_with_both_markers(self):
        cases = (
            ('driver-error', {'FAKE_SMOKE_JOBS_FAIL': '23'}, 23),
            ('zero', {'FAKE_SMOKE_JOBS_ROWS': ''}, 1),
            ('non-four', {'FAKE_SMOKE_JOBS_ROWS': 'one,two,three'}, 1),
            ('duplicate', {
                'FAKE_SMOKE_JOBS_ROWS': 'one,two,two,four'}, 1),
            ('parse', {
                'FAKE_SMOKE_JOBS_ROWS': 'one,two,three,bad/job'}, 1),
        )
        for name, change, expected in cases:
            with self.subTest(name=name):
                root = self.new_smoke_root(f'smoke-jobs-{name}')
                completed = self.run_launcher(
                    '--smoke', root, env={**self.env, **change})
                self.assertEqual(expected, completed.returncode, completed.stderr)
                self.assertTrue((root / 'SMOKE_FAILED').is_file())
                self.assertTrue((root / 'SMOKE_STOPPED').is_file())
                self.assertFalse((root / 'SMOKE_EXECUTION_SUCCESS').exists())
                self.assertFalse((root / 'FORMAL_EXECUTION_SUCCESS').exists())
        self.assertFalse(self.worker_log.exists())

    def test_copied_smoke_signal_during_job_enumeration_reaps_driver(self):
        root = self.new_smoke_root('smoke-jobs-signal')
        env = self.env.copy()
        env['FAKE_SMOKE_JOBS_DELAY'] = '30'
        initial_commands = set(self.tmpdir.glob('formal-smoke-command.*'))
        initial_jobs = set(self.tmpdir.glob('formal-smoke-jobs.*'))
        foreign = subprocess.Popen(['/bin/sleep', '30'])
        self.addCleanup(self._stop_process, foreign)
        process = subprocess.Popen(
            ['bash', str(self.launcher), '--smoke', str(root)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, start_new_session=True,
        )
        self.addCleanup(self._stop_process, process)
        ready = Path(env['FAKE_SMOKE_JOBS_READY'])
        deadline = time.time() + 8
        while time.time() < deadline and not ready.exists():
            time.sleep(.02)
        self.assertTrue(ready.exists(), 'smoke-jobs driver never became ready')
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
        self.assertEqual(143, process.returncode, (stdout, stderr))
        self.assertTrue((root / 'SMOKE_FAILED').is_file())
        self.assertTrue((root / 'SMOKE_STOPPED').is_file())
        failed = json.loads((root / 'SMOKE_FAILED').read_text())
        stopped = json.loads((root / 'SMOKE_STOPPED').read_text())
        self.assertEqual(('failed', 143),
                         (failed['kind'], failed['exit_code']))
        self.assertEqual(('stopped', 143),
                         (stopped['kind'], stopped['exit_code']))
        self.assertFalse((root / 'SMOKE_EXECUTION_SUCCESS').exists())
        self.assertFalse((root / 'FORMAL_EXECUTION_SUCCESS').exists())
        driver_pid = int(Path(env['FAKE_SMOKE_JOBS_PID_FILE']).read_text())
        deadline = time.time() + 5
        while time.time() < deadline and Path(f'/proc/{driver_pid}').exists():
            time.sleep(.02)
        if Path(f'/proc/{driver_pid}').exists():
            os.kill(driver_pid, signal.SIGKILL)
            self.fail('smoke-jobs driver survived launcher signal cleanup')
        self.assertEqual([], self._live_logged_fake_children())
        self.assertEqual([], self._live_logged_dormant_children())
        self.assertIsNone(foreign.poll())
        self.assertEqual(initial_commands,
                         set(self.tmpdir.glob('formal-smoke-command.*')))
        self.assertEqual(initial_jobs,
                         set(self.tmpdir.glob('formal-smoke-jobs.*')))
        self.assertEqual(set(), self._new_child_gates())

    def test_copied_launcher_real_driver_plans_then_stops_before_gpu_command(self):
        from three_dataset_formal_audit import _SOURCE_INVENTORY
        source = SCRIPT.parent
        worktree = self.base / 'real-driver-worktree'
        worktree.mkdir()
        launcher = worktree / SCRIPT.name
        shutil.copy2(SCRIPT, launcher)
        launcher.chmod(0o755)
        shutil.copy2(
            source / 'three_dataset_formal_driver.py',
            worktree / 'three_dataset_formal_driver.py',
        )
        shutil.copy2(
            source / 'prune_completed_runs.py',
            worktree / 'prune_completed_runs.py',
        )
        for logical in _SOURCE_INVENTORY:
            destination = worktree / logical
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / logical, destination)
        root = self.new_smoke_root('real-driver-generated-smoke')
        env = self.env.copy()
        env['FAKE_GPU_FAIL'] = '1'
        env['PYTHONPATH'] = str(source) + os.pathsep + env.get('PYTHONPATH', '')
        completed = subprocess.run(
            ['bash', str(launcher), '--smoke', str(root)],
            text=True, capture_output=True, timeout=30, env=env,
        )
        self.assertEqual(95, completed.returncode, completed.stderr)
        self.assertTrue((root / 'SMOKE_PLAN.json').is_file())
        self.assertTrue((root / 'SMOKE_FAILED').is_file())
        self.assertTrue((root / 'SMOKE_STOPPED').is_file())
        self.assertEqual([], list((root / 'runs').iterdir()))
        self.assertFalse((root / 'SMOKE_EXECUTION_SUCCESS').exists())
        self.assertFalse((root / 'FORMAL_EXECUTION_SUCCESS').exists())

    def test_check_is_exact_nonmutating_and_smoke_or_external_mode_rejects(self):
        before = self.snapshot(self.root)
        checked = self.run_launcher('--check', self.root)
        self.assertEqual(0, checked.returncode, checked.stderr)
        self.assertEqual(before, self.snapshot(self.root))
        smoke = self.run_launcher('--smoke', self.root)
        self.assertNotEqual(0, smoke.returncode)
        self.assertIn('generated smoke root must be empty', smoke.stderr.lower())
        env = dict(self.env, SMOKE_MODE='1')
        external = self.run_launcher('--check', self.root, env=env)
        self.assertNotEqual(0, external.returncode)
        self.assertIn('SMOKE_MODE is not supported', external.stderr)

    def test_check_rejects_git_python_plan_and_disk_mismatches(self):
        cases = (
            ('dirty', {'FAKE_GIT_DIRTY': '1'}),
            ('branch', {'FAKE_GIT_BRANCH': 'wrong'}),
            ('head', {'FAKE_GIT_HEAD': '0' * 40}),
            ('python', {'VFCL_PYTHON': '/usr/bin/python3'}),
            ('plan', {'FAKE_PLAN_BAD': '1'}),
            ('disk', {'FAKE_DISK_KB': '1'}),
        )
        for name, change in cases:
            with self.subTest(name=name):
                completed = self.run_launcher(
                    '--check', self.root, env={**self.env, **change})
                self.assertNotEqual(0, completed.returncode)

    def test_check_rejects_root_symlinks_escape_and_existing_markers(self):
        target = self.base / 'target-root'
        shutil.copytree(self.root, target)
        final_link = self.base / 'final-link'
        final_link.symlink_to(target, target_is_directory=True)
        self.assertNotEqual(
            0, self.run_launcher('--check', final_link).returncode)

        real_parent = self.base / 'real-parent'
        real_parent.mkdir()
        ancestor_root = real_parent / 'root'
        shutil.copytree(self.root, ancestor_root)
        linked_parent = self.base / 'linked-parent'
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        self.assertNotEqual(
            0, self.run_launcher('--check', linked_parent / 'root').returncode)
        self.assertNotEqual(
            0, self.run_launcher('--check', self.root / '..' / 'formal').returncode)

        (self.root / 'FORMAL_STOPPED').write_text('{}\n')
        self.assertNotEqual(
            0, self.run_launcher('--check', self.root).returncode)

    def test_success_partitions_jobs_across_two_gpus_and_finishes_after_summary(self):
        self.env['FAKE_MEMAVAILABLE_KB'] = '0'
        self.queue('formal', 'formal:a:42', 'formal:b:42', 'formal:c:42')
        self.queue('explanation', 'explanation:a:42', 'explanation:b:42')
        completed = self.run_launcher(self.root)
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertFalse(Path(self.env['FAKE_MEMORY_LOG']).exists())
        self.assertFalse((self.root / 'PILOT_EXECUTION_SUCCESS').exists())
        for marker in ('FORMAL_PHASE_SUCCESS', 'EXPLANATION_PHASE_SUCCESS',
                       'FORMAL_EXECUTION_SUCCESS'):
            self.assertTrue((self.root / marker).is_file(), marker)
        records = list((self.root / 'records').glob('*.json'))
        self.assertEqual(5, len(records))
        self.assertTrue((self.root / 'tables' / 'FORMAL_TABLE.csv').is_file())
        rows = self.worker_log.read_text().splitlines()
        self.assertEqual(5, len(rows))
        self.assertEqual({'0', '1'}, {row.split('|')[1] for row in rows})
        self.assertTrue(all('--device cuda:0' in row for row in rows))
        self.assertTrue(all('|:4096:8|1|42' in row for row in rows))
        temp_templates = Path(self.env['FAKE_MKTEMP_LOG']).read_text().splitlines()
        self.assertTrue(temp_templates)
        self.assertTrue(all(
            Path(template.split()[-1]).parent == self.tmpdir
            for template in temp_templates
        ), temp_templates)

    def pipeline_events(self):
        path = Path(self.env['FAKE_PIPELINE_EVENTS'])
        return [json.loads(row) for row in path.read_text().splitlines()] if path.exists() else []

    def memory_reads(self):
        path = Path(self.env['FAKE_MEMORY_LOG'])
        return [row.split('|') for row in path.read_text().splitlines()] if path.exists() else []

    def set_memory(self, **values):
        path = self.base / 'memory-state.json'
        pending = self.base / 'memory-state.pending'
        pending.write_text(canonical(values))
        pending.replace(path)
        self.env['FAKE_MEMAVAILABLE_FILE'] = str(path)

    def assert_no_unstarted_work(self):
        self.assertFalse(self.worker_log.exists())
        self.assertFalse(any((self.root / 'claims').glob('*/owner.json')))
        self.assertFalse(any((self.root / 'gpu_claims').glob('gpu-*')))
        self.assertFalse(any((self.root / 'runs').glob('*')))

    def test_pilot_memory_threshold_waits_without_claim_then_releases(self):
        self.env['VFCL_EXPERIMENT_PROFILE'] = 'seed42-pilot'
        self.set_memory(default='41943039')
        self.queue('formal', 'dataset:method:42')
        process = self.start_pipeline()
        self.wait_for(lambda: {row[1] for row in self.memory_reads()} >= {'0', '1'}, process)
        self.assert_no_unstarted_work()
        self.assertNotIn('claim --root', self.driver_log.read_text())
        self.set_memory(default='41943040')
        _, stderr = process.communicate(timeout=20)
        self.assertEqual(0, process.returncode, stderr)
        self.assertEqual(1, len(self.worker_log.read_text().splitlines()))
        self.assert_pipeline_cleanup()

    def test_pilot_memory_waiting_worker_exits_when_parent_disappears(self):
        self.env.update(VFCL_EXPERIMENT_PROFILE='seed42-pilot', FAKE_MEMAVAILABLE_KB='41943039')
        self.queue('formal', 'dataset:method:42')
        process = self.start_pipeline()
        self.wait_for(lambda: {row[1] for row in self.memory_reads()} >= {'0', '1'}, process)
        identities = {(int(row[0]), self.process_identity(int(row[0]))[0])
                      for row in self.memory_reads()}
        auditor, auditor_start = self.logged_auditor_identity()
        process.kill()
        process.wait(timeout=5)
        self.signal_exact_process(auditor, auditor_start, signal.SIGTERM)
        process.communicate(timeout=5)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if all(self.process_identity(pid)[0] != start or
                   self.process_identity(pid)[1] == 'Z' for pid, start in identities):
                break
            time.sleep(.02)
        self.assertTrue(all(self.process_identity(pid)[0] != start or
                            self.process_identity(pid)[1] == 'Z' for pid, start in identities))
        self.assert_no_unstarted_work()

    def test_pilot_memory_wait_does_not_block_other_producer_or_auditor(self):
        self.env['VFCL_EXPERIMENT_PROFILE'] = 'seed42-pilot'
        self.set_memory(default='41943040', **{'0': '41943039'})
        self.queue('formal', 'dataset:method:42')
        process = self.start_pipeline()
        self.wait_for(lambda: any(e['kind'] == 'complete' for e in self.pipeline_events()), process)
        self.assertTrue(any(row[1:] == ['0', '41943039'] for row in self.memory_reads()))
        self.assertFalse(any((self.root / 'gpu_claims').glob('gpu-*')))
        owners = list((self.root / 'claims').glob('*/owner.json'))
        self.assertEqual(['formal-worker-1'], [json.loads(path.read_text())['worker_role'] for path in owners])
        self.assertEqual({'0', '1'}, {row[1] for row in self.memory_reads()})
        self.wait_for(lambda: process.poll() is not None, process, timeout=4)
        _, stderr = process.communicate(timeout=20)
        self.assertEqual(0, process.returncode, stderr)
        self.assert_pipeline_cleanup()

    def test_pilot_memory_recovery_does_not_bypass_dead_parent(self):
        release = self.base / 'memory-ready-release'
        self.env.update(VFCL_EXPERIMENT_PROFILE='seed42-pilot',
                        FAKE_MEMORY_READY_RELEASE=str(release))
        self.set_memory(default='41943039')
        self.queue('formal', 'dataset:method:42')
        process = self.start_pipeline()
        self.wait_for(lambda: {row[1] for row in self.memory_reads()} >= {'0', '1'}, process)
        self.set_memory(default='41943040')
        self.wait_for(lambda: {row[1] for row in self.memory_reads() if row[2] == '41943040'}
                      >= {'0', '1'}, process)
        auditor, auditor_start = self.logged_auditor_identity()
        process.kill()
        process.wait(timeout=5)
        self.signal_exact_process(auditor, auditor_start, signal.SIGTERM)
        release.touch()
        process.communicate(timeout=5)
        self.assertNotIn('claim --root', self.driver_log.read_text())
        self.assert_no_unstarted_work()

    def logged_auditor_identity(self):
        owner_line = next(row for row in self.driver_log.read_text().splitlines()
                          if row.startswith('owner ') and '--worker-role formal-worker-2 ' in row)
        pid = int(owner_line.split('--pid ')[1].split()[0])
        return pid, self.process_identity(pid)[0]

    def test_pilot_parent_death_during_memory_recheck_never_prepares_run(self):
        release = self.base / 'memory-ready-release'
        self.env.update(VFCL_EXPERIMENT_PROFILE='seed42-pilot',
                        FAKE_MEMORY_READY_RELEASE=str(release),
                        FAKE_MEMORY_BLOCK_READ_NUMBER='2',
                        FAKE_REQUIRED_CLAIM_WORKER='formal-worker-0')
        self.queue('formal', 'dataset:method:42')
        process = self.start_pipeline()
        self.wait_for(lambda: sum(row[1] == '0' for row in self.memory_reads()) >= 2, process)
        auditor, auditor_start = self.logged_auditor_identity()
        process.kill()
        process.wait(timeout=5)
        self.signal_exact_process(auditor, auditor_start, signal.SIGTERM)
        release.touch()
        process.communicate(timeout=10)
        self.assertNotIn('prepare-run ', self.driver_log.read_text())
        self.assert_no_unstarted_work()

    def test_pilot_parent_start_identity_mismatch_never_claims(self):
        self.env.update(VFCL_EXPERIMENT_PROFILE='seed42-pilot', FAKE_PARENT_START_CORRUPT='1')
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertNotIn('claim --root', self.driver_log.read_text())
        self.assert_no_unstarted_work()

    def test_pilot_low_memory_already_drained_exits_without_claims(self):
        self.env.update(VFCL_EXPERIMENT_PROFILE='seed42-pilot', FAKE_MEMAVAILABLE_KB='41943039')
        process = self.start_pipeline()
        self.wait_for(lambda: process.poll() is not None, process, timeout=4)
        _, stderr = process.communicate(timeout=5)
        self.assertEqual(0, process.returncode, stderr)
        self.assertNotIn('claim --root', self.driver_log.read_text())
        self.assert_no_unstarted_work()
        self.assertTrue((self.root / 'PILOT_EXECUTION_SUCCESS').is_file())
        self.assert_pipeline_cleanup()

    def test_pilot_memory_drop_releases_only_unstarted_owned_claims(self):
        self.env.update(VFCL_EXPERIMENT_PROFILE='seed42-pilot',
                        FAKE_REQUIRED_CLAIM_WORKER='formal-worker-0',
                        FAKE_MEMORY_DROP_AFTER_GPU=str(self.base / 'memory-dropped'))
        self.set_memory(default='41943040')
        self.queue('formal', 'dataset:method:42')
        process = self.start_pipeline()
        self.wait_for(lambda: any(e['kind'] == 'release-claim' for e in self.pipeline_events()), process)
        self.assert_no_unstarted_work()
        self.assertNotIn('prepare-run ', self.driver_log.read_text())
        self.set_memory(default='41943040')
        _, stderr = process.communicate(timeout=20)
        self.assertEqual(0, process.returncode, stderr)
        self.assertEqual(1, len(self.worker_log.read_text().splitlines()))
        self.assert_pipeline_cleanup()

    def test_pilot_fifteen_jobs_skip_explanation_and_mark_only_after_tables(self):
        self.env['VFCL_EXPERIMENT_PROFILE'] = 'seed42-pilot'
        self.queue('formal', *(f'dataset:method-{i}:42' for i in range(15)))
        completed = self.run_launcher(self.root, timeout=40)
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual(15, len(list((self.root / 'records').glob('*.json'))))
        log = self.driver_log.read_text()
        self.assertNotIn('--phase explanation', log)
        self.assertTrue((self.root / 'tables' / 'PILOT_TABLE.csv').is_file())
        for name in ('PILOT_PHASE_SUCCESS', 'PILOT_EXECUTION_SUCCESS'):
            self.assertTrue((self.root / name).is_file(), name)
            self.assertLess(log.index('finalize '), log.index('--name ' + name))
        for name in ('FORMAL_PHASE_SUCCESS', 'EXPLANATION_PHASE_SUCCESS', 'FORMAL_EXECUTION_SUCCESS'):
            self.assertFalse((self.root / name).exists(), name)
        self.assert_pipeline_cleanup()

    def recovery_environment(self):
        self.env.update(VFCL_EXPERIMENT_PROFILE='seed42-adaptive-recovery',
                        FAKE_GIT_BRANCH=RECOVERY_BRANCH)

    def full_matrix_environment(self, missing=()):
        self.env.update(VFCL_EXPERIMENT_PROFILE='full-public-matrix',
                        FAKE_GIT_BRANCH=FULL_MATRIX_BRANCH)
        jobs = [f'{dataset}:{method}:{seed}' for dataset in
                ('cifar100', 'isolet', 'upmc_food101') for method in
                ('finetune', 'lwf', 'ewc', 'er', 'der_pp', 'er_ace', 'gpm',
                 'fedprotip_vfl', 'target', 'afc', 'lwf_wa', 'adagauss',
                 'proto_fedspace', 'adaptive') for seed in (42, 43, 44)]
        self.assertTrue(set(missing) <= set(jobs))
        (self.root / 'FORMAL_PLAN.json').write_text(json.dumps({
            'formal_jobs': jobs, 'explanation_jobs': [], 'missing_jobs': list(missing)}))
        (self.root / 'FULL_MATRIX_REUSE.json').write_text(json.dumps(
            [key for key in jobs if key not in missing]))
        self.queue('formal', *missing)
        self.queue('explanation')

    def test_single_dataset_42_jobs_finalize_without_explanations(self):
        dataset = 'isolet'
        jobs = [f'{dataset}:{method}:{seed}' for method in
                ('finetune', 'lwf', 'ewc', 'er', 'der_pp', 'er_ace', 'gpm',
                 'fedprotip_vfl', 'target', 'afc', 'lwf_wa', 'adagauss',
                 'proto_fedspace', 'adaptive') for seed in (42, 43, 44)]
        self.env.update(VFCL_EXPERIMENT_PROFILE='single-dataset-full-matrix',
                        VFCL_FORMAL_DATASET=dataset,
                        FAKE_GIT_BRANCH=DATASET_BRANCH,
                        FAKE_GPU_ROWS='0, GPU-0, 8000, 0',
                        VFCL_GPU_COUNT='1')
        (self.root / 'FORMAL_PLAN.json').write_text(json.dumps({
            'formal_jobs': jobs, 'explanation_jobs': [], 'missing_jobs': jobs}))
        self.queue('formal', *jobs)
        self.queue('explanation')
        completed = self.run_launcher(self.root, timeout=120)
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual(42, len(list((self.root / 'records').glob('*.json'))))
        self.assertFalse((self.root / 'FORMAL_EXECUTION_SUCCESS').exists())
        self.assertTrue((self.root / 'DATASET_PHASE_SUCCESS').is_file())
        self.assertTrue((self.root / 'DATASET_EXECUTION_SUCCESS').is_file())
        self.assertTrue((self.root / 'tables' / 'FORMAL_TABLE.csv').is_file())
        self.assertNotIn('--phase explanation', self.driver_log.read_text())
        calls = Path(self.env['FAKE_RETENTION_LOG']).read_text().splitlines()
        self.assertEqual(84, len(calls))
        for index, call in enumerate(calls):
            self.assertEqual('--terminate-blocking-sftp', call.split()[-2])
            self.assertEqual('--dry-run' if index % 2 == 0 else '--apply',
                             call.split()[-1])
        self.assert_pipeline_cleanup()

    def test_continuation_only_runs_twenty_four_and_marks_separately(self):
        methods = ('finetune', 'lwf', 'ewc', 'er', 'der_pp', 'er_ace', 'gpm',
                   'fedprotip_vfl', 'target', 'afc', 'lwf_wa', 'adagauss',
                   'proto_fedspace', 'adaptive')
        jobs = [f'cifar100:{method}:{seed}' for method in methods
                for seed in (42, 43, 44)]
        missing = jobs[18:]
        self.env.update(
            VFCL_EXPERIMENT_PROFILE='single-dataset-verified-continuation-v1',
            VFCL_FORMAL_DATASET='cifar100',
            FAKE_GIT_BRANCH=CONTINUATION_BRANCH, VFCL_GPU_COUNT='2')
        (self.root / 'FORMAL_PLAN.json').write_text(json.dumps({
            'formal_jobs': jobs, 'explanation_jobs': [],
            'missing_jobs': missing}))
        self.queue('formal', *missing)
        self.queue('explanation')
        completed = self.run_launcher(self.root, timeout=120)
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual(24, len(list((self.root / 'records').glob('*.json'))))
        self.assertEqual(42, next(event['rows'] for event in
                                  self.pipeline_events()
                                  if event['kind'] == 'finalize'))
        self.assertTrue((self.root / 'tables' /
                         'CIFAR_CONTINUATION_TABLE.csv').is_file())
        self.assertTrue((self.root /
                         'DATASET_CONTINUATION_PHASE_SUCCESS').is_file())
        self.assertTrue((self.root /
                         'DATASET_CONTINUATION_SUCCESS').is_file())
        self.assertFalse((self.root / 'DATASET_EXECUTION_SUCCESS').exists())
        self.assertEqual(24, len([event for event in self.pipeline_events()
                                  if event['kind'] == 'claim']))
        self.assertIn('--requested-slots 2', self.driver_log.read_text())
        calls = Path(self.env['FAKE_RETENTION_LOG']).read_text().splitlines()
        self.assertEqual(48, len(calls))
        self.assertTrue(all('--terminate-blocking-sftp' in call
                            for call in calls))
        self.assert_pipeline_cleanup()

    def test_continuation_disk_gate_rejects_before_claim(self):
        missing = [f'cifar100:{method}:{seed}' for method in (
            'gpm', 'fedprotip_vfl', 'target', 'afc', 'lwf_wa', 'adagauss',
            'proto_fedspace', 'adaptive') for seed in (42, 43, 44)]
        self.env.update(
            VFCL_EXPERIMENT_PROFILE='single-dataset-verified-continuation-v1',
            VFCL_FORMAL_DATASET='cifar100',
            FAKE_GIT_BRANCH=CONTINUATION_BRANCH, VFCL_GPU_COUNT='2',
            FAKE_DISK_STATUS_OUTPUT='{"safe":false}')
        self.queue('formal', *missing)
        self.queue('explanation')
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertIn('insufficient single-dataset disk capacity',
                      completed.stderr)
        self.assertIn('--requested-slots 2', self.driver_log.read_text())
        self.assertNotIn('claim --root', self.driver_log.read_text())

    def test_single_dataset_two_gpu_disk_gate_rejects_before_claim(self):
        jobs = [f'cifar100:{method}:{seed}' for method in
                ('finetune', 'lwf', 'ewc', 'er', 'der_pp', 'er_ace', 'gpm',
                 'fedprotip_vfl', 'target', 'afc', 'lwf_wa', 'adagauss',
                 'proto_fedspace', 'adaptive') for seed in (42, 43, 44)]
        self.env.update(VFCL_EXPERIMENT_PROFILE='single-dataset-full-matrix',
                        VFCL_FORMAL_DATASET='cifar100',
                        FAKE_GIT_BRANCH=DATASET_BRANCH, VFCL_GPU_COUNT='2',
                        FAKE_DISK_STATUS_OUTPUT='{"safe":false}')
        (self.root / 'FORMAL_PLAN.json').write_text(json.dumps({
            'formal_jobs': jobs, 'explanation_jobs': [], 'missing_jobs': jobs}))
        self.queue('formal', *jobs)
        self.queue('explanation')
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertIn('insufficient single-dataset disk capacity', completed.stderr)
        self.assertIn('--requested-slots 2', self.driver_log.read_text())
        self.assertNotIn('claim --root', self.driver_log.read_text())

    def test_single_dataset_two_gpu_workers_overlap_on_distinct_devices(self):
        jobs = [f'cifar100:{method}:{seed}' for method in
                ('finetune', 'lwf', 'ewc', 'er', 'der_pp', 'er_ace', 'gpm',
                 'fedprotip_vfl', 'target', 'afc', 'lwf_wa', 'adagauss',
                 'proto_fedspace', 'adaptive') for seed in (42, 43, 44)]
        self.env.update(VFCL_EXPERIMENT_PROFILE='single-dataset-full-matrix',
                        VFCL_FORMAL_DATASET='cifar100',
                        FAKE_GIT_BRANCH=DATASET_BRANCH, VFCL_GPU_COUNT='2',
                        FAKE_WORKER_DELAY='1')
        Path(self.env['FAKE_GPU_LOCK_ROOT']).mkdir()
        (self.root / 'FORMAL_PLAN.json').write_text(json.dumps({
            'formal_jobs': jobs, 'explanation_jobs': [], 'missing_jobs': jobs}))
        self.queue('formal', *jobs)
        self.queue('explanation')
        completed = self.run_launcher(self.root, timeout=180)
        self.assertEqual(0, completed.returncode, completed.stderr)
        events = self.pipeline_events()
        first_record = next(i for i, event in enumerate(events)
                            if event['kind'] == 'record-installed')
        self.assertGreaterEqual(sum(event['kind'] == 'training-start'
                                    for event in events[:first_record]), 2)
        self.assertEqual({'0', '1'}, {row.split('|')[1] for row in
                                     self.worker_log.read_text().splitlines()})
        self.assertFalse(Path(self.env['FAKE_GPU_OVERLAP_LOG']).exists())
        self.assertTrue((self.root / 'DATASET_EXECUTION_SUCCESS').exists())

    def test_single_dataset_gpu_count_is_frozen_after_preflight(self):
        jobs = [f'cifar100:{method}:{seed}' for method in
                ('finetune', 'lwf', 'ewc', 'er', 'der_pp', 'er_ace', 'gpm',
                 'fedprotip_vfl', 'target', 'afc', 'lwf_wa', 'adagauss',
                 'proto_fedspace', 'adaptive') for seed in (42, 43, 44)]
        (self.root / 'FORMAL_PLAN.json').write_text(json.dumps({
            'formal_jobs': jobs, 'explanation_jobs': [], 'missing_jobs': jobs}))
        self.queue('formal', *jobs)
        self.queue('explanation')
        library = self.worktree / 'launcher-library.sh'
        library.write_text(self.launcher.read_text().rsplit('\nmain "$@"', 1)[0])
        completed = subprocess.run(
            ['bash', '-c', 'source "$1"; preflight check "$2" || exit; '
             'export VFCL_GPU_COUNT=1; check_frozen_state',
             'test', str(library), str(self.root)],
            env={**self.env, 'VFCL_EXPERIMENT_PROFILE': 'single-dataset-full-matrix',
                 'VFCL_FORMAL_DATASET': 'cifar100',
                 'FAKE_GIT_BRANCH': DATASET_BRANCH, 'VFCL_GPU_COUNT': '2'},
            capture_output=True, text=True)
        self.assertNotEqual(0, completed.returncode)
        self.assertIn('GPU count changed', completed.stderr)

    def test_full_matrix_review_branch_and_exact_plan(self):
        self.full_matrix_environment()
        completed = self.run_launcher('--check', self.root)
        self.assertEqual(0, completed.returncode, completed.stderr)
        for branch in (RECOVERY_BRANCH, BRANCH, 'codex/full-public-matrix',
                       'codex/arbitrary', FULL_MATRIX_BRANCH + '-ad-hoc', ''):
            with self.subTest(branch=branch):
                completed = self.run_launcher('--check', self.root,
                    env={**self.env, 'FAKE_GIT_BRANCH': branch})
                self.assertNotEqual(0, completed.returncode)
                self.assertIn('implementation branch differs', completed.stderr)
        plan = json.loads((self.root / 'FORMAL_PLAN.json').read_text())
        for change in ({'formal_jobs': plan['formal_jobs'][:-1]},
                       {'explanation_jobs': ['isolet:adaptive:42']}):
            (self.root / 'FORMAL_PLAN.json').write_text(json.dumps({**plan, **change}))
            self.assertNotEqual(0, self.run_launcher('--check', self.root).returncode)

    def test_full_matrix_reuse_never_claimed_and_success_requires_merged_drain(self):
        jobs = ('cifar100:finetune:42', 'isolet:adaptive:43')
        self.full_matrix_environment(jobs)
        completed = self.run_launcher(self.root)
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual(set(jobs), {e['key'] for e in self.pipeline_events() if e['kind'] == 'claim'})
        self.assertEqual({driver.safe_spec_name(driver.spec_for_key(k)) for k in jobs},
                         {p.name for p in (self.root / 'runs').iterdir()})
        log = self.driver_log.read_text()
        self.assertNotIn('--phase explanation', log)
        self.assertTrue((self.root / 'tables/FULL_MATRIX_TABLE.csv').is_file())
        self.assertEqual([126], [e['rows'] for e in self.pipeline_events() if e['kind'] == 'finalize'])
        for name in ('FULL_MATRIX_PHASE_SUCCESS', 'FULL_MATRIX_EXECUTION_SUCCESS'):
            self.assertTrue((self.root / name).is_file())
            self.assertLess(log.index('finalize '), log.index('--name ' + name))
        self.assertFalse((self.root / 'FORMAL_EXECUTION_SUCCESS').exists())
        self.assert_pipeline_cleanup()

    def test_full_matrix_existing_success_marker_rejects_check(self):
        self.full_matrix_environment()
        for name in ('FULL_MATRIX_PHASE_SUCCESS', 'FULL_MATRIX_EXECUTION_SUCCESS'):
            (self.root / name).write_text('{}\n')
            completed = self.run_launcher('--check', self.root)
            self.assertNotEqual(0, completed.returncode)
            (self.root / name).unlink()

    def test_full_matrix_all_reused_drains_without_gpu_even_under_disk_pressure(self):
        self.full_matrix_environment()
        slots = self.base / 'disk-slots'
        slots.write_text('0')
        self.env.update(FAKE_DISK_SLOTS_FILE=str(slots), FAKE_MEMAVAILABLE_KB='0')
        completed = self.run_launcher(self.root)
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertFalse((self.root / 'runs').exists())
        self.assertFalse(Path(self.env['FAKE_GPU_QUERY_LOG']).exists())
        self.assertTrue((self.root / 'FULL_MATRIX_EXECUTION_SUCCESS').exists())
        self.assert_pipeline_cleanup()

    def test_full_matrix_incomplete_merged_rows_never_install_success(self):
        self.full_matrix_environment()
        reuse_path = self.root / 'FULL_MATRIX_REUSE.json'
        reuse_path.write_text(json.dumps(json.loads(reuse_path.read_text())[:-1]))
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertTrue((self.root / 'FORMAL_STOPPED').exists())
        self.assertFalse((self.root / 'FULL_MATRIX_PHASE_SUCCESS').exists())
        self.assertFalse((self.root / 'FULL_MATRIX_EXECUTION_SUCCESS').exists())
        self.assert_pipeline_cleanup()

    def test_full_matrix_either_gpu_excludes_any_compute_and_nonidle(self):
        job = 'isolet:finetune:42'
        self.full_matrix_environment((job,))
        # Producer 0 must use GPU 1 when GPU 0 hosts a non-Python application.
        self.env.update(FAKE_REQUIRED_CLAIM_WORKER='formal-worker-0',
                        FAKE_COMPUTE_ROWS='GPU-0, 900001, external-renderer',
                        FAKE_GPU_ROWS='0, GPU-0, 9000, 0\\n1, GPU-1, 9000, 0')
        completed = self.run_launcher(self.root)
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual(['1'], [r.split('|')[1] for r in self.worker_log.read_text().splitlines()])

    def test_full_matrix_nonidle_gpu_falls_back_to_zero(self):
        self.full_matrix_environment(('isolet:finetune:42',))
        self.env.update(FAKE_REQUIRED_CLAIM_WORKER='formal-worker-1',
                        FAKE_GPU_ROWS='0, GPU-0, 9000, 0\\n1, GPU-1, 9000, 1')
        completed = self.run_launcher(self.root)
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual(['0'], [r.split('|')[1] for r in self.worker_log.read_text().splitlines()])

    def test_full_matrix_disk_status_error_fails_before_claim_without_retry(self):
        for output in ('not-json', '{}', '{"safe":1}', '{"safe":"false"}'):
            with self.subTest(output=output):
                self.full_matrix_environment(('isolet:finetune:42',))
                self.env['FAKE_DISK_STATUS_OUTPUT'] = output
                completed = self.run_launcher(self.root)
                self.assertNotEqual(0, completed.returncode)
                self.assertTrue((self.root / 'FAILED_JOB').is_file())
                self.assertFalse(self.worker_log.exists())
                self.assertNotIn('claim --root', self.driver_log.read_text())
                for name in ('FAILED_JOB', 'FORMAL_STOPPED'):
                    (self.root / name).unlink()
        self.env.pop('FAKE_DISK_STATUS_OUTPUT')
        self.env['FAKE_DISK_STATUS_EXIT'] = '1'
        self.assertNotEqual(0, self.run_launcher(self.root).returncode)
        self.assertTrue((self.root / 'FAILED_JOB').exists())
        self.assertNotIn('claim --root', self.driver_log.read_text())

    def test_full_matrix_disk_pressure_waits_zero_then_one_then_two(self):
        jobs = ('isolet:finetune:42', 'isolet:finetune:43', 'isolet:finetune:44')
        self.full_matrix_environment(jobs)
        slots = self.base / 'disk-slots'
        slots.write_text('0')
        self.env.update(FAKE_DISK_SLOTS_FILE=str(slots), FAKE_AUDIT_BLOCK=jobs[0])
        process = self.start_pipeline()
        self.wait_for(lambda: any(e['kind'] == 'disk-status' for e in self.pipeline_events()), process)
        self.assertFalse(self.worker_log.exists())
        self.assertFalse((self.root / 'FAILED_JOB').exists())
        slots.write_text('1')
        self.wait_for(Path(self.env['FAKE_AUDIT_READY']).exists, process)
        self.assertEqual(1, sum(e['kind'] == 'claim' for e in self.pipeline_events()))
        slots.write_text('2')
        self.wait_for(lambda: sum(e['kind'] == 'training-start' for e in self.pipeline_events()) == 2, process)
        slots.write_text('0')
        self.wait_for(lambda: any(e['kind'] == 'disk-status' and not e['safe'] for e in self.pipeline_events()[-3:]), process)
        self.assertIsNone(process.poll())
        self.assertFalse((self.root / 'FAILED_JOB').exists())
        Path(self.env['FAKE_AUDIT_RELEASE']).touch()
        slots.write_text('2')
        _, stderr = process.communicate(timeout=25)
        self.assertEqual(0, process.returncode, stderr)
        self.assert_pipeline_cleanup()

    def test_full_matrix_fake_disk_reads_wait_for_claim_lock(self):
        self.full_matrix_environment(('isolet:finetune:42',))
        slots = self.base / 'disk-slots'
        slots.write_text('1')
        self.env['FAKE_DISK_SLOTS_FILE'] = str(slots)
        queue = self.root / 'FAKE_FORMAL_QUEUE'
        owner = {'pid': os.getpid(), 'worker_role': 'formal-worker-1'}
        for action, options in (('disk-status', ['--requested-slots', '1']),
                                ('claim', ['--phase', 'formal', '--pipeline',
                                           '--owner-json', json.dumps(owner)])):
            with self.subTest(action=action), queue.open('r+') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                slots.unlink()
                process = subprocess.Popen([sys.executable,
                    str(self.worktree / 'three_dataset_formal_driver.py'), action,
                    '--root', str(self.root), *options], env=self.env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    with self.assertRaises(subprocess.TimeoutExpired,
                                           msg='disk state read outside claims lock'):
                        process.communicate(timeout=1)
                finally:
                    slots.write_text('0')
                    fcntl.flock(stream, fcntl.LOCK_UN)
                    stdout, stderr = process.communicate(timeout=5)
                self.assertEqual(0 if action == 'disk-status' else 75,
                                 process.returncode, stderr)
                if action == 'disk-status':
                    self.assertIs(json.loads(stdout)['safe'], False)
        self.assertEqual([], list((self.root / 'claims').glob('*')))

    def test_full_matrix_last_handoff_keeps_producers_alive_until_retention_drains(self):
        jobs = ('isolet:finetune:42', 'isolet:finetune:43')
        self.full_matrix_environment(jobs)
        self.env.update(FAKE_AUDIT_BLOCK=jobs[0], FAKE_DISK_REQUIRE_LIVE_OWNER='1')
        process = self.start_pipeline()
        self.wait_for(Path(self.env['FAKE_AUDIT_READY']).exists, process)
        self.wait_for(lambda: sum(e['kind'] == 'claim-empty' for e in self.pipeline_events()) >= 4, process)
        for claim in (self.root / 'claims').iterdir():
            owner = json.loads((claim / 'owner.json').read_text())
            start, state = self.process_identity(owner['pid'])
            self.assertEqual(owner['process_start_time'], start)
            self.assertNotEqual('Z', state)
        self.assertFalse((self.root / 'FAILED_JOB').exists())
        Path(self.env['FAKE_AUDIT_RELEASE']).touch()
        _, stderr = process.communicate(timeout=25)
        self.assertEqual(0, process.returncode, stderr)
        self.assert_pipeline_cleanup()

    def test_full_matrix_cpu_audit_overlaps_next_training_and_retention_handoff(self):
        jobs = ('isolet:finetune:42', 'isolet:finetune:43', 'isolet:finetune:44', 'isolet:adaptive:42')
        self.full_matrix_environment(jobs)
        Path(self.env['FAKE_GPU_LOCK_ROOT']).mkdir()
        self.env.update(FAKE_AUDIT_BLOCK=jobs[0], FAKE_WAIT_FIRST_AUDIT='1')
        process = self.start_pipeline()
        self.wait_for(Path(self.env['FAKE_AUDIT_READY']).exists, process)
        self.wait_for(lambda: sum(e['kind'] == 'queue' for e in self.pipeline_events()) == 3, process)
        events = self.pipeline_events()
        audit_index = next(i for i, e in enumerate(events) if e['kind'] == 'audit-start')
        self.assertTrue(any(e['kind'] == 'training-start' for e in events[audit_index + 1:]))
        self.assertFalse((self.root / 'FULL_MATRIX_EXECUTION_SUCCESS').exists())
        Path(self.env['FAKE_AUDIT_RELEASE']).touch()
        _, stderr = process.communicate(timeout=25)
        self.assertEqual(0, process.returncode, stderr + '\n' + self.driver_log.read_text())
        self.assertFalse(Path(self.env['FAKE_GPU_OVERLAP_LOG']).exists())
        self.assertEqual({'0', '1'}, {r.split('|')[1] for r in self.worker_log.read_text().splitlines()})
        events = self.pipeline_events()
        for key in jobs:
            stages = [e['kind'] for e in events if e.get('key') == key and e['kind'] in
                      ('record-installed', 'retention-dry-run', 'retention-apply', 'complete')]
            self.assertEqual(['record-installed', 'retention-dry-run', 'retention-apply', 'complete'], stages)
        self.assertFalse(any('gpu-claim ' in row and 'formal-worker-2' in row
                             for row in self.driver_log.read_text().splitlines()))
        self.assert_pipeline_cleanup()

    def test_full_matrix_deterministic_failure_blocks_next_claim(self):
        for failure in ('FAKE_FAIL_JOB', 'FAKE_AUDIT_FAIL', 'FAKE_RETENTION_FAIL'):
            with self.subTest(failure=failure):
                jobs = ('isolet:finetune:42', 'isolet:finetune:43')
                self.full_matrix_environment(jobs)
                slots = self.base / 'disk-slots'
                slots.write_text('1')
                self.env.update(FAKE_DISK_SLOTS_FILE=str(slots))
                self.env[failure] = 'dry-run' if failure == 'FAKE_RETENTION_FAIL' else jobs[0]
                completed = self.run_launcher(self.root)
                self.assertNotEqual(0, completed.returncode)
                claimed = [e['key'] for e in self.pipeline_events() if e['kind'] == 'claim']
                self.assertEqual([jobs[0]], claimed)
                self.assertFalse((self.root / 'FULL_MATRIX_EXECUTION_SUCCESS').exists())
                self.assertTrue((self.root / 'FAILED_JOB').is_file())
                self.env.pop(failure)
                # Each subcase owns a fresh disposable fixture, not a retry.
                self.doCleanups()
                self.setUp()

    def test_recovery_branch_mapping_is_exact(self):
        self.recovery_environment()
        completed = self.run_launcher('--check', self.root)
        self.assertEqual(0, completed.returncode, completed.stderr)
        for profile, branch in (('seed42-adaptive-recovery', 'codex/seed42-adaptive-recovery'),
                                ('seed42-adaptive-recovery', BRANCH),
                                ('seed42-adaptive-recovery', 'codex/arbitrary'),
                                ('formal', RECOVERY_BRANCH),
                                ('seed42-pilot', RECOVERY_BRANCH)):
            completed = self.run_launcher('--check', self.root, env={**self.env,
                'VFCL_EXPERIMENT_PROFILE': profile, 'FAKE_GIT_BRANCH': branch})
            self.assertNotEqual(0, completed.returncode)
            self.assertIn('branch differs', completed.stderr)

    def test_recovery_serializes_training_audit_and_record_admission(self):
        self.recovery_environment()
        jobs = ['cifar100:adaptive:42', 'isolet:adaptive:42', 'upmc_food101:adaptive:42']
        self.queue('formal', *jobs)
        self.env.update(FAKE_AUDIT_BLOCK=jobs[0], FAKE_RECORD_BLOCK=jobs[0],
                        FAKE_RECORD_READY=str(self.base / 'record-ready'),
                        FAKE_RECORD_RELEASE=str(self.base / 'record-release'))
        process = self.start_pipeline()
        self.wait_for(Path(self.env['FAKE_AUDIT_READY']).exists, process)
        for release, ready in ((self.env['FAKE_AUDIT_RELEASE'], self.env['FAKE_RECORD_READY']),
                               (self.env['FAKE_RECORD_RELEASE'], None)):
            events = self.pipeline_events()
            self.assertEqual([jobs[0]], [e['key'] for e in events if e['kind'] == 'claim'])
            self.assertEqual([], list((self.root / 'gpu_claims').glob('gpu-*')))
            self.assertFalse((self.root / 'RECOVERY_PHASE_SUCCESS').exists())
            Path(release).touch()
            if ready:
                self.wait_for(Path(ready).exists, process)
        _, stderr = process.communicate(timeout=25)
        self.assertEqual(0, process.returncode, stderr)
        events = self.pipeline_events()
        active = None
        for event in events:
            if event['kind'] == 'claim':
                self.assertIsNone(active)
                active = event['key']
            elif event['kind'] == 'complete':
                self.assertEqual(active, event['key'])
                active = None
        self.assertIsNone(active)
        self.assertEqual(3, len(list((self.root / 'records').glob('*.json'))))
        calls = self.driver_log.read_text()
        self.assertNotIn('--phase explanation', calls)
        self.assertTrue((self.root / 'tables' / 'RECOVERY_TABLE.csv').is_file())
        for name in ('RECOVERY_PHASE_SUCCESS', 'RECOVERY_EXECUTION_SUCCESS'):
            self.assertTrue((self.root / name).exists())
            self.assertLess(calls.index('finalize '), calls.index('--name ' + name))
        self.assertFalse((self.root / 'PILOT_EXECUTION_SUCCESS').exists())
        self.assertFalse((self.root / 'FORMAL_EXECUTION_SUCCESS').exists())
        self.assert_pipeline_cleanup()

    def test_recovery_retention_precedes_audit_release_and_next_claim(self):
        self.recovery_environment()
        first, second = 'cifar100:adaptive:42', 'isolet:adaptive:42'
        self.queue('formal', first, second)
        completed = self.run_launcher(self.root)
        self.assertEqual(0, completed.returncode, completed.stderr)
        relevant = [(event['kind'], event.get('key')) for event in self.pipeline_events()
                    if event['kind'] in {'record-installed', 'retention-dry-run',
                                         'retention-apply', 'complete', 'claim'}]
        self.assertEqual([
            ('claim', first), ('record-installed', first),
            ('retention-dry-run', first), ('retention-apply', first),
            ('complete', first), ('claim', second),
            ('record-installed', second), ('retention-dry-run', second),
            ('retention-apply', second), ('complete', second),
        ], relevant)
        calls = Path(self.env['FAKE_RETENTION_LOG']).read_text().splitlines()
        self.assertEqual(4, len(calls))
        for index, key in enumerate((first, second)):
            dry = calls[index * 2].split()
            apply = calls[index * 2 + 1].split()
            owner = json.loads(dry[dry.index('--owner-json') + 1])
            self.assertEqual('', owner['job'])
            self.assertEqual('formal-worker-2', owner['worker_role'])
            expected = ['--root', str(self.root), '--worktree', str(self.worktree),
                        '--expected-head', HEAD, '--spec-key', key,
                        '--owner-json', canonical(owner)]
            self.assertEqual(expected + ['--dry-run'], dry)
            self.assertEqual(expected + ['--apply'], apply)
            self.assertNotIn('--terminate-blocking-sftp', dry)
            self.assertNotIn('--terminate-blocking-sftp', apply)
            run = self.root / 'runs' / urllib.parse.quote(key, safe='')
            self.assertTrue((run / 'PRUNE_PLAN.json').is_file())
            self.assertTrue((run / 'PRUNED_EVIDENCE.json').is_file())
            tombstone = run / 'retention-quarantine/event_0_CIL.pt'
            self.assertTrue(tombstone.is_file())
            self.assertEqual(0, tombstone.stat().st_size)
        self.assert_pipeline_cleanup()

    def test_recovery_retention_failure_blocks_next_claim_without_deletion(self):
        for failure in ('dry-run', 'apply'):
            with self.subTest(failure=failure):
                fixture = FakeLauncherTests('runTest')
                fixture.setUp()
                self.addCleanup(fixture.doCleanups)
                fixture.recovery_environment()
                first, second = 'cifar100:adaptive:42', 'isolet:adaptive:42'
                fixture.queue('formal', first, second)
                fixture.env['FAKE_RETENTION_FAIL'] = failure
                completed = fixture.run_launcher(fixture.root)
                self.assertNotEqual(0, completed.returncode)
                marker = json.loads((fixture.root / 'FAILED_JOB').read_text())
                self.assertEqual('failed_retention', marker['kind'])
                mark_calls = [line for line in fixture.driver_log.read_text().splitlines()
                              if line.startswith('mark ') and '--name FAILED_JOB' in line]
                retention_mark = next(index for index, line in enumerate(mark_calls)
                                      if '"kind":"failed_retention"' in line)
                setup_marks = [index for index, line in enumerate(mark_calls)
                               if '"kind":"failed_setup"' in line]
                self.assertTrue(all(retention_mark < index for index in setup_marks), mark_calls)
                active = fixture.root / 'audit_queue/active.json'
                safe_first = urllib.parse.quote(first, safe='')
                queued = fixture.root / 'audit_queue' / (safe_first + '.json')
                record = fixture.root / 'records' / (safe_first + '.json')
                self.assertTrue(active.is_file())
                self.assertTrue(queued.is_file())
                self.assertTrue(record.is_file())
                self.assertEqual(first, json.loads(active.read_text())['spec_key'])
                self.assertEqual(first, json.loads(record.read_text())['spec_key'])
                self.assertEqual([first], [event['key'] for event in fixture.pipeline_events()
                                           if event['kind'] == 'claim'])
                self.assertFalse(any(event['kind'] == 'complete'
                                     for event in fixture.pipeline_events()))
                for name in ('RECOVERY_PHASE_SUCCESS', 'RECOVERY_EXECUTION_SUCCESS'):
                    self.assertFalse((fixture.root / name).exists())
                calls = Path(fixture.env['FAKE_RETENTION_LOG']).read_text().splitlines()
                self.assertEqual(1 if failure == 'dry-run' else 2, len(calls))
                run = fixture.root / 'runs' / safe_first
                if failure == 'apply':
                    self.assertTrue((run / 'PRUNE_PLAN.json').is_file())
                    self.assertTrue((run / 'PRUNED_EVIDENCE.json').is_file())
                    tombstone = run / 'retention-quarantine/event_0_CIL.pt'
                    self.assertTrue(tombstone.is_file())
                    self.assertEqual(0, tombstone.stat().st_size)
                else:
                    self.assertFalse((run / 'PRUNE_PLAN.json').exists())

        for interruption in ('signal', 'parent-death'):
            with self.subTest(interruption=interruption):
                fixture = FakeLauncherTests('runTest')
                fixture.setUp()
                self.addCleanup(fixture.doCleanups)
                fixture.recovery_environment()
                key = 'cifar100:adaptive:42'
                fixture.queue('formal', key)
                fixture.env['FAKE_RETENTION_BLOCK'] = '1'
                process = fixture.start_pipeline()
                fixture.wait_for(Path(fixture.env['FAKE_RETENTION_READY']).exists, process)
                workers = [int(row.split('--pid ')[1].split()[0])
                           for row in fixture.driver_log.read_text().splitlines()
                           if row.startswith('owner ')]
                if interruption == 'signal':
                    process.terminate()
                else:
                    process.kill()
                    Path(fixture.env['FAKE_RETENTION_RELEASE']).touch()
                process.wait(timeout=5)
                process.communicate(timeout=8)
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline and any(
                        fixture.process_identity(pid)[1] not in (None, 'Z') for pid in workers):
                    time.sleep(.02)
                self.assertTrue(all(fixture.process_identity(pid)[1] in (None, 'Z')
                                    for pid in workers))
                run = fixture.root / 'runs' / urllib.parse.quote(key, safe='')
                self.assertTrue((run / 'PRUNE_PLAN.json').is_file())
                self.assertTrue((run / 'PRUNED_EVIDENCE.json').is_file())
                self.assertTrue((run / 'retention-quarantine/event_0_CIL.pt').is_file())
                self.assertTrue((fixture.root / 'audit_queue/active.json').is_file())
                self.assertFalse(any(event['kind'] == 'complete'
                                     for event in fixture.pipeline_events()))
                self.assertFalse((fixture.root / 'RECOVERY_EXECUTION_SUCCESS').exists())

    def test_recovery_retention_failure_marker_passes_real_driver_validation(self):
        self.recovery_environment()
        key = 'cifar100:adaptive:42'
        self.queue('formal', key)
        real_root = self.base / 'real-marker-root'
        real_root.mkdir(mode=0o700)
        self.env.update({
            'FAKE_RETENTION_FAIL': 'dry-run',
            'FAKE_REAL_MARK_DRIVER': str(SCRIPT.with_name(
                'three_dataset_formal_driver.py')),
            'FAKE_REAL_MARK_ROOT': str(real_root),
        })

        completed = self.run_launcher(self.root)

        self.assertNotEqual(0, completed.returncode)
        fake = (self.root / 'FAILED_JOB').read_bytes()
        real = (real_root / 'FAILED_JOB').read_bytes()
        self.assertEqual(fake, real)
        marker = json.loads(real)
        self.assertEqual({
            'kind': 'failed_retention', 'role': 'formal-worker-2',
            'spec_key': key, 'exit_code': 31,
        }, marker)
        self.assertTrue((self.root / 'audit_queue/active.json').is_file())
        self.assertNotIn('complete-audit ', self.driver_log.read_text())

    def test_recovery_signal_windows_preserve_active_handoff(self):
        for stage in ('post-next-audit', 'post-audit-run'):
            for interruption in ('signal', 'parent-death'):
                with self.subTest(stage=stage, interruption=interruption):
                    fixture = FakeLauncherTests('runTest')
                    fixture.setUp()
                    self.addCleanup(fixture.doCleanups)
                    fixture.recovery_environment()
                    key = 'cifar100:adaptive:42'
                    fixture.queue('formal', key)
                    ready = fixture.base / f'{stage}-ready'
                    release = fixture.base / f'{stage}-release'
                    if stage == 'post-next-audit':
                        fixture.env.update({
                            'FAKE_POST_NEXT_AUDIT_BLOCK': '1',
                            'FAKE_POST_NEXT_AUDIT_READY': str(ready),
                            'FAKE_POST_NEXT_AUDIT_RELEASE': str(release),
                        })
                    else:
                        fixture.env.update({
                            'FAKE_POST_AUDIT_RUN_BLOCK': key,
                            'FAKE_POST_AUDIT_RUN_READY': str(ready),
                            'FAKE_POST_AUDIT_RUN_RELEASE': str(release),
                        })
                    process = fixture.start_pipeline()
                    fixture.wait_for(ready.exists, process)
                    workers = [int(row.split('--pid ')[1].split()[0])
                               for row in fixture.driver_log.read_text().splitlines()
                               if row.startswith('owner ')]
                    next_calls = fixture.driver_log.read_text().count('next-audit ')
                    if interruption == 'signal':
                        process.terminate()
                    else:
                        process.kill()
                    release.touch()
                    process.wait(timeout=8)
                    process.communicate(timeout=8)
                    deadline = time.monotonic() + 8
                    while time.monotonic() < deadline and any(
                            fixture.process_identity(pid)[1] not in (None, 'Z')
                            for pid in workers):
                        time.sleep(.02)
                    self.assertTrue(all(
                        fixture.process_identity(pid)[1] in (None, 'Z')
                        for pid in workers))
                    safe_key = urllib.parse.quote(key, safe='')
                    self.assertTrue((fixture.root / 'audit_queue/active.json').is_file())
                    self.assertTrue((fixture.root / 'audit_queue' /
                                     (safe_key + '.json')).is_file())
                    record = fixture.root / 'records' / (safe_key + '.json')
                    self.assertEqual(stage == 'post-audit-run', record.is_file())
                    calls = fixture.driver_log.read_text()
                    self.assertEqual(next_calls, calls.count('next-audit '))
                    self.assertNotIn('complete-audit ', calls)
                    self.assertNotIn('cancel-audit ', calls)
                    self.assertFalse(Path(fixture.env['FAKE_RETENTION_LOG']).exists())

    def test_recovery_retention_control_payload_failure_preserves_active_audit(self):
        self.recovery_environment()
        key = 'cifar100:adaptive:42'
        self.queue('formal', key)
        reviewed_python = self.base / 'reviewed-python'
        self._write_executable(reviewed_python, r'''#!/bin/bash
if [[ "${FAKE_RETENTION_CONTROL_PAYLOAD_FAIL:-}" == 1 &&
      "${1:-}" == -c && "${2:-}" == *formal_retention_active* ]]; then
  shopt -s nullglob
  pending=("$TMPDIR"/formal-worker-gates.*/.retention-active)
  [[ ${#pending[@]} -eq 1 && -f ${pending[0]} ]] || exit 88
  printf 'ready\n' > "$FAKE_RETENTION_CONTROL_PAYLOAD_READY"
  exit 41
fi
exec "$REAL_REVIEWED_PYTHON" "$@"
''')
        source = self.launcher.read_text()
        pinned = "readonly REVIEWED_PYTHON='/home/c3080/YangXiaoXiang/envs/vfcl/bin/python'"
        self.assertEqual(1, source.count(pinned))
        self.launcher.write_text(source.replace(
            pinned, f"readonly REVIEWED_PYTHON='{reviewed_python}'"))
        ready = self.base / 'retention-control-payload-ready'
        self.env.update({
            'VFCL_PYTHON': str(reviewed_python),
            'REAL_REVIEWED_PYTHON': PYTHON,
            'FAKE_RETENTION_CONTROL_PAYLOAD_FAIL': '1',
            'FAKE_RETENTION_CONTROL_PAYLOAD_READY': str(ready),
        })

        completed = self.run_launcher(self.root)

        self.assertNotEqual(0, completed.returncode)
        self.assertTrue(ready.is_file(), completed.stderr)
        marker = json.loads((self.root / 'FAILED_JOB').read_text())
        self.assertEqual('failed_retention', marker['kind'])
        active = self.root / 'audit_queue/active.json'
        self.assertTrue(active.is_file())
        self.assertEqual(key, json.loads(active.read_text())['spec_key'])
        safe_key = urllib.parse.quote(key, safe='')
        self.assertTrue((self.root / 'audit_queue' / (safe_key + '.json')).is_file())
        self.assertFalse((self.root / 'records' / (safe_key + '.json')).exists())
        calls = self.driver_log.read_text()
        self.assertNotIn('audit-run --root', calls)
        self.assertNotIn('complete-audit ', calls)
        self.assertNotIn('cancel-audit ', calls)
        self.assertFalse(Path(self.env['FAKE_RETENTION_LOG']).exists())
        self.assertFalse((self.root / 'RECOVERY_PHASE_SUCCESS').exists())
        self.assertFalse((self.root / 'RECOVERY_EXECUTION_SUCCESS').exists())
        self.assertFalse(any(self.tmpdir.glob('formal-worker-gates.*')))

    def test_recovery_sibling_failure_during_retention_preserves_active_audit(self):
        self.recovery_environment()
        first, second = 'cifar100:adaptive:42', 'isolet:adaptive:42'
        self.queue('formal', first, second)
        producer_failure = self.base / 'producer-failure'
        self.env.update({
            'FAKE_REQUIRED_CLAIM_WORKER': 'formal-worker-0',
            'FAKE_RETENTION_BLOCK': '1',
            'FAKE_CLAIM_FAIL_FILE': str(producer_failure),
            'FAKE_CLAIM_FAIL_WORKER': 'formal-worker-0',
        })
        process = self.start_pipeline()
        self.wait_for(Path(self.env['FAKE_RETENTION_READY']).exists, process)
        (self.root / 'FAILED_JOB').write_text(
            canonical({'kind': 'failed_job', 'spec_key': second}) + '\n')
        producer_failure.touch()
        _, stderr = process.communicate(timeout=20)

        self.assertNotEqual(0, process.returncode)
        marker = json.loads((self.root / 'FAILED_JOB').read_text())
        self.assertEqual('failed_job', marker['kind'])
        safe_first = urllib.parse.quote(first, safe='')
        active = self.root / 'audit_queue' / 'active.json'
        self.assertTrue(active.is_file())
        self.assertEqual(first, json.loads(active.read_text())['spec_key'])
        self.assertTrue((self.root / 'audit_queue' / (safe_first + '.json')).is_file())
        self.assertTrue((self.root / 'records' / (safe_first + '.json')).is_file())
        self.assertEqual([first], [event['key'] for event in self.pipeline_events()
                                   if event['kind'] == 'claim'])
        calls = self.driver_log.read_text()
        self.assertNotIn('complete-audit ', calls)
        self.assertNotIn('cancel-audit ', calls)
        self.assertFalse((self.root / 'RECOVERY_PHASE_SUCCESS').exists())
        self.assertFalse((self.root / 'RECOVERY_EXECUTION_SUCCESS').exists())
        self.assertFalse(any(self.tmpdir.glob('formal-worker-gates.*')))

    def test_recovery_malformed_retention_control_fails_closed(self):
        self.recovery_environment()
        first, second = 'cifar100:adaptive:42', 'isolet:adaptive:42'
        self.queue('formal', first, second)
        producer_failure = self.base / 'producer-failure'
        self.env.update({
            'FAKE_REQUIRED_CLAIM_WORKER': 'formal-worker-0',
            'FAKE_RETENTION_BLOCK': '1',
            'FAKE_CLAIM_FAIL_FILE': str(producer_failure),
            'FAKE_CLAIM_FAIL_WORKER': 'formal-worker-0',
        })
        process = self.start_pipeline()
        self.wait_for(Path(self.env['FAKE_RETENTION_READY']).exists, process)
        gate_roots = list(self.tmpdir.glob('formal-worker-gates.*'))
        self.assertEqual(1, len(gate_roots))
        control = gate_roots[0] / 'retention-active'
        self.wait_for(control.exists, process)
        control.write_text('malformed\n')
        (self.root / 'FAILED_JOB').write_text(
            canonical({'kind': 'failed_job', 'spec_key': second}) + '\n')
        producer_failure.touch()
        _, stderr = process.communicate(timeout=20)

        self.assertNotEqual(0, process.returncode)
        self.assertTrue((self.root / 'audit_queue' / 'active.json').is_file())
        calls = self.driver_log.read_text()
        self.assertNotIn('complete-audit ', calls)
        self.assertNotIn('cancel-audit ', calls)
        self.assertFalse((self.root / 'RECOVERY_EXECUTION_SUCCESS').exists())
        self.assertFalse(any(self.tmpdir.glob('formal-worker-gates.*')))

    def test_nonrecovery_and_smoke_never_invoke_retention(self):
        cases = (
            ('formal', 'formal', 'formal:baseline:42'),
            ('pilot', 'seed42-pilot', 'formal:pilot:42'),
            ('explanation', 'formal', 'explanation:baseline:42'),
        )
        for name, profile, job in cases:
            with self.subTest(name=name):
                fixture = FakeLauncherTests('runTest')
                fixture.setUp()
                self.addCleanup(fixture.doCleanups)
                fixture.env['VFCL_EXPERIMENT_PROFILE'] = profile
                fixture.queue('formal' if name != 'explanation' else 'explanation', job)
                completed = fixture.run_launcher(fixture.root)
                self.assertEqual(0, completed.returncode, completed.stderr)
                self.assertFalse(Path(fixture.env['FAKE_RETENTION_LOG']).exists())
                fixture.assert_pipeline_cleanup()
        self.recovery_environment()
        smoke = self.new_smoke_root('retention-free-smoke')
        completed = self.run_launcher('--smoke', smoke)
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertFalse(Path(self.env['FAKE_RETENTION_LOG']).exists())

    def test_recovery_resource_gate_precedes_check_smoke_and_launch(self):
        self.recovery_environment()
        for mode in ('--check', '--smoke', None):
            for field, value in (('FAKE_MEMAVAILABLE_KB', '41943039'),
                                 ('FAKE_DISK_KB', str(30 * 1024 * 1024 - 1))):
                with self.subTest(mode=mode, field=field):
                    root = self.new_smoke_root(f'gate-{mode}-{field}') if mode == '--smoke' else self.root
                    before = self.snapshot(root)
                    args = [mode, root] if mode else [root]
                    result = self.run_launcher(*args, env={**self.env, field: value})
                    self.assertNotEqual(0, result.returncode)
                    self.assertIn('insufficient', result.stderr)
                    self.assertEqual(before, self.snapshot(root))
                    self.assert_no_unstarted_work()

    def test_recovery_memory_wait_and_parent_identity_hold_no_claim(self):
        self.recovery_environment()
        self.set_memory(default='41943040', **{'0': '41943039', '1': '41943039'})
        self.queue('formal', 'cifar100:adaptive:42')
        process = self.start_pipeline()
        self.wait_for(lambda: {r[1] for r in self.memory_reads()} >= {'0', '1'}, process)
        self.assert_no_unstarted_work()
        self.set_memory(default='41943040')
        _, stderr = process.communicate(timeout=20)
        self.assertEqual(0, process.returncode, stderr)
        self.assert_pipeline_cleanup()

    def test_recovery_parent_identity_mismatch_never_claims(self):
        self.recovery_environment()
        self.env['FAKE_PARENT_START_CORRUPT'] = '1'
        result = self.run_launcher(self.root)
        self.assertNotEqual(0, result.returncode)
        self.assertTrue(self.driver_log.exists(), 'recovery preflight rejected before parent check')
        self.assert_no_unstarted_work()
        self.assert_pipeline_cleanup()

    def test_recovery_waiters_exit_after_parent_exits_without_claim(self):
        self.recovery_environment()
        self.set_memory(default='41943040', **{'0': '41943039', '1': '41943039'})
        self.queue('formal', 'cifar100:adaptive:42')
        process = self.start_pipeline()
        self.wait_for(lambda: {r[1] for r in self.memory_reads()} >= {'0', '1'}, process)
        self.assert_no_unstarted_work()
        workers = [int(row.split('--pid ')[1].split()[0])
                   for row in self.driver_log.read_text().splitlines()
                   if row.startswith('owner ')]
        self.assertEqual(3, len(workers))
        process.kill()  # only this test's fake launcher
        process.wait(timeout=5)
        process.communicate(timeout=8)
        self.assertTrue(all(self.process_identity(pid)[1] in (None, 'Z') for pid in workers))
        self.assert_no_unstarted_work()

    def start_recovery_resource_drop_during_unsafe_gpu_probe(self, resource):
        self.recovery_environment()
        self.queue('formal', 'cifar100:adaptive:42')
        self.set_memory(default='41943040')
        disk = self.base / 'disk-available'
        disk.write_text('31457280\n')
        self.env.update(FAKE_DISK_FILE=str(disk), FAKE_DISK_LOG=str(self.base / 'disk.log'),
                        FAKE_GPU_ROWS='0, GPU-0, 1, 0\\n1, GPU-1, 1, 0',
                        FAKE_GPU_QUERY_DROPPED=str(self.base / 'resource-dropped'))
        target, value = ((self.env['FAKE_MEMAVAILABLE_FILE'], '{"default":"41943039"}')
                         if resource == 'memory' else (str(disk), '31457279'))
        self.env.update(FAKE_GPU_QUERY_RESOURCE_FILE=target, FAKE_GPU_QUERY_RESOURCE_VALUE=value)
        process = self.start_pipeline()
        self.wait_for(lambda: any(e['kind'] == 'release-claim' for e in self.pipeline_events()),
                      process, timeout=5)
        if resource == 'memory':
            released = next(row for row in self.driver_log.read_text().splitlines()
                            if row.startswith('release-claim '))
            owner = json.loads(released.split('--owner-json ', 1)[1])
            worker = owner['worker_role'].rsplit('-', 1)[1]
            self.wait_for(lambda: any(r[1:] == [worker, '41943039'] for r in self.memory_reads()),
                          process)
        else:
            log = Path(self.env['FAKE_DISK_LOG'])
            self.wait_for(lambda: log.read_text().splitlines().count('31457279') >= 3, process)
        self.assert_no_unstarted_work()
        self.assertNotIn('prepare-run ', self.driver_log.read_text())
        return process

    def test_recovery_unsafe_gpu_releases_claim_and_reenters_disk_gate(self):
        process = self.start_recovery_resource_drop_during_unsafe_gpu_probe('disk')
        Path(self.env['FAKE_GPU0_LOW_MEMORY_FILE']).touch()  # now only physical GPU 1 is safe
        before = len([e for e in self.pipeline_events() if e['kind'] == 'claim'])
        log = Path(self.env['FAKE_DISK_LOG'])
        reads = len(log.read_text().splitlines())
        self.wait_for(lambda: len(log.read_text().splitlines()) >= reads + 2, process)
        self.assertEqual(before, len([e for e in self.pipeline_events() if e['kind'] == 'claim']))
        self.assert_no_unstarted_work()
        Path(self.env['FAKE_DISK_FILE']).write_text('31457280\n')
        _, stderr = process.communicate(timeout=20)
        self.assertEqual(0, process.returncode, stderr)
        self.assertEqual(1, len(self.worker_log.read_text().splitlines()))
        self.assertEqual('1', self.worker_log.read_text().split('|')[1])
        self.assert_pipeline_cleanup()

    def test_recovery_unsafe_gpu_waiter_exits_on_parent_death_without_claim_or_launch(self):
        process = self.start_recovery_resource_drop_during_unsafe_gpu_probe('memory')
        workers = [int(row.split('--pid ')[1].split()[0])
                   for row in self.driver_log.read_text().splitlines() if row.startswith('owner ')]
        identities = {pid: self.process_identity(pid)[0] for pid in workers}
        process.kill()  # only this fixture's fake parent
        process.wait(timeout=5)
        process.communicate(timeout=3)
        self.assertEqual(3, len(workers))
        for pid, start in identities.items():
            current, state = self.process_identity(pid)
            self.assertTrue(current is None or (current == start and state == 'Z'))
        self.assert_no_unstarted_work()
        self.assertEqual([], self._live_logged_fake_children())
        self.assertEqual([], self._live_logged_dormant_children())
        self.assertEqual(set(), self._new_child_gates())
        self.assertFalse(any(self.tmpdir.glob('formal-worker-gates.*/child-*')))
        for marker in ('FAILED_JOB', 'FORMAL_STOPPED', 'RECOVERY_PHASE_SUCCESS',
                       'RECOVERY_EXECUTION_SUCCESS'):
            self.assertFalse((self.root / marker).exists(), marker)

    def test_recovery_unsafe_gpu_cooperative_stop_cleans_owned_registration_gates(self):
        process = self.start_recovery_resource_drop_during_unsafe_gpu_probe('memory')
        process.terminate()
        process.communicate(timeout=15)
        self.assertEqual(143, process.returncode)
        self.assert_no_unstarted_work()
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertFalse((self.root / 'FAILED_JOB').exists())
        self.assert_pipeline_cleanup()

    def test_recovery_selects_either_only_safe_physical_gpu(self):
        for gpu in (0, 1):
            with self.subTest(gpu=gpu):
                fixture = FakeLauncherTests('runTest')
                fixture.setUp()
                self.addCleanup(fixture.doCleanups)
                fixture.recovery_environment()
                fixture.queue('formal', 'cifar100:adaptive:42')
                fixture.env['FAKE_GPU_ROWS'] = (
                    f'0, GPU-0, {8000 if gpu == 0 else 1}, 0\\n'
                    f'1, GPU-1, {8000 if gpu == 1 else 1}, 0')
                result = fixture.run_launcher(fixture.root)
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertEqual(str(gpu), fixture.worker_log.read_text().split('|')[1])
                fixture.assert_pipeline_cleanup()

    def test_recovery_audit_failure_never_retries_and_cleans_claims(self):
        self.recovery_environment()
        key = 'cifar100:adaptive:42'
        self.queue('formal', key, 'isolet:adaptive:42', 'upmc_food101:adaptive:42')
        self.env['FAKE_AUDIT_FAIL'] = key
        result = self.run_launcher(self.root)
        self.assertNotEqual(0, result.returncode)
        self.assertEqual([key], [e['key'] for e in self.pipeline_events() if e['kind'] == 'training-start'])
        self.assertEqual('failed_audit', json.loads((self.root / 'FAILED_JOB').read_text())['kind'])
        self.assertEqual(1, self.driver_log.read_text().count('--name FAILED_JOB '))
        self.assertFalse((self.root / 'RECOVERY_EXECUTION_SUCCESS').exists())
        self.assert_pipeline_cleanup()

    def test_recovery_pre_retention_exit_marks_failed_setup_and_cleans_worker_state(self):
        self.recovery_environment()
        key = 'cifar100:adaptive:42'
        self.queue('formal', key)
        self.env['FAKE_PREPARE_FAIL'] = key

        completed = self.run_launcher(self.root)

        self.assertNotEqual(0, completed.returncode)
        self.assertNotIn('retention_active: unbound variable', completed.stderr)
        marker = json.loads((self.root / 'FAILED_JOB').read_text())
        self.assertEqual('failed_setup', marker['kind'])
        self.assertFalse(Path(self.env['FAKE_RETENTION_LOG']).exists())
        self.assertEqual([], list((self.root / 'claims').iterdir()))
        self.assert_pipeline_cleanup()

    def test_recovery_generated_smoke_cannot_install_recovery_success(self):
        self.recovery_environment()
        root = self.new_smoke_root()
        result = self.run_launcher('--smoke', root)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertTrue((root / 'SMOKE_EXECUTION_SUCCESS').exists())
        for name in ('RECOVERY_PHASE_SUCCESS', 'RECOVERY_EXECUTION_SUCCESS'):
            self.assertFalse((root / name).exists())
            blocked = self.new_smoke_root(name)
            (blocked / name).write_text('{}')
            result = self.run_launcher('--smoke', blocked)
            self.assertNotEqual(0, result.returncode)
            self.assertIn(name, result.stderr)

    def test_check_rejects_existing_pilot_markers(self):
        for name in ('PILOT_PHASE_SUCCESS', 'PILOT_EXECUTION_SUCCESS'):
            with self.subTest(name=name):
                marker = self.root / name
                marker.write_text('{}\n')
                completed = self.run_launcher('--check', self.root)
                self.assertNotEqual(0, completed.returncode)
                self.assertIn(name, completed.stderr)
                marker.unlink()

    def test_pilot_memory_predicate_fails_closed_on_invalid_or_missing_values(self):
        library = self.worktree / 'launcher-library.sh'
        library.write_text(self.launcher.read_text().rsplit('\nmain "$@"', 1)[0])
        for profile in ('seed42-pilot',
                        'single-dataset-verified-continuation-v1'):
            for available in ('unavailable', '', '-1', 'invalid',
                              '41943039', '41943040'):
                with self.subTest(profile=profile, available=available):
                    completed = subprocess.run(
                        ['bash', '-c', 'source "$1"; pilot_memory_ready',
                         'test', str(library)],
                        env={**self.env, 'VFCL_EXPERIMENT_PROFILE': profile,
                             'FAKE_MEMAVAILABLE_KB': available},
                        capture_output=True, text=True)
                    self.assertEqual(
                        0 if available == '41943040' else 1,
                        completed.returncode, completed.stderr)

    def test_selected_profile_is_part_of_frozen_state(self):
        library = self.worktree / 'launcher-library.sh'
        library.write_text(self.launcher.read_text().rsplit('\nmain "$@"', 1)[0])
        completed = subprocess.run(
            ['bash', '-c', 'source "$1"; preflight check "$2" || exit; '
             'export VFCL_EXPERIMENT_PROFILE=formal; check_frozen_state',
             'test', str(library), str(self.root)],
            env={**self.env, 'VFCL_EXPERIMENT_PROFILE': 'seed42-pilot'},
            capture_output=True, text=True)
        self.assertNotEqual(0, completed.returncode)
        self.assertIn('profile changed', completed.stderr)

    def wait_for(self, predicate, process, timeout=12):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            if process.poll() is not None:
                break
            time.sleep(.02)
        self.assertTrue(predicate(), 'expected pipeline event not observed')

    def start_pipeline(self):
        process = subprocess.Popen(
            ['bash', str(self.launcher), str(self.root)], env=self.env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self._stop_process, process)
        return process

    def assert_pipeline_cleanup(self, *, abrupt_producer=False):
        self.assertFalse((self.root / 'audit_queue' / 'active.json').exists())
        self.assertEqual([], [(str(p), (p / 'owner.json').read_text() if (p / 'owner.json').exists() else 'missing owner')
                              for p in (self.root / 'gpu_claims').glob('gpu-*')],
                         self.driver_log.read_text())
        self.assertEqual([], self._live_logged_fake_children())
        self.assertEqual([], self._live_logged_dormant_children())
        self.assertEqual(set(), self._new_child_gates())
        self.assertFalse(any(self.tmpdir.glob('formal-worker-gates.*')))

        if not abrupt_producer:
            self.assertFalse(any(self.tmpdir.glob('formal-peak.*')))
        self.assertFalse(any(self.tmpdir.glob('formal-command.*')))

    def test_pipeline_blocked_audit_overlaps_training_caps_three_and_drains(self):
        jobs = ['formal:pipeline:42', 'formal:pipeline:43',
                'formal:pipeline:44', 'formal:later:42']
        self.queue('formal', *jobs)
        self.queue('explanation', 'explanation:after-drain:44')
        self.env.update({'FAKE_AUDIT_BLOCK': jobs[0],
                         'FAKE_WAIT_FIRST_AUDIT': '1',
                         'FAKE_REQUIRE_POSTPROCESS_GPU_MATCH': '1'})
        process = self.start_pipeline()
        ready = Path(self.env['FAKE_AUDIT_READY'])
        self.wait_for(ready.exists, process)
        self.wait_for(lambda: any(e['kind'] == 'backpressure' for e in self.pipeline_events()), process)
        finished = Path(self.env['FAKE_FINISHED_LOG'])
        self.wait_for(lambda: finished.exists() and len(finished.read_text().splitlines()) == 3, process)
        self.assertFalse((self.root / 'FORMAL_PHASE_SUCCESS').exists())
        self.assertFalse(any(e.get('key', '').startswith('explanation:') for e in self.pipeline_events()))
        self.assertNotIn(jobs[3], finished.read_text().splitlines())
        held_events = self.pipeline_events()
        audit_index = next(i for i, e in enumerate(held_events) if e['kind'] == 'audit-start')
        self.assertTrue(any(e['kind'] == 'training-start' for e in held_events[audit_index + 1:]))
        Path(self.env['FAKE_AUDIT_RELEASE']).touch()
        _, stderr = process.communicate(timeout=20)
        self.assertEqual(0, process.returncode, stderr)
        events = self.pipeline_events()
        claimed = [e['key'] for e in events if e['kind'] == 'claim']
        audited = [e['key'] for e in events if e['kind'] == 'audit-start']
        self.assertEqual(5, len(claimed))
        self.assertEqual(set(claimed), set(audited))
        self.assertEqual(5, len(audited))
        self.assertLessEqual(max(e['outstanding'] for e in events if 'outstanding' in e), 3)
        active = 0
        for event in events:
            if event['kind'] == 'audit-start':
                active += 1
                self.assertEqual(1, active)
            elif event['kind'] == 'complete':
                active -= 1
        self.assertEqual(0, active)
        owners = [line for line in self.driver_log.read_text().splitlines() if line.startswith('owner ')]
        self.assertEqual(6, len(owners))
        for phase in ('formal', 'explanation'):
            self.assertEqual({f'{phase}-worker-{i}' for i in range(3)},
                             {row.split('--worker-role ')[1].split()[0] for row in owners if f'--phase {phase} ' in row})
        self.assertTrue((self.root / 'FORMAL_EXECUTION_SUCCESS').exists())
        self.assert_pipeline_cleanup()

    def test_pipeline_resource_failure_never_hands_off(self):
        key = 'formal:resource-fail:42'
        self.queue('formal', key)
        self.env['FAKE_RESOURCE_FAIL'] = key
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertNotIn('queue-audit ', self.driver_log.read_text())
        self.assertNotIn('audit-run ', self.driver_log.read_text())
        self.assert_pipeline_cleanup()

    def test_pipeline_three_plus_one_jobs_bind_environment_without_phase_in_spec_key(self):
        jobs = ['dataset:method:42', 'dataset:method:43', 'dataset:method:44']
        explanation = 'dataset:explanation-method:44'
        self.queue('formal', *jobs)
        self.queue('explanation', explanation)
        self.env['FAKE_REQUIRE_POSTPROCESS_GPU_MATCH'] = '1'
        completed = self.run_launcher(self.root)
        self.assertEqual(0, completed.returncode, completed.stderr)
        workers = {row[0]: (row[1], row[5]) for row in
                   (line.split('|') for line in self.worker_log.read_text().splitlines())}
        self.assertEqual(set(jobs + [explanation]), set(workers))
        evidence = [line.split('|') for line in
                    Path(self.env['FAKE_POSTPROCESS_ENV_LOG']).read_text().splitlines()]
        self.assertEqual(8, len(evidence))
        for action, key, gpu, seed in evidence:
            self.assertEqual(workers[key], (gpu, seed))
            self.assertEqual(key.rsplit(':', 1)[-1], seed)
        self.assert_pipeline_cleanup()

    def test_pipeline_queue_failure_preserves_resource_and_stops(self):
        key = 'formal:queue-fail:42'
        self.queue('formal', key)
        self.env['FAKE_QUEUE_FAIL'] = key
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertEqual(1, len(list((self.root / 'runs').glob('*/RESOURCE_EVIDENCE.json'))))
        self.assertNotIn('audit-run ', self.driver_log.read_text())
        self.assert_pipeline_cleanup()

    def test_pipeline_handoff_release_failure_retains_pending_without_audit(self):
        self.queue('formal', 'formal:release-fail:42')
        self.env['FAKE_GPU_RELEASE_FAIL_ONCE_FILE'] = str(self.base / 'release-failed')
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertEqual(1, len(list((self.root / 'audit_queue').glob('*.json'))))
        self.assertNotIn('audit-run ', self.driver_log.read_text())
        self.assert_pipeline_cleanup()

    def test_pipeline_malformed_payload_fails_closed(self):
        self._assert_pipeline_bad_payload('{"physical_gpu":true}')

    def test_pipeline_boolean_seed_payload_fails_closed(self):
        self._assert_pipeline_bad_payload('{"seed":true}')

    def test_pipeline_wrong_spec_payload_fails_closed(self):
        self._assert_pipeline_bad_payload('{"spec_key":"formal:other:42"}')

    def test_pipeline_traversal_run_payload_fails_closed(self):
        self._assert_pipeline_bad_payload(canonical({'run_dir': str(self.root / 'runs' / '..')}))

    def _assert_pipeline_bad_payload(self, payload):
        self.queue('formal', 'formal:payload:42')
        self.env['FAKE_AUDIT_PAYLOAD'] = payload
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertNotIn('audit-run ', self.driver_log.read_text())
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assert_pipeline_cleanup()

    def test_pipeline_exhausted_producers_with_missing_record_fail_promptly(self):
        self.queue('formal')
        self.env['FAKE_FORMAL_TOTAL'] = '1'
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assert_pipeline_cleanup()

    def test_pipeline_audit_failure_stops_active_training(self):
        first, slow = 'formal:audit-fail:42', 'formal:slow:43'
        self.queue('formal', first, slow, 'formal:waiting:44')
        self.env.update({'FAKE_AUDIT_BLOCK': first, 'FAKE_AUDIT_FAIL': first,
                         'FAKE_SLOW_JOB': slow})
        process = self.start_pipeline()
        self.wait_for(Path(self.env['FAKE_AUDIT_READY']).exists, process)
        self.wait_for(Path(self.env['FAKE_READY_LOG']).exists, process)
        Path(self.env['FAKE_AUDIT_RELEASE']).touch()
        _, stderr = process.communicate(timeout=20)
        self.assertNotEqual(0, process.returncode)
        self.assertIn('fake audit rejected', stderr)
        self.assertIn(slow, Path(self.env['FAKE_KILLED_LOG']).read_text())
        self.assertEqual('failed_audit', json.loads((self.root / 'FAILED_JOB').read_text())['kind'])
        self.assertFalse((self.root / 'FORMAL_PHASE_SUCCESS').exists())
        self.assert_pipeline_cleanup()

    def test_pipeline_signal_during_producer_cleanup_finishes_gpu_release(self):
        key = 'formal:cleanup-signal:42'
        self.queue('formal', key)
        self.env.update({'FAKE_QUEUE_FAIL': key, 'FAKE_CLEANUP_MARK_BLOCK': '1'})
        process = self.start_pipeline()
        self.wait_for(Path(self.env['FAKE_AUDIT_READY']).exists, process)
        owner_path = next((self.root / 'gpu_claims').glob('gpu-*/owner.json'))
        owner = json.loads(owner_path.read_text())
        os.killpg(owner['pgid'], signal.SIGTERM)
        Path(self.env['FAKE_AUDIT_RELEASE']).touch()
        process.communicate(timeout=20)
        self.assertNotEqual(0, process.returncode)
        self.assert_pipeline_cleanup()

    def _assert_audit_tree_stopped(self, abrupt_worker_exit):
        first = 'formal:audit-tree:42'
        self.queue('formal', first)
        self.env.update({'FAKE_AUDIT_BLOCK': first, 'FAKE_AUDIT_GRANDCHILD': '1'})
        foreign = subprocess.Popen(['setsid', 'sleep', '120'])
        self.addCleanup(self._stop_owned_process_group, foreign)
        process = self.start_pipeline()
        self.wait_for(Path(self.env['FAKE_AUDIT_READY']).exists, process)
        events = self.pipeline_events()
        audit = next(e for e in events if e['kind'] == 'audit-start')
        child = next(e for e in events if e['kind'] == 'audit-grandchild')
        owner = json.loads((self.root / 'audit-owner.json').read_text())
        self.assertEqual(audit['parent'], owner['pid'])
        if abrupt_worker_exit:
            # The owned audit child remains alive when its registered worker dies.
            os.kill(owner['pid'], signal.SIGKILL)
        else:
            process.send_signal(signal.SIGTERM)
        process.communicate(timeout=20)
        self.assertNotEqual(0, process.returncode)
        self.assertIsNone(foreign.poll())
        for pid in (audit['pid'], child['pid']):
            stat = Path(f'/proc/{pid}/stat')
            self.assertTrue(not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z')
        self.assertEqual(1, len(list((self.root / 'audit_queue').glob('*.json'))))
        self.assertTrue((self.root / 'FORMAL_STOPPED').exists())
        self.assert_pipeline_cleanup()

    def test_pipeline_signal_cleans_owned_audit_child_and_grandchild(self):
        self._assert_audit_tree_stopped(False)

    def test_pipeline_abrupt_auditor_exit_parent_cleans_registered_subtree(self):
        self._assert_audit_tree_stopped(True)

    @staticmethod
    def process_identity(pid):
        try:
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            return fields[19], fields[0]
        except FileNotFoundError:
            return None, None

    @staticmethod
    def signal_exact_process(pid, start, sig=signal.SIGKILL):
        if os.uname().machine != 'x86_64':
            raise RuntimeError('pidfd test helper requires x86_64')
        libc = ctypes.CDLL(None, use_errno=True)
        syscall = libc.syscall
        syscall.restype = ctypes.c_long
        fd = syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
        if fd < 0:
            return
        try:
            if FakeLauncherTests.process_identity(pid)[0] == start:
                syscall(ctypes.c_long(424), ctypes.c_int(fd), ctypes.c_int(sig),
                        ctypes.c_void_p(0), ctypes.c_uint(0))
        finally:
            os.close(fd)

    def test_pipeline_dead_audit_child_cleans_owned_descendants_not_foreign_token(self):
        key = 'formal:dead-audit-child:42'
        self.queue('formal', key)
        child_ready = self.base / 'audit-grandchild-ready'
        child_term = self.base / 'audit-grandchild-term'
        self.env.update({'FAKE_AUDIT_BLOCK': key, 'FAKE_AUDIT_GRANDCHILD': '1',
                         'FAKE_AUDIT_FOREIGN_CHILD': '1',
                         'FAKE_AUDIT_GRANDCHILD_READY': str(child_ready),
                         'FAKE_AUDIT_GRANDCHILD_TERM_MARK': str(child_term)})
        process = self.start_pipeline()
        self.wait_for(Path(self.env['FAKE_AUDIT_READY']).exists, process)
        self.wait_for(child_ready.exists, process)
        events = self.pipeline_events()
        audit = next(e['pid'] for e in events if e['kind'] == 'audit-start')
        owned = next(e['pid'] for e in events if e['kind'] == 'audit-grandchild')
        foreign = next(e['pid'] for e in events if e['kind'] == 'foreign-grandchild')
        owned_start = self.process_identity(owned)[0]
        foreign_start = self.process_identity(foreign)[0]
        self.addCleanup(self.signal_exact_process, owned, owned_start)
        self.addCleanup(self.signal_exact_process, foreign, foreign_start)
        self.signal_exact_process(audit, self.process_identity(audit)[0])
        self.wait_for(child_term.exists, process)
        self.assertIsNone(process.poll())
        controls = list(self.tmpdir.glob('formal-worker-gates.*/child-2'))
        self.assertEqual(1, len(controls))
        self.assertTrue(controls[0].is_file())
        self.wait_for(lambda: process.poll() is not None, process)
        self.assertIn(self.process_identity(owned)[1], (None, 'Z'))
        self.assertEqual((foreign_start, 'S'), self.process_identity(foreign))
        process.communicate(timeout=5)
        self.assertEqual(137, process.returncode)
        self.assert_pipeline_cleanup()

    def test_pipeline_dead_training_child_cleans_owned_descendant(self):
        key = 'formal:dead-training-child:42'
        self.queue('formal', key)
        self.env.update({'FAKE_SLOW_JOB': key, 'FAKE_TRAIN_GRANDCHILD': key})
        process = self.start_pipeline()
        self.wait_for(Path(self.env['FAKE_READY_LOG']).exists, process)
        leader = int(Path(self.env['FAKE_CHILD_PID_LOG']).read_text().split('|')[0])
        child = next(e['pid'] for e in self.pipeline_events() if e['kind'] == 'training-grandchild')
        start = self.process_identity(child)[0]
        self.addCleanup(self.signal_exact_process, child, start)
        self.signal_exact_process(leader, self.process_identity(leader)[0])
        self.wait_for(lambda: process.poll() is not None, process)
        self.assertIn(self.process_identity(child)[1], (None, 'Z'))
        process.communicate(timeout=5)
        self.assertEqual(137, process.returncode)
        self.assert_pipeline_cleanup()

    def test_pipeline_abrupt_producer_exit_parent_releases_exact_owned_gpu(self):
        key = 'formal:producer-death:42'
        self.queue('formal', key)
        self.env['FAKE_SLOW_JOB'] = key
        process = self.start_pipeline()
        self.wait_for(Path(self.env['FAKE_READY_LOG']).exists, process)
        gpu_owner = next((self.root / 'gpu_claims').glob('gpu-*/owner.json'))
        owner = json.loads(gpu_owner.read_text())
        os.kill(owner['pid'], signal.SIGKILL)
        process.communicate(timeout=20)
        self.assertNotEqual(0, process.returncode)
        self.assertTrue((self.root / 'FORMAL_STOPPED').exists())
        self.assert_pipeline_cleanup(abrupt_producer=True)
        residual = list(self.tmpdir.glob('formal-peak.*'))
        self.assertEqual(1, len(residual))
        print(f'CRASH_TEMP_PRESERVED {residual[0]} {residual[0].stat().st_size} bytes')

    def test_pipeline_parent_gpu_cleanup_preserves_changed_owner(self):
        key = 'formal:changed-gpu-owner:42'
        self.queue('formal', key)
        self.env['FAKE_SLOW_JOB'] = key
        process = self.start_pipeline()
        self.wait_for(Path(self.env['FAKE_READY_LOG']).exists, process)
        gpu_owner = next((self.root / 'gpu_claims').glob('gpu-*/owner.json'))
        owner = json.loads(gpu_owner.read_text())
        changed = {**owner, 'launcher_token': 'foreign-token', 'job': 'formal:another-job:43'}
        gpu_owner.write_text(canonical(changed) + '\n')
        os.kill(owner['pid'], signal.SIGKILL)
        process.communicate(timeout=20)
        self.assertNotEqual(0, process.returncode)
        self.assertEqual(changed, json.loads(gpu_owner.read_text()))
        self.assertNotIn('gpu-release --root', self.driver_log.read_text())
        self.assertEqual([], self._live_logged_fake_children())
        self.assertEqual(set(), self._new_child_gates())

    def test_pipeline_missing_resource_after_selection_fails_without_completion(self):
        self.queue('formal', 'formal:vanished:42')
        self.env['FAKE_REMOVE_RESOURCE'] = '1'
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertNotIn('complete-audit ', self.driver_log.read_text())
        self.assert_pipeline_cleanup()

    def test_pipeline_claim_error_is_not_retried_or_given_a_gpu(self):
        self.queue('formal', 'formal:claim-error:42')
        self.env['FAKE_CLAIM_ERROR'] = '23'
        completed = self.run_launcher(self.root)
        self.assertEqual(23, completed.returncode)
        calls = self.driver_log.read_text()
        self.assertNotIn('gpu-claim ', calls)
        self.assertLessEqual(sum(row.startswith('claim ') for row in calls.splitlines()), 2)
        self.assert_pipeline_cleanup()

    def test_formal_workers_bind_pythonhashseed_to_each_command_seed(self):
        self.queue(
            'formal', 'formal:hash-seed:42', 'formal:hash-seed:43',
            'formal:hash-seed:44')
        self.queue('explanation')
        env = self.env.copy()
        env['FAKE_REQUIRE_HASH_SEED_MATCH'] = '1'
        completed = self.run_launcher(self.root, env=env)
        self.assertEqual(0, completed.returncode, completed.stderr)
        rows = [line.split('|') for line in self.worker_log.read_text().splitlines()]
        self.assertEqual(3, len(rows))
        self.assertEqual(
            {'formal:hash-seed:42': '42',
             'formal:hash-seed:43': '43',
             'formal:hash-seed:44': '44'},
            {row[0]: row[5] for row in rows},
        )
        self.assertEqual(3, len(list((self.root / 'records').glob('*.json'))))
        self.assertTrue((self.root / 'FORMAL_EXECUTION_SUCCESS').is_file())
        self.assertFalse(any((self.root / 'gpu_claims').iterdir()))
        self.assertEqual([], self._live_logged_fake_children())
        self.assertEqual([], self._live_logged_dormant_children())
        self.assertEqual(set(), self._new_child_gates())

    def test_formal_postprocess_inherits_claimed_gpu_and_job_seed(self):
        jobs = [
            'formal:postprocess:42', 'formal:postprocess:43',
            'formal:postprocess:44',
        ]
        self.queue('formal', *jobs)
        self.queue('explanation')
        env = self.env.copy()
        env.pop('CUDA_VISIBLE_DEVICES', None)
        env.update({
            'FAKE_REQUIRE_HASH_SEED_MATCH': '1',
            'FAKE_REQUIRE_POSTPROCESS_GPU_MATCH': '1',
        })
        completed = self.run_launcher(self.root, env=env)
        self.assertEqual(0, completed.returncode, completed.stderr)
        workers = {
            row[0]: (row[1], row[5])
            for row in (line.split('|') for line in
                        self.worker_log.read_text().splitlines())
        }
        self.assertEqual(set(jobs), set(workers))
        evidence = [
            line.split('|')
            for line in Path(env['FAKE_POSTPROCESS_ENV_LOG']).read_text().splitlines()
        ]
        self.assertEqual(6, len(evidence))
        for action, key, visible, hash_seed in evidence:
            self.assertIn(action, {'resource-record', 'audit-run'})
            self.assertEqual(workers[key], (visible, hash_seed))
            self.assertEqual(key.rsplit(':', 1)[-1], hash_seed)
        self.assertEqual(3, len(list((self.root / 'records').glob('*.json'))))
        self.assertFalse(any((self.root / 'gpu_claims').iterdir()))

    def test_formal_worker_replaces_postprocess_affinity_for_next_gpu(self):
        first = 'formal:affinity-switch:42'
        second = 'formal:affinity-switch:44'
        self.queue('formal', first, second)
        self.queue('explanation')
        env = self.env.copy()
        env.pop('CUDA_VISIBLE_DEVICES', None)
        env.update({
            'FAKE_REQUIRED_CLAIM_WORKER': 'formal-worker-0',
            'FAKE_REQUIRE_HASH_SEED_MATCH': '1',
            'FAKE_REQUIRE_POSTPROCESS_GPU_MATCH': '1',
            'FAKE_SWITCH_GPU_AFTER_JOB': first,
        })
        completed = self.run_launcher(self.root, env=env)
        self.assertEqual(0, completed.returncode, completed.stderr)
        workers = [line.split('|') for line in
                   self.worker_log.read_text().splitlines()]
        self.assertEqual([(first, '0', '42'), (second, '1', '44')],
                         [(row[0], row[1], row[5]) for row in workers])
        evidence = [line.split('|') for line in
                    Path(env['FAKE_POSTPROCESS_ENV_LOG']).read_text().splitlines()]
        by_job = {}
        for action, key, visible, hash_seed in evidence:
            by_job.setdefault(key, set()).add((action, visible, hash_seed))
        self.assertEqual({
            first: {('resource-record', '0', '42'),
                    ('audit-run', '0', '42')},
            second: {('resource-record', '1', '44'),
                     ('audit-run', '1', '44')},
        }, by_job)

    def _assert_formal_command_seed_rejected(self, mode):
        self.queue('formal', 'formal:bad-seed:42')
        self.queue('explanation')
        env = self.env.copy()
        env.update({
            'FAKE_COMMAND_SEED_MODE': mode,
            'FAKE_REQUIRE_HASH_SEED_MATCH': '1',
        })
        completed = self.run_launcher(self.root, env=env)
        self.assertNotEqual(0, completed.returncode)
        self.assertTrue((self.root / 'FAILED_JOB').is_file())
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertFalse((self.root / 'FORMAL_EXECUTION_SUCCESS').exists())
        self.assertFalse(self.worker_log.exists())
        self.assertFalse((self.root / 'records').exists())
        self.assertFalse(any((self.root / 'gpu_claims').iterdir()))
        self.assertEqual([], self._live_logged_fake_children())
        self.assertEqual([], self._live_logged_dormant_children())
        self.assertEqual(set(), self._new_child_gates())
        calls = self.driver_log.read_text()
        self.assertIn('command --spec formal:bad-seed:42', calls)
        self.assertNotIn('resource-record --root', calls)
        self.assertNotIn('audit-run --root', calls)

    def test_formal_command_missing_seed_is_rejected_prelaunch(self):
        self._assert_formal_command_seed_rejected('missing')

    def test_formal_command_duplicate_seed_is_rejected_prelaunch(self):
        self._assert_formal_command_seed_rejected('duplicate')

    def test_formal_command_valueless_seed_is_rejected_prelaunch(self):
        self._assert_formal_command_seed_rejected('valueless')

    def test_formal_command_noninteger_seed_is_rejected_prelaunch(self):
        self._assert_formal_command_seed_rejected('noninteger')

    def test_formal_command_noncanonical_seed_is_rejected_prelaunch(self):
        self._assert_formal_command_seed_rejected('noncanonical')

    def test_formal_command_spec_mismatched_seed_is_rejected_prelaunch(self):
        self._assert_formal_command_seed_rejected('mismatch')

    def test_serial_owner_construction_can_exceed_worker_registration_12s(self):
        self.queue('formal', 'formal:slow-owner-a:42', 'formal:slow-owner-b:42')
        self.queue('explanation')
        env = self.env.copy()
        env.update({
            'FAKE_OWNER_DELAY': '5',
            'FAKE_OWNER_DELAY_PHASE': 'formal',
        })
        started = time.monotonic()
        completed = self.run_launcher(self.root, env=env, timeout=30)
        elapsed = time.monotonic() - started
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertGreaterEqual(elapsed, 13.5)
        rows = self.worker_log.read_text().splitlines()
        self.assertEqual(2, len(rows))
        self.assertEqual(2, len(list((self.root / 'records').glob('*.json'))))
        for marker in (
                'FORMAL_PHASE_SUCCESS', 'EXPLANATION_PHASE_SUCCESS',
                'FORMAL_EXECUTION_SUCCESS'):
            self.assertTrue((self.root / marker).is_file(), marker)
        self.assertEqual([], self._live_logged_fake_children())
        self.assertEqual([], self._live_logged_dormant_children())
        self.assertEqual(set(), self._new_child_gates())
        self.assertFalse(any(self.tmpdir.glob('formal-worker-gates.*')))
        self.assertFalse(any((self.root / 'gpu_claims').iterdir()))

    def test_formal_foreign_compute_rows_share_only_capacity_eligible_gpu(self):
        self.queue('formal', 'formal:shared-a:42', 'formal:shared-b:42')
        self.queue('explanation')
        env = self.env.copy()
        lock_root = self.base / 'gpu-locks'
        lock_root.mkdir()
        env.update({
            'FAKE_GPU_ROWS': '0, GPU-0, 6000, 20\\n1, GPU-1, 2000, 0',
            'FAKE_COMPUTE_ROWS': (
                'GPU-0, 900001, /foreign/oia/python\\n'
                'GPU-1, 900002, /foreign/oia/python'
            ),
            'FAKE_GPU_LOCK_ROOT': str(lock_root),
            'FAKE_GPU_OVERLAP_LOG': str(self.base / 'gpu-overlap.log'),
        })
        completed = self.run_launcher(self.root, env=env)
        self.assertEqual(0, completed.returncode, completed.stderr)
        rows = [line.split('|') for line in self.worker_log.read_text().splitlines()]
        self.assertEqual(2, len(rows))
        self.assertEqual({'0'}, {row[1] for row in rows})
        self.assertFalse((self.base / 'gpu-overlap.log').exists())
        self.assertFalse(any((self.root / 'gpu_claims').iterdir()))

    def assert_formal_final_recheck_prelaunch_failure(self, completed):
        self.assertEqual(95, completed.returncode, completed.stderr)
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertFalse(self.worker_log.exists())
        self.assertFalse((self.root / 'runs').exists())
        self.assertEqual([], list((self.root / 'claims').iterdir()))
        self.assertEqual([], list((self.root / 'gpu_claims').iterdir()))
        actions = [line.split()[0]
                   for line in self.driver_log.read_text().splitlines()]
        self.assertNotIn('prepare-run', actions)
        self.assertNotIn('command', actions)
        self.assertEqual([], self._live_logged_fake_children())
        self.assertEqual([], self._live_logged_dormant_children())
        self.assertEqual(set(), self._new_child_gates())
        self.assertEqual([], list(self.tmpdir.glob('formal-worker-gates.*')))

    def test_formal_final_recheck_releases_claim_before_retry_and_launch(self):
        self.queue('formal', 'formal:recheck:42')
        self.queue('explanation')
        env = self.env.copy()
        env.update({
            'FAKE_GPU_ROWS': '0, GPU-0, 8000, 0\\n1, GPU-1, 2000, 0',
            'FAKE_GPU0_LOW_MEMORY_CALLS': '2',
            'FAKE_REQUIRED_CLAIM_WORKER': 'formal-worker-0',
        })
        completed = self.run_launcher(self.root, env=env)
        self.assertEqual(0, completed.returncode, completed.stderr)
        rows = [line.split('|') for line in self.worker_log.read_text().splitlines()]
        self.assertEqual(1, len(rows))
        self.assertEqual('0', rows[0][1])
        self.assertGreaterEqual(
            int(Path(env['FAKE_GPU_CALLS_FILE']).read_text()), 4)
        self.assertFalse(any((self.root / 'gpu_claims').iterdir()))

    def test_formal_final_recheck_query_failure_is_95_and_releases_gpu_claim(self):
        self.queue('formal', 'formal:recheck-query-fail:42')
        self.queue('explanation')
        env = self.env.copy()
        env.update({
            'FAKE_GPU_ROWS': '0, GPU-0, 8000, 0\\n1, GPU-1, 2000, 0',
            'FAKE_GPU_FAIL_CALLS': '2',
            'FAKE_REQUIRED_CLAIM_WORKER': 'formal-worker-0',
        })
        completed = self.run_launcher(self.root, env=env)
        self.assert_formal_final_recheck_prelaunch_failure(completed)

    def test_formal_final_recheck_malformed_inventory_cleans_all_prelaunch_state(self):
        self.queue('formal', 'formal:recheck-malformed:42')
        self.queue('explanation')
        env = self.env.copy()
        env.update({
            'FAKE_GPU_ROWS': '0, GPU-0, 8000, 0\\n1, GPU-1, 2000, 0',
            'FAKE_GPU_MALFORMED_CALLS': '2',
            'FAKE_REQUIRED_CLAIM_WORKER': 'formal-worker-0',
        })
        completed = self.run_launcher(self.root, env=env)
        self.assert_formal_final_recheck_prelaunch_failure(completed)

    def test_formal_final_recheck_uuid_change_cleans_all_prelaunch_state(self):
        self.queue('formal', 'formal:recheck-uuid-change:42')
        self.queue('explanation')
        env = self.env.copy()
        env.update({
            'FAKE_GPU_ROWS': '0, GPU-0, 8000, 0\\n1, GPU-1, 2000, 0',
            'FAKE_GPU0_UUID_CHANGE_CALLS': '2',
            'FAKE_REQUIRED_CLAIM_WORKER': 'formal-worker-0',
        })
        completed = self.run_launcher(self.root, env=env)
        self.assert_formal_final_recheck_prelaunch_failure(completed)

    def test_formal_final_recheck_release_failure_once_retries_cleanup(self):
        self.queue('formal', 'formal:recheck-release-fail:42')
        self.queue('explanation')
        env = self.env.copy()
        env.update({
            'FAKE_GPU_ROWS': '0, GPU-0, 8000, 0\\n1, GPU-1, 2000, 0',
            'FAKE_GPU_FAIL_CALLS': '2',
            'FAKE_GPU_RELEASE_FAIL_ONCE_FILE': str(
                self.base / 'gpu-release-failed-once'),
            'FAKE_REQUIRED_CLAIM_WORKER': 'formal-worker-0',
        })
        completed = self.run_launcher(self.root, env=env)
        self.assert_formal_final_recheck_prelaunch_failure(completed)
        actions = [line.split()[0]
                   for line in self.driver_log.read_text().splitlines()]
        self.assertEqual(2, actions.count('gpu-release'))

    def test_formal_gpu_claim_install_then_failure_cleans_all_prelaunch_state(self):
        self.queue('formal', 'formal:claim-install-fail:42')
        self.queue('explanation')
        env = self.env.copy()
        env.update({
            'FAKE_GPU_ROWS': '0, GPU-0, 8000, 0\\n1, GPU-1, 2000, 0',
            'FAKE_GPU_CLAIM_INSTALL_THEN_FAIL': '17',
            'FAKE_REQUIRED_CLAIM_WORKER': 'formal-worker-0',
        })
        completed = self.run_launcher(self.root, env=env)
        self.assert_formal_final_recheck_prelaunch_failure(completed)
        actions = [line.split()[0]
                   for line in self.driver_log.read_text().splitlines()]
        self.assertEqual(1, actions.count('gpu-claim'))
        self.assertEqual(1, actions.count('gpu-release'))

    def test_dynamic_gpu_fallback_and_unsafe_existing_claim_is_never_stolen(self):
        self.queue('formal', 'formal:fallback:42')
        self.queue('explanation')
        claim = self.root / 'gpu_claims' / 'gpu-0'
        claim.mkdir(parents=True)
        (claim / 'owner.json').write_text('unsafe\n')
        self.env['FAKE_GPU_ROWS'] = (
            '0, GPU-0, 9000, 0\\n1, GPU-1, 9000, 0')
        self.env['FAKE_MONITOR_JOB'] = 'formal:fallback:42'
        self.env['FAKE_MONITOR_UUID'] = 'GPU-1'
        completed = self.run_launcher(self.root)
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual('unsafe\n', (claim / 'owner.json').read_text())
        rows = self.worker_log.read_text().splitlines()
        self.assertEqual(1, len(rows))
        self.assertEqual('1', rows[0].split('|')[1])
        self.assertIn(
            '--peak-gpu-memory-bytes 128974848',
            self.driver_log.read_text())

    def test_first_failure_stops_only_owned_sibling_and_foreign_survives(self):
        self.queue('formal', 'formal:fail:42', 'formal:slow:42')
        self.queue('explanation')
        self.env.update({
            'FAKE_FAIL_JOB': 'formal:fail:42',
            'FAKE_SLOW_JOB': 'formal:slow:42',
        })
        foreign = subprocess.Popen(
            ['setsid', 'bash', '-c',
             'trap "exit 0" TERM; while :; do sleep 1; done'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(self._stop_owned_process_group, foreign)
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertIsNone(foreign.poll())
        self.assertTrue((self.root / 'FAILED_JOB').is_file())
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        deadline = time.time() + 3
        killed = self.base / 'killed.log'
        while time.time() < deadline and not killed.exists():
            time.sleep(.02)
        self.assertIn('formal:slow:42', killed.read_text())
        self.assertFalse((self.root / 'FORMAL_EXECUTION_SUCCESS').exists())

    def test_zero_exit_with_audit_failure_stops_without_success(self):
        self.queue('formal', 'formal:incomplete:42')
        self.queue('explanation')
        self.env['FAKE_AUDIT_FAIL'] = 'formal:incomplete:42'
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertFalse((self.root / 'FORMAL_EXECUTION_SUCCESS').exists())

    def test_peak_monitor_failure_stops_before_resource_or_audit(self):
        self.queue('formal', 'formal:monitor-fail:42')
        self.queue('explanation')
        self.env['FAKE_MONITOR_JOB'] = 'formal:monitor-fail:42'
        self.env['FAKE_MONITOR_FAIL'] = '1'
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        calls = self.driver_log.read_text()
        self.assertNotIn('resource-record --root', calls)
        self.assertNotIn('audit-run --root', calls)

    def test_peak_temp_failure_never_leaves_spawned_job_alive(self):
        job = 'formal:mktemp-fail:42'
        self.queue('formal', job)
        self.queue('explanation')
        self.env['FAKE_PEAK_MKTEMP_FAIL'] = '1'
        self.env['FAKE_SLOW_JOB'] = job
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertTrue((self.root / 'FAILED_JOB').is_file())
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertEqual([], self._live_logged_fake_children())

    def test_child_pgid_probe_failure_never_executes_job(self):
        self.queue('formal', 'formal:child-pgid-fail:42')
        self.queue('explanation')
        self.env['FAKE_CHILD_PGID_FAIL'] = '1'
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertTrue((self.root / 'FAILED_JOB').is_file())
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertFalse(self.worker_log.exists())
        self.assertEqual([], self._live_logged_dormant_children())

    def test_child_group_identity_mismatch_never_executes_job(self):
        self.queue('formal', 'formal:child-pgid-wrong:42')
        self.queue('explanation')
        self.env['FAKE_CHILD_PGID_WRONG'] = '1'
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertTrue((self.root / 'FAILED_JOB').is_file())
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertFalse(self.worker_log.exists())
        self.assertEqual([], self._live_logged_dormant_children())

    def test_child_token_corruption_never_releases_command(self):
        self.queue('formal', 'formal:child-token-corrupt:42')
        self.queue('explanation')
        self.env['FAKE_CHILD_TOKEN_CORRUPT'] = '1'
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertTrue((self.root / 'FAILED_JOB').is_file())
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertFalse(self.worker_log.exists())
        self.assertEqual([], self._live_logged_dormant_children())

    def test_signal_during_child_registration_kills_dormant_group(self):
        job = 'formal:child-register-signal:42'
        self.queue('formal', job)
        self.queue('explanation')
        self.env['FAKE_CHILD_PGID_DELAY'] = '0.8'
        self.env['FAKE_SLOW_JOB'] = job
        process = subprocess.Popen(
            ['bash', str(self.launcher), str(self.root)], env=self.env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self._stop_process, process)
        ready = self.base / 'child-register-ready'
        deadline = time.time() + 8
        while time.time() < deadline and not ready.exists():
            if process.poll() is not None:
                break
            time.sleep(.02)
        self.assertTrue(ready.exists())
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)
        self.assertNotEqual(0, process.returncode)
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertFalse(self.worker_log.exists())
        self.assertEqual([], self._live_logged_dormant_children())

    def test_signal_during_child_gate_release_never_executes_job(self):
        job = 'formal:child-gate-signal:42'
        self.queue('formal', job)
        self.queue('explanation')
        self.env['FAKE_CHILD_GATE_DELAY'] = '0.8'
        self.env['FAKE_SLOW_JOB'] = job
        process = subprocess.Popen(
            ['bash', str(self.launcher), str(self.root)], env=self.env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self._stop_process, process)
        ready = self.base / 'child-gate-ready'
        deadline = time.time() + 2
        while time.time() < deadline and not ready.exists():
            if process.poll() is not None:
                break
            time.sleep(.02)
        self.assertTrue(ready.exists())
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)
        self.assertNotEqual(0, process.returncode)
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertFalse(self.worker_log.exists())
        self.assertEqual([], self._live_logged_dormant_children())

    def test_term_ignoring_child_is_reaped_before_parent_worker_grace(self):
        job = 'formal:ignore-term:42'
        self.queue('formal', job)
        self.queue('explanation')
        self.env['FAKE_IGNORE_TERM_JOB'] = job
        self.env['FAKE_GPU_RELEASE_DELAY'] = '3'
        process = subprocess.Popen(
            ['bash', str(self.launcher), str(self.root)], env=self.env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self._stop_process, process)
        ready = self.base / 'ready.log'
        deadline = time.time() + 8
        while time.time() < deadline and not ready.exists():
            if process.poll() is not None:
                break
            time.sleep(.02)
        self.assertTrue(ready.exists())
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=20)
        self.assertNotEqual(0, process.returncode)
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertEqual([], self._live_logged_dormant_children())
        self.assertEqual(set(), self._new_child_gates())

    def test_parent_timeout_cleans_child_control_and_gate_before_worker_kill(self):
        job = 'formal:parent-timeout:42'
        self.queue('formal', job)
        self.queue('explanation')
        self.env['FAKE_IGNORE_TERM_JOB'] = job
        self.env['FAKE_GPU_RELEASE_DELAY'] = '15'
        process = subprocess.Popen(
            ['bash', str(self.launcher), str(self.root)], env=self.env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self._stop_process, process)
        ready = self.base / 'ready.log'
        deadline = time.time() + 8
        while time.time() < deadline and not ready.exists():
            if process.poll() is not None:
                break
            time.sleep(.02)
        self.assertTrue(ready.exists())
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=25)
        self.assertNotEqual(0, process.returncode)
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertEqual([], self._live_logged_dormant_children())
        self.assertEqual(set(), self._new_child_gates())
        self.assertFalse(any((self.root / 'gpu_claims').glob('gpu-*')))

    def test_prepare_control_failure_stops_before_worker_spawn(self):
        job = 'formal:prepare-fail:42'
        self.queue('formal', job)
        self.queue('explanation')
        self.env['FAKE_PREPARE_FAIL'] = job
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertTrue((self.root / 'FAILED_JOB').is_file())
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertFalse(self.worker_log.exists())
        self.assertEqual([], self._live_logged_fake_children())

    def test_compute_query_exit_is_prelaunch_failure(self):
        self.queue('formal', 'formal:compute-exit:42')
        self.queue('explanation')
        self.env['FAKE_COMPUTE_FAIL'] = '1'
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertTrue((self.root / 'FAILED_JOB').is_file())
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertFalse(self.worker_log.exists())

    def test_malformed_compute_query_is_prelaunch_failure(self):
        self.queue('formal', 'formal:compute-malformed:42')
        self.queue('explanation')
        self.env['FAKE_COMPUTE_ROWS'] = 'GPU-0, not-a-pid'
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertTrue((self.root / 'FAILED_JOB').is_file())
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertFalse(self.worker_log.exists())

    def test_compute_query_timeout_is_prelaunch_failure(self):
        self.queue('formal', 'formal:compute-timeout:42')
        self.queue('explanation')
        self.env['FAKE_COMPUTE_SLEEP'] = '3'
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertTrue((self.root / 'FAILED_JOB').is_file())
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertFalse(self.worker_log.exists())

    def test_gpu_inventory_query_exit_is_prelaunch_failure(self):
        self.queue('formal', 'formal:gpu-query-exit:42')
        self.queue('explanation')
        self.env['FAKE_GPU_FAIL'] = '1'
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertTrue((self.root / 'FAILED_JOB').is_file())
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertFalse(self.worker_log.exists())

    def test_malformed_gpu_inventory_is_prelaunch_failure(self):
        self.queue('formal', 'formal:gpu-query-malformed:42')
        self.queue('explanation')
        self.env['FAKE_GPU_ROWS'] = (
            '0, GPU-0, not-memory, 0\\n1, GPU-1, 8000, 0')
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertTrue((self.root / 'FAILED_JOB').is_file())
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertFalse(self.worker_log.exists())

    def test_signal_uses_owned_cleanup_and_preserves_nonzero(self):
        self.queue('formal', 'formal:slow-a:42', 'formal:slow-b:42')
        self.queue('explanation')
        self.env['FAKE_SLOW_JOB'] = 'formal:slow-a:42'
        # Both fake jobs become slow for this test through a tiny wrapper match.
        original = self.worker.read_text()
        self.worker.write_text(original.replace(
            '[[ "$spec" == "${FAKE_SLOW_JOB:-never}" ]]',
            '[[ "$spec" == formal:slow-* ]]'))
        process = subprocess.Popen(
            ['bash', str(self.launcher), str(self.root)], env=self.env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ready = self.base / 'ready.log'
        deadline = time.time() + 8
        while time.time() < deadline:
            if ready.exists() and len(ready.read_text().splitlines()) >= 2:
                break
            if process.poll() is not None:
                break
            time.sleep(.05)
        self.assertIsNone(process.poll())
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)
        self.assertNotEqual(0, process.returncode)
        killed = self.base / 'killed.log'
        self.assertEqual(2, len(killed.read_text().splitlines()))
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())

    def test_signal_during_delayed_setsid_never_releases_job(self):
        self.queue('formal', 'formal:delayed-setsid:42')
        self.queue('explanation')
        self.env['FAKE_SETSID_DELAY'] = '0.6'
        process = subprocess.Popen(
            ['bash', str(self.launcher), str(self.root)], env=self.env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ready = self.base / 'setsid-ready.log'
        deadline = time.time() + 8
        while time.time() < deadline and not ready.exists():
            if process.poll() is not None:
                break
            time.sleep(.02)
        self.assertTrue(ready.exists())
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)
        self.assertNotEqual(0, process.returncode)
        time.sleep(.8)
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertFalse(self.worker_log.exists())
        self.assertEqual([], self._live_logged_fake_children())

    def test_signal_while_gate_owner_is_delayed_never_runs_job(self):
        self.queue('formal', 'formal:delayed-gate:42')
        self.queue('explanation')
        self.env['FAKE_OWNER_DELAY'] = '0.6'
        process = subprocess.Popen(
            ['bash', str(self.launcher), str(self.root)], env=self.env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ready = self.base / 'owner-ready.log'
        deadline = time.time() + 8
        while time.time() < deadline and not ready.exists():
            if process.poll() is not None:
                break
            time.sleep(.02)
        self.assertTrue(ready.exists())
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)
        self.assertNotEqual(0, process.returncode)
        time.sleep(.8)
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertFalse(self.worker_log.exists())
        self.assertEqual([], self._live_logged_fake_children())

    def test_plan_change_before_gate_release_stops_dormant_workers(self):
        self.queue('formal', 'formal:pre-release-plan-change:42')
        self.queue('explanation')
        self.env['FAKE_MUTATE_PLAN_ON_OWNER'] = '1'
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertFalse(self.worker_log.exists())
        self.assertEqual([], self._live_logged_fake_children())

    def test_plan_change_after_empty_claim_is_caught_after_worker_wait(self):
        self.queue('formal')
        self.queue('explanation')
        self.env['FAKE_MUTATE_PLAN_ON_EMPTY_CLAIM'] = '1'
        completed = self.run_launcher(self.root)
        self.assertNotEqual(0, completed.returncode)
        self.assertTrue((self.root / 'FORMAL_STOPPED').is_file())
        self.assertFalse((self.root / 'FORMAL_PHASE_SUCCESS').exists())
        self.assertFalse(self.worker_log.exists())
        self.assertEqual([], self._live_logged_fake_children())


if __name__ == '__main__':
    unittest.main()
