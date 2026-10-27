"""Single-job remote broker. Never run against the production queue without audit.

The broker reports its PID, waits for GO, and directly spawns the frozen worker.
It never claims keys, edits queue locks, or appends ATTEMPTS.jsonl.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from party_drift_balanced_resume import gpu_snapshot
from party_drift_mailbox import replace_current, sha256, write_once
from party_drift_multihost import validate_remote_gpu
from party_drift_queue import active_queue_lock


def parse_thermal_audit(line):
    fields = [field.strip() for field in line.split(',')]
    if len(fields) != 3 or fields[2] not in ('Active', 'Not Active'):
        raise ValueError('GPU thermal audit format differs')
    return {'temperature_c': int(fields[0]), 'sm_clock_mhz': int(fields[1]),
            'sw_thermal_slowdown': fields[2] == 'Active'}


def audited_gpu_snapshot(index):
    snapshot = gpu_snapshot(index)
    result = subprocess.check_output([
        'nvidia-smi', '-i', str(index),
        '--query-gpu=temperature.gpu,clocks.sm,clocks_throttle_reasons.sw_thermal_slowdown',
        '--format=csv,noheader,nounits'], text=True).strip()
    return {**snapshot, **parse_thermal_audit(result)}


def broker_once(command, env, incoming, outgoing, *, spawn=subprocess.Popen,
                report=None, cwd=None, log_path=None):
    report = {**(report or {}), 'pid': os.getpid()}
    outgoing.write(json.dumps(report, sort_keys=True) + '\n')
    outgoing.flush()
    if incoming.readline() != 'GO\n':
        raise RuntimeError('broker did not receive exact GO authorization')
    if log_path is None:
        process = spawn(command, env=env, cwd=cwd)
        code = process.wait()
    else:
        with Path(log_path).open('x', encoding='utf-8') as stream:
            process = spawn(command, env=env, cwd=cwd, stdout=stream,
                            stderr=subprocess.STDOUT)
            code = process.wait()
    outgoing.write(json.dumps({'kind': 'exit', 'returncode': code},
                              sort_keys=True) + '\n')
    outgoing.flush()
    return code


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nfs_mount_source(findmnt_output):
    sources = [fields[1] for line in findmnt_output.splitlines()
               if len(fields := line.split()) >= 2 and
               fields[0] in ('nfs', 'nfs4')]
    if len(sources) != 1:
        raise ValueError('remote result path has no unique NFS source')
    return sources[0]


def verify_remote_authorization(result_root, assignment, broker_pid):
    """Read the same lock, plan, and start evidence as the frozen worker."""
    root = Path(result_root).resolve()
    key = assignment['job_key']
    job = (root / key).resolve()
    if not job.is_relative_to(root):
        raise ValueError('remote job escaped result root')
    try:
        lock = active_queue_lock(root)
        plan = json.loads((job / 'attempt-01.plan.json').read_text(
            encoding='utf-8'))
        history = [json.loads(line) for line in (root / 'ATTEMPTS.jsonl').read_text(
            encoding='utf-8').splitlines() if line]
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError('remote authorization evidence absent or incomplete') from error
    history = [row for row in history if row.get('job_key') == key]
    if (lock.get('pid') != broker_pid or
            lock.get('manifest_sha256') != assignment['manifest_sha256'] or
            lock.get('production_commit') != assignment['production_commit'] or
            plan != {'attempt': 1, 'job_key': key} or
            len(history) != 1 or
            history[0].get('kind') != 'start' or
            history[0].get('attempt') != 1 or
            history[0].get('config_sha256') != assignment['config_sha256']):
        raise ValueError('remote broker cannot see complete worker authorization')


def _wait_control(path, seconds=900):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if Path(path).is_file():
            return json.loads(Path(path).read_text(encoding='utf-8'))
        time.sleep(1)
    raise RuntimeError(f'control message absent: {path}')


def _wait_go_or_cancel(control, tag, seconds=900):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for kind in ('CANCEL', 'GO'):
            path = Path(control) / f'{kind}_{tag}.json'
            if path.is_file():
                return kind, json.loads(path.read_text(encoding='utf-8'))
        time.sleep(1)
    raise RuntimeError('GO or CANCEL absent; broker stops without spawning')


class HeartbeatWriter:
    def __init__(self, control, *, pid, snapshot):
        self.control = Path(control)
        self.pid = pid
        self.snapshot = snapshot
        self.sequence = 0
        self.stop = threading.Event()

    def pulse(self):
        self.sequence += 1
        replace_current(self.control / 'HEARTBEAT_CURRENT.json', {
            'protocol': 1, 'sequence': self.sequence, 'pid': self.pid,
            'gpu': self.snapshot(), 'time_utc': datetime.now(timezone.utc).isoformat(),
        })

    def run(self):
        while not self.stop.is_set():
            self.pulse()
            self.stop.wait(5)


def broker_mailbox_once(control, sequence, *, report, manifest,
                        manifest_sha256, producer_commit, command, env,
                        log_path, result_root, gpu_uuid, snapshot,
                        spawn=subprocess.Popen, cwd=None):
    """Handle one assigned key; spawn exactly once after a bound GO record."""
    control = Path(control)
    tag = f'{sequence:06d}'
    assignment_path = control / f'ASSIGN_{tag}.json'
    assignment = _wait_control(assignment_path)
    if sha256(manifest) != manifest_sha256:
        raise ValueError('remote manifest changed')
    rows = [json.loads(line) for line in Path(manifest).read_text(
        encoding='utf-8').splitlines()]
    row = next((item for item in rows
                if item.get('job_key') == assignment.get('job_key')), None)
    if (row is None or assignment.get('protocol') != 1 or
            assignment.get('sequence') != sequence or
            assignment.get('manifest_sha256') != manifest_sha256 or
            assignment.get('production_commit') != producer_commit or
            assignment.get('config_sha256') != row.get('config_sha256') or
            not assignment.get('scheduler_commit')):
        raise ValueError('remote assignment differs from frozen manifest')
    control_anchor_path = control / 'CONTROL_ANCHOR.json'
    if control_anchor_path.is_file():
        control_anchor = json.loads(control_anchor_path.read_text(
            encoding='utf-8'))
        if assignment['scheduler_commit'] != control_anchor.get('scheduler_commit'):
            raise ValueError('remote assignment scheduler differs from control anchor')
    bind = {
        'protocol': 1, 'sequence': sequence,
        'job_key': assignment['job_key'],
        'manifest_sha256': manifest_sha256,
        'production_commit': producer_commit,
        'scheduler_commit': assignment['scheduler_commit'],
        'assignment_sha256': sha256(assignment_path),
    }
    ready_path = control / f'READY_{tag}.json'
    write_once(ready_path, {**bind, 'report': {**report, 'pid': os.getpid()}})
    kind, instruction = _wait_go_or_cancel(control, tag)
    if instruction != {**bind, 'ready_sha256': sha256(ready_path)}:
        raise ValueError('remote GO/CANCEL differs from assignment and READY')
    if kind == 'CANCEL':
        return None
    current = snapshot()
    write_once(control / f'LAUNCH_CHECK_{tag}.json', {
        **bind, 'gpu': current, 'ready_sha256': sha256(ready_path)})
    try:
        validate_remote_gpu(current, gpu_uuid)
        verify_remote_authorization(result_root, assignment, os.getpid())
    except ValueError as error:
        write_once(control / f'ABORT_{tag}.json', {
            **bind, 'reason': str(error),
            'launch_check_sha256': sha256(control / f'LAUNCH_CHECK_{tag}.json')})
        raise RuntimeError('remote launch authorization failed after GO') from error
    command = command(assignment) if callable(command) else command
    env = env(assignment) if callable(env) else env
    log_path = log_path(assignment) if callable(log_path) else log_path
    with Path(log_path).open('x', encoding='utf-8') as stream:
        process = spawn(command, env=env, cwd=cwd, stdout=stream,
                        stderr=subprocess.STDOUT)
        code = process.wait()
    write_once(control / f'EXIT_{tag}.json', {**bind, 'returncode': code})
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer-checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--production-commit', required=True)
    parser.add_argument('--control-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--expected-mount-source', required=True)
    parser.add_argument('--min-disk-free-mib', type=int, default=4096)
    parser.add_argument('--result-root', type=Path, required=True)
    args = parser.parse_args()

    producer = args.producer_checkout.resolve()
    manifest = args.manifest.resolve()
    root = args.result_root.resolve()
    control = args.control_dir.resolve()
    if control.is_relative_to(root) or not control.is_dir():
        raise ValueError('existing control directory must be outside result root')
    if os.name == 'posix' and (control.stat().st_mode & 0o077 or
                               control.stat().st_uid != os.getuid()):
        raise ValueError('control directory must be owned by current user, mode 0700')
    control_anchor = json.loads((control / 'CONTROL_ANCHOR.json').read_text(
        encoding='utf-8'))
    if (control_anchor.get('protocol') != 1 or
            control_anchor.get('manifest_sha256') != args.manifest_sha256 or
            control_anchor.get('production_commit') != args.production_commit or
            not control_anchor.get('scheduler_commit')):
        raise ValueError('prepared control anchor differs')
    scheduler_checkout = Path(__file__).resolve().parent
    scheduler_commit = subprocess.check_output([
        'git', '-C', str(scheduler_checkout), 'rev-parse', 'HEAD'],
        text=True).strip()
    scheduler_dirty = subprocess.check_output([
        'git', '-C', str(scheduler_checkout), 'status', '--porcelain'],
        text=True).strip()
    if scheduler_commit != control_anchor['scheduler_commit'] or scheduler_dirty:
        raise ValueError('remote scheduler checkout differs or is dirty')
    if _sha256(manifest) != args.manifest_sha256:
        raise ValueError('remote manifest bytes differ')
    commit = subprocess.check_output(
        ['git', '-C', str(producer), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(
        ['git', '-C', str(producer), 'status', '--porcelain'], text=True).strip()
    if commit != args.production_commit or dirty:
        raise ValueError('remote frozen producer differs or is dirty')
    mount = nfs_mount_source(subprocess.check_output(
        ['findmnt', '-n', '-o', 'FSTYPE,SOURCE,TARGET', '-T', str(root)],
        text=True))
    if mount != args.expected_mount_source:
        raise ValueError('remote result mount differs')
    disk_free_mib = shutil.disk_usage(root).free // (1024 * 1024)
    if disk_free_mib < args.min_disk_free_mib:
        raise ValueError('remote result filesystem lacks free space')
    scratch = Path(tempfile.gettempdir()).resolve()
    scratch_free_mib = shutil.disk_usage(scratch).free // (1024 * 1024)
    if scratch_free_mib < args.min_disk_free_mib:
        raise ValueError('remote local scratch filesystem lacks free space')
    gpu = audited_gpu_snapshot(args.gpu_index)
    if gpu['name'] != 'NVIDIA GeForce RTX 4090' or gpu['uuid'] != args.gpu_uuid:
        raise ValueError('remote GPU identity differs')
    driver = subprocess.check_output([
        'nvidia-smi', '-i', str(args.gpu_index),
        '--query-gpu=driver_version', '--format=csv,noheader,nounits',
    ], text=True).strip()
    import torch
    report = {
        'pid': os.getpid(),
        'host': os.uname().nodename,
        'producer_commit': commit,
        'scheduler_commit': scheduler_commit,
        'manifest_sha256': args.manifest_sha256,
        'mount_source': mount,
        'result_disk_free_mib': disk_free_mib,
        'scratch_path': str(scratch),
        'scratch_free_mib': scratch_free_mib,
        'gpu': gpu,
        'driver': driver,
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
    }
    write_once(control / 'BROKER_INFO.json', report)
    def command(assignment):
        return [sys.executable, '-u', str(producer /
                'run_party_drift_development.py'),
                '--job-key', assignment['job_key'], '--manifest', str(manifest),
                '--manifest-sha256', args.manifest_sha256, '--attempt', '1']

    def environment(assignment):
        seed = int(assignment['job_key'].rsplit('/', 1)[1])
        return {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpu_uuid,
                'PYTHONHASHSEED': str(seed),
                'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}

    def log_path(assignment):
        job = (root / assignment['job_key']).resolve()
        if not job.is_relative_to(root):
            raise ValueError('remote job escaped result root')
        return job / 'attempt-01.log'

    heartbeat = HeartbeatWriter(control, pid=os.getpid(),
                                snapshot=lambda: audited_gpu_snapshot(args.gpu_index))
    thread = threading.Thread(target=heartbeat.run, daemon=True)
    thread.start()
    try:
        sequence = 1
        while True:
            assignment = control / f'ASSIGN_{sequence:06d}.json'
            while not assignment.is_file():
                if (control / 'STOP.json').is_file():
                    stop_record = json.loads((control / 'STOP.json').read_text(
                        encoding='utf-8'))
                    if (stop_record.get('protocol') != 1 or
                            stop_record.get('manifest_sha256') !=
                            args.manifest_sha256 or
                            stop_record.get('production_commit') != commit or
                            stop_record.get('scheduler_commit') !=
                            control_anchor['scheduler_commit'] or
                            not stop_record.get('attempts_sha256')):
                        raise ValueError('control STOP differs from active queue')
                    return 0
                time.sleep(1)
            broker_mailbox_once(
                control, sequence,
                report={**report, 'gpu': audited_gpu_snapshot(args.gpu_index)},
                manifest=manifest, manifest_sha256=args.manifest_sha256,
                producer_commit=commit, command=command, env=environment,
                log_path=log_path, gpu_uuid=args.gpu_uuid,
                snapshot=lambda: audited_gpu_snapshot(args.gpu_index),
                result_root=root,
                cwd=producer)
            sequence += 1
    finally:
        heartbeat.stop.set()
        thread.join(timeout=10)


if __name__ == '__main__':
    raise SystemExit(main())
