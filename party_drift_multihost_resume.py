"""Central, single-writer three-lane resume for the frozen development queue."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import party_drift_balanced_resume as balanced
import party_drift_queue as queue
from party_drift_mailbox import FileBroker, write_once
from party_drift_multihost import (HandoffGate, NotAdmitted, dispatch_three,
    execute_local_job, execute_remote_job, next_lock, select_remote_keys,
    validate_broker_report, validate_remote_key_anchor, wait_remote_ready)
from party_drift_remote_broker import audited_gpu_snapshot


def validate_remote_anchor(anchor, pending):
    validate_remote_key_anchor(pending, anchor.get('remote_job_keys'))
    remote = anchor.get('remote')
    if not isinstance(remote, dict):
        raise ValueError('remote anchor absent')
    for field in ('host', 'gpu_uuid', 'mount_source'):
        if not isinstance(remote.get(field), str) or not remote[field]:
            raise ValueError(f'remote {field} anchor absent')
    if not remote['gpu_uuid'].startswith('GPU-'):
        raise ValueError('remote GPU UUID anchor differs')
    environment = remote.get('environment')
    if (not isinstance(environment, dict) or
            set(environment) != {'driver', 'python', 'torch', 'torch_cuda'} or
            any(not isinstance(value, str) or not value
                for value in environment.values())):
        raise ValueError('remote software anchor incomplete')


def read_control_gpu(control, *, max_age_seconds=90):
    path = Path(control) / 'HEARTBEAT_CURRENT.json'
    if not path.is_file():
        raise ValueError('remote broker has no heartbeat')
    if time.time() - path.stat().st_mtime > max_age_seconds:
        raise ValueError('remote heartbeat is stale')
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as error:
        raise RuntimeError('remote heartbeat JSON is corrupt') from error
    if (record.get('protocol') != 1 or
            type(record.get('sequence')) is not int or
            record['sequence'] < 1 or
            type(record.get('pid')) is not int or record['pid'] <= 1 or
            not isinstance(record.get('gpu'), dict)):
        raise RuntimeError('remote heartbeat is corrupt')
    return record['gpu']


def prepare_control(control, anchor_path, anchor_sha256, *, result_root,
                    manifest_sha256, producer_commit, scheduler_commit):
    control = Path(control).resolve()
    root = Path(result_root).resolve()
    if control.is_relative_to(root):
        raise ValueError('control directory must be outside result root')
    parent = control.parent
    if not parent.is_dir():
        raise ValueError('control directory parent is absent')
    if os.name == 'posix' and (parent.stat().st_uid != os.getuid() or
                               parent.stat().st_mode & 0o022):
        raise ValueError('control directory parent is not privately owned')
    if queue._sha256(anchor_path) != anchor_sha256:
        raise ValueError('external anchor bytes differ')
    control.mkdir(mode=0o700, exist_ok=False)
    write_once(control / 'CONTROL_ANCHOR.json', {
        'protocol': 1, 'anchor_sha256': anchor_sha256,
        'manifest_sha256': manifest_sha256,
        'production_commit': producer_commit,
        'scheduler_commit': scheduler_commit,
    })
    return control


def validate_control(control, anchor_sha256, *, manifest_sha256,
                     producer_commit, scheduler_commit):
    control = Path(control).resolve()
    if not control.is_dir():
        raise ValueError('prepared control directory absent')
    if os.name == 'posix' and (control.stat().st_mode & 0o077 or
                               control.stat().st_uid != os.getuid()):
        raise ValueError('control directory is not private')
    record = json.loads((control / 'CONTROL_ANCHOR.json').read_text(
        encoding='utf-8'))
    expected = {'protocol': 1, 'anchor_sha256': anchor_sha256,
        'manifest_sha256': manifest_sha256,
        'production_commit': producer_commit,
        'scheduler_commit': scheduler_commit}
    if record != expected:
        raise ValueError('prepared control directory differs from anchor')


def reject_prior_multihost(root):
    if list(Path(root).glob('MULTIHOST_HANDOFF*.json')):
        raise ValueError('prior multihost handoff needs manual remote process audit; '
                         'automatic recovery is forbidden')


def require_lock_budget(root, *, remote_count):
    numbers = [int(path.stem.rsplit('_', 1)[1]) for path in
               Path(root).glob('QUEUE_LOCK_RESUME_*.json')]
    if max(numbers, default=0) + 1 + 2 * remote_count > 99:
        raise ValueError('insufficient two-digit resume lock slots')


def write_completion_and_stop(root, control, completion, *, manifest_sha256,
                              producer_commit, scheduler_commit, events):
    write_once(Path(control) / 'STOP.json', {
        'protocol': 1, 'manifest_sha256': manifest_sha256,
        'production_commit': producer_commit,
        'scheduler_commit': scheduler_commit,
        'attempts_sha256': queue._sha256(events),
    })
    queue._write_once(Path(root) / 'QUEUE_COMPLETE.json', completion)


def execute_multihost_in_root(rows, configs, manifest, root, producer_checkout,
                              manifest_sha256, scheduler_commit,
                              producer_commit, gpu_uuids, anchor,
                              anchor_sha256, control, *,
                              local_snapshot=audited_gpu_snapshot,
                              local_spawn=subprocess.Popen,
                              ready_local=None, ready_remote=None):
    root = Path(root).resolve()
    reject_prior_multihost(root)
    manifest = Path(manifest).resolve()
    events = root / 'ATTEMPTS.jsonl'
    if (root / 'QUEUE_COMPLETE.json').exists():
        raise ValueError('completed queue cannot be resumed')
    prior_lock = queue.active_queue_lock(root)
    if (prior_lock.get('manifest_sha256') != manifest_sha256 or
            prior_lock.get('production_commit') != producer_commit):
        raise ValueError('prior queue lock differs from frozen protocol')
    source_sha = queue._sha256(events)
    source_events = [json.loads(line) for line in events.read_text(
        encoding='utf-8').splitlines()]
    source_state = balanced.validate_history(rows, source_events, root,
                                             manifest_sha256)
    validate_remote_anchor(anchor, source_state['pending'])
    require_lock_budget(root, remote_count=len(anchor['remote_job_keys']))
    queue.recover_interrupted_attempts(
        root, manifest_sha256, 'verified multihost scheduler handoff',
        {row['job_key']: row['config_sha256'] for row in rows})
    recovered_events = [json.loads(line) for line in events.read_text(
        encoding='utf-8').splitlines()]
    state = balanced.validate_history(rows, recovered_events, root,
                                      manifest_sha256)
    validate_remote_anchor(anchor, state['pending'])
    if any(configs[key]['device'] != 'cuda:0' for key in state['pending']):
        raise ValueError('pending job uses unsupported logical GPU')
    queue._write_once(queue._next_record(root, 'MULTIHOST_HANDOFF'), {
        'status': 'scheduler_only_multihost_handoff',
        'new_scheduler_pid': os.getpid(),
        'startup_command': shlex.join([sys.executable, *sys.argv]),
        'manifest_sha256': manifest_sha256,
        'production_commit': producer_commit,
        'scheduler_commit': scheduler_commit,
        'original_queue_pid': prior_lock['pid'],
        'source_ledger_sha256': source_sha,
        'recovered_ledger_sha256': queue._sha256(events),
        'anchor_sha256': anchor_sha256,
        'gpu_uuids': list(gpu_uuids),
        'remote': anchor['remote'],
        'remote_job_keys': anchor['remote_job_keys'],
        'pending_keys': state['pending'],
        'reconstructed_keys': source_state['reconstructable'],
        'started_utc': datetime.now(timezone.utc).isoformat(),
    })
    queue._write_once(next_lock(root), {
        'pid': os.getpid(), 'status': 'running',
        'manifest_sha256': manifest_sha256,
        'production_commit': producer_commit,
        'scheduler_commit': scheduler_commit,
        'started_utc': datetime.now(timezone.utc).isoformat(),
    })
    gate = HandoffGate(root, events, state['pending'],
        manifest_sha256=manifest_sha256, producer_commit=producer_commit,
        scheduler_commit=scheduler_commit)
    by_key = {row['job_key']: row for row in rows}
    remote = anchor['remote']
    sequence = [0]

    def local(key, gpu):
        row = by_key[key]
        approved = manifest.parent / row['config_relpath']
        return execute_local_job(row, configs[key], approved, gate,
            producer_checkout=producer_checkout, manifest=manifest, gpu=gpu,
            gpu_uuid=gpu_uuids[gpu], snapshot_fn=local_snapshot,
            spawn=local_spawn)

    def remote_job(key):
        if key not in gate.pending:
            raise NotAdmitted('local lane claimed candidate during remote wait')
        sequence[0] += 1
        row = by_key[key]
        approved = manifest.parent / row['config_relpath']
        broker = FileBroker(control, sequence[0], result_root=root, key=key,
            manifest_sha256=manifest_sha256, producer_commit=producer_commit,
            scheduler_commit=scheduler_commit,
            config_sha256=row['config_sha256'])
        broker.assign()
        try:
            return execute_remote_job(row, configs[key], approved, gate, broker,
                gpu_uuid=remote['gpu_uuid'], host=remote['host'],
                mount_source=remote['mount_source'],
                environment=remote['environment'])
        except NotAdmitted:
            broker.cancel()
            raise

    try:
        dispatch_three(gate, anchor['remote_job_keys'],
            ready_local=ready_local or (lambda gpu, stop: balanced.wait_ready(
                gpu, gate.pending, stop, snapshot_fn=local_snapshot)),
            run_local=local,
            ready_remote=ready_remote or (lambda stop: wait_remote_ready(
                lambda: read_control_gpu(control), remote['gpu_uuid'],
                stop=stop) is not None),
            run_remote=remote_job)
        final_events = [json.loads(line) for line in events.read_text(
            encoding='utf-8').splitlines()]
        final_state = balanced.validate_history(rows, final_events, root,
                                                manifest_sha256)
        completion = balanced.completion_record(
            final_state, len(rows), manifest_sha256, queue._sha256(events))
        write_completion_and_stop(root, control, completion,
            manifest_sha256=manifest_sha256,
            producer_commit=producer_commit,
            scheduler_commit=scheduler_commit, events=events)
        return completion
    except BaseException as error:
        path = (root / 'QUEUE_FAILURE.json' if not
                (root / 'QUEUE_FAILURE.json').exists() else
                queue._next_record(root, 'QUEUE_FAILURE'))
        queue._write_once(path, {
            'status': 'multihost_queue_failed',
            'type': type(error).__name__, 'message': str(error),
            'manifest_sha256': manifest_sha256,
            'attempts_sha256': queue._sha256(events),
            'ended_utc': datetime.now(timezone.utc).isoformat(),
        })
        raise


def run_multihost_queue(manifest, manifest_sha256, result_root,
                        producer_checkout, anchor_path, anchor_sha256):
    root = Path(result_root).resolve()
    with queue.queue_run_lock(root):
        reject_prior_multihost(root)
        balanced.assert_original_queue_gone(root, manifest)
        rows, configs, state, producer, snapshots = balanced.load_run_inputs(
            manifest, manifest_sha256, root, producer_checkout,
            allow_active=False)
        if state['active'] or not state['pending']:
            raise ValueError('handoff has no safe unstarted jobs')
        scheduler = balanced._git_commit(Path(__file__).resolve().parent,
                                         require_clean=True)
        anchor_path = Path(anchor_path).resolve()
        if queue._sha256(anchor_path) != anchor_sha256:
            raise ValueError('external anchor bytes differ')
        anchor = json.loads(anchor_path.read_text(encoding='utf-8'))
        balanced.validate_anchor(anchor, producer=producer,
            scheduler=scheduler, manifest=manifest_sha256,
            ledger=queue._sha256(root / 'ATTEMPTS.jsonl'),
            gpu_uuids=[snapshot['uuid'] for snapshot in snapshots])
        proof = balanced.load_anchored_admission_proof(anchor)
        proof_gpu = proof.get('gpu_index', 1)
        if anchor.get('admission_gpu_index') != proof_gpu:
            raise ValueError('local GPU admission proof differs')
        balanced.validate_admission_proof(
            proof, manifest=manifest_sha256,
            ledger=queue._sha256(root / 'ATTEMPTS.jsonl'),
            producer=producer, scheduler=scheduler,
            snapshot=balanced.gpu_snapshot(proof_gpu))
        if (proof.get('original_queue_pid') !=
                queue.active_queue_lock(root)['pid'] or
                proof.get('pending_count') != len(state['pending']) or
                proof.get('reconstructable') != state['reconstructable']):
            raise ValueError('local GPU admission proof differs from paused queue')
        validate_remote_anchor(anchor, state['pending'])
        if anchor['remote']['gpu_uuid'] in [item['uuid'] for item in snapshots]:
            raise ValueError('remote GPU UUID matches a local card')
        control = Path(anchor['control_dir']).resolve()
        if control.is_relative_to(root):
            raise ValueError('control directory must be outside result root')
        validate_control(control, anchor_sha256,
            manifest_sha256=manifest_sha256, producer_commit=producer,
            scheduler_commit=scheduler)
        broker_info = json.loads((control / 'BROKER_INFO.json').read_text(
            encoding='utf-8'))
        validate_broker_report(broker_info,
            host=anchor['remote']['host'],
            mount_source=anchor['remote']['mount_source'],
            manifest_sha256=manifest_sha256,
            producer_commit=producer,
            scheduler_commit=scheduler,
            gpu_uuid=anchor['remote']['gpu_uuid'],
            environment=anchor['remote']['environment'])
        read_control_gpu(control)
        heartbeat = json.loads((control / 'HEARTBEAT_CURRENT.json').read_text(
            encoding='utf-8'))
        if heartbeat['pid'] != broker_info['pid']:
            raise ValueError('prepared broker PID differs from heartbeat')
        return execute_multihost_in_root(
            rows, configs, manifest, root, producer_checkout,
            manifest_sha256, scheduler, producer,
            [item['uuid'] for item in snapshots], anchor,
            anchor_sha256, control)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--producer-checkout', type=Path, required=True)
    parser.add_argument('--anchor', type=Path, required=True)
    parser.add_argument('--anchor-sha256', required=True)
    parser.add_argument('--prepare-control', action='store_true')
    args = parser.parse_args()
    if args.prepare_control:
        anchor = json.loads(args.anchor.read_text(encoding='utf-8'))
        producer = balanced._git_commit(args.producer_checkout,
                                        require_clean=True)
        scheduler = balanced._git_commit(Path(__file__).resolve().parent,
                                         require_clean=True)
        prepared = prepare_control(anchor['control_dir'], args.anchor,
            args.anchor_sha256, result_root=args.result_root,
            manifest_sha256=args.manifest_sha256,
            producer_commit=producer, scheduler_commit=scheduler)
        print(json.dumps({'status': 'control_prepared',
                          'control_dir': str(prepared)}, sort_keys=True))
        return
    print(json.dumps(run_multihost_queue(
        args.manifest, args.manifest_sha256, args.result_root,
        args.producer_checkout, args.anchor, args.anchor_sha256),
        sort_keys=True))


if __name__ == '__main__':
    main()
