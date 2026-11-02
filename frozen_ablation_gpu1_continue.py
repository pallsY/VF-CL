"""Explicit, one-attempt continuation after a verified quiescent retirement.

Importing this module uses only the standard library. The frozen producer is
loaded from a separately pinned clean checkout, after worker authorization and
the parent-death watchdog have been installed.
"""

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ORIGIN_COMMIT = 'f4735cacf51c41d63a9f87e8933c968cb268637b'
MANIFEST_SHA = 'bb10fa6c62719b68124e7b625a204defc6b3e396ac285c068849fff8866e8fca'
GPU_UUID = {0: 'GPU-6c7fee92-a6fe-5525-c554-bddb8e14943f',
            1: 'GPU-6569e526-43ea-74e1-264e-4b66417181e1'}
EXTENSION = 'gpu1-rebalance-v1'
CANDIDATE_CAP = frozenset({'cifar100/no_partykd/43'} | {
    f'isolet/{arm}/{seed}' for arm in ('no_partykd', 'fim_off') for seed in (42, 43, 44)})


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_once(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def publish_once(path, value):
    """Publish an exclusive, fully written handshake without a partial-file window."""
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    try:
        write_once(temporary, value)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_event(root, value):
    with (Path(root) / 'ATTEMPTS.jsonl').open('ab') as stream:
        stream.write((json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())


def process_identity(pid):
    from frozen_ablation_quiescent import identity
    return identity(pid)


def identities_match(expected, actual):
    from frozen_ablation_quiescent import same_identity
    return same_identity(expected, actual)


def pending_candidates(keys):
    if not set(keys) <= CANDIDATE_CAP:
        raise ValueError('pending work exceeds the frozen candidate cap')
    return sorted(keys)


def execution_binding(row, index, epoch_sha, parent):
    if index not in GPU_UUID:
        raise ValueError('unknown physical GPU')
    return {'execution_epoch_sha256': epoch_sha, 'controller_identity': parent,
            'gpu_index': index, 'gpu_uuid': GPU_UUID[index],
            'logical_device': 'cuda:0',
            'cuda_visible_devices': [GPU_UUID[index], GPU_UUID[1 - index]]}


def worker_environment(config, index):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=','.join((GPU_UUID[index], GPU_UUID[1 - index])),
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED=str(config['seed']),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return env


def observed_cuda_order(index):
    import torch
    if torch.cuda.device_count() != 2:
        raise ValueError('worker must retain exactly two visible CUDA devices')
    observed = ['GPU-' + str(uuid.UUID(str(torch.cuda.get_device_properties(i).uuid).removeprefix('GPU-')))
                for i in range(2)]
    if observed != [GPU_UUID[index], GPU_UUID[1 - index]]:
        raise ValueError('observed CUDA logical-to-physical UUID order differs')
    return observed


def pinned_checkout(root, commit):
    root = Path(root).resolve()
    if (subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() != commit
            or subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()):
        raise ValueError('orchestration checkout is not the pinned clean commit')
    return root


def load_legacy(root):
    root = pinned_checkout(root, ORIGIN_COMMIT)
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('_gpu1_legacy_f473', root / 'frozen_ablation_run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_old_parent(expected, legacy_root, manifest, code_roots):
    actual = process_identity(expected['pid'])
    if not identities_match(expected, actual):
        raise ValueError('old controller differs from external process identity anchor')
    command = actual['cmdline']
    script = str(Path(legacy_root).resolve() / 'frozen_ablation_run.py')
    if (script not in command or '--worker' in command or command.count(script) != 1):
        raise ValueError('old controller command is not the pinned legacy queue')
    required = {'--manifest': str(Path(manifest).resolve()), '--manifest-sha256': MANIFEST_SHA,
                '--base-code': str(Path(code_roots['no_partykd']).resolve()),
                '--patch-code': str(Path(code_roots['fim_off']).resolve())}
    for flag, value in required.items():
        if command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
            raise ValueError('old controller command binding is absent or duplicated')
        supplied = command[command.index(flag) + 1]
        if (supplied != value if flag.endswith('sha256') else str(Path(supplied).resolve()) != value):
            raise ValueError('old controller command binding differs')
    return actual


def epoch_record(root, path, digest):
    root, path = Path(root).resolve(), Path(path).resolve()
    if path != root / EXTENSION / 'EXECUTION_EPOCH.json' or sha(path) != digest:
        raise ValueError('execution epoch path or external SHA differs')
    epoch = read_json(path)
    if sha(path.parent / 'ARMED.json') != epoch.get('armed_sha256'):
        raise ValueError('armed record changed')
    raw = (root / 'ATTEMPTS.jsonl').read_bytes()
    length = epoch['ledger_prefix_bytes_length']
    if (type(length) is not int or length < 0 or len(raw) < length or
            hashlib.sha256(raw[:length]).hexdigest() != epoch.get('ledger_prefix_sha256') or
            (length and raw[length - 1:length] != b'\n') or (raw and not raw.endswith(b'\n'))):
        raise ValueError('original ledger prefix changed or ledger is partial')
    return epoch


def authorize_worker(manifest, digest, key, epoch_path, epoch_sha):
    manifest = Path(manifest).resolve()
    root = manifest.parent.parent
    epoch = epoch_record(root, epoch_path, epoch_sha)
    parent = epoch['controller_identity']
    if (sha(manifest) != digest or epoch.get('manifest_sha256') != digest or
            os.getppid() != parent['pid'] or not identities_match(parent, process_identity(parent['pid']))):
        raise ValueError('worker parent or manifest authorization differs')
    pending_candidates(epoch['assignments'])
    if key not in epoch['assignments']:
        raise ValueError('worker is not an epoch candidate')
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    selected = [row for row in rows if row['job_key'] == key]
    if len(selected) != 1:
        raise ValueError('worker job is absent or duplicated')
    row = selected[0]
    config_path = (manifest.parent / row['config_relpath']).resolve()
    if (not config_path.is_relative_to(manifest.parent / 'configs') or
            sha(config_path) != row['config_sha256']):
        raise ValueError('worker original config bytes changed')
    config = read_json(config_path)
    job_root = Path(config['output_dir']).parent.resolve()
    if not job_root.is_relative_to(root) or config['device'] != 'cuda:0':
        raise ValueError('worker config output or logical device differs')
    binding = execution_binding(row, epoch['assignments'][key], epoch_sha, parent)
    own = [json.loads(line) for line in (root / 'ATTEMPTS.jsonl').read_text().splitlines()
           if json.loads(line).get('job_key') == key]
    if (len(own) != 1 or own[0].get('kind') != 'start' or own[0].get('attempt') != 1 or
            own[0].get('config_sha256') != row['config_sha256'] or
            own[0].get('code_commit') != row['code_commit'] or
            any(own[0].get(k) != v for k, v in binding.items()) or
            read_json(job_root / 'attempt-01.plan.json') != {'job_key': key, 'attempt': 1}):
        raise ValueError('worker is not one uniquely authorized queued attempt')
    expected_env = worker_environment(config, binding['gpu_index'])
    # Environment is checked in worker(), keeping this metadata verifier usable
    # by the independent audit without importing or initializing CUDA.
    return dict(row=row, config=config, config_path=config_path, job_root=job_root,
                binding=binding, epoch=epoch, expected_env=expected_env)


def protocol_binding(auth):
    return dict(job_key=auth['row']['job_key'], attempt=1,
                config_sha256=auth['row']['config_sha256'],
                training_code_commit=auth['row']['code_commit'], **auth['binding'])


def validate_ready(ready, expected, pid):
    identity = ready.get('worker_identity', {})
    if (ready != dict(expected, worker_identity=identity) or identity.get('pid') != pid or
            identity.get('session') != pid or identity.get('pgrp') != pid or
            not identities_match(identity, process_identity(pid))):
        raise ValueError('READY/GO worker identity or epoch differs')
    return identity


def parent_watchdog(parent, finished, interval=0.2):
    while not finished.is_set():
        try:
            alive = identities_match(parent, process_identity(parent['pid']))
        except (OSError, ValueError):
            alive = False
        if not alive:
            pid = os.getpid()
            if os.getsid(0) != pid or os.getpgrp() != pid:
                os._exit(70)
            os.killpg(pid, signal.SIGKILL)
            return
        finished.wait(interval)


def worker(manifest, digest, key, epoch_path, epoch_sha, legacy_root, code_roots):
    auth = authorize_worker(manifest, digest, key, epoch_path, epoch_sha)
    if (str(Path(legacy_root).resolve()) != auth['epoch']['legacy_root'] or
            {arm: str(Path(path).resolve()) for arm, path in code_roots.items()} != auth['epoch']['code_roots']):
        raise ValueError('worker frozen checkout roots differ from execution epoch')
    for name in ('CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG', 'PYTHONHASHSEED',
                 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(name) != auth['expected_env'][name]:
            raise ValueError('worker startup environment differs: ' + name)
    parent = auth['epoch']['controller_identity']
    finished = threading.Event()
    threading.Thread(target=parent_watchdog, args=(parent, finished), daemon=True).start()
    # The watchdog precedes READY, GO, frozen imports and all expensive checks.
    try:
        mine = process_identity(os.getpid())
        expected = protocol_binding(auth)
        validate_ready(dict(expected, worker_identity=mine), expected, os.getpid())
        protocol = Path(epoch_path).parent / 'workers' / key.replace('/', '__')
        publish_once(protocol / 'READY.json', dict(expected, worker_identity=mine))
        deadline = time.monotonic() + 120
        while not (protocol / 'GO.json').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('worker GO timed out; attempt consumed')
            time.sleep(0.05)
        go = read_json(protocol / 'GO.json')
        validate_ready(go, expected, os.getpid())
        # Recheck unique claim and parent after the handshake, before imports.
        authorize_worker(manifest, digest, key, epoch_path, epoch_sha)
        legacy = load_legacy(legacy_root)
        jobs = legacy.load_jobs(manifest, digest, code_roots)
        row, config, config_path, job_root = next(job for job in jobs if job[0]['job_key'] == key)
        if (row != auth['row'] or config != auth['config'] or
                Path(config_path).read_bytes() != auth['config_path'].read_bytes()):
            raise ValueError('legacy validated job differs from authorized original config')
        run = Path(job_root) / 'attempt-01'
        run.mkdir(parents=True, exist_ok=False)
        for name in ('APPROVED_CONFIG.json', 'config.json'):
            (run / name).write_bytes(Path(config_path).read_bytes())
        metadata = dict(expected, manifest_sha256=digest, worker_identity=mine,
                        observed_cuda_uuid_order=observed_cuda_order(auth['binding']['gpu_index']))
        write_once(run / 'JOB_RUNTIME.json', dict(metadata, runtime=legacy.runtime_identity()))
        sys.path.insert(0, str(Path(code_roots[row['arm']]).resolve()))
        from config import (validate_adaptive_head_consolidation,
                            validate_party_kd_variant, validate_sample_replay_only)
        import runner
        args = SimpleNamespace(**config)
        validate_party_kd_variant(config, config['expected_party_kd_variant'])
        validate_sample_replay_only(args)
        validate_adaptive_head_consolidation(args)
        if args.formal_deferred_evaluation is not True:
            raise ValueError('formal deferred evaluation is disabled')
        runner.run_experiment(args)
        sealed = legacy.verify_formal_seal(run)
        write_once(run / 'JOB_SUCCESS.json', dict(metadata, status='success', **sealed,
                                                 runtime=legacy.runtime_identity()))
    finally:
        finished.set()


def settle_worker(entry, legacy, cancelled=False):
    process, row = entry['process'], entry['row']
    returncode = process.wait()
    log, root = entry['log'], entry['job_root']
    text = log.read_text(encoding='utf-8', errors='replace') if log.exists() else ''
    gate = root / 'attempt-01/JOB_SUCCESS.json'
    freeze_count = (len(re.findall(r'FIM applied freeze: 0/\d+ bottom elements '
                                   r'\(fraction=0\.000000\)', text))
                    if row['arm'] == 'fim_off' else None)
    valid = returncode == 0 and gate.is_file()
    seal_error = None
    if valid:
        try:
            sealed = legacy.verify_formal_seal(root / 'attempt-01')
            payload = read_json(gate)
            if any(payload.get(k) != v for k, v in sealed.items()):
                raise ValueError('JOB_SUCCESS differs from original formal seal')
        except Exception as error:
            valid, seal_error = False, str(error)
    if valid and (freeze_count is None or freeze_count == entry['config']['num_tasks']):
        classification = 'success'
    elif cancelled and returncode != 0:
        classification = 'cancelled_by_peer_failure'
    elif 'out of memory' in text.lower() or returncode in (-9, 137):
        classification = 'infrastructure_candidate'
    else:
        classification = 'training_or_audit_failure'
    return dict(kind='finish', job_key=row['job_key'], attempt=1,
                classification=classification, returncode=returncode,
                log_sha256=sha(log) if log.exists() else None,
                job_success_sha256=sha(gate) if classification == 'success' else None,
                test_accessed=(root / 'attempt-01/formal_access/test.consumed.json').exists(),
                fim_zero_freeze_logs=freeze_count, seal_error=seal_error,
                time_utc=now(), **entry['binding'])


def _pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    function = ctypes.CDLL(None, use_errno=True).pidfd_open
    function.argtypes, function.restype = (ctypes.c_int, ctypes.c_uint), ctypes.c_int
    fd = function(pid, 0)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def session_members(session):
    result = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                raw = (path / 'stat').read_text()
                fields = raw[raw.rfind(')') + 2:].split()
                if int(fields[3]) != session or fields[0] in ('Z', 'X'):
                    continue
            except (OSError, ValueError, IndexError):
                continue
            try:
                current = process_identity(int(path.name))
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (OSError, ValueError) as exc:
                result.append({'pid': int(path.name), 'identity_error': str(exc)})
                continue
            if current['session'] == session and current['state'] not in ('Z', 'X'):
                result.append(current)
    return result


def stop_owned_session(entry):
    """Kill individually bound members, including loaders after leader death."""
    process, expected = entry['process'], entry.get('identity')
    try:
        leader = process_identity(process.pid)
    except (FileNotFoundError, ProcessLookupError):
        leader = None
    if expected is None:
        if (leader is None or leader['ppid'] != os.getpid() or
                leader['session'] != process.pid or leader['pgrp'] != process.pid):
            raise ValueError('cannot bind newly spawned worker session for cleanup')
        expected = entry['identity'] = leader
    if leader is not None and leader['start_ticks'] != expected['start_ticks']:
        raise ValueError('refusing to signal a reused worker session ID')
    for sweep in range(50):
        members = session_members(process.pid)
        if not members:
            return
        for member in members:
            if 'identity_error' in member:
                continue
            if member['start_ticks'] < expected['start_ticks'] or member['boot_id'] != expected['boot_id']:
                raise ValueError('worker session contains an unbound process')
            try:
                fd = _pidfd_open(member['pid'])
                try:
                    current = process_identity(member['pid'])
                    # Killing a leader reparents its loaders. Every immutable
                    # identity field stays binding; only ppid may change here.
                    if not identities_match(dict(member, ppid=current['ppid']), current):
                        continue
                    signal.pidfd_send_signal(fd, signal.SIGKILL)
                finally:
                    os.close(fd)
            except (OSError, ValueError):
                pass
        time.sleep(0.02)
    raise RuntimeError('worker session did not stop')


def stop_all_owned(entries):
    errors = []
    for entry in entries:
        try:
            stop_owned_session(entry)
        except Exception as exc:
            errors.append(str(exc))
    return errors


def verify_final_ledger(root, epoch):
    raw = (Path(root) / 'ATTEMPTS.jsonl').read_bytes()
    length = epoch['ledger_prefix_bytes_length']
    if (not raw.endswith(b'\n') or hashlib.sha256(raw[:length]).hexdigest() != epoch['ledger_prefix_sha256']):
        raise ValueError('original ledger prefix changed before completion')
    events = [json.loads(line) for line in raw.splitlines()]
    keys = set(epoch['success_keys']) | set(epoch['assignments'])
    by_key = {key: [] for key in keys}
    for event in events:
        if event.get('job_key') not in by_key or event.get('attempt') != 1:
            raise ValueError('unknown final ledger event')
        by_key[event['job_key']].append(event)
    if len(keys) != 18 or len(events) != 36:
        raise ValueError('final ledger is not eighteen one-attempt jobs')
    for key, own in by_key.items():
        if (len(own) != 2 or own[0].get('kind') != 'start' or own[1].get('kind') != 'finish' or
                own[1].get('classification') != 'success' or own[1].get('returncode') != 0):
            raise ValueError('final ledger contains an unsettled or failed job')
    return sha(Path(root) / 'ATTEMPTS.jsonl')


def controller_loop(jobs, legacy, root, epoch_path, epoch_sha, poll_seconds=0.1):
    """One serial event loop: settle exits before every claim and every GO."""
    epoch = read_json(epoch_path)
    parent = epoch['controller_identity']
    pending = {i: sorted((job for job in jobs if epoch['assignments'].get(job[0]['job_key']) == i),
                         key=lambda job: job[0]['job_key']) for i in (0, 1)}
    active, outcomes, since = {}, {}, {0: None, 1: None}
    failed = False

    def drain():
        nonlocal failed
        for index, entry in list(active.items()):
            if entry['process'].poll() is not None:
                # A dead leader may have left live data-loader descendants.
                stop_owned_session(entry)
                event = settle_worker(entry, legacy, cancelled=failed)
                append_event(root, event)
                outcomes[event['job_key']] = event['classification']
                del active[index]
                if event['classification'] != 'success':
                    failed = True
        return failed

    error = None
    try:
        while active or any(pending.values()):
            if drain():
                break
            for index in (0, 1):
                if index in active:
                    entry = active[index]
                    protocol = entry['protocol']
                    if not entry['go'] and (protocol / 'READY.json').exists():
                        mine = validate_ready(read_json(protocol / 'READY.json'),
                                              entry['protocol_binding'], entry['process'].pid)
                        if drain():
                            break
                        if index not in active:
                            continue
                        entry['identity'] = mine
                        publish_once(protocol / 'GO.json', dict(entry['protocol_binding'], worker_identity=mine))
                        entry['go'] = True
                    elif not entry['go'] and time.monotonic() - entry['started'] > 120:
                        raise RuntimeError('READY timed out; attempt consumed')
                    continue
                if not pending[index]:
                    continue
                snapshot = legacy.gpu_snapshot(index)
                if drain():
                    break
                if snapshot['uuid'] != GPU_UUID[index]:
                    raise ValueError('GPU physical identity changed')
                t = time.monotonic()
                since[index] = ((t if since[index] is None else since[index])
                                if legacy.gpu_eligible(snapshot, index) else None)
                if since[index] is None or t - since[index] < 120:
                    continue
                immediate = legacy.gpu_snapshot(index)
                if drain():
                    break
                if not legacy.gpu_eligible(immediate, index):
                    since[index] = None
                    continue
                row, config, config_path, job_root = pending[index].pop(0)
                job_root = Path(job_root)
                if config['device'] != 'cuda:0' or sha(config_path) != row['config_sha256']:
                    raise ValueError('original config changed before launch')
                binding = execution_binding(row, index, epoch_sha, parent)
                job_root.mkdir(parents=True, exist_ok=False)
                write_once(job_root / 'attempt-01.plan.json', {'job_key': row['job_key'], 'attempt': 1})
                protocol = Path(epoch_path).parent / 'workers' / row['job_key'].replace('/', '__')
                protocol.mkdir(parents=True, exist_ok=False)
                if drain():
                    break
                append_event(root, dict(kind='start', job_key=row['job_key'], attempt=1,
                    config_sha256=row['config_sha256'], code_commit=row['code_commit'], time_utc=now(), **binding))
                log = job_root / 'attempt-01.log'
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
                    '--manifest', str(epoch['manifest']), '--manifest-sha256', epoch['manifest_sha256'],
                    '--legacy-root', epoch['legacy_root'], '--base-code', epoch['code_roots']['no_partykd'],
                    '--patch-code', epoch['code_roots']['fim_off'], '--orchestration-commit', epoch['orchestration_commit'],
                    '--epoch', str(epoch_path), '--epoch-sha256', epoch_sha, '--job-key', row['job_key']]
                try:
                    if drain():
                        raise RuntimeError('peer failed before worker launch; attempt consumed')
                    with log.open('x', encoding='utf-8') as stream:
                        process = subprocess.Popen(command, cwd=epoch['code_roots'][row['arm']],
                            env=worker_environment(config, index), stdout=stream,
                            stderr=subprocess.STDOUT, start_new_session=True)
                except Exception as exc:
                    append_event(root, dict(kind='finish', job_key=row['job_key'], attempt=1,
                        classification='infrastructure_candidate', returncode=None,
                        log_sha256=sha(log) if log.exists() else None, job_success_sha256=None,
                        test_accessed=False, launch_error=str(exc), time_utc=now(), **binding))
                    outcomes[row['job_key']] = 'infrastructure_candidate'
                    failed = True
                    break
                auth = dict(row=row, binding=binding)
                active[index] = dict(process=process, row=row, config=config, job_root=job_root,
                    log=log, binding=binding, protocol=protocol, protocol_binding=protocol_binding(auth),
                    identity=None, go=False, started=time.monotonic())
                active[index]['identity'] = process_identity(process.pid)
                since[index] = None
            if failed:
                break
            time.sleep(poll_seconds)
    except BaseException as exc:
        failed, error = True, str(exc)
        raise
    finally:
        if active:
            failed = True
            cleanup_errors = stop_all_owned(list(active.values()))
            error = '; '.join(filter(None, [error] + cleanup_errors)) or None
            for entry in list(active.values()):
                # If identity is unverifiable, exiting the parent activates the
                # worker watchdog. Never hang here waiting on an unverified PID.
                if entry['process'].poll() is not None:
                    event = settle_worker(entry, legacy, cancelled=True)
                    append_event(root, event)
                    outcomes[event['job_key']] = event['classification']
            active.clear()
        success = len(epoch['success_keys']) + sum(v == 'success' for v in outcomes.values())
        complete = not failed and success == 18 and len(epoch['success_keys']) + len(outcomes) == 18
        if complete:
            try:
                verify_final_ledger(root, epoch)
            except Exception as exc:
                complete, error = False, str(exc)
        record = dict(status='complete' if complete else 'failed_or_interrupted',
            manifest_sha256=epoch['manifest_sha256'], successes=success,
            accounted_for=len(epoch['success_keys']) + len(outcomes), attempts_sha256=sha(Path(root) / 'ATTEMPTS.jsonl'),
            orchestration_commit=epoch['orchestration_commit'], original_orchestration_commit=ORIGIN_COMMIT,
            execution_epoch_sha256=epoch_sha, error=error, ended_utc=now())
        write_once(Path(root) / ('QUEUE_COMPLETE.json' if complete else 'QUEUE_FAILURE.json'), record)
    if not complete:
        raise RuntimeError('ablation queue stopped; preserve all evidence')
    return record


def run_continuation(manifest, digest, legacy_root, code_roots, orchestration_commit,
                     old_identity_path, old_identity_sha256):
    from frozen_ablation_quiescent import ledger_state, try_quiescent_fence, assign_pending
    pinned_checkout(Path(__file__).resolve().parent, orchestration_commit)
    if digest != MANIFEST_SHA:
        raise ValueError('continuation requires the original externally pinned manifest')
    legacy = load_legacy(legacy_root)
    jobs = legacy.load_jobs(manifest, digest, code_roots)
    rows = [job[0] for job in jobs]
    root = Path(legacy.RESULT_ROOT)
    if shutil.disk_usage(root).free < 40 * (1 << 30):
        raise RuntimeError('formal ablation needs at least 40 GiB free before continuation')
    if any((root / name).exists() for name in ('QUEUE_COMPLETE.json', 'QUEUE_FAILURE.json')):
        raise ValueError('queue already ended')
    state = ledger_state(root, rows)
    pending_candidates(state['pending'])
    assignments = assign_pending(state['pending'])
    if len(set(assignments.values())) < 2:
        return {'status': 'not_needed', 'reason': 'fewer than two usable pending lanes'}
    lock = read_json(root / 'QUEUE_LOCK.json')
    if sha(old_identity_path) != old_identity_sha256:
        raise ValueError('external old process identity anchor changed')
    old = read_json(old_identity_path)
    if old['pid'] != lock['pid']:
        raise ValueError('old queue lock PID differs from external identity anchor')
    validate_old_parent(old, legacy_root, manifest, code_roots)
    if lock.get('manifest_sha256') != digest or lock.get('orchestration_commit') != ORIGIN_COMMIT:
        raise ValueError('original queue lock differs')
    extension = root / EXTENSION
    extension.mkdir(exist_ok=False)
    write_once(extension / 'ARMED.json', dict(schema_version=1, manifest_sha256=digest,
        candidate_cap=sorted(CANDIDATE_CAP), initial_pending=state['pending'], old_identity=old,
        old_identity_anchor_sha256=old_identity_sha256,
        ledger_sha256=state['raw_sha256'], ledger_bytes_length=state['raw_bytes_length'],
        orchestration_commit=orchestration_commit, original_orchestration_commit=ORIGIN_COMMIT,
        orchestration_files_sha256={name: sha(Path(__file__).resolve().parent / name) for name in
            ('frozen_ablation_gpu1_continue.py', 'frozen_ablation_quiescent.py', 'frozen_ablation_audit.py')},
        training_commits={'no_partykd': legacy.BASE_COMMIT, 'fim_off': legacy.PATCH_COMMIT},
        gpu_uuid={str(i): GPU_UUID[i] for i in (0, 1)}, controller_identity=process_identity(os.getpid()),
        legacy_root=str(Path(legacy_root).resolve()), code_roots={k: str(v) for k, v in code_roots.items()},
        allocation='cifar43_gpu1_else_isolet_alternating_0_1', armed_utc=now()))
    while True:
        state = ledger_state(root, rows)
        pending_candidates(state['pending'])
        assignments = assign_pending(state['pending'])
        if len(set(assignments.values())) < 2:
            return {'status': 'not_needed', 'reason': 'old queue consumed remaining parallel work'}
        receipt = try_quiescent_fence(root, rows, old)
        if receipt is not None:
            break
        time.sleep(1)
    pending_candidates(receipt['pending_keys'])
    assignments = assign_pending(receipt['pending_keys'])
    if len(set(assignments.values())) < 2:
        raise RuntimeError('retirement receipt lost the two-lane pending guarantee')
    epoch_path = extension / 'EXECUTION_EPOCH.json'
    epoch = dict(receipt, schema_version=1, manifest=str(Path(manifest).resolve()),
        manifest_sha256=digest, assignments=assignments, controller_identity=process_identity(os.getpid()),
        orchestration_commit=orchestration_commit, original_orchestration_commit=ORIGIN_COMMIT,
        armed_sha256=sha(extension / 'ARMED.json'), legacy_root=str(Path(legacy_root).resolve()),
        code_roots={k: str(v) for k, v in code_roots.items()}, created_utc=now())
    write_once(epoch_path, epoch)
    return controller_loop(jobs, legacy, root, epoch_path, sha(epoch_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--legacy-root', type=Path, required=True)
    parser.add_argument('--base-code', type=Path, required=True)
    parser.add_argument('--patch-code', type=Path, required=True)
    parser.add_argument('--orchestration-commit', required=True)
    parser.add_argument('--old-identity', type=Path)
    parser.add_argument('--old-identity-sha256')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--job-key')
    parser.add_argument('--epoch', type=Path)
    parser.add_argument('--epoch-sha256')
    args = parser.parse_args()
    if args.manifest_sha256 != MANIFEST_SHA:
        raise ValueError('manifest differs from original external anchor')
    roots = {'no_partykd': args.base_code.resolve(), 'fim_off': args.patch_code.resolve()}
    if args.worker:
        if not args.job_key or not args.epoch or not args.epoch_sha256:
            raise ValueError('worker needs complete epoch authorization')
        epoch = read_json(args.epoch)
        if epoch.get('orchestration_commit') != args.orchestration_commit:
            raise ValueError('worker orchestration producer differs')
        pinned_checkout(Path(__file__).resolve().parent, args.orchestration_commit)
        worker(args.manifest, args.manifest_sha256, args.job_key, args.epoch,
               args.epoch_sha256, args.legacy_root, roots)
    else:
        if args.job_key or args.epoch or args.epoch_sha256:
            raise ValueError('controller cannot select an individual job or adopt an epoch')
        if not args.old_identity or not args.old_identity_sha256:
            raise ValueError('controller needs an externally anchored old process identity')
        print(json.dumps(run_continuation(args.manifest.resolve(), args.manifest_sha256,
              args.legacy_root.resolve(), roots, args.orchestration_commit,
              args.old_identity, args.old_identity_sha256), sort_keys=True))


if __name__ == '__main__':
    main()
