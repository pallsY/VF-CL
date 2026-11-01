"""One-attempt, fail-stop queue for the frozen method's formal ablations."""

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from frozen_ablation_manifest import (
    AUX_DATA_SHA, BASE_COMMIT, DATA_SHA, PATCH_COMMIT, RESULT_ROOT, SOURCE_ROOT,
    SOURCE_ANCHOR_SHA,
    _sha, auxiliary_data_paths, derive_config, expected_keys,
    patch_diff_sha256, runtime_identity, semantic_diff,
    validate_smoke_comparison, _bytes,
)

GPU_UUID = {
    0: 'GPU-6c7fee92-a6fe-5525-c554-bddb8e14943f',
    1: 'GPU-6569e526-43ea-74e1-264e-4b66417181e1',
}
MIN_FREE_MIB = {0: 3072, 1: 5120}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_once(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def gpu_eligible(snapshot, index):
    return (snapshot['index'] == index and
            snapshot['name'] == 'NVIDIA GeForce RTX 3080' and
            snapshot['uuid'] == GPU_UUID[index] and
            snapshot['free_mib'] >= MIN_FREE_MIB[index])


def gpu_snapshot(index):
    output = subprocess.run([
        'nvidia-smi', '-i', str(index),
        '--query-gpu=index,name,uuid,memory.free',
        '--format=csv,noheader,nounits',
    ], capture_output=True, text=True, check=True).stdout.strip()
    fields = [part.strip() for part in output.split(',')]
    if len(fields) != 4:
        raise ValueError('unexpected GPU status format')
    return dict(index=int(fields[0]), name=fields[1], uuid=fields[2],
                free_mib=int(fields[3]))


def wait_headroom(index, stop):
    since = None
    while not stop.is_set():
        snapshot = gpu_snapshot(index)
        if snapshot['uuid'] != GPU_UUID[index]:
            raise ValueError('GPU physical identity changed')
        t = time.monotonic()
        since = (t if since is None else since) if gpu_eligible(snapshot, index) else None
        if since is not None and t - since >= 120:
            immediate = gpu_snapshot(index)
            if gpu_eligible(immediate, index):
                return immediate
            since = None
        stop.wait(20)
    return None


def verify_formal_seal(run):
    run = Path(run)
    paths = {
        'results_sha256': run / 'results.json',
        'checkpoint_sha256': run / 'checkpoints/formal_final.pt',
        'formal_complete_sha256': run / 'FORMAL_EVALUATION_COMPLETE.json',
        'formal_sealed_sha256': run / 'FORMAL_EVALUATION_SEALED.json',
        'formal_published_sha256': run / 'FORMAL_EVALUATION_PUBLISHED.json',
        'test_access_sha256': run / 'formal_access/test.consumed.json',
    }
    if any(not path.is_file() for path in paths.values()):
        raise ValueError('formal result or single test-access seal is absent')
    complete = json.loads(paths['formal_complete_sha256'].read_text())
    sealed = json.loads(paths['formal_sealed_sha256'].read_text())
    published = json.loads(paths['formal_published_sha256'].read_text())
    access = json.loads(paths['test_access_sha256'].read_text())
    if (complete.get('status') != 'complete' or
            sealed.get('status') != 'sealed' or
            published.get('status') != 'published' or
            published.get('results_sha256') != _sha(paths['results_sha256']) or
            access.get('status') != 'consumed' or
            access.get('access', {}).get('split') != 'test' or
            access.get('access', {}).get('event') != 'first_iteration'):
        raise ValueError('formal test did not seal one authorized access')
    return {key: _sha(path) for key, path in paths.items()}


def code_identity(path, expected):
    path = Path(path).resolve()
    head = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                                   text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'],
                                    text=True).strip()
    if head != expected or dirty:
        raise ValueError('training checkout differs from the frozen code commit')
    return path


def verify_preflight(manifest, rows):
    manifest = Path(manifest).resolve()
    evidence = manifest.parent / 'preflight'
    record_path = evidence / 'PRELAUNCH_PREFLIGHT.json'
    smoke_path = evidence / 'SMOKE_COMPARISON.json'
    review_path = evidence / 'PATCH_REVIEW.json'
    if any(not path.is_file() for path in (record_path, smoke_path, review_path)):
        raise ValueError('preflight evidence is missing')
    values = {name: {row.get(name) for row in rows} for name in (
        'preflight_sha256', 'smoke_comparison_sha256',
        'patch_review_sha256', 'patch_diff_sha256')}
    if any(len(value) != 1 for value in values.values()):
        raise ValueError('preflight identity differs across manifest rows')
    record = json.loads(record_path.read_text())
    review = json.loads(review_path.read_text())
    comparison = json.loads(smoke_path.read_text())
    validate_smoke_comparison(comparison)
    from frozen_ablation_smoke import validate_smoke_records
    validate_smoke_records(evidence / 'smokes', comparison)
    preflight_sha = next(iter(values['preflight_sha256']))
    smoke_sha = next(iter(values['smoke_comparison_sha256']))
    review_sha = next(iter(values['patch_review_sha256']))
    diff_sha = next(iter(values['patch_diff_sha256']))
    if (preflight_sha != _sha(record_path) or
            smoke_sha != _sha(smoke_path) or
            review_sha != _sha(review_path) or
            diff_sha != patch_diff_sha256() or
            record.get('status') != 'approved' or
            record.get('schema_version') != 1 or
            record.get('reviewer_id') != review.get('reviewer_id') or
            record.get('base_commit') != BASE_COMMIT or
            record.get('patch_commit') != PATCH_COMMIT or
            record.get('patch_diff_sha256') != diff_sha or
            record.get('smoke_comparison_sha256') != smoke_sha or
            record.get('patch_review_sha256') != review_sha or
            review.get('verdict') != 'approved' or
            review.get('base_commit') != BASE_COMMIT or
            review.get('patch_commit') != PATCH_COMMIT or
            review.get('patch_diff_sha256') != diff_sha or
            review.get('blocking_findings') != []):
        raise ValueError('preflight evidence does not match the frozen code and manifest')
    return record


def fail_stop_peers(active, failed_index, stop):
    stop.set()
    signaled = False
    for lane_index, process in active.items():
        if lane_index != failed_index and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                signaled = True
            except ProcessLookupError:
                pass
    return signaled


def load_jobs(manifest, digest, code_roots):
    manifest = Path(manifest).resolve()
    if (_sha(manifest) != digest or
            manifest != Path(RESULT_ROOT) / 'manifests/manifest.jsonl'):
        raise ValueError('manifest path or external SHA differs')
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if len(rows) != 18 or {row.get('job_key') for row in rows} != expected_keys():
        raise ValueError('formal ablation allowlist is incomplete or duplicated')
    verify_preflight(manifest, rows)
    code_identity(Path(__file__).resolve().parent,
                  rows[0]['orchestration_commit'])
    if _sha(SOURCE_ROOT / 'audit/ADAPTIVE_SAMPLE_ONLY_FINAL_ANCHOR.json') != SOURCE_ANCHOR_SHA:
        raise ValueError('original formal anchor changed after manifest generation')
    jobs = []
    data_cache = {}
    current_runtime = runtime_identity()
    for row in rows:
        dataset, arm, seed_text = row['job_key'].split('/')
        seed = int(seed_text)
        expected_commit = BASE_COMMIT if arm == 'no_partykd' else PATCH_COMMIT
        if (row.get('dataset') != dataset or row.get('arm') != arm or
                row.get('seed') != seed or
                row.get('code_commit') != expected_commit or
                row.get('orchestration_commit') != rows[0]['orchestration_commit'] or
                row.get('source_commit') != BASE_COMMIT or
                row.get('source_formal_anchor_sha256') != SOURCE_ANCHOR_SHA or
                row.get('gpu_uuid') != GPU_UUID[row['physical_gpu']]):
            raise ValueError('manifest row identity differs')
        code_identity(code_roots[arm], expected_commit)
        source_path = (SOURCE_ROOT / 'full' / dataset / 'adaptive' /
                       str(seed) / 'run/config.json')
        if _sha(source_path) != row['source_config_sha256']:
            raise ValueError('original formal source config changed')
        source = json.loads(source_path.read_text())
        original_success = json.loads(
            (source_path.parent.parent / 'JOB_SUCCESS.json').read_text())
        formal_runtime = original_success.get('runtime')
        runtime_sha = hashlib.sha256(_bytes(formal_runtime)).hexdigest()
        if (row.get('runtime') != formal_runtime or
                row.get('runtime_sha256') != runtime_sha or
                current_runtime != formal_runtime):
            raise ValueError('training runtime differs from original formal job')
        data_path = (Path(source['data_path']) / 'cifar-100-python/train'
                     if dataset == 'cifar100' else Path(source['vector_npz']))
        if data_path not in data_cache:
            data_cache[data_path] = _sha(data_path)
        if (data_cache[data_path] != DATA_SHA[dataset] or
                row.get('source_data_sha256') != data_cache[data_path]):
            raise ValueError('original formal training data changed')
        if row.get('auxiliary_data_sha256') != AUX_DATA_SHA[dataset]:
            raise ValueError('formal auxiliary data anchor differs')
        for name, path in auxiliary_data_paths(source, dataset).items():
            if path not in data_cache:
                data_cache[path] = _sha(path)
            if data_cache[path] != AUX_DATA_SHA[dataset][name]:
                raise ValueError(f'original formal auxiliary data changed: {name}')
        config_path = (manifest.parent / row['config_relpath']).resolve()
        if (not config_path.is_relative_to(manifest.parent / 'configs') or
                _sha(config_path) != row['config_sha256']):
            raise ValueError('ablation config SHA differs')
        config = json.loads(config_path.read_text())
        if (config != derive_config(source, dataset, arm, seed) or
                row['config_semantic_diff'] != semantic_diff(source, config) or
                row['physical_gpu'] != int(config['device'][-1])):
            raise ValueError('ablation config has an unapproved change')
        job_root = Path(config['output_dir']).parent
        if not job_root.is_relative_to(Path(RESULT_ROOT)):
            raise ValueError('ablation result path escaped the new root')
        jobs.append((row, config, config_path, job_root))
    return jobs


def worker(manifest, digest, key, code_root, code_roots):
    jobs = load_jobs(manifest, digest, code_roots)
    row, config, config_path, job_root = next(
        (job for job in jobs if job[0]['job_key'] == key), (None, None, None, None))
    if row is None or Path(code_root).resolve() != Path(code_roots[row['arm']]).resolve():
        raise ValueError('worker code/job binding differs')
    lock = json.loads((Path(RESULT_ROOT) / 'QUEUE_LOCK.json').read_text())
    ledger = [json.loads(line) for line in
              (Path(RESULT_ROOT) / 'ATTEMPTS.jsonl').read_text().splitlines()
              if line]
    own = [event for event in ledger if event.get('job_key') == key]
    plan = json.loads((job_root / 'attempt-01.plan.json').read_text())
    if (lock.get('pid') != os.getppid() or
            lock.get('manifest_sha256') != digest or
            plan != {'job_key': key, 'attempt': 1} or
            len(own) != 1 or own[0].get('kind') != 'start' or
            own[0].get('attempt') != 1 or
            own[0].get('config_sha256') != row['config_sha256']):
        raise ValueError('worker is not one authorized queued attempt')
    run = job_root / 'attempt-01'
    run.mkdir(parents=True, exist_ok=False)
    (run / 'APPROVED_CONFIG.json').write_bytes(config_path.read_bytes())
    (run / 'config.json').write_bytes(config_path.read_bytes())
    write_once(run / 'JOB_RUNTIME.json', {
        'job_key': key, 'attempt': 1, 'manifest_sha256': digest,
        'config_sha256': row['config_sha256'],
        'training_code_commit': row['code_commit'],
        'runtime': runtime_identity(),
    })
    sys.path.insert(0, str(Path(code_root).resolve()))
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
    sealed = verify_formal_seal(run)
    write_once(run / 'JOB_SUCCESS.json', {
        'status': 'success', 'job_key': key, 'attempt': 1,
        'manifest_sha256': digest, 'config_sha256': row['config_sha256'],
        'training_code_commit': row['code_commit'], **sealed,
        'runtime': runtime_identity(),
    })


def run_queue(manifest, digest, code_roots):
    jobs = load_jobs(manifest, digest, code_roots)
    root = Path(RESULT_ROOT)
    if shutil.disk_usage(root).free < 40 * (1 << 30):
        raise RuntimeError('formal ablation needs at least 40 GiB free before launch')
    if any((root / name).exists() for name in
           ('QUEUE_LOCK.json', 'ATTEMPTS.jsonl', 'QUEUE_COMPLETE.json',
            'QUEUE_FAILURE.json')):
        raise FileExistsError('ablation queue already launched')
    write_once(root / 'QUEUE_LOCK.json', {
        'pid': os.getpid(), 'manifest_sha256': digest,
        'orchestration_commit': jobs[0][0]['orchestration_commit'],
        'status': 'running', 'started_utc': now(),
    })
    with (root / 'ATTEMPTS.jsonl').open('x', encoding='utf-8'):
        pass
    mutex, stop, settling = (threading.Lock(), threading.Event(),
                             threading.Event())
    settlement_lock = threading.Lock()
    settling_lanes = set()
    outcomes, active = {}, {}

    def begin_settlement(lane_index):
        with settlement_lock:
            settling_lanes.add(lane_index)
            settling.set()

    def end_settlement(lane_index):
        with settlement_lock:
            settling_lanes.discard(lane_index)
            if not settling_lanes:
                settling.clear()

    def stop_active(except_index=None):
        fail_stop_peers(active, except_index, stop)

    def append_unlocked(value):
        with (root / 'ATTEMPTS.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def append(value):
        with mutex:
            append_unlocked(value)

    def lane(index):
        assigned = sorted((job for job in jobs if job[0]['physical_gpu'] == index),
                          key=lambda job: job[0]['job_key'])
        for row, config, _, job_root in assigned:
            if stop.is_set():
                return
            admission = wait_headroom(index, stop)
            if admission is None:
                return
            key = row['job_key']
            log = job_root / 'attempt-01.log'
            env = os.environ.copy()
            env.update({'CUDA_VISIBLE_DEVICES': ','.join(GPU_UUID[i] for i in (0, 1)),
                        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                        'PYTHONHASHSEED': str(config['seed']),
                        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
            command = [sys.executable, '-u', str(Path(__file__).resolve()),
                       '--worker', '--manifest', str(manifest),
                       '--manifest-sha256', digest, '--job-key', key,
                       '--base-code', str(code_roots['no_partykd']),
                       '--patch-code', str(code_roots['fim_off']),
                       '--code-root', str(code_roots[row['arm']])]
            while True:
                waiting_for_settlement = False
                with mutex:
                    if stop.is_set():
                        return
                    for active_lane, active_process in active.items():
                        if active_process.poll() is not None:
                            begin_settlement(active_lane)
                    if settling.is_set():
                        waiting_for_settlement = True
                    else:
                        snapshot = gpu_snapshot(index)
                        if stop.is_set() or settling.is_set():
                            waiting_for_settlement = True
                        elif not gpu_eligible(snapshot, index):
                            raise RuntimeError('GPU headroom disappeared at launch')
                        else:
                            job_root.mkdir(parents=True, exist_ok=False)
                            write_once(job_root / 'attempt-01.plan.json',
                                       {'job_key': key, 'attempt': 1})
                            append_unlocked({'kind': 'start', 'job_key': key,
                                             'attempt': 1,
                                             'config_sha256': row['config_sha256'],
                                             'code_commit': row['code_commit'],
                                             'gpu_index': index,
                                             'gpu_uuid': GPU_UUID[index],
                                             'time_utc': now()})
                            try:
                                with log.open('x', encoding='utf-8') as stream:
                                    process = subprocess.Popen(
                                        command, cwd=code_roots[row['arm']],
                                        env=env, stdout=stream,
                                        stderr=subprocess.STDOUT,
                                        start_new_session=True)
                                    active[index] = process
                            except Exception as exc:
                                outcomes[key] = 'infrastructure_candidate'
                                stop_active(except_index=index)
                                append_unlocked({
                                    'kind': 'finish', 'job_key': key,
                                    'attempt': 1,
                                    'classification': 'infrastructure_candidate',
                                    'returncode': None,
                                    'log_sha256': _sha(log) if log.is_file() else None,
                                    'job_success_sha256': None,
                                    'test_accessed': False,
                                    'launch_error': str(exc),
                                    'time_utc': now(),
                                })
                                return
                if not waiting_for_settlement:
                    break
                stop.wait(0.05)
            process.wait()
            begin_settlement(index)
            try:
                stopped_before_exit = stop.is_set()
                gate = job_root / 'attempt-01/JOB_SUCCESS.json'
                if process.returncode != 0 or not gate.is_file():
                    stop.set()
                with mutex:
                    active.pop(index, None)
                    if stop.is_set():
                        fail_stop_peers(active, index, stop)
                log_text = log.read_text(encoding='utf-8', errors='replace')
                freeze_count = (len(re.findall(
                    r'FIM applied freeze: 0/\d+ bottom elements '
                    r'\(fraction=0\.000000\)', log_text))
                    if row['arm'] == 'fim_off' else None)
                if (process.returncode == 0 and gate.is_file() and
                        (freeze_count is None or
                         freeze_count == config['num_tasks'])):
                    classification = 'success'
                elif stopped_before_exit and process.returncode != 0:
                    classification = 'cancelled_by_peer_failure'
                elif ('out of memory' in log_text.lower() or
                      process.returncode in (-9, 137)):
                    classification = 'infrastructure_candidate'
                else:
                    classification = 'training_or_audit_failure'
                with mutex:
                    outcomes[key] = classification
                    if classification != 'success':
                        stop_active(except_index=index)
                append({'kind': 'finish', 'job_key': key, 'attempt': 1,
                        'classification': classification,
                        'returncode': process.returncode,
                        'log_sha256': _sha(log),
                        'job_success_sha256': (
                            _sha(gate) if classification == 'success' else None),
                        'test_accessed': (job_root /
                            'attempt-01/formal_access/test.consumed.json').exists(),
                        'fim_zero_freeze_logs': freeze_count,
                        'time_utc': now()})
            except BaseException:
                with mutex:
                    stop_active(except_index=index)
                raise
            finally:
                end_settlement(index)
            if classification != 'success':
                return

    error = None
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(lane, index) for index in (0, 1)]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                with mutex:
                    stop_active()
                error = str(exc)
    complete = not stop.is_set() and len(outcomes) == 18 and all(
        value == 'success' for value in outcomes.values())
    record = {'status': 'complete' if complete else 'failed_or_interrupted',
              'manifest_sha256': digest, 'successes': sum(
                  value == 'success' for value in outcomes.values()),
              'orchestration_commit': jobs[0][0]['orchestration_commit'],
              'accounted_for': len(outcomes),
              'attempts_sha256': _sha(root / 'ATTEMPTS.jsonl'),
              'error': error, 'ended_utc': now()}
    write_once(root / ('QUEUE_COMPLETE.json' if complete else 'QUEUE_FAILURE.json'),
               record)
    if not complete:
        raise RuntimeError('ablation queue stopped; preserve all evidence')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--base-code', type=Path, required=True)
    parser.add_argument('--patch-code', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--job-key')
    parser.add_argument('--code-root', type=Path)
    args = parser.parse_args()
    roots = {'no_partykd': args.base_code.resolve(),
             'fim_off': args.patch_code.resolve()}
    manifest = args.manifest.resolve()
    if args.worker:
        if not args.job_key or not args.code_root:
            raise ValueError('worker needs job and code-root binding')
        worker(manifest, args.manifest_sha256, args.job_key,
               args.code_root.resolve(), roots)
    elif args.job_key or args.code_root:
        raise ValueError('queue cannot select a single unlisted job')
    else:
        print(json.dumps(run_queue(manifest, args.manifest_sha256,
                                   roots), sort_keys=True))


if __name__ == '__main__':
    main()
