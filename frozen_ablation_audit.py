"""Independent metadata audit before formal ablation accuracy summarization."""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from frozen_ablation_seal import verify_formal_seal

from frozen_ablation_manifest import (
    AUX_DATA_SHA, BASE_COMMIT, DATASETS, DATA_SHA, PATCH_COMMIT,
    RESULT_ROOT, SEEDS, SOURCE_ANCHOR_SHA, SOURCE_ROOT, SMOKE_PRODUCER_COMMIT,
    auxiliary_data_paths, derive_config, expected_keys,
    original_tree_sha256, runtime_identity, semantic_diff,
    validate_smoke_comparison, validate_publication_preflight, _bytes,
)

GPU_UUID = {
    0: 'GPU-6c7fee92-a6fe-5525-c554-bddb8e14943f',
    1: 'GPU-6569e526-43ea-74e1-264e-4b66417181e1',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_run_gate(run, gate):
    sealed_hashes = verify_formal_seal(run)
    if any(gate.get(field) != value for field, value in sealed_hashes.items()):
        raise ValueError('formal ablation result hashes differ from gate')
    return sealed_hashes


def verify_historical_source(run):
    """Use only after the entire historical tree matches its pinned final audit.

    NFS mount device numbers can change between sessions; all other file
    identity fields and both result digest definitions remain binding.
    """
    import adaptive_consolidation_audit as producer
    run = Path(run)
    published = producer._safe_json(run / 'FORMAL_EVALUATION_PUBLISHED.json')
    for name in ('results', 'checkpoint'):
        expected = published[name]
        path = (run / expected['path']).resolve()
        if not path.is_relative_to(run.resolve()):
            raise ValueError('historical artifact escaped anchored root')
        actual = producer._formal_file_record(path, run)
        if {k: v for k, v in actual.items() if k != 'device'} != {
                k: v for k, v in expected.items() if k != 'device'}:
            raise ValueError('historical artifact file identity changed')
    results = producer._safe_json(run / 'results.json')
    content_sha = hashlib.sha256(producer._strict_json(results).encode('utf-8')).hexdigest()
    if content_sha != published['results_sha256']:
        raise ValueError('historical result content changed')
    return content_sha


def party_kd_contribution_label(deltas):
    if set(deltas) != set(DATASETS):
        raise ValueError('PartyKD requires three dataset means')
    return ('supported' if sum(delta > 0 for delta in deltas.values()) >= 2
            else 'inconsistent')


def verify_execution_extension(root, rows, manifest_sha, path, external_sha):
    """Validate an explicitly anchored execution epoch, never select it implicitly."""
    root, path = Path(root).resolve(), Path(path).resolve()
    origin = 'f4735cacf51c41d63a9f87e8933c968cb268637b'
    candidates = {'cifar100/no_partykd/43'} | {
        f'isolet/{arm}/{seed}' for arm in ('no_partykd', 'fim_off') for seed in (42, 43, 44)}
    if (path != root / 'gpu1-rebalance-v1/EXECUTION_EPOCH.json' or
            path.is_symlink() or sha(path) != external_sha):
        raise ValueError('execution extension path or external SHA differs')
    epoch = json.loads(path.read_text())
    armed_path = path.parent / 'ARMED.json'
    armed = json.loads(armed_path.read_text())
    raw = (root / 'ATTEMPTS.jsonl').read_bytes()
    length, initial_length = epoch.get('ledger_prefix_bytes_length'), armed.get('ledger_bytes_length')
    if (type(length) is not int or type(initial_length) is not int or
            initial_length < 0 or length < initial_length or length > len(raw) or
            (raw and not raw.endswith(b'\n')) or (length and raw[length-1:length] != b'\n') or
            hashlib.sha256(raw[:length]).hexdigest() != epoch.get('ledger_prefix_sha256') or
            hashlib.sha256(raw[:initial_length]).hexdigest() != armed.get('ledger_sha256')):
        raise ValueError('original success prefix or armed ledger bytes changed')
    prefix = [json.loads(line) for line in raw[:length].splitlines()]
    by_key = defaultdict(list)
    for event in prefix:
        if event.get('execution_epoch_sha256') is not None or event.get('attempt') != 1:
            raise ValueError('execution extension polluted the original ledger prefix')
        by_key[event.get('job_key')].append(event)
    for key, events in by_key.items():
        if (len(events) != 2 or events[0].get('kind') != 'start' or events[1].get('kind') != 'finish' or
                events[1].get('classification') != 'success' or events[1].get('returncode') != 0):
            raise ValueError('execution prefix is not wholly successful and settled')
    keys = {row['job_key'] for row in rows}
    assignments = epoch.get('assignments', {})
    pending = keys - set(by_key)
    if (not isinstance(assignments, dict) or not pending <= candidates or
            set(assignments) != pending or set(epoch.get('pending_keys', [])) != pending or
            len(epoch.get('pending_keys', [])) != len(pending) or
            set(epoch.get('success_keys', [])) != set(by_key) or
            len(epoch.get('success_keys', [])) != len(by_key) or not set(by_key) <= keys or
            set(armed.get('candidate_cap', [])) != candidates or
            not pending <= set(armed.get('initial_pending', [])) <= candidates):
        raise ValueError('execution extension expanded or duplicated the allowed pending matrix')
    cifar = 'cifar100/no_partykd/43'
    expected_map = {key: (int(key == cifar) if cifar in pending else i % 2)
                    for i, key in enumerate(sorted(pending))}
    if assignments != expected_map or set(assignments.values()) != {0, 1} or any(type(v) is not int for v in assignments.values()):
        raise ValueError('execution extension physical assignment differs')
    old, retired = epoch.get('old_identity', {}), epoch.get('retired_identity', {})
    parent = epoch.get('controller_identity', {})
    identity_fields = ('pid', 'ppid', 'pgrp', 'session', 'start_ticks', 'boot_id', 'cmdline')
    if (epoch.get('schema_version') != 1 or armed.get('schema_version') != 1 or
            epoch.get('manifest_sha256') != manifest_sha or armed.get('manifest_sha256') != manifest_sha or
            epoch.get('armed_sha256') != sha(armed_path) or
            epoch.get('original_orchestration_commit') != origin or
            armed.get('original_orchestration_commit') != origin or
            not re.fullmatch('[0-9a-f]{40}', epoch.get('orchestration_commit', '')) or
            epoch.get('orchestration_commit') == origin or
            epoch.get('orchestration_commit') != armed.get('orchestration_commit') or
            not epoch.get('legacy_root') or epoch.get('legacy_root') != armed.get('legacy_root') or
            set(epoch.get('code_roots', {})) != {'no_partykd', 'fim_off'} or
            epoch.get('code_roots') != armed.get('code_roots') or
            parent != armed.get('controller_identity') or old != armed.get('old_identity') or
            parent.get('pid') == old.get('pid') or
            any(field not in old or field not in parent for field in identity_fields) or
            any(retired.get(field) != old.get(field) for field in ('pid', 'start_ticks', 'boot_id')) or
            retired.get('state') not in ('Z', 'X', 'gone') or
            not epoch.get('stopped_utc') or not epoch.get('retired_utc') or
            armed.get('gpu_uuid') != {str(i): GPU_UUID[i] for i in (0, 1)}):
        raise ValueError('execution retirement, writer or source identity differs')
    snapshots = epoch.get('stopped_snapshots', [])
    if len(snapshots) != 2:
        raise ValueError('execution retirement lacks two stopped snapshots')
    for snapshot in snapshots:
        observed = snapshot.get('identity', {})
        threads = snapshot.get('threads', [])
        if (any(observed.get(field) != old.get(field) for field in identity_fields) or
                observed.get('state') != 'T' or not threads or
                any(len(thread) != 4 or thread[2] != 'T' or thread[3] or
                    type(thread[0]) is not int or type(thread[1]) is not int or
                    thread[0] <= 0 or thread[1] <= 0 for thread in threads) or
                snapshot.get('ledger_sha256') != epoch['ledger_prefix_sha256'] or
                snapshot.get('ledger_bytes_length') != length or not snapshot.get('observed_utc')):
            raise ValueError('execution stopped snapshot has children, changed identity or ledger')
    if snapshots[0]['threads'] != snapshots[1]['threads']:
        raise ValueError('execution stopped thread inventories differ')
    files = epoch.get('metadata_files_sha256', {})
    expected_paths = {'QUEUE_LOCK.json', 'QUEUE_COMPLETE.json', 'QUEUE_FAILURE.json',
                      'manifests/manifest.jsonl', 'manifests/manifest.sha256',
                      'gpu1-rebalance-v1/ARMED.json'}
    plan_paths = {f'{k.split("/")[1]}/{k.split("/")[0]}/{k.split("/")[2]}/attempt-01.plan.json': k
                  for k in keys}
    expected_paths.update(plan_paths)
    if (set(files) != expected_paths or
            hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest() != epoch.get('metadata_sha256') or
            files.get('manifests/manifest.jsonl') != manifest_sha or
            files.get('gpu1-rebalance-v1/ARMED.json') != epoch['armed_sha256']):
        raise ValueError('execution immutable metadata inventory differs')
    for relative, expected_sha in files.items():
        target = root / relative
        if expected_sha is not None:
            if target.is_symlink() or not target.is_file() or sha(target) != expected_sha:
                raise ValueError('execution immutable metadata bytes changed')
        elif relative == 'QUEUE_COMPLETE.json' or (relative in plan_paths and plan_paths[relative] in pending):
            continue
        elif target.exists() or target.is_symlink():
            raise ValueError('execution created an unapproved metadata path')
    lock = json.loads((root / 'QUEUE_LOCK.json').read_text())
    if (lock.get('pid') != old['pid'] or lock.get('manifest_sha256') != manifest_sha or
            lock.get('orchestration_commit') != origin):
        raise ValueError('original queue lock identity differs')
    return epoch


def verify_execution_source(epoch, armed):
    root = Path(__file__).resolve().parent
    names = ('frozen_ablation_gpu1_continue.py', 'frozen_ablation_quiescent.py', 'frozen_ablation_audit.py')
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()
    if (head != epoch['orchestration_commit'] or dirty or
            armed.get('orchestration_files_sha256') != {name: sha(root / name) for name in names} or
            armed.get('training_commits') != {'no_partykd': BASE_COMMIT, 'fim_off': PATCH_COMMIT}):
        raise ValueError('execution audit source is not the clean pinned orchestration and training code')


def audit_publication_preflight(path, epoch):
    if epoch is None:
        return validate_publication_preflight(path)
    from frozen_ablation_gpu1_continue import load_legacy
    return load_legacy(epoch['legacy_root']).validate_publication_preflight(path)


def verify_execution_job(root, row, config, start, finish, gate, runtime, epoch, epoch_sha):
    key = row['job_key']
    index = epoch['assignments'][key]
    binding = dict(execution_epoch_sha256=epoch_sha, controller_identity=epoch['controller_identity'],
                   gpu_index=index, gpu_uuid=GPU_UUID[index], logical_device='cuda:0',
                   cuda_visible_devices=[GPU_UUID[index], GPU_UUID[1-index]])
    if config['device'] != 'cuda:0':
        raise ValueError('execution extension changed original logical CUDA device')
    for name, record in (('start', start), ('finish', finish), ('gate', gate), ('runtime', runtime)):
        if any(record.get(k) != v for k, v in binding.items()):
            raise ValueError(f'execution {name} physical/logical identity differs: {key}')
    for record in (runtime, gate):
        if (record.get('observed_cuda_uuid_order') != binding['cuda_visible_devices'] or
                record.get('job_key') != key or record.get('attempt') != 1 or
                record.get('manifest_sha256') != epoch['manifest_sha256'] or
                record.get('config_sha256') != row['config_sha256'] or
                record.get('training_code_commit') != row['code_commit']):
            raise ValueError(f'execution observed CUDA identities differ: {key}')
    job_root = Path(config['output_dir']).parent
    if (json.loads((job_root / 'attempt-01.plan.json').read_text()) != {'job_key': key, 'attempt': 1} or
            (job_root / 'attempt-01/APPROVED_CONFIG.json').read_bytes() !=
            (Path(root) / 'manifests' / row['config_relpath']).read_bytes() or
            (job_root / 'attempt-01/config.json').read_bytes() !=
            (Path(root) / 'manifests' / row['config_relpath']).read_bytes()):
        raise ValueError(f'execution original config copy or plan differs: {key}')
    protocol = Path(root) / 'gpu1-rebalance-v1/workers' / key.replace('/', '__')
    ready, go = (json.loads((protocol / name).read_text()) for name in ('READY.json', 'GO.json'))
    worker = ready.get('worker_identity', {})
    expected = dict(job_key=key, attempt=1, config_sha256=row['config_sha256'],
                    training_code_commit=row['code_commit'], **binding, worker_identity=worker)
    if (ready != expected or go != expected or runtime.get('worker_identity') != worker or
            gate.get('worker_identity') != worker or worker.get('ppid') != epoch['controller_identity']['pid'] or
            not isinstance(worker.get('pid'), int) or worker.get('pid') <= 0 or
            worker.get('session') != worker['pid'] or worker.get('pgrp') != worker['pid'] or
            not worker.get('start_ticks') or not worker.get('boot_id') or not worker.get('cmdline')):
        raise ValueError(f'execution READY/GO or worker identity differs: {key}')


def audit_metadata(manifest, digest, *, execution_epoch=None, execution_epoch_sha256=None):
    if bool(execution_epoch) != bool(execution_epoch_sha256):
        raise ValueError('execution extension requires both explicit path and external SHA')
    manifest = Path(manifest).resolve()
    root = Path(RESULT_ROOT)
    if (manifest != root / 'manifests/manifest.jsonl' or
            sha(manifest) != digest or
            (root / 'QUEUE_FAILURE.json').exists()):
        raise ValueError('formal ablation manifest or queue changed')
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if len(rows) != 18 or {row['job_key'] for row in rows} != expected_keys():
        raise ValueError('formal ablation matrix is incomplete')
    epoch = (verify_execution_extension(root, rows, digest, execution_epoch, execution_epoch_sha256)
             if execution_epoch else None)
    if epoch:
        verify_execution_source(epoch, json.loads((Path(execution_epoch).parent / 'ARMED.json').read_text()))
    evidence = manifest.parent / 'preflight'
    preflight_path = evidence / 'PRELAUNCH_PREFLIGHT.json'
    smoke_path = evidence / 'SMOKE_COMPARISON.json'
    review_path = evidence / 'PATCH_REVIEW.json'
    if any(not path.is_file() for path in
           (preflight_path, smoke_path, review_path)):
        raise ValueError('preflight evidence is missing from the sealed root')
    preflight = json.loads(preflight_path.read_text())
    review = json.loads(review_path.read_text())
    comparison = json.loads(smoke_path.read_text())
    validate_smoke_comparison(comparison)
    from frozen_ablation_smoke import validate_smoke_records
    validate_smoke_records(evidence / 'smokes', comparison,
                           producer_commit=SMOKE_PRODUCER_COMMIT)
    publication_sha = audit_publication_preflight(evidence / 'publication', epoch)
    if (any(row.get('publication_preflight_sha256') != publication_sha for row in rows) or
            preflight.get('publication_preflight_sha256') != publication_sha):
        raise ValueError('publication preflight identity differs')
    row_hashes = {
        name: {row.get(name) for row in rows}
        for name in ('preflight_sha256', 'smoke_comparison_sha256',
                     'patch_review_sha256', 'patch_diff_sha256')
    }
    if any(len(values) != 1 for values in row_hashes.values()):
        raise ValueError('preflight hashes differ across manifest rows')
    expected_diff = hashlib.sha256(subprocess.check_output([
        'git', '-C', str(Path(__file__).parent), 'diff',
        f'{BASE_COMMIT}..{PATCH_COMMIT}', '--',
        'config.py', 'cl_methods/proto_evolve.py',
    ])).hexdigest()
    if (next(iter(row_hashes['preflight_sha256'])) != sha(preflight_path) or
            next(iter(row_hashes['smoke_comparison_sha256'])) != sha(smoke_path) or
            next(iter(row_hashes['patch_review_sha256'])) != sha(review_path) or
            next(iter(row_hashes['patch_diff_sha256'])) != expected_diff or
            preflight.get('status') != 'approved' or
            preflight.get('schema_version') != 1 or
            preflight.get('base_commit') != BASE_COMMIT or
            preflight.get('patch_commit') != PATCH_COMMIT or
            preflight.get('patch_diff_sha256') != expected_diff or
            preflight.get('smoke_comparison_sha256') != sha(smoke_path) or
            preflight.get('patch_review_sha256') != sha(review_path) or
            preflight.get('reviewer_id') != review.get('reviewer_id') or
            review.get('verdict') != 'approved' or
            review.get('base_commit') != BASE_COMMIT or
            review.get('patch_commit') != PATCH_COMMIT or
            review.get('patch_diff_sha256') != expected_diff or
            review.get('blocking_findings') != []):
        raise ValueError('preflight evidence does not bind the approved smoke and FIM patch')
    queue = json.loads((root / 'QUEUE_COMPLETE.json').read_text())
    ledger_path = root / 'ATTEMPTS.jsonl'
    if (queue.get('status') != 'complete' or
            queue.get('manifest_sha256') != digest or
            queue.get('orchestration_commit') != (epoch['orchestration_commit'] if epoch else rows[0]['orchestration_commit']) or
            queue.get('accounted_for') != 18 or
            queue.get('successes') != 18 or
            queue.get('attempts_sha256') != sha(ledger_path)):
        raise ValueError('formal ablation queue is not completely sealed')
    if epoch and (queue.get('original_orchestration_commit') != rows[0]['orchestration_commit'] or
                  queue.get('execution_epoch_sha256') != execution_epoch_sha256):
        raise ValueError('queue completion is not bound to the actual execution extension')
    events = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    if len(events) != 36:
        raise ValueError('formal ablation ledger has missing or extra events')
    by_key = defaultdict(list)
    for event in events:
        by_key[event['job_key']].append(event)
    if set(by_key) != expected_keys():
        raise ValueError('formal ablation ledger contains missing or extra jobs')
    source_anchor = SOURCE_ROOT / 'audit/ADAPTIVE_SAMPLE_ONLY_FINAL_ANCHOR.json'
    anchor = json.loads(source_anchor.read_text())
    if (sha(source_anchor) != SOURCE_ANCHOR_SHA or
            anchor.get('producer_commit') != BASE_COMMIT or
            anchor.get('basic_passed') != 9):
        raise ValueError('original formal anchor changed')
    final_gate_path = SOURCE_ROOT / anchor['final_gate']['relative_path']
    if sha(final_gate_path) != anchor['final_gate']['sha256']:
        raise ValueError('original formal final gate changed')
    original_trees = json.loads(final_gate_path.read_text())['original_tree_sha256']
    old_ledger = SOURCE_ROOT / 'audit/attestations.jsonl'
    if sha(old_ledger) != anchor['ledger']['sha256']:
        raise ValueError('original formal audit ledger changed')
    old_attest = {}
    for line in old_ledger.read_text().splitlines():
        record = json.loads(line)
        if record.get('stage') == 'basic' and record.get('status') == 'passed':
            if record['job'] in old_attest:
                raise ValueError('duplicate original formal attestation')
            old_attest[record['job']] = record
    if len(old_attest) != 9:
        raise ValueError('original formal attestations are incomplete')
    cache, tree_cache, source_seals = {}, {}, {}
    def cached_sha(path):
        path = Path(path)
        if path not in cache:
            cache[path] = sha(path)
        return cache[path]
    audited = {}
    for row in rows:
        key = row['job_key']
        dataset, arm, seed_text = key.split('/')
        seed = int(seed_text)
        source = SOURCE_ROOT / 'full' / dataset / 'adaptive' / str(seed) / 'run'
        source_config_path = source / 'config.json'
        source_config = json.loads(source_config_path.read_text())
        original_success = json.loads(
            (source.parent / 'JOB_SUCCESS.json').read_text())
        formal_runtime = original_success.get('runtime')
        old_key = f'{dataset}/adaptive/{seed}'
        if source.parent not in tree_cache:
            tree_cache[source.parent] = original_tree_sha256(source.parent)
        if (row.get('original_tree_sha256') != original_trees.get(old_key) or
                tree_cache[source.parent] != row['original_tree_sha256']):
            raise ValueError(f'original formal tree changed: {key}')
        if (not isinstance(formal_runtime, dict) or
                row.get('runtime') != formal_runtime or
                row.get('runtime_sha256') != hashlib.sha256(
                    _bytes(formal_runtime)).hexdigest() or
                runtime_identity() != formal_runtime):
            raise ValueError(f'formal training runtime differs: {key}')
        original_published = source / 'FORMAL_EVALUATION_PUBLISHED.json'
        if (cached_sha(source_config_path) != row['source_config_sha256'] or
                cached_sha(original_published) != old_attest[old_key]['file_sha256'][
                    'run/FORMAL_EVALUATION_PUBLISHED.json']):
            raise ValueError(f'original formal job changed: {key}')
        if source not in source_seals:
            source_seals[source] = verify_historical_source(source)
        data_path = (Path(source_config['data_path']) / 'cifar-100-python/train'
                     if dataset == 'cifar100' else Path(source_config['vector_npz']))
        if (cached_sha(data_path) != DATA_SHA[dataset] or
                row['source_data_sha256'] != DATA_SHA[dataset] or
                row.get('auxiliary_data_sha256') != AUX_DATA_SHA[dataset]):
            raise ValueError(f'original data differs: {key}')
        for name, path in auxiliary_data_paths(source_config, dataset).items():
            if cached_sha(path) != AUX_DATA_SHA[dataset][name]:
                raise ValueError(f'original auxiliary data differs: {key}, {name}')
        config_path = manifest.parent / row['config_relpath']
        config = json.loads(config_path.read_text())
        code = BASE_COMMIT if arm == 'no_partykd' else PATCH_COMMIT
        gpu = row['physical_gpu']
        if (cached_sha(config_path) != row['config_sha256'] or
                config != derive_config(source_config, dataset, arm, seed) or
                row['config_semantic_diff'] != semantic_diff(source_config, config) or
                row['code_commit'] != code or
                row['source_commit'] != BASE_COMMIT or
                row['source_formal_anchor_sha256'] != SOURCE_ANCHOR_SHA or
                row['gpu_uuid'] != GPU_UUID[gpu] or
                gpu != int(config['device'][-1])):
            raise ValueError(f'formal ablation identity differs: {key}')
        start, finish = by_key[key]
        execution_gpu = epoch['assignments'].get(key, gpu) if epoch else gpu
        if (start.get('kind') != 'start' or finish.get('kind') != 'finish' or
                start.get('attempt') != 1 or
                finish.get('attempt') != 1 or
                start.get('config_sha256') != row['config_sha256'] or
                start.get('code_commit') != code or
                start.get('gpu_index') != execution_gpu or
                start.get('gpu_uuid') != GPU_UUID[execution_gpu] or
                finish.get('classification') != 'success' or
                finish.get('returncode') != 0 or
                finish.get('test_accessed') is not True):
            raise ValueError(f'formal ablation attempt differs: {key}')
        job_root = Path(config['output_dir']).parent
        run = job_root / 'attempt-01'
        gate_path = run / 'JOB_SUCCESS.json'
        gate = json.loads(gate_path.read_text())
        log = job_root / 'attempt-01.log'
        if (cached_sha(gate_path) != finish['job_success_sha256'] or
                cached_sha(log) != finish['log_sha256'] or
                gate.get('status') != 'success' or
                gate.get('job_key') != key or
                gate.get('attempt') != 1 or
                gate.get('manifest_sha256') != digest or
                gate.get('config_sha256') != row['config_sha256'] or
                gate.get('training_code_commit') != code):
            raise ValueError(f'formal ablation gate differs: {key}')
        runtime_path = run / 'JOB_RUNTIME.json'
        if (not runtime_path.is_file() or
                json.loads(runtime_path.read_text()).get('runtime') !=
                formal_runtime or gate.get('runtime') != formal_runtime):
            raise ValueError(f'ablation runtime identity differs: {key}')
        if epoch and key in epoch['assignments']:
            verify_execution_job(root, row, config, start, finish, gate,
                                 json.loads(runtime_path.read_text()), epoch, execution_epoch_sha256)
        elif any(record.get('execution_epoch_sha256') is not None for record in
                 (start, finish, gate, json.loads(runtime_path.read_text()))):
            raise ValueError(f'unapproved execution extension metadata: {key}')
        verify_run_gate(run, gate)
        if arm == 'fim_off':
            count = len(re.findall(r'FIM applied freeze: 0/\d+ bottom elements '
                                   r'\(fraction=0\.000000\)', log.read_text()))
            if count != config['num_tasks'] or finish.get('fim_zero_freeze_logs') != count:
                raise ValueError(f'FIM-off applied a nonzero or unlogged freeze: {key}')
        audited[key] = {'run': run, 'original': source,
                        'gate_sha256': cached_sha(gate_path)}
    return audited, sha(ledger_path)


def final_score(path):
    result = json.loads(Path(path).read_text())
    value = result['cl_metrics']['AA_final']
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not 0 <= value <= 1):
        raise ValueError('formal final Class-IL is invalid')
    return Fraction(str(value))


def summarize(audited):
    per_seed, summaries = [], []
    means = {}
    for dataset in DATASETS:
        for arm in ('no_partykd', 'fim_off'):
            main_scores, ablated_scores, deltas, exact_deltas = [], [], [], []
            for seed in SEEDS:
                job = audited[f'{dataset}/{arm}/{seed}']
                main = final_score(job['original'] / 'results.json')
                ablated = final_score(job['run'] / 'results.json')
                delta = main - ablated
                main_scores.append(float(main))
                ablated_scores.append(float(ablated))
                deltas.append(float(delta))
                exact_deltas.append(delta)
                per_seed.append({'dataset': dataset, 'arm': arm, 'seed': seed,
                    'main_class_il': float(main),
                    'ablated_class_il': float(ablated),
                    'paired_delta_pp': float(delta * 100)})
            means[(dataset, arm)] = sum(exact_deltas) / 3
            summaries.append({'dataset': dataset, 'arm': arm,
                'main_mean': statistics.mean(main_scores),
                'main_sample_sd': statistics.stdev(main_scores),
                'ablated_mean': statistics.mean(ablated_scores),
                'ablated_sample_sd': statistics.stdev(ablated_scores),
                'paired_delta_mean_pp': 100 * statistics.mean(deltas),
                'paired_delta_sample_sd_pp': 100 * statistics.stdev(deltas)})
    label = party_kd_contribution_label({dataset: means[(dataset, 'no_partykd')]
                                         for dataset in DATASETS})
    return per_seed, summaries, label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--execution-epoch', type=Path)
    parser.add_argument('--execution-epoch-sha256')
    args = parser.parse_args()
    root = Path(RESULT_ROOT)
    audited, ledger_sha = audit_metadata(args.manifest, args.manifest_sha256,
        execution_epoch=args.execution_epoch, execution_epoch_sha256=args.execution_epoch_sha256)
    pass_path = root / 'INDEPENDENT_AUDIT_PASS.json'
    if not args.summarize:
        if pass_path.exists():
            raise FileExistsError('ablation audit already sealed')
        with pass_path.open('x', encoding='utf-8') as stream:
            json.dump({'status': 'passed', 'jobs': 18,
                       'manifest_sha256': args.manifest_sha256,
                       'ledger_sha256': ledger_sha,
                       'execution_epoch_sha256': args.execution_epoch_sha256,
                       'job_gate_sha256': {key: item['gate_sha256']
                                           for key, item in audited.items()},
                       'accuracy_opened': False}, stream,
                      indent=2, sort_keys=True)
            stream.write('\n')
        print(json.dumps({'audit': 'passed', 'jobs': 18,
                          'accuracy_opened': False}))
        return
    passed = json.loads(pass_path.read_text())
    if (passed.get('status') != 'passed' or passed.get('jobs') != 18 or
            passed.get('manifest_sha256') != args.manifest_sha256 or
            passed.get('ledger_sha256') != ledger_sha or
            passed.get('execution_epoch_sha256') != args.execution_epoch_sha256 or
            passed.get('job_gate_sha256') != {key: item['gate_sha256']
                                              for key, item in audited.items()}):
        raise ValueError('independent audit seal changed before summary')
    per_seed, summaries, label = summarize(audited)
    for filename, rows in (('ABLATION_PER_SEED.csv', per_seed),
                           ('ABLATION_SUMMARY.csv', summaries)):
        with (root / filename).open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (root / 'ABLATION_INTERPRETATION.json').open('x', encoding='utf-8') as stream:
        json.dump({'party_kd_contribution': label,
                   'main_method_changed': False,
                   'formal_results_seen_before_ablation': True,
                   'significance_claimed': False}, stream,
                  indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'status': 'summarized', 'party_kd_contribution': label}))


if __name__ == '__main__':
    main()
