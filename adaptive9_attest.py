"""Append-only audit and final gate for the nine adaptive originals."""

import json
import hashlib
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import adaptive9_evidence as evidence
import replay11_attest as ledger


DATASETS = ('cifar100', 'isolet', 'upmc_food101')
SEEDS = (42, 43, 44)


def finalize_records(records, *, producer_commit, audit_commit, results_root):
    """Require latest basic success for nine jobs and numeric for seed 42."""
    basic_count = numeric_count = 0
    root = str(results_root).replace('\\', '/').rstrip('/')
    for dataset, seed in product(DATASETS, SEEDS):
        job = f'{dataset}/adaptive/{seed}'
        relevant = [row for row in records if row.get('job') == job]
        expected_stage = 'numeric' if seed == 42 else 'basic'
        if not relevant or relevant[-1].get('stage') != expected_stage:
            raise ValueError(f'latest {expected_stage} verdict is missing: {job}')
        latest = relevant[-1]
        required = [latest]
        if seed == 42:
            prior = next((row for row in reversed(relevant[:-1])
                          if row.get('stage') == 'basic'), None)
            if prior is None or prior.get('status') != 'passed':
                raise ValueError(f'latest basic verdict differs: {job}')
            required.append(prior)
            numeric_count += 1
        for row in required:
            digest = row.get('original_tree_sha256')
            admission = ('basic_admission' if row['stage'] == 'numeric'
                         else 'admission')
            if (row.get('status') != 'passed'
                    or row.get('producer_commit') != producer_commit
                    or row.get('audit_commit') != audit_commit
                    or row.get('source_kind') != 'original'
                    or row.get('source_path', '').replace('\\', '/')
                    != f'{root}/{job}'
                    or type(digest) is not str or len(digest) != 64
                    or row.get('original_tree_after_sha256') != digest
                    or (row.get('result') or {}).get(admission) != 'formal'):
                raise ValueError(f'latest {row["stage"]} verdict differs: {job}')
        if len(required) == 2 and (required[0]['original_tree_sha256']
                                   != required[1]['original_tree_sha256']):
            raise ValueError(f'basic/numeric original tree differs: {job}')
        basic_count += 1
    return {
        'basic_passed': basic_count, 'numeric_passed': numeric_count,
        'producer_commit': producer_commit, 'audit_commit': audit_commit,
        'ledger_head_sha256': (records[-1]['event_sha256'] if records else None),
    }


def _snapshot(job_dir):
    return ledger._tree_snapshot(job_dir)


def numeric_visibility(source_host):
    if source_host not in evidence.HOST_GPU_COUNT:
        raise ValueError('adaptive numeric source host differs')
    return ','.join(map(str, range(evidence.HOST_GPU_COUNT[source_host])))


def _event(job, stage, status, result, error, *, job_dir,
           producer_commit, audit_commit, before, after, seconds):
    return {
        'schema_version': 1,
        'job': job, 'stage': stage, 'status': status,
        'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'seconds': round(seconds, 3), 'host': platform.node(),
        'producer_commit': producer_commit, 'audit_commit': audit_commit,
        'source_kind': 'original', 'source_path': str(job_dir),
        'file_sha256': ledger._file_hashes(job_dir),
        'original_tree_sha256': before['sha256'],
        'original_tree_after_sha256': after['sha256'],
        'original_tree_error': before['error'] or after['error'],
        'result': result, 'error': error,
    }


def _numeric_in_new_process(job_dir, producer_commit, producer_root,
                            data_root):
    import torch

    plan = evidence.shared._json(Path(job_dir) / 'JOB_PLAN.json')
    runtime = evidence.validate_runtime(
        plan['runtime'], plan['command'][0])
    source_host = runtime['host']
    count = evidence.HOST_GPU_COUNT[source_host]
    if torch.cuda.device_count() < count:
        raise ValueError('audit host lacks source GPU count')
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = numeric_visibility(source_host)
    completed = subprocess.run([
        sys.executable,
        str(Path(__file__).resolve().parent / 'replay11_numeric.py'),
        '--worker', str(job_dir), producer_commit,
        str(producer_root), str(data_root),
    ], capture_output=True, text=True, check=False, env=env)
    if completed.returncode:
        raise ValueError('adaptive numeric worker failed: '
                         + completed.stderr[-4000:])
    prefix = 'REPLAY11_NUMERIC_JSON='
    rows = [line[len(prefix):] for line in completed.stdout.splitlines()
            if line.startswith(prefix)]
    if len(rows) != 1:
        raise ValueError('adaptive numeric worker returned no single result')
    return json.loads(rows[0])


def audit_one(dataset, seed, *, results_root, data_root, producer_root,
              producer_commit, ledger_path, audit_commit):
    """Append a basic verdict and, for seed 42, a numeric verdict."""
    if dataset not in DATASETS or seed not in SEEDS:
        raise ValueError('job is outside the adaptive nine-cell matrix')
    if ledger._audit_commit() != audit_commit:
        raise ValueError('adaptive audit checkout differs from frozen commit')
    job = f'{dataset}/adaptive/{seed}'
    job_dir = Path(results_root) / dataset / 'adaptive' / str(seed)
    existing = ledger.read_ledger(ledger_path)
    prior = [row for row in existing if row.get('job') == job]
    if prior and prior[-1].get('status') != 'passed':
        raise ValueError('adaptive latest verdict failed; preserve evidence')
    if prior and (prior[-1].get('stage') == 'numeric'
                  or seed != 42):
        raise ValueError('adaptive job already audited')
    if prior and prior[-1].get('stage') != 'basic':
        raise ValueError('adaptive audit stage order differs')
    common = dict(
        results_root=results_root, data_root=data_root,
        producer_root=producer_root, producer_commit=producer_commit)
    basic = None
    if not prior:
        started = time.perf_counter()
        before = _snapshot(job_dir)
        try:
            basic = evidence.verify_pinned_job(job_dir, **common)
            after = _snapshot(job_dir)
            if (before['error'] or after['error']
                    or before['sha256'] != after['sha256']):
                raise ValueError('adaptive original changed during basic audit')
            status, error = 'passed', None
        except Exception as problem:
            after = _snapshot(job_dir)
            status, error = 'failed', {
                'type': type(problem).__name__, 'message': str(problem)}
        event = _event(
            job, 'basic', status, basic if status == 'passed' else None,
            error, job_dir=job_dir, producer_commit=producer_commit,
            audit_commit=audit_commit, before=before, after=after,
            seconds=time.perf_counter() - started)
        ledger.append_event(ledger_path, event)
        if status != 'passed':
            raise ValueError(f'adaptive basic audit failed: {error}')
    if seed != 42:
        return
    started = time.perf_counter()
    before = _snapshot(job_dir)
    try:
        if prior:
            if (before['sha256'] != prior[-1]['original_tree_sha256']
                    or prior[-1].get('audit_commit') != audit_commit
                    or prior[-1].get('producer_commit') != producer_commit):
                raise ValueError('adaptive seed42 basic evidence changed')
            basic = evidence.verify_pinned_job(job_dir, **common)
        numeric = _numeric_in_new_process(
            job_dir, producer_commit, producer_root, data_root)
        if numeric['metrics'] != basic['metrics']:
            raise ValueError('adaptive numeric metrics differ from basic audit')
        numeric['basic_admission'] = 'formal'
        after = _snapshot(job_dir)
        if (before['error'] or after['error']
                or before['sha256'] != after['sha256']):
            raise ValueError('adaptive original changed during numeric audit')
        status, error = 'passed', None
    except Exception as problem:
        after = _snapshot(job_dir)
        status, error = 'failed', {
            'type': type(problem).__name__, 'message': str(problem)}
    event = _event(
        job, 'numeric', status, numeric if status == 'passed' else None,
        error, job_dir=job_dir, producer_commit=producer_commit,
        audit_commit=audit_commit, before=before, after=after,
        seconds=time.perf_counter() - started)
    ledger.append_event(ledger_path, event)
    if status != 'passed':
        raise ValueError(f'adaptive numeric audit failed: {error}')


def write_final_gate(*, results_root, data_root, producer_root,
                     producer_commit, ledger_path, audit_commit):
    """Recheck every original before publishing the nine-cell final gate."""
    if ledger._audit_commit() != audit_commit:
        raise ValueError('adaptive final audit checkout differs')
    records = ledger.read_ledger(ledger_path)
    summary = finalize_records(
        records, producer_commit=producer_commit,
        audit_commit=audit_commit, results_root=results_root)
    trees = {}
    for dataset, seed in product(DATASETS, SEEDS):
        job = f'{dataset}/adaptive/{seed}'
        job_dir = Path(results_root) / dataset / 'adaptive' / str(seed)
        before = _snapshot(job_dir)
        checked = evidence.verify_pinned_job(
            job_dir, results_root=results_root, data_root=data_root,
            producer_root=producer_root,
            producer_commit=producer_commit)
        after = _snapshot(job_dir)
        latest = next(row for row in reversed(records)
                      if row.get('job') == job)
        if (checked['admission'] != 'formal' or before['error']
                or after['error'] or before['sha256'] != after['sha256']
                or before['sha256'] != latest['original_tree_sha256']):
            raise ValueError(f'adaptive original changed before final gate: {job}')
        trees[job] = before['sha256']
    output = {
        **summary, 'schema_version': 1,
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'ledger_sha256': hashlib.sha256(
            Path(ledger_path).read_bytes()).hexdigest(),
        'original_tree_sha256': trees,
    }
    target = Path(ledger_path).parent / 'FINAL_GATE.json'
    with target.open('x', encoding='utf-8') as handle:
        json.dump(output, handle, indent=2, sort_keys=True,
                  ensure_ascii=False, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    return output
