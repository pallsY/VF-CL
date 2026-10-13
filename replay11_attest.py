"""Append-only, externally pinned Replay-11 v3 audit runner."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import replay11_evidence as evidence
import replay11_matrix as matrix
import replay11_numeric as numeric


LEDGER_PATH = (evidence.FROZEN_RESULTS_ROOT.parent /
               'audit-v3' / 'attestations.jsonl')
_FILES = ('JOB_PLAN.json', 'JOB_SUCCESS.json',
          'run/FORMAL_STATE_FROZEN.json',
          'run/FORMAL_EVALUATION_PUBLISHED.json',
          'run/checkpoints/formal_final.pt')


def _digest(value):
    packed = json.dumps(value, sort_keys=True, ensure_ascii=False,
                        separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(packed.encode('utf-8')).hexdigest()


def read_ledger(path=LEDGER_PATH):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    previous = None
    for line in path.read_text(encoding='utf-8').splitlines():
        record = json.loads(line)
        declared = record.pop('event_sha256', None)
        if (declared != _digest(record)
                or record.get('previous_event_sha256') != previous):
            raise ValueError('audit ledger hash chain differs')
        record['event_sha256'] = declared
        records.append(record)
        previous = declared
    return records


def append_event(path, event):
    """Preserve every previous verdict; verify the chain before appending."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+', encoding='utf-8', newline='\n') as handle:
        try:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        except ImportError:  # Single-process Windows unit tests only.
            pass
        handle.seek(0)
        existing = handle.read()
        previous = None
        for line in existing.splitlines():
            record = json.loads(line)
            declared = record.pop('event_sha256', None)
            if (declared != _digest(record)
                    or record.get('previous_event_sha256') != previous):
                raise ValueError('audit ledger hash chain differs')
            previous = declared
        if 'event_sha256' in event or 'previous_event_sha256' in event:
            raise ValueError('audit event has reserved hash fields')
        record = dict(event, previous_event_sha256=previous)
        record['event_sha256'] = _digest(record)
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False,
                                separators=(',', ':'), allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    return record


def _audit_commit():
    root = Path(__file__).resolve().parent
    commit = subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(
        ['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()
    if dirty or len(commit) != 40:
        raise ValueError('audit checkout is dirty or uncommitted')
    return commit


def _file_hashes(job_dir):
    found = {}
    for name in _FILES:
        path = job_dir / name
        if path.is_file() and not path.is_symlink():
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            found[name] = digest.hexdigest()
        else:
            found[name] = None
    return found


def _tree_snapshot(job_dir):
    try:
        files, size, digest = evidence._tree_identity(job_dir)
        return {'sha256': digest, 'files': files, 'bytes': size, 'error': None}
    except Exception as problem:
        return {'sha256': None, 'files': None, 'bytes': None,
                'error': f'{type(problem).__name__}: {problem}'}


def _seed42_numeric_is_latest(ledger, dataset, method, audit_commit):
    target = f'{dataset}/{method}/42'
    matches = [row for row in ledger
               if row.get('job') == target
               and row.get('audit_commit') == audit_commit
               and row.get('producer_commit') == evidence.CORRECTED_COMMIT
               and row.get('stage') in ('basic', 'numeric')]
    return bool(matches and matches[-1]['stage'] == 'numeric'
                and matches[-1]['status'] == 'passed')


def audit_one(job, producer_root):
    """Run basic, then only preselected seed-42 numeric, logging each verdict."""
    if tuple(job) not in matrix.jobs():
        raise ValueError('job is outside the frozen 99-cell matrix')
    dataset, method, seed = job
    job_dir = evidence.FROZEN_RESULTS_ROOT / dataset / method / str(seed)
    common = {
        'schema_version': 1,
        'job': f'{dataset}/{method}/{seed}',
        'source_kind': 'original',
        'source_path': str(job_dir),
        'archive_path': None,
        'producer_commit': evidence.CORRECTED_COMMIT,
        'audit_commit': None,
        'host': platform.node(),
        'file_sha256': None,
    }
    started = time.perf_counter()
    try:
        common['audit_commit'] = _audit_commit()
        common['file_sha256'] = _file_hashes(job_dir)
    except Exception as problem:
        snapshot = _tree_snapshot(job_dir)
        append_event(LEDGER_PATH, dict(
            common, stage='preflight', status='failed',
            audited_at_utc=datetime.now(timezone.utc).isoformat(),
            seconds=round(time.perf_counter() - started, 3), result=None,
            original_tree_sha256=snapshot['sha256'],
            original_tree_after_sha256=snapshot['sha256'],
            original_tree_error=snapshot['error'],
            error={'type': type(problem).__name__, 'message': str(problem)}))
        raise ValueError(f'preflight audit failed: {problem}') from problem
    if seed != 42 and not _seed42_numeric_is_latest(
            read_ledger(LEDGER_PATH), dataset, method, common['audit_commit']):
        snapshot = _tree_snapshot(job_dir)
        append_event(LEDGER_PATH, dict(
            common, stage='prerequisite', status='failed',
            audited_at_utc=datetime.now(timezone.utc).isoformat(),
            seconds=0.0, result=None,
            original_tree_sha256=snapshot['sha256'],
            original_tree_after_sha256=snapshot['sha256'],
            original_tree_error=snapshot['error'],
            error={'type': 'ValueError',
                   'message': 'seed 42 numeric admission is not the latest verdict'}))
        raise ValueError('seed 42 numeric admission is required before seed 43/44')
    outputs = []
    for stage in ('basic', 'numeric') if seed == 42 else ('basic',):
        started = time.perf_counter()
        before = _tree_snapshot(job_dir)
        try:
            if before['error'] is not None:
                raise ValueError('original tree cannot be hashed: ' + before['error'])
            result = (evidence.verify_v3_formal_job(job_dir)
                      if stage == 'basic' else
                      numeric.audit_job(job_dir, evidence.CORRECTED_COMMIT,
                                        producer_root, evidence.FROZEN_DATA_ROOT))
            if stage == 'numeric' and result.get('basic_admission') != 'formal':
                raise ValueError('numeric worker lacks formal basic admission')
            status = 'passed'
            error = None
        except Exception as problem:
            result = None
            status = 'failed'
            error = {'type': type(problem).__name__, 'message': str(problem)}
        after = _tree_snapshot(job_dir)
        if before != after:
            status = 'failed'
            result = None
            error = {'type': 'ValueError',
                     'message': 'original tree changed during audit'}
        event = dict(common, stage=stage, status=status,
                     audited_at_utc=datetime.now(timezone.utc).isoformat(),
                     seconds=round(time.perf_counter() - started, 3),
                     original_tree_sha256=before['sha256'],
                     original_tree_after_sha256=after['sha256'],
                     original_tree_files=before['files'],
                     original_tree_bytes=before['bytes'],
                     original_tree_error=before['error'] or after['error'],
                     result=result, error=error)
        outputs.append(append_event(LEDGER_PATH, event))
        if status == 'failed':
            raise ValueError(f'{stage} audit failed: {error["message"]}')
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job', help='dataset/method/seed from the frozen matrix')
    parser.add_argument('--producer-root', required=True, type=Path)
    args = parser.parse_args()
    parts = args.job.split('/')
    if len(parts) != 3:
        parser.error('job must be dataset/method/seed')
    records = audit_one((parts[0], parts[1], int(parts[2])), args.producer_root)
    print(json.dumps([{'job': row['job'], 'stage': row['stage'],
                       'status': row['status'],
                       'event_sha256': row['event_sha256']}
                      for row in records], sort_keys=True))


if __name__ == '__main__':
    main()
