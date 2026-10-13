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


def append_event(path, event, *, require_seed42=None):
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
        existing_records = []
        for line in existing.splitlines():
            record = json.loads(line)
            declared = record.pop('event_sha256', None)
            if (declared != _digest(record)
                    or record.get('previous_event_sha256') != previous):
                raise ValueError('audit ledger hash chain differs')
            previous = declared
            record['event_sha256'] = declared
            existing_records.append(record)
        if 'event_sha256' in event or 'previous_event_sha256' in event:
            raise ValueError('audit event has reserved hash fields')
        record = dict(event)
        if require_seed42 is not None and not _seed42_numeric_is_latest(
                existing_records, *require_seed42):
            record.update(stage='prerequisite', status='failed', result=None,
                          error={'type': 'ValueError',
                                 'message': 'seed 42 numeric admission changed during audit'})
        record['previous_event_sha256'] = previous
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
        if job_dir.is_symlink() or not job_dir.is_dir():
            raise ValueError('original directory is missing or a symlink')
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


def finalize_records(records, audit_commit, *, jobs=None):
    """Require the latest verdict for all 99 basic and 33 numeric cells."""
    selected = list(matrix.jobs() if jobs is None else jobs)
    basic_count = numeric_count = 0
    for dataset, method, seed in selected:
        key = f'{dataset}/{method}/{seed}'
        relevant = [row for row in records if row.get('job') == key]
        if not relevant:
            raise ValueError(f'latest basic verdict is missing: {key}')
        latest = relevant[-1]
        stage = 'numeric' if seed == 42 else 'basic'
        if latest.get('stage') != stage or latest.get('status') != 'passed':
            raise ValueError(f'latest {stage} verdict is not passed: {key}')
        required = [latest]
        if seed == 42:
            prior_basic = next((row for row in reversed(relevant[:-1])
                                if row.get('stage') == 'basic'), None)
            if prior_basic is None or prior_basic.get('status') != 'passed':
                raise ValueError(f'latest basic verdict is not passed: {key}')
            required.append(prior_basic)
            numeric_count += 1
        for row in required:
            expected_path = (evidence.FROZEN_RESULTS_ROOT / dataset / method /
                             str(seed))
            digest = row.get('original_tree_sha256')
            if (row.get('audit_commit') != audit_commit
                    or row.get('producer_commit') != evidence.CORRECTED_COMMIT
                    or row.get('source_kind') != 'original'
                    or row.get('source_path') != str(expected_path)
                    or not isinstance(digest, str) or len(digest) != 64
                    or digest != row.get('original_tree_after_sha256')
                    or row.get('original_tree_error') is not None
                    or not row.get('file_sha256')):
                raise ValueError(f'latest evidence identity differs: {key}')
            result = row.get('result') or {}
            expected_admission = ('basic_admission' if row['stage'] == 'numeric'
                                  else 'admission')
            if result.get(expected_admission) != 'formal':
                raise ValueError(f'latest {row["stage"]} admission differs: {key}')
        if len(required) == 2 and (required[0]['original_tree_sha256']
                                    != required[1]['original_tree_sha256']):
            raise ValueError(f'basic/numeric original tree differs: {key}')
        basic_count += 1
    return {'basic_passed': basic_count, 'numeric_passed': numeric_count,
            'audit_commit': audit_commit,
            'producer_commit': evidence.CORRECTED_COMMIT,
            'ledger_head_sha256': (records[-1]['event_sha256'] if records
                                   else None)}


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
            if (stage == 'numeric' and before['sha256']
                    != outputs[-1]['original_tree_after_sha256']):
                raise ValueError('original tree changed since basic admission')
            result = (evidence.verify_v3_formal_job(
                          job_dir, producer_root=producer_root)
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
        require = ((dataset, method, common['audit_commit'])
                   if seed != 42 and stage == 'basic' else None)
        saved = append_event(LEDGER_PATH, event, require_seed42=require)
        outputs.append(saved)
        if saved['status'] == 'failed':
            raise ValueError(f'{saved["stage"]} audit failed: '
                             f'{saved["error"]["message"]}')
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job', help='dataset/method/seed or finalize')
    parser.add_argument('--producer-root', type=Path)
    args = parser.parse_args()
    if args.job == 'finalize':
        print(json.dumps(finalize_records(read_ledger(LEDGER_PATH),
                                          _audit_commit()), sort_keys=True))
        return
    if args.producer_root is None:
        parser.error('--producer-root is required for a job audit')
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
