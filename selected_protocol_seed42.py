"""All-method seed-42 validation under the frozen vector protocols."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from dataset_protocol_smoke import replace_arg
from fair_main_table_3datasets import (
    CE_PROTOCOL_SELECTION,
    DATASETS,
    METHODS,
    base_command,
    complete_run,
    file_sha256,
    find_resume,
)


SEED = 42


def jobs(worker=None, workers=2):
    specs = [
        f'{dataset}:{method}:{SEED}'
        for method in METHODS
        for dataset in DATASETS
    ]
    if worker is None:
        return specs
    return [spec for index, spec in enumerate(specs) if index % workers == worker]


def parse_job(spec):
    dataset, method, seed_text = spec.split(':')
    seed = int(seed_text)
    if dataset not in DATASETS or method not in METHODS or seed != SEED:
        raise ValueError(f'unknown selected-protocol job: {spec}')
    return dataset, method, seed


def validation_command(spec, device, results_dir, resume_run_dir=None):
    dataset, method, seed = parse_job(spec)
    cfg = DATASETS[dataset]
    command = base_command(
        f'{dataset}:{method}:{seed}', device, results_dir,
        resume_run_dir=resume_run_dir, smoke=False,
    )
    replace_arg(command, '--lambda_validation_enabled', 1)
    replace_arg(
        command, '--lambda_validation_per_class', cfg['validation_per_class'],
    )
    replace_arg(
        command, '--lambda_validation_split_seed', cfg['validation_split_seed'],
    )
    replace_arg(command, '--exp_name', 'selected_protocol_seed42')
    return command


def deterministic_env():
    env = os.environ.copy()
    env.update({
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'PYTHONHASHSEED': str(SEED),
    })
    return env


def metric(result, name):
    value = result['cl_metrics'][name]
    if isinstance(value, dict):
        value = value.get('mean', value.get('value'))
    return float(value)


def job_root(matrix_root, spec):
    return Path(matrix_root) / 'runs' / spec.replace(':', '_')


def protocol_evidence(dataset):
    report_path = CE_PROTOCOL_SELECTION['report']
    report = json.loads(report_path.read_text(encoding='utf-8'))
    actual_hash = file_sha256(report_path)
    cfg = DATASETS[dataset]
    checks = {
        'selection_report_passed': report.get('passed') is True,
        'selection_test_not_used': report.get('selection_test_used') is False,
        'ours_not_used_for_selection': report.get('ours_used_for_selection') is False,
        'selection_hash_matches': actual_hash == CE_PROTOCOL_SELECTION['sha256'],
        'selected_ce_matches_command': (
            report['selected'][dataset]['ce_mode'] == cfg['task_ce_mode']
        ),
    }
    return {
        'report': str(CE_PROTOCOL_SELECTION['source_report']),
        'local_report': str(report_path),
        'expected_sha256': CE_PROTOCOL_SELECTION['sha256'],
        'actual_sha256': actual_hash,
        'selection_code_commit': CE_PROTOCOL_SELECTION['code_commit'],
        'checks': checks,
        'passed': all(checks.values()),
    }


def audit_run(spec, command, run_dir):
    dataset, method, seed = parse_job(spec)
    cfg = DATASETS[dataset]
    result_path = run_dir / 'results.json'
    manifest_path = run_dir / 'validation' / 'validation_manifest.json'
    if not result_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f'{spec} is missing results or validation manifest')
    result = json.loads(result_path.read_text(encoding='utf-8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    selection = result.get('selection_audit', {})
    trajectory = result['cl_metrics'].get('AA_trajectory_taskil', [])
    evidence = protocol_evidence(dataset)
    checks = {
        'selection_audit_passed': selection.get('passed') is True,
        'test_not_used': selection.get('test_used_for_selection') is False,
        'training_validation_only': (
            selection.get('evaluation_source') == 'vector-train-validation'
        ),
        'all_tasks_completed': len(trajectory) == len(cfg['tasks']),
        'ce_protocol_evidence_passed': evidence['passed'],
        'effective_ce_is_frozen': (
            command[command.index('--task_ce_mode') + 1] == cfg['task_ce_mode']
        ),
    }
    record = {
        'job': spec,
        'dataset': dataset,
        'method': method,
        'seed': seed,
        'AA_final': metric(result, 'AA_final'),
        'AA_cil': metric(result, 'AA_cil'),
        'BWT': metric(result, 'BWT'),
        'AA_final_taskil': metric(result, 'AA_final_taskil'),
        'validation_manifest_sha256': manifest['sha256'],
        'ce_mode': cfg['task_ce_mode'],
        'ce_protocol_evidence': evidence,
        'run_dir': str(run_dir),
        'checks': checks,
        'passed': all(checks.values()),
        'command': command,
    }
    if not record['passed']:
        raise RuntimeError(f'{spec} protocol audit failed: {checks}')
    return record


def discover_runs(root):
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        path for path in root.iterdir()
        if path.is_dir() and (path / 'config.json').is_file()
    )


def find_complete(root, expected_tasks):
    return [
        run_dir for run_dir in discover_runs(root)
        if complete_run(run_dir, expected_tasks)
    ]


def run_job(spec, device, matrix_root):
    dataset, _, _ = parse_job(spec)
    expected_tasks = len(DATASETS[dataset]['tasks'])
    root = job_root(matrix_root, spec)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'SUCCESS').is_file() and (root / 'record.json').is_file():
        print(f'SKIP complete {spec}')
        return 0
    complete = find_complete(root, expected_tasks)
    resume = None if complete else find_resume(discover_runs(root))
    command = validation_command(spec, device, root, resume_run_dir=resume)
    (root / 'planned_protocol.json').write_text(
        json.dumps({
            'job': spec,
            'selection_source': 'training-validation',
            'selection_test_used': False,
            'frozen_ce_evidence': protocol_evidence(dataset),
            'command': command,
        }, indent=2) + '\n', encoding='utf-8',
    )
    if not complete:
        with open(root / 'job.log', 'a', encoding='utf-8') as log:
            completed = subprocess.run(
                command, env=deterministic_env(), stdout=log,
                stderr=subprocess.STDOUT, check=False,
            )
        if completed.returncode:
            print(f'FAILED {spec}: exit={completed.returncode}', file=sys.stderr)
            return completed.returncode
        complete = find_complete(root, expected_tasks)
    if not complete:
        print(f'FAILED {spec}: no complete run', file=sys.stderr)
        return 90
    try:
        record = audit_run(spec, command, complete[-1])
    except Exception as error:
        print(f'FAILED {spec}: {error}', file=sys.stderr)
        return 91
    (root / 'record.json').write_text(
        json.dumps(record, indent=2) + '\n', encoding='utf-8',
    )
    (root / 'SUCCESS').touch()
    print(
        f"COMPLETE {spec}: AA={record['AA_final']:.4f} "
        f"BWT={record['BWT']:.4f} TIL={record['AA_final_taskil']:.4f}"
    )
    return 0


def summarize(matrix_root):
    matrix_root = Path(matrix_root)
    records, missing = [], []
    for spec in jobs():
        path = job_root(matrix_root, spec) / 'record.json'
        if not path.is_file():
            missing.append(spec)
        else:
            records.append(json.loads(path.read_text(encoding='utf-8')))
    report = matrix_root / 'formal_report'
    report.mkdir(parents=True, exist_ok=True)
    fields = [
        'dataset', 'method', 'seed', 'ce_mode', 'AA_final', 'AA_cil',
        'BWT', 'AA_final_taskil', 'validation_manifest_sha256', 'run_dir',
    ]
    with open(report / 'PER_RUN.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in records)
    manifest_checks = {}
    rankings = {}
    for dataset in DATASETS:
        rows = [row for row in records if row['dataset'] == dataset]
        hashes = {row['validation_manifest_sha256'] for row in rows}
        manifest_checks[dataset] = {
            'passed': len(hashes) == 1,
            'unique_hashes': sorted(hashes),
        }
        rankings[dataset] = sorted(
            [{
                key: row[key] for key in (
                    'method', 'AA_final', 'AA_cil', 'BWT', 'AA_final_taskil',
                )
            } for row in rows],
            key=lambda row: row['AA_final'], reverse=True,
        )
    passed = (
        not missing and len(records) == len(jobs())
        and all(row['passed'] for row in records)
        and all(check['passed'] for check in manifest_checks.values())
    )
    summary = {
        'passed': passed,
        'completed_jobs': len(records),
        'expected_jobs': len(jobs()),
        'missing_jobs': missing,
        'seed': SEED,
        'selection_source': 'training-validation',
        'selection_test_used': False,
        'frozen_ce_report_sha256': CE_PROTOCOL_SELECTION['sha256'],
        'manifest_checks': manifest_checks,
        'rankings': rankings,
    }
    (report / 'SELECTED_PROTOCOL_SEED42_SUMMARY.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8',
    )
    (report / 'AUDIT.json').write_text(
        json.dumps({
            key: summary[key] for key in (
                'passed', 'completed_jobs', 'expected_jobs', 'missing_jobs',
                'selection_test_used', 'frozen_ce_report_sha256',
                'manifest_checks',
            )
        }, indent=2) + '\n', encoding='utf-8',
    )
    if not passed:
        return 2
    (matrix_root / 'SELECTED_PROTOCOL_SEED42_SUCCESS').touch()
    print(json.dumps(summary, indent=2))
    return 0


def check():
    assert len(jobs()) == 16
    assert set(jobs(0, 2)).isdisjoint(jobs(1, 2))
    assert set(jobs(0, 2)) | set(jobs(1, 2)) == set(jobs())
    for spec in jobs():
        dataset, _, _ = parse_job(spec)
        command = validation_command(spec, 'cuda:0', Path('/tmp/selected-check'))
        cfg = DATASETS[dataset]
        assert command[command.index('--task_ce_mode') + 1] == cfg['task_ce_mode']
        assert command[command.index('--lambda_validation_enabled') + 1] == '1'
        assert protocol_evidence(dataset)['passed']
    print('SELECTED_PROTOCOL_SEED42_CHECK_SUCCESS')
    return 0


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check')
    jobs_parser = sub.add_parser('jobs')
    jobs_parser.add_argument('--worker', type=int)
    jobs_parser.add_argument('--workers', type=int, default=2)
    run_parser = sub.add_parser('run-job')
    run_parser.add_argument('spec')
    run_parser.add_argument('--device', required=True)
    run_parser.add_argument('--matrix-root', required=True)
    summary_parser = sub.add_parser('summarize')
    summary_parser.add_argument('--matrix-root', required=True)
    args = parser.parse_args()
    if args.command == 'check':
        return check()
    if args.command == 'jobs':
        print('\n'.join(jobs(args.worker, args.workers)))
        return 0
    if args.command == 'run-job':
        return run_job(args.spec, args.device, args.matrix_root)
    return summarize(args.matrix_root)


if __name__ == '__main__':
    raise SystemExit(main())
