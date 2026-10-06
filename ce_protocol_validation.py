"""Training-only CE-scope selection using reference CL methods."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from dataset_protocol_smoke import replace_arg
from fair_main_table_3datasets import DATASETS, base_command


CE_MODES = ('current', 'seen', 'full')
REFERENCE_METHODS = ('finetune', 'lwf', 'er')
SEED = 42
AA_TIE_MARGIN = 0.01


def jobs(worker=None, workers=2):
    specs = [
        f'{dataset}:{mode}:{method}:{SEED}'
        for mode in CE_MODES
        for method in REFERENCE_METHODS
        for dataset in DATASETS
    ]
    if worker is None:
        return specs
    return [spec for index, spec in enumerate(specs) if index % workers == worker]


def parse_job(spec):
    dataset, mode, method, seed_text = spec.split(':')
    seed = int(seed_text)
    if (
        dataset not in DATASETS or mode not in CE_MODES
        or method not in REFERENCE_METHODS or seed != SEED
    ):
        raise ValueError(f'unknown CE protocol job: {spec}')
    return dataset, mode, method, seed


def validation_command(spec, device, results_dir):
    dataset, mode, method, seed = parse_job(spec)
    cfg = DATASETS[dataset]
    command = base_command(
        f'{dataset}:{method}:{seed}', device, results_dir, smoke=False,
    )
    replace_arg(command, '--task_ce_mode', mode)
    replace_arg(command, '--lambda_validation_enabled', 1)
    replace_arg(
        command, '--lambda_validation_per_class', cfg['validation_per_class'],
    )
    replace_arg(
        command, '--lambda_validation_split_seed', cfg['validation_split_seed'],
    )
    replace_arg(command, '--exp_name', 'ce_protocol')
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


def latest_run(job_root):
    runs = sorted(Path(job_root).glob('ce_protocol_*'))
    return runs[-1] if runs else None


def metric(result, name):
    value = result['cl_metrics'][name]
    if isinstance(value, dict):
        value = value.get('mean', value.get('value'))
    return float(value)


def audit_run(spec, command, run_dir):
    dataset, mode, method, seed = parse_job(spec)
    cfg = DATASETS[dataset]
    result_path = run_dir / 'results.json'
    manifest_path = run_dir / 'validation' / 'validation_manifest.json'
    if not result_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f'{spec} is missing results or validation manifest')
    result = json.loads(result_path.read_text(encoding='utf-8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    selection = result.get('selection_audit', {})
    trajectory = result['cl_metrics'].get('AA_trajectory_taskil', [])
    checks = {
        'selection_audit_passed': selection.get('passed') is True,
        'test_not_used': selection.get('test_used_for_selection') is False,
        'training_validation_only': (
            selection.get('evaluation_source') == 'vector-train-validation'
        ),
        'all_tasks_completed': len(trajectory) == len(cfg['tasks']),
    }
    record = {
        'job': spec,
        'dataset': dataset,
        'ce_mode': mode,
        'method': method,
        'seed': seed,
        'AA_final': metric(result, 'AA_final'),
        'AA_cil': metric(result, 'AA_cil'),
        'BWT': metric(result, 'BWT'),
        'AA_final_taskil': metric(result, 'AA_final_taskil'),
        'validation_manifest_sha256': manifest['sha256'],
        'run_dir': str(run_dir),
        'checks': checks,
        'passed': all(checks.values()),
        'command': command,
    }
    if not record['passed']:
        raise RuntimeError(f'{spec} protocol audit failed: {checks}')
    return record


def job_root(matrix_root, spec):
    return Path(matrix_root) / 'runs' / spec.replace(':', '_')


def run_job(spec, device, matrix_root):
    root = job_root(matrix_root, spec)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'SUCCESS').is_file() and (root / 'record.json').is_file():
        print(f'SKIP complete {spec}')
        return 0
    command = validation_command(spec, device, root)
    (root / 'planned_protocol.json').write_text(
        json.dumps({
            'job': spec,
            'selection_source': 'training-validation',
            'selection_test_used': False,
            'command': command,
        }, indent=2) + '\n', encoding='utf-8',
    )
    with open(root / 'job.log', 'a', encoding='utf-8') as log:
        completed = subprocess.run(
            command, env=deterministic_env(), stdout=log,
            stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode:
        print(f'FAILED {spec}: exit={completed.returncode}', file=sys.stderr)
        return completed.returncode
    run_dir = latest_run(root)
    if run_dir is None:
        print(f'FAILED {spec}: no run directory', file=sys.stderr)
        return 90
    try:
        record = audit_run(spec, command, run_dir)
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


def collect_records(matrix_root):
    records, missing = [], []
    for spec in jobs():
        path = job_root(matrix_root, spec) / 'record.json'
        if not path.is_file():
            missing.append(spec)
            continue
        records.append(json.loads(path.read_text(encoding='utf-8')))
    return records, missing


def candidate_summary(dataset, mode, records):
    cfg = DATASETS[dataset]
    rows = [
        row for row in records
        if row['dataset'] == dataset and row['ce_mode'] == mode
    ]
    by_method = {row['method']: row for row in rows}
    if set(by_method) != set(REFERENCE_METHODS):
        raise RuntimeError(f'incomplete candidate {dataset}:{mode}')
    refs = [by_method['lwf'], by_method['er']]
    finetune_taskil = by_method['finetune']['AA_final_taskil']
    return {
        'dataset': dataset,
        'ce_mode': mode,
        'finetune_taskil': finetune_taskil,
        'taskil_gate': cfg['taskil_gate'],
        'taskil_gate_passed': finetune_taskil >= cfg['taskil_gate'],
        'reference_AA_final_mean': statistics.fmean(
            row['AA_final'] for row in refs
        ),
        'reference_BWT_mean': statistics.fmean(row['BWT'] for row in refs),
        'lwf_AA_final': by_method['lwf']['AA_final'],
        'lwf_BWT': by_method['lwf']['BWT'],
        'er_AA_final': by_method['er']['AA_final'],
        'er_BWT': by_method['er']['BWT'],
    }


def select_candidate(dataset, candidates):
    eligible = [row for row in candidates if row['taskil_gate_passed']]
    if not eligible:
        return None
    best_aa = max(row['reference_AA_final_mean'] for row in eligible)
    tied = [
        row for row in eligible
        if best_aa - row['reference_AA_final_mean'] <= AA_TIE_MARGIN
    ]
    return max(tied, key=lambda row: row['reference_BWT_mean'])


def summarize(matrix_root):
    matrix_root = Path(matrix_root)
    records, missing = collect_records(matrix_root)
    report = matrix_root / 'formal_report'
    report.mkdir(parents=True, exist_ok=True)
    if records:
        fields = [
            'dataset', 'ce_mode', 'method', 'seed', 'AA_final', 'AA_cil',
            'BWT', 'AA_final_taskil', 'validation_manifest_sha256', 'run_dir',
        ]
        with open(report / 'PER_RUN.csv', 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in records)
    if missing:
        (report / 'AUDIT.json').write_text(
            json.dumps({'passed': False, 'missing_jobs': missing}, indent=2) + '\n',
            encoding='utf-8',
        )
        return 2
    manifest_checks = {}
    for dataset in DATASETS:
        hashes = {
            row['validation_manifest_sha256']
            for row in records if row['dataset'] == dataset
        }
        manifest_checks[dataset] = {
            'unique_hashes': sorted(hashes), 'passed': len(hashes) == 1,
        }
    candidates = {
        dataset: [candidate_summary(dataset, mode, records) for mode in CE_MODES]
        for dataset in DATASETS
    }
    selected = {
        dataset: select_candidate(dataset, rows)
        for dataset, rows in candidates.items()
    }
    passed = (
        all(item is not None for item in selected.values())
        and all(item['passed'] for item in manifest_checks.values())
    )
    selection = {
        'passed': passed,
        'selection_source': 'training-validation',
        'selection_test_used': False,
        'ours_used_for_selection': False,
        'selection_rule': {
            'gate': 'FineTune AA-final Task-IL meets dataset threshold',
            'primary': 'maximize mean AA-final of LwF and ER',
            'tie': f'within {AA_TIE_MARGIN:.2f} AA, maximize mean BWT of LwF and ER',
        },
        'manifest_checks': manifest_checks,
        'candidates': candidates,
        'selected': selected,
    }
    (report / 'CE_PROTOCOL_SELECTION.json').write_text(
        json.dumps(selection, indent=2) + '\n', encoding='utf-8',
    )
    (report / 'AUDIT.json').write_text(
        json.dumps({
            'passed': passed,
            'completed_jobs': len(records),
            'expected_jobs': len(jobs()),
            'missing_jobs': [],
            'manifest_checks': manifest_checks,
        }, indent=2) + '\n', encoding='utf-8',
    )
    if not passed:
        return 3
    (matrix_root / 'CE_PROTOCOL_VALIDATION_SUCCESS').touch()
    print(json.dumps(selection, indent=2))
    return 0


def check():
    assert len(jobs()) == 18
    assert set(jobs(0, 2)).isdisjoint(jobs(1, 2))
    assert set(jobs(0, 2)) | set(jobs(1, 2)) == set(jobs())
    for spec in jobs():
        dataset, mode, _, _ = parse_job(spec)
        command = validation_command(spec, 'cuda:0', Path('/tmp/ce-check'))
        assert command[command.index('--task_ce_mode') + 1] == mode
        assert command[command.index('--lambda_validation_enabled') + 1] == '1'
        assert command[command.index('--lambda_validation_per_class') + 1] == str(
            DATASETS[dataset]['validation_per_class']
        )
    print('CE_PROTOCOL_VALIDATION_CHECK_SUCCESS')
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
