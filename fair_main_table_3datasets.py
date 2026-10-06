"""Audited fair-main-table matrix for CIFAR-100, ISOLET, and UPMC Food-101."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def deployment_paths(module_file=__file__, executable=sys.executable,
                     common_dir=None):
    worktree = Path(module_file).resolve().parent
    if common_dir is None:
        common_dir = subprocess.check_output(
            ['git', '-C', str(worktree), 'rev-parse', '--git-common-dir'],
            text=True,
        ).strip()
    common = Path(common_dir)
    if not common.is_absolute():
        common = (worktree / common).resolve()
    else:
        common = common.resolve()
    if common.name != '.git' or not common.is_dir():
        raise RuntimeError(f'invalid git common directory: {common}')
    return common.parent, worktree, Path(executable).resolve()


DEPLOYMENT_ROOT, _, _ = deployment_paths()
ROOT = Path('/home/chase/Yangxx/VF-CL')
WORKTREE = ROOT / '.worktrees' / 'fair-main-table-3datasets'
PYTHON = Path('/home/chase/anaconda3/envs/mlz_3.9/bin/python')
SEEDS = (42, 43, 44)
METHODS = (
    'finetune', 'lwf', 'ewc', 'er', 'der_pp', 'gpm', 'fedprotip_vfl', 'ours',
)
CIFAR100_REUSE = {
    'external_table': ROOT / 'results' / 'formal_cifar100_metrics_20260728' / 'FORMAL_CIFAR100_TABLE.csv',
    'external_per_run': ROOT / 'results' / 'formal_cifar100_metrics_20260728' / 'FORMAL_CIFAR100_PER_RUN.csv',
    'external_audit': ROOT / 'results' / 'formal_cifar100_metrics_20260728' / 'FORMAL_CIFAR100_AUDIT.json',
    'external_success': ROOT / 'results' / 'external_baselines_20260723_214415' / 'CIFAR100_EXTERNAL_SEED44_SUCCESS',
    'ours_summary': ROOT / 'results' / 'cifar100_aa_final_improvement_20260805_085859' / 'stage_e_formal' / 'FORMAL_SUMMARY.json',
    'ours_success': ROOT / 'results' / 'cifar100_aa_final_improvement_20260805_085859' / 'stage_e_formal' / 'FORMAL_SUCCESS',
}
CE_PROTOCOL_SELECTION_RELATIVE = (
    Path('results/ce_protocol_validation_20260809_133213_1515853')
    / 'formal_report/CE_PROTOCOL_SELECTION.json'
)
CE_PROTOCOL_SELECTION = {
    'report': DEPLOYMENT_ROOT / CE_PROTOCOL_SELECTION_RELATIVE,
    'source_report': ROOT / CE_PROTOCOL_SELECTION_RELATIVE,
    'sha256': '07f62537f3b203f5575db63042065d9f044eb1acc91d9b8a952b6586a0fbb368',
    'code_commit': '2750942',
}
REPLAY_KIND = {
    'finetune': 'none',
    'lwf': 'distillation',
    'ewc': 'regularization',
    'er': 'exemplar replay',
    'der_pp': 'exemplar+logit replay',
    'gpm': 'subspace constraint',
    'fedprotip_vfl': 'prototype replay',
    'ours': 'balanced prototype replay',
}


def grouped_tasks(num_classes, num_tasks):
    base, remainder = divmod(num_classes, num_tasks)
    tasks = []
    start = 0
    for task in range(num_tasks):
        width = base + (1 if task < remainder else 0)
        tasks.append(list(range(start, start + width)))
        start += width
    assert start == num_classes
    return tasks


DATASETS = {
    'isolet': {
        'vector_npz': ROOT / 'data' / 'isolet' / 'isolet_vfl.npz',
        'metadata': ROOT / 'data' / 'isolet' / 'isolet_vfl.metadata.json',
        'num_classes': 26,
        'tasks': grouped_tasks(26, 13),
        'num_parties': 4,
        'epochs_per_task': 50,
        'batch_size': 128,
        'optimizer': 'adamw',
        'lr': 0.001,
        'bottom_lr_scale': 1.0,
        'weight_decay': 0.0001,
        'task_ce_mode': 'current',
        'validation_per_class': 40,
        'validation_split_seed': 20260809,
        'taskil_gate': 0.95,
    },
    'upmc_food101': {
        'vector_npz': ROOT / 'data' / 'upmc_food101' / 'upmc_food101_vfl.npz',
        'metadata': ROOT / 'data' / 'upmc_food101' / 'upmc_food101_vfl.metadata.json',
        'num_classes': 101,
        'tasks': grouped_tasks(101, 10),
        'num_parties': 2,
        'epochs_per_task': 20,
        'batch_size': 128,
        'optimizer': 'adamw',
        'lr': 0.003,
        'bottom_lr_scale': 0.25,
        'weight_decay': 0.0001,
        'task_ce_mode': 'current',
        'validation_per_class': 64,
        'validation_split_seed': 20260809,
        'taskil_gate': 0.80,
    },
}


METHOD_FLAGS = {
    'finetune': ['--cl_method', 'finetune'],
    'lwf': ['--cl_method', 'lwf'],
    'ewc': ['--cl_method', 'ewc'],
    'er': ['--cl_method', 'er', '--er_per_class', '20', '--er_batch', '64'],
    'der_pp': ['--cl_method', 'der_pp'],
    'gpm': ['--cl_method', 'gpm'],
    'fedprotip_vfl': ['--cl_method', 'fedprotip_vfl'],
    'ours': [
        '--cl_method', 'proto_evolve',
        '--dep_tracking_enabled', '1',
        '--party_kd_enabled', '1',
        '--party_kd_mode', 'uniform',
        '--expected_party_kd_variant', 'uniform',
        '--party_kd_lambda', '1.0',
        '--proto_replay_loss_norm', 'sample_mean',
        '--proto_replay_ratio', '1.0',
        '--proto_lambda_a', '0.15',
        '--fim_freeze_frac', '0',
        '--head_consolidation_enabled', '1',
        '--head_consolidation_mode', 'task_class_bias',
        '--head_consolidation_regularization', '0.01',
        '--head_consolidation_lr', '0.03',
        '--head_consolidation_steps', '600',
        '--head_consolidation_class_regularization', '0.01',
        '--head_consolidation_task_regularization', '0.01',
        '--head_consolidation_task_weight', '1.3',
        '--head_consolidation_samples_per_class', '20',
        '--head_consolidation_schedule', 'final',
        '--distill_weight', '0.25',
        '--feat_distill_weight', '0.05',
        '--current_supcon_weight', '0',
    ],
}


def custom_tasks(tasks):
    return '|'.join(','.join(str(label) for label in task) for task in tasks)


def jobs(worker=None, workers=None):
    specs = [
        f'{dataset}:{method}:{seed}'
        for seed in SEEDS
        for method in METHODS
        for dataset in DATASETS
    ]
    if worker is None:
        return specs
    return [spec for index, spec in enumerate(specs) if index % workers == worker]


def parse_job(spec):
    dataset, method, seed_text = spec.split(':')
    seed = int(seed_text)
    if dataset not in DATASETS or method not in METHODS or seed not in SEEDS:
        raise ValueError(f'unknown job: {spec}')
    return dataset, method, seed


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def base_command(spec, device, results_dir, resume_run_dir=None, smoke=False):
    dataset, method, seed = parse_job(spec)
    cfg = DATASETS[dataset]
    num_tasks = len(cfg['tasks'])
    epochs = 1 if smoke else cfg['epochs_per_task']
    tasks = cfg['tasks'][:2] if smoke else cfg['tasks']
    num_tasks = len(tasks)
    command = [
        str(PYTHON), str(WORKTREE / 'main.py'),
        '--data', 'tabvfl',
        '--data_path', str(ROOT / 'data'),
        '--vector_npz', str(cfg['vector_npz']),
        '--num_classes', str(cfg['num_classes']),
        '--num_tasks', str(num_tasks),
        '--custom_tasks', custom_tasks(tasks),
        '--classes_per_task', str(max(len(task) for task in tasks)),
        '--unlearn_after_tasks', '999',
        '--unlearn_classes', '0',
        '--num_parties', str(cfg['num_parties']),
        '--model_type', 'mlp',
        '--aggregation', 'concat',
        '--epochs_per_task', str(epochs),
        '--batch_size', str(cfg['batch_size']),
        '--num_workers', '2',
        '--optimizer', cfg['optimizer'],
        '--lr', str(cfg['lr']),
        '--bottom_lr_scale', str(cfg['bottom_lr_scale']),
        '--task_ce_mode', cfg['task_ce_mode'],
        '--momentum', '0.9',
        '--weight_decay', str(cfg['weight_decay']),
        '--device', device,
        '--ul_method', 'retrain',
        '--replay_mode', 'prototype',
        '--deterministic', '1',
        '--data_flow_audit', '1',
        '--bic_enabled', '0',
        '--lambda_validation_enabled', '0',
        '--der_buffer_size', str(20 * cfg['num_classes']),
        '--der_batch', '64',
        '--save_task_checkpoints', '3',
        '--seed', str(seed),
        '--results_dir', str(results_dir),
        '--exp_name', 'run',
    ]
    command.extend(METHOD_FLAGS[method])
    if resume_run_dir:
        command.extend(['--resume_run_dir', str(resume_run_dir)])
    return command


def run_directories(job_root):
    return sorted(job_root.glob('run_*'))


def complete_run(run_dir, expected_tasks):
    return (
        (run_dir / 'results.json').is_file()
        and (run_dir / 'checkpoints' / f'event_{expected_tasks - 1}_CIL.pt').is_file()
    )


def find_resume(run_dirs):
    for run_dir in reversed(run_dirs):
        if (run_dir / 'config.json').is_file() and (
            (run_dir / 'checkpoints' / 'resume_latest.pt').is_file()
        ):
            return run_dir
    return None


def protocol_record(spec, command, smoke=False):
    dataset, method, seed = parse_job(spec)
    cfg = DATASETS[dataset]
    return {
        'job': spec,
        'dataset': dataset,
        'method': method,
        'replay_kind': REPLAY_KIND[method],
        'seed': seed,
        'num_classes': cfg['num_classes'],
        'tasks': cfg['tasks'][:2] if smoke else cfg['tasks'],
        'num_parties': cfg['num_parties'],
        'feature_cache': str(cfg['vector_npz']),
        'feature_cache_sha256': file_sha256(cfg['vector_npz']),
        'feature_metadata_sha256': file_sha256(cfg['metadata']),
        'test_used_for_fit': False,
        'selection_test_used': False,
        'replay_budget_items_per_class': 20,
        'dataset_training_protocol': {
            'optimizer': cfg['optimizer'],
            'lr': cfg['lr'],
            'bottom_lr_scale': cfg['bottom_lr_scale'],
            'weight_decay': cfg['weight_decay'],
            'task_ce_mode': cfg['task_ce_mode'],
        },
        'ce_protocol_selection_evidence': {
            'report': str(CE_PROTOCOL_SELECTION['source_report']),
            'local_report': str(CE_PROTOCOL_SELECTION['report']),
            'expected_sha256': CE_PROTOCOL_SELECTION['sha256'],
            'actual_sha256': file_sha256(CE_PROTOCOL_SELECTION['report']),
            'selection_code_commit': CE_PROTOCOL_SELECTION['code_commit'],
        },
        'head_consolidation': {
            'enabled': method == 'ours',
            'mode': 'task_class_bias' if method == 'ours' else None,
            'source': 'balanced_current_encoder_raw_replay' if method == 'ours' else None,
            'samples_per_class': 20 if method == 'ours' else 0,
            'schedule': 'final' if method == 'ours' else None,
            'class_regularization': 0.01 if method == 'ours' else None,
            'task_regularization': 0.01 if method == 'ours' else None,
            'task_weight': 1.3 if method == 'ours' else None,
            'replay_selection': 'normalized_feature_herding' if method == 'ours' else None,
            'persistent_raw_example_count_per_class': 20 if method == 'ours' else 0,
            'persistent_embedding_count': 0,
            'validation_used': False,
            'test_used': False,
        },
        'command': command,
        'code_commit': subprocess.check_output(
            ['git', '-C', str(WORKTREE), 'rev-parse', 'HEAD'], text=True,
        ).strip(),
    }


def run_job(spec, device, matrix_root, smoke=False):
    dataset, method, _ = parse_job(spec)
    cfg = DATASETS[dataset]
    for required in (cfg['vector_npz'], cfg['metadata']):
        if not required.is_file():
            raise FileNotFoundError(required)
    suffix = '_smoke' if smoke else ''
    job_root = Path(matrix_root) / ('smoke' if smoke else 'runs') / (spec.replace(':', '_') + suffix)
    job_root.mkdir(parents=True, exist_ok=True)
    expected_tasks = 2 if smoke else len(cfg['tasks'])
    existing = run_directories(job_root)
    complete = [path for path in existing if complete_run(path, expected_tasks)]
    if complete:
        print(f'SKIP complete {spec}: {complete[-1]}')
        return 0

    resume = find_resume(existing)
    command = base_command(spec, device, job_root, resume, smoke=smoke)
    record = protocol_record(spec, command, smoke=smoke)
    (job_root / 'protocol.json').write_text(
        json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8',
    )
    log_path = job_root / 'job.log'
    env = os.environ.copy()
    env.update({
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'PYTHONHASHSEED': str(parse_job(spec)[2]),
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    })
    print(f'RUN {spec} on {device}' + (f' resume={resume}' if resume else ''))
    with open(log_path, 'a', encoding='utf-8') as log:
        log.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {' '.join(command)}\n")
        completed = subprocess.run(command, cwd=WORKTREE, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        print(f'FAILED {spec}: exit={completed.returncode}', file=sys.stderr)
        return completed.returncode

    complete = [path for path in run_directories(job_root) if complete_run(path, expected_tasks)]
    if not complete:
        print(f'FAILED audit {spec}: no complete run', file=sys.stderr)
        return 90
    result = json.loads((complete[-1] / 'results.json').read_text(encoding='utf-8'))
    metrics = result.get('cl_metrics', {})
    required_metrics = ('AA_final', 'AA_cil', 'BWT')
    if any(name not in metrics for name in required_metrics):
        print(f'FAILED audit {spec}: missing metrics', file=sys.stderr)
        return 91
    (job_root / 'SUCCESS').touch()
    print(f'COMPLETE {spec}: {complete[-1]}')
    return 0


def metric_value(result, key):
    value = result['cl_metrics'][key]
    if isinstance(value, dict):
        return float(value.get('mean', value.get('value')))
    return float(value)


def summarize(matrix_root):
    matrix_root = Path(matrix_root)
    records = []
    missing = []
    for spec in jobs():
        dataset, method, seed = parse_job(spec)
        job_root = matrix_root / 'runs' / spec.replace(':', '_')
        runs = [path for path in run_directories(job_root) if complete_run(path, len(DATASETS[dataset]['tasks']))]
        if not runs:
            missing.append(spec)
            continue
        result = json.loads((runs[-1] / 'results.json').read_text(encoding='utf-8'))
        records.append({
            'dataset': dataset,
            'method': method,
            'replay_kind': REPLAY_KIND[method],
            'seed': seed,
            'AA_final': metric_value(result, 'AA_final'),
            'AA_cil': metric_value(result, 'AA_cil'),
            'BWT': metric_value(result, 'BWT'),
            'run_dir': str(runs[-1]),
        })
    report = matrix_root / 'formal_report'
    report.mkdir(parents=True, exist_ok=True)
    with open(report / 'PER_RUN.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else [
            'dataset', 'method', 'replay_kind', 'seed', 'AA_final', 'AA_cil', 'BWT', 'run_dir',
        ])
        writer.writeheader()
        writer.writerows(records)
    grouped = []
    for dataset in DATASETS:
        for method in METHODS:
            rows = [r for r in records if r['dataset'] == dataset and r['method'] == method]
            if len(rows) != len(SEEDS):
                continue
            item = {'dataset': dataset, 'method': method, 'replay_kind': REPLAY_KIND[method]}
            for metric in ('AA_final', 'AA_cil', 'BWT'):
                values = [r[metric] for r in rows]
                item[metric + '_mean'] = statistics.fmean(values)
                item[metric + '_std'] = statistics.pstdev(values)
            grouped.append(item)
    with open(report / 'FAIR_MAIN_TABLE.csv', 'w', newline='', encoding='utf-8') as handle:
        fieldnames = list(grouped[0]) if grouped else [
            'dataset', 'method', 'replay_kind', 'AA_final_mean', 'AA_final_std',
            'AA_cil_mean', 'AA_cil_std', 'BWT_mean', 'BWT_std',
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(grouped)
    audit = {'passed': not missing, 'missing_jobs': missing, 'completed_jobs': len(records), 'expected_jobs': len(jobs())}
    (report / 'AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    if missing:
        print('\n'.join(missing))
        return 2
    (matrix_root / 'FAIR_MAIN_TABLE_SUCCESS').touch()
    return 0


def audit_cifar_reuse(matrix_root):
    missing = [str(path) for path in CIFAR100_REUSE.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('missing CIFAR-100 formal artifacts: ' + ', '.join(missing))
    destination = Path(matrix_root) / 'cifar100_reuse'
    destination.mkdir(parents=True, exist_ok=True)
    record = {
        'dataset': 'cifar100',
        'policy': 'reuse completed three-seed formal runs; do not spend GPU on duplicates',
        'artifacts': {
            name: {'path': str(path), 'sha256': file_sha256(path)}
            for name, path in CIFAR100_REUSE.items()
        },
        'passed': True,
    }
    (destination / 'AUDIT.json').write_text(
        json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8',
    )
    (destination / 'CIFAR100_REUSE_SUCCESS').touch()
    print('CIFAR100_REUSE_AUDIT_SUCCESS')
    return 0


def check():
    assert len(jobs()) == 48
    for dataset, cfg in DATASETS.items():
        flattened = [label for task in cfg['tasks'] for label in task]
        assert flattened == list(range(cfg['num_classes'])), dataset
        assert len(set(flattened)) == cfg['num_classes']
    report_path = CE_PROTOCOL_SELECTION['report']
    assert report_path.is_file(), report_path
    assert file_sha256(report_path) == CE_PROTOCOL_SELECTION['sha256']
    selection = json.loads(report_path.read_text(encoding='utf-8'))
    assert selection['passed'] is True
    assert selection['selection_test_used'] is False
    assert selection['ours_used_for_selection'] is False
    for dataset, cfg in DATASETS.items():
        assert (
            selection['selected'][dataset]['ce_mode'] == cfg['task_ce_mode']
        ), dataset
    for spec in jobs():
        dataset, method, _ = parse_job(spec)
        command = base_command(spec, 'cuda:0', Path('/tmp/check'))
        assert '--bic_enabled' in command and command[command.index('--bic_enabled') + 1] == '0'
        assert '--lambda_validation_enabled' in command
        assert str(20 * DATASETS[dataset]['num_classes']) in command
        for flag, key in (('--optimizer', 'optimizer'), ('--lr', 'lr'),
                          ('--bottom_lr_scale', 'bottom_lr_scale'),
                          ('--weight_decay', 'weight_decay'), ('--task_ce_mode', 'task_ce_mode')):
            assert command[command.index(flag) + 1] == str(DATASETS[dataset][key])
        if method == 'er':
            assert command[command.index('--er_per_class') + 1] == '20'
        if method == 'ours':
            expected = {
                '--head_consolidation_enabled': '1',
                '--head_consolidation_mode': 'task_class_bias',
                '--head_consolidation_regularization': '0.01',
                '--head_consolidation_lr': '0.03',
                '--head_consolidation_steps': '600',
                '--head_consolidation_class_regularization': '0.01',
                '--head_consolidation_task_regularization': '0.01',
                '--head_consolidation_task_weight': '1.3',
                '--head_consolidation_samples_per_class': '20',
                '--head_consolidation_schedule': 'final',
            }
            assert all(command[command.index(flag) + 1] == value for flag, value in expected.items())
    print('FAIR_MAIN_TABLE_3DATASETS_CHECK_SUCCESS')
    return 0


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check')
    audit_parser = sub.add_parser('audit-cifar')
    audit_parser.add_argument('--matrix-root', required=True)
    jobs_parser = sub.add_parser('jobs')
    jobs_parser.add_argument('--worker', type=int)
    jobs_parser.add_argument('--workers', type=int, default=2)
    run_parser = sub.add_parser('run-job')
    run_parser.add_argument('spec')
    run_parser.add_argument('--device', required=True)
    run_parser.add_argument('--matrix-root', required=True)
    run_parser.add_argument('--smoke', action='store_true')
    summarize_parser = sub.add_parser('summarize')
    summarize_parser.add_argument('--matrix-root', required=True)
    args = parser.parse_args()
    if args.command == 'check':
        return check()
    if args.command == 'audit-cifar':
        return audit_cifar_reuse(args.matrix_root)
    if args.command == 'jobs':
        print('\n'.join(jobs(args.worker, args.workers)))
        return 0
    if args.command == 'run-job':
        return run_job(args.spec, args.device, args.matrix_root, smoke=args.smoke)
    return summarize(args.matrix_root)


if __name__ == '__main__':
    raise SystemExit(main())
