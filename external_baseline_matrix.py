"""Frozen two-dataset external baseline matrix, audit, and reporting."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


DATASETS = {
    'cifar100': {'num_classes': 100, 'classes_per_task': 10},
    'tinyimagenet': {'num_classes': 200, 'classes_per_task': 20},
}
METHODS = {
    'finetune': {'cl_method': 'finetune'},
    'lwf': {'cl_method': 'lwf'},
    'ewc': {'cl_method': 'ewc'},
    'er': {'cl_method': 'er'},
    'der_pp': {'cl_method': 'der_pp'},
    'gpm': {'cl_method': 'gpm'},
    'fedprotip_vfl': {'cl_method': 'fedprotip_vfl'},
    'proto_uniform': {
        'cl_method': 'proto_evolve',
        'dep_tracking_enabled': 1,
        'party_kd_enabled': 1,
        'party_kd_mode': 'uniform',
        'expected_party_kd_variant': 'uniform',
        'party_kd_lambda': 1.0,
    },
}
SEEDS = (42, 43)
SUPPORTED_SEEDS = (*SEEDS, 44)
DATA_PATH = '/home/chase/Yangxx/VF-CL/data'
SEED44_EXTERNAL_METHODS = (
    'er', 'lwf', 'der_pp', 'ewc', 'gpm', 'fedprotip_vfl', 'finetune',
)


def parse_job(spec):
    dataset, method, seed = spec.split(':')
    if (dataset not in DATASETS or method not in METHODS
            or int(seed) not in SUPPORTED_SEEDS):
        raise ValueError(f'unknown job {spec}')
    return dataset, method, int(seed)


def job_specs():
    return [
        f'{dataset}:{method}:{seed}'
        for dataset in DATASETS
        for method in METHODS
        for seed in SEEDS
    ]


def seed44_job_specs():
    return [f'cifar100:{method}:44' for method in SEED44_EXTERNAL_METHODS]


def claim_next_job(claim_root, specs=None):
    claim_root = Path(claim_root)
    claim_root.mkdir(parents=True, exist_ok=True)
    for spec in specs or job_specs():
        try:
            (claim_root / spec.replace(':', '_')).mkdir()
        except FileExistsError:
            continue
        return spec
    return None


def expected_config(spec):
    dataset, method, seed = parse_job(spec)
    values = {
        'data': dataset,
        'data_path': DATA_PATH,
        'num_classes': DATASETS[dataset]['num_classes'],
        'num_tasks': 10,
        'classes_per_task': DATASETS[dataset]['classes_per_task'],
        'unlearn_after_tasks': [99],
        'unlearn_classes': [[0]],
        'num_parties': 4,
        'model_type': 'resnet18',
        'aggregation': 'sum',
        'epochs_per_task': 50,
        'batch_size': 64,
        'num_workers': 2,
        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'cl_method': METHODS[method]['cl_method'],
        'ul_method': 'retrain',
        'deterministic': 1,
        'data_flow_audit': 1,
        'bic_enabled': 1,
        'bic_per_class': 25,
        'bic_split_seed': 20260722,
        'bic_lr': 0.05,
        'bic_steps': 1000,
        'bic_fit_mode': 'joint_each_stage' if method == 'proto_uniform' else 'joint_final',
        'save_task_checkpoints': (
            3 if seed == 44 and method in SEED44_EXTERNAL_METHODS else 2
        ),
        'seed': seed,
        'device': 'cuda:0',
        'replay_mode': 'prototype',
    }
    values.update({key: value for key, value in METHODS[method].items()
                   if key != 'cl_method'})
    if method == 'lwf':
        values.update(lwf_temperature=2.0, lwf_lambda=1.0,
                      lwf_ce_newonly=True, feat_distill_weight=0.0)
    elif method == 'ewc':
        values.update(ewc_lambda=1000.0, ewc_fisher_decay=0.9,
                      ewc_fisher_samples=1024, lwf_ce_newonly=True,
                      feat_distill_weight=0.0)
    elif method == 'er':
        values.update(er_per_class=300, er_batch=64, er_alpha=1.0)
    elif method == 'der_pp':
        values.update(der_buffer_size=20 * values['num_classes'], der_batch=64,
                      der_alpha=0.5, der_beta=0.5)
    elif method == 'gpm':
        values.update(gpm_threshold=0.95)
    return values


def command_for(spec, matrix_root, repo_root, python):
    dataset, method, seed = parse_job(spec)
    values = expected_config(spec)
    values['results_dir'] = str(Path(matrix_root) / 'runs')
    values['exp_name'] = f'external_{dataset}_{method}_seed{seed}'
    command = [str(python), str(Path(repo_root) / 'main.py')]
    for key, value in values.items():
        if key == 'unlearn_after_tasks':
            cli_value = ','.join(str(item) for item in value)
        elif key == 'unlearn_classes':
            cli_value = ';'.join(','.join(str(item) for item in group)
                                 for group in value)
        else:
            cli_value = str(value)
        command.extend((f'--{key}', cli_value))
    return command


def _normalized_config(config, expected):
    normalized = {}
    for key, value in expected.items():
        actual = config.get(key)
        if isinstance(value, list):
            actual = actual
        elif isinstance(value, bool):
            actual = bool(actual)
        elif isinstance(value, int):
            actual = int(actual)
        elif isinstance(value, float):
            actual = float(actual)
        else:
            actual = str(actual)
        normalized[key] = actual
    return normalized


def audit_run(spec, run_dir, code_commit):
    run_dir = Path(run_dir)
    config_path = run_dir / 'config.json'
    result_path = run_dir / 'results.json'
    checkpoint = run_dir / 'checkpoints' / 'event_9_CIL.pt'
    if not config_path.is_file() or not result_path.is_file():
        raise ValueError(f'{spec}: missing config or results')
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f'{spec}: missing final checkpoint')
    config = json.loads(config_path.read_text())
    expected = expected_config(spec)
    actual = _normalized_config(config, expected)
    if actual != expected:
        mismatch = {key: (actual[key], expected[key]) for key in expected
                    if actual[key] != expected[key]}
        raise ValueError(f'{spec}: protocol mismatch {mismatch}')
    result = json.loads(result_path.read_text())
    for key in ('AA_final', 'AA_cil', 'BWT'):
        if key not in result.get('cl_metrics', {}):
            raise ValueError(f'{spec}: missing metric {key}')
    calibration = result.get('calibration_audit', {})
    if not calibration.get('passed') or calibration.get('test_used_for_fit'):
        raise ValueError(f'{spec}: calibration privacy audit failed')
    _, method, _ = parse_job(spec)
    if method == 'proto_uniform' and len(result.get('bic_history', [])) != 10:
        raise ValueError(f'{spec}: calibrated trajectory must have 10 stages')
    if method == 'fedprotip_vfl':
        for key in ('AA_final_class_il_global', 'AA_final_task_prediction'):
            if key not in result['cl_metrics']:
                raise ValueError(f'{spec}: missing FedProTIP companion {key}')
    protocol = {'job': spec, 'code_commit': code_commit, 'config': expected}
    encoded = json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()
    record = {
        **protocol,
        'protocol_sha256': hashlib.sha256(encoded).hexdigest(),
        'run_dir': str(run_dir.resolve()),
        'result': str(result_path.resolve()),
        'checkpoint': str(checkpoint.resolve()),
        'calibration_manifest_sha256': calibration.get('manifest_sha256'),
    }
    (run_dir / 'protocol_digest.json').write_text(
        json.dumps(record, indent=2, sort_keys=True) + '\n'
    )
    return record


def recorded_code_commit(spec, run_dir):
    record = json.loads((Path(run_dir) / 'protocol_digest.json').read_text())
    if record.get('job') != spec:
        raise ValueError(f'{spec}: protocol digest job mismatch')
    commit = record.get('code_commit')
    if not isinstance(commit, str) or not commit.strip():
        raise ValueError(f'{spec}: protocol digest has no code commit')
    return commit


def _is_complete_run(root, expected):
    root = Path(root)
    config_path = root / 'config.json'
    result_path = root / 'results.json'
    checkpoint = root / 'checkpoints' / f"event_{expected['num_tasks'] - 1}_CIL.pt"
    if not config_path.is_file() or not result_path.is_file():
        return False
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        return False
    try:
        config = json.loads(config_path.read_text())
        result = json.loads(result_path.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if _normalized_config(config, expected) != expected:
        return False
    return all(
        key in result.get('cl_metrics', {})
        for key in ('AA_final', 'AA_cil', 'BWT')
    )


def find_complete(spec, matrix_root):
    dataset, method, seed = parse_job(spec)
    prefix = f'external_{dataset}_{method}_seed{seed}_'
    roots = sorted((Path(matrix_root) / 'runs').glob(prefix + '*'))
    expected = expected_config(spec)
    complete = [root for root in roots if _is_complete_run(root, expected)]
    if not complete:
        raise FileNotFoundError(spec)
    return complete[-1]


def find_incomplete(spec, matrix_root):
    dataset, method, seed = parse_job(spec)
    prefix = f'external_{dataset}_{method}_seed{seed}_'
    roots = sorted((Path(matrix_root) / 'runs').glob(prefix + '*'), reverse=True)
    expected = expected_config(spec)
    for root in roots:
        if _is_complete_run(root, expected):
            continue
        config_path = root / 'config.json'
        checkpoints = root / 'checkpoints'
        if not config_path.is_file() or not (
            (checkpoints / 'resume_latest.pt').is_file()
            or (checkpoints / f"event_{expected['num_tasks'] - 1}_CIL.pt").is_file()
        ):
            continue
        try:
            config = json.loads(config_path.read_text())
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if _normalized_config(config, expected) == expected:
            return root
    raise FileNotFoundError(spec)


def calibrated_trajectory(history):
    per_stage = [record['paired']['calibrated']['per_task_accuracy']
                 for record in history]
    aa = [statistics.fmean(values.values()) for values in per_stage]
    bwt = []
    for previous, current in zip(per_stage, per_stage[1:]):
        bwt.extend(current[key] - previous[key] for key in previous
                   if key in current)
    return {
        'AA_final': aa[-1],
        'AA_cil': statistics.fmean(aa),
        'BWT': statistics.fmean(bwt) if bwt else 0.0,
        'final_per_task': per_stage[-1],
    }


def _raw_metrics(result):
    final = result['task_acc_history'][-1]['per_task_accs']
    return {
        'AA_final': result['cl_metrics']['AA_final'],
        'AA_cil': result['cl_metrics']['AA_cil'],
        'BWT': result['cl_metrics']['BWT'],
        'final_per_task': final,
        'training_hours': sum(item.get('train_time', 0.0)
                              for item in result.get('step_results', [])
                              if item.get('type') == 'CIL') / 3600.0,
        'communication_mb': sum(item.get('megabytes_transmitted', 0.0)
                                for item in result.get('comm_stats', [])),
    }


def _mean_std(values):
    return statistics.fmean(values), statistics.pstdev(values)


def _rows(matrix_root):
    records = {}
    for spec in job_specs():
        dataset, method, seed = parse_job(spec)
        run_dir = find_complete(spec, matrix_root)
        result = json.loads((run_dir / 'results.json').read_text())
        records.setdefault((dataset, method), {})[seed] = _raw_metrics(result)
        if method == 'proto_uniform':
            calibrated = calibrated_trajectory(result['bic_history'])
            calibrated['training_hours'] = records[(dataset, method)][seed]['training_hours']
            calibrated['communication_mb'] = records[(dataset, method)][seed]['communication_mb']
            records.setdefault((dataset, 'proto_uniform_calibrated'), {})[seed] = calibrated
    labels = {
        'finetune': 'FineTune', 'lwf': 'LwF', 'ewc': 'EWC', 'er': 'ER',
        'der_pp': 'DER++', 'gpm': 'GPM', 'fedprotip_vfl': 'FedProTIP-VFL',
        'proto_uniform': 'ProtoEvolve + Uniform PartyKD',
        'proto_uniform_calibrated': 'ProtoEvolve + Uniform PartyKD + joint alpha/beta',
    }
    rows = []
    for (dataset, method), by_seed in sorted(records.items()):
        if set(by_seed) != set(SEEDS):
            raise ValueError(f'{dataset}/{method}: incomplete seeds')
        row = {'dataset': dataset, 'method_key': method, 'method': labels[method]}
        for metric in ('AA_final', 'AA_cil', 'BWT', 'training_hours',
                       'communication_mb'):
            mean, std = _mean_std([by_seed[seed][metric] for seed in SEEDS])
            row[metric + '_mean'], row[metric + '_std'] = mean, std
        tasks = sorted(by_seed[42]['final_per_task'])
        row['final_per_task_mean'] = {
            task: statistics.fmean(
                by_seed[seed]['final_per_task'][task] for seed in SEEDS
            )
            for task in tasks
        }
        row['replay'] = method in ('er', 'der_pp')
        row['calibration'] = method == 'proto_uniform_calibrated'
        row['task_prediction'] = method == 'fedprotip_vfl'
        rows.append(row)
    return rows


def summarize(matrix_root):
    matrix_root = Path(matrix_root)
    rows = _rows(matrix_root)
    fields = [
        'dataset', 'method_key', 'method', 'AA_final_mean', 'AA_final_std',
        'AA_cil_mean', 'AA_cil_std', 'BWT_mean', 'BWT_std',
        'training_hours_mean', 'training_hours_std',
        'communication_mb_mean', 'communication_mb_std',
        'replay', 'calibration', 'task_prediction', 'final_per_task_mean',
    ]
    with (matrix_root / 'MAIN_TABLE.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                'final_per_task_mean': json.dumps(
                    row['final_per_task_mean'], sort_keys=True
                ),
            })
    lines = [
        '| Dataset | Method | AA_final Class-IL | AA_cil | BWT | Train h | Comm MiB |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | "
            f"{row['AA_final_mean']:.4f}±{row['AA_final_std']:.4f} | "
            f"{row['AA_cil_mean']:.4f}±{row['AA_cil_std']:.4f} | "
            f"{row['BWT_mean']:.4f}±{row['BWT_std']:.4f} | "
            f"{row['training_hours_mean']:.2f} | "
            f"{row['communication_mb_mean']:.1f} |"
        )
    (matrix_root / 'MAIN_TABLE.md').write_text('\n'.join(lines) + '\n')
    summary = {'jobs': 32, 'rows': rows}
    (matrix_root / 'SUMMARY.json').write_text(
        json.dumps(summary, indent=2) + '\n'
    )
    (matrix_root / 'ORIGINAL_FEDPROTIP_REFERENCE.md').write_text(
        '# Original FedProTIP reference (non-comparable)\n\n'
        '| Federation | Clients | Backbone | Pretraining | Role |\n'
        '|---|---:|---|---|---|\n'
        '| Horizontal | 5 | ResNet18 | Yes | Protocol reference only |\n\n'
        'The unified rank uses FedProTIP-VFL, not this original implementation.\n'
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('jobs')
    sub.add_parser('seed44-jobs')
    claim = sub.add_parser('claim')
    claim.add_argument('--claims-root', required=True, type=Path)
    seed44_claim = sub.add_parser('seed44-claim')
    seed44_claim.add_argument('--claims-root', required=True, type=Path)
    command = sub.add_parser('command')
    command.add_argument('--job', required=True)
    command.add_argument('--matrix-root', required=True, type=Path)
    command.add_argument('--repo-root', required=True, type=Path)
    command.add_argument('--python', required=True, type=Path)
    find = sub.add_parser('find')
    find.add_argument('--job', required=True)
    find.add_argument('--matrix-root', required=True, type=Path)
    incomplete = sub.add_parser('find-incomplete')
    incomplete.add_argument('--job', required=True)
    incomplete.add_argument('--matrix-root', required=True, type=Path)
    audit = sub.add_parser('audit')
    audit.add_argument('--job', required=True)
    audit.add_argument('--run-dir', required=True, type=Path)
    audit.add_argument('--code-commit', required=True)
    provenance = sub.add_parser('provenance')
    provenance.add_argument('--job', required=True)
    provenance.add_argument('--run-dir', required=True, type=Path)
    report = sub.add_parser('summarize')
    report.add_argument('--matrix-root', required=True, type=Path)
    args = parser.parse_args()
    if args.action == 'jobs':
        print('\n'.join(job_specs()))
    elif args.action == 'seed44-jobs':
        print('\n'.join(seed44_job_specs()))
    elif args.action == 'claim':
        job = claim_next_job(args.claims_root)
        if job is None:
            raise SystemExit(1)
        print(job)
    elif args.action == 'seed44-claim':
        job = claim_next_job(args.claims_root, seed44_job_specs())
        if job is None:
            raise SystemExit(1)
        print(job)
    elif args.action == 'command':
        print('\n'.join(command_for(
            args.job, args.matrix_root, args.repo_root, args.python
        )))
    elif args.action == 'find':
        print(find_complete(args.job, args.matrix_root))
    elif args.action == 'find-incomplete':
        print(find_incomplete(args.job, args.matrix_root))
    elif args.action == 'audit':
        print(json.dumps(
            audit_run(args.job, args.run_dir, args.code_commit), sort_keys=True
        ))
    elif args.action == 'provenance':
        print(recorded_code_commit(args.job, args.run_dir))
    else:
        summarize(args.matrix_root)


if __name__ == '__main__':
    main()
