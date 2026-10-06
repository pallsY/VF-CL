"""Training-only smoke for dataset-level vector training protocols."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from fair_main_table_3datasets import DATASETS, base_command


def replace_arg(command, flag, value):
    if flag in command:
        index = command.index(flag)
        command[index + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def validation_command(dataset, device, output_root, epochs=3):
    cfg = DATASETS[dataset]
    results_dir = Path(output_root) / dataset
    command = base_command(
        f'{dataset}:finetune:42', device, results_dir, smoke=True,
    )
    replace_arg(command, '--epochs_per_task', epochs)
    replace_arg(command, '--lambda_validation_enabled', 1)
    replace_arg(
        command, '--lambda_validation_per_class', cfg['validation_per_class'],
    )
    replace_arg(
        command, '--lambda_validation_split_seed', cfg['validation_split_seed'],
    )
    replace_arg(command, '--data_flow_audit', 0)
    replace_arg(command, '--exp_name', 'protocol_smoke')
    return command


def latest_run(results_dir):
    runs = sorted(Path(results_dir).glob('protocol_smoke_*'))
    if not runs:
        raise RuntimeError(f'no protocol smoke run in {results_dir}')
    return runs[-1]


def run(dataset, device, output_root, epochs=3):
    cfg = DATASETS[dataset]
    results_dir = Path(output_root) / dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    command = validation_command(dataset, device, output_root, epochs=epochs)
    env = os.environ.copy()
    env.update({
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'PYTHONHASHSEED': '42',
    })
    completed = subprocess.run(command, check=False, env=env)
    if completed.returncode:
        return completed.returncode
    run_dir = latest_run(results_dir)
    manifest_path = run_dir / 'validation' / 'validation_manifest.json'
    result_path = run_dir / 'results.json'
    if not manifest_path.is_file() or not result_path.is_file():
        raise RuntimeError('training-validation smoke is missing audited artifacts')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    result = json.loads(result_path.read_text(encoding='utf-8'))
    metrics = result['cl_metrics']
    summary = {
        'dataset': dataset,
        'status': 'complete',
        'selection_source': 'training-validation',
        'selection_test_used': False,
        'test_used_for_fit': False,
        'optimizer': cfg['optimizer'],
        'lr': cfg['lr'],
        'bottom_lr_scale': cfg['bottom_lr_scale'],
        'weight_decay': cfg['weight_decay'],
        'task_ce_mode': cfg['task_ce_mode'],
        'epochs_per_task': epochs,
        'tasks': cfg['tasks'][:2],
        'validation_per_class': cfg['validation_per_class'],
        'validation_split_seed': cfg['validation_split_seed'],
        'validation_manifest': str(manifest_path),
        'validation_manifest_dataset': manifest.get('dataset'),
        'AA_final': metrics['AA_final'],
        'BWT': metrics['BWT'],
        'AA_final_taskil': metrics.get('AA_final_taskil'),
        'run_dir': str(run_dir),
        'command': command,
    }
    output = Path(output_root) / f'{dataset}_PROTOCOL_SMOKE.json'
    output.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    parser.add_argument('--device', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()
    return run(args.dataset, args.device, args.output_root, epochs=args.epochs)


if __name__ == '__main__':
    raise SystemExit(main())
