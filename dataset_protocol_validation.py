"""Full-stream, training-only validation for vector dataset protocols."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from dataset_protocol_smoke import replace_arg
from fair_main_table_3datasets import DATASETS, base_command


def validation_command(dataset, device, output_root):
    cfg = DATASETS[dataset]
    results_dir = Path(output_root) / dataset
    command = base_command(
        f'{dataset}:finetune:42', device, results_dir, smoke=False,
    )
    replace_arg(command, '--lambda_validation_enabled', 1)
    replace_arg(
        command, '--lambda_validation_per_class', cfg['validation_per_class'],
    )
    replace_arg(
        command, '--lambda_validation_split_seed', cfg['validation_split_seed'],
    )
    replace_arg(command, '--exp_name', 'protocol_validation')
    return command


def deterministic_env():
    env = os.environ.copy()
    env.update({
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'PYTHONHASHSEED': '42',
    })
    return env


def latest_run(results_dir):
    runs = sorted(Path(results_dir).glob('protocol_validation_*'))
    if not runs:
        raise RuntimeError(f'no protocol validation run in {results_dir}')
    return runs[-1]


def audit_result(dataset, command, run_dir):
    cfg = DATASETS[dataset]
    manifest_path = run_dir / 'validation' / 'validation_manifest.json'
    result_path = run_dir / 'results.json'
    if not manifest_path.is_file() or not result_path.is_file():
        raise RuntimeError('protocol validation is missing audited artifacts')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    result = json.loads(result_path.read_text(encoding='utf-8'))
    metrics = result['cl_metrics']
    selection = result.get('selection_audit', {})
    taskil = metrics.get('AA_final_taskil')
    trajectory = metrics.get('AA_trajectory_taskil', [])
    checks = {
        'selection_audit_passed': selection.get('passed') is True,
        'test_not_used_for_selection': (
            selection.get('test_used_for_selection') is False
        ),
        'evaluation_is_training_validation': (
            selection.get('evaluation_source') == 'vector-train-validation'
        ),
        'all_tasks_completed': len(trajectory) == len(cfg['tasks']),
        'taskil_gate_passed': (
            taskil is not None and float(taskil) >= cfg['taskil_gate']
        ),
    }
    return {
        'dataset': dataset,
        'status': 'passed' if all(checks.values()) else 'failed',
        'selection_source': 'training-validation',
        'selection_test_used': False,
        'test_used_for_fit': False,
        'seed': 42,
        'optimizer': cfg['optimizer'],
        'lr': cfg['lr'],
        'bottom_lr_scale': cfg['bottom_lr_scale'],
        'weight_decay': cfg['weight_decay'],
        'task_ce_mode': cfg['task_ce_mode'],
        'epochs_per_task': cfg['epochs_per_task'],
        'completed_tasks': len(trajectory),
        'expected_tasks': len(cfg['tasks']),
        'validation_per_class': cfg['validation_per_class'],
        'validation_split_seed': cfg['validation_split_seed'],
        'validation_manifest': str(manifest_path),
        'validation_manifest_sha256': manifest.get('sha256'),
        'taskil_gate': cfg['taskil_gate'],
        'AA_final': metrics['AA_final'],
        'AA_cil': metrics['AA_cil'],
        'BWT': metrics['BWT'],
        'AA_final_taskil': taskil,
        'checks': checks,
        'selection_audit': selection,
        'run_dir': str(run_dir),
        'command': command,
    }


def run(dataset, device, output_root):
    results_dir = Path(output_root) / dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    command = validation_command(dataset, device, output_root)
    completed = subprocess.run(command, check=False, env=deterministic_env())
    if completed.returncode:
        return completed.returncode
    run_dir = latest_run(results_dir)
    summary = audit_result(dataset, command, run_dir)
    output = Path(output_root) / f'{dataset}_PROTOCOL_VALIDATION.json'
    output.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    if summary['status'] != 'passed':
        return 2
    (Path(output_root) / f'{dataset}_PROTOCOL_VALIDATION_SUCCESS').touch()
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    parser.add_argument('--device', required=True)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    return run(args.dataset, args.device, args.output_root)


if __name__ == '__main__':
    raise SystemExit(main())
