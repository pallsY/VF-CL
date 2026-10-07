"""Launch one preregistered seed-45 CIFAR development run from audited config."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace


SOURCE_COMMIT = '7bfe6b1d724fb1206bc0053a9008126bad86332d'
DESIGN_COMMIT = 'f8487fa00519b35f1a1b2fbf543188d4d7a34208'
HEAD_BUDGET_DESIGN_COMMIT = '645bb89f97feac5386c09ffec61bde7c10814eb8'
OVERRIDE_KEYS = frozenset({
    'seed', 'lambda_validation_split_seed', 'formal_deferred_evaluation',
    'head_consolidation_enabled', 'head_consolidation_mode',
    'results_dir', 'output_dir', 'exp_name',
})


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def derive_config(source, root, seed=45):
    """Return the only allowed deviations from the audited seed-42 config."""
    expected = {
        'data': 'cifar100', 'seed': 42, 'num_tasks': 10,
        'classes_per_task': 10, 'custom_tasks': '',
        'lambda_validation_enabled': 1,
        'lambda_validation_per_class': 25,
        'lambda_validation_split_seed': 20260729,
        'formal_deferred_evaluation': True,
        'head_consolidation_enabled': 1,
        'head_consolidation_mode': 'adaptive_dual_branch',
        'dep_tracking_enabled': 1, 'party_kd_enabled': 1,
        'party_kd_mode': 'uniform', 'save_task_checkpoints': 3,
        'deterministic': 1, 'data_flow_audit': 1,
        'num_workers': 2, 'bic_enabled': 1,
    }
    if any(source.get(key) != value for key, value in expected.items()):
        raise ValueError('source config is not the audited CIFAR Adaptive run')
    if any(int(task) < 10 for task in source['unlearn_after_tasks']):
        raise ValueError('pilot requires a CL-only timeline')
    if seed not in (45, 46):
        raise ValueError('pilot seed is not registered')
    root = Path(root).resolve()
    config = copy.deepcopy(source)
    config.update(
        seed=seed,
        lambda_validation_split_seed=(20261007 if seed == 45 else 20261008),
        formal_deferred_evaluation=False,
        head_consolidation_enabled=0,
        head_consolidation_mode='full_classifier',
        results_dir=str(root),
        output_dir=str(root / f'seed_{seed}_baseline'),
        exp_name=('cifar_head_offset_seed45_baseline' if seed == 45
                  else 'cifar_head_budget_seed46_baseline'),
    )
    changed = {
        key: {'source': source.get(key), 'pilot': config.get(key)}
        for key in set(source) | set(config)
        if source.get(key) != config.get(key)
    }
    if set(changed) != OVERRIDE_KEYS:
        raise ValueError('pilot configuration changed an unregistered option')
    return config, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-config', required=True, type=Path)
    parser.add_argument('--source-record', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--seed', type=int, choices=(45, 46), default=45)
    parser.add_argument('--check', action='store_true')
    options = parser.parse_args()
    source_config = options.source_config.resolve(strict=True)
    source_record = options.source_record.resolve(strict=True)
    root = options.root.resolve()
    if root.exists() or source_config.parent == root or root in source_config.parents:
        raise ValueError('pilot output root must be new and separate')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], text=True)
    if commit != SOURCE_COMMIT or dirty:
        raise ValueError('pilot requires the exact clean formal producer checkout')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('pilot is pinned to physical GPU 1')
    if (os.environ.get('PYTHONHASHSEED') != str(options.seed)
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'):
        raise ValueError('deterministic launcher environment is incomplete')

    source = json.loads(source_config.read_text(encoding='utf-8'))
    record = json.loads(source_record.read_text(encoding='utf-8'))
    if (record.get('kind') != 'formal_completed_run'
            or record.get('source_commit') != SOURCE_COMMIT
            or record.get('method') != 'adaptive'
            or record.get('seed') != 42
            or file_sha256(source_config) != record['artifact_sha256']['config']):
        raise ValueError('source record/config identity is invalid')
    config, changed = derive_config(source, root, options.seed)
    source_data = {
        key: file_sha256(Path(config['data_path']) / key.split(':', 1)[1])
        for key in record['artifact_sha256'] if key.startswith('data:')
    }
    if any(digest != record['artifact_sha256'][key]
           for key, digest in source_data.items()):
        raise ValueError('CIFAR data differ from formal source')

    import torch
    from config import validate_adaptive_head_consolidation, validate_party_kd_variant
    from data_utils import TaskManager
    from runner import run_experiment

    args = SimpleNamespace(**config)
    validate_party_kd_variant(config, config['expected_party_kd_variant'])
    validate_adaptive_head_consolidation(args)
    if any(event['type'] != 'CIL' for event in TaskManager(args).get_timeline()):
        raise ValueError('pilot timeline contains unlearning')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('masked pilot GPU is unavailable')
    free, _ = torch.cuda.mem_get_info(0)
    if free < 4 * 1024 ** 3:
        raise ValueError('pilot GPU has less than 4 GiB free')
    if options.check:
        print(json.dumps({
            'status': 'ready', 'root': str(root),
            'source_commit': SOURCE_COMMIT,
            'overrides': changed,
            'source_data_sha256': source_data,
        }, indent=2, sort_keys=True))
        return

    root.mkdir(parents=True, exist_ok=False)
    run = Path(config['output_dir'])
    run.mkdir()
    (run / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    protocol = {
        'schema_version': 1,
        'source_commit': SOURCE_COMMIT,
        'design_commit': (DESIGN_COMMIT if options.seed == 45
                          else HEAD_BUDGET_DESIGN_COMMIT),
        'source_record_sha256': file_sha256(source_record),
        'source_config_sha256': file_sha256(source_config),
        'source_data_sha256': source_data,
        'launcher_sha256': file_sha256(__file__),
        'overrides': changed,
    }
    (run / 'PILOT_PROTOCOL.json').write_text(
        json.dumps(protocol, indent=2, sort_keys=True), encoding='utf-8',
    )
    print('PILOT_OUTPUT_DIR=' + str(run), flush=True)
    run_experiment(args)
    checkpoint = run / 'checkpoints' / 'event_9_CIL.pt'
    results = run / 'results.json'
    if not checkpoint.is_file() or not results.is_file():
        raise RuntimeError('pilot did not write its final checkpoint and results')
    audit = run / 'data_flow_audit.jsonl'
    if not audit.is_file():
        raise RuntimeError('pilot data-flow audit is missing')
    for line in audit.read_text(encoding='utf-8').splitlines():
        entry = json.loads(line)
        if entry.get('split') == 'test' or str(entry.get('loader_key', '')).startswith("('test'"):
            raise RuntimeError('pilot accessed the final-test loader')
    complete = {
        'schema_version': 1,
        'source_commit': SOURCE_COMMIT,
        'config_sha256': file_sha256(run / 'config.json'),
        'checkpoint_sha256': file_sha256(checkpoint),
        'results_sha256': file_sha256(results),
        'data_flow_audit_sha256': file_sha256(audit),
    }
    (run / 'PILOT_TRAINING_COMPLETE.json').write_text(
        json.dumps(complete, indent=2, sort_keys=True), encoding='utf-8',
    )
    print('PILOT_TRAINING_COMPLETE=' + str(run), flush=True)


if __name__ == '__main__':
    main()
