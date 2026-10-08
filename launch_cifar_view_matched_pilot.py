"""Train seed-48 Adaptive through its frozen checkpoint, before test access."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace


SOURCE_COMMIT = '7bfe6b1d724fb1206bc0053a9008126bad86332d'
DESIGN_COMMIT = 'f6c6708f11b29c32717ed6da83e85154079a64ed'
OVERRIDE_KEYS = frozenset({
    'seed', 'formal_deferred_evaluation', 'results_dir', 'output_dir', 'exp_name',
})


class TrainingOnlyStop(Exception):
    """The runner reached deferred evaluation after freezing the final state."""


def stop_before_deferred_evaluation(*_args, **_kwargs):
    raise TrainingOnlyStop


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def derive_config(source, root):
    expected = {
        'data': 'cifar100', 'seed': 42, 'num_tasks': 10,
        'classes_per_task': 10, 'custom_tasks': '',
        'lambda_validation_enabled': 1,
        'lambda_validation_per_class': 25,
        'lambda_validation_split_seed': 20260729,
        'formal_deferred_evaluation': True,
        'head_consolidation_enabled': 1,
        'head_consolidation_mode': 'adaptive_dual_branch',
        'head_consolidation_samples_per_class': 20,
        'bic_enabled': 1, 'bic_per_class': 25,
        'deterministic': 1, 'data_flow_audit': 1,
    }
    if any(source.get(key) != value for key, value in expected.items()):
        raise ValueError('source config is not the audited CIFAR Adaptive run')
    if any(int(task) < 10 for task in source['unlearn_after_tasks']):
        raise ValueError('pilot requires a ten-task CL-only timeline')
    root = Path(root).resolve()
    config = copy.deepcopy(source)
    config.update(
        seed=48,
        formal_deferred_evaluation=False,
        results_dir=str(root),
        output_dir=str(root / 'seed_48_training_only'),
        exp_name='cifar_view_matched_seed48_training_only',
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
    parser.add_argument('--check', action='store_true')
    cli = parser.parse_args()
    source_config = cli.source_config.resolve(strict=True)
    source_record = cli.source_record.resolve(strict=True)
    root = cli.root.resolve()
    if root.exists() or root == source_config.parent or root in source_config.parents:
        raise ValueError('pilot output root must be new and separate')
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
            != SOURCE_COMMIT
            or subprocess.check_output(['git', 'status', '--porcelain'], text=True)):
        raise ValueError('pilot requires exact clean formal producer checkout')
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '1'
            or os.environ.get('PYTHONHASHSEED') != '48'
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'):
        raise ValueError('deterministic physical-GPU-1 environment is incomplete')

    source = json.loads(source_config.read_text(encoding='utf-8'))
    record = json.loads(source_record.read_text(encoding='utf-8'))
    if (record.get('kind') != 'formal_completed_run'
            or record.get('source_commit') != SOURCE_COMMIT
            or record.get('method') != 'adaptive'
            or record.get('seed') != 42
            or file_sha256(source_config) != record['artifact_sha256']['config']):
        raise ValueError('formal source record/config identity mismatch')
    config, changed = derive_config(source, root)
    source_data = {
        key: file_sha256(Path(config['data_path']) / key.split(':', 1)[1])
        for key in record['artifact_sha256'] if key.startswith('data:')
    }
    if any(digest != record['artifact_sha256'][key]
           for key, digest in source_data.items()):
        raise ValueError('CIFAR payload differs from formal source')

    import torch
    from config import validate_adaptive_head_consolidation, validate_party_kd_variant
    from data_utils import TaskManager
    import runner

    args = SimpleNamespace(**config)
    validate_party_kd_variant(config, config['expected_party_kd_variant'])
    validate_adaptive_head_consolidation(args)
    if any(event['type'] != 'CIL' for event in TaskManager(args).get_timeline()):
        raise ValueError('pilot timeline contains unlearning')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('masked physical GPU 1 is unavailable')
    free, _ = torch.cuda.mem_get_info(0)
    if free < 4 * 1024 ** 3:
        raise ValueError('masked GPU has less than 4 GiB free')
    if cli.check:
        print(json.dumps({
            'status': 'ready', 'root': str(root),
            'source_commit': SOURCE_COMMIT, 'overrides': changed,
            'source_data_sha256': source_data,
        }, indent=2, sort_keys=True))
        return

    root.mkdir(parents=True, exist_ok=False)
    run = Path(config['output_dir'])
    run.mkdir()
    (run / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (run / 'PILOT_PROTOCOL.json').write_text(json.dumps({
        'schema_version': 1,
        'source_commit': SOURCE_COMMIT,
        'design_commit': DESIGN_COMMIT,
        'source_record_sha256': file_sha256(source_record),
        'source_config_sha256': file_sha256(source_config),
        'source_data_sha256': source_data,
        'launcher_sha256': file_sha256(__file__),
        'overrides': changed,
        'planned_stop': 'before_deferred_test_evaluation',
    }, indent=2, sort_keys=True), encoding='utf-8')
    print('PILOT_OUTPUT_DIR=' + str(run), flush=True)

    original_evaluator = runner.evaluate_deferred_cil_trajectory
    runner.evaluate_deferred_cil_trajectory = stop_before_deferred_evaluation
    stopped = False
    try:
        runner.run_experiment(args)
    except TrainingOnlyStop:
        stopped = True
    finally:
        runner.evaluate_deferred_cil_trajectory = original_evaluator
    if not stopped:
        raise RuntimeError('pilot did not stop at deferred test evaluation')
    checkpoint = run / 'adaptive_final.pt'
    freeze = run / 'ADAPTIVE_STATE_FROZEN.json'
    events = [run / 'checkpoints' / f'event_{task}_CIL.pt'
              for task in range(10)]
    bic_manifest = run / 'bic' / 'calibration_manifest.json'
    validation_manifest = run / 'validation' / 'validation_manifest.json'
    audit = run / 'data_flow_audit.jsonl'
    if (not checkpoint.is_file() or not freeze.is_file()
            or not all(path.is_file() for path in events)
            or not bic_manifest.is_file() or not validation_manifest.is_file()
            or not audit.is_file() or (run / 'results.json').exists()):
        raise RuntimeError('training-only checkpoint boundary is incomplete')
    accesses = [json.loads(line) for line in audit.read_text(encoding='utf-8').splitlines()]
    if any(entry.get('split') == 'test'
           or str(entry.get('loader_key', '')).startswith("('test'")
           for entry in accesses):
        raise RuntimeError('pilot accessed the test loader')
    complete = {
        'schema_version': 1,
        'status': 'training_only_before_deferred_test_evaluation',
        'source_commit': SOURCE_COMMIT,
        'config_sha256': file_sha256(run / 'config.json'),
        'checkpoint_sha256': file_sha256(checkpoint),
        'adaptive_freeze_sha256': file_sha256(freeze),
        'event_checkpoint_sha256': [file_sha256(path) for path in events],
        'bic_manifest_sha256': file_sha256(bic_manifest),
        'validation_manifest_sha256': file_sha256(validation_manifest),
        'data_flow_audit_sha256': file_sha256(audit),
    }
    (run / 'PILOT_TRAINING_ONLY_COMPLETE.json').write_text(
        json.dumps(complete, indent=2, sort_keys=True), encoding='utf-8',
    )
    print('PILOT_TRAINING_ONLY_COMPLETE=' + str(run), flush=True)


if __name__ == '__main__':
    main()
