"""Recover a training-only development readout without touching its run tree."""

import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from run_resource_lite_pilot import (
    PilotDataset, evaluate_holdout, sha256, validate_source_data_hash,
    validate_training_only_artifacts,
)


def recover(failed_root, recovery_root, expected_training_commit):
    failed_root = Path(failed_root).resolve()
    recovery_root = Path(recovery_root).resolve()
    queue_failure = failed_root / 'QUEUE_FAILURE.json'
    queue_lock = failed_root / 'QUEUE_LOCK.json'
    failure = json.loads(queue_failure.read_text(encoding='utf-8'))
    lock = json.loads(queue_lock.read_text(encoding='utf-8'))
    if (failure.get('producer_commit') != expected_training_commit
            or lock.get('producer_commit') != expected_training_commit
            or 'isolet/54/lite18 exited 1' not in failure.get('error', '')):
        raise ValueError('recovery source failure identity differs')
    run = failed_root / 'isolet_seed54_lite18'
    if (run / 'HOLDOUT_READOUT.json').exists():
        raise ValueError('source run already has a development readout')
    validate_training_only_artifacts(run)
    config_path = run / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if (config.get('pilot_dataset') != 'isolet'
            or config.get('pilot_variant') != 'lite18'
            or config.get('seed') != 54
            or Path(config['output_dir']).resolve() != run):
        raise ValueError('recovery run config differs')
    validate_source_data_hash(config)
    recovery_root.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(**config)
    args.output_dir = str(recovery_root)
    dataset = PilotDataset(args)
    manifest = run / 'pilot_holdout' / 'manifest.json'
    recovered_manifest = recovery_root / 'pilot_holdout' / 'manifest.json'
    if sha256(manifest) != sha256(recovered_manifest):
        raise ValueError('reconstructed holdout manifest differs')
    args.output_dir = str(run)
    checkpoint = run / 'checkpoints' / 'formal_final.pt'
    metrics = evaluate_holdout(args, dataset, checkpoint)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True,
    ).strip()
    dirty = subprocess.check_output(
        ['git', 'status', '--porcelain'], text=True,
    ).strip()
    if dirty:
        raise RuntimeError('readout code checkout is dirty')
    result = {
        'status': 'recovered_training_only_holdout_readout',
        'dataset': 'isolet', 'seed': 54, 'variant': 'lite18',
        'training_commit': expected_training_commit,
        'readout_commit': commit,
        'source_run': str(run),
        'source_failure_sha256': sha256(queue_failure),
        'config_sha256': sha256(config_path),
        'checkpoint_sha256': sha256(checkpoint),
        'freeze_sha256': sha256(run / 'FORMAL_STATE_FROZEN.json'),
        'holdout_manifest_sha256': sha256(manifest),
        **metrics,
    }
    with (recovery_root / 'HOLDOUT_READOUT_RECOVERED.json').open(
            'x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failed-root', type=Path, required=True)
    parser.add_argument('--recovery-root', type=Path, required=True)
    parser.add_argument('--expected-training-commit', required=True)
    args = parser.parse_args()
    recover(args.failed_root, args.recovery_root,
            args.expected_training_commit)
