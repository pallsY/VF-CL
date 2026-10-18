"""Rescore a frozen iCaRL dev checkpoint with its actual NME inference."""

import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from run_resource_lite_pilot import (
    PilotDataset, sha256, validate_source_data_hash,
    validate_training_only_artifacts,
)
from run_selected_baseline_dev_pilot import score_selected_baseline_checkpoint


def recover(source_run, recovery_root, expected_training_commit):
    source_run, recovery_root = Path(source_run).resolve(), Path(recovery_root).resolve()
    if (expected_training_commit
            != 'e289f8eb3fdae7eda5eda9de14af09aa9fe84b28'
            or source_run != Path(
                '/brf/users/yangxx/VFCL-results/'
                'resource-lite-baseline-dev-v1-20261018/'
                'isolet_seed54_icarl').resolve()):
        raise ValueError('frozen iCaRL training identity differs')
    validate_training_only_artifacts(source_run)
    config_path = source_run / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if (config.get('cl_method') != 'icarl'
            or config.get('pilot_dataset') != 'isolet'
            or config.get('seed') != 54
            or Path(config['output_dir']).resolve() != source_run):
        raise ValueError('iCaRL recovery source identity differs')
    validate_source_data_hash(config)
    generic = source_run / 'HOLDOUT_READOUT.json'
    if not generic.is_file():
        raise ValueError('original generic readout is missing')
    prior = json.loads(generic.read_text(encoding='utf-8'))
    if prior.get('metrics', {}).get('readout_rule'):
        raise ValueError('source is already method-aware')
    recovery_root.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(**config)
    args.output_dir = str(recovery_root)
    dataset = PilotDataset(args)
    manifest = source_run / 'pilot_holdout' / 'manifest.json'
    if sha256(manifest) != sha256(
            recovery_root / 'pilot_holdout' / 'manifest.json'):
        raise ValueError('recovery holdout differs')
    args.output_dir = str(source_run)
    checkpoint = source_run / 'checkpoints' / 'formal_final.pt'
    metrics = score_selected_baseline_checkpoint(args, dataset, checkpoint)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
    if dirty:
        raise RuntimeError('readout checkout is dirty')
    result = {
        'status': 'recovered_method_aware_training_holdout_readout',
        'method': 'icarl', 'dataset': 'isolet', 'seed': 54,
        'training_commit': expected_training_commit,
        'readout_commit': commit,
        'source_run': str(source_run),
        'source_generic_readout_sha256': sha256(generic),
        'config_sha256': sha256(config_path),
        'checkpoint_sha256': sha256(checkpoint),
        'holdout_manifest_sha256': sha256(manifest),
        'metrics': metrics,
    }
    with (recovery_root / 'METHOD_AWARE_READOUT.json').open(
            'x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--recovery-root', type=Path, required=True)
    parser.add_argument('--expected-training-commit', required=True)
    args = parser.parse_args()
    recover(args.source_run, args.recovery_root,
            args.expected_training_commit)
