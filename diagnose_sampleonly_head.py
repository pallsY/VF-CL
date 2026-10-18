"""Compare frozen sample-only development checkpoints on a training holdout."""

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace


def validate_run_inputs(run):
    """Reject changed or test-exposed development runs before loading models."""
    from run_resource_lite_pilot import sha256

    run = Path(run)
    readout = json.loads((run / 'HOLDOUT_READOUT.json').read_text())
    config_path = run / 'config.json'
    config = json.loads(config_path.read_text())
    if (config.get('pilot_dataset') not in ('isolet', 'upmc_food101')
            or config.get('pilot_variant') != 'reference20'
            or config.get('seed') not in (54, 55)
            or config.get('num_tasks') not in (10, 13)
            or readout.get('status')
            != 'independent_training_holdout_readout'
            or (readout.get('dataset'), readout.get('seed'),
                readout.get('variant'))
            != (config['pilot_dataset'], config['seed'], 'reference20')):
        raise ValueError('run is not an approved reference20 development run')
    if not (run / 'FORMAL_EVALUATION_PENDING.json').is_file() or any(
            (run / name).exists() for name in (
                'results.json', 'FORMAL_EVALUATION_CONSUMING.json',
                'FORMAL_EVALUATION_PUBLISHED.json',
                'FORMAL_EVALUATION_COMPLETE.json',
                'formal_access/test.consumed.json')):
        raise ValueError('development run reached test access')
    manifest = run / 'pilot_holdout' / 'manifest.json'
    pre = run / 'checkpoints' / f"event_{config['num_tasks'] - 1}_CIL.pt"
    final = run / 'checkpoints' / 'formal_final.pt'
    for path, expected, label in (
            (config_path, readout.get('config_sha256'), 'config'),
            (manifest, readout.get('holdout_manifest_sha256'),
             'holdout manifest'),
            (final, readout.get('checkpoint_sha256'), 'final checkpoint')):
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f'{label} hash differs from frozen readout')
    if not pre.is_file():
        raise ValueError('last event checkpoint is missing')
    return config, readout, pre, final


def diagnose_run(run):
    from run_resource_lite_pilot import (
        PilotDataset, score_holdout_checkpoint, sha256,
        validate_source_data_hash, validate_training_only_artifacts,
    )

    run = Path(run)
    config, readout, pre, final = validate_run_inputs(run)
    validate_training_only_artifacts(run)
    validate_source_data_hash(config)
    with tempfile.TemporaryDirectory(prefix='vfcl-sampleonly-diagnostic-') as tmp:
        args = SimpleNamespace(**config)
        args.output_dir = tmp
        args.device = 'cpu'
        dataset = PilotDataset(args)
        if sha256(Path(tmp) / 'pilot_holdout' / 'manifest.json') \
                != readout['holdout_manifest_sha256']:
            raise ValueError('reconstructed holdout manifest differs')
        before = score_holdout_checkpoint(args, dataset, pre)
        after = score_holdout_checkpoint(args, dataset, final)
    if (before['count'] != after['count']
            or abs(after['cil'] - readout['metrics']['cil']) > 1e-12):
        raise ValueError('post-head readout differs from frozen development run')
    return {
        'status': 'training_holdout_diagnostic',
        'dataset': config['pilot_dataset'], 'seed': config['seed'],
        'source': str(run),
        'config_sha256': readout['config_sha256'],
        'holdout_manifest_sha256': readout['holdout_manifest_sha256'],
        'pre_checkpoint_sha256': sha256(pre),
        'final_checkpoint_sha256': sha256(final),
        'pre_head': before, 'post_head': after,
        'head_delta_cil': after['cil'] - before['cil'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = diagnose_run(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({
        'dataset': result['dataset'], 'seed': result['seed'],
        'pre_head_cil': result['pre_head']['cil'],
        'post_head_cil': result['post_head']['cil'],
        'head_delta_cil': result['head_delta_cil'],
    }, sort_keys=True))


if __name__ == '__main__':
    main()
