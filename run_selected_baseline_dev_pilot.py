"""Training-only holdout readout for the three selected replay baselines."""

import argparse
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from run_resource_lite_pilot import (
    DATASET_IDENTITY, HOLDOUT_PER_CLASS, HOLDOUT_SEED,
    PilotDataset, StopBeforeTest, _identity, score_holdout_checkpoint,
    sha256, validate_source_data_hash, validate_training_only_artifacts,
)


SELECTED = {
    'isolet': ('icarl', 40, 20260809, 50),
    'upmc_food101': ('der_pp', 64, 20260809, 20),
    'cifar100': ('er_ace', 25, 20260729, 50),
}
PROTOCOL = 'selected-baseline-train-holdout-v1'


def _validate_source(source, dataset):
    _identity(source, dataset)
    method, per_class, split_seed, epochs = SELECTED[dataset]
    if source.get('cl_method') != method:
        raise ValueError('selected baseline method differs')
    required = {
        'seed': 42,
        'formal_deferred_evaluation': True,
        'head_consolidation_enabled': 0,
        'lambda_validation_enabled': 1,
        'lambda_validation_per_class': per_class,
        'lambda_validation_split_seed': split_seed,
        'epochs_per_task': epochs,
    }
    if any(source.get(key) != value for key, value in required.items()):
        raise ValueError('selected baseline source config drifted')
    if bool(source.get('bic_enabled', 0)) != (dataset == 'cifar100'):
        raise ValueError('selected baseline calibration drifted')


def derive_config(source, root, dataset, seed):
    if dataset not in DATASET_IDENTITY:
        raise ValueError('unknown selected baseline dataset')
    _validate_source(source, dataset)
    if seed not in (54, 55):
        raise ValueError('development seed must be 54 or 55')
    root = (PurePosixPath(root) if str(root).startswith('/')
            else Path(root).resolve())
    method = SELECTED[dataset][0]
    name = f'{dataset}_seed{seed}_{method}'
    return {
        **source,
        'seed': seed,
        'pilot_protocol': PROTOCOL,
        'pilot_dataset': dataset,
        'pilot_variant': method,
        'pilot_holdout_seed': HOLDOUT_SEED,
        'pilot_holdout_per_class': HOLDOUT_PER_CLASS[dataset],
        'results_dir': str(root),
        'output_dir': str(root / name),
        'exp_name': f'selected_baseline_dev_{name}',
    }


def run_pilot(config_path):
    import runner
    from data_utils import TaskManager

    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    dataset_name = config['pilot_dataset']
    _validate_source({**config, 'seed': 42}, dataset_name)
    method = SELECTED[dataset_name][0]
    if (config.get('pilot_protocol') != PROTOCOL
            or config.get('pilot_variant') != method
            or config.get('seed') not in (54, 55)
            or config.get('pilot_holdout_seed') != HOLDOUT_SEED
            or config.get('pilot_holdout_per_class')
            != HOLDOUT_PER_CLASS[dataset_name]):
        raise ValueError('selected baseline development config drifted')
    validate_source_data_hash(config)
    args = SimpleNamespace(**config)
    if any(event['type'] != 'CIL' for event in TaskManager(args).get_timeline()):
        raise ValueError('selected baseline requires a CL-only task stream')
    run = Path(args.output_dir)
    run.mkdir(parents=True, exist_ok=False)
    (run / 'config.json').write_bytes(config_path.read_bytes())
    dataset = PilotDataset(args)
    original = runner.VFLDataset
    runner.VFLDataset = lambda _args: dataset
    stopped = False
    try:
        runner.run_experiment(args)
    except StopBeforeTest:
        stopped = True
    finally:
        runner.VFLDataset = original
    if not stopped:
        raise RuntimeError('baseline pilot did not stop before formal test')
    validate_training_only_artifacts(run)
    final_path = run / 'checkpoints' / 'formal_final.pt'
    result = {
        'status': 'independent_training_holdout_readout',
        'method': method,
        'seed': args.seed,
        'dataset': dataset_name,
        'config_sha256': sha256(run / 'config.json'),
        'checkpoint_sha256': sha256(final_path),
        'freeze_sha256': sha256(run / 'FORMAL_STATE_FROZEN.json'),
        'holdout_manifest_sha256': sha256(
            run / 'pilot_holdout' / 'manifest.json'),
        'metrics': score_holdout_checkpoint(args, dataset, final_path),
    }
    (run / 'HOLDOUT_READOUT.json').write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    print(json.dumps(result, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--source-config', type=Path, required=True)
    prepare.add_argument('--root', required=True)
    prepare.add_argument('--dataset', choices=SELECTED, required=True)
    prepare.add_argument('--seed', type=int, required=True)
    run = sub.add_parser('run')
    run.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        run_pilot(args.config)
        return
    source = json.loads(args.source_config.read_text(encoding='utf-8'))
    config = derive_config(source, args.root, args.dataset, args.seed)
    method = SELECTED[args.dataset][0]
    target = Path(args.root) / (
        f'{args.dataset}_seed{args.seed}_{method}.config.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(config, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print(target)


if __name__ == '__main__':
    main()
