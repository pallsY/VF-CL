"""Frozen-method A/B development runs; never consume the official test."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from run_resource_lite_pilot import (
    HOLDOUT_PER_CLASS, PilotDataset, StopBeforeTest, _identity,
    evaluate_holdout, sha256, validate_source_data_hash,
    validate_training_only_artifacts,
)


SOURCE_SHA256 = {
    'isolet': '9a524718f29e3dbae0a0157a875ce2b4eccb27aa51db7f757dd00858240bc8db',
    'upmc_food101': '4e8c76695ad2c8cd4d2ddd895e134b386597281a756d80c2c3555ffb5a9042e1',
}
PROTOCOL = 'sampleonly-train-replay-ab-v1'
HOLDOUT_SEED = 20261021


def derive_config(source_path, root, dataset, seed, arm, *, pilot=False):
    source_path, root = Path(source_path), Path(root)
    if dataset not in SOURCE_SHA256 or sha256(source_path) != SOURCE_SHA256[dataset]:
        raise ValueError('source formal config hash differs')
    source = json.loads(source_path.read_text(encoding='utf-8'))
    _identity(source, dataset)
    if (source.get('seed') != 42 or source.get('sample_replay_only') != 1
            or source.get('head_consolidation_samples_per_class') != 20
            or source.get('formal_deferred_evaluation') is not True):
        raise ValueError('source is not frozen sample-only reference20')
    if seed not in ((63,) if pilot else (60, 61, 62)):
        raise ValueError('seed does not belong to this stage')
    if arm not in ('A', 'B'):
        raise ValueError('A/B arm differs')
    name = f'{dataset}_seed{seed}_{arm}' + ('_pilot' if pilot else '')
    config = {
        **source,
        'seed': seed, 'pilot_dataset': dataset,
        'pilot_variant': 'reference20', 'pilot_protocol': PROTOCOL,
        'pilot_holdout_seed': HOLDOUT_SEED,
        'pilot_holdout_per_class': HOLDOUT_PER_CLASS[dataset],
        'pilot_smoke': pilot,
        'train_replay_enabled': 1 if arm == 'B' else 0,
        'train_replay_arm': arm,
        'source_config_sha256': SOURCE_SHA256[dataset],
        'results_dir': str(root), 'exp_name': name,
        'output_dir': str(root / name),
    }
    if pilot:
        config.update(device='cpu', epochs_per_task=1)
    return config


def run_config(config_path):
    import runner
    from config import validate_adaptive_head_consolidation, validate_sample_replay_only
    from data_utils import TaskManager

    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    args = SimpleNamespace(**config)
    dataset_name, seed, arm = args.pilot_dataset, args.seed, args.train_replay_arm
    if (args.pilot_protocol != PROTOCOL or dataset_name not in SOURCE_SHA256
            or arm not in ('A', 'B')
            or args.train_replay_enabled != (arm == 'B')
            or args.pilot_holdout_seed != HOLDOUT_SEED
            or args.pilot_holdout_per_class != HOLDOUT_PER_CLASS[dataset_name]
            or args.source_config_sha256 != SOURCE_SHA256[dataset_name]
            or args.pilot_variant != 'reference20'
            or args.formal_deferred_evaluation is not True):
        raise ValueError('A/B development config differs')
    _identity(config, dataset_name)
    if args.pilot_smoke:
        if seed != 63 or args.epochs_per_task != 1 or args.device != 'cpu':
            raise ValueError('pilot budget or identity differs')
    elif (seed not in (60, 61, 62)
          or args.epochs_per_task != (20 if dataset_name == 'upmc_food101' else 50)):
        raise ValueError('full development budget or identity differs')
    validate_sample_replay_only(args)
    validate_adaptive_head_consolidation(args)
    validate_source_data_hash(config)
    if any(event['type'] != 'CIL' for event in TaskManager(args).get_timeline()):
        raise ValueError('development stream differs')
    run = Path(args.output_dir)
    if run != Path(args.results_dir) / args.exp_name:
        raise ValueError('output path differs')
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
        raise RuntimeError('development run did not stop before official test')
    validate_training_only_artifacts(run)
    checkpoint = run / 'checkpoints' / 'formal_final.pt'
    result = {
        'status': 'training_only_replay_ab_readout',
        'dataset': dataset_name, 'seed': seed, 'arm': arm,
        'pilot': bool(args.pilot_smoke),
        'config_sha256': sha256(run / 'config.json'),
        'checkpoint_sha256': sha256(checkpoint),
        'freeze_sha256': sha256(run / 'FORMAL_STATE_FROZEN.json'),
        'holdout_manifest_sha256': sha256(run / 'pilot_holdout' / 'manifest.json'),
    }
    audit_dir = run / 'train_replay_audit'
    if arm == 'B':
        events = [json.loads((audit_dir / f'event_{task_id}.json').read_text())
                  for task_id in range(args.num_tasks)]
        if (events[0]['steps'] != 0
                or any(event['steps'] < 1 for event in events[1:])
                or any(event['samples'] != 64 * event['steps'] for event in events)):
            raise ValueError('replay audit counts differ from the fixed protocol')
        result['train_replay_audit'] = {
            'events': events,
            'totals': {key: sum(event[key] for event in events)
                       for key in ('steps', 'samples', 'duplicates',
                                   'upload_bytes', 'return_bytes')},
        }
    elif audit_dir.exists():
        raise ValueError('A arm unexpectedly wrote replay audit files')
    if not args.pilot_smoke:
        result.update(evaluate_holdout(args, dataset, checkpoint))
    path = run / ('PILOT_GATE.json' if args.pilot_smoke else 'HOLDOUT_READOUT.json')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--source-config', type=Path, required=True)
    prepare.add_argument('--root', type=Path, required=True)
    prepare.add_argument('--dataset', choices=SOURCE_SHA256, required=True)
    prepare.add_argument('--seed', type=int, required=True)
    prepare.add_argument('--arm', choices=('A', 'B'), required=True)
    prepare.add_argument('--pilot', action='store_true')
    run = sub.add_parser('run')
    run.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        config = derive_config(args.source_config, args.root, args.dataset,
                               args.seed, args.arm, pilot=args.pilot)
        path = args.root / (config['exp_name'] + '.config.json')
        args.root.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(config, stream, indent=2, sort_keys=True)
            stream.write('\n')
        print(path)
    else:
        run_config(args.config)


if __name__ == '__main__':
    main()
