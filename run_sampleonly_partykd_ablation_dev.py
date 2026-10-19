"""Training-only PartyKD variants from four approved reference20 jobs."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


APPROVED_HASHES = {
    ('isolet', 54): '5775c71254a8f0f0eb41b420670fcbec5201acd071d059c0058355964f740cfe',
    ('isolet', 55): '0282d27f205b258ba2ed53a51dc0b16702799544dcda34d4547433da4ac8fd58',
    ('upmc_food101', 54): '1eeb07c79238cdcfa16a437cee14f1299254e238e2323a2655692ccd4dadb202',
    ('upmc_food101', 55): '37562007542b87d45f075eeedbf3482462710afd8870a688e9ef89212c6a09a6',
}


def derive_config(source, source_hash, root, smoke=False, mode='base'):
    dataset, seed = source.get('pilot_dataset'), source.get('seed')
    if source_hash != APPROVED_HASHES.get((dataset, seed)):
        raise ValueError('source config hash is not approved')
    required = {
        'num_parties': 4, 'sample_replay_only': 1,
        'resource_lite': 0, 'head_consolidation_samples_per_class': 20,
        'head_consolidation_mode': 'adaptive_dual_branch',
        'party_kd_enabled': 1, 'dep_tracking_enabled': 1,
        'party_kd_mode': 'uniform',
        'expected_party_kd_variant': 'uniform',
    }
    if any(source.get(key) != value for key, value in required.items()):
        raise ValueError('reference20 source protocol differs')
    if mode not in ('base', 'task_marginal'):
        raise ValueError('unknown PartyKD development mode')
    variant = 'no_partykd20' if mode == 'base' else 'task_marginal20'
    name = f'{dataset}_seed{seed}_{variant}' + ('_smoke' if smoke else '')
    result = {
        **source,
        'party_kd_enabled': 0 if mode == 'base' else 1,
        'dep_tracking_enabled': 0 if mode == 'base' else 1,
        'party_kd_mode': 'uniform' if mode == 'base' else 'task_marginal',
        'expected_party_kd_variant': mode,
        'pilot_variant': variant,
        'results_dir': str(root),
        'output_dir': str(Path(root) / name),
        'exp_name': name,
    }
    if smoke:
        tasks = source['custom_tasks'].split('|')
        result.update(
            num_tasks=2, custom_tasks='|'.join(tasks[:2]),
            epochs_per_task=1, device='cpu', pilot_smoke=True,
        )
    return result


def run_ablation(source_path, root, smoke=False, mode='base'):
    import runner
    from config import (validate_adaptive_head_consolidation,
                        validate_party_kd_variant,
                        validate_sample_replay_only)
    from run_resource_lite_pilot import (
        PilotDataset, StopBeforeTest, evaluate_holdout, sha256,
        validate_source_data_hash, validate_training_only_artifacts,
    )

    source_path = Path(source_path)
    source = json.loads(source_path.read_text(encoding='utf-8'))
    config = derive_config(source, sha256(source_path), root, smoke, mode)
    validate_party_kd_variant(config, mode)
    args = SimpleNamespace(**config)
    validate_sample_replay_only(args)
    validate_adaptive_head_consolidation(args)
    validate_source_data_hash(config)
    if not smoke and args.epochs_per_task != (
            20 if args.pilot_dataset == 'upmc_food101' else 50):
        raise ValueError('full task training budget differs')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    run = Path(args.output_dir)
    config_path = root / (run.name + '.config.json')
    if run.exists() or config_path.exists():
        raise FileExistsError('ablation job already exists')
    with config_path.open('x', encoding='utf-8') as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
        stream.write('\n')
    run.mkdir(exist_ok=False)
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
        raise RuntimeError('ablation reached formal test access')
    validate_training_only_artifacts(run)
    final = run / 'checkpoints' / 'formal_final.pt'
    record = {
        'status': 'two_task_training_only_smoke' if smoke
        else 'independent_training_holdout_readout',
        'variant': args.pilot_variant, 'dataset': args.pilot_dataset,
        'seed': args.seed, 'source_config_sha256': sha256(source_path),
        'config_sha256': sha256(run / 'config.json'),
        'checkpoint_sha256': sha256(final),
        'freeze_sha256': sha256(run / 'FORMAL_STATE_FROZEN.json'),
        'holdout_manifest_sha256': sha256(
            run / 'pilot_holdout' / 'manifest.json'),
    }
    if not smoke:
        record.update(evaluate_holdout(args, dataset, final))
    target = run / ('SMOKE_GATE.json' if smoke else 'HOLDOUT_READOUT.json')
    with target.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({key: record[key] for key in (
        'status', 'variant', 'dataset', 'seed')}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--mode', choices=('base', 'task_marginal'),
                        default='base')
    args = parser.parse_args()
    run_ablation(args.source_config, args.root, args.smoke, args.mode)


if __name__ == '__main__':
    main()
