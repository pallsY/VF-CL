"""Paired, training-only development for full-width versus old-class KD."""

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

import runner
from config import (validate_adaptive_head_consolidation,
                    validate_party_kd_variant, validate_sample_replay_only)
from data_utils import TaskManager
from run_resource_lite_pilot import (
    HOLDOUT_PER_CLASS, PilotDataset, StopBeforeTest, _identity,
    evaluate_holdout, expected_tasks, sha256, validate_source_data_hash,
    validate_training_only_artifacts,
)


SOURCE_SHA256 = {
    'isolet': '9a524718f29e3dbae0a0157a875ce2b4eccb27aa51db7f757dd00858240bc8db',
    'upmc_food101': '4e8c76695ad2c8cd4d2ddd895e134b386597281a756d80c2c3555ffb5a9042e1',
    'cifar100': '673ca34abe87179a75abff9d10d7e7a7392f786587df8c4d668340b57418a8c2',
}
PROTOCOL = 'old-class-kd-training-holdout-v1'
HOLDOUT_SEED = 20261019
VARIANTS = ('oldonly20', 'fullwidth20')


def git_commit():
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip()


def build_config(source, source_sha, dataset, seed, variant, root, *,
                 code_commit, smoke=False, device=None, source_path=''):
    if dataset not in SOURCE_SHA256 or source_sha != SOURCE_SHA256[dataset]:
        raise ValueError('source config hash differs from frozen sample-only run')
    _identity(source, dataset)
    if seed not in ((54,) if smoke else (56, 57)):
        raise ValueError('development seed is outside the preselected set')
    if variant not in VARIANTS:
        raise ValueError('unknown KD scope variant')
    if len(code_commit) != 40 or any(c not in '0123456789abcdef' for c in code_commit):
        raise ValueError('invalid code commit')
    required = {
        'seed': 42, 'cl_method': 'proto_evolve',
        'sample_replay_only': 1, 'resource_lite': 0,
        'formal_deferred_evaluation': True,
        'head_consolidation_enabled': 1,
        'head_consolidation_schedule': 'final',
        'head_consolidation_mode': 'adaptive_dual_branch',
        'head_consolidation_samples_per_class': 20,
        'head_consolidation_steps': 500,
        'proto_replay_gradient_enabled': 0, 'proto_sdc': False,
        'feat_distill_weight': 0.0, 'fim_freeze_frac': 0.0,
        'party_kd_enabled': 1, 'party_kd_mode': 'uniform',
        'dep_tracking_enabled': 1,
        'expected_party_kd_variant': 'uniform',
        'lambda_validation_enabled': 1, 'distill_weight': 0.25,
        'bic_enabled': int(dataset == 'cifar100'),
    }
    if any(source.get(key) != value for key, value in required.items()):
        raise ValueError('source config differs from approved sample-only protocol')
    budget = 20 if dataset == 'upmc_food101' else 50
    if source.get('epochs_per_task') != budget:
        raise ValueError('source training budget differs')
    if device is not None and device not in ('cuda:0', 'cuda:1'):
        raise ValueError('development device must be a qualified 3080 GPU')
    name = f'{dataset}_seed{seed}_{variant}' + ('_smoke' if smoke else '')
    result = {
        **source,
        'seed': seed,
        'pilot_protocol': PROTOCOL,
        'pilot_dataset': dataset,
        'pilot_variant': variant,
        'pilot_smoke': bool(smoke),
        'pilot_holdout_seed': HOLDOUT_SEED,
        'pilot_holdout_per_class': HOLDOUT_PER_CLASS[dataset],
        'pilot_source_config_sha256': source_sha,
        'pilot_source_config_path': str(source_path),
        'pilot_code_commit': code_commit,
        'results_dir': str(root),
        'output_dir': str(Path(root) / name),
        'exp_name': f'old_class_kd_dev_{name}',
    }
    if smoke:
        result.update(epochs_per_task=1, device='cpu')
        if dataset != 'cifar100':
            result.update(
                num_tasks=2,
                custom_tasks='|'.join(expected_tasks(dataset).split('|')[:2]),
            )
    elif device is not None:
        result['device'] = device
    return result


def run(config_path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    args = SimpleNamespace(**config)
    if (args.pilot_protocol != PROTOCOL
            or args.pilot_variant not in VARIANTS
            or args.pilot_code_commit != git_commit()
            or config['pilot_source_config_sha256']
            != SOURCE_SHA256[args.pilot_dataset]
            or sha256(args.pilot_source_config_path)
            != config['pilot_source_config_sha256']
            or args.pilot_holdout_seed != HOLDOUT_SEED
            or args.pilot_holdout_per_class
            != HOLDOUT_PER_CLASS[args.pilot_dataset]
            or args.epochs_per_task != (1 if args.pilot_smoke else
                                        (20 if args.pilot_dataset == 'upmc_food101' else 50))):
        raise ValueError('KD development protocol or code commit differs')
    if args.pilot_smoke and (args.device != 'cpu' or
                             (args.pilot_dataset == 'cifar100' and args.num_tasks != 10)):
        raise ValueError('invalid CPU preflight task schedule')
    if not args.pilot_smoke and args.device not in ('cuda:0', 'cuda:1'):
        raise ValueError('full development requires a qualified 3080 GPU')
    _identity(config, args.pilot_dataset,
              smoke=bool(args.pilot_smoke and args.pilot_dataset != 'cifar100'))
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('KD development checkout must be clean')
    validate_party_kd_variant(config, 'uniform')
    validate_sample_replay_only(args)
    validate_adaptive_head_consolidation(args)
    validate_source_data_hash(config)
    if any(event['type'] != 'CIL' for event in TaskManager(args).get_timeline()):
        raise ValueError('KD development requires a CIL-only stream')
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'config.json').write_bytes(config_path.read_bytes())
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
        raise RuntimeError('KD development reached the formal test')
    validate_training_only_artifacts(root)
    checkpoint = root / 'checkpoints' / 'formal_final.pt'
    payload = runner._decode_checkpoint_value(torch.load(
        checkpoint, map_location='cpu', weights_only=True))
    state = payload['cl_state']
    previous = sorted({int(class_id)
                       for task_id, classes in payload['seen_task_classes'].items()
                       if int(task_id) < args.num_tasks - 1
                       for class_id in classes})
    if args.pilot_variant == 'oldonly20':
        if state.get('kd_old_classes') != previous:
            raise ValueError('old-only checkpoint KD class scope differs')
    elif 'kd_old_classes' in state:
        raise ValueError('full-width control unexpectedly has old-only scope')
    record = {
        'status': 'training_only_preflight' if args.pilot_smoke
                  else 'paired_training_holdout_readout',
        'dataset': args.pilot_dataset, 'seed': args.seed,
        'variant': args.pilot_variant,
        'code_commit': args.pilot_code_commit,
        'config_sha256': sha256(root / 'config.json'),
        'checkpoint_sha256': sha256(checkpoint),
        'freeze_sha256': sha256(root / 'FORMAL_STATE_FROZEN.json'),
        'holdout_manifest_sha256': sha256(root / 'pilot_holdout' / 'manifest.json'),
        'old_class_count': len(previous),
    }
    if not args.pilot_smoke:
        record.update(evaluate_holdout(args, dataset, checkpoint))
    target = root / ('PREFLIGHT_GATE.json' if args.pilot_smoke
                     else 'HOLDOUT_READOUT.json')
    with target.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(record, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--source-config', type=Path, required=True)
    prepare.add_argument('--root', type=Path, required=True)
    prepare.add_argument('--dataset', choices=SOURCE_SHA256, required=True)
    prepare.add_argument('--seed', type=int, required=True)
    prepare.add_argument('--variant', choices=VARIANTS, required=True)
    prepare.add_argument('--smoke', action='store_true')
    prepare.add_argument('--device', choices=('cuda:0', 'cuda:1'))
    running = sub.add_parser('run')
    running.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        run(args.config)
        return
    source = json.loads(args.source_config.read_text(encoding='utf-8'))
    config = build_config(
        source, sha256(args.source_config), args.dataset, args.seed,
        args.variant, args.root, code_commit=git_commit(),
        smoke=args.smoke, device=args.device, source_path=args.source_config,
    )
    args.root.mkdir(parents=True, exist_ok=True)
    path = args.root / (Path(config['output_dir']).name + '.config.json')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(path)


if __name__ == '__main__':
    main()
