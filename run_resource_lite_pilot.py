"""Training-only holdout trial; stop before the official test is opened."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from calibration_split import build_manifest, manifest_indices, write_manifest
from data_utils import VFLDataset


HOLDOUT_SEED = 20261013
HOLDOUT_PER_CLASS = {'isolet': 40, 'upmc_food101': 64, 'cifar100': 25}
DATASET_IDENTITY = {
    'isolet': ('tabvfl', 26, 13, 'isolet/isolet_vfl.npz'),
    'upmc_food101': (
        'tabvfl', 101, 10,
        'upmc_food101_image_clean_4party_v1/upmc_food101_vfl.npz',
    ),
    'cifar100': ('cifar100', 100, 10, ''),
}
VECTOR_SHA256 = {
    'isolet': 'd34312670de93198afcd2b126c95b79bae2b4cffeb30d480f097faf046b69514',
    'upmc_food101': 'dbdd3a2979203ba0ee804de7d3b128426f869b0a127e4b87ebbe023e5750023b',
}
VARIANTS = ('reference20', 'lite18')
PROTOCOL = 'resource-lite-train-holdout-v1'


class StopBeforeTest(Exception):
    """The model was frozen; official test access is forbidden in development."""


def expected_tasks(dataset):
    if dataset == 'isolet':
        sizes = [2] * 13
    elif dataset == 'upmc_food101':
        sizes = [11] + [10] * 9
    elif dataset == 'cifar100':
        sizes = [10] * 10
    else:
        raise ValueError('unknown pilot dataset identity')
    tasks, start = [], 0
    for size in sizes:
        tasks.append(','.join(str(value) for value in range(start, start + size)))
        start += size
    return '|'.join(tasks)


def _identity(source, dataset, *, smoke=False):
    if dataset not in DATASET_IDENTITY:
        raise ValueError('unknown pilot dataset identity')
    data, classes, tasks, vector_suffix = DATASET_IDENTITY[dataset]
    path = source.get('vector_npz', '')
    schedule = ('' if dataset == 'cifar100' and not smoke
                else expected_tasks(dataset))
    if smoke:
        schedule = '|'.join(schedule.split('|')[:2])
    if (source.get('data') != data
            or source.get('num_classes') != classes
            or source.get('num_tasks') != (2 if smoke else tasks)
            or source.get('num_parties') != 4
            or source.get('custom_tasks') != schedule
            or (dataset == 'cifar100' and source.get('classes_per_task') != 10)
            or (not path if vector_suffix else bool(path))
            or (vector_suffix and not path.replace('\\', '/').endswith(
                '/data/' + vector_suffix))):
        raise ValueError('pilot dataset identity differs from four-party protocol')


def derive_config(source, root, dataset, seed, variant, *, smoke=False):
    """Create a new dev config from one frozen four-party sample-only config."""
    _identity(source, dataset)
    if seed not in (54, 55):
        raise ValueError('development seed must be 54 or 55')
    if variant not in VARIANTS:
        raise ValueError('unknown development variant')
    required = {
        'seed': 42, 'cl_method': 'proto_evolve',
        'sample_replay_only': 1, 'head_consolidation_enabled': 1,
        'head_consolidation_schedule': 'final',
        'head_consolidation_mode': 'adaptive_dual_branch',
        'head_consolidation_samples_per_class': 20,
        'head_consolidation_steps': 500,
        'formal_deferred_evaluation': True,
        'proto_replay_gradient_enabled': 0,
        'proto_sdc': False, 'feat_distill_weight': 0.0,
        'lambda_validation_enabled': 1,
        'distill_weight': 0.25,
        'party_kd_enabled': 1,
        'dep_tracking_enabled': 1,
        'expected_party_kd_variant': 'uniform',
        'party_kd_mode': 'uniform',
    }
    if any(source.get(key) != value for key, value in required.items()):
        raise ValueError('source config drifted from frozen sample-only protocol')
    if bool(source.get('bic_enabled', 0)) != (dataset == 'cifar100'):
        raise ValueError('source calibration protocol drifted')
    root = (PurePosixPath(root) if str(root).startswith('/')
            else Path(root).resolve())
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
        'results_dir': str(root),
        'output_dir': str(root / name),
        'exp_name': f'resource_lite_dev_{name}',
    }
    if variant == 'lite18':
        result.update(
            resource_lite=1,
            head_consolidation_mode='full_classifier',
            head_consolidation_samples_per_class=18,
            distill_weight=0.0,
            party_kd_enabled=0,
            dep_tracking_enabled=0,
            expected_party_kd_variant='base',
            lambda_validation_enabled=1 if dataset == 'cifar100' else 0,
        )
    else:
        result['resource_lite'] = 0
    if smoke:
        tasks = expected_tasks(dataset).split('|')
        if len(tasks) < 2:
            raise ValueError('smoke requires at least two locked tasks')
        result.update(
            num_tasks=2, custom_tasks='|'.join(tasks[:2]),
            epochs_per_task=1, device='cpu',
        )
    return result


def build_holdout_manifest(targets, excluded, per_class, seed, dataset):
    manifest = build_manifest(
        targets, per_class=per_class, seed=seed,
        dataset=f'{dataset}-train', excluded_indices=excluded,
    )
    indices = manifest_indices(manifest)
    if indices & set(excluded):
        raise ValueError('pilot holdout overlaps calibration or validation')
    return manifest, indices


def paired_vector_validation(targets, args):
    """Reconstruct the reference split without granting validation access."""
    manifest = build_manifest(
        targets,
        per_class=args.lambda_validation_per_class,
        seed=args.lambda_validation_split_seed,
        dataset=f'{Path(args.vector_npz).name}-train',
    )
    return manifest, manifest_indices(manifest)


class PilotDataset(VFLDataset):
    def __init__(self, args):
        super().__init__(args)
        if args.data == 'tabvfl':
            paired_manifest, paired_validation = paired_vector_validation(
                self.trainset.targets, args)
            if (getattr(args, 'lambda_validation_enabled', 0)
                    and paired_validation != self.validation_indices):
                raise ValueError('paired validation reconstruction differs')
        else:
            paired_manifest = self.validation_manifest
            paired_validation = self.validation_indices
        self.pilot_pair_excluded = (
            self.calibration_indices | paired_validation)
        excluded = self.pilot_pair_excluded
        self.holdout_manifest, self.holdout_indices = build_holdout_manifest(
            self.trainset.targets, excluded, args.pilot_holdout_per_class,
            args.pilot_holdout_seed, args.pilot_dataset,
        )
        if args.data == 'cifar100' and not hasattr(self, 'validationset'):
            from torchvision import datasets, transforms
            norm = transforms.Normalize([.507, .487, .441], [.267, .256, .276])
            self.validationset = datasets.CIFAR100(
                args.data_path, True, download=False,
                transform=transforms.Compose([transforms.ToTensor(), norm]),
            )
        path = Path(args.output_dir) / 'pilot_holdout' / 'manifest.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        write_manifest(self.holdout_manifest, path)
        if paired_manifest is not None:
            write_manifest(
                paired_manifest,
                path.parent / 'paired_validation_manifest.json',
            )

    def _loader(self, dataset, indices, key, shuffle, audit=False):
        if key[0] == 'train':
            indices = [index for index in indices
                       if index not in getattr(self, 'holdout_indices', set())
                       and index not in getattr(self, 'pilot_pair_excluded', set())]
        return super()._loader(dataset, indices, key, shuffle, audit=audit)

    def holdout_loader(self, classes):
        view = getattr(self, 'validationset', self.trainset)
        indices = [index for index in self.get_class_indices(view, classes)
                   if index in self.holdout_indices]
        return self._loader(
            view, indices,
            ('pilot_holdout', tuple(int(c) for c in classes)), False,
        )

    def authorize_formal_access(self, **kwargs):
        if kwargs.get('split') == 'test':
            history, pending = self._formal_durable_state()
            if pending:
                raise RuntimeError('pilot has an unconsumed formal access')
            self._validate_formal_phase_transition({'split': 'test'}, history)
            raise StopBeforeTest
        return super().authorize_formal_access(**kwargs)

    def get_test_loader(self, classes):
        raise AssertionError('pilot must not create a formal test loader')


def validate_training_only_artifacts(run):
    run = Path(run)
    for name in ('FORMAL_STATE_FROZEN.json',
                 'FORMAL_EVALUATION_PENDING.json'):
        if not (run / name).is_file():
            raise ValueError(f'incomplete training-only freeze: {name}')
    if not (run / 'checkpoints' / 'formal_final.pt').is_file():
        raise ValueError('incomplete training-only final checkpoint')
    if any((run / name).exists() for name in (
            'FORMAL_EVALUATION_CONSUMING.json',
            'FORMAL_EVALUATION_PUBLISHED.json',
            'FORMAL_EVALUATION_COMPLETE.json', 'results.json')):
        raise ValueError('pilot reached formal test access or publication')
    audit = run / 'data_flow_audit.jsonl'
    if not audit.is_file():
        raise ValueError('missing data flow audit')
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    if any(row.get('split') == 'test'
           or str(row.get('loader_key', '')).startswith("('test'")
           for row in rows):
        raise ValueError('pilot reached formal test access')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_source_data_hash(config):
    dataset = config['pilot_dataset']
    if dataset in VECTOR_SHA256:
        if sha256(config['vector_npz']) != VECTOR_SHA256[dataset]:
            raise ValueError('pilot vector data hash differs from frozen protocol')


@torch.no_grad()
def score_holdout_checkpoint(args, dataset, checkpoint):
    import runner
    from analyze_isolet_head_branches import score_branch
    from data_utils import split_features
    from models import build_models
    from vfl_trainer import VFLTrainer

    payload = runner._decode_checkpoint_value(torch.load(
        checkpoint, map_location='cpu', weights_only=True,
    ))
    bottoms, top = build_models(args)
    trainer = VFLTrainer(bottoms, top, args)
    trainer.load_state(payload['trainer_state'])
    for model in (*trainer.bottoms, trainer.top_model):
        model.eval()
    logits, labels = [], []
    for batch_x, batch_y in dataset.holdout_loader(
            list(range(args.num_classes))):
        parts = split_features(batch_x.to(args.device), args)
        embedded = [bottom(part) for bottom, part in zip(trainer.bottoms, parts)]
        logits.append(trainer.top_model(trainer._aggregate(embedded)).cpu())
        labels.append(batch_y.cpu())
    metrics = score_branch(
        F.log_softmax(torch.cat(logits), dim=1), torch.cat(labels).long(),
        payload['seen_task_classes'],
    )
    if metrics['count'] != args.num_classes * args.pilot_holdout_per_class:
        raise ValueError('pilot holdout example count differs')
    return metrics


@torch.no_grad()
def evaluate_holdout(args, dataset, checkpoint):
    import runner

    metrics = score_holdout_checkpoint(args, dataset, checkpoint)
    payload = runner._decode_checkpoint_value(torch.load(
        checkpoint, map_location='cpu', weights_only=True,
    ))
    state = payload['cl_state']
    raw = state['head_raw_replay']
    expected = 18 if args.pilot_variant == 'lite18' else 20
    if (set(raw) != set(range(args.num_classes))
            or any(value.size(0) != expected for value in raw.values())
            or state.get('global_protos') != {}
            or state.get('prev_protos') != {}):
        raise ValueError('pilot retained cache differs from variant')
    if args.pilot_variant == 'lite18' and state.get('has_old_teacher'):
        raise ValueError('resource-lite retained a frozen teacher')
    return {
        'metrics': metrics,
        'raw_replay_per_class': expected,
        'raw_replay_bytes': sum(
            value.numel() * value.element_size() for value in raw.values()),
        'prototype_bytes': 0,
    }


def run_pilot(config_path):
    import runner
    from config import (validate_adaptive_head_consolidation,
                        validate_party_kd_variant,
                        validate_sample_replay_only)
    from data_utils import TaskManager

    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    args = SimpleNamespace(**config)
    _identity(config, args.pilot_dataset,
              smoke=bool(getattr(args, 'pilot_smoke', False)))
    if (args.pilot_protocol != PROTOCOL or args.seed not in (54, 55)
            or args.pilot_variant not in VARIANTS
            or args.pilot_holdout_seed != HOLDOUT_SEED
            or args.pilot_holdout_per_class
            != HOLDOUT_PER_CLASS[args.pilot_dataset]
            or args.formal_deferred_evaluation is not True
            or args.head_consolidation_samples_per_class
            != (18 if args.pilot_variant == 'lite18' else 20)
            or bool(args.resource_lite) != (args.pilot_variant == 'lite18')):
        raise ValueError('config differs from resource-lite pilot')
    if bool(getattr(args, 'pilot_smoke', False)):
        if args.epochs_per_task != 1 or args.device != 'cpu':
            raise ValueError('smoke must use one epoch per task on CPU')
    elif args.epochs_per_task != (20 if args.pilot_dataset == 'upmc_food101' else 50):
        raise ValueError('full development budget drifted')
    if args.pilot_variant == 'reference20' and (
            args.distill_weight != 0.25
            or args.expected_party_kd_variant != 'uniform'
            or args.party_kd_enabled != 1
            or args.dep_tracking_enabled != 1
            or args.lambda_validation_enabled != 1):
        raise ValueError('reference20 config drifted')
    if (args.pilot_variant == 'lite18'
            and bool(args.lambda_validation_enabled)
            != (args.pilot_dataset == 'cifar100')):
        raise ValueError('lite18 validation access drifted')
    validate_party_kd_variant(config, config['expected_party_kd_variant'])
    validate_sample_replay_only(args)
    validate_adaptive_head_consolidation(args)
    validate_source_data_hash(config)
    if any(event['type'] != 'CIL' for event in TaskManager(args).get_timeline()):
        raise ValueError('pilot must have a CL-only task stream')
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
        raise RuntimeError('pilot did not stop before formal test')
    validate_training_only_artifacts(run)
    final_path = run / 'checkpoints' / 'formal_final.pt'
    if bool(getattr(args, 'pilot_smoke', False)):
        gate = {
            'status': 'two_task_training_only_smoke',
            'dataset': args.pilot_dataset,
            'variant': args.pilot_variant,
            'seed': args.seed,
            'config_sha256': sha256(run / 'config.json'),
            'checkpoint_sha256': sha256(final_path),
            'freeze_sha256': sha256(run / 'FORMAL_STATE_FROZEN.json'),
            'holdout_manifest_sha256': sha256(
                run / 'pilot_holdout' / 'manifest.json'),
        }
        (run / 'SMOKE_GATE.json').write_text(
            json.dumps(gate, indent=2, sort_keys=True) + '\n',
            encoding='utf-8')
        print(json.dumps(gate, sort_keys=True))
        return
    result = {
        'status': 'independent_training_holdout_readout',
        'variant': args.pilot_variant,
        'seed': args.seed,
        'dataset': args.pilot_dataset,
        'config_sha256': sha256(run / 'config.json'),
        'checkpoint_sha256': sha256(final_path),
        'freeze_sha256': sha256(run / 'FORMAL_STATE_FROZEN.json'),
        'holdout_manifest_sha256': sha256(
            run / 'pilot_holdout' / 'manifest.json'),
        **evaluate_holdout(args, dataset, final_path),
    }
    path = run / 'HOLDOUT_READOUT.json'
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n',
                    encoding='utf-8')
    print(json.dumps(result, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--source-config', type=Path, required=True)
    prepare.add_argument('--root', required=True)
    prepare.add_argument('--dataset', choices=DATASET_IDENTITY, required=True)
    prepare.add_argument('--seed', type=int, required=True)
    prepare.add_argument('--variant', choices=VARIANTS, required=True)
    prepare.add_argument('--smoke', action='store_true')
    run = sub.add_parser('run')
    run.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        run_pilot(args.config)
        return
    source = json.loads(args.source_config.read_text(encoding='utf-8'))
    config = derive_config(
        source, args.root, args.dataset, args.seed, args.variant,
        smoke=args.smoke)
    target = Path(args.root) / (
        f'{args.dataset}_seed{args.seed}_{args.variant}'
        f'{"_smoke" if args.smoke else ""}.config.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(config, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print(target)


if __name__ == '__main__':
    main()
