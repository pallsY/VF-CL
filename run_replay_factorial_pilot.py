"""Training-only paired replay-factor ablation on a disjoint holdout."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from calibration_split import build_manifest, manifest_indices, write_manifest
from data_utils import VFLDataset


ARMS = {
    'P1R1': (1, 1, 'adaptive_dual_branch'),
    'P0R1': (0, 1, 'adaptive_dual_branch'),
    'P1R0': (1, 0, 'full_classifier'),
    'P0R0': (0, 0, 'full_classifier'),
}
HOLDOUT_PER_CLASS = {'isolet': 40, 'upmc': 64, 'cifar': 25}
HOLDOUT_SEED = 20261013
DATASET_IDENTITY = {
    'isolet': ('tabvfl', 26, 'isolet_vfl.npz'),
    'upmc': ('tabvfl', 101, 'upmc_food101_vfl.npz'),
    'cifar': ('cifar100', 100, ''),
}


class StopBeforeTest(Exception):
    """The final model is frozen; formal test access is forbidden."""


def derive_config(source, root, dataset, seed, arm):
    if dataset not in DATASET_IDENTITY:
        raise ValueError('unknown pilot dataset')
    expected_data, expected_classes, expected_vector = DATASET_IDENTITY[dataset]
    if ((source.get('data'), source.get('num_classes'),
         Path(source.get('vector_npz', '')).name)
            != (expected_data, expected_classes, expected_vector)):
        raise ValueError('pilot source dataset identity differs')
    if (arm not in ARMS
            or seed not in (54, 55)
            or source.get('head_consolidation_mode') != 'adaptive_dual_branch'
            or source.get('head_consolidation_samples_per_class') != 40
            or not source.get('formal_deferred_evaluation')
            or (source.get('proto_lambda_a'), source.get('distill_weight'),
                source.get('feat_distill_weight')) != (0.05, 0.10, 0.02)
            or source.get('party_drift_telemetry', 0)):
        raise ValueError('pilot source or arm differs from locked design')
    p, r, mode = ARMS[arm]
    root = (PurePosixPath(root) if str(root).startswith('/')
            else Path(root).resolve())
    name = f'{dataset}_seed{seed}_{arm}'
    return {
        **source,
        'seed': seed,
        'proto_replay_gradient_enabled': p,
        'head_consolidation_enabled': r,
        'head_consolidation_mode': mode,
        'head_consolidation_samples_per_class': 20,
        'pilot_dataset': dataset,
        'pilot_holdout_seed': HOLDOUT_SEED,
        'pilot_holdout_per_class': HOLDOUT_PER_CLASS[dataset],
        'results_dir': str(root),
        'output_dir': str(root / name),
        'exp_name': f'replay_factorial_{name}',
    }


def build_holdout_manifest(targets, excluded, per_class, seed, dataset):
    manifest = build_manifest(
        targets, per_class=per_class, seed=seed,
        dataset=f'{dataset}-train', excluded_indices=excluded,
    )
    indices = manifest_indices(manifest)
    if indices & set(excluded):
        raise ValueError('pilot holdout overlaps calibration or validation')
    return manifest, indices


class PilotDataset(VFLDataset):
    def __init__(self, args):
        super().__init__(args)
        excluded = self.calibration_indices | self.validation_indices
        self.holdout_manifest, self.holdout_indices = build_holdout_manifest(
            self.trainset.targets, excluded,
            args.pilot_holdout_per_class, args.pilot_holdout_seed,
            args.pilot_dataset,
        )
        path = Path(args.output_dir) / 'pilot_holdout' / 'manifest.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        write_manifest(self.holdout_manifest, path)

    def _loader(self, dataset, indices, key, shuffle, audit=False):
        if key[0] == 'train':
            indices = [index for index in indices
                       if index not in getattr(self, 'holdout_indices', set())]
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
            'FORMAL_EVALUATION_PUBLISHED.json', 'results.json')):
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


@torch.no_grad()
def evaluate_holdout(args, dataset, checkpoint):
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
    state = payload['cl_state']
    raw = state['head_raw_replay']
    if args.head_consolidation_enabled:
        if (set(raw) != set(range(args.num_classes))
                or any(values.size(0) != 20 for values in raw.values())):
            raise ValueError('sample-backed head has wrong raw replay')
    elif raw:
        raise ValueError('no-head arm retained raw examples')
    proto = state['global_protos']
    if set(proto) != set(range(args.num_classes)):
        raise ValueError('prototype memory has wrong classes')
    return {
        'metrics': metrics,
        'raw_replay_per_class': 20 if args.head_consolidation_enabled else 0,
        'raw_replay_bytes': sum(
            value.numel() * value.element_size() for value in raw.values()
        ),
        'prototype_bytes': sum(
            value.numel() * value.element_size()
            for record in proto.values() for value in record.values()
        ),
    }


def sample_only_result_fields(args, measured):
    if not bool(getattr(args, 'sample_replay_only', 0)):
        return {}
    expected = {'prototype_bytes': 0, 'raw_replay_per_class': 20}
    for key, value in expected.items():
        if measured.get(key) != value:
            raise ValueError(f'sample replay only requires {key}={value}')
    if measured.get('raw_replay_bytes', 0) <= 0:
        raise ValueError('sample replay only requires raw_replay_bytes > 0')
    return {'variant': 'sample_replay_only'}


def run_pilot(config_path):
    import runner
    from config import (validate_adaptive_head_consolidation,
                        validate_party_kd_variant,
                        validate_sample_replay_only)
    from data_utils import TaskManager

    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    args = SimpleNamespace(**config)
    if (args.pilot_dataset not in DATASET_IDENTITY
            or (args.data, args.num_classes, Path(args.vector_npz).name)
            != DATASET_IDENTITY[args.pilot_dataset]
            or args.seed not in (54, 55)
            or args.pilot_holdout_seed != HOLDOUT_SEED
            or args.pilot_holdout_per_class
            != HOLDOUT_PER_CLASS[args.pilot_dataset]
            or args.head_consolidation_samples_per_class != 20
            or args.formal_deferred_evaluation is not True
            or (args.proto_replay_gradient_enabled,
                args.head_consolidation_enabled,
                args.head_consolidation_mode) not in ARMS.values()):
        raise ValueError('config differs from replay-factorial pilot')
    validate_party_kd_variant(config, config['expected_party_kd_variant'])
    validate_sample_replay_only(args)
    validate_adaptive_head_consolidation(args)
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
    source_path = run / 'checkpoints' / f'event_{args.num_tasks - 1}_CIL.pt'
    source = runner._decode_checkpoint_value(torch.load(
        source_path, map_location='cpu', weights_only=True,
    ))
    result = {
        'status': 'independent_training_holdout_readout',
        'arm': next(name for name, factors in ARMS.items() if factors == (
            args.proto_replay_gradient_enabled, args.head_consolidation_enabled,
            args.head_consolidation_mode)),
        'seed': args.seed,
        'dataset': args.pilot_dataset,
        'config_sha256': sha256(run / 'config.json'),
        'checkpoint_sha256': sha256(final_path),
        'freeze_sha256': sha256(run / 'FORMAL_STATE_FROZEN.json'),
        'holdout_manifest_sha256': sha256(
            run / 'pilot_holdout' / 'manifest.json'),
        'pre_head_trainer_sha256': runner.trainer_state_sha256(
            source['trainer_state']),
        **evaluate_holdout(args, dataset, final_path),
    }
    result.update(sample_only_result_fields(args, result))
    path = run / 'HOLDOUT_READOUT.json'
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n',
                    encoding='utf-8')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    run_pilot(parser.parse_args().config)
