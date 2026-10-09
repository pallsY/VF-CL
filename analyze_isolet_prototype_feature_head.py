"""Read-only raw20 plus transient-prototype20 Full-head pilot."""

import argparse
import copy
import json
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from adaptive_consolidation_audit import _safe_torch_load
from analyze_isolet_frozen_head_capacity import encode_loader, score_top
from analyze_isolet_head_branches import sha256
from data_utils import split_features
from determinism import derive_seed
from head_consolidation import balanced_prototype_batch, consolidate_classifier, hash_top_state
import launch_isolet_replay_selection_pilot as pilot
from models import TopModel, build_models
from vfl_trainer import VFLTrainer


def synthetic_by_class(prototypes, seed, per_class):
    features, labels, classes = balanced_prototype_batch(
        prototypes, per_class, seed, 'cpu',
    )
    return {class_id: features[labels == class_id].detach().cpu()
            for class_id in classes}


def merge_raw_synthetic(raw, synthetic, raw_count, synthetic_count):
    if set(raw) != set(synthetic):
        raise ValueError('raw and synthetic class sets differ')
    combined = {}
    for class_id in sorted(raw):
        real = torch.as_tensor(raw[class_id]).detach().cpu()
        generated = torch.as_tensor(synthetic[class_id]).detach().cpu()
        if (real.ndim != 2 or generated.ndim != 2
                or real.shape != (raw_count, generated.shape[1])
                or generated.shape[0] != synthetic_count
                or not torch.isfinite(real).all()
                or not torch.isfinite(generated).all()):
            raise ValueError('raw/synthetic feature shape or finiteness differs')
        combined[class_id] = torch.cat((real, generated), dim=0)
    return combined


def one_seed(root, seed):
    run = root / f'seed{seed}-herding' / 'run'
    complete = json.loads((run / 'PILOT_COMPLETE.json').read_text())
    protocol = json.loads((run / 'PILOT_PROTOCOL.json').read_text())
    checkpoint_path = run / 'adaptive_final.pt'
    if (complete['status'] != 'training_only_before_test'
            or sha256(checkpoint_path) != complete['checkpoint_sha256']
            or protocol['holdout_split_seed'] != 20261011
            or protocol['selector'] != 'herding'
            or protocol['source_commit'] != 'a575bbf446ae501cf8e580ba62c2cc5492a25f30'
            or (run / 'results.json').exists()):
        raise ValueError('training-only seed source differs')
    config = json.loads((run / 'config.json').read_text())
    if (config['seed'] != seed or config['proto_lambda_a'] != 0.05
            or config['distill_weight'] != 0.10
            or config['feat_distill_weight'] != 0.02):
        raise ValueError('frozen source tuple differs')
    payload = _safe_torch_load(checkpoint_path)
    bundle = payload['cl_state']['adaptive_audit_bundle']
    raw = payload['cl_state']['head_raw_replay']
    if (set(raw) != set(range(26))
            or any(samples.size(0) != 20 for samples in raw.values())
            or hash_top_state(bundle['installed_state'])
            != hash_top_state(payload['trainer_state']['top_model'])):
        raise ValueError('source replay or installed head differs')
    pilot.HOLDOUT_SEED = 20261011
    with tempfile.TemporaryDirectory(prefix='isolet-proto-feature-head-') as scratch:
        args = SimpleNamespace(**config)
        args.output_dir = scratch
        dataset = pilot.PilotDataset(args)
        if (sha256(Path(scratch) / 'pilot_holdout' / 'manifest.json')
                != complete['holdout_manifest_sha256']
                or sha256(Path(scratch) / 'validation' / 'validation_manifest.json')
                != complete['gate_manifest_sha256']
                or dataset.validation_indices & dataset.holdout_indices):
            raise ValueError('fresh holdout or gate identity differs')
        bottoms, installed = build_models(args)
        trainer = VFLTrainer(bottoms, installed, args)
        trainer.load_state(payload['trainer_state'])
        for model in (*trainer.bottoms, trainer.top_model):
            model.eval()
        holdout_x, holdout_y = encode_loader(
            trainer, args, dataset.holdout_loader(list(range(26))),
        )
    if holdout_y.numel() != 1040:
        raise ValueError('fresh holdout count differs')
    pre = TopModel(holdout_x.size(1), 26, cosine=bool(config['cosine_head']))
    pre.load_state_dict(bundle['pre_state'], strict=True)
    pre = pre.to(args.device).eval()
    fit_seed = derive_seed(seed, 'head_consolidation', 12)
    full20 = copy.deepcopy(pre)
    torch.cuda.synchronize()
    started = time.perf_counter()
    consolidate_classifier(
        full20, payload['cl_state']['global_protos'],
        0.01, 500, 0.01, 20, fit_seed, args.device,
        replay_embeddings=bundle['replay_embeddings'],
        replay_source='current_raw20', persistent_raw_example_count=520,
    )
    torch.cuda.synchronize()
    control_seconds = time.perf_counter() - started
    if hash_top_state(full20) != bundle['result']['candidate_hashes']['full']:
        raise ValueError('control Full20 refit did not reproduce frozen state')
    control = score_top(full20, holdout_x, holdout_y, bundle['task_classes'])
    synthetic_seed = derive_seed(seed, 'head_proto_feature_aug', 12)
    started = time.perf_counter()
    generated = synthetic_by_class(
        payload['cl_state']['global_protos'], synthetic_seed, 20,
    )
    if set(generated) != set(range(26)):
        raise ValueError('prototype generation class coverage differs')
    combined = merge_raw_synthetic(bundle['replay_embeddings'], generated, 20, 20)
    generation_seconds = time.perf_counter() - started
    candidate = copy.deepcopy(pre)
    torch.cuda.synchronize()
    started = time.perf_counter()
    consolidate_classifier(
        candidate, payload['cl_state']['global_protos'],
        0.01, 500, 0.01, 40, fit_seed, args.device,
        replay_embeddings=combined,
        replay_source='raw20_plus_transient_prototype20',
        persistent_raw_example_count=520,
    )
    torch.cuda.synchronize()
    candidate_seconds = time.perf_counter() - started
    augmented = score_top(candidate, holdout_x, holdout_y, bundle['task_classes'])
    return {
        'seed': seed, 'checkpoint_sha256': sha256(checkpoint_path),
        'config_sha256': sha256(run / 'config.json'),
        'gate_manifest_sha256': complete['gate_manifest_sha256'],
        'holdout_manifest_sha256': complete['holdout_manifest_sha256'],
        'control_full20_sha256': hash_top_state(full20),
        'candidate_full40_sha256': hash_top_state(candidate),
        'synthetic_seed': synthetic_seed,
        'persistent_raw_per_class': 20,
        'transient_synthetic_per_class': 20,
        'control_fit_seconds': control_seconds,
        'generation_seconds': generation_seconds,
        'candidate_fit_seconds': candidate_seconds,
        'control': control, 'candidate': augmented,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    cli = parser.parse_args()
    root = cli.root.resolve(strict=True)
    output = cli.output.resolve()
    if output.exists() or root in output.parents:
        raise ValueError('analysis output must be new and outside source runs')
    rows = [one_seed(root, seed) for seed in (49, 50)]
    paired = [
        {'seed': row['seed'],
         'cil_delta': row['candidate']['cil'] - row['control']['cil'],
         'new_cil_delta': row['candidate']['new_cil'] - row['control']['new_cil'],
         'task_id_delta': row['candidate']['task_id'] - row['control']['task_id'],
         'old_to_old_error_delta': (
             sum(row['candidate']['task_confusion'][i][j]
                 for i in range(12) for j in range(12) if i != j)
             - sum(row['control']['task_confusion'][i][j]
                   for i in range(12) for j in range(12) if i != j)),
         } for row in rows
    ]
    mean_gain = statistics.fmean(row['cil_delta'] for row in paired)
    passed = (all(row['cil_delta'] > 0 and row['new_cil_delta'] >= -0.01
                  for row in paired) and mean_gain >= 0.01)
    result = {
        'schema_version': 1, 'status': 'training_holdout_head_only_pilot',
        'source_root': str(root),
        'decision_rule': 'both CIL deltas > 0, mean >= 0.01, each newest delta >= -0.01',
        'runs': rows, 'paired': paired, 'mean_cil_gain': mean_gain,
        'passed': passed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'paired': paired, 'mean_cil_gain': mean_gain,
                      'passed': passed}, indent=2))


if __name__ == '__main__':
    main()
