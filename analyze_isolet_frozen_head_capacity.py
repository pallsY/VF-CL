"""Read-only, non-deployable ISOLET frozen-encoder head-capacity screen."""

import argparse
import copy
import json
import statistics
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from adaptive_consolidation_audit import _safe_torch_load
from analyze_isolet_head_branches import score_branch, sha256
from data_utils import split_features
from determinism import derive_seed
from head_consolidation import consolidate_classifier, hash_top_state
from launch_isolet_replay_selection_pilot import PilotDataset
from models import TopModel, build_models
from vfl_trainer import VFLTrainer


def encode_loader(trainer, args, loader):
    features, labels = [], []
    with torch.no_grad():
        for bx, by in loader:
            parts = split_features(bx.to(args.device), args)
            party = [bottom(part) for bottom, part in zip(trainer.bottoms, parts)]
            features.append(trainer._aggregate(party).detach().cpu())
            labels.append(by.cpu())
    return torch.cat(features), torch.cat(labels)


def score_top(top, features, labels, task_classes):
    with torch.no_grad():
        logp = torch.log_softmax(top(features.to(next(top.parameters()).device)).double(), dim=1)
    return score_branch(logp.cpu(), labels, task_classes)


def one_seed(root, seed):
    run = root / f'seed{seed}-herding' / 'run'
    done = json.loads((run / 'PILOT_COMPLETE.json').read_text())
    readout = json.loads((run / 'HOLDOUT_READOUT.json').read_text())
    checkpoint_path = run / 'adaptive_final.pt'
    if (done['status'] != 'training_only_before_test'
            or sha256(checkpoint_path) != done['checkpoint_sha256']
            or (run / 'results.json').exists()):
        raise ValueError('frozen development source differs')
    config = json.loads((run / 'config.json').read_text())
    if (config['seed'] != seed or config['proto_lambda_a'] != 0.05
            or config['distill_weight'] != 0.10
            or config['feat_distill_weight'] != 0.02):
        raise ValueError('source tuple or seed differs')
    payload = _safe_torch_load(checkpoint_path)
    bundle = payload['cl_state']['adaptive_audit_bundle']
    if hash_top_state(bundle['installed_state']) != hash_top_state(payload['trainer_state']['top_model']):
        raise ValueError('installed head differs from frozen trainer')
    with tempfile.TemporaryDirectory(prefix='isolet-head-capacity-') as scratch:
        args = SimpleNamespace(**config)
        args.output_dir = scratch
        dataset = PilotDataset(args)
        if (sha256(Path(scratch) / 'pilot_holdout' / 'manifest.json')
                != done['holdout_manifest_sha256']
                or dataset.validation_indices & dataset.holdout_indices):
            raise ValueError('training-only holdout differs')
        bottoms, installed = build_models(args)
        trainer = VFLTrainer(bottoms, installed, args)
        trainer.load_state(payload['trainer_state'])
        for model in (*trainer.bottoms, trainer.top_model):
            model.eval()
        holdout_x, holdout_y = encode_loader(
            trainer, args, dataset.holdout_loader(list(range(26))),
        )
        full_logits, _bias_logits = trainer.top_model.branch_log_probabilities(
            holdout_x.to(args.device),
        )
        saved_full = score_branch(full_logits.cpu(), holdout_y, bundle['task_classes'])
        mixed = score_branch(
            trainer.top_model(holdout_x.to(args.device)).cpu(),
            holdout_y, bundle['task_classes'],
        )
        if abs(mixed['cil'] - readout['metrics']['cil']) > 1e-12:
            raise ValueError('holdout readout differs from frozen checkpoint')
        train_x, train_y = encode_loader(
            trainer, args, dataset.get_train_loader(list(range(26)), shuffle=False),
        )
    per_class = {c: train_x[train_y == c] for c in range(26)}
    available = {c: int(rows.size(0)) for c, rows in per_class.items()}
    minimum = min(available.values())
    if minimum < 150 or max(available.values()) > 160 or holdout_y.numel() != 1040:
        raise ValueError('training pool or holdout count differs')
    balanced = {}
    for c, rows in per_class.items():
        generator = torch.Generator().manual_seed(seed * 100 + c)
        balanced[c] = rows[torch.randperm(rows.size(0), generator=generator)[:minimum]]
    pre = TopModel(holdout_x.size(1), 26, cosine=bool(config['cosine_head']))
    pre.load_state_dict(bundle['pre_state'], strict=True)
    pre = pre.to(args.device).eval()
    fit_seed = derive_seed(seed, 'head_consolidation', 12)
    full20 = copy.deepcopy(pre)
    consolidate_classifier(
        full20, payload['cl_state']['global_protos'],
        0.01, 500, 0.01, 20, fit_seed, args.device,
        replay_embeddings=bundle['replay_embeddings'],
        replay_source='existing_20_per_class',
        persistent_raw_example_count=520,
    )
    if hash_top_state(full20) != bundle['result']['candidate_hashes']['full']:
        raise ValueError('20/class refit did not reproduce frozen Full head')
    refit20 = score_top(full20, holdout_x, holdout_y, bundle['task_classes'])
    if abs(refit20['cil'] - saved_full['cil']) > 1e-12:
        raise ValueError('20/class refit accuracy differs')
    full_pool = copy.deepcopy(pre)
    consolidate_classifier(
        full_pool, payload['cl_state']['global_protos'],
        0.01, 500, 0.01, minimum, fit_seed, args.device,
        replay_embeddings=balanced,
        replay_source='offline_full_training_pool_diagnostic',
        persistent_raw_example_count=0,
    )
    oracle = score_top(full_pool, holdout_x, holdout_y, bundle['task_classes'])
    return {
        'seed': seed, 'checkpoint_sha256': sha256(checkpoint_path),
        'holdout_manifest_sha256': done['holdout_manifest_sha256'],
        'gate_manifest_sha256': done['gate_manifest_sha256'],
        'training_counts_per_class': available,
        'balanced_oracle_per_class': minimum,
        'saved_full': saved_full, 'refit20': refit20,
        'offline_full_pool': oracle, 'mixed': mixed,
        'full_pool_head_sha256': hash_top_state(full_pool),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    cli = parser.parse_args()
    root = cli.root.resolve(strict=True)
    output = cli.output.resolve()
    if output.exists() or root in output.parents:
        raise ValueError('output must be new and outside source runs')
    rows = [one_seed(root, seed) for seed in (47, 48)]
    result = {'schema_version': 1, 'status': 'offline_capacity_diagnostic_only',
              'root': str(root), 'runs': rows,
              'mean_saved_full_cil': statistics.fmean(r['saved_full']['cil'] for r in rows),
              'mean_offline_full_pool_cil': statistics.fmean(
                  r['offline_full_pool']['cil'] for r in rows)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'runs'}, indent=2))
    for row in rows:
        print(row['seed'], row['saved_full']['cil'], row['offline_full_pool']['cil'],
              row['balanced_oracle_per_class'])


if __name__ == '__main__':
    main()
