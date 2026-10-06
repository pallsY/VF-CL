"""Read-only replay utility screen for saved vector VFL checkpoints."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from adaptive_consolidation_audit import _safe_torch_load
from data_utils import VFLDataset, split_features
from models import build_models
from vfl_trainer import VFLTrainer


@torch.no_grad()
def replay_scores(old_trainer, new_trainer, replay, seen, frozen_weights, args):
    """Aggregate fixed-model party utility and next-task logit change by class."""
    seen = [int(c) for c in seen]
    parties = int(args.num_parties)
    if (args.aggregation != 'concat' or parties < 2 or not seen
            or len(set(seen)) != len(seen) or set(replay) != set(seen)
            or set(frozen_weights) != set(seen)):
        raise ValueError('screen requires exact seen-class replay and concat parties')
    if any(len(trainer.bottoms) != parties for trainer in (old_trainer, new_trainer)):
        raise ValueError('party count differs from model')

    rows = []
    for class_id in seen:
        inputs = replay[class_id]
        if (not isinstance(inputs, torch.Tensor) or inputs.ndim < 2
                or inputs.size(0) == 0 or not torch.isfinite(inputs).all()):
            raise ValueError('replay is empty or non-finite')
        parts = split_features(inputs.to(args.device), args)
        if len(parts) != parties:
            raise ValueError('replay party split is malformed')

        outputs = []
        for trainer in (old_trainer, new_trainer):
            embeddings = [bottom(part) for bottom, part in zip(trainer.bottoms, parts)]
            top = trainer.top_model
            weight = top.classifier.weight
            if (weight.size(1) % parties or top.cosine
                    or bool(top._adaptive_enabled)):
                raise ValueError('screen requires an ordinary linear concat head')
            width = weight.size(1) // parties
            per_party = [F.linear(embedding,
                                  weight[:, p * width:(p + 1) * width], None)
                         for p, embedding in enumerate(embeddings)]
            full = sum(per_party)
            if top.classifier.bias is not None:
                full = full + top.classifier.bias
            if (not torch.isfinite(full).all()
                    or not all(torch.isfinite(x).all() for x in per_party)
                    or not torch.allclose(full, top(trainer._aggregate(embeddings)),
                                          atol=1e-5, rtol=1e-5)):
                raise ValueError('party logits do not reconstruct the full head')
            outputs.append((per_party, full))

        old_party, old_full = outputs[0]
        new_party, _ = outputs[1]
        position = seen.index(class_id)
        target = torch.full((inputs.size(0),), position,
                            dtype=torch.long, device=old_full.device)
        baseline = F.cross_entropy(old_full[:, seen].double(), target,
                                   reduction='none')
        utilities = torch.stack([
            (F.cross_entropy((old_full - logits)[:, seen].double(), target,
                             reduction='none') - baseline).mean()
            for logits in old_party
        ])
        positive = utilities.clamp_min(0)
        fallback = bool(positive.sum() == 0)
        utility_weights = (torch.full_like(positive, 1 / parties) if fallback
                           else positive / positive.sum())
        frozen = torch.as_tensor(frozen_weights[class_id], dtype=torch.float64,
                                 device=old_full.device)
        if (frozen.shape != (parties,) or not torch.isfinite(frozen).all()
                or (frozen < 0).any() or frozen.sum() <= 0):
            raise ValueError('frozen contribution weights are malformed')
        frozen = frozen / frozen.sum()
        drift = torch.stack([
            (new_logits[:, class_id].double()
             - old_logits[:, class_id].double()).square().mean()
            for old_logits, new_logits in zip(old_party, new_party)
        ])
        if not torch.isfinite(drift).all() or not torch.isfinite(utilities).all():
            raise ValueError('party utility or drift is non-finite')
        uniform = torch.full_like(drift, 1 / parties)
        weights = {
            'utility': utility_weights,
            'uniform': uniform,
            'frozen': frozen,
            'shuffled': utility_weights.roll(1),
        }
        rows.append({
            'class_id': class_id,
            'replay_count': int(inputs.size(0)),
            'utility_fallback': fallback,
            'marginal_utility': utilities.tolist(),
            'utility_weights': utility_weights.tolist(),
            'party_logit_drift': drift.tolist(),
            **{name: float((weight * drift).sum())
               for name, weight in weights.items()},
        })
    return rows


@torch.no_grad()
def validation_outcome(old_trainer, new_trainer, batches, seen, next_seen, args):
    """Measure one old class on validation, preserving the old-class CE scope."""
    seen = [int(c) for c in seen]
    next_seen = [int(c) for c in next_seen]
    if (not seen or len(set(seen)) != len(seen) or not set(seen) < set(next_seen)):
        raise ValueError('validation classes must add at least one new class')
    old_ce = new_ce = old_errors = new_errors = count = 0
    class_id = None
    for inputs, labels in batches:
        if (inputs.size(0) == 0 or labels.ndim != 1
                or labels.numel() != inputs.size(0)):
            raise ValueError('validation batch is malformed')
        labels = labels.to(args.device)
        if class_id is None:
            class_id = int(labels[0])
        if class_id not in seen or not bool((labels == class_id).all()):
            raise ValueError('validation loader must contain one old class')
        parts = split_features(inputs.to(args.device), args)
        outputs = []
        for trainer in (old_trainer, new_trainer):
            embeddings = [bottom(part) for bottom, part in zip(trainer.bottoms, parts)]
            logits = trainer.top_model(trainer._aggregate(embeddings))
            if not torch.isfinite(logits).all():
                raise ValueError('validation logits are non-finite')
            outputs.append(logits)
        old_logits, new_logits = outputs
        target = torch.full_like(labels, seen.index(class_id))
        old_ce += float(F.cross_entropy(old_logits[:, seen].double(), target,
                                        reduction='sum'))
        new_ce += float(F.cross_entropy(new_logits[:, seen].double(), target,
                                        reduction='sum'))
        old_errors += int((old_logits[:, seen].argmax(1) != target).sum())
        new_errors += int((new_logits[:, next_seen].argmax(1)
                           != next_seen.index(class_id)).sum())
        count += labels.numel()
    if count == 0:
        raise ValueError('validation class has no examples')
    old_ce, new_ce = old_ce / count, new_ce / count
    old_error, new_error = old_errors / count, new_errors / count
    return {
        'validation_count': count,
        'old_ce': old_ce,
        'new_ce': new_ce,
        'delta_ce': new_ce - old_ce,
        'old_cil_error': old_error,
        'new_cil_error': new_error,
        'delta_cil_error': new_error - old_error,
    }


def summarize_rows(rows):
    """Apply the prespecified within-boundary association and stop rule."""
    if not rows:
        raise ValueError('screen has no class-boundary rows')
    names = ('utility', 'uniform', 'frozen', 'shuffled')
    groups = {}
    for row in rows:
        groups.setdefault(int(row['boundary']), []).append(row)
        if not all(np.isfinite(float(row[key])) for key in (*names, 'delta_ce')):
            raise ValueError('screen row contains a non-finite score or outcome')

    centered = {name: [] for name in (*names, 'delta_ce')}
    positive_boundaries = eligible_boundaries = 0
    for group in groups.values():
        for name in centered:
            values = np.asarray([row[name] for row in group], dtype=np.float64)
            centered[name].extend((values - values.mean()).tolist())
        if len(group) >= 3:
            eligible_boundaries += 1
            utility = np.asarray([row['utility'] for row in group])
            outcome = np.asarray([row['delta_ce'] for row in group])
            if (np.ptp(utility) > 0 and np.ptp(outcome) > 0
                    and spearmanr(utility, outcome).statistic > 0):
                positive_boundaries += 1

    outcome = np.asarray(centered['delta_ce'])
    correlations = {}
    for name in names:
        values = np.asarray(centered[name])
        correlations[name] = (
            float(spearmanr(values, outcome).statistic)
            if np.ptp(values) > 0 and np.ptp(outcome) > 0 else None
        )
    utility = correlations['utility']
    controls = [correlations[name] for name in ('uniform', 'frozen')]
    passed = bool(utility is not None and utility > 0
                  and all(value is not None and utility - value >= 0.10
                          for value in controls)
                  and eligible_boundaries > 0
                  and positive_boundaries > eligible_boundaries / 2)
    return {
        'rows': len(rows),
        'boundaries': len(groups),
        'centered_spearman': correlations,
        'positive_boundaries': positive_boundaries,
        'eligible_boundaries': eligible_boundaries,
        'utility_fallback_fraction': sum(bool(row['utility_fallback']) for row in rows) / len(rows),
        'mean_abs_utility_uniform_score_gap': float(np.mean([
            abs(row['utility'] - row['uniform']) for row in rows
        ])),
        'passed': passed,
    }


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_trainer(checkpoint, task_id, seen, args):
    if (type(checkpoint) is not dict or checkpoint.get('task_id') != task_id
            or 'trainer_state' not in checkpoint or 'cl_state' not in checkpoint):
        raise ValueError('task checkpoint is malformed or out of order')
    task_classes = checkpoint.get('seen_task_classes')
    if (type(task_classes) is not dict
            or sorted(int(c) for classes in task_classes.values() for c in classes)
            != sorted(seen)):
        raise ValueError('checkpoint seen classes differ from task plan')
    bottoms, top = build_models(args)
    trainer = VFLTrainer(bottoms, top, args)
    trainer.load_state(checkpoint['trainer_state'])
    for model in (*trainer.bottoms, trainer.top_model):
        model.eval()
    return trainer


def screen_run(run_dir, output_dir):
    """Screen saved adjacent checkpoints and write only aggregate evidence."""
    run_dir = Path(run_dir).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError('screen output must be outside the source run')
    config_path = run_dir / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if (config.get('data') != 'tabvfl' or config.get('model_type') != 'mlp'
            or config.get('aggregation') != 'concat'
            or config.get('cosine_head') is not False
            or config.get('lambda_validation_enabled') != 1
            or config.get('party_drift_telemetry') not in (None, 1)
            or config.get('head_consolidation_mode', 'full_classifier')
            != 'full_classifier'):
        raise ValueError('screen requires vector concat diagnostic with validation')
    task_spec = config.get('custom_tasks')
    if type(task_spec) is not str or not task_spec:
        raise ValueError('screen requires an explicit task plan')
    tasks = [[int(c) for c in group.split(',')] for group in task_spec.split('|')]
    if (len(tasks) != config.get('num_tasks') or len(tasks) < 3
            or any(not task for task in tasks)
            or len(set(sum(tasks, []))) != sum(len(task) for task in tasks)):
        raise ValueError('task plan is malformed')
    output_dir.mkdir(parents=True, exist_ok=False)

    rows = []
    checkpoint_hashes = {}
    with tempfile.TemporaryDirectory(prefix='vfcl-party-utility-') as scratch:
        args = SimpleNamespace(**config)
        args.output_dir = scratch
        args.data_flow_audit = 0
        args.num_workers = 0
        dataset = VFLDataset(args)
        manifest_sha = dataset.validation_manifest['sha256']
        for boundary in range(len(tasks) - 2):
            seen = sum(tasks[:boundary + 1], [])
            next_seen = seen + tasks[boundary + 1]
            checkpoints = []
            for task_id, expected in ((boundary, seen), (boundary + 1, next_seen)):
                name = f'event_{task_id}_CIL.pt'
                path = run_dir / 'checkpoints' / name
                checkpoint_hashes[name] = _file_sha256(path)
                checkpoint = _safe_torch_load(path)
                checkpoints.append((checkpoint,
                                    _load_trainer(checkpoint, task_id, expected, args)))
            old_state, old_trainer = checkpoints[0]
            _, new_trainer = checkpoints[1]
            replay = {int(c): x for c, x in
                      old_state['cl_state']['head_raw_replay'].items()}
            frozen = {int(c): x for c, x in
                      old_state['cl_state']['class_party_weights'].items()}
            for row in replay_scores(old_trainer, new_trainer, replay,
                                     seen, frozen, args):
                outcome = validation_outcome(
                    old_trainer, new_trainer,
                    dataset.get_validation_loader([row['class_id']]),
                    seen, next_seen, args,
                )
                rows.append({'boundary': boundary, **row, **outcome})
        if dataset.validation_manifest['sha256'] != manifest_sha:
            raise ValueError('validation manifest changed during screen')

    result = {
        'schema_version': 1,
        'source_run': str(run_dir),
        'source_config_sha256': _file_sha256(config_path),
        'source_dataset_sha256': _file_sha256(config['vector_npz']),
        'source_checkpoints_sha256': checkpoint_hashes,
        'validation_manifest_sha256': manifest_sha,
        'excluded_final_transition': len(tasks) - 2,
        'summary': summarize_rows(rows),
        'rows': rows,
    }
    temporary = output_dir / 'screen.json.tmp'
    with open(temporary, 'x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_dir / 'screen.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    options = parser.parse_args()
    print(json.dumps(screen_run(options.run_dir, options.output_dir)['summary'],
                     indent=2, sort_keys=True))
