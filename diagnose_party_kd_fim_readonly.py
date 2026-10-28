"""Read-only old-exemplar representation and FIM-mask diagnostic.

Uses saved training exemplars and task checkpoints only. Output is hypothesis
generation, never a mechanism choice or a formal test result.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from data_utils import split_features
from models import build_models


def frozen_fraction(state, mask):
    if not mask:
        return 0.0
    if any(name not in state for name in mask):
        raise ValueError('FIM mask references a missing parameter')
    total = sum(state[name].numel() for name in mask)
    frozen = sum(state[name].numel() for name, marked in mask.items() if marked)
    return frozen / total


def _scatter_log_ratio(embeddings, labels):
    classes = labels.unique(sorted=True)
    if len(classes) < 2:
        raise ValueError('old exemplar probe needs at least two classes')
    means = torch.stack([embeddings[labels == cls].mean(0) for cls in classes])
    grand = means.mean(0)
    between = ((means - grand) ** 2).sum(1).mean()
    within = torch.stack([
        ((embeddings[labels == cls] - means[i]) ** 2).sum(1).mean()
        for i, cls in enumerate(classes)
    ]).mean()
    return float(torch.log((between + 1e-8) / (within + 1e-8)))


def drift_metrics(old, current, labels):
    if old.shape != current.shape or len(old) != len(labels) or old.ndim != 2:
        raise ValueError('representation probe shapes differ')
    with torch.no_grad():
        rel = (current - old).norm(dim=1) / old.norm(dim=1).clamp_min(1e-8)
        cosine = F.cosine_similarity(old, current, dim=1, eps=1e-8)
        old_scatter = _scatter_log_ratio(old, labels)
        current_scatter = _scatter_log_ratio(current, labels)
    return {
        'relative_l2_mean': float(rel.mean()),
        'cosine_mean': float(cosine.mean()),
        'scatter_log_old': old_scatter,
        'scatter_log_current': current_scatter,
        'scatter_log_change': current_scatter - old_scatter,
    }


def _saved_training_exemplars(snapshot):
    memory = snapshot['cl_state']['head_raw_replay']
    classes = sorted(int(cls) for cls in memory)
    if len(classes) < 2 or any(len(memory[cls]) != 20 for cls in classes):
        raise ValueError('reference20 memory is incomplete')
    values = [memory[cls].detach().cpu() for cls in classes]
    raw = torch.cat(values)
    labels = torch.cat([
        torch.full((len(value),), cls, dtype=torch.long)
        for cls, value in zip(classes, values)
    ])
    return raw, labels, classes


def _models_for_snapshot(snapshot, args):
    bottoms, _ = build_models(args)
    if len(bottoms) != len(snapshot['trainer_state']['bottoms']):
        raise ValueError('party count changed between configuration and checkpoint')
    for bottom, state in zip(bottoms, snapshot['trainer_state']['bottoms']):
        bottom.load_state_dict(state, strict=True)
        bottom.eval()
    return bottoms


def analyze_transition(previous, current, config):
    if current['task_id'] != previous['task_id'] + 1:
        raise ValueError('snapshots are not adjacent tasks')
    if config['sample_replay_only'] != 1 or config['model_type'] != 'mlp':
        raise ValueError('this diagnostic is restricted to sample-only vector MLP')
    args = SimpleNamespace(**{**config, 'device': 'cpu'})
    raw, labels, classes = _saved_training_exemplars(previous)
    state_prev = previous['trainer_state']['bottoms']
    state_curr = current['trainer_state']['bottoms']
    widths = [int(state['net.0.weight'].shape[1]) for state in state_prev]
    if sum(widths) != raw.shape[1] or len(widths) != args.num_parties:
        raise ValueError('saved raw input does not match bottom feature widths')
    offset, ranges = 0, []
    for width in widths:
        ranges.append((offset, offset + width))
        offset += width
    args.party_col_ranges = ranges
    old_bottoms = _models_for_snapshot(previous, args)
    new_bottoms = _models_for_snapshot(current, args)
    parts = split_features(raw, args)
    old_embeddings, new_embeddings, party_rows = [], [], []
    with torch.no_grad():
        for party, part in enumerate(parts):
            old = old_bottoms[party](part)
            new = new_bottoms[party](part)
            old_embeddings.append(old)
            new_embeddings.append(new)
            party_rows.append({
                'party': party,
                'fim_frozen_param_fraction_during_task': frozen_fraction(
                    state_prev[party], previous['cl_state']['fim_masks'][party]),
                'fim_frozen_param_fraction_after_task': frozen_fraction(
                    state_curr[party], current['cl_state']['fim_masks'][party]),
                **drift_metrics(old, new, labels),
            })
    return {
        'task_id': int(current['task_id']),
        'introduced_classes': [int(x) for x in current['introduced_classes']],
        'old_classes': len(classes),
        'old_training_exemplars': len(raw),
        'party': party_rows,
        'aggregate_concat': drift_metrics(
            torch.cat(old_embeddings, dim=1),
            torch.cat(new_embeddings, dim=1), labels),
    }


def analyze_run(run):
    run = Path(run)
    config_path = run / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    snapshots = sorted((run / 'formal_snapshots').glob('event_*_CIL.pt'),
                       key=lambda path: int(path.stem.split('_')[1]))
    if len(snapshots) != 13:
        raise ValueError('ISOLET reference20 should have exactly 13 task snapshots')
    rows = []
    previous = torch.load(snapshots[0], map_location='cpu', weights_only=True)
    for task_id, path in enumerate(snapshots[1:], start=1):
        current = torch.load(path, map_location='cpu', weights_only=True)
        if current['task_id'] != task_id:
            raise ValueError('snapshot task order differs')
        rows.append(analyze_transition(previous, current, config))
        previous = current
    return {
        'run': str(run),
        'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'rows': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    runs = [args.base / f'isolet_seed{seed}_reference20' for seed in (54, 55)]
    if any(not run.is_dir() for run in runs):
        raise FileNotFoundError('frozen ISOLET reference20 runs missing')
    if args.output.resolve().is_relative_to(args.base.resolve()):
        raise ValueError('diagnostic output must be outside source results')
    records = [analyze_run(run) for run in runs]
    summary = {}
    for party in range(4):
        vals = [row['party'][party] for record in records for row in record['rows']]
        summary[str(party)] = {
            'median_relative_l2': median(x['relative_l2_mean'] for x in vals),
            'mean_scatter_log_change': sum(x['scatter_log_change'] for x in vals) / len(vals),
            'high_fim_freeze_transition_count': sum(
                x['fim_frozen_param_fraction_during_task'] > 0.5 for x in vals),
            'transition_count': len(vals),
        }
    payload = {
        'scope': 'training_exemplars_only_hypothesis_generation',
        'dataset': 'isolet', 'seeds': [54, 55],
        'metric_note': ('Representation drift and scatter are descriptive endpoint '
                        'probes on herded old training exemplars, not causal '
                        'attribution or deployable selection signals.'),
        'runs': records, 'summary': summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'read_only_diagnostic_complete',
                      'runs': len(records), 'transitions': sum(
                          len(record['rows']) for record in records),
                      'output': str(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
