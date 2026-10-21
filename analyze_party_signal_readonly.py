"""Read-only, hypothesis-generating analysis of party task signals."""

import argparse
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from diagnose_party_task_evidence import _task_scores
from diagnose_sampleonly_head import validate_run_inputs
from run_resource_lite_pilot import (
    PilotDataset, sha256, validate_training_only_artifacts,
    validate_source_data_hash,
)


def normalized_margin(scores):
    """Fixed, label-free task-score confidence, invariant to positive scale."""
    if scores.ndim < 2 or scores.size(-1) < 2:
        raise ValueError('at least two task scores are required')
    top = scores.topk(2, dim=-1).values
    return (top[..., 0] - top[..., 1]) / (
        scores.std(dim=-1, unbiased=False) + 1e-6)


def _wilson(correct, count):
    z = 1.959963984540054
    p = correct / count
    denom = 1 + z * z / count
    middle = (p + z * z / (2 * count)) / denom
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denom
    return [middle - radius, middle + radius]


def analyze_scores(party_scores, fused_scores, labels, task_classes):
    """Labels score a fixed selector; they never enter its selection rule."""
    from sklearn.metrics import roc_auc_score

    if (party_scores.ndim != 3 or fused_scores.ndim != 2
            or party_scores.size(1) != fused_scores.size(0)
            or party_scores.size(2) != fused_scores.size(1)
            or labels.numel() != fused_scores.size(0)):
        raise ValueError('task-score shapes differ')
    class_task = torch.full((max(max(v) for v in task_classes.values()) + 1,), -1)
    for task, classes in task_classes.items():
        class_task[list(map(int, classes))] = int(task)
    if (bool((class_task < 0).any()) or labels.min() < 0
            or labels.max() >= len(class_task)):
        raise ValueError('class-to-task mapping differs')
    truth = class_task[labels]
    party_pred = party_scores.argmax(-1)
    fused_pred = fused_scores.argmax(-1)
    party_correct = party_pred == truth.unsqueeze(0)
    fused_correct = fused_pred == truth
    margin = normalized_margin(party_scores)
    chosen = margin.argmax(0)
    selected_pred = party_pred.gather(0, chosen.unsqueeze(0)).squeeze(0)
    selected_correct = selected_pred == truth
    fused_errors = ~fused_correct

    class_rows = []
    for class_id in sorted(int(x) for x in labels.unique().tolist()):
        mask = labels == class_id
        count = int(mask.sum())
        correct = party_correct[:, mask].sum(1).tolist()
        accuracies = [value / count for value in correct]
        class_rows.append({
            'class_id': class_id, 'count': count,
            'party_task_correct_counts': correct,
            'party_task_accuracy': accuracies,
            'party_task_wilson95': [_wilson(value, count) for value in correct],
            'party_accuracy_range': max(accuracies) - min(accuracies),
        })
    task_rows = []
    for task, classes in sorted(task_classes.items()):
        mask = torch.isin(labels, torch.tensor(list(classes)))
        count = int(mask.sum())
        accuracies = (party_correct[:, mask].double().mean(1)).tolist()
        task_rows.append({
            'task_id': int(task), 'count': count,
            'party_task_accuracy': accuracies,
            'party_accuracy_range': max(accuracies) - min(accuracies),
        })
    aucs = []
    for party in range(party_scores.size(0)):
        target = party_correct[party].numpy().astype(int)
        aucs.append(float(roc_auc_score(target, margin[party].numpy()))
                    if len(set(target)) == 2 else None)
    summary = {
        'count': int(labels.numel()),
        'fused_task_accuracy': float(fused_correct.double().mean()),
        'party_task_accuracy': party_correct.double().mean(1).tolist(),
        'fused_error_count': int(fused_errors.sum()),
        'oracle_rescue_count': int((party_correct.any(0) & fused_errors).sum()),
        'selector_rescue_count': int((selected_correct & fused_errors).sum()),
        'selector_damage_count': int((~selected_correct & fused_correct).sum()),
        'selector_task_accuracy': float(selected_correct.double().mean()),
        'normalized_margin_auc_by_party': aucs,
        'mean_class_party_range': sum(row['party_accuracy_range'] for row in class_rows) / len(class_rows),
        'mean_task_party_range': sum(row['party_accuracy_range'] for row in task_rows) / len(task_rows),
    }
    return {'summary': summary, 'per_class': class_rows, 'per_task': task_rows}


def compare_seeds(first, second):
    rows_a, rows_b = first['per_class'], second['per_class']
    if [row['class_id'] for row in rows_a] != [row['class_id'] for row in rows_b]:
        raise ValueError('class orders differ across seeds')
    a = torch.tensor([row['party_task_accuracy'] for row in rows_a], dtype=torch.double)
    b = torch.tensor([row['party_task_accuracy'] for row in rows_b], dtype=torch.double)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError('party accuracy matrices differ')
    max_a, max_b = a.max(1).values, b.max(1).values
    unique_a = (a == max_a[:, None]).sum(1) == 1
    unique_b = (b == max_b[:, None]).sum(1) == 1
    both = unique_a & unique_b
    matches = (a.argmax(1) == b.argmax(1)) & both
    residual_a = a - a.mean(1, keepdim=True) - a.mean(0, keepdim=True) + a.mean()
    residual_b = b - b.mean(1, keepdim=True) - b.mean(0, keepdim=True) + b.mean()
    x, y = residual_a.flatten(), residual_b.flatten()
    denom = x.norm() * y.norm()
    corr = float((x @ y) / denom) if denom > 0 else None
    return {
        'class_count': int(a.size(0)),
        'seed54_tied_winner_classes': int((~unique_a).sum()),
        'seed55_tied_winner_classes': int((~unique_b).sum()),
        'unique_winner_classes': int(both.sum()),
        'winner_agreement_fraction': float(matches.sum() / both.sum()) if both.any() else None,
        'class_party_residual_correlation': corr,
    }


def validate_output_root(output_root, source_roots):
    output = Path(output_root).resolve()
    if output.exists() or any(output == Path(source).resolve()
                              or output.is_relative_to(Path(source).resolve())
                              for source in source_roots):
        raise ValueError('diagnostic root must be fresh and outside frozen sources')


def analyze_one(run):
    import runner
    from data_utils import split_features
    from models import build_models
    from vfl_trainer import VFLTrainer

    run = Path(run)
    config, readout, _, final = validate_run_inputs(run)
    validate_training_only_artifacts(run)
    validate_source_data_hash(config)
    if config['pilot_dataset'] != 'isolet' or config['seed'] not in (54, 55):
        raise ValueError('only frozen ISOLET reference20 seeds 54/55 are approved')
    if config['aggregation'] not in ('concat', 'sum') or config['cosine_head']:
        raise ValueError('party decomposition requires a linear head')
    paths = {
        'config': run / 'config.json',
        'readout': run / 'HOLDOUT_READOUT.json',
        'manifest': run / 'pilot_holdout' / 'manifest.json',
        'freeze': run / 'FORMAL_STATE_FROZEN.json',
        'checkpoint': final,
        'data_flow': run / 'data_flow_audit.jsonl',
    }
    before = {name: sha256(path) for name, path in paths.items()}
    if before['freeze'] != readout.get('freeze_sha256'):
        raise ValueError('freeze record differs from frozen readout')
    with tempfile.TemporaryDirectory(prefix='vfcl-party-signal-readonly-') as tmp:
        args = SimpleNamespace(**config)
        args.output_dir = tmp
        args.device = 'cpu'
        dataset = PilotDataset(args)
        if sha256(Path(tmp) / 'pilot_holdout' / 'manifest.json') != before['manifest']:
            raise ValueError('reconstructed holdout manifest differs')
        payload = runner._decode_checkpoint_value(torch.load(
            final, map_location='cpu', weights_only=True))
        state = payload['cl_state']['adaptive_audit_bundle']['full_state']
        weights, bias = state['classifier.weight'], state['classifier.bias']
        bottoms, top = build_models(args)
        trainer = VFLTrainer(bottoms, top, args)
        trainer.load_state(payload['trainer_state'])
        for model in (*trainer.bottoms, trainer.top_model):
            model.eval()
        party_rows = [[] for _ in range(args.num_parties)]
        fused_rows, label_rows = [], []
        with torch.no_grad():
            for batch_x, batch_y in dataset.holdout_loader(
                    list(range(args.num_classes))):
                pieces = split_features(batch_x.to('cpu'), args)
                embeddings = [bottom(piece) for bottom, piece in zip(
                    trainer.bottoms, pieces)]
                sections = (torch.split(weights,
                    [embedding.size(1) for embedding in embeddings], dim=1)
                    if args.aggregation == 'concat'
                    else [weights] * args.num_parties)
                rows = [F.linear(embedding, section, None)
                        for embedding, section in zip(embeddings, sections)]
                fused = F.linear(trainer._aggregate(embeddings), weights, bias)
                if not torch.allclose(sum(rows) + bias, fused, atol=1e-5):
                    raise ValueError('party logits do not sum to full head')
                for bucket, row in zip(party_rows, rows):
                    bucket.append(row.cpu())
                fused_rows.append(fused.cpu())
                label_rows.append(batch_y.long().cpu())
        task_classes = payload['seen_task_classes']
        party_scores = torch.stack([
            _task_scores(torch.cat(rows), task_classes)
            for rows in party_rows])
        fused_scores = _task_scores(torch.cat(fused_rows), task_classes)
        labels = torch.cat(label_rows)
        analysis = analyze_scores(party_scores, fused_scores, labels, task_classes)
    if (analysis['summary']['count'] != 1040
            or len(analysis['per_class']) != 26
            or any(row['count'] != 40 for row in analysis['per_class'])):
        raise ValueError('ISOLET holdout counts differ')
    if {name: sha256(path) for name, path in paths.items()} != before:
        raise RuntimeError('frozen source changed during read-only analysis')
    return {
        'seed': int(config['seed']), 'source_run': str(run),
        'source_sha256': before, 'holdout_manifest_sha256': before['manifest'],
        'full_branch_analysis': analysis,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-root', required=True, type=Path)
    args = parser.parse_args()
    source = Path('/brf/users/yangxx/VFCL-results/resource-lite-dev-full-v2-20261018')
    runs = [source / f'isolet_seed{seed}_reference20' for seed in (54, 55)]
    validate_output_root(args.diagnostic_root, runs)
    torch.set_num_threads(2)
    results = [analyze_one(run) for run in runs]
    if results[0]['holdout_manifest_sha256'] != results[1]['holdout_manifest_sha256']:
        raise ValueError('paired seeds do not share a holdout manifest')
    pair = compare_seeds(
        results[0]['full_branch_analysis'], results[1]['full_branch_analysis'])
    for item in results:
        if any(sha256(Path(item['source_run']) / {'config': 'config.json',
                'readout': 'HOLDOUT_READOUT.json',
                'manifest': 'pilot_holdout/manifest.json',
                'freeze': 'FORMAL_STATE_FROZEN.json',
                'checkpoint': 'checkpoints/formal_final.pt',
                'data_flow': 'data_flow_audit.jsonl'}[name]) != digest
               for name, digest in item['source_sha256'].items()):
            raise RuntimeError('source changed before diagnostic publication')
    report = {
        'status': 'training_holdout_readonly_hypothesis_generation',
        'protocol': '2026-10-21-party-signal-readonly-analysis',
        'not_for_final_mechanism_hardcoding': True,
        'sources': results,
        'cross_seed': pair,
    }
    args.diagnostic_root.mkdir(parents=True, exist_ok=False)
    output = args.diagnostic_root / 'PARTY_SIGNAL_READONLY.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({
        'output': str(output), 'cross_seed': pair,
        'seed_summary': {str(item['seed']): item['full_branch_analysis']['summary']
                         for item in results},
    }, sort_keys=True))


if __name__ == '__main__':
    main()
