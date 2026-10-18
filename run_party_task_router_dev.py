"""Evaluate a frozen four-party task router without formal test access."""

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from diagnose_sampleonly_head import validate_run_inputs
from party_task_router_dev import fit_router, route
from run_resource_lite_pilot import (
    PilotDataset, score_branch, sha256, validate_training_only_artifacts,
)


@torch.no_grad()
def _scores(loader, args, trainer, full_state):
    from data_utils import split_features

    weights = full_state['classifier.weight']
    bias = full_state['classifier.bias']
    party_rows = [[] for _ in range(args.num_parties)]
    base_rows, labels, aggregates = [], [], []
    for batch_x, batch_y in loader:
        parts = split_features(batch_x.to('cpu'), args)
        embeddings = [bottom(part) for bottom, part in zip(
            trainer.bottoms, parts)]
        aggregate = trainer._aggregate(embeddings)
        if args.aggregation == 'concat':
            sections = torch.split(
                weights, [embedding.size(1) for embedding in embeddings],
                dim=1)
        elif args.aggregation == 'sum':
            sections = [weights] * args.num_parties
        else:
            raise ValueError('unknown aggregation')
        rows = [F.linear(embedding, section, None) for embedding, section
                in zip(embeddings, sections)]
        full = F.linear(aggregate, weights, bias)
        if not torch.allclose(sum(rows) + bias, full, atol=1e-5):
            raise ValueError('party logits do not reconstruct full head')
        branch_full, _ = trainer.top_model.branch_log_probabilities(aggregate)
        if not torch.allclose(F.log_softmax(full.double(), dim=1),
                              branch_full, atol=1e-5):
            raise ValueError('full branch differs from party reconstruction')
        for bucket, row in zip(party_rows, rows):
            bucket.append(row.double().cpu())
        base_rows.append(trainer.top_model(aggregate).double().cpu())
        labels.append(batch_y.long().cpu())
        aggregates.append(aggregate.cpu())
    return (torch.cat(base_rows),
            [torch.cat(bucket) for bucket in party_rows],
            torch.cat(labels), torch.cat(aggregates))


def evaluate_run(run):
    import runner
    from adaptive_consolidation_audit import _canonical_validation_evidence
    from models import build_models
    from vfl_trainer import VFLTrainer

    run = Path(run)
    config, readout, _, final = validate_run_inputs(run)
    validate_training_only_artifacts(run)
    if config.get('cosine_head') or config['num_parties'] != 4:
        raise ValueError('router requires a linear four-party head')
    payload = runner._decode_checkpoint_value(torch.load(
        final, map_location='cpu', weights_only=True))
    bundle = payload['cl_state']['adaptive_audit_bundle']
    with tempfile.TemporaryDirectory(prefix='vfcl-party-router-') as tmp:
        args = SimpleNamespace(**config)
        args.output_dir = tmp
        args.device = 'cpu'
        dataset = PilotDataset(args)
        if sha256(Path(tmp) / 'pilot_holdout' / 'manifest.json') \
                != readout['holdout_manifest_sha256']:
            raise ValueError('reconstructed holdout manifest differs')
        if (dataset.validation_manifest['sha256']
                != payload['cl_state']['head_validation_sha256']):
            raise ValueError('training validation manifest differs')
        bottoms, top = build_models(args)
        trainer = VFLTrainer(bottoms, top, args)
        trainer.load_state(payload['trainer_state'])
        for model in (*trainer.bottoms, trainer.top_model):
            model.eval()
        classes = list(range(args.num_classes))
        dataset.authorize_formal_access(
            split='validation', phase='final_validation_pre_install',
            event_idx=payload['event_idx'], task_id=payload['task_id'],
            timeline_step=payload['step'], classes=classes)
        val_base, val_party, val_y, val_aggregate = _scores(
            dataset.get_validation_loader(classes), args, trainer,
            bundle['full_state'])
        _, validation_order = _canonical_validation_evidence(
            dataset.validation_manifest, val_y, classes)
        val_base = val_base.index_select(0, validation_order)
        val_party = [part.index_select(0, validation_order)
                     for part in val_party]
        val_y = val_y.index_select(0, validation_order)
        val_aggregate = val_aggregate.index_select(0, validation_order)
        if (not torch.equal(val_y, bundle['validation_labels'])
                or not torch.allclose(
                    val_aggregate, bundle['validation_embeddings'],
                    atol=1e-5)):
            raise ValueError(
                'reloaded validation embedding differs: '
                f'labels_equal={torch.equal(val_y, bundle["validation_labels"])} '
                f'shape={tuple(val_aggregate.shape)} '
                f'original_shape={tuple(bundle["validation_embeddings"].shape)} '
                f'max_abs={float((val_aggregate - bundle["validation_embeddings"]).abs().max())}'
            )
        hold_base, hold_party, hold_y, _ = _scores(
            dataset.holdout_loader(classes), args, trainer,
            bundle['full_state'])
    tasks = payload['seen_task_classes']
    baseline = score_branch(F.log_softmax(hold_base, dim=1), hold_y, tasks)
    if abs(baseline['cil'] - readout['metrics']['cil']) > 1e-12:
        raise ValueError('router baseline differs from frozen readout')
    fitted = fit_router(val_base, val_party, val_y, tasks)
    routed_val = route(val_base, val_party, tasks, fitted)
    routed_hold = route(hold_base, hold_party, tasks, fitted)
    routed = score_branch(routed_hold, hold_y, tasks)
    mapping = torch.full((args.num_classes,), -1, dtype=torch.long)
    for task, members in tasks.items():
        mapping[list(map(int, members))] = int(task)
    val_tasks = mapping[val_y]
    def task_nll(scores):
        return float(F.nll_loss(torch.stack([
            torch.logsumexp(scores[:, list(map(int, tasks[task]))], 1)
            for task in sorted(tasks)
        ], dim=1), val_tasks))
    return {
        'status': 'training_holdout_four_party_router_control',
        'dataset': config['pilot_dataset'], 'seed': config['seed'],
        'source': str(run), 'checkpoint_sha256': sha256(final),
        'holdout_manifest_sha256': readout['holdout_manifest_sha256'],
        'validation_manifest_sha256': dataset.validation_manifest['sha256'],
        'validation_count': int(val_y.numel()),
        'validation_task_nll_before': task_nll(val_base),
        'validation_task_nll_after': task_nll(routed_val),
        'party_weights': fitted['weights'].tolist(),
        'party_scales': fitted['scales'].tolist(),
        'baseline': baseline, 'routed': routed,
        'holdout_delta_cil': routed['cil'] - baseline['cil'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    result = evaluate_run(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({
        'dataset': result['dataset'], 'seed': result['seed'],
        'baseline_cil': result['baseline']['cil'],
        'routed_cil': result['routed']['cil'],
        'delta_cil': result['holdout_delta_cil'],
        'weights': result['party_weights'],
    }, sort_keys=True))


if __name__ == '__main__':
    main()
