"""Read-only party task evidence on frozen development holdouts."""

import argparse
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from diagnose_sampleonly_head import validate_run_inputs
from run_resource_lite_pilot import (
    PilotDataset, sha256, validate_training_only_artifacts,
)


def _task_scores(logits, task_classes):
    return torch.stack([
        torch.logsumexp(logits[:, list(map(int, task_classes[task]))], 1)
        - math.log(len(task_classes[task]))
        for task in sorted(task_classes)
    ], dim=1)


def summarize_task_evidence(party_logits, fused_logits, labels,
                            task_classes):
    if (not party_logits or fused_logits.ndim != 2
            or any(part.shape != fused_logits.shape for part in party_logits)
            or labels.ndim != 1 or labels.numel() != fused_logits.size(0)):
        raise ValueError('party evidence shape mismatch')
    mapping = torch.full((fused_logits.size(1),), -1, dtype=torch.long)
    for task, members in task_classes.items():
        mapping[list(map(int, members))] = int(task)
    if bool((mapping < 0).any()):
        raise ValueError('task class map is incomplete')
    truth = mapping[labels]
    fused = _task_scores(fused_logits, task_classes).argmax(1)
    party = torch.stack([
        _task_scores(part, task_classes).argmax(1)
        for part in party_logits
    ])
    fused_correct = fused == truth
    party_correct = party == truth.unsqueeze(0)
    errors = ~fused_correct
    return {
        'count': int(labels.numel()),
        'fused_task_accuracy': float(fused_correct.double().mean()),
        'party_task_accuracy': [float(row.double().mean())
                                for row in party_correct],
        'party_agreement_rate': float((party == party[0:1]).all(0)
                                      .double().mean()),
        # This uses truth only as an upper-bound diagnostic, never at inference.
        'fused_error_rescue_fraction': float(
            (party_correct.any(0) & errors).sum() / errors.sum())
        if bool(errors.any()) else 0.0,
    }


@torch.no_grad()
def evaluate_run(run):
    import runner
    from data_utils import split_features
    from models import build_models
    from vfl_trainer import VFLTrainer

    run = Path(run)
    config, readout, _, final = validate_run_inputs(run)
    validate_training_only_artifacts(run)
    if config['aggregation'] not in ('concat', 'sum') or config['cosine_head']:
        raise ValueError('party decomposition requires a linear top head')
    payload = runner._decode_checkpoint_value(torch.load(
        final, map_location='cpu', weights_only=True))
    state = payload['cl_state']['adaptive_audit_bundle']['full_state']
    weights, bias = state['classifier.weight'], state['classifier.bias']
    with tempfile.TemporaryDirectory(prefix='vfcl-party-evidence-') as tmp:
        args = SimpleNamespace(**config)
        args.output_dir = tmp
        args.device = 'cpu'
        dataset = PilotDataset(args)
        if sha256(Path(tmp) / 'pilot_holdout' / 'manifest.json') \
                != readout['holdout_manifest_sha256']:
            raise ValueError('reconstructed holdout manifest differs')
        bottoms, top = build_models(args)
        trainer = VFLTrainer(bottoms, top, args)
        trainer.load_state(payload['trainer_state'])
        for model in (*trainer.bottoms, trainer.top_model):
            model.eval()
        party_rows = [[] for _ in range(args.num_parties)]
        fused_rows, labels = [], []
        for batch_x, batch_y in dataset.holdout_loader(
                list(range(args.num_classes))):
            parts = split_features(batch_x.to('cpu'), args)
            embeddings = [bottom(part) for bottom, part in zip(
                trainer.bottoms, parts)]
            if args.aggregation == 'concat':
                widths = [embedding.size(1) for embedding in embeddings]
                slices = torch.split(weights, widths, dim=1)
            else:
                slices = [weights] * args.num_parties
            rows = [F.linear(embedding, section, None)
                    for embedding, section in zip(embeddings, slices)]
            fused = F.linear(trainer._aggregate(embeddings), weights, bias)
            if not torch.allclose(sum(rows) + bias, fused, atol=1e-5):
                raise ValueError('party logits do not sum to the full head')
            for bucket, row in zip(party_rows, rows):
                bucket.append(row.cpu())
            fused_rows.append(fused.cpu())
            labels.append(batch_y.long().cpu())
    summary = summarize_task_evidence(
        [torch.cat(rows) for rows in party_rows],
        torch.cat(fused_rows), torch.cat(labels),
        payload['seen_task_classes'])
    return {
        'status': 'training_holdout_party_task_diagnostic',
        'dataset': config['pilot_dataset'], 'seed': config['seed'],
        'source': str(run), 'checkpoint_sha256': sha256(final),
        'holdout_manifest_sha256': readout['holdout_manifest_sha256'],
        'summary': summary,
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
    print(json.dumps({'dataset': result['dataset'], 'seed': result['seed'],
                      **result['summary']}, sort_keys=True))


if __name__ == '__main__':
    main()
