"""Rebuild task-boundary 20/class exemplars without touching formal tests."""

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

from diagnose_sampleonly_head import validate_run_inputs
from party_balanced_herding_dev import party_balanced_features
from run_resource_lite_pilot import (
    PilotDataset, sha256, validate_training_only_artifacts,
)


def _row_counts(rows):
    return Counter(hashlib.sha256(row.contiguous().numpy().tobytes()).digest()
                   for row in rows)


@torch.no_grad()
def reconstruct_run(run):
    import runner
    from cl_methods.proto_evolve import herding_indices
    from data_utils import split_features
    from models import build_models
    from vfl_trainer import VFLTrainer

    run = Path(run)
    config, readout, _, final_path = validate_run_inputs(run)
    validate_training_only_artifacts(run)
    final = runner._decode_checkpoint_value(torch.load(
        final_path, map_location='cpu', weights_only=True))
    reference = final['cl_state']['head_raw_replay']
    baseline_raw, balanced_raw, overlaps = {}, {}, {}
    event_hashes = {}
    with tempfile.TemporaryDirectory(prefix='vfcl-exemplar-rebuild-') as tmp:
        args = SimpleNamespace(**config)
        args.output_dir = tmp
        args.device = 'cpu'
        dataset = PilotDataset(args)
        if sha256(Path(tmp) / 'pilot_holdout' / 'manifest.json') \
                != readout['holdout_manifest_sha256']:
            raise ValueError('reconstructed holdout manifest differs')
        for task in range(args.num_tasks):
            event_path = run / 'checkpoints' / f'event_{task}_CIL.pt'
            event_hashes[str(task)] = sha256(event_path)
            event = runner._decode_checkpoint_value(torch.load(
                event_path, map_location='cpu', weights_only=True))
            if event['task_id'] != task:
                raise ValueError('historical checkpoint task ID differs')
            bottoms, top = build_models(args)
            trainer = VFLTrainer(bottoms, top, args)
            trainer.load_state(event['trainer_state'])
            for model in trainer.bottoms:
                model.eval()
            raw_by, aggregate_by, party_by = {}, {}, {}
            for batch_x, batch_y in dataset.get_train_loader(
                    list(event['new_classes']), shuffle=False):
                parts = split_features(batch_x.to('cpu'), args)
                embeddings = [bottom(part) for bottom, part in zip(
                    trainer.bottoms, parts)]
                aggregate = trainer._aggregate(embeddings)
                for row, label in enumerate(batch_y.tolist()):
                    class_id = int(label)
                    raw_by.setdefault(class_id, []).append(
                        batch_x[row].cpu().clone())
                    aggregate_by.setdefault(class_id, []).append(
                        aggregate[row].cpu().clone())
                    if class_id not in party_by:
                        party_by[class_id] = [list() for _ in embeddings]
                    for bucket, embedding in zip(party_by[class_id],
                                                 embeddings):
                        bucket.append(embedding[row].cpu().clone())
            if set(raw_by) != set(map(int, event['new_classes'])):
                raise ValueError('training task class set differs')
            for class_id in sorted(raw_by):
                raw = torch.stack(raw_by[class_id])
                aggregate = torch.stack(aggregate_by[class_id])
                parties = [torch.stack(bucket) for bucket
                           in party_by[class_id]]
                baseline = herding_indices(aggregate, 20)
                balanced = herding_indices(
                    party_balanced_features(parties), 20)
                baseline_raw[class_id] = raw.index_select(0, baseline)
                balanced_raw[class_id] = raw.index_select(0, balanced)
                if class_id not in reference or reference[class_id].shape \
                        != baseline_raw[class_id].shape:
                    raise ValueError('frozen reference exemplars differ in shape')
                common = sum((_row_counts(baseline_raw[class_id])
                              & _row_counts(reference[class_id])).values())
                overlaps[str(class_id)] = common
    return {
        'baseline_raw': baseline_raw, 'balanced_raw': balanced_raw,
        'summary': {
            'status': 'training_only_historical_exemplar_reconstruction',
            'dataset': config['pilot_dataset'], 'seed': config['seed'],
            'source': str(run),
            'final_checkpoint_sha256': sha256(final_path),
            'holdout_manifest_sha256': readout['holdout_manifest_sha256'],
            'event_checkpoint_sha256': event_hashes,
            'class_count': len(overlaps),
            'baseline_overlap_per_class': overlaps,
            'baseline_exact_reproduction': all(value == 20 for value
                                               in overlaps.values()),
            'balanced_overlap_per_class': {
                str(class_id): sum((
                    _row_counts(baseline_raw[class_id])
                    & _row_counts(balanced_raw[class_id])).values())
                for class_id in sorted(baseline_raw)
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    summary = reconstruct_run(args.run)['summary']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True)
                           + '\n')
    print(json.dumps({
        'dataset': summary['dataset'], 'seed': summary['seed'],
        'exact': summary['baseline_exact_reproduction'],
        'class_count': summary['class_count'],
        'min_overlap': min(summary['baseline_overlap_per_class'].values()),
    }, sort_keys=True))


if __name__ == '__main__':
    main()
