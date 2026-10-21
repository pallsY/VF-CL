"""Read-only execution diagnostic for the locked train-replay A/B results.

All derived files go to a new diagnostic root. No formal test loader is used.
Intermediate scores use the raw checkpoint head and are not final-head scores.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from run_resource_lite_pilot import PilotDataset, validate_source_data_hash


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def score(args, dataset, checkpoint, seen):
    import runner
    from data_utils import split_features
    from models import build_models
    from vfl_trainer import VFLTrainer

    payload = runner._decode_checkpoint_value(torch.load(
        checkpoint, map_location='cpu', weights_only=True))
    score_args = SimpleNamespace(**vars(args))
    score_args.device = 'cpu'
    bottoms, top = build_models(score_args)
    trainer = VFLTrainer(bottoms, top, score_args)
    trainer.load_state(payload['trainer_state'])
    for model in (*trainer.bottoms, trainer.top_model):
        model.eval()
    task_classes = {
        int(task): [int(label) for label in labels]
        for task, labels in payload['seen_task_classes'].items()
    }
    classes = sorted(label for labels in task_classes.values() for label in labels)
    if classes != list(range(len(classes))) or classes != list(seen):
        raise ValueError('checkpoint task classes differ from expected stream')
    true, predicted, taskil = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in dataset.holdout_loader(classes):
            parts = split_features(batch_x.cpu(), score_args)
            embedded = [bottom(part) for bottom, part in zip(trainer.bottoms, parts)]
            logits = trainer.top_model(trainer._aggregate(embedded))
            pred = logits[:, classes].argmax(dim=1)
            true.extend(int(value) for value in batch_y.tolist())
            predicted.extend(int(value) for value in pred.tolist())
            for row, label in enumerate(batch_y.tolist()):
                matching = [members for members in task_classes.values()
                            if int(label) in members]
                if len(matching) != 1:
                    raise ValueError('task mapping is not unique')
                members = matching[0]
                local = members[logits[row, members].argmax().item()]
                taskil.append(int(local == int(label)))
    per_class = {}
    for label in classes:
        indices = [i for i, value in enumerate(true) if value == label]
        per_class[str(label)] = {
            'count': len(indices),
            'errors': sum(predicted[i] != label for i in indices),
        }
        if len(indices) != args.pilot_holdout_per_class:
            raise ValueError('held-out class count differs')
    return {
        'count': len(true),
        'cil': sum(a == b for a, b in zip(true, predicted)) / len(true),
        'task_il': sum(taskil) / len(taskil),
        'per_class': per_class,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--diagnostic-root', required=True, type=Path)
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    output = args.diagnostic_root.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('diagnostic root must be separate from source tree')
    if output.exists():
        raise FileExistsError(output)
    if not (source / 'QUEUE_COMPLETE.json').is_file() or (source / 'QUEUE_FAILURE.json').exists():
        raise ValueError('source queue is incomplete or failed')
    jobs = [f'isolet_seed{seed}_{arm}' for seed in (60, 61, 62) for arm in 'AB']
    tracked = []
    for job in jobs:
        directory = source / job
        if (directory / 'results.json').exists() or not (directory / 'FORMAL_EVALUATION_PENDING.json').is_file():
            raise ValueError('source reached official test or lacks test gate')
        tracked += [directory / 'config.json', directory / 'pilot_holdout/manifest.json',
                    directory / 'checkpoints/formal_final.pt']
        tracked += [directory / f'formal_snapshots/event_{event}_CIL.pt'
                    for event in range(13)]
    before = {str(path): digest(path) for path in tracked}
    output.mkdir(parents=True)
    results = {'source_root': str(source), 'source_hashes_before': before,
               'note': 'intermediate raw-head scores are distinct from final-head readout',
               'jobs': {}}
    try:
        for job in jobs:
            directory = source / job
            config = json.loads((directory / 'config.json').read_text())
            if (config['pilot_dataset'] != 'isolet' or config['pilot_smoke']
                    or config['formal_deferred_evaluation'] is not True):
                raise ValueError('source job identity differs')
            validate_source_data_hash(config)
            config['output_dir'] = str(output / job)
            config['results_dir'] = str(output)
            config['data_flow_audit'] = 0
            config['num_workers'] = 0
            work = output / job
            work.mkdir()
            instance = SimpleNamespace(**config)
            dataset = PilotDataset(instance)
            if digest(work / 'pilot_holdout/manifest.json') != before[str(directory / 'pilot_holdout/manifest.json')]:
                raise ValueError('reconstructed holdout manifest differs')
            curve = []
            for event in range(13):
                checkpoint = directory / f'formal_snapshots/event_{event}_CIL.pt'
                value = score(instance, dataset, checkpoint, range(2 * (event + 1)))
                curve.append({'event': event, 'count': value['count'],
                              'cil': value['cil'], 'task_il': value['task_il']})
            final = score(instance, dataset, directory / 'checkpoints/formal_final.pt', range(26))
            existing = json.loads((directory / 'HOLDOUT_READOUT.json').read_text())['metrics']
            if (abs(final['cil'] - existing['cil']) > 1e-12
                    or abs(final['task_il'] - existing['task_il']) > 1e-12):
                raise ValueError('recomputed final metrics differ from frozen readout')
            results['jobs'][job] = {'intermediate_raw_head': curve,
                                    'final_head': final}
        after = {str(path): digest(path) for path in tracked}
        if before != after:
            raise ValueError('source artifact hashes changed during diagnostic')
        results['source_hashes_after'] = after
        results['status'] = 'passed'
        (output / 'DIAGNOSTIC.json').write_text(
            json.dumps(results, indent=2, sort_keys=True) + '\n')
        print('diagnostic_passed', output / 'DIAGNOSTIC.json')
    except Exception as error:
        after = {str(path): digest(path) for path in tracked}
        (output / 'DIAGNOSTIC_FAILURE.json').write_text(json.dumps({
            'error': repr(error), 'source_hashes_unchanged': before == after,
        }, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
