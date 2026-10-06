#!/usr/bin/env python3
"""Read-only replay and offline output-calibration ablation for P1 checkpoints."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from analyze_output_bias import (
    _load_checkpoint,
    configure_evaluation_determinism,
    final_checkpoint_path,
    validate_run_contract,
)
from bic_calibration import (
    TaskAffineCalibrator,
    fit_final_calibrator,
    summarize_paired_logits,
)
from data_utils import VFLDataset
from models import build_models
from vfl_trainer import VFLTrainer


CACHE_KEYS = (
    'calibration_logits',
    'calibration_labels',
    'calibration_indices',
    'test_logits',
    'test_labels',
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def nested_budget_indices(manifest, budget):
    budget = int(budget)
    per_class = int(manifest['per_class'])
    if budget <= 0 or budget > per_class:
        raise ValueError('budget must be between 1 and manifest per_class')
    rows = []
    for class_id in sorted(manifest['by_class'], key=int):
        values = [int(value) for value in manifest['by_class'][class_id]]
        if len(values) != per_class:
            raise ValueError('manifest class count mismatch')
        rows.extend(values[:budget])
    if len(rows) != len(set(rows)):
        raise ValueError('manifest indices must be unique')
    return rows


def validate_raw_replay(saved_predictions, replayed_predictions,
                        saved_accuracy, replayed_accuracy):
    saved_predictions = np.asarray(saved_predictions)
    replayed_predictions = np.asarray(replayed_predictions)
    if not np.array_equal(saved_predictions, replayed_predictions):
        raise ValueError('raw replay mismatch: predictions differ')
    if not math.isclose(float(saved_accuracy), float(replayed_accuracy),
                        rel_tol=0.0, abs_tol=1e-12):
        raise ValueError('raw replay mismatch: accuracy differs')


def validate_cache_schema(cache):
    if set(cache) != set(CACHE_KEYS):
        raise ValueError('cache schema contains missing or forbidden arrays')
    calibration_count = len(cache['calibration_labels'])
    test_count = len(cache['test_labels'])
    if cache['calibration_logits'].shape[0] != calibration_count:
        raise ValueError('cache schema has inconsistent calibration rows')
    if len(cache['calibration_indices']) != calibration_count:
        raise ValueError('cache schema has inconsistent calibration indices')
    if cache['test_logits'].shape[0] != test_count:
        raise ValueError('cache schema has inconsistent test rows')


def _nll(logits, labels, task_classes):
    seen = [value for task_id in sorted(task_classes)
            for value in task_classes[task_id]]
    positions = {class_id: position for position, class_id in enumerate(seen)}
    targets = torch.tensor([positions[int(label)] for label in labels])
    return float(F.cross_entropy(logits[:, seen], targets))


def _evaluation_record(method, budget, test_logits, test_labels, task_classes,
                       calibrator, audit):
    paired = summarize_paired_logits(
        test_logits, test_labels, task_classes, calibrator
    )
    paired['raw']['nll'] = _nll(test_logits, test_labels, task_classes)
    calibrated_logits = calibrator.apply(test_logits)
    paired['calibrated']['nll'] = _nll(
        calibrated_logits, test_labels, task_classes
    )
    task_il_delta = max(
        abs(paired['calibrated']['task_il'][key] - paired['raw']['task_il'][key])
        for key in paired['raw']['task_il']
    )
    params = {
        str(task_id): calibrator.parameters_for(task_id)
        for task_id in sorted(calibrator.tasks)
    }
    return {
        'method': method,
        'budget': int(budget),
        'raw': paired['raw'],
        'calibrated': paired['calibrated'],
        'parameters': params,
        'task_il_max_abs_delta': float(task_il_delta),
        'audit': dict(audit, passed=(
            bool(audit.get('passed', True)) and task_il_delta <= 1e-6
        )),
    }


def evaluate_cache(cache, task_classes, manifest, sequential_state=None,
                   budgets=(5, 10, 25), lr=0.05, steps=1000):
    validate_cache_schema(cache)
    task_classes = {
        int(key): [int(value) for value in values]
        for key, values in task_classes.items()
    }
    calibration_logits = torch.as_tensor(cache['calibration_logits']).float()
    calibration_labels = torch.as_tensor(cache['calibration_labels']).long()
    test_logits = torch.as_tensor(cache['test_logits']).float()
    test_labels = torch.as_tensor(cache['test_labels']).long()
    row_for_index = {
        int(index): row for row, index in enumerate(cache['calibration_indices'])
    }
    records = [_evaluation_record(
        'identity', 0, test_logits, test_labels, task_classes,
        TaskAffineCalibrator(), {'passed': True, 'fit_source': 'none'},
    )]
    for budget in budgets:
        selected = nested_budget_indices(manifest, budget)
        try:
            rows = [row_for_index[index] for index in selected]
        except KeyError as error:
            raise ValueError(f'budget index missing from cache: {error}') from error
        for mode in ('beta_only', 'alpha_only', 'joint_alpha_beta'):
            calibrator, fit = fit_final_calibrator(
                calibration_logits[rows], calibration_labels[rows],
                task_classes, mode, lr=lr, steps=steps,
            )
            records.append(_evaluation_record(
                mode, budget, test_logits, test_labels, task_classes, calibrator,
                {'passed': fit['loss_after'] < fit['loss_before'],
                 'fit_source': 'heldout_calibration_logits', 'fit': fit},
            ))
    if sequential_state is not None:
        calibrator = TaskAffineCalibrator()
        calibrator.load_state_dict(sequential_state)
        records.append(_evaluation_record(
            'sequential_alpha_beta', int(manifest['per_class']),
            test_logits, test_labels, task_classes, calibrator,
            {'passed': True, 'fit_source': 'saved_p1_calibrator'},
        ))
    return records


def _read_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def _args_for_replay(config, data_path, output_dir, device):
    values = dict(config)
    values.update(data_path=data_path or config['data_path'],
                  output_dir=str(output_dir), device=device, num_workers=0)
    return argparse.Namespace(**values)


def replay_run(run_dir, data_path, output_dir, device='cuda:0'):
    """Replay one P1 checkpoint and persist only aggregate output arrays."""
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _read_json(run_dir / 'config.json')
    validate_run_contract(config)
    configure_evaluation_determinism(int(config['seed']))
    args = _args_for_replay(config, data_path, output_dir, device)
    dataset = VFLDataset(args)

    saved_manifest = _read_json(run_dir / 'bic' / 'calibration_manifest.json')
    if dataset.calibration_manifest != saved_manifest:
        raise ValueError('calibration manifest replay mismatch')
    if dataset.calibration_audit()['passed'] is not True:
        raise ValueError('calibration data audit failed')

    checkpoint_path = final_checkpoint_path(str(run_dir))
    checkpoint = _load_checkpoint(checkpoint_path)
    bottoms, top = build_models(args)
    trainer = VFLTrainer(bottoms, top, args)
    trainer.load_state(checkpoint['trainer_state'])
    classes = list(range(int(config['num_classes'])))
    calibration_loader = dataset.get_calibration_loader(classes)
    _, test_loader = dataset.get_task_loaders(classes, shuffle_train=False)
    calibration_logits, calibration_labels = trainer.collect_logits(calibration_loader)
    test_logits, test_labels = trainer.collect_logits(test_loader)
    replay_logits, replay_labels = trainer.collect_logits(test_loader)
    if not torch.equal(test_labels, replay_labels) or not torch.equal(test_logits, replay_logits):
        raise ValueError('raw replay mismatch: fresh deterministic replay differs')

    saved = np.load(run_dir / 'final_probs.npz')
    if not np.array_equal(saved['labels'], test_labels.numpy()):
        raise ValueError('raw replay mismatch: label order differs')
    saved_predictions = saved['probs'].argmax(axis=1)
    replayed_predictions = test_logits.argmax(1).numpy()
    results = _read_json(run_dir / 'results.json')
    saved_accuracy = float(results['cl_metrics']['AA_final'])
    replayed_accuracy = float(np.mean(replayed_predictions == test_labels.numpy()))
    validate_raw_replay(
        saved_predictions, replayed_predictions, saved_accuracy, replayed_accuracy
    )

    calibration_indices = np.array(
        sorted(int(value) for value in dataset.calibration_indices), dtype=np.int64
    )
    arrays = {
        'calibration_logits': calibration_logits.numpy(),
        'calibration_labels': calibration_labels.numpy(),
        'calibration_indices': calibration_indices,
        'test_logits': test_logits.numpy(),
        'test_labels': test_labels.numpy(),
    }
    validate_cache_schema(arrays)
    cache_path = output_dir / 'logits.npz'
    np.savez_compressed(cache_path, **arrays)
    record = {
        'run_dir': str(run_dir),
        'variant': config['expected_party_kd_variant'],
        'seed': int(config['seed']),
        'checkpoint_path': checkpoint_path,
        'checkpoint_sha256': _sha256(checkpoint_path),
        'cache_path': str(cache_path),
        'cache_sha256': _sha256(cache_path),
        'manifest_sha256': saved_manifest['sha256'],
        'raw_aa_final': replayed_accuracy,
        'raw_prediction_replay_exact': True,
        'fresh_replay_logits_exact': True,
        'calibration_count': int(len(calibration_labels)),
        'test_count': int(len(test_labels)),
        'cache_keys': list(CACHE_KEYS),
        'privacy_audit': {
            'passed': True,
            'test_used_for_fit': False,
            'raw_images_saved': False,
            'party_embeddings_saved': False,
        },
    }
    with open(output_dir / 'replay.json', 'w', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    return record


def evaluate_saved_cache(cache_path, run_dir, output_path, lr=0.05, steps=1000):
    run_dir = Path(run_dir)
    with np.load(cache_path) as loaded:
        cache = {key: loaded[key] for key in loaded.files}
    manifest = _read_json(run_dir / 'bic' / 'calibration_manifest.json')
    state = _load_checkpoint(run_dir / 'bic' / 'calibrator.pt')
    config = _read_json(run_dir / 'config.json')
    classes_per_task = int(config['classes_per_task'])
    task_classes = {
        task_id: list(range(task_id * classes_per_task,
                            (task_id + 1) * classes_per_task))
        for task_id in range(int(config['num_tasks']))
    }
    records = evaluate_cache(
        cache, task_classes, manifest, sequential_state=state, lr=lr, steps=steps
    )
    run_key = f"{config['expected_party_kd_variant']}_seed{int(config['seed'])}"
    for record in records:
        record['run_key'] = run_key
        record['variant'] = config['expected_party_kd_variant']
        record['seed'] = int(config['seed'])
        record['privacy_audit'] = {
            'passed': True, 'test_used_for_fit': False,
            'raw_images_saved': False, 'party_embeddings_saved': False,
        }
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(records, handle, indent=2, sort_keys=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--data-path')
    parser.add_argument('--output-dir')
    parser.add_argument('--cache')
    parser.add_argument('--output-json')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.cache:
        if not args.output_json:
            parser.error('--output-json is required with --cache')
        records = evaluate_saved_cache(
            args.cache, args.run_dir, args.output_json, steps=args.steps
        )
        print(json.dumps({'records': len(records), 'output': args.output_json}))
        return
    if not args.data_path or not args.output_dir:
        parser.error('--data-path and --output-dir are required for replay')
    print(json.dumps(replay_run(
        args.run_dir, args.data_path, args.output_dir, args.device
    ), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
