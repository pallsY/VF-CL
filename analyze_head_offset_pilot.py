"""Fit one frozen-head new-task offset from training images, then evaluate it."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize_scalar


def select_calibration_indices(labels, heldout, per_class=20):
    """Take fixed first eligible training indices for every observed class."""
    if per_class <= 0:
        raise ValueError('per-class calibration count must be positive')
    heldout = {int(index) for index in heldout}
    if any(index < 0 or index >= len(labels) for index in heldout):
        raise ValueError('held-out index is outside the training corpus')
    selected = []
    for class_id in sorted(set(int(label) for label in labels)):
        candidates = [index for index, label in enumerate(labels)
                      if int(label) == class_id and index not in heldout]
        if len(candidates) < per_class:
            raise ValueError(f'class {class_id} lacks calibration rows')
        selected.extend(candidates[:per_class])
    if set(selected) & heldout:
        raise ValueError('calibration examples overlap validation')
    return selected


def fit_scalar_offset(logits, labels, newest_classes):
    """Minimize training-only CE for one shared new-task logit offset."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    newest = sorted(set(int(value) for value in newest_classes))
    if (logits.ndim != 2 or logits.shape[0] == 0
            or labels.shape != (logits.shape[0],)
            or not np.isfinite(logits).all()
            or np.any(labels < 0) or np.any(labels >= logits.shape[1])
            or not newest or newest[0] < 0 or newest[-1] >= logits.shape[1]
            or not np.isin(labels, newest).any()
            or np.isin(labels, newest).all()):
        raise ValueError('calibration logits/labels are malformed')
    mask = np.zeros(logits.shape[1], dtype=np.float64)
    mask[newest] = 1.0
    rows = np.arange(len(labels))

    def nll(delta):
        shifted = logits + float(delta) * mask
        return float(np.mean(np.logaddexp.reduce(shifted, axis=1)
                             - shifted[rows, labels]))

    def derivative(delta):
        shifted = logits + float(delta) * mask
        log_z = np.logaddexp.reduce(shifted, axis=1)
        new_probability = np.exp(shifted[:, newest] - log_z[:, None]).sum(axis=1)
        return float(np.mean(new_probability - np.isin(labels, newest)))

    if not derivative(-10.0) < 0 < derivative(10.0):
        raise ValueError('scalar optimum is outside the prespecified bounds')

    result = minimize_scalar(nll, bounds=(-10.0, 10.0), method='bounded',
                             options={'xatol': 1e-6})
    if (not result.success or not np.isfinite(result.x)
            or not np.isfinite(result.fun) or not -10.0 < result.x < 10.0):
        raise ValueError('bounded scalar fit did not find an interior optimum')
    return float(result.x), nll(result.x)


def pilot_passes(baseline, corrected):
    """Apply the prespecified one-seed development gate."""
    if (baseline['examples'] != corrected['examples']
            or baseline['old_examples'] != corrected['old_examples']
            or baseline['new_examples'] != corrected['new_examples']):
        raise ValueError('pilot groups changed between baseline and correction')
    return bool(
        100 * (corrected['correct'] - baseline['correct'])
        >= baseline['examples']
        and 20 * (corrected['new_correct'] - baseline['new_correct'])
        >= 3 * baseline['new_examples']
        and 100 * (baseline['old_correct'] - corrected['old_correct'])
        <= baseline['old_examples']
        and corrected['taskil_correct'] == baseline['taskil_correct']
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def summarize_logits(logits, labels, tasks, newest):
    """Return aggregate class and task metrics without saving predictions."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if (logits.shape != (len(labels), 100) or not np.isfinite(logits).all()
            or len(labels) == 0 or np.any(labels < 0) or np.any(labels >= 100)):
        raise ValueError('evaluation logits or labels are malformed')
    newest = set(newest)
    old_mask = ~np.isin(labels, sorted(newest))
    new_mask = ~old_mask
    if not old_mask.any() or not new_mask.any():
        raise ValueError('both old and new classes must be present')
    predictions = logits.argmax(axis=1)
    correct = predictions == labels
    taskil_correct = np.zeros(len(labels), dtype=bool)
    task_accuracy = []
    for classes in tasks:
        classes = np.asarray(classes, dtype=np.int64)
        selected = np.isin(labels, classes)
        if not selected.any():
            raise ValueError('evaluation omits a task')
        local_pred = classes[logits[selected][:, classes].argmax(axis=1)]
        taskil_correct[selected] = local_pred == labels[selected]
        task_accuracy.append(float(correct[selected].mean()))
    rows = np.arange(len(labels))
    nll = np.mean(np.logaddexp.reduce(logits, axis=1) - logits[rows, labels])
    return {
        'examples': int(len(labels)),
        'old_examples': int(old_mask.sum()),
        'new_examples': int(new_mask.sum()),
        'correct': int(correct.sum()),
        'old_correct': int(correct[old_mask].sum()),
        'new_correct': int(correct[new_mask].sum()),
        'taskil_correct': int(taskil_correct.sum()),
        'accuracy': float(correct.mean()),
        'old_accuracy': float(correct[old_mask].mean()),
        'new_accuracy': float(correct[new_mask].mean()),
        'old_to_new_rate': float(np.isin(predictions[old_mask], sorted(newest)).mean()),
        'new_to_old_rate': float((~np.isin(predictions[new_mask], sorted(newest))).mean()),
        'taskil_accuracy': float(taskil_correct.mean()),
        'per_task_accuracy': task_accuracy,
        'nll': float(nll),
    }


def analyze(run_dir, output_dir, device):
    import torch
    from torch.utils.data import DataLoader, Subset
    from adaptive_consolidation_audit import _safe_torch_load, trainer_state_sha256
    from data_utils import VFLDataset
    from models import build_models
    from vfl_trainer import VFLTrainer

    run_dir = Path(run_dir).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() or output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError('analysis output must be a new root outside training')
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
            != '7bfe6b1d724fb1206bc0053a9008126bad86332d'
            or subprocess.check_output(['git', 'status', '--porcelain'], text=True)):
        raise ValueError('analysis requires the exact formal source checkout')
    complete = json.loads((run_dir / 'PILOT_TRAINING_COMPLETE.json').read_text())
    protocol = json.loads((run_dir / 'PILOT_PROTOCOL.json').read_text())
    config_path = run_dir / 'config.json'
    checkpoint_path = run_dir / 'checkpoints' / 'event_9_CIL.pt'
    results_path = run_dir / 'results.json'
    for path, field in ((config_path, 'config_sha256'),
                        (checkpoint_path, 'checkpoint_sha256'),
                        (results_path, 'results_sha256')):
        if file_sha256(path) != complete[field]:
            raise ValueError('completed pilot artifact hash mismatch')
    config = json.loads(config_path.read_text())
    if (config.get('seed') != 45
            or config.get('lambda_validation_split_seed') != 20261007
            or config.get('lambda_validation_enabled') != 1
            or config.get('head_consolidation_enabled') != 0
            or config.get('formal_deferred_evaluation') is not False
            or protocol.get('source_commit')
            != '7bfe6b1d724fb1206bc0053a9008126bad86332d'):
        raise ValueError('pilot config/protocol differs from prespecification')
    expected_data = {
        'data:cifar-100-python/train',
        'data:cifar-100-python/test',
        'data:cifar-100-python/meta',
    }
    source_data = protocol.get('source_data_sha256')
    if (type(source_data) is not dict or set(source_data) != expected_data
            or any(file_sha256(Path(config['data_path'])
                               / key.split(':', 1)[1]) != digest
                   for key, digest in source_data.items())):
        raise ValueError('CIFAR payload differs from pilot launch')
    checkpoint = _safe_torch_load(checkpoint_path)
    if checkpoint.get('task_id') != 9:
        raise ValueError('final task checkpoint is missing')
    tasks = [list(map(int, checkpoint['seen_task_classes'][task]))
             for task in range(10)]
    if tasks != [list(range(task * 10, (task + 1) * 10)) for task in range(10)]:
        raise ValueError('checkpoint task plan is malformed')
    newest = tasks[-1]

    with tempfile.TemporaryDirectory(prefix='vfcl-head-offset-analysis-') as scratch:
        args = SimpleNamespace(**config)
        args.output_dir = scratch
        args.device = device
        args.data_flow_audit = 0
        args.num_workers = 0
        dataset = VFLDataset(args)
        manifest_path = run_dir / 'validation' / 'validation_manifest.json'
        if json.loads(manifest_path.read_text()) != dataset.validation_manifest:
            raise ValueError('validation selection differs from training run')
        bic_manifest_path = run_dir / 'bic' / 'calibration_manifest.json'
        if json.loads(bic_manifest_path.read_text()) != dataset.calibration_manifest:
            raise ValueError('BiC calibration selection differs from training run')
        heldout = dataset.validation_indices | dataset.calibration_indices
        selected = select_calibration_indices(
            dataset.validationset.targets, heldout, 20,
        )
        if len(selected) != 2000 or set(selected) & heldout:
            raise ValueError('calibration set is not 20 train-only examples per class')
        bottoms, top = build_models(args)
        trainer = VFLTrainer(bottoms, top, args)
        trainer.load_state(checkpoint['trainer_state'])
        if bool(top._adaptive_enabled) or bool(top._logit_calibration_enabled):
            raise ValueError('pilot head was previously consolidated/calibrated')
        for model in (*trainer.bottoms, trainer.top_model):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        initial_state = trainer_state_sha256(trainer.get_state())
        calibration_loader = DataLoader(
            Subset(dataset.validationset, selected), batch_size=64,
            shuffle=False, num_workers=0,
        )
        train_logits, train_labels = trainer.collect_logits(calibration_loader)
        validation_logits, validation_labels = trainer.collect_logits(
            dataset.get_validation_loader(list(range(100)))
        )
        if trainer_state_sha256(trainer.get_state()) != initial_state:
            raise RuntimeError('frozen model state changed during analysis')

    train_logits = train_logits.numpy()
    train_labels = train_labels.numpy()
    validation_logits = validation_logits.numpy()
    validation_labels = validation_labels.numpy()
    if (not all(np.count_nonzero(train_labels == c) == 20 for c in range(100))
            or not all(np.count_nonzero(validation_labels == c) == 25
                       for c in range(100))):
        raise ValueError('calibration/validation class coverage is malformed')
    delta, train_nll = fit_scalar_offset(train_logits, train_labels, newest)
    mask = np.zeros(100, dtype=np.float64)
    mask[newest] = 1.0
    shifted_validation = validation_logits + delta * mask
    for classes in tasks:
        if not np.array_equal(
                validation_logits[:, classes].argmax(axis=1),
                shifted_validation[:, classes].argmax(axis=1)):
            raise ValueError('task-il predictions changed under common task offset')
    baseline = summarize_logits(validation_logits, validation_labels,
                                tasks, newest)
    corrected = summarize_logits(shifted_validation, validation_labels,
                                 tasks, newest)
    if (baseline['taskil_accuracy'] != corrected['taskil_accuracy']
            or baseline['examples'] != 2500):
        raise ValueError('task-il changed or validation cohort is incomplete')
    runner_results = json.loads(results_path.read_text())
    runner_aa = runner_results['cl_metrics']['AA_final']
    if abs(baseline['accuracy'] - runner_aa) > 0.001:
        raise ValueError('baseline accuracy differs from training result')
    bic_pair = runner_results['bic_final']['paired']
    if abs(bic_pair['raw']['overall_accuracy'] - baseline['accuracy']) > 0.001:
        raise ValueError('baseline accuracy differs from saved BiC readout')
    bic_calibrated = bic_pair['calibrated']
    bic_context = {
        'accuracy': float(bic_calibrated['overall_accuracy']),
        'old_accuracy': float(np.mean([
            bic_calibrated['per_task_accuracy'][f'task_{task}']
            for task in range(9)
        ])),
        'new_accuracy': float(bic_calibrated['per_task_accuracy']['task_9']),
    }
    result = {
        'schema_version': 1,
        'source_run': str(run_dir),
        'source_commit': protocol['source_commit'],
        'analyzer_sha256': file_sha256(__file__),
        'training_config_sha256': file_sha256(config_path),
        'training_checkpoint_sha256': file_sha256(checkpoint_path),
        'source_data_sha256': source_data,
        'validation_manifest_sha256': file_sha256(manifest_path),
        'bic_manifest_sha256': file_sha256(bic_manifest_path),
        'calibration_indices_sha256': hashlib.sha256(
            json.dumps(selected, separators=(',', ':')).encode()
        ).hexdigest(),
        'calibration_count': len(selected),
        'validation_count': len(validation_labels),
        'model_state_sha256': initial_state,
        'delta': delta,
        'calibration_nll_after_fit': train_nll,
        'baseline': baseline,
        'corrected': corrected,
        'existing_bic_context': bic_context,
        'passed': pilot_passes(baseline, corrected),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    temp = output_dir / 'head_offset.json.tmp'
    with open(temp, 'x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, output_dir / 'head_offset.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    cli = parser.parse_args()
    result = analyze(cli.run_dir, cli.output_dir, cli.device)
    print(json.dumps({key: result[key] for key in
                      ('delta', 'baseline', 'corrected', 'passed')},
                     indent=2, sort_keys=True))
