"""Read-only Full/Bias/Mixed error attribution on saved CIFAR validation data."""

import argparse
import hashlib
import json
import os
import statistics
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

import adaptive_head_consolidation
import data_utils
import determinism
import models
import vfl_trainer
from adaptive_head_consolidation import mix_log_probabilities
from data_utils import VFLDataset, split_features
from models import build_models
from vfl_trainer import VFLTrainer


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def new_counts(task_count):
    return dict(total=0, correct=0, old_total=0, old_correct=0,
                new_total=0, new_correct=0, old_to_new=0,
                new_to_old=0, nll_sum=0.0,
                task_correct=[0] * task_count,
                task_total=[0] * task_count,
                taskil_correct=[0] * task_count,
                task_confusion=[[0] * task_count for _ in range(task_count)])


def finish(counts):
    total = counts['total']
    old_total = counts['old_total']
    new_total = counts['new_total']
    old_errors = old_total - counts['old_correct']
    new_errors = new_total - counts['new_correct']
    if total == 0 or old_total == 0 or new_total == 0:
        raise ValueError('validation cohort is incomplete')
    return dict(
        examples=total, old_examples=old_total, new_examples=new_total,
        accuracy=counts['correct'] / total,
        old_accuracy=counts['old_correct'] / old_total,
        new_accuracy=counts['new_correct'] / new_total,
        taskil_accuracy=sum(counts['taskil_correct']) / total,
        old_to_new_rate=counts['old_to_new'] / old_total,
        new_to_old_rate=counts['new_to_old'] / new_total,
        old_error_to_new_fraction=(counts['old_to_new'] / old_errors
                                   if old_errors else 0.0),
        new_error_to_old_fraction=(counts['new_to_old'] / new_errors
                                   if new_errors else 0.0),
        nll=counts['nll_sum'] / total,
        per_task_accuracy=[right / count for right, count in zip(
            counts['task_correct'], counts['task_total'])],
        per_task_taskil=[right / count for right, count in zip(
            counts['taskil_correct'], counts['task_total'])],
        task_confusion=counts['task_confusion'],
    )


@torch.no_grad()
def evaluate_seed(root, seed, device):
    key = f'cifar100%3Aadaptive%3A{seed}'
    run = root / 'runs' / key
    record_path = root / 'records' / f'{key}.json'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    if (record.get('kind') != 'formal_completed_run'
            or record.get('method') != 'adaptive'
            or record.get('seed') != seed):
        raise ValueError('formal completed record identity mismatch')
    config_path = run / 'config.json'
    checkpoint_path = run / 'checkpoints' / 'formal_final.pt'
    source_manifest_path = run / 'validation' / 'validation_manifest.json'
    if (sha256(config_path) != record['artifact_sha256']['config']
            or sha256(checkpoint_path) != record['artifact_sha256']['checkpoint']
            or sha256(source_manifest_path)
            != record['artifact_sha256']['validation_manifest']):
        raise ValueError('formal source artifact hash mismatch')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if (config['data'] != 'cifar100' or config['num_tasks'] != 10
            or config['classes_per_task'] != 10
            or config['lambda_validation_enabled'] != 1
            or config['head_consolidation_mode'] != 'adaptive_dual_branch'):
        raise ValueError('unexpected CIFAR Adaptive protocol')
    source_modules = {
        name: sha256(Path(module.__file__))
        for name, module in (
            ('models.py', models),
            ('data_utils.py', data_utils),
            ('determinism.py', determinism),
            ('adaptive_head_consolidation.py', adaptive_head_consolidation),
            ('vfl_trainer.py', vfl_trainer),
        )
    }
    if any(digest != record['artifact_sha256']['source:' + name]
           for name, digest in source_modules.items()):
        raise ValueError('analysis model code differs from formal producer')
    source_data = {
        key: sha256(Path(config['data_path']) / key.split(':', 1)[1])
        for key in record['artifact_sha256'] if key.startswith('data:')
    }
    if any(digest != record['artifact_sha256'][key]
           for key, digest in source_data.items()):
        raise ValueError('CIFAR payload differs from formal producer')

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    task_classes = checkpoint['seen_task_classes']
    ordered_tasks = [list(map(int, task_classes[i])) for i in range(10)]
    if sorted(sum(ordered_tasks, [])) != list(range(100)):
        raise ValueError('checkpoint task classes are incomplete')
    newest = set(ordered_tasks[-1])
    old = set(range(100)) - newest
    class_task = torch.empty(100, dtype=torch.long, device=device)
    for task_id, classes in enumerate(ordered_tasks):
        class_task[classes] = task_id

    with tempfile.TemporaryDirectory(prefix='vfcl-head-error-') as scratch:
        args = SimpleNamespace(**config)
        args.output_dir = scratch
        args.device = device
        args.formal_deferred_evaluation = False
        args.data_flow_audit = 0
        args.num_workers = 0
        dataset = VFLDataset(args)
        source_manifest = json.loads(source_manifest_path.read_text(encoding='utf-8'))
        if dataset.validation_manifest != source_manifest:
            raise ValueError('rebuilt validation cohort differs from source')
        bottoms, top = build_models(args)
        trainer = VFLTrainer(bottoms, top, args)
        trainer.load_state(checkpoint['trainer_state'])
        for model in (*trainer.bottoms, trainer.top_model):
            model.eval()
        if (not bool(top._adaptive_enabled)
                or top._adaptive_full_weight.numel() == 0
                or top._adaptive_class_order.tolist() != list(range(100))):
            raise ValueError('final checkpoint lacks full adaptive branches')
        gate = float(top._adaptive_gate)
        if not 0.0 < gate < 1.0:
            raise ValueError('adaptive gate is an endpoint')

        counts = {name: new_counts(10) for name in ('full', 'bias', 'mixed')}
        validation_loader = dataset.get_validation_loader(list(range(100)))
        for batch_x, batch_y in validation_loader:
            batch_x = batch_x.to(device)
            labels = batch_y.to(device)
            parts = split_features(batch_x, args)
            embeddings = [bottom(part) for bottom, part in zip(trainer.bottoms, parts)]
            aggregate = trainer._aggregate(embeddings)
            full, bias = top.branch_log_probabilities(aggregate)
            mixed = top(aggregate)
            expected = mix_log_probabilities(full, bias, gate)
            if (not torch.allclose(mixed, expected, rtol=0, atol=1e-10)
                    or not all(torch.isfinite(x).all() for x in (full, bias, mixed))):
                raise ValueError('adaptive branch decomposition failed')
            old_mask = torch.tensor([int(y) in old for y in labels], device=device)
            new_mask = ~old_mask
            true_tasks = class_task[labels]
            for name, logp in (('full', full), ('bias', bias), ('mixed', mixed)):
                item = counts[name]
                pred = logp.argmax(dim=1)
                predicted_tasks = class_task[pred]
                correct = pred == labels
                item['total'] += labels.numel()
                item['correct'] += int(correct.sum())
                item['old_total'] += int(old_mask.sum())
                item['old_correct'] += int((correct & old_mask).sum())
                item['new_total'] += int(new_mask.sum())
                item['new_correct'] += int((correct & new_mask).sum())
                item['old_to_new'] += int((old_mask & torch.isin(pred, torch.tensor(
                    sorted(newest), device=device))).sum())
                item['new_to_old'] += int((new_mask & torch.isin(pred, torch.tensor(
                    sorted(old), device=device))).sum())
                item['nll_sum'] += float(-logp.gather(1, labels[:, None]).sum())
                confusion = torch.bincount(
                    true_tasks * 10 + predicted_tasks, minlength=100,
                ).reshape(10, 10).cpu().tolist()
                for true_task in range(10):
                    item['task_total'][true_task] += int((true_tasks == true_task).sum())
                    item['task_correct'][true_task] += int(
                        (correct & (true_tasks == true_task)).sum())
                    for predicted_task in range(10):
                        item['task_confusion'][true_task][predicted_task] += confusion[
                            true_task][predicted_task]
                    subset = true_tasks == true_task
                    if bool(subset.any()):
                        allowed = torch.tensor(ordered_tasks[true_task], device=device)
                        local_pred = allowed[logp[subset][:, allowed].argmax(dim=1)]
                        item['taskil_correct'][true_task] += int(
                            (local_pred == labels[subset]).sum())

        branches = {name: finish(value) for name, value in counts.items()}
        history = checkpoint['cl_state']['head_consolidation_history']
        if not history or not isinstance(history[-1], dict):
            raise ValueError('adaptive head history is missing')
        gate_evidence = history[-1]['gate']
        for branch, evidence_name in (('full', 'full_branch_nll'),
                                      ('bias', 'bias_branch_nll'),
                                      ('mixed', 'mixture_nll')):
            if abs(branches[branch]['nll'] - gate_evidence[evidence_name]) > 1e-4:
                raise ValueError(f'{branch} validation NLL differs from frozen gate evidence')
        return dict(
            seed=seed, source_record_sha256=sha256(record_path),
            source_commit=record['source_commit'],
            source_config_sha256=sha256(config_path),
            source_checkpoint_sha256=sha256(checkpoint_path),
            source_validation_manifest_sha256=sha256(source_manifest_path),
            source_modules_sha256=source_modules,
            source_data_sha256=source_data,
            validation_manifest_identity=source_manifest['sha256'],
            gate=gate, task_classes=ordered_tasks,
            gate_evidence_nll={key: gate_evidence[key] for key in
                               ('full_branch_nll', 'bias_branch_nll', 'mixture_nll')},
            branches=branches,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    output = args.output_dir.resolve()
    if output == root or root in output.parents or output.exists():
        raise ValueError('output must be a new directory outside the formal root')
    if not (root / 'METHOD_SHARD_SUCCESS').is_file():
        raise ValueError('formal Adaptive shard lacks success marker')
    seeds = [evaluate_seed(root, seed, args.device) for seed in (42, 43, 44)]
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema_version=1, source_root=str(root), device=args.device,
                  seeds=seeds)
    temporary = output / 'head_error.json.tmp'
    with open(temporary, 'x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output / 'head_error.json')
    print(json.dumps({
        branch: {
            key: statistics.fmean(
                seed['branches'][branch][key] for seed in seeds)
            for key in ('accuracy', 'old_accuracy', 'new_accuracy',
                        'taskil_accuracy', 'old_to_new_rate',
                        'new_to_old_rate', 'nll')
        }
        for branch in ('full', 'bias', 'mixed')
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
