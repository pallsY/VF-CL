"""One-time paired test readout from an already frozen Adaptive checkpoint."""

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

from adaptive_consolidation_audit import _fresh_trainer, _strict_json, _strict_load_trainer_state
from data_utils import VFLDataset, split_features


_METRICS = ('AA_final', 'BWT', 'AA_final_taskil')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value):
    return hashlib.sha256(_strict_json(value).encode('utf-8')).hexdigest()


def summarize_paired(measured, history, baseline):
    """Reuse the frozen diagonal only after Mixed reproduces published metrics."""
    final = history[-1]
    tasks = list(final['deferred_diagonal'])
    if (set(measured) != set(tasks)
            or set(final['per_task_accs']) != set(tasks)
            or set(final['per_task_accs_taskil']) != set(tasks)):
        raise ValueError('paired task identity mismatch')
    rows = {}
    for task in tasks:
        row = measured[task]
        if row['count'] <= 0:
            raise ValueError('paired task count is empty')
        rows[task] = {
            name: round(row[name] / row['count'], 4)
            for name in ('mixed_correct', 'mixed_taskil_correct',
                         'factorized_correct', 'factorized_taskil_correct')
        }
        if (rows[task]['mixed_correct'] != final['per_task_accs'][task]
                or rows[task]['mixed_taskil_correct']
                != final['per_task_accs_taskil'][task]):
            raise ValueError('Mixed baseline mismatch on ' + task)
    def metrics(class_key, taskil_key):
        class_final = [rows[task][class_key] for task in tasks]
        taskil_final = [rows[task][taskil_key] for task in tasks]
        diagonal = [final['deferred_diagonal'][task] for task in tasks]
        return {
            'AA_final': round(statistics.fmean(class_final), 4),
            'BWT': round(statistics.fmean(
                class_final[i] - diagonal[i] for i in range(len(tasks) - 1)
            ), 4),
            'AA_final_taskil': round(statistics.fmean(taskil_final), 4),
        }
    mixed = metrics('mixed_correct', 'mixed_taskil_correct')
    if any(abs(mixed[key] - baseline[key]) > 1e-4 for key in _METRICS):
        raise ValueError('Mixed baseline mismatch in summary')
    return {
        'mixed': mixed,
        'factorized': metrics('factorized_correct', 'factorized_taskil_correct'),
        'per_task': rows,
    }


@torch.no_grad()
def evaluate_run(run_dir, output_dir):
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    for name in ('FORMAL_STATE_FROZEN.json', 'FORMAL_EVALUATION_PUBLISHED.json'):
        if not (run_dir / name).is_file():
            raise ValueError('source run is not formally frozen and published')
    checkpoint_path = run_dir / 'checkpoints' / 'formal_final.pt'
    config_path = run_dir / 'config.json'
    results_path = run_dir / 'results.json'
    config = json.loads(config_path.read_text())
    results = json.loads(results_path.read_text())
    published_path = run_dir / 'FORMAL_EVALUATION_PUBLISHED.json'
    published = json.loads(published_path.read_text())
    checkpoint_hash = _sha256(checkpoint_path)
    results_hash = _sha256(results_path)
    if (published.get('status') != 'published'
            or published.get('checkpoint', {}).get('sha256') != checkpoint_hash
            or published.get('results', {}).get('sha256') != results_hash
            or published.get('results_sha256') != _canonical_sha256(results)):
        raise ValueError('published source artifact identity mismatch')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if (checkpoint.get('protocol', {}).get('head_consolidation_mode')
            != 'adaptive_dual_branch'):
        raise ValueError('source checkpoint is not Adaptive')
    output_dir.mkdir(parents=True)
    args = SimpleNamespace(**config)
    args.device = 'cpu'
    args.num_workers = 0
    args.output_dir = str(output_dir)
    # This is a separate read-only posthoc test access after method freeze.
    args.formal_deferred_evaluation = False
    dataset = VFLDataset(args)
    trainer = _fresh_trainer(checkpoint, args)
    _strict_load_trainer_state(trainer, checkpoint['trainer_state'],
                               checkpoint['protocol'])
    for bottom in trainer.bottoms:
        bottom.eval()
    top = trainer.top_model.eval()
    if not bool(top._adaptive_enabled):
        raise ValueError('source checkpoint lacks installed Adaptive head')
    output_classes = [int(value) for value in top._adaptive_class_order.tolist()]
    task_classes = checkpoint['seen_task_classes']
    expected_classes = [int(c) for task in sorted(task_classes)
                        for c in task_classes[task]]
    if output_classes != expected_classes:
        raise ValueError('source task and output class order mismatch')
    class_to_column = {class_id: i for i, class_id in enumerate(output_classes)}
    output_class_tensor = torch.tensor(output_classes, dtype=torch.long)
    measured = {}
    for task_id in sorted(task_classes):
        classes = [int(c) for c in task_classes[task_id]]
        task_columns = torch.tensor([class_to_column[c] for c in classes])
        task_class_tensor = torch.tensor(classes, dtype=torch.long)
        counts = {'count': 0, 'mixed_correct': 0, 'mixed_taskil_correct': 0,
                  'factorized_correct': 0, 'factorized_taskil_correct': 0}
        for batch_x, labels in dataset.get_test_loader(classes):
            parts = split_features(batch_x, args)
            embeddings = [bottom(part) for bottom, part in zip(trainer.bottoms, parts)]
            features = trainer._aggregate(embeddings)
            for name, log_p in (
                    ('mixed', top(features)),
                    ('factorized', top.factorized_log_probabilities(features))):
                predicted = output_class_tensor[log_p.argmax(1).cpu()]
                within = task_class_tensor[log_p[:, task_columns].argmax(1).cpu()]
                counts[name + '_correct'] += int((predicted == labels).sum())
                counts[name + '_taskil_correct'] += int((within == labels).sum())
            counts['count'] += labels.numel()
        measured[f'task_{task_id}'] = counts
    comparison = summarize_paired(measured, results['task_acc_history'],
                                  results['cl_metrics'])
    test_source = (Path(args.vector_npz) if getattr(args, 'vector_npz', None)
                   else Path(args.data_path) / 'cifar-100-python' / 'test')
    report = {
        'method': 'factorized_pre_within_v1',
        'access_protocol': 'paired_posthoc_test_readout_v1',
        'evaluator_commit': subprocess.check_output(
            ['git', '-C', str(Path(__file__).resolve().parent), 'rev-parse', 'HEAD'],
            text=True,
        ).strip(),
        'published_marker_sha256': _sha256(published_path),
        'source_run': str(run_dir),
        'checkpoint_sha256': checkpoint_hash,
        'config_sha256': _sha256(config_path),
        'results_sha256': results_hash,
        'test_source_sha256': _sha256(test_source),
        'test_used_for_fit': False,
        'test_used_for_selection': False,
        **comparison,
    }
    target = output_dir / 'comparison.json'
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    result = evaluate_run(args.run_dir, args.output_dir)
    print(json.dumps({key: result[key] for key in ('mixed', 'factorized')},
                     sort_keys=True))


if __name__ == '__main__':
    main()
