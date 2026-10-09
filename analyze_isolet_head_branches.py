"""Read-only ISOLET Full/Bias/Mixed head diagnosis on frozen gate validation."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from adaptive_consolidation_audit import _safe_torch_load, _tensor_sha256
from adaptive_head_consolidation import mix_log_probabilities
from head_consolidation import hash_top_state
from models import TopModel


SOURCE_COMMIT = 'a575bbf446ae501cf8e580ba62c2cc5492a25f30'
PARAMETERS = {'proto_lambda_a': 0.05, 'distill_weight': 0.10,
              'feat_distill_weight': 0.02}
BASELINE_PARAMETERS = {'proto_lambda_a': 0.15, 'distill_weight': 0.25,
                       'feat_distill_weight': 0.05}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def score_branch(log_probs, labels, task_classes):
    """Aggregate only counts and accuracies; preserve no individual records."""
    if (log_probs.ndim != 2 or labels.ndim != 1
            or log_probs.shape[0] != labels.numel() or not labels.numel()
            or not torch.isfinite(log_probs).all()):
        raise ValueError('invalid branch scores or labels')
    n, class_count = log_probs.shape
    tasks = sorted(task_classes)
    if tasks != list(range(len(tasks))) or len(tasks) < 2:
        raise ValueError('invalid task IDs')
    class_task = torch.full((class_count,), -1, dtype=torch.long)
    for task_id in tasks:
        for class_id in task_classes[task_id]:
            if not 0 <= class_id < class_count or class_task[class_id] >= 0:
                raise ValueError('invalid task class map')
            class_task[class_id] = task_id
    if (bool((class_task < 0).any()) or labels.dtype != torch.long
            or bool(((labels < 0) | (labels >= class_count)).any())):
        raise ValueError('incomplete task class map or invalid labels')
    predicted = log_probs.argmax(dim=1)
    true_task, predicted_task = class_task[labels], class_task[predicted]
    correct = predicted == labels
    task_correct = true_task == predicted_task
    taskil = torch.zeros(n, dtype=torch.bool)
    per_task_cil, per_task_taskil = [], []
    for task_id in tasks:
        mask = true_task == task_id
        members = torch.tensor(task_classes[task_id], dtype=torch.long)
        if not bool(mask.any()):
            raise ValueError('empty task in validation cohort')
        local = members[log_probs[mask][:, members].argmax(dim=1)]
        taskil[mask] = local == labels[mask]
        per_task_cil.append(float(correct[mask].double().mean()))
        per_task_taskil.append(float(taskil[mask].double().mean()))
    old, newest = true_task != tasks[-1], true_task == tasks[-1]
    confusion = torch.bincount(
        true_task * len(tasks) + predicted_task,
        minlength=len(tasks) ** 2,
    ).reshape(len(tasks), len(tasks)).tolist()
    return {
        'count': n,
        'cil': float(correct.double().mean()),
        'task_id': float(task_correct.double().mean()),
        'task_il': float(taskil.double().mean()),
        'old_cil': float(correct[old].double().mean()),
        'new_cil': float(correct[newest].double().mean()),
        'old_to_new_rate': float((old & (predicted_task == tasks[-1])).sum() / old.sum()),
        'new_to_old_rate': float((newest & (predicted_task != tasks[-1])).sum() / newest.sum()),
        'cross_task_error_fraction': float((~task_correct).sum() / (~correct).sum())
        if bool((~correct).any()) else 0.0,
        'nll': float(-log_probs.gather(1, labels[:, None]).mean()),
        'per_task_cil': per_task_cil,
        'per_task_taskil': per_task_taskil,
        'task_confusion': confusion,
    }


@torch.no_grad()
def analyze_run(run, variant, seed):
    if (run / 'exit.code').read_text(encoding='utf-8').strip() != '0':
        raise ValueError(f'{run} did not complete')
    published = json.loads((run / 'FORMAL_EVALUATION_PUBLISHED.json').read_text())
    config_path = run / 'config.json'
    checkpoint_path = run / 'checkpoints' / 'formal_final.pt'
    manifest_path = run / 'validation' / 'validation_manifest.json'
    config = json.loads(config_path.read_text())
    expected = PARAMETERS if variant == 'selected' else BASELINE_PARAMETERS
    if (published.get('status') != 'published'
            or published['checkpoint']['sha256'] != sha256(checkpoint_path)
            or published['source_provenance']['source_commit'] != SOURCE_COMMIT
            or config['seed'] != seed or config['data'] != 'tabvfl'
            or config['num_classes'] != 26 or config['num_tasks'] != 13
            or config['lambda_validation_per_class'] != 40
            or config['formal_deferred_evaluation'] is not True
            or any(config[key] != value for key, value in expected.items())):
        raise ValueError(f'{run} protocol differs')
    checkpoint = _safe_torch_load(checkpoint_path)
    bundle = checkpoint['cl_state']['adaptive_audit_bundle']
    x = bundle['validation_embeddings'].detach().cpu()
    y = bundle['validation_labels'].detach().cpu().long()
    result = bundle['result']
    manifest = json.loads(manifest_path.read_text())
    if (y.shape != (1040,) or x.shape[0] != 1040
            or any(int((y == c).sum()) != 40 for c in range(26))
            or _tensor_sha256(x) != bundle['validation_embeddings_sha256']
            or _tensor_sha256(y) != bundle['validation_labels_sha256']
            or result['validation_manifest'] != manifest
            or bundle['diagnostics']['source_splits']['test_used'] is not False):
        raise ValueError(f'{run} validation cache differs')
    top = TopModel(x.shape[1], 26, cosine=bool(config['cosine_head']))
    top.load_state_dict(bundle['installed_state'], strict=True)
    top.eval()
    if (hash_top_state(top) != hash_top_state(checkpoint['trainer_state']['top_model'])
            or top._adaptive_class_order.tolist() != list(range(26))):
        raise ValueError(f'{run} installed head differs')
    full, bias = top.branch_log_probabilities(x)
    mixed = top(x)
    gate = float(result['gate']['g'])
    if not torch.allclose(mixed, mix_log_probabilities(full, bias, gate),
                          rtol=0, atol=1e-10):
        raise ValueError(f'{run} adaptive mixture differs')
    scores = {name: score_branch(logp, y, bundle['task_classes'])
              for name, logp in (('full', full), ('bias', bias), ('mixed', mixed))}
    diagnostic = bundle['diagnostics']['validation']
    for name, key in (('full', 'full_branch_nll'),
                      ('bias', 'bias_branch_nll'), ('mixed', 'mixture_nll')):
        if abs(scores[name]['nll'] - diagnostic[key]) > 1e-8:
            raise ValueError(f'{run} {name} NLL differs from frozen audit')
    if abs(scores['mixed']['task_id'] - diagnostic['task_id_accuracy']['after']) > 1e-8:
        raise ValueError(f'{run} mixed task accuracy differs from frozen audit')
    return {'variant': variant, 'seed': seed,
            'checkpoint_sha256': sha256(checkpoint_path),
            'config_sha256': sha256(config_path),
            'validation_manifest_sha256': sha256(manifest_path),
            'gate': gate, 'scores': scores,
            'frozen_diagnostics': bundle['diagnostics']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or root in output.parents:
        raise ValueError('output must be new and outside the source root')
    runs = []
    for seed in (42, 43, 44):
        for variant, name in (('selected', f'isolet_seed{seed}'),
                              ('baseline', f'isolet_baseline_seed{seed}')):
            runs.append(analyze_run(root / name, variant, seed))
    means = {
        variant: {
            branch: {key: statistics.fmean(
                row['scores'][branch][key] for row in runs
                if row['variant'] == variant)
                for key in ('cil', 'task_id', 'task_il', 'old_cil',
                            'new_cil', 'old_to_new_rate', 'new_to_old_rate',
                            'cross_task_error_fraction', 'nll')}
            for branch in ('full', 'bias', 'mixed')}
        for variant in ('selected', 'baseline')
    }
    payload = {'schema_version': 1, 'status': 'gate_used_validation_descriptive',
               'source_root': str(root), 'runs': runs, 'means': means}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps(means, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
