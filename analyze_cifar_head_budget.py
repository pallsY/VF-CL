"""Compare three fixed head-fit sample budgets on fresh CIFAR validation."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import adaptive_consolidation_audit
import analyze_head_offset_pilot
import data_utils
import head_consolidation
import models
import vfl_trainer
from analyze_head_offset_pilot import (
    file_sha256, select_calibration_indices, summarize_logits,
)
from adaptive_consolidation_audit import _safe_torch_load
from data_utils import VFLDataset
from head_consolidation import consolidate_classifier
from models import build_models
from vfl_trainer import VFLTrainer


SOURCE_COMMIT = '7bfe6b1d724fb1206bc0053a9008126bad86332d'
HELPER_SHA256 = '346b3b95a5abe2f5d7c749bf2b4780b1fbbebd5a5ce7843ff43cab684b9682f9'


def tensor_tree_sha256(states):
    digest = hashlib.sha256()
    for index, state in enumerate(states):
        for name, value in sorted(state.items()):
            digest.update(str(index).encode())
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def classify_budget_response(small, large):
    """Apply the fixed descriptive 20-versus-400 comparison."""
    if any(small[key] != large[key] for key in
           ('examples', 'old_examples', 'new_examples')):
        raise ValueError('head budgets used different validation cohorts')
    gain = large['correct'] - small['correct']
    if (20 * gain >= small['examples']
            and 50 * (small['old_correct'] - large['old_correct'])
            <= small['old_examples']
            and 50 * (small['new_correct'] - large['new_correct'])
            <= small['new_examples']):
        return 'sample_sufficiency_signal'
    if (50 * gain < small['examples']
            and large['nll'] >= small['nll']):
        return 'early_saturation'
    return 'inconclusive'


def validate_budget_config(source, recorded, run_dir, overrides):
    """Require exactly the seed-46 preregistered deviations from formal source."""
    run_dir = Path(run_dir).resolve()
    expected = copy.deepcopy(source)
    expected.update(
        seed=46,
        lambda_validation_split_seed=20261008,
        formal_deferred_evaluation=False,
        head_consolidation_enabled=0,
        head_consolidation_mode='full_classifier',
        results_dir=str(run_dir.parent),
        output_dir=str(run_dir),
        exp_name='cifar_head_budget_seed46_baseline',
    )
    changes = {
        key: {'source': source.get(key), 'pilot': expected.get(key)}
        for key in set(source) | set(expected)
        if source.get(key) != expected.get(key)
    }
    if recorded != expected or overrides != changes or len(changes) != 8:
        raise ValueError('pilot training config differs from preregistered overrides')
    return changes


def run(run_dir, output_dir, device):
    run_dir = Path(run_dir).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() or output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError('output must be a fresh root outside training')
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
            != SOURCE_COMMIT
            or subprocess.check_output(['git', 'status', '--porcelain'], text=True)):
        raise ValueError('diagnostic requires the exact clean producer worktree')
    complete = json.loads((run_dir / 'PILOT_TRAINING_COMPLETE.json').read_text())
    protocol = json.loads((run_dir / 'PILOT_PROTOCOL.json').read_text())
    if complete.get('source_commit') != SOURCE_COMMIT:
        raise ValueError('pilot completion marker source differs')
    launcher_path = Path('/tmp/launch_head_budget_screen.py')
    if file_sha256(launcher_path) != protocol.get('launcher_sha256'):
        raise ValueError('pilot launcher differs from completed protocol')
    config_path = run_dir / 'config.json'
    checkpoint_path = run_dir / 'checkpoints' / 'event_9_CIL.pt'
    results_path = run_dir / 'results.json'
    audit_path = run_dir / 'data_flow_audit.jsonl'
    for path, key in ((config_path, 'config_sha256'),
                      (checkpoint_path, 'checkpoint_sha256'),
                      (results_path, 'results_sha256'),
                      (audit_path, 'data_flow_audit_sha256')):
        if file_sha256(path) != complete[key]:
            raise ValueError('training artifact differs from complete marker')
    with open(audit_path, encoding='utf-8') as handle:
        for line in handle:
            entry = json.loads(line)
            if (entry.get('split') == 'test'
                    or str(entry.get('loader_key', '')).startswith("('test'")):
                raise ValueError('pilot data flow includes final-test access')
    config = json.loads(config_path.read_text())
    source_config_path = Path(
        '/home/c3080/YangXiaoXiang/VF-CL/results/formal-method-adaptive-20261005-v1/'
        'runs/cifar100%3Aadaptive%3A42/config.json'
    )
    if file_sha256(source_config_path) != protocol.get('source_config_sha256'):
        raise ValueError('formal source config differs from pilot launch')
    validate_budget_config(
        json.loads(source_config_path.read_text()), config,
        run_dir, protocol.get('overrides'),
    )
    expected_data = {
        'data:cifar-100-python/train',
        'data:cifar-100-python/test',
        'data:cifar-100-python/meta',
    }
    if (protocol.get('source_commit') != SOURCE_COMMIT
            or protocol.get('design_commit')
            != '645bb89f97feac5386c09ffec61bde7c10814eb8'
            or config.get('seed') != 46
            or config.get('head_consolidation_enabled') != 0
            or config.get('lambda_validation_split_seed') != 20261008
            or set(protocol.get('source_data_sha256', {})) != expected_data
            or any(file_sha256(Path(config['data_path']) / key.split(':', 1)[1])
                   != digest
                   for key, digest in protocol['source_data_sha256'].items())):
        raise ValueError('training configuration/data differ from pilot protocol')
    record_path = Path(
        '/home/c3080/YangXiaoXiang/VF-CL/results/formal-method-adaptive-20261005-v1/'
        'records/cifar100%3Aadaptive%3A42.json'
    )
    if file_sha256(record_path) != protocol['source_record_sha256']:
        raise ValueError('formal source record differs from pilot launch')
    record = json.loads(record_path.read_text())
    if (file_sha256(source_config_path)
            != record['artifact_sha256']['config']
            or any(digest != record['artifact_sha256'][key]
                   for key, digest in protocol['source_data_sha256'].items())):
        raise ValueError('formal source config/data hashes differ')
    source_modules = {
        name: file_sha256(Path(module.__file__))
        for name, module in (
            ('adaptive_consolidation_audit.py', adaptive_consolidation_audit),
            ('data_utils.py', data_utils),
            ('head_consolidation.py', head_consolidation),
            ('models.py', models),
            ('vfl_trainer.py', vfl_trainer),
        )
    }
    if any(digest != record['artifact_sha256']['source:' + name]
           for name, digest in source_modules.items()):
        raise ValueError('imported model or fitter code differs from producer')
    helper_sha = file_sha256(Path(analyze_head_offset_pilot.__file__))
    if helper_sha != HELPER_SHA256:
        raise ValueError('imported pilot analysis helper differs from frozen source')
    checkpoint = _safe_torch_load(checkpoint_path)
    tasks = [list(map(int, checkpoint['seen_task_classes'][task]))
             for task in range(10)]
    if tasks != [list(range(task * 10, (task + 1) * 10)) for task in range(10)]:
        raise ValueError('checkpoint task plan differs')

    with tempfile.TemporaryDirectory(prefix='vfcl-head-budget-') as scratch:
        args = SimpleNamespace(**config)
        args.output_dir = scratch
        args.device = device
        args.num_workers = 0
        args.data_flow_audit = 0
        dataset = VFLDataset(args)
        validation_manifest = run_dir / 'validation' / 'validation_manifest.json'
        bic_manifest = run_dir / 'bic' / 'calibration_manifest.json'
        if (dataset.validation_manifest != json.loads(validation_manifest.read_text())
                or dataset.calibration_manifest != json.loads(bic_manifest.read_text())):
            raise ValueError('holdout manifests differ from training')
        heldout = dataset.validation_indices | dataset.calibration_indices
        selected = select_calibration_indices(
            dataset.validationset.targets, heldout, 400,
        )
        if len(selected) != 40000 or set(selected) & heldout:
            raise ValueError('train-only class-balanced calibration failed')

        bottoms, top = build_models(args)
        trainer = VFLTrainer(bottoms, top, args)
        trainer.load_state(checkpoint['trainer_state'])
        if bool(top._adaptive_enabled) or bool(top._logit_calibration_enabled):
            raise ValueError('final head is already consolidated')
        for model in (*trainer.bottoms, trainer.top_model):
            model.eval()
        for bottom in trainer.bottoms:
            for parameter in bottom.parameters():
                parameter.requires_grad_(False)
        bottom_before = tensor_tree_sha256([b.state_dict() for b in trainer.bottoms])
        train_loader = DataLoader(Subset(dataset.validationset, selected),
                                  batch_size=64, shuffle=False, num_workers=0)
        train_features, train_labels = trainer.compute_embeddings(train_loader)
        validation_features, validation_labels = trainer.compute_embeddings(
            dataset.get_validation_loader(list(range(100)))
        )
        if (not all(int((train_labels == c).sum()) == 400 for c in range(100))
                or not all(int((validation_labels == c).sum()) == 25
                           for c in range(100))):
            raise ValueError('calibration/validation class coverage differs')
        replay = {
            class_id: train_features[train_labels == class_id]
            for class_id in range(100)
        }
        budget_index_hashes = {
            str(budget): hashlib.sha256(json.dumps(
                [index for class_id in range(100)
                 for index in selected[class_id * 400:class_id * 400 + budget]],
                separators=(',', ':'),
            ).encode()).hexdigest()
            for budget in (20, 100, 400)
        }
        with torch.no_grad():
            baseline_logits = top(validation_features.to(device)).cpu().numpy()
        labels = validation_labels.numpy()
        fitted, audits = {}, {}
        for budget in (20, 100, 400):
            candidate = copy.deepcopy(top).eval()
            audit = consolidate_classifier(
                candidate, prototypes={}, regularization=0.01,
                steps=500, lr=0.01, samples_per_class=budget,
                seed=46, device=device,
                replay_embeddings=replay,
                replay_source=f'fixed_train_only_{budget}_per_class',
                persistent_raw_example_count=budget * 100,
            )
            if (audit['validation_used'] or audit['test_used']
                    or audit['fit_sample_count'] != budget * 100):
                raise RuntimeError('head fitter used unexpected data')
            with torch.no_grad():
                logits = candidate(validation_features.to(device)).cpu().numpy()
            fitted[str(budget)] = summarize_logits(logits, labels, tasks, tasks[-1])
            audits[str(budget)] = audit
        bottom_after = tensor_tree_sha256([b.state_dict() for b in trainer.bottoms])
        if bottom_after != bottom_before:
            raise RuntimeError('bottom encoders changed during head fit')

    baseline = summarize_logits(baseline_logits, labels, tasks, tasks[-1])
    results = json.loads(results_path.read_text())
    if abs(baseline['accuracy'] - results['cl_metrics']['AA_final']) > 0.001:
        raise ValueError('unmodified baseline differs from saved run')
    bic_pair = results['bic_final']['paired']
    if abs(bic_pair['raw']['overall_accuracy'] - baseline['accuracy']) > 0.001:
        raise ValueError('raw baseline differs from saved BiC comparison')
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
        'status': 'exploratory_fresh_seed46_holdout',
        'source_run': str(run_dir),
        'source_checkpoint_sha256': file_sha256(checkpoint_path),
        'source_config_sha256': file_sha256(config_path),
        'launcher_sha256': file_sha256(launcher_path),
        'formal_source_config_sha256': file_sha256(source_config_path),
        'data_flow_audit_sha256': file_sha256(audit_path),
        'source_modules_sha256': source_modules,
        'formal_source_record_sha256': file_sha256(record_path),
        'analysis_helper_sha256': helper_sha,
        'validation_manifest_sha256': file_sha256(validation_manifest),
        'bic_manifest_sha256': file_sha256(bic_manifest),
        'calibration_index_sha256': hashlib.sha256(
            json.dumps(selected, separators=(',', ':')).encode()
        ).hexdigest(),
        'script_sha256': file_sha256(__file__),
        'bottom_state_sha256': bottom_before,
        'budgets_per_class': [20, 100, 400],
        'calibration_indices_sha256_by_budget': budget_index_hashes,
        'head_fit_audits': audits,
        'baseline': baseline,
        'fitted_heads': fitted,
        'existing_bic_context': bic_context,
        'decision': classify_budget_response(fitted['20'], fitted['400']),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    tmp = output_dir / 'head_budget.json.tmp'
    with open(tmp, 'x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, output_dir / 'head_budget.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    cli = parser.parse_args()
    result = run(cli.run_dir, cli.output_dir, cli.device)
    print(json.dumps({key: result[key] for key in
                      ('baseline', 'fitted_heads', 'decision')},
                     indent=2, sort_keys=True))
