"""Exploratory frozen-feature full-head capacity check; no method selection."""

import argparse
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
    expected_data = {
        'data:cifar-100-python/train',
        'data:cifar-100-python/test',
        'data:cifar-100-python/meta',
    }
    if (protocol.get('source_commit') != SOURCE_COMMIT
            or config.get('seed') != 45
            or config.get('head_consolidation_enabled') != 0
            or config.get('lambda_validation_split_seed') != 20261007
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

    with tempfile.TemporaryDirectory(prefix='vfcl-frozen-head-') as scratch:
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
            dataset.validationset.targets, heldout, 20,
        )
        if len(selected) != 2000 or set(selected) & heldout:
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
        if (not all(int((train_labels == c).sum()) == 20 for c in range(100))
                or not all(int((validation_labels == c).sum()) == 25
                           for c in range(100))):
            raise ValueError('calibration/validation class coverage differs')
        replay = {
            class_id: train_features[train_labels == class_id]
            for class_id in range(100)
        }
        with torch.no_grad():
            baseline_logits = top(validation_features.to(device)).cpu().numpy()
        audit = consolidate_classifier(
            top, prototypes={}, regularization=0.01,
            steps=500, lr=0.01, samples_per_class=20,
            seed=45, device=device,
            replay_embeddings=replay,
            replay_source='fixed_train_only_20_per_class',
            persistent_raw_example_count=2000,
        )
        with torch.no_grad():
            fitted_logits = top(validation_features.to(device)).cpu().numpy()
        bottom_after = tensor_tree_sha256([b.state_dict() for b in trainer.bottoms])
        if bottom_after != bottom_before:
            raise RuntimeError('bottom encoders changed during head fit')
        if audit['validation_used'] or audit['test_used']:
            raise RuntimeError('head fitter accessed validation or test')

    labels = validation_labels.numpy()
    baseline = summarize_logits(baseline_logits, labels, tasks, tasks[-1])
    fitted = summarize_logits(fitted_logits, labels, tasks, tasks[-1])
    results = json.loads(results_path.read_text())
    if abs(baseline['accuracy'] - results['cl_metrics']['AA_final']) > 0.001:
        raise ValueError('unmodified baseline differs from saved run')
    result = {
        'schema_version': 1,
        'status': 'exploratory_reused_validation',
        'source_run': str(run_dir),
        'source_checkpoint_sha256': file_sha256(checkpoint_path),
        'source_config_sha256': file_sha256(config_path),
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
        'head_fit_audit': audit,
        'baseline': baseline,
        'frozen_feature_full_head': fitted,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    tmp = output_dir / 'frozen_head_capacity.json.tmp'
    with open(tmp, 'x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, output_dir / 'frozen_head_capacity.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    cli = parser.parse_args()
    result = run(cli.run_dir, cli.output_dir, cli.device)
    print(json.dumps({key: result[key] for key in
                      ('baseline', 'frozen_feature_full_head')},
                     indent=2, sort_keys=True))
