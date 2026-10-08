"""Run one guarded ISOLET Adaptive development hyperparameter candidate."""

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


SOURCE_COMMIT = 'a575bbf446ae501cf8e580ba62c2cc5492a25f30'
SOURCE_RECORD_SHA256 = '7dc5bf930600d097134948d6ee166fd55490327d16cde1fe2496ac374a73299b'
SOURCE_CONFIG_SHA256 = '5702b8846ca8e9d279728a725550c1b053b8580fec832a9695b56d6ea7f2af34'
DESIGN_COMMIT = '5b3a4e9636b5adfafb10aeadc9f73504c741af54'
FORMAL = (.15, .25, .05)
GRID = ({.05, .15, .30}, {.10, .25, .50}, {.02, .05, .10})
ALLOWED_OVERRIDES = frozenset({
    'seed', 'data_path', 'vector_npz', 'formal_deferred_evaluation',
    'results_dir', 'output_dir', 'exp_name',
    'proto_lambda_a', 'distill_weight', 'feat_distill_weight',
})


class TrainingOnlyStop(Exception):
    """The frozen final state was reached before deferred test evaluation."""


def stop_before_deferred_evaluation(*_args, **_kwargs):
    raise TrainingOnlyStop


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def derive_config(source, root, seed, proto_lambda_a, distill_weight,
                  feat_distill_weight):
    expected = {
        'data': 'tabvfl', 'num_classes': 26, 'num_tasks': 13,
        'classes_per_task': 2, 'num_parties': 4,
        'aggregation': 'concat', 'model_type': 'mlp',
        'seed': 42, 'device': 'cuda:0',
        'formal_deferred_evaluation': True,
        'head_consolidation_enabled': 1,
        'head_consolidation_mode': 'adaptive_dual_branch',
        'head_consolidation_samples_per_class': 20,
        'lambda_validation_enabled': 1,
        'lambda_validation_per_class': 40,
        'lambda_validation_split_seed': 20260809,
        'bic_enabled': 0, 'epochs_per_task': 50,
        'batch_size': 128, 'optimizer': 'adamw',
        'proto_lambda_a': .15,
        'distill_weight': .25,
        'feat_distill_weight': .05,
    }
    if any(source.get(key) != value for key, value in expected.items()):
        raise ValueError('source config is not the audited ISOLET Adaptive run')
    if (Path(source['vector_npz']).name != 'isolet_vfl.npz'
            or any(int(task) < 13 for task in source['unlearn_after_tasks'])):
        raise ValueError('ISOLET source data or timeline differs')
    params = (float(proto_lambda_a), float(distill_weight),
              float(feat_distill_weight))
    if int(seed) not in (45, 46) or any(value not in choices
                                          for value, choices in zip(params, GRID)):
        raise ValueError('seed or candidate is outside the locked grid')
    root = Path(root).resolve()
    tag = (f'seed_{seed}_a{round(params[0] * 100):03d}'
           f'_d{round(params[1] * 100):03d}'
           f'_f{round(params[2] * 100):03d}')
    config = copy.deepcopy(source)
    config.update(
        seed=int(seed),
        data_path='/home/c3080/YangXiaoXiang/VF-CL/data',
        vector_npz='/home/c3080/YangXiaoXiang/VF-CL/data/isolet/isolet_vfl.npz',
        formal_deferred_evaluation=False,
        results_dir=str(root),
        output_dir=str(root / 'run'),
        exp_name='isolet_hparam_' + tag,
        proto_lambda_a=params[0],
        distill_weight=params[1],
        feat_distill_weight=params[2],
    )
    changed = {
        key: {'formal': source.get(key), 'pilot': config.get(key)}
        for key in set(source) | set(config)
        if source.get(key) != config.get(key)
    }
    if not set(changed) <= ALLOWED_OVERRIDES:
        raise ValueError('candidate config has an unregistered override')
    return config, changed


def rank_candidate(metrics, params):
    if any(key not in metrics or not math.isfinite(float(metrics[key]))
           for key in ('cil', 'old_cil', 'til')):
        raise ValueError('candidate metric is missing or non-finite')
    distance = sum(abs(float(value) - baseline) / baseline
                   for value, baseline in zip(params, FORMAL))
    return (float(metrics['cil']), float(metrics['old_cil']),
            float(metrics['til']), -distance)


def readout_metrics(log_probabilities, labels):
    if (not isinstance(log_probabilities, torch.Tensor)
            or log_probabilities.ndim != 2
            or log_probabilities.shape[1] != 26
            or not isinstance(labels, torch.Tensor)
            or labels.ndim != 1
            or labels.numel() == 0
            or labels.shape[0] != log_probabilities.shape[0]):
        raise ValueError('ISOLET readout must have shape [N, 26] with N labels')
    if (labels.dtype != torch.long
            or not bool(((labels >= 0) & (labels < 26)).all())):
        raise ValueError('ISOLET readout label is outside 0..25')
    if not bool(torch.isfinite(log_probabilities).all()):
        raise ValueError('ISOLET readout log probabilities are non-finite')
    log_probabilities = log_probabilities.detach().cpu().double()
    labels = labels.detach().cpu()
    if not bool(torch.allclose(
            torch.logsumexp(log_probabilities, dim=1),
            torch.zeros(labels.numel(), dtype=torch.float64),
            atol=1e-6, rtol=0)):
        raise ValueError('readout rows are not normalized log probabilities')
    pred = log_probabilities.argmax(dim=1)
    starts = (labels // 2) * 2
    pair = torch.stack((log_probabilities[torch.arange(len(labels)), starts],
                        log_probabilities[torch.arange(len(labels)), starts + 1]),
                       dim=1)
    til_pred = starts + pair.argmax(dim=1)
    losses = -log_probabilities[torch.arange(len(labels)), labels]

    def group(mask):
        total = int(mask.sum())
        correct = int(((pred == labels) & mask).sum())
        return {'correct': correct, 'total': total,
                'accuracy': correct / total if total else None}

    til_correct = int((til_pred == labels).sum())
    return {
        'cil': group(torch.ones_like(labels, dtype=torch.bool)),
        'til': {'correct': til_correct, 'total': labels.numel(),
                'accuracy': til_correct / labels.numel()},
        'old_cil': group(labels < 24),
        'new_cil': group(labels >= 24),
        'nll': {'sum': float(losses.sum()), 'total': labels.numel(),
                'mean': float(losses.mean())},
        'per_task': {
            str(task): group(labels // 2 == task)
            for task in range(13)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-config', required=True, type=Path)
    parser.add_argument('--source-record', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int, choices=(45, 46))
    parser.add_argument('--proto-lambda-a', required=True, type=float)
    parser.add_argument('--distill-weight', required=True, type=float)
    parser.add_argument('--feat-distill-weight', required=True, type=float)
    parser.add_argument('--check', action='store_true')
    cli = parser.parse_args()
    source_config = cli.source_config.resolve(strict=True)
    source_record = cli.source_record.resolve(strict=True)
    root = cli.root.resolve()
    if root.exists() or root == source_config.parent or root in source_config.parents:
        raise ValueError('candidate output root must be new and separate')
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
            != SOURCE_COMMIT
            or subprocess.check_output(['git', 'status', '--porcelain'], text=True)):
        raise ValueError('ISOLET sweep requires exact clean formal producer source')
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '1'
            or os.environ.get('PYTHONHASHSEED') != str(cli.seed)
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'
            or os.environ.get('OMP_NUM_THREADS') != '1'
            or os.environ.get('MKL_NUM_THREADS') != '1'):
        raise ValueError('deterministic physical-GPU-1 environment is incomplete')
    if shutil.disk_usage(root.parent).free < 5 * 1024 ** 3:
        raise ValueError('candidate host has less than 5 GiB free')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('masked physical GPU 1 is unavailable')
    free, _ = torch.cuda.mem_get_info(0)
    if free < 4 * 1024 ** 3:
        raise ValueError('masked GPU has less than 4 GiB free')

    record = json.loads(source_record.read_text(encoding='utf-8'))
    source = json.loads(source_config.read_text(encoding='utf-8'))
    if (file_sha256(source_record) != SOURCE_RECORD_SHA256
            or file_sha256(source_config) != SOURCE_CONFIG_SHA256
            or record.get('kind') != 'formal_completed_run'
            or record.get('dataset') != 'isolet'
            or record.get('method') != 'adaptive'
            or record.get('seed') != 42
            or record.get('source_commit') != SOURCE_COMMIT
            or record['artifact_sha256']['config'] != SOURCE_CONFIG_SHA256):
        raise ValueError('frozen formal ISOLET record/config identity mismatch')
    config, changed = derive_config(
        source, root, cli.seed, cli.proto_lambda_a,
        cli.distill_weight, cli.feat_distill_weight,
    )
    source_hashes = {
        name: file_sha256(Path(name))
        for name in (key[7:] for key in record['artifact_sha256']
                     if key.startswith('source:'))
    }
    if any(digest != record['artifact_sha256']['source:' + name]
           for name, digest in source_hashes.items()):
        raise ValueError('formal producer source file hashes differ')
    data_hashes = {
        key: file_sha256(Path(config['data_path']) / key.split(':', 1)[1])
        for key in record['artifact_sha256'] if key.startswith('data:')
    }
    if (set(data_hashes) != {
            'data:isolet/isolet_vfl.npz',
            'data:isolet/isolet_vfl.metadata.json',
            } or any(digest != record['artifact_sha256'][key]
                     for key, digest in data_hashes.items())):
        raise ValueError('ISOLET data payload differs from formal source')

    from config import validate_adaptive_head_consolidation, validate_party_kd_variant
    from data_utils import TaskManager, VFLDataset
    import runner

    args = SimpleNamespace(**config)
    validate_party_kd_variant(config, config['expected_party_kd_variant'])
    validate_adaptive_head_consolidation(args)
    if any(event['type'] != 'CIL' for event in TaskManager(args).get_timeline()):
        raise ValueError('ISOLET sweep timeline contains unlearning')
    with tempfile.TemporaryDirectory(prefix='isolet-hparam-preflight-') as scratch:
        preflight_args = SimpleNamespace(**config)
        preflight_args.output_dir = scratch
        dataset = VFLDataset(preflight_args)
        manifest_path = Path(scratch) / 'validation' / 'validation_manifest.json'
        if (file_sha256(manifest_path)
                != record['artifact_sha256']['validation_manifest']
                or dataset.validation_manifest['sha256']
                != '487e81663a12d4a663a1f421407aa3f88d788cc3c83f7323166cf7aff82902d3'
                or len(dataset.validation_indices) != 1040):
            raise ValueError('ISOLET validation cohort differs from formal source')
    if cli.check:
        print(json.dumps({
            'status': 'ready', 'root': str(root),
            'source_commit': SOURCE_COMMIT,
            'overrides': changed,
            'data_sha256': data_hashes,
            'validation_manifest_sha256': file_sha256(manifest_path)
            if manifest_path.exists() else record['artifact_sha256']['validation_manifest'],
        }, indent=2, sort_keys=True))
        return

    root.mkdir(parents=True, exist_ok=False)
    run = Path(config['output_dir'])
    run.mkdir()
    (run / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (run / 'PILOT_PROTOCOL.json').write_text(json.dumps({
        'schema_version': 1,
        'source_commit': SOURCE_COMMIT,
        'design_commit': DESIGN_COMMIT,
        'source_record_sha256': SOURCE_RECORD_SHA256,
        'source_config_sha256': SOURCE_CONFIG_SHA256,
        'source_files_sha256': source_hashes,
        'data_sha256': data_hashes,
        'launcher_sha256': file_sha256(__file__),
        'overrides': changed,
        'planned_stop': 'before_deferred_test_evaluation',
    }, indent=2, sort_keys=True), encoding='utf-8')
    print('PILOT_OUTPUT_DIR=' + str(run), flush=True)

    original_evaluator = runner.evaluate_deferred_cil_trajectory
    runner.evaluate_deferred_cil_trajectory = stop_before_deferred_evaluation
    stopped = False
    try:
        runner.run_experiment(args)
    except TrainingOnlyStop:
        stopped = True
    finally:
        runner.evaluate_deferred_cil_trajectory = original_evaluator
    if not stopped:
        raise RuntimeError('candidate did not stop before deferred test evaluation')
    checkpoint_path = run / 'adaptive_final.pt'
    freeze_path = run / 'ADAPTIVE_STATE_FROZEN.json'
    events = [run / 'checkpoints' / f'event_{task}_CIL.pt'
              for task in range(13)]
    audit_path = run / 'data_flow_audit.jsonl'
    saved_manifest = run / 'validation' / 'validation_manifest.json'
    if (not checkpoint_path.is_file() or not freeze_path.is_file()
            or not all(path.is_file() for path in events)
            or not audit_path.is_file() or not saved_manifest.is_file()
            or (run / 'results.json').exists()
            or file_sha256(saved_manifest)
            != record['artifact_sha256']['validation_manifest']):
        raise RuntimeError('candidate training-only checkpoint boundary is incomplete')
    access_rows = [json.loads(line) for line in audit_path.read_text(
        encoding='utf-8',
    ).splitlines()]
    if any(row.get('split') == 'test'
           or str(row.get('loader_key', '')).startswith("('test'")
           for row in access_rows):
        raise RuntimeError('candidate accessed final test loader')

    from adaptive_consolidation_audit import _safe_torch_load
    from head_consolidation import hash_top_state
    from models import TopModel
    checkpoint = _safe_torch_load(checkpoint_path)
    bundle = checkpoint['cl_state']['adaptive_audit_bundle']
    labels = bundle['validation_labels'].detach().cpu().long()
    embeddings = bundle['validation_embeddings'].detach().cpu()
    if (labels.shape != (1040,) or embeddings.shape[0] != 1040
            or any(int((labels == class_id).sum()) != 40
                   for class_id in range(26))
            or bundle['result']['validation_manifest']['sha256']
            != dataset.validation_manifest['sha256']):
        raise ValueError('frozen ISOLET validation cache differs from manifest')
    top = TopModel(
        embeddings.shape[1], 26, cosine=bool(config.get('cosine_head', False)),
    )
    top.load_state_dict(bundle['installed_state'], strict=True)
    top.eval()
    if hash_top_state(top) != hash_top_state(checkpoint['trainer_state']['top_model']):
        raise ValueError('frozen installed top differs from checkpoint')
    with torch.no_grad():
        metrics = readout_metrics(top(embeddings), labels)
    readout = {
        'schema_version': 1,
        'status': 'same_gate_validation_development_readout',
        'seed': cli.seed,
        'parameters': {
            'proto_lambda_a': config['proto_lambda_a'],
            'distill_weight': config['distill_weight'],
            'feat_distill_weight': config['feat_distill_weight'],
        },
        'validation_manifest_sha256': file_sha256(saved_manifest),
        'validation_embedding_sha256': bundle['validation_embeddings_sha256'],
        'checkpoint_sha256': file_sha256(checkpoint_path),
        'gate': bundle['result']['gate']['g'],
        'metrics': metrics,
    }
    readout_path = run / 'DEVELOPMENT_READOUT.json'
    readout_path.write_text(json.dumps(
        readout, indent=2, sort_keys=True, allow_nan=False,
    ), encoding='utf-8')
    complete = {
        'schema_version': 1,
        'status': 'training_only_before_deferred_test_evaluation',
        'source_commit': SOURCE_COMMIT,
        'config_sha256': file_sha256(run / 'config.json'),
        'checkpoint_sha256': file_sha256(checkpoint_path),
        'adaptive_freeze_sha256': file_sha256(freeze_path),
        'event_checkpoint_sha256': [file_sha256(path) for path in events],
        'validation_manifest_sha256': file_sha256(saved_manifest),
        'data_flow_audit_sha256': file_sha256(audit_path),
        'development_readout_sha256': file_sha256(readout_path),
    }
    (run / 'PILOT_TRAINING_ONLY_COMPLETE.json').write_text(json.dumps(
        complete, indent=2, sort_keys=True,
    ), encoding='utf-8')
    print(json.dumps({
        'status': complete['status'],
        'seed': cli.seed,
        'parameters': readout['parameters'],
        'cil': metrics['cil']['accuracy'],
        'til': metrics['til']['accuracy'],
        'old_cil': metrics['old_cil']['accuracy'],
        'new_cil': metrics['new_cil']['accuracy'],
        'nll': metrics['nll']['mean'],
        'gate': readout['gate'],
        'run': str(run),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
