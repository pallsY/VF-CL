"""Read-only ISOLET task-time head-memory capacity curve."""

import argparse
import copy
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from adaptive_consolidation_audit import _safe_torch_load, trainer_state_sha256
from analyze_isolet_frozen_head_capacity import encode_loader, score_top
from analyze_isolet_head_branches import sha256
from determinism import derive_seed
from head_consolidation import consolidate_classifier, hash_top_state
import launch_isolet_replay_selection_pilot as pilot
from models import TopModel, build_models
from vfl_trainer import VFLTrainer

SOURCE_COMMIT = 'a575bbf446ae501cf8e580ba62c2cc5492a25f30'
SOURCE_CONFIG_SHA256 = '5702b8846ca8e9d279728a725550c1b053b8580fec832a9695b56d6ea7f2af34'
SOURCE_RECORD_SHA256 = '7dc5bf930600d097134948d6ee166fd55490327d16cde1fe2496ac374a73299b'
LAUNCHER_SHA256 = '419dc05641a90bc4834ae9ca4f4bbb8079a8226df1226c7b72019a691cd3043f'
DATA_SHA256 = {
    'data:isolet/isolet_vfl.npz': 'd34312670de93198afcd2b126c95b79bae2b4cffeb30d480f097faf046b69514',
    'data:isolet/isolet_vfl.metadata.json': '79396dea1751b6094a5769f2dd789ad58b3ea9582a12c8c98d3ec07d8d1eb3cd',
}


def validate_provenance_fields(protocol, done, config, seed, capacity, audit_rows):
    if (protocol.get('source_commit') != SOURCE_COMMIT
            or protocol.get('source_config_sha256') != SOURCE_CONFIG_SHA256
            or protocol.get('source_record_sha256') != SOURCE_RECORD_SHA256
            or protocol.get('launcher_sha256') != LAUNCHER_SHA256
            or protocol.get('data_sha256') != DATA_SHA256
            or protocol.get('holdout_split_seed') != 20261012
            or protocol.get('selector') != 'herding'
            or protocol.get('raw_replay_capacity_per_class') != capacity
            or protocol.get('adaptive_contract_capacity_override') is not (capacity != 20)
            or done.get('status') != 'training_only_before_test'
            or config.get('seed') != seed
            or config.get('head_consolidation_samples_per_class') != capacity
            or config.get('formal_deferred_evaluation') is not False
            or tuple(config.get(k) for k in (
                'proto_lambda_a', 'distill_weight', 'feat_distill_weight'
            )) != (0.05, 0.10, 0.02)
            or any(row.get('split') == 'test'
                   or str(row.get('loader_key', '')).startswith("('test'")
                   for row in audit_rows)):
        raise ValueError('producer, protocol, config or test-access evidence differs')

def memory_stats(replay):
    if not replay or any(not isinstance(rows, torch.Tensor) or rows.ndim < 2
                         for rows in replay.values()):
        raise ValueError('raw replay must contain per-class tensors')
    counts = {int(class_id): int(rows.size(0)) for class_id, rows in replay.items()}
    if len(set(counts.values())) != 1 or min(counts.values()) <= 0:
        raise ValueError('raw replay is not class-balanced')
    return {'classes': len(replay), 'examples_per_class': next(iter(counts.values())),
            'total_examples': sum(counts.values()),
            'raw_bytes': sum(rows.numel() * rows.element_size()
                             for rows in replay.values())}


def passes_capacity(paired):
    return (len(paired) == 2
            and all(row['cil_delta'] > 0 and row['new_cil_delta'] >= -0.01
                    for row in paired)
            and statistics.fmean(row['cil_delta'] for row in paired) >= 0.01)


def one_run(root, seed, capacity):
    run = root / f'seed{seed}-cap{capacity}' / 'run'
    done = json.loads((run / 'PILOT_COMPLETE.json').read_text())
    protocol = json.loads((run / 'PILOT_PROTOCOL.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    checkpoint_path = run / 'adaptive_final.pt'
    audit_path = run / 'data_flow_audit.jsonl'
    audit_rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
    validate_provenance_fields(protocol, done, config, seed, capacity, audit_rows)
    events = [run / 'checkpoints' / f'event_{index}_CIL.pt'
              for index in range(13)]
    if (sha256(pilot.__file__) != LAUNCHER_SHA256
            or (run.parent.parent / f'seed{seed}-cap{capacity}.exit.code').read_text().strip() != '0'
            or sha256(run / 'config.json') != done['config_sha256']
            or sha256(audit_path) != done['data_flow_audit_sha256']
            or sha256(run / 'validation' / 'validation_manifest.json')
            != done['gate_manifest_sha256']
            or sha256(run / 'pilot_holdout' / 'manifest.json')
            != done['holdout_manifest_sha256']
            or [sha256(path) for path in events] != done['event_checkpoint_sha256']
            or sha256(checkpoint_path) != done['checkpoint_sha256']
            or (run / 'results.json').exists()):
        raise ValueError('completed source artifact identity differs')
    payload = _safe_torch_load(checkpoint_path)
    bundle = payload['cl_state']['adaptive_audit_bundle']
    raw = payload['cl_state']['head_raw_replay']
    stored = memory_stats(raw)
    if (stored['classes'] != 26 or stored['examples_per_class'] != capacity
            or set(bundle['replay_embeddings']) != set(raw)
            or any(bundle['replay_embeddings'][c].size(0) != capacity
                   for c in raw)
            or hash_top_state(bundle['installed_state'])
            != hash_top_state(payload['trainer_state']['top_model'])):
        raise ValueError('task-time raw replay or frozen head differs')
    pretrainer = trainer_state_sha256({
        'bottoms': payload['trainer_state']['bottoms'],
        'top_model': bundle['pre_state'],
    })
    pilot.HOLDOUT_SEED = 20261012
    with tempfile.TemporaryDirectory(prefix='isolet-online-memory-') as scratch:
        args = SimpleNamespace(**config)
        args.output_dir = scratch
        dataset = pilot.PilotDataset(args)
        if (sha256(Path(scratch) / 'pilot_holdout' / 'manifest.json')
                != done['holdout_manifest_sha256']
                or sha256(Path(scratch) / 'validation' / 'validation_manifest.json')
                != done['gate_manifest_sha256']
                or dataset.validation_indices & dataset.holdout_indices):
            raise ValueError('gate/holdout cohort differs')
        bottoms, installed = build_models(args)
        trainer = VFLTrainer(bottoms, installed, args)
        trainer.load_state(payload['trainer_state'])
        for model in (*trainer.bottoms, trainer.top_model):
            model.eval()
        holdout_x, holdout_y = encode_loader(
            trainer, args, dataset.holdout_loader(list(range(26))),
        )
    if holdout_y.numel() != 1040:
        raise ValueError('holdout sample count differs')
    pre = TopModel(holdout_x.size(1), 26, cosine=bool(config['cosine_head']))
    pre.load_state_dict(bundle['pre_state'], strict=True)
    pre = pre.to(args.device).eval()
    fit_seed = derive_seed(seed, 'head_consolidation', 12)
    full20 = copy.deepcopy(pre)
    torch.cuda.synchronize()
    started = time.perf_counter()
    consolidate_classifier(
        full20, payload['cl_state']['global_protos'],
        0.01, 500, 0.01, 20, fit_seed, args.device,
        replay_embeddings=bundle['replay_embeddings'],
        replay_source='task_time_raw_replay_first20',
        persistent_raw_example_count=capacity * 26,
    )
    torch.cuda.synchronize()
    first20_fit_seconds = time.perf_counter() - started
    first20_hash = hash_top_state(full20)
    if first20_hash != bundle['result']['candidate_hashes']['full']:
        raise ValueError('frozen Full20 candidate was not reproduced')
    first20 = score_top(full20, holdout_x, holdout_y, bundle['task_classes'])
    if capacity == 20:
        fitted, fitted_seconds = full20, first20_fit_seconds
    else:
        fitted = copy.deepcopy(pre)
        torch.cuda.synchronize()
        started = time.perf_counter()
        consolidate_classifier(
            fitted, payload['cl_state']['global_protos'],
            0.01, 500, 0.01, capacity, fit_seed, args.device,
            replay_embeddings=bundle['replay_embeddings'],
            replay_source='task_time_raw_replay_full_capacity',
            persistent_raw_example_count=capacity * 26,
        )
        torch.cuda.synchronize()
        fitted_seconds = time.perf_counter() - started
    metrics = score_top(fitted, holdout_x, holdout_y, bundle['task_classes'])
    confusion = metrics['task_confusion']
    old_to_old = sum(confusion[i][j] for i in range(12)
                     for j in range(12) if i != j)
    return {
        'seed': seed, 'capacity': capacity,
        'checkpoint_sha256': sha256(checkpoint_path),
        'config_sha256': sha256(run / 'config.json'),
        'gate_manifest_sha256': done['gate_manifest_sha256'],
        'holdout_manifest_sha256': done['holdout_manifest_sha256'],
        'pre_final_head_trainer_sha256': pretrainer,
        'raw_first20_sha256': hash_top_state({
            str(class_id): raw[class_id][:20] for class_id in sorted(raw)
        }),
        'full20_sha256': first20_hash,
        'fitted_head_sha256': hash_top_state(fitted),
        'first20_fit_seconds': first20_fit_seconds,
        'capacity_fit_seconds': fitted_seconds,
        'persistent_replay': stored,
        'first20': first20, 'metrics': metrics,
        'old_to_old_task_errors': old_to_old,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    cli = parser.parse_args()
    root = cli.root.resolve(strict=True)
    import models
    source_root = Path(models.__file__).resolve().parent
    if (subprocess.check_output(['git', '-C', str(source_root), 'rev-parse', 'HEAD'], text=True).strip()
            != SOURCE_COMMIT
            or subprocess.check_output(['git', '-C', str(source_root), 'status', '--porcelain'], text=True)):
        raise ValueError('analysis model source is not the exact clean producer')
    output = cli.output.resolve()
    if output.exists() or root in output.parents:
        raise ValueError('analysis output must be new and outside source runs')
    rows = [one_run(root, seed, capacity)
            for seed in (51, 52) for capacity in (20, 40, 80)]
    prefix_numerical_differences = {}
    for seed in (51, 52):
        paired = [row for row in rows if row['seed'] == seed]
        control = paired[0]
        if (len({row['gate_manifest_sha256'] for row in paired}) != 1
                or len({row['holdout_manifest_sha256'] for row in paired}) != 1
                or len({row['pre_final_head_trainer_sha256'] for row in paired}) != 1
                or len({row['raw_first20_sha256'] for row in paired}) != 1
                or any(row['first20']['cil'] != control['first20']['cil']
                       or row['first20']['task_id'] != control['first20']['task_id']
                       or abs(row['first20']['nll'] - control['first20']['nll']) > 1e-6
                       for row in paired)):
            raise ValueError('capacity runs do not share paired training/first20 identity')
        base_replay = _safe_torch_load(
            root / f'seed{seed}-cap20' / 'run' / 'adaptive_final.pt'
        )['cl_state']['adaptive_audit_bundle']['replay_embeddings']
        for capacity in (40, 80):
            other_replay = _safe_torch_load(
                root / f'seed{seed}-cap{capacity}' / 'run' / 'adaptive_final.pt'
            )['cl_state']['adaptive_audit_bundle']['replay_embeddings']
            difference = max(float((base_replay[c] - other_replay[c][:20]).abs().max())
                             for c in range(26))
            if difference > 1e-6:
                raise ValueError('first20 re-encoding differs beyond numerical tolerance')
            prefix_numerical_differences[f'{seed}:{capacity}'] = difference
    comparisons = {}
    for capacity in (40, 80):
        pairs = []
        for seed in (51, 52):
            baseline = next(row for row in rows
                            if row['seed'] == seed and row['capacity'] == 20)
            candidate = next(row for row in rows
                             if row['seed'] == seed and row['capacity'] == capacity)
            pairs.append({
                'seed': seed,
                'cil_delta': candidate['metrics']['cil'] - baseline['metrics']['cil'],
                'new_cil_delta': (candidate['metrics']['new_cil']
                                  - baseline['metrics']['new_cil']),
                'task_id_delta': (candidate['metrics']['task_id']
                                  - baseline['metrics']['task_id']),
                'old_to_old_error_delta': (candidate['old_to_old_task_errors']
                                           - baseline['old_to_old_task_errors']),
            })
        comparisons[str(capacity)] = {
            'paired': pairs,
            'mean_cil_gain': statistics.fmean(p['cil_delta'] for p in pairs),
            'passed': passes_capacity(pairs),
        }
    result = {
        'schema_version': 1, 'status': 'online_task_time_memory_development_screen',
        'source_root': str(root),
        'decision_rule': 'both CIL deltas > 0, mean >= 0.01, each newest delta >= -0.01',
        'runs': rows, 'comparisons': comparisons,
        'first20_reencoding_max_abs_diff': prefix_numerical_differences,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(comparisons, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
