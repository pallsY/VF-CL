"""Compare task-time stochastic and deterministic herding on one final encoder."""

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset

import analyze_online_offline_herding_gap as feature_audit
import calibration_split
import cifar_replay_id_match
import cl_methods.proto_evolve as proto_evolve
import data_utils
import determinism
import launch_cifar_selection_view_pilot as launcher
import models
import vfl_trainer
from adaptive_consolidation_audit import _safe_torch_load
from cifar_replay_id_match import CifarReplayMatcher
from cl_methods.proto_evolve import herding_indices
from data_utils import VFLDataset


SOURCE_COMMIT = launcher.SOURCE_COMMIT
FEATURE_AUDIT_SHA256 = 'c980e3347becc75984f91009aeccfe50fac8ffa87a022d43502690e194a6f5c3'
DATA_KEYS = ('data:cifar-100-python/train',
             'data:cifar-100-python/test',
             'data:cifar-100-python/meta')


def _paired_summary(rows):
    stochastic = [row['stochastic'] for row in rows]
    deterministic = [row['deterministic'] for row in rows]
    if not all(isinstance(value, (int, float)) and math.isfinite(value)
               and value >= 0 for value in stochastic + deterministic):
        raise ValueError('malformed paired centroid error')
    differences = [s - d for s, d in zip(stochastic, deterministic)]
    result = {
        'stochastic_mean': statistics.fmean(stochastic),
        'deterministic_mean': statistics.fmean(deterministic),
        'stochastic_median': statistics.median(stochastic),
        'deterministic_median': statistics.median(deterministic),
        's_minus_d_mean': statistics.fmean(differences),
        's_minus_d_median': statistics.median(differences),
        'fraction_s_worse': sum(value > 0 for value in differences) / len(rows),
        'fraction_s_better': sum(value < 0 for value in differences) / len(rows),
    }
    if all('offline' in row for row in rows):
        offline = [row['offline'] for row in rows]
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   and value >= 0 for value in offline):
            raise ValueError('malformed offline centroid error')
        residual = [d - o for d, o in zip(deterministic, offline)]
        result.update(
            offline_mean=statistics.fmean(offline),
            offline_median=statistics.median(offline),
            d_minus_o_mean=statistics.fmean(residual),
            d_minus_o_median=statistics.median(residual),
            fraction_d_worse_offline=sum(value > 0 for value in residual) / len(rows),
        )
    elif any('offline' in row for row in rows):
        raise ValueError('inconsistent offline panel')
    return result


def summarize_classes(rows):
    if len(rows) != 100:
        raise ValueError('expected 100 class rows')
    if any(len(row['final_parties']) != 4
           or not isinstance(row['content_overlap_count'], int)
           or not 0 <= row['content_overlap_count'] <= 20 for row in rows):
        raise ValueError('party or overlap panel is malformed')
    return {
        'classes': 100,
        'selection_time': _paired_summary([row['selection_time'] for row in rows]),
        'final_aggregate': _paired_summary([row['final_aggregate'] for row in rows]),
        'final_parties': [
            _paired_summary([row['final_parties'][party] for row in rows])
            for party in range(4)
        ],
        'content_overlap_total': sum(row['content_overlap_count'] for row in rows),
        'content_overlap_mean': statistics.fmean(
            row['content_overlap_count'] for row in rows),
    }


def _verify_inputs(run, source_config, source_record):
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
            != SOURCE_COMMIT
            or subprocess.check_output(['git', 'status', '--porcelain'], text=True)):
        raise ValueError('analysis requires exact clean producer checkout')
    if (launcher.file_sha256(source_record) != launcher.SOURCE_RECORD_SHA256
            or launcher.file_sha256(source_config) != launcher.SOURCE_CONFIG_SHA256
            or launcher.file_sha256(feature_audit.__file__) != FEATURE_AUDIT_SHA256):
        raise ValueError('formal source or embedding helper differs from locked input')
    record = json.loads(source_record.read_text(encoding='utf-8'))
    if (record.get('kind') != 'formal_completed_run'
            or record.get('method') != 'adaptive'
            or record.get('seed') != 42
            or record.get('source_commit') != SOURCE_COMMIT):
        raise ValueError('formal source record identity mismatch')
    source_modules = {
        name: launcher.file_sha256(Path(module.__file__))
        for name, module in (
            ('calibration_split.py', calibration_split),
            ('cl_methods/proto_evolve.py', proto_evolve),
            ('data_utils.py', data_utils),
            ('determinism.py', determinism),
            ('models.py', models),
            ('vfl_trainer.py', vfl_trainer),
        )
    }
    if any(digest != record['artifact_sha256']['source:' + name]
           for name, digest in source_modules.items()):
        raise ValueError('producer module differs from audited source')
    protocol_path = run / 'PILOT_PROTOCOL.json'
    complete_path = run / 'PILOT_TRAINING_ONLY_COMPLETE.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    complete = json.loads(complete_path.read_text(encoding='utf-8'))
    config_path = run / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    expected, overrides = launcher.derive_config(
        json.loads(source_config.read_text(encoding='utf-8')), run.parent,
    )
    if (config != expected or Path(config['output_dir']).resolve() != run
            or protocol['source_commit'] != SOURCE_COMMIT
            or protocol['overrides'] != overrides
            or protocol['source_record_sha256'] != launcher.SOURCE_RECORD_SHA256
            or protocol['source_config_sha256'] != launcher.SOURCE_CONFIG_SHA256
            or protocol['launcher_sha256'] != launcher.file_sha256(launcher.__file__)
            or protocol['planned_stop'] != 'before_deferred_test_evaluation'
            or complete['status'] != 'training_only_before_deferred_test_evaluation'
            or complete['source_commit'] != SOURCE_COMMIT
            or (run / 'results.json').exists()):
        raise ValueError('training-only protocol/completion mismatch')
    data_hashes = {
        key: launcher.file_sha256(Path(config['data_path']) / key.split(':', 1)[1])
        for key in DATA_KEYS
    }
    if (data_hashes != protocol['source_data_sha256']
            or any(digest != record['artifact_sha256'][key]
                   for key, digest in data_hashes.items())):
        raise ValueError('CIFAR data differ from audited source')
    artifacts = {
        'config_sha256': config_path,
        'checkpoint_sha256': run / 'adaptive_final.pt',
        'adaptive_freeze_sha256': run / 'ADAPTIVE_STATE_FROZEN.json',
        'bic_manifest_sha256': run / 'bic' / 'calibration_manifest.json',
        'validation_manifest_sha256': run / 'validation' / 'validation_manifest.json',
        'data_flow_audit_sha256': run / 'data_flow_audit.jsonl',
    }
    if any(launcher.file_sha256(path) != complete[key]
           for key, path in artifacts.items()):
        raise ValueError('training-only artifact hash mismatch')
    events = [run / 'checkpoints' / f'event_{task}_CIL.pt'
              for task in range(10)]
    if ([launcher.file_sha256(path) for path in events]
            != complete['event_checkpoint_sha256']):
        raise ValueError('task checkpoint hash mismatch')
    accesses = [json.loads(line) for line in (run / 'data_flow_audit.jsonl').read_text(
        encoding='utf-8',
    ).splitlines()]
    if any(entry.get('split') == 'test'
           or str(entry.get('loader_key', '')).startswith("('test'")
           for entry in accesses):
        raise ValueError('training run accessed test loader')
    return config, complete, data_hashes, source_modules


def _embed_state(state, dataset, indices, args):
    bottoms, top = models.build_models(args)
    trainer = vfl_trainer.VFLTrainer(bottoms, top, args)
    trainer.load_state(state)
    for bottom in trainer.bottoms:
        bottom.eval()
        for parameter in bottom.parameters():
            parameter.requires_grad_(False)
    before = feature_audit.bottom_state_sha256(trainer.bottoms)
    aggregate, parties, labels = feature_audit.embed_batches(
        trainer.bottoms,
        DataLoader(Subset(dataset.validationset, indices),
                   batch_size=64, shuffle=False, num_workers=0),
        args,
    )
    if feature_audit.bottom_state_sha256(trainer.bottoms) != before:
        raise RuntimeError('bottom parameters changed during embedding')
    return aggregate, parties, labels, before


def _hash_indices(indices):
    return hashlib.sha256(json.dumps(
        indices, separators=(',', ':'),
    ).encode()).hexdigest()


def _content_overlap(data, stochastic_ids, deterministic_ids):
    def counts(ids):
        return Counter(hashlib.sha256(data[index].tobytes()).digest()
                       for index in ids)
    return sum((counts(stochastic_ids) & counts(deterministic_ids)).values())


def _analyze(run, config, device):
    final_checkpoint = _safe_torch_load(run / 'adaptive_final.pt')
    if (final_checkpoint.get('schema_version') != 1
            or final_checkpoint.get('kind') != 'adaptive_final_checkpoint'
            or final_checkpoint['provenance']['source_commit'] != SOURCE_COMMIT):
        raise ValueError('unexpected final Adaptive checkpoint')
    saved = {int(class_id): value for class_id, value in
             final_checkpoint['cl_state']['head_raw_replay'].items()}
    if (set(saved) != set(range(100))
            or any(not isinstance(value, torch.Tensor)
                   or value.shape != (20, 3, 32, 32)
                   or not bool(torch.isfinite(value).all())
                   for value in saved.values())):
        raise ValueError('final online replay is not exactly 20/class')

    with tempfile.TemporaryDirectory(prefix='vfcl-selection-view-') as scratch:
        args = SimpleNamespace(**config)
        args.output_dir = scratch
        args.device = device
        args.num_workers = 0
        args.data_flow_audit = 0
        dataset = VFLDataset(args)
        if (dataset.calibration_manifest != json.loads((
                run / 'bic' / 'calibration_manifest.json').read_text(encoding='utf-8'))
                or dataset.validation_manifest != json.loads((
                    run / 'validation' / 'validation_manifest.json').read_text(encoding='utf-8'))):
            raise ValueError('rebuilt holdouts differ from training run')
        targets = dataset.validationset.targets
        heldout = dataset.calibration_indices | dataset.validation_indices
        available = [index for index in range(len(targets)) if index not in heldout]
        if (len(available) != 45000
                or Counter(int(targets[index]) for index in available)
                != {class_id: 450 for class_id in range(100)}):
            raise ValueError('full training candidate pool is not 450/class')
        eligible = [index for class_id in range(100) for index in available
                    if int(targets[index]) == class_id]
        candidates = [eligible[class_id * 450:(class_id + 1) * 450]
                      for class_id in range(100)]

        stochastic_ids, duplicate_matches = [], 0
        for class_id in range(100):
            matcher = CifarReplayMatcher(dataset.validationset.data,
                                         candidates[class_id])
            selected = [matcher.recover_id(raw) for raw in saved[class_id]]
            if len(selected) != 20 or set(selected) & heldout:
                raise ValueError('stochastic replay is outside training pool')
            stochastic_ids.append(selected)
            duplicate_matches += matcher.identical_duplicate_matches

        deterministic_ids = [None] * 100
        selection_errors = [None] * 100
        event_bottom_hashes = []
        for task_id in range(10):
            checkpoint = _safe_torch_load(
                run / 'checkpoints' / f'event_{task_id}_CIL.pt',
            )
            classes = list(range(task_id * 10, task_id * 10 + 10))
            if (checkpoint.get('schema_version') != 4
                    or checkpoint.get('event_idx') != task_id
                    or checkpoint.get('task_id') != task_id
                    or checkpoint.get('step') != f'event_{task_id}_CIL'
                    or checkpoint.get('new_classes') != classes):
                raise ValueError('task checkpoint identity mismatch')
            event_replay = {int(class_id): value for class_id, value in
                            checkpoint['cl_state']['head_raw_replay'].items()}
            if (set(event_replay) != set(range((task_id + 1) * 10))
                    or any(not torch.equal(event_replay[class_id], saved[class_id])
                           for class_id in classes)):
                raise ValueError('task-time stochastic replay differs from final')
            task_indices = eligible[task_id * 4500:(task_id + 1) * 4500]
            aggregate, _parties, labels, bottom_hash = _embed_state(
                checkpoint['trainer_state'], dataset, task_indices, args,
            )
            event_bottom_hashes.append(bottom_hash)
            if any(not bool((labels[local * 450:(local + 1) * 450]
                             == class_id).all())
                   for local, class_id in enumerate(classes)):
                raise ValueError('task feature labels are misaligned')
            for local, class_id in enumerate(classes):
                full = aggregate[local * 450:(local + 1) * 450]
                ids = candidates[class_id]
                positions = {index: position for position, index in enumerate(ids)}
                stochastic_local = torch.tensor(
                    [positions[index] for index in stochastic_ids[class_id]],
                    dtype=torch.long,
                )
                deterministic_local = herding_indices(full, 20)
                if (deterministic_local.numel() != 20
                        or deterministic_local.unique().numel() != 20):
                    raise ValueError('task-time herding selection is malformed')
                deterministic_ids[class_id] = [ids[int(index)]
                                               for index in deterministic_local]
                selection_errors[class_id] = {
                    'stochastic': feature_audit.normalized_centroid_error(
                        full, full.index_select(0, stochastic_local)),
                    'deterministic': feature_audit.normalized_centroid_error(
                        full, full.index_select(0, deterministic_local)),
                }
            del checkpoint

        full_agg, full_parties, labels, final_bottom_hash = _embed_state(
            final_checkpoint['trainer_state'], dataset, eligible, args,
        )
        if (final_bottom_hash != event_bottom_hashes[-1]
                or any(not bool((labels[class_id * 450:(class_id + 1) * 450]
                                 == class_id).all()) for class_id in range(100))):
            raise ValueError('final/task-9 bottom state or class rows differ')

        rows, offline_ids = [], []
        for class_id in range(100):
            full_slice = slice(class_id * 450, (class_id + 1) * 450)
            ids = candidates[class_id]
            positions = {index: position for position, index in enumerate(ids)}
            stochastic_local = torch.tensor(
                [positions[index] for index in stochastic_ids[class_id]],
                dtype=torch.long,
            )
            deterministic_local = torch.tensor(
                [positions[index] for index in deterministic_ids[class_id]],
                dtype=torch.long,
            )
            offline_local = herding_indices(full_agg[full_slice], 20)
            if (offline_local.numel() != 20
                    or offline_local.unique().numel() != 20):
                raise ValueError('final offline selection is malformed')
            offline_ids.extend(ids[int(index)] for index in offline_local)

            def errors(full):
                return {
                    'stochastic': feature_audit.normalized_centroid_error(
                        full, full.index_select(0, stochastic_local)),
                    'deterministic': feature_audit.normalized_centroid_error(
                        full, full.index_select(0, deterministic_local)),
                    'offline': feature_audit.normalized_centroid_error(
                        full, full.index_select(0, offline_local)),
                }

            rows.append({
                'class_id': class_id,
                'task_id': class_id // 10,
                'memory_per_selector': 20,
                'candidate_count': 450,
                'selection_time': selection_errors[class_id],
                'final_aggregate': errors(full_agg[full_slice]),
                'final_parties': [errors(full_parties[party][full_slice])
                                  for party in range(4)],
                'content_overlap_count': _content_overlap(
                    dataset.validationset.data,
                    stochastic_ids[class_id], deterministic_ids[class_id],
                ),
            })
        if getattr(dataset, '_first_accessed_splits', set()):
            raise RuntimeError('analysis accessed a protected evaluation loader')
    flat_stochastic = [index for class_ids in stochastic_ids for index in class_ids]
    flat_deterministic = [index for class_ids in deterministic_ids for index in class_ids]
    if (len(flat_stochastic) != 2000 or len(flat_deterministic) != 2000
            or len(set(flat_deterministic)) != 2000
            or len(offline_ids) != 2000 or len(set(offline_ids)) != 2000):
        raise ValueError('selection memory count is malformed')
    return {
        'eligible_indices_sha256': _hash_indices(eligible),
        'stochastic_ids_sha256': _hash_indices(flat_stochastic),
        'deterministic_ids_sha256': _hash_indices(flat_deterministic),
        'offline_ids_sha256': _hash_indices(offline_ids),
        'identical_duplicate_matches': duplicate_matches,
        'event_bottom_state_sha256': event_bottom_hashes,
        'final_bottom_state_sha256': final_bottom_hash,
        'summary': summarize_classes(rows),
        'classes': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--source-config', required=True, type=Path)
    parser.add_argument('--source-record', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    cli = parser.parse_args()
    run = cli.run.resolve(strict=True)
    source_config = cli.source_config.resolve(strict=True)
    source_record = cli.source_record.resolve(strict=True)
    output = cli.output_dir.resolve()
    if output.exists() or output == run or run in output.parents:
        raise ValueError('analysis output must be a new root outside training run')
    config, complete, data_hashes, modules = _verify_inputs(
        run, source_config, source_record,
    )
    result = _analyze(run, config, cli.device)
    payload = {
        'schema_version': 1,
        'status': 'training_only_task_time_selection_view_pilot',
        'seed': 49,
        'run': str(run),
        'source_commit': SOURCE_COMMIT,
        'source_record_sha256': launcher.SOURCE_RECORD_SHA256,
        'source_config_sha256': launcher.SOURCE_CONFIG_SHA256,
        'pilot_protocol_sha256': launcher.file_sha256(run / 'PILOT_PROTOCOL.json'),
        'pilot_completion_sha256': launcher.file_sha256(
            run / 'PILOT_TRAINING_ONLY_COMPLETE.json'),
        'checkpoint_sha256': complete['checkpoint_sha256'],
        'data_sha256': data_hashes,
        'source_modules_sha256': modules,
        'script_sha256': launcher.file_sha256(__file__),
        'matcher_sha256': launcher.file_sha256(cifar_replay_id_match.__file__),
        'feature_audit_sha256': FEATURE_AUDIT_SHA256,
        **result,
    }
    output.mkdir(parents=True, exist_ok=False)
    temporary = output / 'selection_view_herding.json.tmp'
    with open(temporary, 'x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output / 'selection_view_herding.json')
    print(json.dumps({
        'summary': payload['summary'],
        'identical_duplicate_matches': payload['identical_duplicate_matches'],
        'stochastic_ids_sha256': payload['stochastic_ids_sha256'],
        'deterministic_ids_sha256': payload['deterministic_ids_sha256'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
