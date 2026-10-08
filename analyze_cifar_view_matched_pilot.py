"""Separate CIFAR replay-view error from the optimistic offline herding gap."""

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
from torch.utils.data import DataLoader, Subset, TensorDataset

import analyze_online_offline_herding_gap as previous_audit
import calibration_split
import cifar_replay_id_match
import cl_methods.proto_evolve as proto_evolve
import data_utils
import determinism
import launch_cifar_view_matched_pilot as launcher
import models
import vfl_trainer
from adaptive_consolidation_audit import _safe_torch_load
from cifar_replay_id_match import CifarReplayMatcher
from cl_methods.proto_evolve import herding_indices
from data_utils import VFLDataset


SOURCE_COMMIT = launcher.SOURCE_COMMIT
SOURCE_RECORD_SHA256 = 'aad02da1ba8e87f29e79f5702e71fe8e99c8ef4cdf554976cc1b881af6dc0871'
SOURCE_CONFIG_SHA256 = '5ef9984e091b83d7f84ac373cd702811c0ed15595dc6f4075ca23cc515535955'
PREVIOUS_AUDIT_SHA256 = 'c980e3347becc75984f91009aeccfe50fac8ffa87a022d43502690e194a6f5c3'
DATA_KEYS = ('data:cifar-100-python/train',
             'data:cifar-100-python/test',
             'data:cifar-100-python/meta')


def _summary_for_space(triplets):
    values = {key: [row[key] for row in triplets]
              for key in ('augmented', 'matched', 'offline')}
    if not all(isinstance(value, (int, float)) and math.isfinite(value)
               and value >= 0
               for series in values.values() for value in series):
        raise ValueError('malformed centroid error')
    view = [a - b for a, b in zip(values['augmented'], values['matched'])]
    remaining = [b - c for b, c in zip(values['matched'], values['offline'])]
    return {
        **{f'{key}_mean': statistics.fmean(series)
           for key, series in values.items()},
        **{f'{key}_median': statistics.median(series)
           for key, series in values.items()},
        'view_gap_mean': statistics.fmean(view),
        'view_gap_median': statistics.median(view),
        'remaining_gap_mean': statistics.fmean(remaining),
        'remaining_gap_median': statistics.median(remaining),
        'fraction_view_gap_positive': sum(value > 0 for value in view) / len(view),
        'fraction_remaining_gap_positive': sum(value > 0 for value in remaining) / len(remaining),
    }


def summarize_classes(rows):
    if len(rows) != 100:
        raise ValueError('expected 100 class rows')
    if any(len(row['parties']) != 4 for row in rows):
        raise ValueError('expected four party errors per class')
    return {
        'classes': 100,
        'aggregate': _summary_for_space([row['aggregate'] for row in rows]),
        'parties': [
            _summary_for_space([row['parties'][party] for row in rows])
            for party in range(4)
        ],
    }


def _verify_source(run, source_config, source_record):
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
            != SOURCE_COMMIT
            or subprocess.check_output(['git', 'status', '--porcelain'], text=True)):
        raise ValueError('analysis requires exact clean producer checkout')
    if (launcher.file_sha256(source_record) != SOURCE_RECORD_SHA256
            or launcher.file_sha256(source_config) != SOURCE_CONFIG_SHA256):
        raise ValueError('formal source record/config are not the published audit inputs')
    record = json.loads(source_record.read_text(encoding='utf-8'))
    if (record.get('kind') != 'formal_completed_run'
            or record.get('method') != 'adaptive'
            or record.get('seed') != 42
            or record.get('source_commit') != SOURCE_COMMIT
            or launcher.file_sha256(source_config) != record['artifact_sha256']['config']):
        raise ValueError('formal source identity differs')
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
    if any(value != record['artifact_sha256']['source:' + name]
           for name, value in source_modules.items()):
        raise ValueError('imported producer module differs from formal source')
    if (launcher.file_sha256(previous_audit.__file__) != PREVIOUS_AUDIT_SHA256
            or launcher.file_sha256(launcher.__file__)
            != json.loads((run / 'PILOT_PROTOCOL.json').read_text())['launcher_sha256']):
        raise ValueError('imported analysis/launcher helper differs from locked version')
    return record, source_modules


def _verify_run(run, source_config, source_record, record):
    protocol_path = run / 'PILOT_PROTOCOL.json'
    complete_path = run / 'PILOT_TRAINING_ONLY_COMPLETE.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    complete = json.loads(complete_path.read_text(encoding='utf-8'))
    config_path = run / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    expected, changed = launcher.derive_config(
        json.loads(source_config.read_text(encoding='utf-8')), run.parent,
    )
    if config != expected or Path(config['output_dir']).resolve() != run:
        raise ValueError('pilot config differs from registered overrides')
    if (protocol['source_commit'] != SOURCE_COMMIT
            or protocol['overrides'] != changed
            or protocol['source_record_sha256'] != launcher.file_sha256(source_record)
            or protocol['source_config_sha256'] != launcher.file_sha256(source_config)
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
            or any(value != record['artifact_sha256'][key]
                   for key, value in data_hashes.items())):
        raise ValueError('pilot CIFAR payload differs from formal source')
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
        raise ValueError('pilot artifact hash mismatch')
    events = [run / 'checkpoints' / f'event_{task}_CIL.pt'
              for task in range(10)]
    if ([launcher.file_sha256(path) for path in events]
            != complete['event_checkpoint_sha256']):
        raise ValueError('pilot CIL event checkpoint hash mismatch')
    accesses = [json.loads(line) for line in artifacts['data_flow_audit_sha256'].read_text(
        encoding='utf-8',
    ).splitlines()]
    if any(entry.get('split') == 'test'
           or str(entry.get('loader_key', '')).startswith("('test'")
           for entry in accesses):
        raise ValueError('pilot accessed test loader')
    return config, protocol, complete, data_hashes


def _analyze(run, config, device):
    checkpoint = _safe_torch_load(run / 'adaptive_final.pt')
    if (checkpoint.get('schema_version') != 1
            or checkpoint.get('kind') != 'adaptive_final_checkpoint'
            or checkpoint['provenance']['source_commit'] != SOURCE_COMMIT):
        raise ValueError('unexpected adaptive checkpoint schema/provenance')
    saved = {int(class_id): value for class_id, value in
             checkpoint['cl_state']['head_raw_replay'].items()}
    if (set(saved) != set(range(100))
            or any(not isinstance(value, torch.Tensor)
                   or value.shape != (20, 3, 32, 32)
                   or not bool(torch.isfinite(value).all())
                   for value in saved.values())):
        raise ValueError('final replay is not 20 raw tensors for each class')

    with tempfile.TemporaryDirectory(prefix='vfcl-view-matched-') as scratch:
        args = SimpleNamespace(**config)
        args.output_dir = scratch
        args.device = device
        args.num_workers = 0
        args.data_flow_audit = 0
        dataset = VFLDataset(args)
        if (dataset.validation_manifest != json.loads((
                run / 'validation' / 'validation_manifest.json').read_text(encoding='utf-8'))
                or dataset.calibration_manifest != json.loads((
                    run / 'bic' / 'calibration_manifest.json').read_text(encoding='utf-8'))):
            raise ValueError('rebuilt holdouts differ from training run')
        heldout = dataset.validation_indices | dataset.calibration_indices
        targets = dataset.validationset.targets
        available = [index for index in range(len(targets)) if index not in heldout]
        if (len(available) != 45000
                or Counter(int(targets[index]) for index in available)
                != {class_id: 450 for class_id in range(100)}):
            raise ValueError('full train-minus-holdouts pool is not 450/class')
        eligible = [index for class_id in range(100) for index in available
                    if int(targets[index]) == class_id]
        bottoms, top = models.build_models(args)
        trainer = vfl_trainer.VFLTrainer(bottoms, top, args)
        trainer.load_state(checkpoint['trainer_state'])
        for bottom in trainer.bottoms:
            bottom.eval()
            for parameter in bottom.parameters():
                parameter.requires_grad_(False)
        bottom_hash = previous_audit.bottom_state_sha256(trainer.bottoms)
        full_agg, full_party, full_labels = previous_audit.embed_batches(
            trainer.bottoms,
            DataLoader(Subset(dataset.validationset, eligible),
                       batch_size=64, shuffle=False, num_workers=0),
            args,
        )
        online_raw = torch.cat([saved[class_id] for class_id in range(100)])
        online_labels = torch.arange(100).repeat_interleave(20)
        online_agg, online_party, observed_online = previous_audit.embed_batches(
            trainer.bottoms,
            DataLoader(TensorDataset(online_raw, online_labels),
                       batch_size=64, shuffle=False, num_workers=0),
            args,
        )
        if (not torch.equal(observed_online, online_labels)
                or any(not bool((full_labels[c * 450:(c + 1) * 450] == c).all())
                       for c in range(100))):
            raise ValueError('feature rows are not aligned by class')

        rows, recovered_ids, offline_ids, duplicate_matches = [], [], [], 0
        for class_id in range(100):
            full_slice = slice(class_id * 450, (class_id + 1) * 450)
            online_slice = slice(class_id * 20, (class_id + 1) * 20)
            candidates = eligible[full_slice]
            matcher = CifarReplayMatcher(dataset.validationset.data, candidates)
            matched_ids = [matcher.recover_id(raw) for raw in saved[class_id]]
            if len(matched_ids) != 20 or set(matched_ids) & heldout:
                raise ValueError('recovered replay is outside eligible training pool')
            recovered_ids.extend(matched_ids)
            duplicate_matches += matcher.identical_duplicate_matches
            candidate_positions = {index: position for position, index in enumerate(candidates)}
            matched_local = torch.tensor(
                [candidate_positions[index] for index in matched_ids], dtype=torch.long,
            )
            class_full = full_agg[full_slice]
            offline_local = herding_indices(class_full, 20)
            if (offline_local.numel() != 20
                    or offline_local.unique().numel() != 20
                    or int(offline_local.min()) < 0
                    or int(offline_local.max()) >= 450):
                raise ValueError('offline herding selection is malformed')
            offline_ids.extend(candidates[int(local)] for local in offline_local)

            def errors(full, augmented):
                return {
                    'augmented': previous_audit.normalized_centroid_error(
                        full, augmented),
                    'matched': previous_audit.normalized_centroid_error(
                        full, full.index_select(0, matched_local)),
                    'offline': previous_audit.normalized_centroid_error(
                        full, full.index_select(0, offline_local)),
                }

            rows.append({
                'class_id': class_id,
                'online_count': 20,
                'candidate_count': 450,
                'offline_count': 20,
                'identical_duplicate_matches': matcher.identical_duplicate_matches,
                'aggregate': errors(class_full, online_agg[online_slice]),
                'parties': [errors(full_party[party][full_slice],
                                   online_party[party][online_slice])
                            for party in range(4)],
            })
        if (previous_audit.bottom_state_sha256(trainer.bottoms) != bottom_hash
                or getattr(dataset, '_first_accessed_splits', set())):
            raise RuntimeError('analysis changed model state or accessed protected loader')
    if (len(recovered_ids) != 2000 or len(offline_ids) != 2000
            or len(set(offline_ids)) != 2000):
        raise ValueError('recovered/offline selection count is malformed')
    digest = lambda ids: hashlib.sha256(json.dumps(
        ids, separators=(',', ':'),
    ).encode()).hexdigest()
    return {
        'bottom_state_sha256': bottom_hash,
        'eligible_indices_sha256': digest(eligible),
        'recovered_online_ids_sha256': digest(recovered_ids),
        'offline_selected_ids_sha256': digest(offline_ids),
        'identical_duplicate_matches': duplicate_matches,
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
    record, modules = _verify_source(run, source_config, source_record)
    config, protocol, complete, data_hashes = _verify_run(
        run, source_config, source_record, record,
    )
    result = _analyze(run, config, cli.device)
    payload = {
        'schema_version': 1,
        'status': 'training_only_view_matched_herding_pilot',
        'seed': 48,
        'run': str(run),
        'source_commit': SOURCE_COMMIT,
        'source_record_sha256': launcher.file_sha256(source_record),
        'source_config_sha256': launcher.file_sha256(source_config),
        'pilot_protocol_sha256': launcher.file_sha256(run / 'PILOT_PROTOCOL.json'),
        'pilot_completion_sha256': launcher.file_sha256(
            run / 'PILOT_TRAINING_ONLY_COMPLETE.json'),
        'checkpoint_sha256': complete['checkpoint_sha256'],
        'data_sha256': data_hashes,
        'source_modules_sha256': modules,
        'script_sha256': launcher.file_sha256(__file__),
        'matcher_sha256': launcher.file_sha256(cifar_replay_id_match.__file__),
        'previous_audit_sha256': PREVIOUS_AUDIT_SHA256,
        **result,
    }
    output.mkdir(parents=True, exist_ok=False)
    temporary = output / 'view_matched_herding.json.tmp'
    with open(temporary, 'x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output / 'view_matched_herding.json')
    print(json.dumps({
        'summary': payload['summary'],
        'identical_duplicate_matches': payload['identical_duplicate_matches'],
        'recovered_online_ids_sha256': payload['recovered_online_ids_sha256'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
