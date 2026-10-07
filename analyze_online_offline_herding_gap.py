"""Compare task-time and final-encoder herding replay without evaluation data."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

import analyze_head_offset_pilot
import calibration_split
import cl_methods.proto_evolve as proto_evolve
import data_utils
import determinism
import models
import vfl_trainer
from adaptive_consolidation_audit import _safe_torch_load
from analyze_head_offset_pilot import file_sha256, select_calibration_indices
from cl_methods.proto_evolve import herding_indices
from data_utils import VFLDataset, split_features
from models import build_models
from vfl_trainer import VFLTrainer


SOURCE_COMMIT = '7bfe6b1d724fb1206bc0053a9008126bad86332d'
HELPER_SHA256 = '346b3b95a5abe2f5d7c749bf2b4780b1fbbebd5a5ce7843ff43cab684b9682f9'
DATA_KEYS = ('data:cifar-100-python/train',
             'data:cifar-100-python/test',
             'data:cifar-100-python/meta')


def normalized_centroid_error(full_features, selected_features):
    """L2 error of selected normalized-feature mean against full class mean."""
    if (full_features.ndim != 2 or selected_features.ndim != 2
            or full_features.shape[1] != selected_features.shape[1]
            or full_features.shape[0] < selected_features.shape[0]
            or selected_features.shape[0] == 0
            or not torch.isfinite(full_features).all()
            or not torch.isfinite(selected_features).all()):
        raise ValueError('class feature matrices are malformed')
    full = F.normalize(full_features.float(), dim=1)
    chosen = F.normalize(selected_features.float(), dim=1)
    return float((chosen.mean(dim=0) - full.mean(dim=0)).norm())


def bottom_state_sha256(bottoms):
    digest = hashlib.sha256()
    for index, bottom in enumerate(bottoms):
        for name, value in sorted(bottom.state_dict().items()):
            digest.update(str(index).encode())
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def embed_batches(bottoms, loader, args):
    for bottom in bottoms:
        bottom.eval()
    aggregates, parties, labels = [], [[] for _ in bottoms], []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(args.device)
        parts = split_features(batch_x, args)
        embeddings = [bottom(part) for bottom, part in zip(bottoms, parts)]
        if (len(embeddings) != args.num_parties
                or not all(torch.isfinite(value).all() for value in embeddings)):
            raise ValueError('party embedding batch is malformed')
        aggregate = sum(embeddings) if args.aggregation == 'sum' else torch.cat(embeddings, dim=1)
        aggregates.append(aggregate.detach().cpu())
        for index, value in enumerate(embeddings):
            parties[index].append(value.detach().cpu())
        labels.append(batch_y.detach().cpu())
    return (torch.cat(aggregates),
            [torch.cat(values) for values in parties],
            torch.cat(labels))


def summarize_rows(rows):
    if len(rows) != 100:
        raise ValueError('expected exactly 100 paired class records')
    online = [row['online_aggregate_error'] for row in rows]
    offline = [row['offline_aggregate_error'] for row in rows]
    if not all(0 <= value < float('inf') for value in online + offline):
        raise ValueError('non-finite centroid error')
    parties = len(rows[0]['party_online_errors'])
    return {
        'classes': 100,
        'online_mean': statistics.fmean(online),
        'offline_mean': statistics.fmean(offline),
        'online_median': statistics.median(online),
        'offline_median': statistics.median(offline),
        'mean_difference': statistics.fmean(a - b for a, b in zip(online, offline)),
        'fraction_online_worse': sum(a > b for a, b in zip(online, offline)) / 100,
        'party_online_means': [statistics.fmean(row['party_online_errors'][p]
                                                  for row in rows)
                               for p in range(parties)],
        'party_offline_means': [statistics.fmean(row['party_offline_errors'][p]
                                                   for row in rows)
                                for p in range(parties)],
    }


def screen_seed(root, seed, device):
    key = f'cifar100%3Aadaptive%3A{seed}'
    record_path = root / 'records' / f'{key}.json'
    run = root / 'runs' / key
    record = json.loads(record_path.read_text(encoding='utf-8'))
    if (record.get('kind') != 'formal_completed_run'
            or record.get('source_commit') != SOURCE_COMMIT
            or record.get('method') != 'adaptive'
            or record.get('seed') != seed):
        raise ValueError('formal record identity mismatch')
    config_path = run / 'config.json'
    checkpoint_path = run / 'checkpoints' / 'formal_final.pt'
    validation_manifest_path = run / 'validation' / 'validation_manifest.json'
    for path, key_name in (
            (config_path, 'config'),
            (checkpoint_path, 'checkpoint'),
            (validation_manifest_path, 'validation_manifest')):
        if file_sha256(path) != record['artifact_sha256'][key_name]:
            raise ValueError('formal source artifact hash mismatch')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if (config.get('data') != 'cifar100' or config.get('seed') != seed
            or config.get('num_classes') != 100
            or config.get('num_parties') != 4
            or config.get('aggregation') != 'sum'
            or config.get('lambda_validation_enabled') != 1
            or config.get('bic_enabled') != 1):
        raise ValueError('unexpected CIFAR Adaptive source config')
    data_hashes = {
        key_name: file_sha256(Path(config['data_path']) / key_name.split(':', 1)[1])
        for key_name in DATA_KEYS
    }
    if any(digest != record['artifact_sha256'][key_name]
           for key_name, digest in data_hashes.items()):
        raise ValueError('CIFAR source payload hash mismatch')
    checkpoint = _safe_torch_load(checkpoint_path)
    saved = {int(class_id): value for class_id, value in
             checkpoint['cl_state']['head_raw_replay'].items()}
    if (set(saved) != set(range(100))
            or any(not isinstance(value, torch.Tensor)
                   or value.shape != (20, 3, 32, 32)
                   or not torch.isfinite(value).all()
                   for value in saved.values())):
        raise ValueError('online replay is not exactly 20 raw rows per class')

    with tempfile.TemporaryDirectory(prefix='vfcl-herding-gap-') as scratch:
        args = SimpleNamespace(**config)
        args.output_dir = scratch
        args.device = device
        args.num_workers = 0
        args.data_flow_audit = 0
        args.formal_deferred_evaluation = False
        dataset = VFLDataset(args)
        if (dataset.validation_manifest
                != json.loads(validation_manifest_path.read_text(encoding='utf-8'))
                or dataset.calibration_manifest
                != json.loads((run / 'bic' / 'calibration_manifest.json').read_text(
                    encoding='utf-8'))):
            raise ValueError('rebuilt holdouts differ from formal source')
        heldout = dataset.validation_indices | dataset.calibration_indices
        eligible = select_calibration_indices(
            dataset.validationset.targets, heldout, 450,
        )
        if len(eligible) != 45000 or set(eligible) & heldout:
            raise ValueError('eligible pool is not class-balanced train-only data')
        bottoms, top = build_models(args)
        trainer = VFLTrainer(bottoms, top, args)
        trainer.load_state(checkpoint['trainer_state'])
        for bottom in trainer.bottoms:
            bottom.eval()
            for parameter in bottom.parameters():
                parameter.requires_grad_(False)
        state_before = bottom_state_sha256(trainer.bottoms)
        pool_loader = DataLoader(
            Subset(dataset.validationset, eligible),
            batch_size=64, shuffle=False, num_workers=0,
        )
        full_agg, full_party, full_labels = embed_batches(
            trainer.bottoms, pool_loader, args,
        )
        online_raw = torch.cat([saved[class_id] for class_id in range(100)])
        online_labels = torch.arange(100).repeat_interleave(20)
        online_loader = DataLoader(
            TensorDataset(online_raw, online_labels),
            batch_size=64, shuffle=False, num_workers=0,
        )
        online_agg, online_party, observed_online = embed_batches(
            trainer.bottoms, online_loader, args,
        )
        if (not torch.equal(online_labels, observed_online)
                or any(not bool((full_labels[c * 450:(c + 1) * 450] == c).all())
                       for c in range(100))):
            raise ValueError('class labels are not aligned with feature rows')
        rows, offline_indices = [], []
        for class_id in range(100):
            full_slice = slice(class_id * 450, (class_id + 1) * 450)
            online_slice = slice(class_id * 20, (class_id + 1) * 20)
            class_full = full_agg[full_slice]
            selected_local = herding_indices(class_full, 20)
            if (selected_local.numel() != 20
                    or selected_local.unique().numel() != 20
                    or int(selected_local.min()) < 0
                    or int(selected_local.max()) >= 450):
                raise ValueError('offline herding selection is malformed')
            offline_indices.extend(eligible[class_id * 450 + int(local)]
                                   for local in selected_local.tolist())
            rows.append({
                'class_id': class_id,
                'online_count': 20,
                'candidate_count': 450,
                'offline_count': 20,
                'online_aggregate_error': normalized_centroid_error(
                    class_full, online_agg[online_slice]),
                'offline_aggregate_error': normalized_centroid_error(
                    class_full, class_full.index_select(0, selected_local)),
                'party_online_errors': [
                    normalized_centroid_error(full_party[p][full_slice],
                                              online_party[p][online_slice])
                    for p in range(4)
                ],
                'party_offline_errors': [
                    normalized_centroid_error(
                        full_party[p][full_slice],
                        full_party[p][full_slice].index_select(0, selected_local))
                    for p in range(4)
                ],
            })
        if bottom_state_sha256(trainer.bottoms) != state_before:
            raise RuntimeError('frozen bottom state changed during audit')
        if getattr(dataset, '_first_accessed_splits', set()):
            raise RuntimeError('audit accessed a protected evaluation loader')
    if (len(offline_indices) != 2000
            or len(set(offline_indices)) != 2000
            or set(offline_indices) & heldout):
        raise ValueError('offline selection contains held-out or duplicate rows')
    return {
        'seed': seed,
        'formal_record_sha256': file_sha256(record_path),
        'config_sha256': file_sha256(config_path),
        'checkpoint_sha256': file_sha256(checkpoint_path),
        'validation_manifest_sha256': file_sha256(validation_manifest_path),
        'bic_manifest_sha256': file_sha256(run / 'bic' / 'calibration_manifest.json'),
        'data_sha256': data_hashes,
        'eligible_indices_sha256': hashlib.sha256(json.dumps(
            eligible, separators=(',', ':'),
        ).encode()).hexdigest(),
        'offline_selected_indices_sha256': hashlib.sha256(json.dumps(
            offline_indices, separators=(',', ':'),
        ).encode()).hexdigest(),
        'bottom_state_sha256': state_before,
        'summary': summarize_rows(rows),
        'classes': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    cli = parser.parse_args()
    root = cli.root.resolve(strict=True)
    output = cli.output_dir.resolve()
    if output.exists() or output == root or root in output.parents:
        raise ValueError('audit output must be a new root outside formal evidence')
    if not (root / 'METHOD_SHARD_SUCCESS').is_file():
        raise ValueError('formal Adaptive shard lacks success marker')
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
            != SOURCE_COMMIT
            or subprocess.check_output(['git', 'status', '--porcelain'], text=True)):
        raise ValueError('audit requires exact clean formal producer source')
    source_modules = {
        name: file_sha256(Path(module.__file__))
        for name, module in (
            ('calibration_split.py', calibration_split),
            ('cl_methods/proto_evolve.py', proto_evolve),
            ('data_utils.py', data_utils),
            ('determinism.py', determinism),
            ('models.py', models),
            ('vfl_trainer.py', vfl_trainer),
        )
    }
    helper_hash = file_sha256(Path(analyze_head_offset_pilot.__file__))
    if helper_hash != HELPER_SHA256:
        raise ValueError('training-index helper differs from frozen source')
    for seed in (42, 43, 44):
        record = json.loads((root / 'records' / f'cifar100%3Aadaptive%3A{seed}.json').read_text())
        if any(digest != record['artifact_sha256']['source:' + name]
               for name, digest in source_modules.items()):
            raise ValueError('imported model or selector code differs from producer')
    runs = [screen_seed(root, seed, cli.device) for seed in (42, 43, 44)]
    result = {
        'schema_version': 1,
        'status': 'training_only_online_offline_herding_gap',
        'source_root': str(root),
        'source_commit': SOURCE_COMMIT,
        'script_sha256': file_sha256(__file__),
        'helper_sha256': helper_hash,
        'source_modules_sha256': source_modules,
        'seeds': runs,
        'overall': {
            'class_rows': 300,
            'online_mean': statistics.fmean(run['summary']['online_mean'] for run in runs),
            'offline_mean': statistics.fmean(run['summary']['offline_mean'] for run in runs),
            'fraction_online_worse': statistics.fmean(
                run['summary']['fraction_online_worse'] for run in runs),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    temporary = output / 'online_offline_herding_gap.json.tmp'
    with open(temporary, 'x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output / 'online_offline_herding_gap.json')
    print(json.dumps({
        'per_seed': {run['seed']: run['summary'] for run in runs},
        'overall': result['overall'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
