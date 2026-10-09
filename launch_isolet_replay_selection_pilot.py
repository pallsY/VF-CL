"""Training-only ISOLET equal-memory replay selection pilot."""

import argparse
import copy
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from calibration_split import build_manifest, manifest_indices, write_manifest
from cl_methods.proto_evolve import herding_indices
from data_utils import VFLDataset
from analyze_isolet_head_branches import score_branch, sha256


HOLDOUT_SEED = 20261010
HOLDOUT_PER_CLASS = 40

SOURCE_EXPECTED = {
    'data': 'tabvfl', 'num_classes': 26, 'num_tasks': 13,
    'classes_per_task': 2, 'num_parties': 4,
    'seed': 42, 'formal_deferred_evaluation': True,
    'head_consolidation_enabled': 1,
    'head_consolidation_mode': 'adaptive_dual_branch',
    'head_consolidation_samples_per_class': 20,
    'head_consolidation_schedule': 'final',
    'lambda_validation_enabled': 1,
    'lambda_validation_per_class': 40,
    'lambda_validation_split_seed': 20260809,
    'epochs_per_task': 50, 'batch_size': 128,
    'optimizer': 'adamw',
    'proto_lambda_a': 0.15, 'distill_weight': 0.25,
    'feat_distill_weight': 0.05,
}
ALLOWED_OVERRIDES = {
    'seed', 'formal_deferred_evaluation', 'data_path', 'vector_npz',
    'results_dir', 'output_dir', 'exp_name', 'proto_lambda_a',
    'distill_weight', 'feat_distill_weight',
}


def derive_config(source, root, seed, variant):
    if (int(seed) not in (47, 48, 49, 50) or variant not in ('herding', 'hybrid')
            or any(source.get(key) != value for key, value in SOURCE_EXPECTED.items())
            or Path(source.get('vector_npz', '')).name != 'isolet_vfl.npz'):
        raise ValueError('pilot source, seed, or variant differs from locked design')
    root = Path(root).resolve()
    config = copy.deepcopy(source)
    config.update(
        seed=int(seed), formal_deferred_evaluation=False,
        data_path='/home/c3080/YangXiaoXiang/VF-CL/data',
        vector_npz='/home/c3080/YangXiaoXiang/VF-CL/data/isolet/isolet_vfl.npz',
        results_dir=str(root), output_dir=str(root / 'run'),
        exp_name=f'isolet_replay_{variant}_seed{seed}',
        proto_lambda_a=0.05, distill_weight=0.10,
        feat_distill_weight=0.02,
    )
    changed = {key: {'source': source.get(key), 'pilot': config.get(key)}
               for key in set(source) | set(config)
               if source.get(key) != config.get(key)}
    if not set(changed) <= ALLOWED_OVERRIDES:
        raise ValueError('pilot changed an unapproved config field')
    return config, changed


def reject_test_access(rows):
    if any(row.get('split') == 'test'
           or str(row.get('loader_key', '')).startswith("('test'")
           for row in rows):
        raise ValueError('pilot accessed a test loader')

def hybrid_indices(embeddings, capacity):
    """Keep half the mean-herding set, then cover remaining feature directions."""
    capacity = min(int(capacity), embeddings.size(0))
    if capacity <= 0:
        raise ValueError('capacity must be positive')
    selected = herding_indices(embeddings, max(1, capacity // 2)).tolist()
    features = F.normalize(embeddings.float(), dim=1)
    available = torch.ones(features.size(0), dtype=torch.bool)
    available[selected] = False
    while len(selected) < capacity:
        nearest = (1.0 - features @ features[selected].T).min(dim=1).values
        nearest[~available] = -float('inf')
        index = int(nearest.argmax())
        selected.append(index)
        available[index] = False
    return torch.tensor(selected, dtype=torch.long)


def build_holdout_manifest(targets, gate_indices, per_class, seed):
    manifest = build_manifest(
        targets, per_class=per_class, seed=seed,
        dataset='isolet_vfl.npz-train', excluded_indices=gate_indices,
    )
    indices = manifest_indices(manifest)
    if indices & set(gate_indices):
        raise ValueError('pilot holdout overlaps gate validation')
    return manifest, indices


class PilotDataset(VFLDataset):
    def __init__(self, args):
        super().__init__(args)
        self.holdout_manifest, self.holdout_indices = build_holdout_manifest(
            self.trainset.targets, self.validation_indices,
            HOLDOUT_PER_CLASS, HOLDOUT_SEED,
        )
        write_manifest(
            self.holdout_manifest,
            Path(args.output_dir) / 'pilot_holdout' / 'manifest.json',
        )

    def _loader(self, dataset, indices, key, shuffle, audit=False):
        if key[0] == 'train':
            indices = [i for i in indices
                       if i not in getattr(self, 'holdout_indices', set())]
        return super()._loader(dataset, indices, key, shuffle, audit=audit)

    def holdout_loader(self, classes):
        indices = [i for i in self.get_class_indices(self.trainset, classes)
                   if i in self.holdout_indices]
        return self._loader(
            self.trainset, indices,
            ('pilot_holdout', tuple(int(c) for c in classes)), False,
        )

SOURCE_COMMIT = 'a575bbf446ae501cf8e580ba62c2cc5492a25f30'
SOURCE_CONFIG_SHA256 = '5702b8846ca8e9d279728a725550c1b053b8580fec832a9695b56d6ea7f2af34'
SOURCE_RECORD_SHA256 = '7dc5bf930600d097134948d6ee166fd55490327d16cde1fe2496ac374a73299b'
DATA_SHA256 = {
    'data:isolet/isolet_vfl.npz': 'd34312670de93198afcd2b126c95b79bae2b4cffeb30d480f097faf046b69514',
    'data:isolet/isolet_vfl.metadata.json': '79396dea1751b6094a5769f2dd789ad58b3ea9582a12c8c98d3ec07d8d1eb3cd',
}


class TrainingOnlyStop(Exception):
    """Final adaptive state is frozen; deferred test evaluation must not run."""


def stop_before_test(*_args, **_kwargs):
    raise TrainingOnlyStop


def preflight(cli):
    root = cli.root.resolve()
    if (cli.seed in (47, 48)) != (HOLDOUT_SEED == 20261010):
        raise ValueError('pilot seed/holdout protocol differs')
    if root.exists():
        raise ValueError('pilot root must be new')
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
            != SOURCE_COMMIT
            or subprocess.check_output(['git', 'status', '--porcelain'], text=True)):
        raise ValueError('pilot requires exact clean ISOLET producer worktree')
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '1'
            or os.environ.get('PYTHONHASHSEED') != str(cli.seed)
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'
            or os.environ.get('OMP_NUM_THREADS') != '1'
            or os.environ.get('MKL_NUM_THREADS') != '1'):
        raise ValueError('deterministic 3080 GPU environment differs')
    if shutil.disk_usage(root.parent).free < 5 * 1024 ** 3:
        raise ValueError('less than 5 GiB of result storage remains')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('masked physical GPU 1 is unavailable')
    if torch.cuda.mem_get_info(0)[0] < 4 * 1024 ** 3:
        raise ValueError('masked GPU has less than 4 GiB free')
    source_path = cli.source_config.resolve(strict=True)
    record_path = cli.source_record.resolve(strict=True)
    source = json.loads(source_path.read_text(encoding='utf-8'))
    record = json.loads(record_path.read_text(encoding='utf-8'))
    if (sha256(source_path) != SOURCE_CONFIG_SHA256
            or sha256(record_path) != SOURCE_RECORD_SHA256
            or record.get('kind') != 'formal_completed_run'
            or record.get('dataset') != 'isolet'
            or record.get('method') != 'adaptive'
            or record.get('seed') != 42
            or record.get('source_commit') != SOURCE_COMMIT
            or record['artifact_sha256']['config'] != SOURCE_CONFIG_SHA256):
        raise ValueError('archived ISOLET config/record identity differs')
    config, changed = derive_config(source, root, cli.seed, cli.variant)
    data_hashes = {
        key: sha256(Path(config['data_path']) / key.split(':', 1)[1])
        for key in DATA_SHA256
    }
    if data_hashes != DATA_SHA256:
        raise ValueError('ISOLET data payload differs from formal source')
    from config import validate_adaptive_head_consolidation, validate_party_kd_variant
    from data_utils import TaskManager
    args = SimpleNamespace(**config)
    validate_party_kd_variant(config, config['expected_party_kd_variant'])
    validate_adaptive_head_consolidation(args)
    if any(event['type'] != 'CIL' for event in TaskManager(args).get_timeline()):
        raise ValueError('pilot timeline contains unlearning')
    with tempfile.TemporaryDirectory(prefix='isolet-replay-pilot-check-') as scratch:
        check_args = SimpleNamespace(**config)
        check_args.output_dir = scratch
        dataset = PilotDataset(check_args)
        gate_manifest = Path(scratch) / 'validation' / 'validation_manifest.json'
        if (sha256(gate_manifest)
                != record['artifact_sha256']['validation_manifest']
                or len(dataset.validation_indices) != 1040
                or len(dataset.holdout_indices) != 1040
                or dataset.validation_indices & dataset.holdout_indices):
            raise ValueError('gate/holdout manifest differs or overlaps')
        remaining = set(range(len(dataset.trainset))) - dataset.validation_indices - dataset.holdout_indices
        if any(sum(dataset.trainset.targets[i] == c for i in remaining) < 20
               for c in range(26)):
            raise ValueError('pilot leaves fewer than 20 train examples per class')
        holdout_hash = dataset.holdout_manifest['sha256']
        gate_hash = dataset.validation_manifest['sha256']
    return args, changed, data_hashes, gate_hash, holdout_hash


@torch.no_grad()
def evaluate_holdout(args, checkpoint_path, expected_manifest_hash):
    from adaptive_consolidation_audit import _safe_torch_load
    from data_utils import split_features
    from head_consolidation import hash_top_state
    from models import build_models
    from vfl_trainer import VFLTrainer

    dataset = PilotDataset(args)
    if dataset.holdout_manifest['sha256'] != expected_manifest_hash:
        raise ValueError('pilot holdout changed after training')
    checkpoint = _safe_torch_load(checkpoint_path)
    bundle = checkpoint['cl_state']['adaptive_audit_bundle']
    if (hash_top_state(bundle['installed_state'])
            != hash_top_state(checkpoint['trainer_state']['top_model'])
            or bundle['result']['validation_manifest']['sha256']
            != dataset.validation_manifest['sha256']):
        raise ValueError('final installed head or gate validation differs')
    raw = checkpoint['cl_state']['head_raw_replay']
    if set(raw) != set(range(26)) or any(rows.size(0) != 20 for rows in raw.values()):
        raise ValueError('head replay is not exactly 20 examples per class')
    bottoms, top = build_models(args)
    trainer = VFLTrainer(bottoms, top, args)
    trainer.load_state(checkpoint['trainer_state'])
    for model in (*trainer.bottoms, trainer.top_model):
        model.eval()
    scores, labels = [], []
    for bx, by in dataset.holdout_loader(list(range(26))):
        parts = split_features(bx.to(args.device), args)
        embeddings = [bottom(part) for bottom, part in zip(trainer.bottoms, parts)]
        scores.append(trainer.top_model(trainer._aggregate(embeddings)).cpu())
        labels.append(by.cpu())
    metrics = score_branch(torch.cat(scores), torch.cat(labels), bundle['task_classes'])
    if metrics['count'] != 1040:
        raise ValueError('pilot holdout readout count differs')
    confusion = metrics['task_confusion']
    metrics['old_to_old_errors'] = sum(
        confusion[i][j] for i in range(12) for j in range(12) if i != j
    )
    return metrics, bundle['result']['pre_head_sha256'], bundle['result']['gate']['g']


def main():
    global HOLDOUT_SEED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--source-record', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(47, 48, 49, 50), required=True)
    parser.add_argument('--holdout-seed', type=int, choices=(20261010, 20261011),
                        default=20261010)
    parser.add_argument('--variant', choices=('herding', 'hybrid'), required=True)
    parser.add_argument('--check', action='store_true')
    cli = parser.parse_args()
    HOLDOUT_SEED = cli.holdout_seed
    args, changed, data_hashes, gate_hash, holdout_hash = preflight(cli)
    if cli.check:
        print(json.dumps({'status': 'ready', 'seed': cli.seed,
                          'variant': cli.variant, 'changed': changed,
                          'gate_manifest_sha256': gate_hash,
                          'holdout_manifest_sha256': holdout_hash,
                          'data_sha256': data_hashes}, indent=2, sort_keys=True))
        return
    import cl_methods.proto_evolve as proto_evolve
    import runner
    from adaptive_consolidation_audit import _safe_torch_load

    root = cli.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    run = Path(args.output_dir)
    run.mkdir()
    (run / 'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    (run / 'PILOT_PROTOCOL.json').write_text(json.dumps({
        'source_commit': SOURCE_COMMIT, 'source_config_sha256': SOURCE_CONFIG_SHA256,
        'source_record_sha256': SOURCE_RECORD_SHA256,
        'launcher_sha256': sha256(__file__), 'changed': changed,
        'data_sha256': data_hashes, 'gate_manifest_sha256': gate_hash,
        'holdout_manifest_sha256': holdout_hash,
        'holdout_per_class': HOLDOUT_PER_CLASS, 'holdout_split_seed': HOLDOUT_SEED,
        'selector': cli.variant, 'planned_stop': 'before_deferred_test_evaluation',
    }, indent=2, sort_keys=True), encoding='utf-8')
    original_dataset = runner.VFLDataset
    original_selector = proto_evolve.herding_indices
    original_evaluator = runner.evaluate_deferred_cil_trajectory
    calls = []

    def select(embeddings, capacity):
        if capacity != 20:
            raise ValueError('pilot replay capacity differs')
        calls.append(int(embeddings.size(0)))
        return (original_selector(embeddings, capacity)
                if cli.variant == 'herding'
                else hybrid_indices(embeddings, capacity))

    runner.VFLDataset = PilotDataset
    proto_evolve.herding_indices = select
    runner.evaluate_deferred_cil_trajectory = stop_before_test
    stopped = False
    try:
        runner.run_experiment(args)
    except TrainingOnlyStop:
        stopped = True
    finally:
        runner.VFLDataset = original_dataset
        proto_evolve.herding_indices = original_selector
        runner.evaluate_deferred_cil_trajectory = original_evaluator
    if not stopped or len(calls) != 26:
        raise RuntimeError('pilot did not freeze the complete task stream')
    checkpoint = run / 'adaptive_final.pt'
    events = [run / 'checkpoints' / f'event_{t}_CIL.pt' for t in range(13)]
    audit = run / 'data_flow_audit.jsonl'
    manifest = run / 'pilot_holdout' / 'manifest.json'
    if (not checkpoint.is_file() or not (run / 'ADAPTIVE_STATE_FROZEN.json').is_file()
            or any(not path.is_file() for path in events)
            or not audit.is_file() or (run / 'results.json').exists()
            or not manifest.is_file()):
        raise RuntimeError('pilot stopped at an incomplete final checkpoint')
    rows = [json.loads(line) for line in audit.read_text(encoding='utf-8').splitlines()]
    reject_test_access(rows)
    if json.loads(manifest.read_text())['sha256'] != holdout_hash:
        raise ValueError('saved holdout manifest changed')
    metrics, pre_head_sha256, gate = evaluate_holdout(args, checkpoint, holdout_hash)
    readout = {'status': 'independent_training_holdout_readout',
               'seed': cli.seed, 'variant': cli.variant,
               'holdout_manifest_sha256': sha256(manifest),
               'checkpoint_sha256': sha256(checkpoint),
               'pre_head_sha256': pre_head_sha256,
               'gate': gate, 'selector_calls': len(calls),
               'head_raw_replay_per_class': 20, 'metrics': metrics}
    readout_path = run / 'HOLDOUT_READOUT.json'
    readout_path.write_text(json.dumps(readout, indent=2, sort_keys=True),
                            encoding='utf-8')
    complete = {'status': 'training_only_before_test',
                'seed': cli.seed, 'variant': cli.variant,
                'config_sha256': sha256(run / 'config.json'),
                'checkpoint_sha256': sha256(checkpoint),
                'holdout_readout_sha256': sha256(readout_path),
                'gate_manifest_sha256': sha256(run / 'validation' / 'validation_manifest.json'),
                'holdout_manifest_sha256': sha256(manifest),
                'event_checkpoint_sha256': [sha256(path) for path in events],
                'data_flow_audit_sha256': sha256(audit)}
    (run / 'PILOT_COMPLETE.json').write_text(
        json.dumps(complete, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({'status': complete['status'], 'seed': cli.seed,
                      'variant': cli.variant, 'cil': metrics['cil'],
                      'new_cil': metrics['new_cil'],
                      'old_to_old_errors': metrics['old_to_old_errors']}))


if __name__ == '__main__':
    main()
