"""Read-only NME diagnostic for frozen sample-only dev checkpoints.

Reuses the stored head_raw_replay (the same per-class raw examples iCaRL
keeps) to compute a nearest-exemplar-mean readout on the existing training
holdout, using iCaRL's own normalize-per-sample-then-average order. No
retraining, no new config, no test-set access. The source run directory is
never written to: PilotDataset is pointed at a separate diagnostic root, and
the source config/manifest/checkpoint hashes are checked before and after to
prove the frozen run was untouched.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from run_resource_lite_pilot import (
    PilotDataset, score_branch, sha256, validate_training_only_artifacts,
)


def nme_readout(args, dataset, checkpoint):
    import runner
    from cl_methods.icarl import ICaRLCL
    from data_utils import split_features
    from models import build_models
    from vfl_trainer import VFLTrainer

    score_args = SimpleNamespace(**vars(args))
    score_args.device = 'cpu'
    payload = runner._decode_checkpoint_value(torch.load(
        checkpoint, map_location='cpu', weights_only=True))
    bottoms, top = build_models(score_args)
    trainer = VFLTrainer(bottoms, top, score_args)
    trainer.load_state(payload['trainer_state'])
    raw_replay = {
        int(class_id): raw
        for class_id, raw in payload['cl_state']['head_raw_replay'].items()
    }
    if not raw_replay:
        raise ValueError('checkpoint has no stored raw replay to score with')
    for model in (*trainer.bottoms, trainer.top_model):
        model.eval()

    # Reuse iCaRL's own feature/mean rule so the two readouts are comparable:
    # _features() normalizes each sample first, _class_means() then averages
    # the already-unit-norm vectors (and renormalizes the mean).
    icarl_shim = ICaRLCL.__new__(ICaRLCL)
    icarl_shim.args = score_args
    icarl_shim.trainer = trainer
    icarl_shim.buffer = SimpleNamespace(data={
        class_id: list(raw) for class_id, raw in raw_replay.items()
    })

    with torch.no_grad():
        class_ids, means = icarl_shim._class_means()
        if set(class_ids.tolist()) != set(range(score_args.num_classes)):
            raise ValueError(
                'raw replay does not cover every class; NME is incomplete')
        per_class_counts = {int(c): len(raw_replay[int(c)])
                            for c in class_ids.tolist()}

        log_probs, labels = [], []
        for batch_x, batch_y in dataset.holdout_loader(
                list(range(score_args.num_classes))):
            features = icarl_shim._features(
                batch_x.to('cpu'), record_comm=False)
            distances = torch.cdist(features, means)
            scores = torch.full((batch_x.size(0), score_args.num_classes),
                                float('-inf'))
            scores[:, class_ids] = -distances
            log_probs.append(F.log_softmax(scores, dim=1))
            labels.append(batch_y)
    metrics = score_branch(
        torch.cat(log_probs), torch.cat(labels).long(),
        payload['seen_task_classes'])
    if metrics['count'] != score_args.num_classes * score_args.pilot_holdout_per_class:
        raise ValueError('pilot holdout example count differs')
    return {'metrics': metrics, 'readout_rule': 'icarl_nearest_exemplar_mean',
            'raw_replay_classes': len(class_ids),
            'raw_replay_per_class_counts': per_class_counts}


def run(run_dir, diagnostic_root):
    """Score an already-frozen reference20 run directory, read-only.

    `diagnostic_root` must be a fresh directory outside `run_dir`; it holds
    the reconstructed holdout manifest and NME_READONLY_CHECK.json. The
    source run directory's own files are only ever read, never written.
    """
    run_dir, diagnostic_root = Path(run_dir), Path(diagnostic_root)
    if diagnostic_root.resolve() == run_dir.resolve() or str(
            diagnostic_root.resolve()).startswith(str(run_dir.resolve()) + '\\') or str(
            diagnostic_root.resolve()).startswith(str(run_dir.resolve()) + '/'):
        raise ValueError('diagnostic root must be outside the frozen run directory')
    validate_training_only_artifacts(run_dir)
    config_path = run_dir / 'config.json'
    config_sha_before = sha256(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if Path(config['output_dir']).resolve() != run_dir.resolve():
        raise ValueError('run directory does not match its own config')
    if (config.get('pilot_dataset') != 'isolet'
            or config.get('seed') not in (54, 55)
            or config.get('pilot_variant') != 'reference20'
            or config.get('head_consolidation_samples_per_class') != 20):
        raise ValueError('source run is not a reference20 (20 samples/class) job')
    readout_path = run_dir / 'HOLDOUT_READOUT.json'
    readout_sha_before = sha256(readout_path)
    original = json.loads(readout_path.read_text(encoding='utf-8'))
    if original.get('status') not in (
            'independent_training_holdout_readout',
            'paired_training_holdout_readout'):
        raise ValueError('source run is not a frozen training-holdout readout')
    if any(original.get(key) != expected for key, expected in (
            ('dataset', 'isolet'), ('seed', config['seed']),
            ('variant', 'reference20'))):
        raise ValueError('source identity differs from its frozen readout')
    if config_sha_before != original.get('config_sha256'):
        raise ValueError('config differs from its frozen readout')
    freeze_path = run_dir / 'FORMAL_STATE_FROZEN.json'
    freeze_sha_before = sha256(freeze_path)
    if freeze_sha_before != original.get('freeze_sha256'):
        raise ValueError('freeze record differs from its frozen readout')
    manifest_path = run_dir / 'pilot_holdout' / 'manifest.json'
    manifest_sha_before = sha256(manifest_path)
    if manifest_sha_before != original.get('holdout_manifest_sha256'):
        raise ValueError('source holdout manifest no longer matches its frozen readout')
    checkpoint = run_dir / 'checkpoints' / 'formal_final.pt'
    checkpoint_sha_before = sha256(checkpoint)
    if checkpoint_sha_before != original.get('checkpoint_sha256'):
        raise ValueError('checkpoint differs from its frozen readout')

    diagnostic_root.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(**config)
    args.output_dir = str(diagnostic_root)
    dataset = PilotDataset(args)
    rebuilt_manifest = diagnostic_root / 'pilot_holdout' / 'manifest.json'
    if sha256(rebuilt_manifest) != manifest_sha_before:
        raise ValueError('reconstructed holdout differs from the frozen manifest')
    result = nme_readout(args, dataset, checkpoint)
    expected_per_class = config['head_consolidation_samples_per_class']
    if any(count != expected_per_class
          for count in result['raw_replay_per_class_counts'].values()):
        raise ValueError('stored raw replay is not exactly reference20')

    if (sha256(config_path) != config_sha_before
            or sha256(manifest_path) != manifest_sha_before
            or sha256(checkpoint) != checkpoint_sha_before
            or sha256(freeze_path) != freeze_sha_before
            or sha256(readout_path) != readout_sha_before):
        raise RuntimeError('source run directory was modified during scoring')

    result.update({
        'status': 'read_only_nme_diagnostic',
        'source_run': str(run_dir),
        'source_config_sha256': config_sha_before,
        'source_freeze_sha256': freeze_sha_before,
        'source_readout_sha256': readout_sha_before,
        'source_holdout_manifest_sha256': manifest_sha_before,
        'source_checkpoint_sha256': checkpoint_sha_before,
        'original_cil': original.get('metrics', {}).get('cil'),
        'original_readout_rule': original.get('metrics', {}).get('readout_rule'),
        'dataset': config.get('pilot_dataset'),
        'seed': config.get('seed'),
    })
    target = diagnostic_root / 'NME_READONLY_CHECK.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='Frozen reference20 run directory to score (read-only)')
    parser.add_argument('--diagnostic-root', type=Path, required=True,
                        help='Fresh directory (outside --run-dir) for diagnostic output')
    args = parser.parse_args()
    run(args.run_dir, args.diagnostic_root)


if __name__ == '__main__':
    main()
