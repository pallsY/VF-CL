"""Manifest-gated training-only development runs."""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from run_resource_lite_pilot import score_branch


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_job_sources(config, row):
    if (config.get('pilot_protocol') != 'party-drift-development-v1' or
            config.get('development_job_key') != row.get('job_key')):
        raise ValueError('development job protocol or identity differs')
    files = {
        'source_config_sha256': config['development_source_config_path'],
        'source_data_sha256': config['development_source_data_path'],
        'holdout_sha256': config['development_holdout_manifest'],
        'validation_manifest_sha256': config['pilot_paired_validation_manifest'],
    }
    for field, path in files.items():
        if _sha256(path) != row[field]:
            raise ValueError(f'development {field} differs from manifest')


def attempt_args(config, attempt):
    if type(attempt) is not int or attempt < 1:
        raise ValueError('attempt number must be a positive integer')
    root = (PurePosixPath(config['output_dir'])
            if str(config['output_dir']).startswith('/') else
            Path(config['output_dir']).resolve())
    name = f'attempt-{attempt:02d}'
    return SimpleNamespace(**{
        **config, 'results_dir': str(root),
        'output_dir': str(root / name), 'exp_name': name,
    })


@torch.no_grad()
def score_icarl_holdout(method, dataset, task_classes, num_classes):
    """Use iCaRL's own normalized exemplar means for Class-IL readout."""
    class_ids, means = method._class_means()
    if class_ids.tolist() != list(range(num_classes)):
        raise ValueError('iCaRL NME means do not cover every seen class')
    rows, labels = [], []
    for examples, target in dataset.holdout_loader(list(range(num_classes))):
        features = method._features(examples, record_comm=False)
        distances = torch.cdist(features, means)
        scores = torch.full((len(target), num_classes), -1e9,
                            device=features.device)
        scores[:, class_ids] = -distances
        rows.append(scores.cpu())
        labels.append(target.cpu().long())
    if not rows:
        raise ValueError('iCaRL development holdout is empty')
    return score_branch(F.log_softmax(torch.cat(rows), dim=1),
                        torch.cat(labels), task_classes)


def score_development_checkpoint(args, dataset, checkpoint):
    from run_resource_lite_pilot import score_holdout_checkpoint

    if args.cl_method != 'icarl':
        return score_holdout_checkpoint(args, dataset, checkpoint), 'model_head'
    import runner
    from cl_methods import get_cl_method
    from models import build_models
    from vfl_trainer import VFLTrainer

    cpu_args = SimpleNamespace(**vars(args))
    cpu_args.device = 'cpu'
    payload = runner._decode_checkpoint_value(torch.load(
        checkpoint, map_location='cpu', weights_only=True))
    bottoms, top = build_models(cpu_args)
    trainer = VFLTrainer(bottoms, top, cpu_args)
    trainer.load_state(payload['trainer_state'])
    method = get_cl_method('icarl', trainer, cpu_args)
    method.load_state(payload['cl_state'])
    for model in (*trainer.bottoms, trainer.top_model):
        model.eval()
    return score_icarl_holdout(
        method, dataset, payload['seen_task_classes'], args.num_classes), 'icarl_nme'


def run_listed_job(job_key, manifest_path, manifest_sha256, attempt):
    import runner
    from config import validate_sample_replay_only
    from party_drift_manifest import load_approved_job
    from run_resource_lite_pilot import (
        PilotDataset, StopBeforeTest, validate_training_only_artifacts,
    )

    checkout = Path(__file__).resolve().parent
    commit = subprocess.check_output(
        ['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    manifest_path = Path(manifest_path).resolve()
    row = load_approved_job(job_key, manifest_path, manifest_sha256,
                            production_commit=commit)
    config_path = (manifest_path.parent / row['config_relpath']).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    validate_job_sources(config, row)
    args = attempt_args(config, attempt)
    if args.formal_deferred_evaluation is not True:
        raise ValueError('development run must defer formal evaluation')
    validate_sample_replay_only(args)

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'config.json').write_bytes(config_path.read_bytes())
    (root / 'JOB_RUNTIME.json').write_text(json.dumps({
        'job_key': job_key, 'attempt': attempt, 'manifest_sha256': manifest_sha256,
        'config_sha256': _sha256(config_path), 'production_commit': commit,
        'actual_output_dir': str(root), 'test_allowed': False,
    }, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    dataset = PilotDataset(args)
    locked_holdout = json.loads(Path(
        args.development_holdout_manifest).read_text(encoding='utf-8'))
    if (dataset.holdout_manifest['sha256'] != locked_holdout['sha256'] or
            dataset.holdout_indices != set(locked_holdout['ordered_indices']) or
            dataset.holdout_indices & dataset.pilot_pair_excluded):
        raise ValueError('generated holdout differs from locked exact-source cohort')
    original = runner.VFLDataset
    runner.VFLDataset = lambda _args: dataset
    stopped = False
    try:
        runner.run_experiment(args)
    except StopBeforeTest:
        stopped = True
    finally:
        runner.VFLDataset = original
    if not stopped:
        raise RuntimeError('development job did not stop before official test')
    validate_training_only_artifacts(root)
    checkpoint = root / 'checkpoints' / 'formal_final.pt'
    metrics, rule = score_development_checkpoint(args, dataset, checkpoint)
    if (metrics['count'] != args.num_classes * 30 or
            not math.isfinite(metrics['cil']) or
            not 0 <= metrics['cil'] <= 1):
        raise ValueError('development holdout readout is incomplete or nonfinite')
    readout = {
        'status': 'training_holdout_readout', 'job_key': job_key,
        'attempt': attempt, 'readout_rule': rule, 'metrics': metrics,
        'config_sha256': _sha256(config_path),
        'checkpoint_sha256': _sha256(checkpoint),
        'holdout_selected_sha256': dataset.holdout_manifest['sha256'],
        'test_accessed': False,
    }
    with (root / 'HOLDOUT_READOUT.json').open('x', encoding='utf-8') as stream:
        json.dump(readout, stream, indent=2, sort_keys=True)
        stream.write('\n')
    success = {
        'status': 'success', 'job_key': job_key, 'attempt': attempt,
        'manifest_sha256': manifest_sha256,
        'readout_sha256': _sha256(root / 'HOLDOUT_READOUT.json'),
        'data_flow_sha256': _sha256(root / 'data_flow_audit.jsonl'),
        'checkpoint_sha256': _sha256(checkpoint),
        'test_accessed': False,
    }
    with (root / 'JOB_SUCCESS.json').open('x', encoding='utf-8') as stream:
        json.dump(success, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'status': 'success', 'job_key': job_key,
                      'attempt': attempt}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-key', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    run_listed_job(args.job_key, args.manifest, args.manifest_sha256,
                   args.attempt)


if __name__ == '__main__':
    main()
