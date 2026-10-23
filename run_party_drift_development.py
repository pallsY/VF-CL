"""Manifest-gated training-only development runs."""

import argparse
import hashlib
import json
import math
import os
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


def authorize_queue_attempt(job_root, job_key, attempt, manifest_sha256,
                            production_commit, config_sha256, *, queue_pid=None):
    from party_drift_queue import active_queue_lock

    job_root = Path(job_root).resolve()
    root = job_root.parents[3]
    queue_pid = os.getppid() if queue_pid is None else queue_pid
    lock = active_queue_lock(root)
    plan = json.loads((job_root / f'attempt-{attempt:02d}.plan.json').read_text(
        encoding='utf-8')) if (job_root / f'attempt-{attempt:02d}.plan.json').is_file() else None
    if (lock.get('pid') != queue_pid or
            lock.get('manifest_sha256') != manifest_sha256 or
            lock.get('production_commit') != production_commit or
            plan != {'attempt': attempt, 'job_key': job_key}):
        raise ValueError('worker is not the approved queued attempt')
    history = [json.loads(line) for line in (root / 'ATTEMPTS.jsonl').read_text(
        encoding='utf-8').splitlines() if line]
    history = [row for row in history if row.get('job_key') == job_key]
    if (not history or history[-1].get('kind') != 'start' or
            history[-1].get('attempt') != attempt or
            history[-1].get('config_sha256') != config_sha256 or
            sum(row.get('kind') == 'start' and row.get('attempt') == attempt
                for row in history) != 1):
        raise ValueError('worker start event is absent or duplicated')
    if attempt == 1:
        if len(history) != 1:
            raise ValueError('first attempt has earlier job history')
    else:
        prior = history[-2] if len(history) >= 2 else {}
        resume_path = job_root / f'attempt-{attempt:02d}.resume.json'
        resume = json.loads(resume_path.read_text(encoding='utf-8')) if resume_path.is_file() else {}
        if (prior.get('kind') != 'finish' or
                prior.get('attempt') != attempt - 1 or
                prior.get('classification') != 'infrastructure' or
                resume.get('job_key') != job_key or
                resume.get('attempt') != attempt or
                resume.get('manifest_sha256') != manifest_sha256 or
                not resume.get('reason')):
            raise ValueError('only a recorded infrastructure failure may retry')


def write_attempt_configs(root, approved_path, args):
    root = Path(root)
    (root / 'APPROVED_CONFIG.json').write_bytes(Path(approved_path).read_bytes())
    (root / 'config.json').write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + '\n',
        encoding='utf-8')


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


def score_bic_logits(calibration_logits, calibration_labels,
                     holdout_logits, holdout_labels, task_classes, *, lr, steps):
    from bic_calibration import fit_final_calibrator

    calibrator, fit = fit_final_calibrator(
        calibration_logits, calibration_labels, task_classes,
        'joint_alpha_beta', lr, steps)
    return {
        'raw': score_branch(F.log_softmax(holdout_logits, dim=1),
                            holdout_labels.long(), task_classes),
        'calibrated': score_branch(F.log_softmax(calibrator.apply(
            holdout_logits), dim=1), holdout_labels.long(), task_classes),
        'fit': fit,
    }


def bic_fit_loader(args, dataset):
    if args.bic_enabled != 1:
        raise ValueError('CIFAR development BiC is not enabled')
    internal = args.cl_method == 'proto_evolve'
    if internal and args.lambda_validation_enabled != 1:
        raise ValueError('adaptive BiC requires its frozen validation split')
    source = 'validation' if internal else 'calibration'
    view = dataset.validationset if internal else dataset.calibrationset
    selected = (dataset.validation_indices if internal else
                dataset.calibration_indices)
    per_class = (args.lambda_validation_per_class if internal else
                 args.bic_per_class)
    if selected & dataset.holdout_indices:
        raise ValueError('BiC fit split overlaps tuning holdout')
    indices = [index for index in dataset.get_class_indices(
        view, list(range(args.num_classes))) if index in selected]
    if len(indices) != args.num_classes * per_class or set(indices) != selected:
        raise ValueError('development BiC fit split is incomplete')
    return dataset._loader(view, indices, ('development_bic_fit', source),
                           False, audit=True), source


def score_development_checkpoint(args, dataset, checkpoint):
    from run_resource_lite_pilot import score_holdout_checkpoint

    if args.pilot_dataset != 'cifar100' and args.cl_method != 'icarl':
        return score_holdout_checkpoint(args, dataset, checkpoint), 'model_head', {}
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
    for model in (*trainer.bottoms, trainer.top_model):
        model.eval()
    if args.pilot_dataset == 'cifar100':
        loader, source = bic_fit_loader(args, dataset)
        calibration_logits, calibration_labels = trainer.collect_logits(loader)
        holdout_logits, holdout_labels = trainer.collect_logits(
            dataset.holdout_loader(list(range(args.num_classes))))
        paired = score_bic_logits(
            calibration_logits, calibration_labels,
            holdout_logits, holdout_labels, payload['seen_task_classes'],
            lr=args.bic_lr, steps=args.bic_steps)
        return paired['calibrated'], 'final_bic_joint_alpha_beta', {
            'bic_fit_source': source, 'bic_fit_count': len(calibration_labels),
            'bic_fit': paired['fit'], 'raw_metrics': paired['raw'],
        }
    method = get_cl_method('icarl', trainer, cpu_args)
    method.load_state(payload['cl_state'])
    return score_icarl_holdout(
        method, dataset, payload['seen_task_classes'], args.num_classes), 'icarl_nme', {}


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
    authorize_queue_attempt(config['output_dir'], job_key, attempt,
                            manifest_sha256, commit, row['config_sha256'])
    if args.formal_deferred_evaluation is not True:
        raise ValueError('development run must defer formal evaluation')
    validate_sample_replay_only(args)

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    write_attempt_configs(root, config_path, args)
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
    metrics, rule, details = score_development_checkpoint(args, dataset, checkpoint)
    if (metrics['count'] != args.num_classes * 30 or
            not math.isfinite(metrics['cil']) or
            not 0 <= metrics['cil'] <= 1):
        raise ValueError('development holdout readout is incomplete or nonfinite')
    from party_drift_audit_gate import verify_selection_audit
    payload = runner._decode_checkpoint_value(torch.load(
        checkpoint, map_location='cpu', weights_only=True))
    selection = verify_selection_audit(
        root / 'party_drift_audit' / 'selection.jsonl', args, dataset,
        payload['seen_task_classes'], root / 'formal_snapshots')
    validate_training_only_artifacts(root)
    readout = {
        'status': 'training_holdout_readout', 'job_key': job_key,
        'attempt': attempt, 'readout_rule': rule, 'metrics': metrics,
        'readout_details': details,
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
        **selection,
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
