"""One approved full-data F-off formal job, with a single formal test transaction."""

import argparse
import json
import subprocess
from pathlib import Path

import torch

from f_off_formal_manifest import digest, load_approved_job, pinned_checkout
from run_party_drift_development import (
    attempt_args, authorize_queue_attempt, write_attempt_configs,
)
from run_replay_fim_development import audit_fim_masks


def validate_sources(config, row):
    if (config.get('formal_followup_job_key') != row.get('job_key')
            or config.get('formal_deferred_evaluation') is not True
            or 'development_holdout_manifest' in config):
        raise ValueError('formal job identity or full-data boundary differs')
    for name, path in (
        ('source_config_sha256', 'source_config_path'),
        ('source_data_sha256', 'source_data_path'),
        ('validation_manifest_sha256', 'validation_manifest_path'),
    ):
        if digest(row[path]) != row[name]:
            raise ValueError(f'formal input differs: {name}')


def run_job(job_key, manifest, manifest_sha):
    import runner
    from config import validate_party_kd_variant, validate_sample_replay_only

    checkout = Path(__file__).resolve().parent
    commit = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'],
                                     text=True).strip()
    pinned_checkout(commit)
    row = load_approved_job(job_key, manifest, manifest_sha, commit)
    config_path = (Path(manifest).resolve().parent / row['config_relpath']).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    validate_sources(config, row)
    args = attempt_args(config, 1)
    authorize_queue_attempt(config['output_dir'], job_key, 1, manifest_sha,
                            commit, row['config_sha256'])
    validate_party_kd_variant(config, 'uniform')
    validate_sample_replay_only(args)

    run = Path(args.output_dir)
    run.mkdir(parents=True, exist_ok=False)
    write_attempt_configs(run, config_path, args)
    with (run / 'JOB_RUNTIME.json').open('x', encoding='utf-8') as stream:
        json.dump({'job_key': job_key, 'stage': 'formal_followup',
                   'attempt': 1, 'manifest_sha256': manifest_sha,
                   'code_commit': commit, 'test_allowed': True}, stream,
                  indent=2, sort_keys=True)
        stream.write('\n')
    runner.run_experiment(args)

    checkpoint = run / 'checkpoints/formal_final.pt'
    results = run / 'results.json'
    published = run / 'FORMAL_EVALUATION_PUBLISHED.json'
    complete = run / 'FORMAL_EVALUATION_COMPLETE.json'
    sealed = run / 'FORMAL_EVALUATION_SEALED.json'
    marker = json.loads(published.read_text(encoding='utf-8'))
    if (marker.get('status') != 'published'
            or marker.get('checkpoint', {}).get('sha256') != digest(checkpoint)
            or marker.get('results', {}).get('sha256') != digest(results)
            or not complete.is_file() or not sealed.is_file()):
        raise ValueError('formal single-access publication is not sealed')
    snapshots = sorted((run / 'formal_snapshots').glob('event_*_CIL.pt'),
                       key=lambda path: int(path.stem.split('_')[1]))
    if len(snapshots) != args.num_tasks:
        raise ValueError('formal task snapshots are incomplete')
    states = [runner._decode_checkpoint_value(torch.load(
        path, map_location='cpu', weights_only=True)) for path in snapshots]
    masks = audit_fim_masks(states, 'off')
    for record, path in zip(masks, snapshots):
        record['snapshot_sha256'] = digest(path)
    mask_path = run / 'FIM_MASK_AUDIT.json'
    with mask_path.open('x', encoding='utf-8') as stream:
        json.dump({'policy': 'off', 'tasks': masks}, stream,
                  indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    with (run / 'JOB_SUCCESS.json').open('x', encoding='utf-8') as stream:
        json.dump({
            'status': 'success', 'stage': 'formal_followup',
            'job_key': job_key, 'attempt': 1, 'manifest_sha256': manifest_sha,
            'code_commit': commit,
            'checkpoint_sha256': digest(checkpoint),
            'results_sha256': digest(results),
            'published_sha256': digest(published),
            'complete_sha256': digest(complete),
            'sealed_sha256': digest(sealed),
            'fim_mask_audit_sha256': digest(mask_path),
            'data_flow_sha256': digest(run / 'data_flow_audit.jsonl'),
        }, stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-key', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    run_job(args.job_key, args.manifest, args.manifest_sha256)
