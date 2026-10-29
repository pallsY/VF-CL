"""Manifest-gated development worker for the prelocked all-party FIM study."""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch

from party_drift_audit_gate import verify_selection_audit
from replay_fim_manifest import PROTOCOL, load_approved_job
from run_party_drift_development import (
    _sha256, attempt_args, authorize_queue_attempt,
    score_development_checkpoint, write_attempt_configs,
)


def validate_job_sources(config, row):
    if (config.get('pilot_protocol') != PROTOCOL or
            config.get('development_job_key') != row.get('job_key')):
        raise ValueError('study job identity differs from allowlist')
    for field, key in (
        ('source_config_sha256', 'development_source_config_path'),
        ('source_data_sha256', 'development_source_data_path'),
        ('holdout_sha256', 'development_holdout_manifest'),
        ('validation_manifest_sha256', 'pilot_paired_validation_manifest'),
    ):
        if _sha256(config[key]) != row[field]:
            raise ValueError(f'study source differs: {field}')


def audit_fim_masks(states, policy):
    if policy not in ('cumulative', 'fresh', 'off'):
        raise ValueError('invalid FIM mask audit policy')
    result, previous = [], None
    for task, state in enumerate(states):
        if state['task_id'] != task:
            raise ValueError('FIM task snapshots are incomplete or unordered')
        bottoms = state['trainer_state']['bottoms']
        masks = state['cl_state']['fim_masks']
        if len(bottoms) != len(masks):
            raise ValueError('FIM mask party count differs')
        parties = []
        for index, (bottom, mask) in enumerate(zip(bottoms, masks)):
            if set(mask) - set(bottom) or any(type(value) is not bool
                                              for value in mask.values()):
                raise ValueError('FIM mask has invalid tensor names or values')
            frozen = sorted(name for name, value in mask.items() if value)
            if policy == 'off' and frozen:
                raise ValueError('F-off froze a tensor')
            if (policy == 'cumulative' and previous is not None and
                    not set(previous[index]) <= set(frozen)):
                raise ValueError('cumulative FIM mask released a tensor')
            total = sum(tensor.numel() for tensor in bottom.values())
            count = sum(bottom[name].numel() for name in frozen)
            if not total:
                raise ValueError('empty FIM bottom state')
            canonical = json.dumps(frozen, separators=(',', ':')).encode()
            parties.append({
                'party': index, 'frozen_names': frozen,
                'frozen_param_fraction': count / total,
                'mask_sha256': hashlib.sha256(canonical).hexdigest(),
            })
        result.append({'task_id': task, 'parties': parties})
        previous = [party['frozen_names'] for party in parties]
    if len(result) < 2:
        raise ValueError('study FIM audit needs adjacent tasks')
    return result


def run_listed_job(job_key, manifest_path, manifest_sha256, stage,
                   selected_policy=None):
    import runner
    from config import validate_party_kd_variant, validate_sample_replay_only
    from run_resource_lite_pilot import (
        PilotDataset, StopBeforeTest, validate_training_only_artifacts,
    )

    checkout = Path(__file__).resolve().parent
    commit = subprocess.check_output(
        ['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    manifest_path = Path(manifest_path).resolve()
    row = load_approved_job(
        job_key, manifest_path, manifest_sha256, code_commit=commit,
        stage=stage, selected_policy=selected_policy)
    config_path = (manifest_path.parent / row['config_relpath']).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    validate_job_sources(config, row)
    args = attempt_args(config, 1)
    authorize_queue_attempt(config['output_dir'], job_key, 1,
                            manifest_sha256, commit, row['config_sha256'])
    if args.formal_deferred_evaluation is not True:
        raise ValueError('study worker must stop before official test')
    validate_party_kd_variant(config, config.get('expected_party_kd_variant', ''))
    if args.cl_method == 'proto_evolve':
        validate_sample_replay_only(args)

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    write_attempt_configs(root, config_path, args)
    (root / 'JOB_RUNTIME.json').write_text(json.dumps({
        'job_key': job_key, 'stage': stage, 'attempt': 1,
        'manifest_sha256': manifest_sha256,
        'config_sha256': _sha256(config_path), 'code_commit': commit,
        'test_allowed': False,
    }, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    dataset = PilotDataset(args)
    locked_holdout = json.loads(Path(
        args.development_holdout_manifest).read_text(encoding='utf-8'))
    if (dataset.holdout_manifest['sha256'] != locked_holdout['sha256'] or
            dataset.holdout_indices != set(locked_holdout['ordered_indices']) or
            dataset.holdout_indices & dataset.pilot_pair_excluded):
        raise ValueError('study holdout differs from locked training split')
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
        raise RuntimeError('study worker did not stop before official test')
    validate_training_only_artifacts(root)
    checkpoint = root / 'checkpoints' / 'formal_final.pt'
    metrics, rule, details = score_development_checkpoint(
        args, dataset, checkpoint)
    if (metrics['count'] != args.num_classes * 30 or
            not math.isfinite(metrics['cil']) or
            not 0 <= metrics['cil'] <= 1):
        raise ValueError('study holdout readout is incomplete or nonfinite')
    payload = runner._decode_checkpoint_value(torch.load(
        checkpoint, map_location='cpu', weights_only=True))
    selection = verify_selection_audit(
        root / 'party_drift_audit' / 'selection.jsonl', args, dataset,
        payload['seen_task_classes'], root / 'formal_snapshots')
    fim_audit_sha = None
    if args.cl_method == 'proto_evolve':
        snapshots = sorted((root / 'formal_snapshots').glob('event_*_CIL.pt'),
                           key=lambda path: int(path.stem.split('_')[1]))
        if len(snapshots) != args.num_tasks:
            raise ValueError('study FIM task snapshots are incomplete')
        states = [runner._decode_checkpoint_value(torch.load(
            path, map_location='cpu', weights_only=True)) for path in snapshots]
        mask_rows = audit_fim_masks(states, args.fim_mask_policy)
        for row, path in zip(mask_rows, snapshots):
            row['snapshot_sha256'] = _sha256(path)
        fim_path = root / 'FIM_MASK_AUDIT.json'
        with fim_path.open('x', encoding='utf-8') as stream:
            json.dump({'policy': args.fim_mask_policy, 'tasks': mask_rows},
                      stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        fim_audit_sha = _sha256(fim_path)
    validate_training_only_artifacts(root)
    readout = {
        'status': 'training_holdout_readout', 'job_key': job_key,
        'stage': stage, 'attempt': 1, 'readout_rule': rule,
        'metrics': metrics, 'readout_details': details,
        'config_sha256': _sha256(config_path),
        'checkpoint_sha256': _sha256(checkpoint),
        'holdout_selected_sha256': dataset.holdout_manifest['sha256'],
        'test_accessed': False,
    }
    with (root / 'HOLDOUT_READOUT.json').open('x', encoding='utf-8') as stream:
        json.dump(readout, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    success = {
        'status': 'success', 'job_key': job_key, 'stage': stage,
        'attempt': 1, 'manifest_sha256': manifest_sha256,
        'readout_sha256': _sha256(root / 'HOLDOUT_READOUT.json'),
        'data_flow_sha256': _sha256(root / 'data_flow_audit.jsonl'),
        'fim_mask_audit_sha256': fim_audit_sha,
        **selection, 'checkpoint_sha256': _sha256(checkpoint),
        'test_accessed': False,
    }
    with (root / 'JOB_SUCCESS.json').open('x', encoding='utf-8') as stream:
        json.dump(success, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'success', 'job_key': job_key},
                     sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-key', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--stage', choices=('A', 'B'), required=True)
    parser.add_argument('--selected-policy', choices=('f0', 'f1', 'f_off'))
    args = parser.parse_args()
    run_listed_job(args.job_key, args.manifest, args.manifest_sha256,
                   args.stage, args.selected_policy)


if __name__ == '__main__':
    main()
