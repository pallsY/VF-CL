"""Read-only gate for a completed development run's party-selection ledger."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from determinism import derive_seed
from party_drift_local import (
    attenuation_weights, static_importance_weights, verify_audit_record,
)
from party_drift_replay import BalancedOldReplay, cmcdr_positions


def validate_cmcdr_fields(row, arm):
    fields = {'cmcdr_examples', 'cmcdr_subsets', 'cmcdr_profile_loss',
              'cmcdr_rows'}
    if (arm in ('cmcdr', 'cmcdr_budget')) != (fields <= set(row)) or (
            arm not in ('cmcdr', 'cmcdr_budget') and fields & set(row)):
        raise ValueError('CMCDR intervention fields differ from replay arm')


def validate_probe_ids(record, seed, task, epoch, old_classes):
    rng = np.random.default_rng(np.random.SeedSequence(
        [seed, task, epoch, 90210]))
    expected = [[int(class_id), int(row)] for class_id in old_classes
                for row in rng.choice(20, 2, replace=False)]
    if (record.get('seed_sequence') != [seed, task, epoch, 90210] or
            record.get('logical_probe_ids') != expected):
        raise ValueError('probe sample IDs differ from preregistered RNG')


def verify_sealed_job(run, job_key, manifest_sha256, *, replay):
    run = Path(run)
    gate = json.loads((run / 'JOB_SUCCESS.json').read_text(encoding='utf-8'))
    readout = run / 'HOLDOUT_READOUT.json'
    audit = run / 'party_drift_audit' / 'selection.jsonl'
    checkpoint = run / 'checkpoints' / 'formal_final.pt'
    data_flow = run / 'data_flow_audit.jsonl'
    readout_payload = json.loads(readout.read_text(encoding='utf-8'))
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if (gate.get('status') != 'success' or gate.get('job_key') != job_key or
            gate.get('manifest_sha256') != manifest_sha256 or
            gate.get('test_accessed') is not False or
            gate.get('readout_sha256') != hashlib.sha256(
                readout.read_bytes()).hexdigest() or
            gate.get('checkpoint_sha256') != checkpoint_sha or
            readout_payload.get('checkpoint_sha256') != checkpoint_sha or
            gate.get('data_flow_sha256') != hashlib.sha256(
                data_flow.read_bytes()).hexdigest()):
        raise ValueError('sealed development job identity or readout differs')
    expected_audit = (hashlib.sha256(audit.read_bytes()).hexdigest()
                      if audit.is_file() else None)
    if (gate.get('selection_audit_sha256') != expected_audit or
            (replay and (expected_audit is None or
                         type(gate.get('replay_steps')) is not int or
                         gate['replay_steps'] < 1)) or
            (not replay and (expected_audit is not None or
                             gate.get('replay_steps') != 0))):
        raise ValueError('sealed party-selection audit differs')
    return gate


def verify_selection_audit(path, args, dataset, seen_task_classes,
                           snapshot_dir=None):
    path = Path(path)
    if not args.party_drift_enabled:
        if path.exists():
            raise ValueError('non-replay job wrote a selection ledger')
        return {'selection_audit_sha256': None, 'replay_steps': 0}
    if not path.is_file():
        raise ValueError('replay job has no selection ledger')
    records = [json.loads(line) for line in path.read_text(
        encoding='utf-8').splitlines() if line]
    if not records or sorted(seen_task_classes) != list(range(args.num_tasks)):
        raise ValueError('selection ledger or task history is incomplete')
    arm, parties = args.party_replay_arm, args.num_parties
    selected_count = (parties if arm in
                      ('all_party', 'cmcdr', 'cmcdr_budget') else
                      (parties + 1) // 2)
    steps = [row for row in records if 'selected' in row]
    refreshes = [row for row in records if row.get('kind') in
                 ('weight_refresh', 'probe')]
    budgets = [row for row in records if row.get('kind') == 'cmcdr_budget']
    if len(steps) + len(refreshes) + len(budgets) != len(records):
        raise ValueError('selection ledger has unknown records')
    total_steps = 0
    static_frozen = {}
    for task in range(1, args.num_tasks):
        old = sorted(class_id for stage in range(task)
                     for class_id in seen_task_classes[stage])
        if arm == 'static':
            if snapshot_dir is None:
                raise ValueError('static selection needs frozen task snapshots')
            import runner
            previous = runner._decode_checkpoint_value(torch.load(
                Path(snapshot_dir) / f'event_{task - 1}_CIL.pt',
                map_location='cpu', weights_only=True))
            contribution = previous['cl_state']['dep_tracker_contrib']
            for class_id in seen_task_classes[task - 1]:
                values = contribution[class_id].abs().float()
                static_frozen[class_id] = (
                    values / values.sum() if float(values.sum()) > 0 else
                    torch.full_like(values, 1.0 / len(values))).tolist()
        current = set(seen_task_classes[task])
        indices = [index for index, label in
                   enumerate(dataset.trainset.targets)
                   if int(label) in current and index not in dataset.holdout_indices
                   and index not in dataset.pilot_pair_excluded]
        per_epoch = math.ceil(len(indices) / args.batch_size)
        expected = per_epoch * args.epochs_per_task
        task_steps = [row for row in steps if row['task'] == task]
        task_refreshes = [row for row in refreshes if row['task'] == task]
        if not old or not indices or len(task_steps) != expected or len(
                task_refreshes) != math.ceil(args.epochs_per_task / 2):
            raise ValueError('selection task step or refresh count differs')
        sampler = BalancedOldReplay(
            {class_id: torch.empty(20) for class_id in old},
            derive_seed(args.seed, 'party_drift_sampler', task))
        refresh_by_epoch = {row['epoch']: row for row in task_refreshes}
        if set(refresh_by_epoch) != set(range(0, args.epochs_per_task, 2)):
            raise ValueError('selection refresh epochs differ')
        if arm == 'cmcdr_budget':
            planned = cmcdr_positions(
                seed=args.seed, task=task, replay_steps=expected,
                batch_size=64, old_classes=len(old),
                probe_events=math.ceil(args.epochs_per_task / 2),
                parties=parties)
            task_budget = [row for row in budgets if row['task'] == task]
            if task_budget != [{
                    'kind': 'cmcdr_budget', 'task': task,
                    'examples': len(planned), 'subsets': 1 << parties,
                    'replay_steps': expected}]:
                raise ValueError('CMCDR budget header differs')
        else:
            planned = set()
            if any(row['task'] == task for row in budgets):
                raise ValueError('non-budget arm wrote a CMCDR budget')
        for epoch in range(args.epochs_per_task):
            if epoch % 2 == 0:
                refresh = refresh_by_epoch[epoch]
                if arm == 'drift':
                    if (refresh.get('kind') != 'probe' or
                            refresh.get('weights') != attenuation_weights(
                                refresh['attenuation'])):
                        raise ValueError('contribution attenuation probe differs')
                    validate_probe_ids(refresh, args.seed, task, epoch, old)
                elif (refresh.get('kind') != 'weight_refresh' or
                      refresh.get('arm') != arm):
                    raise ValueError('control weight refresh differs')
                elif arm == 'static':
                    weights = static_importance_weights(
                        static_frozen, old)
                    if refresh['weights'] != weights:
                        raise ValueError('static weights differ from prior checkpoint')
                elif refresh.get('weights') != [1.0 / parties] * parties:
                    raise ValueError('uniform control weights differ')
            active = refresh_by_epoch[epoch - epoch % 2]
            for local_step in range(per_epoch):
                step = epoch * per_epoch + local_step
                row = task_steps[step]
                validate_cmcdr_fields(row, arm)
                verify_audit_record(row, selected_count)
                _, _, identities, duplicates = sampler.sample()
                selected = (parties if step % 4 == 3 else selected_count)
                if (row.get('task') != task or row.get('epoch') != epoch or
                        row.get('step') != step or row.get('run_seed') != args.seed or
                        row.get('full_every') != 4 or
                        row.get('replay_arm') != arm or
                        row.get('replay_alpha') != args.party_replay_alpha or
                        row.get('weights') != active['weights'] or
                        len(row['selected']) != selected or
                        row.get('replay_batch') != 64 or
                        row.get('embedding_dims') != [args.embed_dim] * parties or
                        row.get('logical_replay_ids') !=
                        [list(item) for item in identities] or
                        row.get('duplicate_samples') != duplicates):
                    raise ValueError('replay selection, samples, or messages differ')
                if arm == 'cmcdr_budget' and set(row['cmcdr_rows']) != {
                        index for index in range(64) if (step, index) in planned}:
                    raise ValueError('budget CMCDR intervention rows differ')
        total_steps += expected
    if (len(steps) != total_steps or len(refreshes) != sum(
            math.ceil(args.epochs_per_task / 2)
            for _ in range(1, args.num_tasks)) or
            len(budgets) != (args.num_tasks - 1 if arm == 'cmcdr_budget' else 0)):
        raise ValueError('selection ledger has extra task records')
    return {
        'selection_audit_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'replay_steps': total_steps,
    }
