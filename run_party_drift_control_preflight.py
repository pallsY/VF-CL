"""Correctness-only two-task pilots for the five new replay controls."""

import argparse
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from party_drift_local import static_importance_weights, verify_audit_record
from run_party_drift_preflight import SOURCE_CONFIG_SHA256, sha256, derive_config
from run_resource_lite_pilot import expected_tasks


CONTROL_ARMS = ('all_party', 'uniform', 'static', 'cmcdr', 'cmcdr_budget')
PROTOCOL = 'party-drift-control-correctness-v1'


def prepare_control_config(base, dataset, arm, root):
    if dataset not in ('isolet', 'upmc_food101', 'cifar100') or arm not in CONTROL_ARMS:
        raise ValueError('unlisted correctness control')
    root = (PurePosixPath(root) if str(root).startswith('/')
            else Path(root).resolve())
    name = f'{dataset}_{arm}_seed74'
    return {
        **base,
        'seed': 74, 'num_tasks': 2,
        'custom_tasks': '|'.join(expected_tasks(dataset).split('|')[:2]),
        'epochs_per_task': 1,
        'device': 'cuda:1' if dataset == 'cifar100' else 'cpu',
        'num_workers': 2,
        'bic_enabled': 0 if dataset == 'cifar100' else base.get('bic_enabled', 0),
        'party_drift_enabled': 1, 'party_drift_preflight': 1,
        'party_replay_arm': arm, 'party_replay_alpha': 1.0,
        'pilot_dataset': dataset, 'pilot_variant': 'reference20',
        'pilot_protocol': PROTOCOL, 'pilot_smoke': True,
        'pilot_holdout_seed': 20261023,
        'pilot_holdout_per_class': 30,
        'results_dir': str(root), 'output_dir': str(root / name),
        'exp_name': name,
    }


def verify_control_steps(steps, arm, num_parties):
    selected_count = (num_parties if arm in
                      ('all_party', 'cmcdr', 'cmcdr_budget') else
                      (num_parties + 1) // 2)
    for index, row in enumerate(steps):
        verify_audit_record(row, selected_count)
        expected = (num_parties if index % 4 == 3 else selected_count)
        if (row['step'] != index or row['task'] != 1 or row['epoch'] != 0 or
                row['replay_arm'] != arm or row['replay_alpha'] != 1.0 or
                len(row['selected']) != expected):
            raise ValueError('control replay step differs')


def run_config(config_path):
    import torch
    import runner
    from config import validate_adaptive_head_consolidation, validate_sample_replay_only
    from data_utils import TaskManager, VFLDataset
    from run_resource_lite_pilot import StopBeforeTest, validate_training_only_artifacts

    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    dataset_name = config.get('pilot_dataset')
    arm = config.get('party_replay_arm')
    args = SimpleNamespace(**config)
    if (config.get('pilot_protocol') != PROTOCOL or arm not in CONTROL_ARMS or
            config.get('pilot_source_config_sha256') !=
            SOURCE_CONFIG_SHA256.get(dataset_name) or
            args.seed != 74 or args.num_tasks != 2 or
            args.epochs_per_task != 1 or args.num_workers != 2 or
            args.device != ('cuda:1' if dataset_name == 'cifar100' else 'cpu') or
            args.custom_tasks != '|'.join(expected_tasks(dataset_name).split('|')[:2]) or
            args.party_drift_enabled != 1 or args.party_drift_preflight != 1 or
            args.party_replay_alpha != 1.0 or
            (dataset_name == 'cifar100' and args.bic_enabled != 0)):
        raise ValueError('control preflight config differs from locked contract')
    validate_sample_replay_only(args)
    validate_adaptive_head_consolidation(args)
    if dataset_name == 'cifar100':
        source = Path(args.data_path) / 'cifar-100-python' / 'train'
        if (sha256(source) !=
                '735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b'
                or not torch.cuda.is_available() or torch.cuda.device_count() < 2):
            raise ValueError('CIFAR source or pilot GPU differs')
    elif sha256(args.vector_npz) != {
            'isolet': 'd34312670de93198afcd2b126c95b79bae2b4cffeb30d480f097faf046b69514',
            'upmc_food101': 'dbdd3a2979203ba0ee804de7d3b128426f869b0a127e4b87ebbe023e5750023b',
            }[dataset_name]:
        raise ValueError('vector source differs from formal data anchor')
    if any(event['type'] != 'CIL' for event in TaskManager(args).get_timeline()):
        raise ValueError('control preflight is not a two-task CIL stream')

    class TestBlockedDataset(VFLDataset):
        def authorize_formal_access(self, **kwargs):
            if kwargs.get('split') == 'test':
                raise StopBeforeTest
            return super().authorize_formal_access(**kwargs)

        def get_test_loader(self, classes):
            raise AssertionError('official test loader forbidden in preflight')

    output = Path(args.output_dir)
    if output != Path(args.results_dir) / f'{dataset_name}_{arm}_seed74':
        raise ValueError('control preflight output path differs')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_bytes(config_path.read_bytes())
    dataset = TestBlockedDataset(args)
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
        raise RuntimeError('control preflight did not stop before official test')
    validate_training_only_artifacts(output)

    audit_path = output / 'party_drift_audit' / 'selection.jsonl'
    records = [json.loads(line) for line in
               audit_path.read_text(encoding='utf-8').splitlines()]
    steps = [row for row in records if 'selected' in row]
    refreshes = [row for row in records if row.get('kind') == 'weight_refresh']
    if (not steps or len(refreshes) != 1 or refreshes[0]['epoch'] != 0 or
            refreshes[0]['arm'] != arm or
            any(row.get('kind') == 'probe' for row in records)):
        raise ValueError('control weight refresh or replay schedule differs')
    verify_control_steps(steps, arm, args.num_parties)
    if arm in ('all_party', 'uniform', 'cmcdr', 'cmcdr_budget') and (
            refreshes[0]['weights'] != [1 / args.num_parties] * args.num_parties):
        raise ValueError('control expected uniform seed weights')
    if arm == 'static':
        event0 = runner._decode_checkpoint_value(torch.load(
            output / 'formal_snapshots' / 'event_0_CIL.pt',
            map_location='cpu', weights_only=True))
        first_classes = list(range(len(expected_tasks(dataset_name).split('|')[0].split(','))))
        contribution = event0['cl_state']['dep_tracker_contrib']
        frozen = {}
        for class_id in first_classes:
            row = contribution[class_id].abs().float()
            frozen[class_id] = (
                row / row.sum() if float(row.sum()) > 0 else
                torch.full_like(row, 1.0 / len(row))
            ).tolist()
        expected = static_importance_weights(frozen, first_classes)
        if refreshes[0]['weights'] != expected:
            raise ValueError('static selection differs from frozen task table')
    if arm in ('cmcdr', 'cmcdr_budget'):
        if any(row['cmcdr_subsets'] != 16 for row in steps):
            raise ValueError('CMCDR did not evaluate all four-party subsets')
        observed = sum(row['cmcdr_examples'] for row in steps)
        if arm == 'cmcdr' and observed != 64 * len(steps):
            raise ValueError('direct CMCDR intervention count differs')
        if arm == 'cmcdr_budget':
            from party_drift_replay import cmcdr_positions
            old_classes = len(expected_tasks(dataset_name).split('|')[0].split(','))
            planned = cmcdr_positions(
                seed=74, task=1, replay_steps=len(steps), batch_size=64,
                old_classes=old_classes, probe_events=1, parties=4)
            actual = {(row['step'], index)
                      for row in steps for index in row['cmcdr_rows']}
            if observed != len(planned) or actual != planned:
                raise ValueError('budget CMCDR intervention count differs')

    final = runner._decode_checkpoint_value(torch.load(
        output / 'checkpoints' / 'formal_final.pt',
        map_location='cpu', weights_only=True))
    method_state = final['cl_state']
    classes = sum(len(task.split(',')) for task in args.custom_tasks.split('|'))
    if (method_state.get('global_protos') != {} or
            method_state.get('prev_protos') != {} or
            set(method_state['head_raw_replay']) != set(range(classes)) or
            any(len(raw) != 20 for raw in method_state['head_raw_replay'].values())):
        raise ValueError('control retained state differs from raw-only budget')
    gate = {
        'status': 'correctness_only_pass', 'dataset': dataset_name,
        'arm': arm, 'seed': 74, 'tasks': 2, 'epochs_per_task': 1,
        'replay_steps': len(steps), 'selection_audit_sha256': sha256(audit_path),
        'checkpoint_sha256': sha256(output / 'checkpoints' / 'formal_final.pt'),
        'test_accessed': False,
    }
    with (output / 'PILOT_GATE.json').open('x', encoding='utf-8') as stream:
        json.dump(gate, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(gate, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--source-config', type=Path, required=True)
    prepare.add_argument('--root', type=Path, required=True)
    prepare.add_argument('--dataset', choices=SOURCE_CONFIG_SHA256, required=True)
    prepare.add_argument('--arm', choices=CONTROL_ARMS, required=True)
    run = sub.add_parser('run')
    run.add_argument('--config', type=Path, required=True)
    options = parser.parse_args()
    if options.command == 'prepare':
        base = derive_config(options.source_config, options.root,
                             options.dataset)
        config = prepare_control_config(base, options.dataset, options.arm,
                                        options.root)
        config['pilot_source_config_sha256'] = sha256(options.source_config)
        options.root.mkdir(parents=True, exist_ok=True)
        path = options.root / (config['exp_name'] + '.config.json')
        with path.open('x', encoding='utf-8') as stream:
            json.dump(config, stream, indent=2, sort_keys=True)
            stream.write('\n')
        print(path)
    else:
        run_config(options.config)


if __name__ == '__main__':
    main()
