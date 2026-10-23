"""Two-task, one-epoch correctness pilot. Stops before formal test access."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from party_drift_local import verify_audit_record
from run_resource_lite_pilot import (
    PilotDataset, StopBeforeTest, _identity, expected_tasks,
    validate_source_data_hash, validate_training_only_artifacts,
)


PROTOCOL = 'party-drift-correctness-preflight-v1'
SOURCE_CONFIG_SHA256 = {
    'isolet': '9a524718f29e3dbae0a0157a875ce2b4eccb27aa51db7f757dd00858240bc8db',
    'upmc_food101': '4e8c76695ad2c8cd4d2ddd895e134b386597281a756d80c2c3555ffb5a9042e1',
    'cifar100': '673ca34abe87179a75abff9d10d7e7a7392f786587df8c4d668340b57418a8c2',
}
HOLDOUT_SHA256 = {
    'isolet': '668cf1ce874c55a587647940d58181d7a42dc0ae0aa75c1dd4625004ef923089',
    'upmc_food101': 'd19918d00acf93cd65f3095ea3a61914da54144e192ee0fd411d76bfaf3e595f',
    'cifar100': '55b8130fee6f828d0e6be8fbc39e8db70f05aebb55130073da47873c77a9e5ea',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def derive_config(source_path, root, dataset):
    source_path, root = Path(source_path), Path(root)
    if dataset not in SOURCE_CONFIG_SHA256 or sha256(source_path) != SOURCE_CONFIG_SHA256[dataset]:
        raise ValueError('source config differs from frozen sample-only run')
    source = json.loads(source_path.read_text(encoding='utf-8'))
    _identity(source, dataset)
    if (source.get('seed') != 42 or source.get('sample_replay_only') != 1
            or source.get('head_consolidation_samples_per_class') != 20
            or source.get('formal_deferred_evaluation') is not True):
        raise ValueError('source config is not frozen reference20')
    name = f'{dataset}_seed73_two_task_correctness'
    return {
        **source,
        'seed': 73, 'pilot_dataset': dataset,
        'pilot_variant': 'reference20', 'pilot_protocol': PROTOCOL,
        'pilot_smoke': True, 'pilot_holdout_seed': 20261023,
        'pilot_holdout_per_class': 30,
        'party_drift_enabled': 1, 'party_drift_preflight': 1,
        'num_tasks': 2, 'custom_tasks': '|'.join(expected_tasks(dataset).split('|')[:2]),
        'epochs_per_task': 1, 'device': 'cpu', 'num_workers': 0,
        'results_dir': str(root), 'output_dir': str(root / name),
        'exp_name': name,
    }


def run_config(config_path):
    import torch
    import runner
    from config import validate_adaptive_head_consolidation, validate_sample_replay_only
    from data_utils import TaskManager

    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    args = SimpleNamespace(**config)
    dataset_name = args.pilot_dataset
    if (args.pilot_protocol != PROTOCOL or dataset_name not in SOURCE_CONFIG_SHA256
            or args.seed != 73 or args.num_tasks != 2 or args.epochs_per_task != 1
            or args.device != 'cpu' or args.num_workers != 0
            or args.party_drift_enabled != 1 or args.party_drift_preflight != 1
            or args.pilot_holdout_seed != 20261023
            or args.pilot_holdout_per_class != 30
            or args.custom_tasks != '|'.join(expected_tasks(dataset_name).split('|')[:2])
            or args.formal_deferred_evaluation is not True):
        raise ValueError('correctness-only preflight contract differs')
    _identity(config, dataset_name, smoke=True)
    validate_sample_replay_only(args)
    validate_adaptive_head_consolidation(args)
    validate_source_data_hash(config)
    if any(event['type'] != 'CIL' for event in TaskManager(args).get_timeline()):
        raise ValueError('preflight is not a two-task CIL stream')
    run = Path(args.output_dir)
    if run != Path(args.results_dir) / args.exp_name:
        raise ValueError('preflight result path differs')
    run.mkdir(parents=True, exist_ok=False)
    (run / 'config.json').write_bytes(config_path.read_bytes())
    dataset = PilotDataset(args)
    if dataset.holdout_manifest['sha256'] != HOLDOUT_SHA256[dataset_name]:
        raise ValueError('preflight holdout differs from locked 30/class split')
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
        raise RuntimeError('preflight did not stop before official test')
    validate_training_only_artifacts(run)
    audit = run / 'party_drift_audit' / 'selection.jsonl'
    records = [json.loads(line) for line in audit.read_text(encoding='utf-8').splitlines()]
    probes = [row for row in records if row.get('kind') == 'probe']
    steps = [row for row in records if 'selected' in row]
    if len(probes) != 1 or probes[0]['task'] != 1 or probes[0]['epoch'] != 0:
        raise ValueError('two-task probe schedule differs')
    for index, row in enumerate(steps):
        verify_audit_record(row, (args.num_parties + 1) // 2)
        if row['step'] != index or row['task'] != 1 or row['epoch'] != 0:
            raise ValueError('replay step order differs')
    checkpoint = run / 'checkpoints' / 'formal_final.pt'
    payload = runner._decode_checkpoint_value(torch.load(
        checkpoint, map_location='cpu', weights_only=True))
    state = payload['cl_state']
    if (state.get('global_protos') != {} or state.get('prev_protos') != {}
            or set(state['head_raw_replay']) != set(
                range(len(config['custom_tasks'].split('|')[0].split(',')) +
                      len(config['custom_tasks'].split('|')[1].split(','))))
            or any(len(raw) != 20 for raw in state['head_raw_replay'].values())):
        raise ValueError('preflight retained non-raw replay state')
    gate = {
        'status': 'correctness_only_pass', 'dataset': dataset_name,
        'seed': 73, 'tasks': 2, 'epochs_per_task': 1,
        'probe_events': len(probes), 'replay_steps': len(steps),
        'selection_audit_sha256': sha256(audit),
        'checkpoint_sha256': sha256(checkpoint),
        'holdout_manifest_sha256': sha256(run / 'pilot_holdout' / 'manifest.json'),
        'test_accessed': False,
    }
    with (run / 'PILOT_GATE.json').open('x', encoding='utf-8') as handle:
        json.dump(gate, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(gate, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--source-config', type=Path, required=True)
    prepare.add_argument('--root', type=Path, required=True)
    prepare.add_argument('--dataset', choices=SOURCE_CONFIG_SHA256, required=True)
    run = sub.add_parser('run')
    run.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        config = derive_config(args.source_config, args.root, args.dataset)
        args.root.mkdir(parents=True, exist_ok=True)
        path = args.root / (config['exp_name'] + '.config.json')
        with path.open('x', encoding='utf-8') as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
            handle.write('\n')
        print(path)
    else:
        run_config(args.config)


if __name__ == '__main__':
    main()
