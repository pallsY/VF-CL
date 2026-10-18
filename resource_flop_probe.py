"""Development-only model FLOP proxy from a pinned frozen train state.

Profiles one synthetic current-task minibatch in memory. The counter covers
supported tensor operators; it is not total experiment FLOPs or test evidence.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.flop_counter import FlopCounterMode


def parse_task_classes(config):
    if config.get('custom_tasks'):
        tasks = [[int(label) for label in group.split(',')]
                 for group in config['custom_tasks'].split('|')]
    else:
        width = config['classes_per_task']
        tasks = [list(range(task * width, (task + 1) * width))
                 for task in range(config['num_tasks'])]
    classes = [label for group in tasks for label in group]
    if (len(tasks) != config['num_tasks'] or not all(tasks)
            or sorted(classes) != list(range(config['num_classes']))):
        raise ValueError('profile task order differs')
    return tasks


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_profile_inputs(job, source, task):
    job, source = Path(job), Path(source)
    config = json.loads((job / 'run/config.json').read_text(encoding='utf-8'))
    tasks = parse_task_classes(config)
    if (not isinstance(task, int) or not 0 <= task < len(tasks)
            or config.get('num_parties') != 4):
        raise ValueError('profile task or party count differs')
    plan = json.loads((job / 'JOB_PLAN.json').read_text(encoding='utf-8'))
    commit = plan.get('code_commit')
    head = subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(
        ['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    if commit != head or dirty:
        raise ValueError('profile producer source differs')
    if task == 0:
        return config, None, commit
    frozen = json.loads((job / 'run/FORMAL_STATE_FROZEN.json').read_text(
        encoding='utf-8'))['identity']
    relative = f'formal_snapshots/event_{task - 1}_CIL.pt'
    try:
        expected = frozen['snapshots'][task - 1]['file']
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError('profile snapshot inventory differs') from error
    path = job / 'run' / relative
    if (expected.get('path') != relative or path.is_symlink()
            or _sha(path) != expected.get('sha256')):
        raise ValueError('profile snapshot bytes differ')
    return config, path, commit


def _input_shape(config):
    batch = int(config['batch_size'])
    if config['data'] == 'cifar100':
        return batch, 3, 32, 32
    if config['data'] == 'tabvfl' and config['num_classes'] == 101:
        return batch, 1536
    if config['data'] == 'tabvfl' and config['num_classes'] == 26:
        return batch, 617
    raise ValueError('profile input shape is outside the three datasets')


def profile_one(job, source, task):
    config, snapshot_path, commit = verify_profile_inputs(job, source, task)
    sys.path.insert(0, str(Path(source).resolve()))
    from adaptive_consolidation_audit import (_fresh_formal_state,
                                               _safe_torch_load)
    from cl_methods import get_cl_method
    from models import build_models
    from vfl_trainer import VFLTrainer

    torch.set_num_threads(1)
    torch.manual_seed(0)
    random.seed(0)
    payload = (_safe_torch_load(snapshot_path)
               if snapshot_path is not None else None)
    classes = parse_task_classes(config)
    with tempfile.TemporaryDirectory(prefix='vfcl_resource_flops_') as scratch:
        options = SimpleNamespace(**config)
        options.device = 'cpu'
        options.num_workers = 0
        options.output_dir = scratch
        options.results_dir = scratch
        if config['data'] == 'tabvfl':
            options.party_col_ranges = (
                [(0, 256), (256, 512), (512, 1024), (1024, 1536)]
                if config['num_classes'] == 101 else
                [(0, 155), (155, 309), (309, 463), (463, 617)]
            )
        if payload is None:
            bottoms, top = build_models(options)
            trainer = VFLTrainer(bottoms, top, options)
            method = get_cl_method(config['cl_method'], trainer, options)
        else:
            trainer, method, _ = _fresh_formal_state(payload, options)
        seen = [label for group in classes[:task + 1] for label in group]
        method.before_task(task, classes[task], seen)
        for holder in (options, method.args, trainer.args):
            holder.epochs_per_task = 1
            holder.data_flow_audit = 0
        shape = _input_shape(config)
        x = torch.randn(shape)
        y = torch.tensor([classes[task][index % len(classes[task])]
                          for index in range(shape[0])], dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=shape[0],
                            shuffle=False, num_workers=0)
        with FlopCounterMode(display=False) as counter:
            method.train_task(loader, task)
    return {
        'schema_version': 1,
        'job': '/'.join(Path(job).parts[-3:]),
        'producer_commit': commit,
        'snapshot_sha256': (_sha(snapshot_path)
                            if snapshot_path is not None else None),
        'profile_task': task,
        'input_shape': list(shape),
        'flop_counter_total': counter.get_total_flops(),
        'torch_version': torch.__version__,
        'device': 'cpu',
        'probe_script_sha256': _sha(__file__),
        'input_origin': 'synthetic_current_task_shape; no test data',
        'coverage': 'torch FlopCounterMode supported ops only; no final-head fit',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = profile_one(args.job, args.source, args.task)
    body = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as handle:
            handle.write(body)
    print('RESOURCE_FLOP_JSON=' + json.dumps(result, sort_keys=True))
