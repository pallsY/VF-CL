"""Two-task, no-test equivalence smoke for the frozen FIM application patch."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from frozen_ablation_manifest import BASE_COMMIT, PATCH_COMMIT, SOURCE_ROOT

TRAIN_PER_CLASS = 64


def smoke_config(source, mode, output):
    if mode not in ('base', 'default_on', 'off'):
        raise ValueError('unknown smoke mode')
    config = dict(source)
    config.update(epochs_per_task=1, device='cpu', num_workers=2,
                  results_dir=str(output), output_dir=str(output / 'run'),
                  exp_name='run')
    if mode != 'base':
        config['fim_apply_mask'] = 0 if mode == 'off' else 1
    return config


def value_hash(value, torch, np):
    digest = hashlib.sha256()
    def feed(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b'T' + str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(b'N' + str(item.dtype).encode() + str(item.shape).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b'D')
            for key in sorted(item, key=repr):
                feed(key); feed(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b'L' if isinstance(item, list) else b'Q')
            digest.update(str(len(item)).encode())
            for part in item:feed(part)
        else:
            digest.update(b'S' + repr(item).encode())
    feed(value)
    return digest.hexdigest()


def run_smoke(dataset, mode, code_root, output):
    import numpy as np
    import torch

    expected = BASE_COMMIT if mode == 'base' else PATCH_COMMIT
    head = subprocess.check_output(['git', '-C', str(code_root), 'rev-parse', 'HEAD'],
                                   text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(code_root), 'status', '--porcelain'],
                                    text=True).strip()
    if head != expected or dirty:
        raise ValueError('smoke training code differs from pinned commit')
    source_path = SOURCE_ROOT / 'full' / dataset / 'adaptive/42/run/config.json'
    source = json.loads(source_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    config = smoke_config(source, mode, output)
    (output / 'run').mkdir()
    (output / 'SMOKE_CONFIG.json').write_text(
        json.dumps(config, sort_keys=True, indent=2) + '\n')
    sys.path.insert(0, str(code_root))
    import data_utils
    import runner
    import vfl_trainer
    from cl_methods.proto_evolve import ProtoEvolveCL

    class StopAfterTwoTasks(Exception):
        pass
    class NoTestDataset(data_utils.VFLDataset):
        def _loader(self, dataset, indices, key, shuffle, audit=False):
            if key[0] == 'train':
                counts, selected = {}, []
                for index in indices:
                    label = int(self.trainset.targets[index])
                    if counts.get(label, 0) < TRAIN_PER_CLASS:
                        selected.append(index)
                        counts[label] = counts.get(label, 0) + 1
                indices = selected
            return super()._loader(dataset, indices, key, shuffle, audit=audit)
        def authorize_formal_access(self, **kwargs):
            if kwargs.get('split') == 'test':
                raise RuntimeError('smoke reached official test authorization')
            return super().authorize_formal_access(**kwargs)
        def get_test_loader(self, classes):
            raise RuntimeError('smoke reached official test loader')

    runner.VFLDataset = NoTestDataset
    task = {'id': None}
    applied_freeze = []
    losses, gradients, fim_scores, states = [], [], [], []
    original_before = ProtoEvolveCL.before_task
    def before_task(self, task_id, *args):
        task['id'] = int(task_id)
        result = original_before(self, task_id, *args)
        applied_freeze.append({'task': int(task_id),
            'frozen_elements': sum(p.numel() for bottom in self.trainer.bottoms
                                   for p in bottom.parameters() if not p.requires_grad)})
        return result
    ProtoEvolveCL.before_task = before_task
    original_after = ProtoEvolveCL.after_task
    def after_task(self, loader, task_id):
        result = original_after(self, loader, task_id)
        states.append({'task': int(task_id),
                       'model_sha256': value_hash(self.trainer.get_state(), torch, np),
                       'memory_mask_sha256': value_hash(self.get_state(), torch, np),
                       'rng_sha256': value_hash(runner._capture_rng_state(), torch, np),
                       'fim_masks_sha256': value_hash(self.fim_masks, torch, np)})
        if int(task_id) == 1:
            raise StopAfterTwoTasks
        return result
    ProtoEvolveCL.after_task = after_task
    original_epoch = vfl_trainer.VFLTrainer.train_epoch
    def train_epoch(self, *args, **kwargs):
        value = original_epoch(self, *args, **kwargs)
        losses.append({'task': task['id'], 'loss': value[0], 'accuracy': value[1]})
        return value
    vfl_trainer.VFLTrainer.train_epoch = train_epoch
    original_quantile = torch.quantile
    def quantile(values, *args, **kwargs):
        result = original_quantile(values, *args, **kwargs)
        fim_scores.append({'task': task['id'],
                           'values_sha256': value_hash(values, torch, np),
                           'threshold_sha256': value_hash(result, torch, np)})
        return result
    torch.quantile = quantile
    for optimizer_class in (torch.optim.SGD, torch.optim.AdamW):
        original_step = optimizer_class.step
        def step(self, *args, _original=original_step, **kwargs):
            grads = [parameter.grad for group in self.param_groups
                     for parameter in group['params']]
            gradients.append({'task': task['id'],
                              'sha256': value_hash(grads, torch, np)})
            return _original(self, *args, **kwargs)
        optimizer_class.step = step
    torch.set_num_threads(1)
    stopped = False
    try:
        runner.run_experiment(SimpleNamespace(**config))
    except StopAfterTwoTasks:
        stopped = True
    if (not stopped or
            (output / 'run/formal_access/test.consumed.json').exists() or
            (output / 'run/results.json').exists()):
        raise ValueError('smoke reached official test or failed to stop after two tasks')
    if len(states) != 2 or [row['task'] for row in states] != [0, 1]:
        raise ValueError('smoke did not capture both task boundaries')
    data_flow = output / 'run/data_flow_audit.jsonl'
    if not data_flow.is_file():
        raise ValueError('smoke training data flow was not recorded')
    record = {'dataset': dataset, 'mode': mode, 'code_commit': expected,
              'source_config_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
              'test_accessed': False, 'stopped_after_two_tasks': True,
              'train_per_class': TRAIN_PER_CLASS,
              'data_flow_sha256': hashlib.sha256(data_flow.read_bytes()).hexdigest(),
              'states': states,
              'losses': losses, 'gradient_steps': gradients,
              'fim_scores': fim_scores,
              'applied_freeze': applied_freeze}
    with (output / 'SMOKE_RESULT.json').open('x') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return record


def compare_smokes(root):
    reports = {}
    for dataset in ('cifar100', 'isolet', 'upmc_food101'):
        entries = {mode: json.loads((root / dataset / mode /
                   'SMOKE_RESULT.json').read_text())
                   for mode in ('base', 'default_on', 'off')}
        base, default, off = (entries[mode]
                              for mode in ('base', 'default_on', 'off'))
        if (any(entries[mode].get('dataset') != dataset or
                entries[mode].get('mode') != mode or
                entries[mode].get('test_accessed') is not False or
                entries[mode].get('stopped_after_two_tasks') is not True or
                entries[mode].get('train_per_class') != TRAIN_PER_CLASS or
                entries[mode].get('code_commit') !=
                (BASE_COMMIT if mode == 'base' else PATCH_COMMIT)
                for mode in entries) or
                len({entries[mode].get('source_config_sha256')
                     for mode in entries}) != 1):
            raise ValueError(f'smoke provenance differs: {dataset}')
        for field in ('states', 'losses', 'gradient_steps',
                      'fim_scores', 'applied_freeze', 'data_flow_sha256'):
            if base[field] != default[field]:
                raise ValueError(f'default-on differs from producer: {dataset}, {field}')
        for field in ('model_sha256', 'memory_mask_sha256', 'rng_sha256',
                      'fim_masks_sha256'):
            if base['states'][0][field] != off['states'][0][field]:
                raise ValueError(f'FIM-off task-0 differs: {dataset}, {field}')
        for field in ('losses', 'gradient_steps', 'fim_scores'):
            if ([row for row in base[field] if row['task'] == 0] !=
                    [row for row in off[field] if row['task'] == 0]):
                raise ValueError(f'FIM-off task-0 {field} differs: {dataset}')
        if off['data_flow_sha256'] != base['data_flow_sha256']:
            raise ValueError(f'FIM-off data order differs: {dataset}')
        if (len(off['applied_freeze']) != 2 or
                any(row['frozen_elements'] != 0 for row in off['applied_freeze'])):
            raise ValueError(f'FIM-off applied a nonzero freeze: {dataset}')
        if (len(default['applied_freeze']) != 2 or
                default['applied_freeze'][1]['frozen_elements'] <= 0 or
                off['states'][1]['model_sha256'] == base['states'][1]['model_sha256']):
            raise ValueError(f'FIM-off did not change task-1 training: {dataset}')
        reports[dataset] = {'default_on_bitwise': True,
                            'off_task0_identical': True,
                            'off_applied_freeze_fraction': 0,
                            'test_accessed': False}
    with (root / 'SMOKE_COMPARISON.json').open('x') as stream:
        json.dump(reports, stream, sort_keys=True, indent=2)
        stream.write('\n')
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('cifar100', 'isolet', 'upmc_food101'))
    parser.add_argument('--mode', choices=('base', 'default_on', 'off'))
    parser.add_argument('--code-root', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    if args.compare:
        print(json.dumps(compare_smokes(args.output_root), sort_keys=True))
    elif args.dataset and args.mode and args.code_root:
        print(json.dumps(run_smoke(args.dataset, args.mode,
                                   args.code_root.resolve(),
                                   args.output_root / args.dataset / args.mode),
                         sort_keys=True))
    else:
        raise ValueError('smoke needs dataset, mode, and pinned code root')


if __name__ == '__main__':
    main()
