"""Two-task, no-test equivalence smoke for the frozen FIM application patch."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from frozen_ablation_manifest import (BASE_COMMIT, PATCH_COMMIT, SOURCE_ROOT,
                                      runtime_identity)

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


def sample_order_sha256(loaders):
    payload = json.dumps(loaders, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def validate_sample_orders(base, default, off):
    records = []
    for mode, entry in (('base', base), ('default_on', default), ('off', off)):
        states = {row.get('task'): row for row in entry.get('states', [])}
        if set(states) != {0, 1}:
            raise ValueError(f'{mode} smoke is missing task sample order')
        records.append((mode, states))
    for task in (0, 1):
        hashes = [states[task].get('sample_order_sha256')
                  for _, states in records]
        if not hashes[0] or len(set(hashes)) != 1:
            raise ValueError(f'smoke sample order differs or is absent at task {task}')
        orders = [entry.get('sample_orders', []) for _, entry in
                  (('base', base), ('default_on', default), ('off', off))]
        task_orders = []
        for records_for_mode in orders:
            task_record = [row for row in records_for_mode
                           if row.get('task') == task]
            if len(task_record) != 1 or sample_order_sha256(
                    task_record[0].get('loaders', [])) != hashes[0]:
                raise ValueError(f'smoke raw sample order is absent or invalid at task {task}')
            task_orders.append(task_record[0])
        if len({json.dumps(row, sort_keys=True) for row in task_orders}) != 1:
            raise ValueError(f'smoke raw sample order differs at task {task}')


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
    orchestration_root = Path(__file__).resolve().parent
    orchestration_commit = subprocess.check_output(
        ['git', '-C', str(orchestration_root), 'rev-parse', 'HEAD'],
        text=True).strip()
    if subprocess.check_output(
            ['git', '-C', str(orchestration_root), 'status', '--porcelain'],
            text=True).strip():
        raise ValueError('smoke orchestration checkout is not clean')
    smoke_script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_path = SOURCE_ROOT / 'full' / dataset / 'adaptive/42/run/config.json'
    source = json.loads(source_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    config = smoke_config(source, mode, output)
    (output / 'run').mkdir()
    smoke_config_bytes = (json.dumps(config, sort_keys=True, indent=2) +
                          '\n').encode('utf-8')
    (output / 'SMOKE_CONFIG.json').write_bytes(smoke_config_bytes)
    smoke_config_sha = hashlib.sha256(smoke_config_bytes).hexdigest()
    sys.path.insert(0, str(code_root))
    import data_utils
    import runner
    import vfl_trainer
    from cl_methods.proto_evolve import ProtoEvolveCL

    train_loaders = []
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
            loader = super()._loader(dataset, indices, key, shuffle, audit=audit)
            if key[0] == 'train':
                sampler = getattr(loader, 'audit_sampler', None)
                if sampler is None:
                    raise ValueError('training sampler does not record sample order')
                train_loaders.append({'task': None,
                                      'loader_key': loader.audit_key,
                                      'sampler': sampler})
            return loader
        def authorize_formal_access(self, **kwargs):
            if kwargs.get('split') == 'test':
                raise RuntimeError('smoke reached official test authorization')
            return super().authorize_formal_access(**kwargs)
        def get_test_loader(self, classes):
            raise RuntimeError('smoke reached official test loader')

    runner.VFLDataset = NoTestDataset
    task = {'id': None}
    applied_freeze = []
    losses, gradients, fim_scores, states, sample_orders = [], [], [], [], []
    original_before = ProtoEvolveCL.before_task
    def before_task(self, task_id, *args):
        task['id'] = int(task_id)
        result = original_before(self, task_id, *args)
        applied_freeze.append({'task': int(task_id),
            'frozen_elements': sum(p.numel() for bottom in self.trainer.bottoms
                                   for p in bottom.parameters() if not p.requires_grad)})
        return result
    ProtoEvolveCL.before_task = before_task
    fim_tensor_values = []
    fim_capture = {'task': None, 'names': [], 'values': []}
    original_compute_fim = ProtoEvolveCL._compute_fim
    original_tensor_mean = torch.Tensor.mean
    def tracked_tensor_mean(self, *args, **kwargs):
        result = original_tensor_mean(self, *args, **kwargs)
        if fim_capture['task'] is not None and not args and not kwargs:
            index = len(fim_capture['values'])
            if index >= len(fim_capture['names']):
                raise ValueError('FIM tensor capture exceeded parameter inventory')
            fim_capture['values'].append({
                'name': fim_capture['names'][index],
                'shape': list(self.shape),
                'dtype': str(self.dtype),
                'sha256': value_hash(self, torch, np),
            })
        return result
    torch.Tensor.mean = tracked_tensor_mean
    def compute_fim(self, loader, task_id):
        fim_capture.update(
            task=int(task_id),
            names=[f'party{party}/{name}'
                   for party, bottom in enumerate(self.trainer.bottoms)
                   for name, _ in bottom.named_parameters()],
            values=[],
        )
        result = original_compute_fim(self, loader, task_id)
        if len(fim_capture['values']) != len(fim_capture['names']):
            raise ValueError('FIM tensor capture missed parameter values')
        fim_tensor_values.append({'task': int(task_id),
                                  'tensors': fim_capture['values']})
        fim_capture['task'] = None
        return result
    ProtoEvolveCL._compute_fim = compute_fim
    original_after = ProtoEvolveCL.after_task
    def after_task(self, loader, task_id):
        result = original_after(self, loader, task_id)
        for entry in train_loaders:
            if entry['task'] is None:
                entry['task'] = int(task_id)
        order_records = []
        for entry in train_loaders:
            if entry['task'] == int(task_id):
                orders = entry['sampler'].epoch_orders
                if not orders:
                    raise ValueError('training sampler recorded no epoch order')
                order_records.append({'loader_key': entry['loader_key'],
                                      'epoch_orders': orders})
        if not order_records:
            raise ValueError('task has no recorded training loader')
        sample_orders.append({'task': int(task_id), 'loaders': order_records})
        states.append({'task': int(task_id),
                       'model_sha256': value_hash(self.trainer.get_state(), torch, np),
                       'memory_mask_sha256': value_hash(self.get_state(), torch, np),
                       'rng_sha256': value_hash(runner._capture_rng_state(), torch, np),
                       'fim_masks_sha256': value_hash(self.fim_masks, torch, np),
                       'fim_tensor_values_sha256': value_hash(
                           [row for row in fim_tensor_values
                            if row['task'] == int(task_id)], torch, np),
                       'sample_order_sha256': sample_order_sha256(order_records)})
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
    torch.Tensor.mean = original_tensor_mean
    no_test_access = {
        'schema_version': 1,
        'status': 'no_test_access',
        'smoke_output_dir': str(output.resolve()),
        'test_accessed': False,
        'stopped_after_two_tasks': True,
        'test_marker_present': False,
        'formal_results_present': False,
        'smoke_config_sha256': smoke_config_sha,
    }
    no_test_bytes = (json.dumps(no_test_access, sort_keys=True, indent=2) +
                     '\n').encode('utf-8')
    (output / 'NO_TEST_ACCESS.json').write_bytes(no_test_bytes)
    record = {'dataset': dataset, 'mode': mode, 'code_commit': expected,
              'smoke_output_dir': str(output.resolve()),
              'orchestration_commit': orchestration_commit,
              'smoke_script_sha256': smoke_script_sha256,
              'source_config_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
              'smoke_config_sha256': smoke_config_sha,
              'no_test_access_sha256': hashlib.sha256(no_test_bytes).hexdigest(),
              'runtime': runtime_identity(),
              'test_accessed': False, 'stopped_after_two_tasks': True,
              'train_per_class': TRAIN_PER_CLASS,
              'states': states,
              'sample_orders': sample_orders,
              'losses': losses, 'gradient_steps': gradients,
              'fim_scores': fim_scores, 'fim_tensor_values': fim_tensor_values,
              'applied_freeze': applied_freeze}
    with (output / 'SMOKE_RESULT.json').open('x') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return record


def _build_smoke_comparison(root, producer_commit=None):
    import numpy as np
    import torch

    root = Path(root).resolve()
    orchestration_root = Path(__file__).resolve().parent
    orchestration_commit = subprocess.check_output(
        ['git', '-C', str(orchestration_root), 'rev-parse', 'HEAD'],
        text=True).strip()
    if subprocess.check_output(
            ['git', '-C', str(orchestration_root), 'status', '--porcelain'],
            text=True).strip():
        raise ValueError('smoke orchestration checkout is not clean')
    if producer_commit is not None:
        orchestration_commit = producer_commit
        script_bytes = subprocess.check_output([
            'git', '-C', str(orchestration_root), 'show',
            f'{producer_commit}:frozen_ablation_smoke.py'])
    else:
        script_bytes = Path(__file__).read_bytes()
    smoke_script_sha256 = hashlib.sha256(script_bytes).hexdigest()
    current_runtime = runtime_identity()
    reports = {}
    for dataset in ('cifar100', 'isolet', 'upmc_food101'):
        entries = {mode: json.loads((root / dataset / mode /
                   'SMOKE_RESULT.json').read_text())
                   for mode in ('base', 'default_on', 'off')}
        source_path = (SOURCE_ROOT / 'full' / dataset /
                       'adaptive/42/run/config.json')
        source = json.loads(source_path.read_text())
        formal_runtime = json.loads((source_path.parent.parent /
                                     'JOB_SUCCESS.json').read_text())['runtime']
        result_sha, config_sha, no_test_sha = {}, {}, {}
        for mode, record in entries.items():
            output = root / dataset / mode
            result_path = output / 'SMOKE_RESULT.json'
            config_path = output / 'SMOKE_CONFIG.json'
            no_test_path = output / 'NO_TEST_ACCESS.json'
            result_sha[mode] = hashlib.sha256(result_path.read_bytes()).hexdigest()
            config_sha[mode] = hashlib.sha256(config_path.read_bytes()).hexdigest()
            no_test_sha[mode] = hashlib.sha256(no_test_path.read_bytes()).hexdigest()
            config = json.loads(config_path.read_text())
            smoke_output = Path(record.get('smoke_output_dir', '')).resolve()
            expected_config = smoke_config(source, mode, smoke_output)
            no_test = json.loads(no_test_path.read_text())
            expected_code = BASE_COMMIT if mode == 'base' else PATCH_COMMIT
            if (not smoke_output.is_absolute() or
                    config != expected_config or
                    no_test.get('schema_version') != 1 or
                    record.get('smoke_config_sha256') != config_sha[mode] or
                    record.get('no_test_access_sha256') != no_test_sha[mode] or
                    no_test.get('status') != 'no_test_access' or
                    no_test.get('test_accessed') is not False or
                    no_test.get('stopped_after_two_tasks') is not True or
                    no_test.get('test_marker_present') is not False or
                    no_test.get('formal_results_present') is not False or
                    no_test.get('smoke_config_sha256') != config_sha[mode] or
                    no_test.get('smoke_output_dir') != str(smoke_output) or
                    record.get('source_config_sha256') != hashlib.sha256(
                        source_path.read_bytes()).hexdigest() or
                    record.get('runtime') != formal_runtime or
                    record.get('runtime') != current_runtime or
                    record.get('orchestration_commit') != orchestration_commit or
                    record.get('smoke_script_sha256') != smoke_script_sha256 or
                    record.get('code_commit') != expected_code or
                    (smoke_output / 'run/formal_access/test.consumed.json').exists() or
                    (smoke_output / 'run/results.json').exists()):
                raise ValueError(f'smoke raw evidence does not match its config/runtime: {dataset}/{mode}')
            if (len(record.get('states', [])) != 2 or
                    [state.get('task') for state in record['states']] != [0, 1] or
                    len(record.get('sample_orders', [])) != 2 or
                    [item.get('task') for item in record['sample_orders']] != [0, 1] or
                    [item.get('task') for item in record.get('applied_freeze', [])] != [0, 1] or
                    {item.get('task') for item in record.get('losses', [])} != {0, 1}):
                raise ValueError(f'smoke raw evidence is incomplete: {dataset}/{mode}')
            tensor_rows = record.get('fim_tensor_values', [])
            if [item.get('task') for item in tensor_rows] != [0, 1]:
                raise ValueError(f'complete FIM tensor evidence is absent: {dataset}/{mode}')
            for task_id, tensor_row in enumerate(tensor_rows):
                tensors = tensor_row.get('tensors', [])
                names = [item.get('name') for item in tensors]
                if (not names or len(names) != len(set(names)) or
                        any(not isinstance(item.get('shape'), list) or
                            not item.get('dtype') or
                            len(item.get('sha256', '')) != 64
                            for item in tensors)):
                    raise ValueError(f'FIM tensor hashes are incomplete: {dataset}/{mode}/task{task_id}')
                state = record['states'][task_id]
                expected_hash = value_hash([tensor_row], torch, np)
                if state.get('fim_tensor_values_sha256') != expected_hash:
                    raise ValueError(f'FIM tensor state hash differs: {dataset}/{mode}/task{task_id}')
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
                      'fim_scores', 'fim_tensor_values', 'applied_freeze'):
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
        if ([row for row in base['fim_tensor_values'] if row['task'] == 0] !=
                [row for row in off['fim_tensor_values'] if row['task'] == 0]):
            raise ValueError(f'FIM-off task-0 FIM tensor values differ: {dataset}')
        validate_sample_orders(base, default, off)
        if (len(off['applied_freeze']) != 2 or
                any(row['frozen_elements'] != 0 for row in off['applied_freeze'])):
            raise ValueError(f'FIM-off applied a nonzero freeze: {dataset}')
        if (len(default['applied_freeze']) != 2 or
                default['applied_freeze'][1]['frozen_elements'] <= 0 or
                off['states'][1]['model_sha256'] == base['states'][1]['model_sha256']):
            raise ValueError(f'FIM-off did not change task-1 training: {dataset}')
        reports[dataset] = {'default_on_bitwise': True,
                            'off_task0_identical': True,
                            'sample_order_identical_all_tasks': True,
                            'fim_tensor_values_compared': True,
                            'off_task1_model_diverged': True,
                            'off_applied_freeze_fraction': 0,
                            'test_accessed': False,
                            'orchestration_commit': orchestration_commit,
                            'smoke_script_sha256': smoke_script_sha256,
                            'raw_smoke_sha256': result_sha,
                            'raw_config_sha256': config_sha,
                            'raw_no_test_sha256': no_test_sha}
    return reports


def validate_smoke_records(root, comparison, producer_commit=None):
    reports = _build_smoke_comparison(root, producer_commit=producer_commit)
    if reports != comparison:
        raise ValueError('preflight comparison does not match raw smoke artifacts')
    return reports


def compare_smokes(root):
    root = Path(root)
    reports = _build_smoke_comparison(root)
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
