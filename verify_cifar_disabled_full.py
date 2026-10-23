"""Training-only CIFAR identity run against an independent checkout.

Run once for each checkout in the same Python/Torch environment. The official
test loader is blocked. The output contains only hashes and BiC fit state, no
accuracy readout.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def snapshot(payload):
    state = payload['trainer_state']
    tensors = {}
    for party, bottom in enumerate(state['bottoms']):
        for name, value in bottom.items():
            tensors[f'bottom{party}.{name}'] = tensor_sha256(value)
    for name, value in state['top_model'].items():
        tensors[f'top.{name}'] = tensor_sha256(value)
    method = payload['cl_state']
    return {
        'model_tensors': tensors,
        'raw_memory': {
            str(label): tensor_sha256(value)
            for label, value in method['head_raw_replay'].items()
        },
        'fim_masks': method['fim_masks'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    sys.path.insert(0, str(checkout))
    os.chdir(checkout)

    import torch
    from torch.utils.data import DataLoader, Subset
    import runner
    from bic_calibration import fit_final_calibrator
    from data_utils import VFLDataset
    from models import build_models
    from vfl_trainer import VFLTrainer

    for name in ('runner', 'bic_calibration', 'data_utils', 'models', 'vfl_trainer'):
        if not Path(sys.modules[name].__file__).resolve().is_relative_to(checkout):
            raise RuntimeError(f'{name} imported outside requested checkout')

    source = json.loads(args.source_config.read_text(encoding='utf-8'))
    if (file_sha256(args.source_config)
            != '673ca34abe87179a75abff9d10d7e7a7392f786587df8c4d668340b57418a8c2'
            or source.get('data') != 'cifar100'
            or source.get('num_tasks') != 10
            or source.get('sample_replay_only') != 1
            or source.get('bic_fit_mode') != 'joint_each_stage'):
        raise ValueError('source config is not frozen formal CIFAR')
    if file_sha256(Path(source['data_path']) / 'cifar-100-python' / 'train') != (
            '735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b'):
        raise ValueError('formal CIFAR training file differs')
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError('requested CUDA device is unavailable')

    output = args.output_root.resolve()
    config = {
        **source,
        'seed': 73,
        'epochs_per_task': 1,
        'device': 'cuda:1',
        'num_workers': 2,
        'party_drift_enabled': 0,
        'party_drift_preflight': 0,
        'output_dir': str(output),
        'results_dir': str(output.parent),
        'exp_name': output.name,
    }

    class StopBeforeTest(Exception):
        pass

    class TestBlockedDataset(VFLDataset):
        def authorize_formal_access(self, **kwargs):
            if kwargs.get('split') == 'test':
                raise StopBeforeTest
            return super().authorize_formal_access(**kwargs)

        def get_test_loader(self, classes):
            raise AssertionError('official test loader forbidden in identity run')

    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(
        json.dumps(config, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    run_args = SimpleNamespace(**config)
    dataset = TestBlockedDataset(run_args)
    original = runner.VFLDataset
    runner.VFLDataset = lambda _args: dataset
    stopped = False
    try:
        runner.run_experiment(run_args)
    except StopBeforeTest:
        stopped = True
    finally:
        runner.VFLDataset = original
    if not stopped or (output / 'results.json').exists():
        raise RuntimeError('identity run did not stop before official test')

    snapshots = {}
    for task in range(10):
        path = output / 'formal_snapshots' / f'event_{task}_CIL.pt'
        payload = runner._decode_checkpoint_value(torch.load(
            path, map_location='cpu', weights_only=True))
        snapshots[str(task)] = snapshot(payload)
    final_path = output / 'checkpoints' / 'formal_final.pt'
    final = runner._decode_checkpoint_value(torch.load(
        final_path, map_location='cpu', weights_only=True))
    bottoms, top = build_models(run_args)
    trainer = VFLTrainer(bottoms, top, run_args)
    trainer.load_state(final['trainer_state'])
    classes = list(range(100))
    view_indices = getattr(dataset, 'validation_view_indices', dataset.validation_indices)
    selected = [index for index in dataset.get_class_indices(dataset.validationset, classes)
                if index in view_indices]
    if len(selected) != 2500:
        raise ValueError('authoritative BiC validation split differs')
    loader = DataLoader(Subset(dataset.validationset, selected),
                        batch_size=run_args.batch_size, shuffle=False,
                        num_workers=0)
    logits, labels = trainer.collect_logits(loader)
    task_classes = {task: list(range(task * 10, (task + 1) * 10))
                    for task in range(10)}
    calibrator, fit = fit_final_calibrator(
        logits, labels, task_classes, 'joint_alpha_beta',
        run_args.bic_lr, run_args.bic_steps,
    )
    record = {
        'status': 'test_blocked_identity_only',
        'checkout': str(checkout),
        'source_config_sha256': file_sha256(args.source_config),
        'validation_manifest_sha256': file_sha256(
            output / 'bic' / 'calibration_manifest.json'),
        'snapshots': snapshots,
        'final': snapshot(final),
        'bic': {
            'input_logits_sha256': tensor_sha256(logits),
            'labels_sha256': tensor_sha256(labels),
            'fit': fit,
            'state': calibrator.state_dict(),
            'output_logits_sha256': tensor_sha256(calibrator.apply(logits)),
        },
    }
    with (output / 'IDENTITY_RECORD.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'status': record['status'], 'output': str(output)},
                     sort_keys=True))


if __name__ == '__main__':
    main()
