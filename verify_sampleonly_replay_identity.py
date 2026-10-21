"""Gate-2 fixed-input identity probe; run in separate interpreter per checkout."""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _hash_tensor(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _snapshot(trainer, method):
    model = {}
    for party, bottom in enumerate(trainer.bottoms):
        for name, value in bottom.state_dict().items():
            model[f'bottom{party}.{name}'] = _hash_tensor(value)
    for name, value in trainer.top_model.state_dict().items():
        model[f'top.{name}'] = _hash_tensor(value)
    memory = {str(class_id): _hash_tensor(raw)
              for class_id, raw in method.head_raw_replay.items()}
    return {'model_tensors': model, 'raw_memory': memory,
            'fim_masks': method.fim_masks}


def first_two_tasks(schedule):
    tasks = [[int(value) for value in task.split(',')]
             for task in schedule.split('|')]
    if len(tasks) < 2 or not tasks[0] or not tasks[1]:
        raise ValueError('identity probe requires two complete tasks')
    return tasks[0], tasks[1]


def probe(checkout, source_config, output, replay_enabled=0):
    checkout = Path(checkout).resolve()
    sys.path = [str(checkout)] + [path for path in sys.path
                 if Path(path or os.getcwd()).resolve() != Path(__file__).parent.resolve()]
    os.chdir(checkout)
    from cl_methods.proto_evolve import ProtoEvolveCL
    from data_utils import VFLDataset
    from models import build_models
    from vfl_trainer import VFLTrainer

    for module in (sys.modules['cl_methods.proto_evolve'],
                   sys.modules['data_utils'], sys.modules['models']):
        if not Path(module.__file__).resolve().is_relative_to(checkout):
            raise RuntimeError('identity probe imported code from another checkout')
    config = json.loads(Path(source_config).read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(prefix='vfcl-replay-identity-') as tmp:
        args = SimpleNamespace(**config)
        args.device, args.num_workers = 'cpu', 0
        args.epochs_per_task, args.output_dir = 1, tmp
        args.train_replay_enabled = int(replay_enabled)
        torch.manual_seed(37021)
        np.random.seed(37021)
        random.seed(37021)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        dataset = VFLDataset(args)
        def loader(classes):
            indices = []
            for class_id in classes:
                selected = dataset.get_class_indices(dataset.trainset, [class_id])[:24]
                if len(selected) < 20:
                    raise ValueError('identity probe lacks class examples')
                indices.extend(selected)
            return dataset._loader(
                dataset.trainset, indices,
                ('identity_train', tuple(classes)), False, audit=True)
        bottoms, top = build_models(args)
        trainer = VFLTrainer(bottoms, top, args)
        method = ProtoEvolveCL(trainer, args)
        first_classes, second_classes = first_two_tasks(args.custom_tasks)
        first = loader(first_classes)
        method.before_task(0, first_classes, first_classes)
        method.train_task(first, 0)
        method.after_task(first, 0)
        task0 = _snapshot(trainer, method)
        second = loader(second_classes)
        method.before_task(1, second_classes, first_classes + second_classes)
        batch_x, batch_y = next(iter(second))
        from data_utils import split_features
        import torch.nn.functional as F
        pieces = split_features(batch_x, args)
        embeddings = [bottom(piece) for bottom, piece in zip(trainer.bottoms, pieces)]
        ce = F.cross_entropy(trainer.top_model(trainer._aggregate(embeddings)), batch_y)
        loss = method._proto_loss(trainer.bottoms, trainer.top_model,
                                  batch_x, batch_y, ce)
        with torch.no_grad():
            old_embs = [bottom(piece) for bottom, piece in zip(
                method._old_bottoms, pieces)]
            old_party, old_logits = method._compute_party_logits(
                old_embs, method._old_top)
            new_embs = [bottom(piece) for bottom, piece in zip(
                trainer.bottoms, pieces)]
            new_party, new_logits = method._compute_party_logits(
                new_embs, trainer.top_model)
            width = min(old_logits.size(1), new_logits.size(1))
            temperature = method.temperature
            kd = method.effective_distill_weight * F.kl_div(
                F.log_softmax(new_logits[:, :width] / temperature, dim=1),
                F.softmax(old_logits[:, :width] / temperature, dim=1),
                reduction='batchmean') * temperature * temperature
            party_kd = method.party_kd_lambda * method._party_kd_loss(
                new_party, old_party, list(range(width)), args.device)
            components = {'CE': float(ce).hex(), 'KD': float(kd).hex(),
                          'PartyKD': float(party_kd).hex()}
        loss.backward()
        gradients = {}
        for party, bottom in enumerate(trainer.bottoms):
            for name, parameter in bottom.named_parameters():
                gradients[f'bottom{party}.{name}'] = (
                    _hash_tensor(parameter.grad) if parameter.grad is not None else None)
        for name, parameter in trainer.top_model.named_parameters():
            gradients[f'top.{name}'] = (
                _hash_tensor(parameter.grad) if parameter.grad is not None else None)
        for model in (*trainer.bottoms, trainer.top_model):
            model.zero_grad()
        method.train_task(second, 1)
        method.after_task(second, 1)
        task1 = _snapshot(trainer, method)
    result = {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'checkout': str(checkout),
        'loaded_modules': {name: str(sys.modules[name].__file__) for name in
                           ('cl_methods.proto_evolve', 'data_utils', 'models')},
        'task0': task0, 'task1': task1,
        'task1_probe_loss_hex': float(loss).hex(),
        'task1_probe_components': components,
        'task1_probe_gradients': gradients,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'commit': result['commit'], 'output': str(output)}, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay-enabled', type=int, choices=(0, 1), default=0)
    options = parser.parse_args()
    probe(options.checkout, options.source_config, options.output,
          options.replay_enabled)
