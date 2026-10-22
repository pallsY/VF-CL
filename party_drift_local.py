"""Local, training-free preparation for the party-drift replay protocol."""

import argparse
import ast
import hashlib
import json
import math
import pickle
import struct
from pathlib import Path

import numpy as np
import torch

from calibration_split import build_manifest


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def encode_notification(task, epoch, step, party):
    values = (task, epoch, step, party)
    if any(type(value) is not int or not 0 <= value < 2**32 for value in values):
        raise ValueError('notification fields must be uint32')
    return struct.pack('>IIII', *values)


def decode_notification(payload):
    if len(payload) != 16:
        raise ValueError('selection notification must be 16 bytes')
    return struct.unpack('>IIII', payload)


def replay_step_message_bytes(selected, embedding_dims, batch_size,
                              *, value_bytes=4, id_bytes=8):
    if batch_size <= 0 or len(set(selected)) != len(selected) or any(
            party < 0 or party >= len(embedding_dims) for party in selected):
        raise ValueError('invalid replay step')
    return sum(3 * batch_size * embedding_dims[party] * value_bytes +
               batch_size * id_bytes + len(encode_notification(0, 0, 0, party))
               for party in selected)


def attenuation_weights(contribution_attenuation, gamma=0.2, epsilon=1e-8):
    values = np.asarray(contribution_attenuation, dtype=np.float64)
    if (values.ndim != 1 or len(values) == 0 or
            not np.all(np.isfinite(values)) or np.any(values < 0) or
            not 0 < gamma <= 1 or epsilon <= 0):
        raise ValueError('invalid contribution attenuation')
    if float(values.sum()) < epsilon:
        return [1.0 / len(values)] * len(values)
    return ((1 - gamma) * values / values.sum() + gamma / len(values)).tolist()


def build_tuning_holdout(labels, source, expected_sha256, validation, bic,
                         output, *, seed, per_class=30):
    """Write only selected training indices; reject source or split mismatch."""
    source, output = Path(source), Path(output)
    if _sha256(source) != expected_sha256:
        raise ValueError('dataset source SHA-256 differs from frozen anchor')
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError('training labels must be a non-empty vector')
    bic_indices = set()
    if bic is not None:
        expected_bic = build_manifest(labels, bic['per_class'], bic['seed'],
                                      dataset=bic['dataset'])
        if expected_bic != bic:
            raise ValueError('BiC manifest differs from frozen source')
        bic_indices = set(bic['ordered_indices'])
    expected_validation = build_manifest(
        labels, validation['per_class'], validation['seed'],
        dataset=validation['dataset'], excluded_indices=bic_indices,
    )
    if expected_validation != validation:
        raise ValueError('validation manifest differs from frozen source')
    excluded = bic_indices | set(validation['ordered_indices'])
    manifest = build_manifest(
        labels, per_class, seed, dataset='tuning-train-holdout',
        excluded_indices=excluded,
    )
    chosen = set(manifest['ordered_indices'])
    if chosen & excluded:
        raise AssertionError('tuning holdout overlaps a frozen holdout')
    manifest.update({
        'source_sha256': expected_sha256,
        'validation_manifest_sha256': validation['sha256'],
        'bic_manifest_sha256': bic['sha256'] if bic is not None else None,
        'remaining_training_count': len(labels) - len(excluded) - len(chosen),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write('\n')
        handle.flush()
    return manifest


def replay_step_audit(weights, m, run_seed, task, epoch, step, full_every,
                      *, attenuation=None, gamma=0.2,
                      embedding_dims=None, replay_batch=None):
    weights = np.asarray(weights, dtype=np.float64)
    if (weights.ndim != 1 or len(weights) == 0 or
            not np.all(np.isfinite(weights)) or np.any(weights <= 0) or
            not 1 <= m <= len(weights) or full_every < 1):
        raise ValueError('invalid selection weights or cardinality')
    if attenuation is not None and weights.tolist() != attenuation_weights(
            attenuation, gamma):
        raise ValueError('selection weights do not match audited attenuation')
    if step % full_every == full_every - 1:
        selected = list(range(len(weights)))
    else:
        rng = np.random.default_rng(np.random.SeedSequence(
            [run_seed, task, epoch, step]
        ))
        keys = np.log(1.0 - rng.random(len(weights))) / weights
        selected = sorted(range(len(weights)), key=lambda k: (-keys[k], k))[:m]
        selected.sort()
    record = {'weights': weights.tolist(), 'selected': selected,
              'run_seed': run_seed, 'task': task, 'epoch': epoch,
              'step': step, 'full_every': full_every}
    if attenuation is not None:
        record.update({'attenuation': list(attenuation), 'gamma': gamma})
    if (embedding_dims is None) != (replay_batch is None):
        raise ValueError('step message dimensions and batch must be supplied together')
    if embedding_dims is not None:
        record.update({
            'embedding_dims': list(embedding_dims),
            'replay_batch': replay_batch,
            'message_bytes': replay_step_message_bytes(
                selected, embedding_dims, replay_batch),
        })
    return record


def verify_audit_record(record, selected_parties):
    expected = replay_step_audit(
        record['weights'], selected_parties, record['run_seed'],
        record['task'], record['epoch'], record['step'],
        record['full_every'], attenuation=record.get('attenuation'),
        gamma=record.get('gamma', 0.2),
        embedding_dims=record.get('embedding_dims'),
        replay_batch=record.get('replay_batch'),
    )
    if record != expected:
        raise ValueError('selection audit does not replay exactly')


def rows_from_audit(path, step_results):
    """Recover one epoch's task sizes without opening source or test data."""
    counts = {}
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            event = json.loads(line)
            if event.get('loader_iteration') != 0:
                continue
            key = ast.literal_eval(event.get('loader_key', 'None'))
            if not (isinstance(key, tuple) and len(key) == 2 and
                    key[0] == 'train' and isinstance(key[1], tuple)):
                continue
            counts[key[1]] = counts.get(key[1], 0) + len(event['indices'])
    rows = []
    for result in step_results:
        classes = tuple(result['new_classes'])
        if classes not in counts:
            raise ValueError(f'missing first-iteration audit for task {classes}')
        rows.append({'train': counts.pop(classes), 'classes': len(classes)})
    if counts:
        raise ValueError('unexpected task in first-iteration audit')
    return rows


def message_ledger(rows, *, epochs, current_batch, replay_batch,
                   embedding_dims, tuning_per_class, validation_per_class=0,
                   bic_per_class=0, bic_each_stage=False,
                   head_samples_per_class=20, probe_per_class=2,
                   probe_every=1, full_every=4, selected_parties=None,
                   control_bytes=16, id_bytes=8, float_bytes=4):
    """Simulated split-learning payload bytes, excluding test inference."""
    dims = [int(value) for value in embedding_dims]
    parties = len(dims)
    if parties < 2 or len(set(dims)) != 1 or any(value <= 0 for value in dims):
        raise ValueError('pre-run ledger requires equal positive party dimensions')
    if selected_parties is None:
        selected_parties = math.ceil(parties / 2)
    if not (1 <= selected_parties <= parties and epochs > 0 and
            current_batch > 0 and replay_batch > 0 and probe_every > 0 and
            full_every > 0 and control_bytes >= 0):
        raise ValueError('invalid message accounting configuration')
    unit = sum(dims) * float_bytes
    old_classes = 0
    current = full = selected = replay_ids = probe = probe_ids = controls = 0
    fim = herding = bic = steps = party_visits = 0
    total_training = 0
    for task, row in enumerate(rows):
        new_classes = int(row['classes'])
        training = int(row['train']) - tuning_per_class * new_classes
        if training <= 0 or new_classes <= 0:
            raise ValueError('holdout leaves no current-task training examples')
        total_training += training
        task_steps = epochs * math.ceil(training / current_batch)
        current += epochs * training * unit * (2 + (task > 0))
        fim += 2 * training * unit
        herding += training * unit
        old_classes += new_classes
        if bic_each_stage:
            bic += old_classes * bic_per_class * unit
        if task == 0:
            continue
        steps += task_steps
        full_steps = task_steps // full_every
        visits = (task_steps - full_steps) * selected_parties + full_steps * parties
        party_visits += visits
        full += task_steps * replay_batch * unit * 3
        selected += visits * replay_batch * dims[0] * float_bytes * 3
        replay_ids += visits * replay_batch * id_bytes
        controls += visits * control_bytes
        probes = math.ceil(epochs / probe_every)
        probe_samples = probes * probe_per_class * (old_classes - new_classes)
        probe += probe_samples * unit * 2
        probe_ids += probe_samples * parties * id_bytes
        controls += probes * parties * control_bytes
    classes = old_classes
    head = head_samples_per_class * classes * unit
    validation = validation_per_class * classes * unit
    if not bic_each_stage:
        bic = bic_per_class * classes * unit
    tuning = tuning_per_class * classes * unit
    common = fim + herding + head + validation + bic + tuning
    saved = full - selected - replay_ids - probe - probe_ids - controls
    reference = current + full + common
    return {
        'current_bytes': current, 'full_replay_bytes': full,
        'selected_replay_bytes': selected, 'replay_id_bytes': replay_ids,
        'probe_bytes': probe, 'probe_id_bytes': probe_ids,
        'control_bytes': controls, 'fim_bytes': fim,
        'herding_bytes': herding, 'head_bytes': head,
        'validation_bytes': validation, 'bic_bytes': bic,
        'tuning_bytes': tuning, 'common_bytes': common,
        'replay_steps': steps, 'selected_party_visits': party_visits,
        'training_samples_once': total_training,
        'net_saved_bytes': saved, 'total_reference_bytes': reference,
        'net_replay_saving': saved / full if full else 0.0,
        'net_total_saving': saved / reference if reference else 0.0,
    }


def verify_replay_invariants(replay_fn, party_modules, selected):
    """Check only the replay call, after any normal current-task update."""
    selected = set(selected)
    counts = [0] * len(party_modules)
    hooks = [module.register_forward_hook(
        lambda _module, _inputs, _output, k=k: counts.__setitem__(k, counts[k] + 1)
    ) for k, module in enumerate(party_modules)]
    def bn_buffers(module):
        return {name: value.detach().clone() for name, value in module.named_buffers()
                if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    before_bn = [bn_buffers(module) for module in party_modules]
    cpu_state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        loss = replay_fn()
        params = [p for module in party_modules for p in module.parameters()
                  if p.requires_grad]
        gradients = torch.autograd.grad(loss, params, allow_unused=True)
        grad_by_param = dict(zip(params, gradients))
        for k, module in enumerate(party_modules):
            if k in selected:
                if counts[k] == 0:
                    raise AssertionError(f'selected party {k} did not run')
            elif counts[k] != 0:
                raise AssertionError(f'unselected party {k} ran replay forward')
            if k not in selected:
                for param in module.parameters():
                    gradient = grad_by_param.get(param)
                    if gradient is not None and torch.count_nonzero(gradient):
                        raise AssertionError(f'unselected party {k} received replay gradient')
                if any(not torch.equal(before_bn[k][name], value)
                       for name, value in bn_buffers(module).items()):
                    raise AssertionError(f'unselected party {k} changed BN state')
        if not torch.equal(cpu_state, torch.get_rng_state()):
            raise AssertionError('replay changed global CPU RNG state')
        if cuda_state and any(not torch.equal(a, b) for a, b in zip(
                cuda_state, torch.cuda.get_rng_state_all())):
            raise AssertionError('replay changed global CUDA RNG state')
        return {'forward_calls': counts}
    finally:
        for hook in hooks:
            hook.remove()


def _main():
    parser = argparse.ArgumentParser(description='Generate an independent tuning holdout')
    parser.add_argument('--dataset', choices=('isolet', 'upmc_food101', 'cifar100'), required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--anchor', type=Path, required=True)
    parser.add_argument('--formal-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261023)
    args = parser.parse_args()
    anchor = json.loads(args.anchor.read_text(encoding='utf-8'))
    records = anchor['data_sha256'][args.dataset]
    suffix = '/train' if args.dataset == 'cifar100' else '.npz'
    matches = [value for key, value in records.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError('frozen anchor does not identify one training source')
    expected = matches[0]
    if _sha256(args.source) != expected:
        raise ValueError('source hash differs from frozen data anchor')
    if args.dataset == 'cifar100':
        with args.source.open('rb') as handle:
            payload = pickle.load(handle, encoding='bytes')
        labels = np.asarray(payload[b'fine_labels'])
    else:
        with np.load(args.source, allow_pickle=False) as payload:
            labels = payload['y'][payload['train_idx']]
    validation = json.loads((args.formal_run / 'validation' /
                             'validation_manifest.json').read_text(encoding='utf-8'))
    bic_file = args.formal_run / 'bic' / 'calibration_manifest.json'
    bic = json.loads(bic_file.read_text(encoding='utf-8')) if bic_file.exists() else None
    result = build_tuning_holdout(labels, args.source, expected, validation,
                                  bic, args.output, seed=args.seed)
    print(json.dumps({'output': str(args.output), 'sha256': result['sha256'],
                      'count': len(result['ordered_indices'])}, sort_keys=True))


if __name__ == '__main__':
    _main()
