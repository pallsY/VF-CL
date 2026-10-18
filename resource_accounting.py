"""Count learning tensor payloads by role in frozen CL method states."""

import numpy as np
import torch


RAW_DATA_METHODS = {
    'er', 'mir', 'lpr', 'balanced_contrast', 'ocar', 'alsar', 'icarl',
}
BASE_META = {'per_class_size', 'seen_count'}
METHOD_FIELDS = {
    'er': BASE_META | {'data'},
    'mir': BASE_META | {'data'},
    'lpr': BASE_META | {'data'},
    'balanced_contrast': BASE_META | {'data', 'task_number'},
    'ocar': BASE_META | {'data', 'factor_schema_version', 'tau',
                          'steps', 'factors'},
    'alsar': BASE_META | {'data', 'usage', 'step', 'fisher',
                           'top_fisher', 'class_similarity',
                           'last_gradient_mean'},
    'icarl': BASE_META | {'data', 'total_budget', 'new_classes',
                           'old_classes', 'teacher_bottoms', 'teacher_top'},
    'hpcr': {'per_class_size', 'memory', 'seen_count', 'seen_classes', 'step'},
    'der_pp': {'buffer_size', 'num_seen', 'examples', 'labels', 'logits'},
    'er_ace': {'buffer_size', 'num_seen', 'examples', 'labels'},
    'adagauss': {'gaussians', 'adapters'},
    'proto_evolve': {
        'global_protos', 'prev_protos', 'fim_masks',
        'dep_tracker_contrib', 'current_task_classes', 'has_old_teacher',
        'class_party_contrib', 'class_party_weights',
        'effective_class_party_weights', 'party_weight_manifest_hash',
        'party_shuffle_seed', 'party_shuffle_mapping',
        'distill_weight_schedule', 'effective_distill_weight',
        'current_task_id', 'head_raw_replay', 'head_task_classes',
        'head_consolidation_history', 'adaptive_method_version',
        'adaptive_top_version', 'adaptive_class_order', 'adaptive_gate',
        'head_validation_sha256', 'adaptive_pending_task_id',
        'adaptive_audit_bundle',
    },
}


def _leaves(value):
    if isinstance(value, (torch.Tensor, np.ndarray)):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _leaves(item)


def _storage(leaf):
    if isinstance(leaf, torch.Tensor):
        storage = leaf.untyped_storage()
        return ('torch', str(leaf.device), storage.data_ptr()), storage.nbytes()
    root = leaf
    while isinstance(root.base, np.ndarray):
        root = root.base
    pointer = root.__array_interface__['data'][0]
    return ('numpy', pointer), root.nbytes


def payload_bytes(value):
    """Count each underlying tensor/array storage once, including views."""
    seen = {}
    for leaf in _leaves(value):
        key, size = _storage(leaf)
        seen[key] = max(seen.get(key, 0), size)
    return sum(seen.values())


def _example_count(value):
    if value is None:
        return 0
    if not isinstance(value, (torch.Tensor, np.ndarray)) or value.ndim < 1:
        raise ValueError('raw example array is invalid')
    return int(value.shape[0])


def account_method_state(method, state):
    """Return tensor-payload lower bounds with explicit algorithmic roles."""
    if method not in METHOD_FIELDS or type(state) is not dict:
        raise ValueError('unsupported method state')
    allowed = METHOD_FIELDS[method]
    unknown = set(state) - allowed
    if any(payload_bytes(state[key]) for key in unknown):
        raise ValueError(f'unclassified tensor fields: {sorted(unknown)}')
    roles = {name: 0 for name in ('raw', 'logits', 'feature',
                                  'teacher', 'other', 'audit')}
    storages = {}

    def add(role, value):
        for leaf in _leaves(value):
            key, size = _storage(leaf)
            prior = storages.get(key)
            if prior is not None and prior[0] != role:
                raise ValueError('tensor storage appears in multiple roles')
            if prior is None:
                storages[key] = role, size
                roles[role] += size

    count = 0
    for key, value in state.items():
        if key in unknown:
            continue
        if key in ('data', 'head_raw_replay'):
            if type(value) is not dict:
                raise ValueError('raw example mapping is invalid')
            count += sum(_example_count(item) for item in value.values())
            add('raw', value)
        elif key == 'examples':
            count += _example_count(value)
            add('raw', value)
        elif key == 'memory' and method == 'hpcr':
            if type(value) is not dict:
                raise ValueError('HPCR memory is invalid')
            for entries in value.values():
                for entry in entries:
                    if type(entry) not in (tuple, list) or len(entry) != 4:
                        raise ValueError('HPCR entry is invalid')
                    raw, _class_id, logits, features = entry
                    count += 1
                    add('raw', raw)
                    add('logits', logits)
                    add('feature', features)
        elif key in ('logits',):
            add('logits', value)
        elif key in ('global_protos', 'prev_protos', 'gaussians'):
            add('feature', value)
        elif key in ('teacher_bottoms', 'teacher_top'):
            add('teacher', value)
        elif key == 'adaptive_audit_bundle':
            add('audit', value)
        else:
            add('other', value)
    return {
        'raw_examples': count,
        'raw_bytes': roles['raw'],
        'logits_bytes': roles['logits'],
        'feature_bytes': roles['feature'],
        'teacher_bytes': roles['teacher'],
        'other_bytes': roles['other'],
        'audit_bytes': roles['audit'],
        'method_learning_bytes': sum(roles[role] for role in
                                     ('raw', 'logits', 'feature',
                                      'teacher', 'other')),
        'unserialized_teacher': (method == 'proto_evolve'
                                 and state.get('has_old_teacher') is True),
    }
