"""Frozen public contract for adaptive dual-branch head consolidation."""

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import numbers
import operator
import re
import sys
from typing import ClassVar

import torch

from head_consolidation import (
    consolidate_classifier,
    consolidate_task_class_bias,
    freeze_state,
    hash_top_state,
)


ADAPTIVE_METHOD_VERSION = 1
FULL_BRANCH_CONFIG = {'mode': 'full_classifier', 'lr': 0.01, 'steps': 500}
BIAS_BRANCH_CONFIG = {'mode': 'task_class_bias', 'lr': 0.03, 'steps': 600}
INACTIVE_BRANCH_CONFIG = {'mode': 'inactive', 'parameters': 0}
SOLVER_TOLERANCE = 1e-12
SOLVER_MAX_ITERATIONS = 80

_PRIMARY_GATE_FIELDS = frozenset({
    'gate_rule', 'is_primary', 'g', 'boundary_derivatives',
    'final_interval', 'final_interval_width', 'iterations', 'converged',
    'tolerance', 'max_iterations', 'full_branch_nll', 'bias_branch_nll',
    'mixture_nll',
})


class _FrozenJSONDict(dict):
    def _immutable(self, *args, **kwargs):
        raise TypeError('adaptive consolidation evidence is immutable')

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable
    __ior__ = _immutable


def _freeze_json(value):
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError('adaptive consolidation evidence keys must be strings')
        return _FrozenJSONDict(
            (key, _freeze_json(item)) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError('adaptive consolidation evidence must be strict JSON data')


def _thaw_json(value):
    if isinstance(value, dict):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _is_sha256(value):
    return (type(value) is str and len(value) == 64
            and all(character in '0123456789abcdef' for character in value))


def _validate_candidate_evidence(pre_hash, hashes, configs):
    if (type(hashes) is not dict
            or set(hashes) != {'pre', 'full', 'bias'}
            or not _is_sha256(pre_hash)
            or any(not _is_sha256(value) for value in hashes.values())
            or hashes['pre'] != pre_hash):
        raise ValueError('adaptive candidate hashes are invalid')
    if type(configs) is not dict or set(configs) != {'full', 'bias'}:
        raise ValueError('adaptive candidate configs are invalid')
    for branch, expected in (
            ('full', FULL_BRANCH_CONFIG), ('bias', BIAS_BRANCH_CONFIG)):
        actual = configs[branch]
        if (type(actual) is not dict or set(actual) != set(expected)
                or any(type(actual[key]) is not type(value)
                       or actual[key] != value
                       for key, value in expected.items())):
            raise ValueError('adaptive candidate configs are invalid')


def _validate_primary_gate(gate):
    if type(gate) is not dict or set(gate) != _PRIMARY_GATE_FIELDS:
        raise ValueError('adaptive primary gate evidence is incomplete')
    if (type(gate['gate_rule']) is not str
            or gate['gate_rule'] != 'class_balanced'
            or gate['is_primary'] is not True
            or gate['converged'] is not True):
        raise ValueError('adaptive primary gate identity is invalid')
    float_fields = (
        'g', 'final_interval_width', 'tolerance', 'full_branch_nll',
        'bias_branch_nll', 'mixture_nll',
    )
    if any(type(gate[name]) is not float or not math.isfinite(gate[name])
           for name in float_fields):
        raise ValueError('adaptive primary gate values are invalid')
    for name in ('boundary_derivatives', 'final_interval'):
        values = gate[name]
        if (type(values) is not list or len(values) != 2
                or any(type(value) is not float or not math.isfinite(value)
                       for value in values)):
            raise ValueError('adaptive primary gate interval is invalid')
    if (type(gate['iterations']) is not int
            or type(gate['max_iterations']) is not int
            or gate['iterations'] < 0
            or gate['iterations'] > gate['max_iterations']
            or gate['max_iterations'] != SOLVER_MAX_ITERATIONS
            or gate['tolerance'] != SOLVER_TOLERANCE):
        raise ValueError('adaptive primary gate solver config is invalid')
    lower, upper = gate['final_interval']
    if (not 0.0 <= lower <= gate['g'] <= upper <= 1.0
            or gate['final_interval_width'] != upper - lower):
        raise ValueError('adaptive primary gate relationship is invalid')


def _validate_validation_manifest(manifest):
    common = {'dataset', 'seed', 'per_class', 'by_class', 'sha256'}
    identity_fields = {'ordered_indices', 'ordered_sample_ids'}
    present = set(manifest) & identity_fields if type(manifest) is dict else set()
    if (type(manifest) is not dict or len(present) != 1
            or set(manifest) != common | present):
        raise ValueError('adaptive validation identity is incomplete')
    identity_field = next(iter(present))
    identity_type = int if identity_field == 'ordered_indices' else str
    if (type(manifest['dataset']) is not str or not manifest['dataset']
            or type(manifest['seed']) is not int
            or type(manifest['per_class']) is not int
            or manifest['per_class'] <= 0
            or not _is_sha256(manifest['sha256'])
            or type(manifest['by_class']) is not dict
            or not manifest['by_class']
            or type(manifest[identity_field]) is not list):
        raise ValueError('adaptive validation identity types are invalid')
    ordered = manifest[identity_field]
    flattened = []
    try:
        class_ids = sorted(manifest['by_class'], key=int)
    except (TypeError, ValueError) as error:
        raise ValueError('adaptive validation class identity is invalid') from error
    for class_id in class_ids:
        identities = manifest['by_class'][class_id]
        if (type(class_id) is not str
                or class_id != str(int(class_id))
                or int(class_id) < 0
                or type(identities) is not list
                or len(identities) != manifest['per_class']
                or any(type(value) is not identity_type for value in identities)):
            raise ValueError('adaptive validation class identity is invalid')
        flattened.extend(identities)
    if (not ordered or flattened != ordered
            or len(set(ordered)) != len(ordered)
            or manifest['sha256'] != hashlib.sha256(json.dumps(
                ordered, separators=(',', ':')
            ).encode('utf-8')).hexdigest()):
        raise ValueError('adaptive validation identities are inconsistent')


@dataclass(frozen=True)
class AdaptiveConsolidationResult:
    pre_head_sha256: str
    candidate_hashes: object
    candidate_configs: object
    gate: object
    validation_manifest: object
    ordered_classes: tuple
    task_id: int
    task_boundary: str
    method_version: int = ADAPTIVE_METHOD_VERSION

    _FIELDS: ClassVar[frozenset] = frozenset({
        'method_version', 'pre_head_sha256', 'candidate_hashes',
        'candidate_configs', 'gate', 'validation_manifest',
        'ordered_classes', 'task_id', 'task_boundary',
    })

    def __post_init__(self):
        if (type(self.method_version) is not int
                or self.method_version != ADAPTIVE_METHOD_VERSION):
            raise ValueError('adaptive method version mismatch')
        if type(self.pre_head_sha256) is not str or not self.pre_head_sha256:
            raise ValueError('adaptive pre-head hash is missing')
        if type(self.task_id) is not int or self.task_id < 0:
            raise ValueError('adaptive task boundary must be non-negative')
        if (type(self.ordered_classes) not in (list, tuple)
                or any(type(class_id) is not int
                       for class_id in self.ordered_classes)):
            raise ValueError('adaptive result classes must be exact integers')
        classes = tuple(self.ordered_classes)
        if not classes or any(
                left >= right for left, right in zip(classes, classes[1:])):
            raise ValueError('adaptive result classes must be strictly ordered')
        object.__setattr__(self, 'ordered_classes', classes)
        boundary = (re.fullmatch(r'event_(\d+)_(CIL|UL)', self.task_boundary)
                    if type(self.task_boundary) is str else None)
        if boundary is None or int(boundary.group(1)) < self.task_id:
            raise ValueError('adaptive task boundary mismatch')
        evidence_names = (
            'candidate_hashes', 'candidate_configs', 'gate',
            'validation_manifest',
        )
        frozen_evidence = {
            name: _freeze_json(getattr(self, name))
            for name in evidence_names
        }
        _validate_candidate_evidence(
            self.pre_head_sha256,
            self.candidate_hashes,
            self.candidate_configs,
        )
        _validate_primary_gate(self.gate)
        _validate_validation_manifest(self.validation_manifest)
        for name in evidence_names:
            object.__setattr__(self, name, frozen_evidence[name])

    @classmethod
    def from_dict(cls, record):
        if type(record) is not dict or set(record) != cls._FIELDS:
            raise ValueError('adaptive consolidation history record is malformed')
        try:
            _freeze_json(record)
            return cls(
                method_version=record['method_version'],
                pre_head_sha256=record['pre_head_sha256'],
                candidate_hashes=record['candidate_hashes'],
                candidate_configs=record['candidate_configs'],
                gate=record['gate'],
                validation_manifest=record['validation_manifest'],
                ordered_classes=record['ordered_classes'],
                task_id=record['task_id'],
                task_boundary=record['task_boundary'],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                'adaptive consolidation history record is invalid'
            ) from error

    def to_dict(self):
        return {
            'method_version': self.method_version,
            'pre_head_sha256': self.pre_head_sha256,
            'candidate_hashes': _thaw_json(self.candidate_hashes),
            'candidate_configs': _thaw_json(self.candidate_configs),
            'gate': _thaw_json(self.gate),
            'validation_manifest': _thaw_json(self.validation_manifest),
            'ordered_classes': list(self.ordered_classes),
            'task_id': int(self.task_id),
            'task_boundary': self.task_boundary,
        }


@dataclass(frozen=True)
class FrozenAdaptiveCandidates:
    pre_head_sha256: str
    full_state: object
    bias_state: object
    full_head_sha256: str
    bias_head_sha256: str
    full_audit: object
    bias_audit: object
    ordered_classes: tuple


def fit_adaptive_candidates(pre_top, replay_embeddings, prototypes,
                            task_classes, persistent_raw_example_count,
                            seed, device):
    """Fit isolated Full and Bias candidates from one immutable head."""
    pre_hash = hash_top_state(pre_top)
    full_top = copy.deepcopy(pre_top)
    bias_top = copy.deepcopy(pre_top)
    full_audit = consolidate_classifier(
        full_top, prototypes, 0.01, 500, 0.01, 20, seed, device,
        replay_embeddings=replay_embeddings,
        replay_source='balanced_current_encoder_raw_replay',
        persistent_raw_example_count=persistent_raw_example_count,
    )
    bias_audit = consolidate_task_class_bias(
        bias_top, replay_embeddings, task_classes, 0.01, 0.01, 1.3,
        600, 0.03, 20, device,
        persistent_raw_example_count=persistent_raw_example_count,
        replay_source='balanced_current_encoder_raw_replay',
    )
    if hash_top_state(pre_top) != pre_hash:
        raise RuntimeError('pre-consolidation head changed during branch fitting')
    return FrozenAdaptiveCandidates(
        pre_head_sha256=pre_hash,
        full_state=freeze_state(full_top.state_dict()),
        bias_state=freeze_state(bias_top.state_dict()),
        full_head_sha256=hash_top_state(full_top),
        bias_head_sha256=hash_top_state(bias_top),
        full_audit=full_audit,
        bias_audit=bias_audit,
        ordered_classes=tuple(sorted(replay_embeddings)),
    )


def fit_fixed_endpoint_candidate(pre_top, branch, replay_embeddings, prototypes,
                                 task_classes, persistent_raw_example_count,
                                 seed, device):
    """Fit exactly one fixed endpoint and leave the other branch empty."""
    if branch not in ('full', 'bias'):
        raise ValueError('fixed endpoint branch must be full or bias')
    pre_hash = hash_top_state(pre_top)
    selected = copy.deepcopy(pre_top)
    if branch == 'full':
        audit = consolidate_classifier(
            selected, prototypes, 0.01, 500, 0.01, 20, seed, device,
            replay_embeddings=replay_embeddings,
            replay_source='balanced_current_encoder_raw_replay',
            persistent_raw_example_count=persistent_raw_example_count,
        )
    else:
        audit = consolidate_task_class_bias(
            selected, replay_embeddings, task_classes, 0.01, 0.01, 1.3,
            600, 0.03, 20, device,
            persistent_raw_example_count=persistent_raw_example_count,
            replay_source='balanced_current_encoder_raw_replay',
        )
    if hash_top_state(pre_top) != pre_hash:
        raise RuntimeError('pre-consolidation head changed during branch fitting')
    selected_state = freeze_state(selected.state_dict())
    inactive_state = freeze_state({})
    selected_hash = hash_top_state(selected_state)
    inactive_hash = hash_top_state(inactive_state)
    inactive_audit = {'mode': 'inactive', 'parameters': 0, 'skipped': True}
    return FrozenAdaptiveCandidates(
        pre_head_sha256=pre_hash,
        full_state=(selected_state if branch == 'full' else inactive_state),
        bias_state=(selected_state if branch == 'bias' else inactive_state),
        full_head_sha256=(selected_hash if branch == 'full' else inactive_hash),
        bias_head_sha256=(selected_hash if branch == 'bias' else inactive_hash),
        full_audit=(audit if branch == 'full' else inactive_audit),
        bias_audit=(audit if branch == 'bias' else inactive_audit),
        ordered_classes=tuple(sorted(replay_embeddings)),
    )


def _loaded_candidate(pre_top, state):
    candidate = copy.deepcopy(pre_top).eval()
    candidate.load_state_dict(state, strict=True)
    return candidate


@torch.no_grad()
def adaptive_candidate_log_probabilities(pre_top, candidates, embeddings):
    """Evaluate only the two frozen candidate heads on frozen embeddings."""
    if hash_top_state(pre_top) != candidates.pre_head_sha256:
        raise RuntimeError('adaptive candidate origin hash mismatch')
    if (hash_top_state(candidates.full_state) != candidates.full_head_sha256
            or hash_top_state(candidates.bias_state) != candidates.bias_head_sha256):
        raise RuntimeError('adaptive candidate hash mismatch')
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError('adaptive validation embeddings must have shape [N, D]')
    if embeddings.size(0) == 0 or not torch.isfinite(embeddings).all():
        raise ValueError('adaptive validation embeddings must be finite and non-empty')

    states = (candidates.full_state, candidates.bias_state)
    if sum(bool(state) for state in states) == 1:
        state = next(state for state in states if state)
        model = _loaded_candidate(pre_top, state)
        device = model.classifier.weight.device
        classes = torch.tensor(
            candidates.ordered_classes, dtype=torch.long, device=device
        )
        logits = model(embeddings.to(device)).index_select(1, classes)
        selected = torch.log_softmax(logits.to(torch.float64), dim=1).cpu()
        return selected.clone(), selected.clone()
    branches = []
    for state in (candidates.full_state, candidates.bias_state):
        model = _loaded_candidate(pre_top, state)
        device = model.classifier.weight.device
        classes = torch.tensor(
            candidates.ordered_classes, dtype=torch.long, device=device
        )
        logits = model(embeddings.to(device)).index_select(1, classes)
        branches.append(torch.log_softmax(logits.to(torch.float64), dim=1).cpu())
    return tuple(branches)


def install_and_reload_verify(pre_top, candidates, gate):
    """Build and strictly reload a complete adaptive top without touching live state."""
    if not isinstance(gate, dict) or gate.get('gate_rule') != 'class_balanced':
        raise ValueError('adaptive gate evidence must use the primary rule')
    if gate.get('is_primary') is not True or gate.get('converged') is not True:
        raise ValueError('adaptive primary gate did not converge')
    if hash_top_state(pre_top) != candidates.pre_head_sha256:
        raise RuntimeError('adaptive candidate origin hash mismatch')
    if (hash_top_state(candidates.full_state) != candidates.full_head_sha256
            or hash_top_state(candidates.bias_state) != candidates.bias_head_sha256):
        raise RuntimeError('adaptive candidate hash mismatch')

    temporary = _loaded_candidate(pre_top, candidates.bias_state)
    classes = torch.tensor(candidates.ordered_classes, dtype=torch.long)
    full_weight = candidates.full_state['classifier.weight'].index_select(0, classes)
    full_bias = candidates.full_state['classifier.bias'].index_select(0, classes)
    temporary.set_adaptive_mixture(
        full_weight, full_bias, gate['g'], candidates.ordered_classes,
        version=ADAPTIVE_METHOD_VERSION,
    )
    installed_state = freeze_state(temporary.state_dict())
    verified = copy.deepcopy(pre_top).eval()
    verified.load_state_dict(installed_state, strict=True)
    if hash_top_state(verified) != hash_top_state(installed_state):
        raise RuntimeError('adaptive strict reload verification failed')
    return verified


def install_fixed_endpoint_and_reload_verify(pre_top, candidates, gate):
    """Install one fitted branch into the live classifier and strictly reload."""
    if not isinstance(gate, dict):
        raise ValueError('fixed endpoint gate evidence is missing')
    branch = {'fixed_full': 'full', 'fixed_bias': 'bias'}.get(
        gate.get('gate_rule')
    )
    expected_gate = {'full': 1.0, 'bias': 0.0}.get(branch)
    if (branch is None or gate.get('is_primary') is not False
            or gate.get('converged') is not True
            or gate.get('g') != expected_gate):
        raise ValueError('fixed endpoint gate evidence is invalid')
    if hash_top_state(pre_top) != candidates.pre_head_sha256:
        raise RuntimeError('adaptive candidate origin hash mismatch')
    selected_state = (candidates.full_state if branch == 'full'
                      else candidates.bias_state)
    inactive_state = (candidates.bias_state if branch == 'full'
                      else candidates.full_state)
    selected_hash = (candidates.full_head_sha256 if branch == 'full'
                     else candidates.bias_head_sha256)
    inactive_hash = (candidates.bias_head_sha256 if branch == 'full'
                     else candidates.full_head_sha256)
    if (not selected_state or dict(inactive_state)
            or hash_top_state(selected_state) != selected_hash
            or hash_top_state(inactive_state) != inactive_hash):
        raise RuntimeError('fixed endpoint candidate hash/state mismatch')
    temporary = _loaded_candidate(pre_top, selected_state)
    temporary.set_adaptive_endpoint(
        expected_gate, candidates.ordered_classes,
        version=ADAPTIVE_METHOD_VERSION,
    )
    installed_state = freeze_state(temporary.state_dict())
    verified = copy.deepcopy(pre_top).eval()
    verified.load_state_dict(installed_state, strict=True)
    if hash_top_state(verified) != hash_top_state(installed_state):
        raise RuntimeError('adaptive strict reload verification failed')
    return verified


def _diagnostic_accuracy(log_probabilities, labels, classes, task_classes):
    predictions = torch.tensor(classes, dtype=torch.long).index_select(
        0, log_probabilities.argmax(dim=1).cpu()
    )
    class_to_task = {
        int(class_id): int(task_id)
        for task_id, members in task_classes.items()
        for class_id in members
    }
    if set(class_to_task) != set(classes):
        raise ValueError('adaptive diagnostic task classes are incomplete')
    task_correct = torch.tensor([
        class_to_task[int(prediction)] == class_to_task[int(label)]
        for prediction, label in zip(predictions, labels)
    ], dtype=torch.float64)
    within_correct = []
    for row, label in enumerate(labels.tolist()):
        members = sorted(
            class_id for class_id, task_id in class_to_task.items()
            if task_id == class_to_task[int(label)]
        )
        columns = torch.tensor(
            [classes.index(class_id) for class_id in members], dtype=torch.long
        )
        predicted = members[int(log_probabilities[row].index_select(
            0, columns
        ).argmax())]
        within_correct.append(predicted == int(label))
    return (
        float(task_correct.mean()),
        float(torch.tensor(within_correct, dtype=torch.float64).mean()),
    )


@torch.no_grad()
def build_adaptive_diagnostics(pre_top, installed_top, candidates,
                               replay_embeddings, validation_embeddings,
                               validation_labels, task_classes, gate):
    """Compute post-freeze diagnostics from training replay/validation only."""
    classes = list(candidates.ordered_classes)
    if sorted(int(class_id) for class_id in replay_embeddings) != classes:
        raise ValueError('adaptive diagnostic replay classes are incomplete')
    replay_rows, replay_labels = [], []
    for class_id in classes:
        rows = replay_embeddings[class_id]
        if not isinstance(rows, torch.Tensor) or rows.ndim != 2 or not rows.numel():
            raise ValueError('adaptive diagnostic replay must contain matrices')
        replay_rows.append(rows.detach().cpu())
        replay_labels.extend([class_id] * rows.size(0))
    replay_x = torch.cat(replay_rows)
    replay_y = torch.tensor(replay_labels, dtype=torch.long)
    validation_x = validation_embeddings.detach().cpu()
    validation_y = validation_labels.detach().cpu().to(torch.long)

    def probabilities(source_x):
        pre_device = pre_top.classifier.weight.device
        selected = torch.tensor(classes, dtype=torch.long, device=pre_device)
        pre = torch.log_softmax(
            pre_top(source_x.to(pre_device)).index_select(1, selected).to(torch.float64),
            dim=1,
        ).cpu()
        full, bias = adaptive_candidate_log_probabilities(
            pre_top, candidates, source_x
        )
        installed_device = installed_top.classifier.weight.device
        mixed = installed_top(source_x.to(installed_device)).to(torch.float64).cpu()
        return pre, full, bias, mixed

    records = {}
    replay_mixture_nll = validation_mixture_nll = None
    for source, source_x, labels in (
            ('replay', replay_x, replay_y),
            ('validation', validation_x, validation_y)):
        pre, full, bias, mixed = probabilities(source_x)
        before_task, before_within = _diagnostic_accuracy(
            pre, labels, classes, task_classes
        )
        after_task, after_within = _diagnostic_accuracy(
            mixed, labels, classes, task_classes
        )
        full_nll = float(class_balanced_mixture_nll(
            full, bias, labels, classes, 1.0
        ))
        bias_nll = float(class_balanced_mixture_nll(
            full, bias, labels, classes, 0.0
        ))
        mixture_nll = float(class_balanced_mixture_nll(
            full, bias, labels, classes, gate['g']
        ))
        records[source] = {
            'count': int(labels.numel()),
            'task_id_accuracy': {'before': before_task, 'after': after_task},
            'within_task_class_accuracy': {
                'before': before_within, 'after': after_within,
            },
            'cross_task_confusion': {
                'before': 1.0 - before_task, 'after': 1.0 - after_task,
            },
            'full_branch_nll': full_nll,
            'bias_branch_nll': bias_nll,
            'mixture_nll': mixture_nll,
        }
        if source == 'replay':
            replay_mixture_nll = mixture_nll
        else:
            validation_mixture_nll = mixture_nll
    if (records['validation']['full_branch_nll'] != gate['full_branch_nll']
            or records['validation']['bias_branch_nll'] != gate['bias_branch_nll']
            or records['validation']['mixture_nll'] != gate['mixture_nll']):
        raise RuntimeError('adaptive diagnostic validation NLL mismatch')
    return {
        'source_splits': {
            'replay': 'persistent_training_replay',
            'validation': 'frozen_training_validation',
            'test_used': False,
        },
        'replay': records['replay'],
        'validation': records['validation'],
        'replay_validation_nll_gap': (
            replay_mixture_nll - validation_mixture_nll
        ),
        'branch_nlls': {
            'full': gate['full_branch_nll'],
            'bias': gate['bias_branch_nll'],
        },
        'g': gate['g'],
        'solver_input': 'class_balanced_validation_nll',
    }


def _validate_log_probabilities(log_p_full, log_p_bias):
    if not isinstance(log_p_full, torch.Tensor) or not isinstance(log_p_bias, torch.Tensor):
        raise TypeError('branch log-probabilities must be torch tensors')
    if log_p_full.dtype != torch.float64 or log_p_bias.dtype != torch.float64:
        raise TypeError('branch log-probabilities must use float64')
    if log_p_full.ndim != 2 or log_p_bias.ndim != 2:
        raise ValueError('branch log-probabilities must have shape [N, C]')
    if log_p_full.shape != log_p_bias.shape or min(log_p_full.shape) == 0:
        raise ValueError('branch log-probability shapes must match and be non-empty')
    if log_p_full.device != log_p_bias.device:
        raise ValueError('branch log-probabilities must be on the same device')
    if not torch.isfinite(log_p_full).all() or not torch.isfinite(log_p_bias).all():
        raise ValueError('branch log-probabilities must be finite')
    expected = torch.zeros(
        log_p_full.shape[0], dtype=torch.float64, device=log_p_full.device
    )
    for branch in (log_p_full, log_p_bias):
        if not torch.allclose(
                torch.logsumexp(branch, dim=1), expected,
                atol=1e-12, rtol=0.0):
            raise ValueError('branch log-probability rows must be normalized')


def _gate_value(g):
    if isinstance(g, bool) or not isinstance(g, numbers.Real):
        raise TypeError('g must be a real scalar')
    value = float(g)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError('g must be finite and in [0, 1]')
    return value


def _validated_problem(log_p_full, log_p_bias, labels, classes):
    _validate_log_probabilities(log_p_full, log_p_bias)
    if not isinstance(labels, torch.Tensor):
        raise TypeError('labels must be a torch tensor')
    integer_dtypes = {
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    }
    if labels.dtype not in integer_dtypes:
        raise TypeError('labels must have an integer dtype')
    if labels.ndim != 1 or labels.numel() != log_p_full.shape[0]:
        raise ValueError('labels must have shape [N]')
    if isinstance(classes, torch.Tensor):
        if classes.ndim != 1 or classes.dtype not in integer_dtypes:
            raise ValueError('classes must be a one-dimensional ordered integer sequence')
        raw_classes = classes.detach().cpu().tolist()
    else:
        try:
            raw_classes = list(classes)
        except TypeError as error:
            raise TypeError('classes must be an ordered integer sequence') from error
    try:
        class_ids = [operator.index(class_id) for class_id in raw_classes]
    except TypeError as error:
        raise TypeError('class IDs must be integers') from error
    if any(isinstance(class_id, bool) for class_id in raw_classes):
        raise TypeError('class IDs must be integers')
    if len(class_ids) != log_p_full.shape[1]:
        raise ValueError('classes must contain one ID per probability column')
    if any(left >= right for left, right in zip(class_ids, class_ids[1:])):
        raise ValueError('class IDs must be unique and strictly ordered')

    class_to_column = {class_id: column for column, class_id in enumerate(class_ids)}
    label_ids = labels.detach().cpu().tolist()
    if any(label not in class_to_column for label in label_ids):
        raise ValueError('every label must occur in the ordered class identity')
    if set(label_ids) != set(class_ids):
        raise ValueError('every ordered class must have at least one label')
    target_columns = torch.tensor(
        [class_to_column[label] for label in label_ids],
        dtype=torch.long,
        device=log_p_full.device,
    )
    return target_columns


def _mix_validated(log_p_full, log_p_bias, g):
    if g == 0.0:
        return log_p_bias.clone()
    if g == 1.0:
        return log_p_full.clone()
    return torch.logaddexp(
        log_p_full + math.log(g),
        log_p_bias + math.log1p(-g),
    )


def mix_log_probabilities(log_p_full, log_p_bias, g):
    """Mix two aligned branch distributions in probability space."""
    _validate_log_probabilities(log_p_full, log_p_bias)
    return _mix_validated(log_p_full, log_p_bias, _gate_value(g))


def _weights(target_columns, class_balanced):
    if not class_balanced:
        return torch.full_like(
            target_columns, 1.0 / target_columns.numel(), dtype=torch.float64
        )
    class_count = int(target_columns.max().item()) + 1
    counts = torch.bincount(target_columns, minlength=class_count).to(torch.float64)
    return 1.0 / (class_count * counts[target_columns])


def _selected_losses(log_p_full, log_p_bias, target_columns, g):
    mixed = _mix_validated(log_p_full, log_p_bias, g)
    rows = torch.arange(target_columns.numel(), device=mixed.device)
    return -mixed[rows, target_columns]


def class_balanced_mixture_nll(log_p_full, log_p_bias, labels, classes, g):
    """Return mean per-class NLL for a probability-space branch mixture."""
    target_columns = _validated_problem(log_p_full, log_p_bias, labels, classes)
    gate = _gate_value(g)
    return (_selected_losses(log_p_full, log_p_bias, target_columns, gate)
            * _weights(target_columns, True)).sum()


def _signed_log_sum(log_terms, positive):
    positive_terms = log_terms[positive]
    negative_terms = log_terms[~positive]
    log_positive = (float(torch.logsumexp(positive_terms, 0).item())
                    if positive_terms.numel() else -math.inf)
    log_negative = (float(torch.logsumexp(negative_terms, 0).item())
                    if negative_terms.numel() else -math.inf)
    if log_positive == log_negative:
        return 0.0
    sign = 1.0 if log_positive > log_negative else -1.0
    high, low = max(log_positive, log_negative), min(log_positive, log_negative)
    if low == -math.inf:
        log_magnitude = high
    else:
        log_magnitude = high + math.log(-math.expm1(low - high))
    if log_magnitude >= math.log(sys.float_info.max):
        return sign * sys.float_info.max
    return sign * math.exp(log_magnitude)


def _derivative(selected_full, selected_bias, weights, g):
    if g == 0.0:
        mixed = selected_bias
    elif g == 1.0:
        mixed = selected_full
    else:
        mixed = torch.logaddexp(
            selected_full + math.log(g),
            selected_bias + math.log1p(-g),
        )
    difference = selected_bias - selected_full
    nonzero = difference != 0
    if not nonzero.any():
        return 0.0
    distance = difference[nonzero].abs()
    high = torch.maximum(selected_full[nonzero], selected_bias[nonzero])
    log_numerator = high + torch.log(-torch.expm1(-distance))
    log_terms = log_numerator - mixed[nonzero] + torch.log(weights[nonzero])
    return _signed_log_sum(log_terms, difference[nonzero] > 0)


def _nll(selected_log_probabilities, weights):
    return float((-selected_log_probabilities * weights).sum().item())


def _derivative_sign(value):
    if abs(value) <= SOLVER_TOLERANCE:
        return 0
    return -1 if value < 0.0 else 1


def _record(gate_rule, is_primary, g, derivatives, interval, iterations,
            converged, selected_full, selected_bias, weights):
    if g == 0.0:
        selected_mixture = selected_bias
    elif g == 1.0:
        selected_mixture = selected_full
    else:
        selected_mixture = torch.logaddexp(
            selected_full + math.log(g),
            selected_bias + math.log1p(-g),
        )
    lower, upper = interval
    return {
        'gate_rule': gate_rule,
        'is_primary': is_primary,
        'g': float(g),
        'boundary_derivatives': [float(derivatives[0]), float(derivatives[1])],
        'final_interval': [float(lower), float(upper)],
        'final_interval_width': float(upper - lower),
        'iterations': int(iterations),
        'converged': bool(converged),
        'tolerance': SOLVER_TOLERANCE,
        'max_iterations': SOLVER_MAX_ITERATIONS,
        'full_branch_nll': _nll(selected_full, weights),
        'bias_branch_nll': _nll(selected_bias, weights),
        'mixture_nll': _nll(selected_mixture, weights),
    }


def _solve(log_p_full, log_p_bias, labels, classes, class_balanced, gate_rule,
           is_primary):
    target_columns = _validated_problem(log_p_full, log_p_bias, labels, classes)
    rows = torch.arange(target_columns.numel(), device=log_p_full.device)
    selected_full = log_p_full[rows, target_columns]
    selected_bias = log_p_bias[rows, target_columns]
    weights = _weights(target_columns, class_balanced)
    derivatives = (
        _derivative(selected_full, selected_bias, weights, 0.0),
        _derivative(selected_full, selected_bias, weights, 1.0),
    )
    left_sign, right_sign = map(_derivative_sign, derivatives)

    if left_sign == 0 and right_sign == 0:
        return _record(gate_rule, is_primary, 0.5, derivatives, (0.5, 0.5), 0,
                       True, selected_full, selected_bias, weights)
    if left_sign >= 0:
        return _record(gate_rule, is_primary, 0.0, derivatives, (0.0, 0.0), 0,
                       True, selected_full, selected_bias, weights)
    if right_sign <= 0:
        return _record(gate_rule, is_primary, 1.0, derivatives, (1.0, 1.0), 0,
                       True, selected_full, selected_bias, weights)

    half_derivative = _derivative(selected_full, selected_bias, weights, 0.5)
    if abs(half_derivative) <= SOLVER_TOLERANCE:
        return _record(gate_rule, is_primary, 0.5, derivatives, (0.5, 0.5), 0,
                       True, selected_full, selected_bias, weights)

    lower, upper = (0.0, 0.5) if half_derivative > 0.0 else (0.5, 1.0)
    iterations = 0
    while upper - lower > SOLVER_TOLERANCE and iterations < SOLVER_MAX_ITERATIONS:
        midpoint = (lower + upper) / 2.0
        midpoint_derivative = _derivative(
            selected_full, selected_bias, weights, midpoint
        )
        if half_derivative > 0.0:
            if midpoint_derivative <= SOLVER_TOLERANCE:
                lower = midpoint
            else:
                upper = midpoint
        elif midpoint_derivative < -SOLVER_TOLERANCE:
            lower = midpoint
        else:
            upper = midpoint
        iterations += 1
    gate = lower if half_derivative > 0.0 else upper
    return _record(
        gate_rule, is_primary, gate, derivatives, (lower, upper), iterations,
        upper - lower <= SOLVER_TOLERANCE,
        selected_full, selected_bias, weights,
    )


def solve_global_mixture_weight(log_p_full, log_p_bias, labels, classes):
    """Solve the primary class-balanced global mixture gate."""
    return _solve(
        log_p_full, log_p_bias, labels, classes, True, 'class_balanced', True
    )


def solve_sample_mean_ablation(log_p_full, log_p_bias, labels, classes):
    """Solve the predeclared ordinary sample-mean NLL ablation."""
    return _solve(
        log_p_full, log_p_bias, labels, classes, False,
        'sample_mean_ablation', False,
    )


def fixed_half_gate_record(log_p_full, log_p_bias, labels, classes):
    """Return the predeclared fixed-g=0.5 ablation audit record."""
    target_columns = _validated_problem(log_p_full, log_p_bias, labels, classes)
    rows = torch.arange(target_columns.numel(), device=log_p_full.device)
    selected_full = log_p_full[rows, target_columns]
    selected_bias = log_p_bias[rows, target_columns]
    weights = _weights(target_columns, True)
    derivatives = (
        _derivative(selected_full, selected_bias, weights, 0.0),
        _derivative(selected_full, selected_bias, weights, 1.0),
    )
    return _record(
        'fixed_half_ablation', False, 0.5, derivatives, (0.5, 0.5), 0, True,
        selected_full, selected_bias, weights,
    )
