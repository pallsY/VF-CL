"""Training-replay-only summaries of previous/current per-party feature drift."""

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def summarize_party_drift(old_bottoms, current_bottoms, replay_by_class,
                          party_weights, splitter):
    """Return per-class mean cosine drift without retaining feature tensors."""
    if (not isinstance(old_bottoms, (list, tuple))
            or not isinstance(current_bottoms, (list, tuple))
            or not old_bottoms or len(old_bottoms) != len(current_bottoms)
            or any(model.training for model in old_bottoms + current_bottoms)):
        raise ValueError('aligned eval-mode party models are required')
    if (not isinstance(replay_by_class, dict) or not replay_by_class
            or not isinstance(party_weights, dict)
            or set(replay_by_class) != set(party_weights)):
        raise ValueError('replay and party-weight classes must match')
    num_parties = len(old_bottoms)
    classes = {}
    total_examples = 0
    for class_id in sorted(replay_by_class):
        raw = replay_by_class[class_id]
        weights = party_weights[class_id]
        if (type(class_id) is not int or class_id < 0
                or not isinstance(raw, torch.Tensor)
                or raw.ndim < 2 or raw.size(0) == 0
                or not raw.is_floating_point() or not torch.isfinite(raw).all()):
            raise ValueError('training replay is malformed or non-finite')
        if (not isinstance(weights, (list, tuple))
                or len(weights) != num_parties
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or value < 0 for value in weights)
                or not math.isclose(sum(weights), 1.0, abs_tol=1e-5)):
            raise ValueError('party weights are incomplete or invalid')
        parts = splitter(raw)
        if (not isinstance(parts, (list, tuple)) or len(parts) != num_parties
                or any(not isinstance(part, torch.Tensor)
                       or part.size(0) != raw.size(0) for part in parts)):
            raise ValueError('vertical replay parts are not aligned')
        drift = []
        for old_model, current_model, part in zip(
                old_bottoms, current_bottoms, parts):
            old = old_model(part)
            current = current_model(part)
            if (not isinstance(old, torch.Tensor)
                    or not isinstance(current, torch.Tensor)
                    or old.ndim != 2 or old.shape != current.shape
                    or old.size(0) != raw.size(0)
                    or not torch.isfinite(old).all()
                    or not torch.isfinite(current).all()):
                raise ValueError('party features are malformed or non-finite')
            old_float, current_float = old.double(), current.double()
            both_zero = ((old_float == 0).all(dim=1)
                         & (current_float == 0).all(dim=1))
            per_sample = (1.0 - F.cosine_similarity(
                old_float, current_float, dim=1, eps=1e-12
            )).clamp_min(0.0)
            distance = torch.where(both_zero, 0.0, per_sample).mean()
            if not torch.isfinite(distance):
                raise ValueError('party cosine drift is non-finite')
            drift.append(float(distance))
        classes[str(class_id)] = {
            'sample_count': int(raw.size(0)),
            'party_cosine_drift': drift,
            'party_weights': [float(value) for value in weights],
        }
        total_examples += int(raw.size(0))
    return {
        'num_parties': num_parties,
        'total_examples': total_examples,
        'classes': classes,
        'test_used': False,
    }
