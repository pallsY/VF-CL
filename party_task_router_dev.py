"""Four-scalar party task correction for frozen development heads."""

import math

import torch
import torch.nn.functional as F


def _task_map(task_classes, count):
    if sorted(map(int, task_classes)) != list(range(len(task_classes))):
        raise ValueError('task IDs must be consecutive')
    mapping = torch.full((count,), -1, dtype=torch.long)
    for task in sorted(task_classes):
        members = list(map(int, task_classes[task]))
        if not members or any(not 0 <= member < count
                              or mapping[member] >= 0 for member in members):
            raise ValueError('task class map is invalid')
        mapping[members] = int(task)
    if bool((mapping < 0).any()):
        raise ValueError('task class map is incomplete')
    return mapping


def _task_scores(logits, task_classes, average):
    return torch.stack([
        torch.logsumexp(logits[:, list(map(int, task_classes[task]))], 1)
        - (math.log(len(task_classes[task])) if average else 0.0)
        for task in sorted(task_classes)
    ], dim=1)


def _party_evidence(base, parties, task_classes):
    if (base.ndim != 2 or not torch.isfinite(base).all()
            or not parties or any(part.shape != base.shape
                                  or not torch.isfinite(part).all()
                                  for part in parties)):
        raise ValueError('party score shape or finiteness is invalid')
    _task_map(task_classes, base.size(1))
    evidence = torch.stack([
        _task_scores(part, task_classes, average=True)
        for part in parties
    ])
    return evidence - evidence.mean(dim=2, keepdim=True)


def route(base_log_probs, party_logits, task_classes, fitted):
    base = F.log_softmax(base_log_probs.to(torch.float64), dim=1)
    mapping = _task_map(task_classes, base.size(1)).to(base.device)
    evidence = _party_evidence(base, party_logits, task_classes)
    weights = fitted['weights'].to(base.device)
    scales = fitted['scales'].to(base.device)
    if (weights.shape != (len(party_logits),)
            or scales.shape != weights.shape
            or bool((scales <= 0).any())):
        raise ValueError('router parameter shape is invalid')
    task_base = _task_scores(base, task_classes, average=False)
    correction = (evidence / scales[:, None, None]
                  * weights[:, None, None]).sum(0)
    task_routed = F.log_softmax(task_base + correction, dim=1)
    return task_routed[:, mapping] + base - task_base[:, mapping]


def fit_router(base_log_probs, party_logits, labels, task_classes):
    base = F.log_softmax(base_log_probs.detach().to(
        dtype=torch.float64, device='cpu'), dim=1)
    parties = [part.detach().to(dtype=torch.float64, device='cpu')
               for part in party_logits]
    evidence = _party_evidence(base, parties, task_classes)
    mapping = _task_map(task_classes, base.size(1))
    targets = labels.detach().to(dtype=torch.long, device='cpu')
    if (targets.ndim != 1 or targets.numel() != base.size(0)
            or bool(((targets < 0) | (targets >= base.size(1))).any())):
        raise ValueError('validation labels are invalid')
    task_targets = mapping[targets]
    scales = evidence.square().mean(dim=(1, 2)).sqrt().clamp_min(1e-6)
    task_base = _task_scores(base, task_classes, average=False)
    weights = torch.zeros(len(parties), dtype=torch.float64,
                          requires_grad=True)
    optimizer = torch.optim.LBFGS([weights], max_iter=80,
                                  line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        scores = task_base + (evidence / scales[:, None, None]
                              * weights[:, None, None]).sum(0)
        loss = F.cross_entropy(scores, task_targets) \
            + 0.01 * weights.square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return {'weights': weights.detach(), 'scales': scales.detach()}
