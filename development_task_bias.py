"""Development-only task logit offset control on frozen head outputs."""

import torch
import torch.nn.functional as F


def _class_tasks(task_classes, count):
    mapping = torch.full((count,), -1, dtype=torch.long)
    for task, members in task_classes.items():
        for member in members:
            class_id = int(member)
            if not 0 <= class_id < count or mapping[class_id] != -1:
                raise ValueError('task class map is invalid')
            mapping[class_id] = int(task)
    if bool((mapping < 0).any()) or sorted(set(mapping.tolist())) \
            != list(range(len(task_classes))):
        raise ValueError('task class map is incomplete')
    return mapping


def apply_task_bias(logits, offsets, task_classes):
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError('logits must be a finite matrix')
    mapping = _class_tasks(task_classes, logits.size(1)).to(logits.device)
    if offsets.ndim != 1 or offsets.numel() != len(task_classes):
        raise ValueError('task offsets have the wrong shape')
    return logits + offsets.to(logits.device)[mapping]


def fit_task_bias(logits, labels, task_classes):
    if labels.ndim != 1 or labels.numel() != logits.size(0):
        raise ValueError('validation labels do not match logits')
    if len(task_classes) < 2:
        raise ValueError('at least two tasks are required')
    scores = logits.detach().to(dtype=torch.float64, device='cpu')
    targets = labels.detach().to(dtype=torch.long, device='cpu')
    if bool(((targets < 0) | (targets >= scores.size(1))).any()):
        raise ValueError('validation labels are outside the classifier')
    offsets = torch.zeros(len(task_classes), dtype=torch.float64,
                          requires_grad=True)
    optimizer = torch.optim.LBFGS([offsets], max_iter=80,
                                  line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        centered = offsets - offsets.mean()
        calibrated = apply_task_bias(scores, centered, task_classes)
        loss = F.cross_entropy(calibrated, targets) \
            + 0.01 * centered.square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return (offsets - offsets.mean()).detach()


def fit_and_score(validation_logits, validation_labels, holdout_logits,
                  holdout_labels, task_classes):
    """Fit on validation only; use holdout labels solely for readout."""
    offsets = fit_task_bias(validation_logits, validation_labels,
                            task_classes)
    calibrated = apply_task_bias(holdout_logits, offsets, task_classes)
    return {
        'offsets': offsets,
        'holdout_log_probs': F.log_softmax(calibrated, dim=1),
        'holdout_accuracy': float((calibrated.argmax(1)
                                   == holdout_labels).double().mean()),
    }
