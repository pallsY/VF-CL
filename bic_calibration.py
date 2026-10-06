"""Minimal task-wise affine output calibration for Class-IL."""
import math

import torch
import torch.nn.functional as F


class TaskAffineCalibrator:
    def __init__(self):
        self.tasks = {}

    @staticmethod
    def _alpha(raw_alpha):
        return F.softplus(torch.as_tensor(raw_alpha, dtype=torch.float32)) + 1e-6

    def set_task(self, task_id, classes, raw_alpha, beta):
        self.tasks[int(task_id)] = {
            'classes': [int(class_id) for class_id in classes],
            'raw_alpha': float(torch.as_tensor(raw_alpha).detach()),
            'beta': float(torch.as_tensor(beta).detach()),
        }

    def parameters_for(self, task_id):
        item = self.tasks[int(task_id)]
        return {
            'alpha': float(self._alpha(item['raw_alpha'])),
            'beta': item['beta'],
        }

    def apply(self, logits):
        output = logits.clone()
        for item in self.tasks.values():
            alpha = self._alpha(item['raw_alpha']).to(logits.device)
            output[:, item['classes']] = alpha * logits[:, item['classes']] + item['beta']
        return output

    def fit_task(self, logits, labels, task_id, task_classes, seen_classes, lr, steps):
        logits = logits.detach().float().cpu()
        labels = labels.detach().long().cpu()
        task_id = int(task_id)
        task_classes = [int(value) for value in task_classes]
        seen_classes = [int(value) for value in seen_classes]
        identity_raw = math.log(math.expm1(1.0 - 1e-6))
        raw_alpha = torch.nn.Parameter(torch.tensor(identity_raw))
        beta = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.Adam([raw_alpha, beta], lr=float(lr))
        target_positions = {class_id: position for position, class_id in enumerate(seen_classes)}
        targets = torch.tensor([target_positions[int(label)] for label in labels])

        def calibrated():
            output = self.apply(logits)
            alpha = self._alpha(raw_alpha)
            output[:, task_classes] = alpha * logits[:, task_classes] + beta
            return output[:, seen_classes]

        with torch.no_grad():
            loss_before = float(F.cross_entropy(calibrated(), targets))
        for _ in range(int(steps)):
            optimizer.zero_grad()
            loss = F.cross_entropy(calibrated(), targets)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            loss_after = float(F.cross_entropy(calibrated(), targets))
        self.set_task(task_id, task_classes, raw_alpha, beta)
        params = self.parameters_for(task_id)
        return {
            'task_id': task_id,
            'classes': task_classes,
            'loss_before': loss_before,
            'loss_after': loss_after,
            **params,
        }

    def state_dict(self):
        return {'tasks': {str(key): dict(value) for key, value in sorted(self.tasks.items())}}

    def load_state_dict(self, state):
        self.tasks = {int(key): dict(value) for key, value in state.get('tasks', {}).items()}


def fit_final_calibrator(logits, labels, task_classes, mode, lr=0.05, steps=1000):
    """Fit all task-wise affine parameters together on final-stage logits."""
    modes = {'beta_only', 'alpha_only', 'joint_alpha_beta'}
    if mode not in modes:
        raise ValueError(f'unknown calibration mode: {mode}')
    logits = torch.as_tensor(logits).detach().float().cpu()
    labels = torch.as_tensor(labels).detach().long().cpu()
    task_classes = {
        int(key): [int(value) for value in values]
        for key, values in task_classes.items()
    }
    ordered_tasks = sorted(task_classes)
    seen = [class_id for task_id in ordered_tasks for class_id in task_classes[task_id]]
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.numel():
        raise ValueError('logits and labels have incompatible shapes')
    if not ordered_tasks or not seen or labels.numel() == 0:
        raise ValueError('calibration data and task classes must be non-empty')
    if len(seen) != len(set(seen)) or min(seen) < 0 or max(seen) >= logits.shape[1]:
        raise ValueError('task classes must be unique valid logit columns')
    positions = {class_id: position for position, class_id in enumerate(seen)}
    if any(int(label) not in positions for label in labels):
        raise ValueError('labels must belong to task classes')
    if not torch.isfinite(logits).all():
        raise ValueError('logits must be finite')
    lr, steps = float(lr), int(steps)
    if lr <= 0 or steps <= 0:
        raise ValueError('lr and steps must be positive')

    identity_raw = math.log(math.expm1(1.0 - 1e-6))
    raw_alpha = torch.full((len(ordered_tasks),), identity_raw)
    beta = torch.zeros(len(ordered_tasks))
    parameters = []
    if mode in {'alpha_only', 'joint_alpha_beta'}:
        raw_alpha = torch.nn.Parameter(raw_alpha)
        parameters.append(raw_alpha)
    if mode in {'beta_only', 'joint_alpha_beta'}:
        beta = torch.nn.Parameter(beta)
        parameters.append(beta)
    optimizer = torch.optim.Adam(parameters, lr=lr)
    targets = torch.tensor([positions[int(label)] for label in labels])

    def calibrated():
        output = logits.clone()
        for position, task_id in enumerate(ordered_tasks):
            alpha = F.softplus(raw_alpha[position]) + 1e-6
            classes = task_classes[task_id]
            output[:, classes] = alpha * logits[:, classes] + beta[position]
        return output[:, seen]

    with torch.no_grad():
        loss_before = float(F.cross_entropy(calibrated(), targets))
    if not math.isfinite(loss_before):
        raise ValueError('calibration loss must be finite')
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(calibrated(), targets)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        loss_after = float(F.cross_entropy(calibrated(), targets))
    if not math.isfinite(loss_after):
        raise ValueError('calibration loss must be finite')

    calibrator = TaskAffineCalibrator()
    for position, task_id in enumerate(ordered_tasks):
        calibrator.set_task(
            task_id, task_classes[task_id], raw_alpha[position], beta[position]
        )
    return calibrator, {
        'mode': mode,
        'lr': lr,
        'steps': steps,
        'loss_before': loss_before,
        'loss_after': loss_after,
    }


def _ece(logits, labels, bins=15):
    probabilities = torch.softmax(logits, dim=1)
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(labels)
    value = torch.tensor(0.0)
    edges = torch.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = confidence.gt(lower) & confidence.le(upper)
        if mask.any():
            value += mask.float().mean() * (confidence[mask].mean() - correct[mask].float().mean()).abs()
    return float(value)


def summarize_paired_logits(logits, labels, task_classes, calibrator):
    task_classes = {int(key): [int(value) for value in values]
                    for key, values in task_classes.items()}
    seen = [class_id for task_id in sorted(task_classes) for class_id in task_classes[task_id]]
    class_to_position = {class_id: position for position, class_id in enumerate(seen)}
    targets = torch.tensor([class_to_position[int(label)] for label in labels])

    def summarize(values):
        sliced = values[:, seen]
        predicted = torch.tensor([seen[position] for position in sliced.argmax(1).tolist()])
        per_task, task_il, fractions = {}, {}, {}
        for task_id, classes in sorted(task_classes.items()):
            key = f'task_{task_id}'
            mask = torch.tensor([int(label) in classes for label in labels], dtype=torch.bool)
            per_task[key] = float(predicted[mask].eq(labels[mask]).float().mean())
            local_pred = values[mask][:, classes].argmax(1)
            local_targets = torch.tensor([classes.index(int(label)) for label in labels[mask]])
            task_il[key] = float(local_pred.eq(local_targets).float().mean())
            fractions[key] = float(torch.tensor([int(value) in classes for value in predicted]).float().mean())
        return {
            'overall_accuracy': float(predicted.eq(labels).float().mean()),
            'per_task_accuracy': per_task,
            'task_il': task_il,
            'task_prediction_fraction': fractions,
            'ece': _ece(sliced, targets),
        }

    return {'raw': summarize(logits), 'calibrated': summarize(calibrator.apply(logits))}
