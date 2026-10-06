"""Task-boundary classifier calibration from bounded balanced replay."""

import hashlib
from collections.abc import Mapping

import torch
import torch.nn.functional as F


class _FrozenState(Mapping):
    def __init__(self, state):
        self.__state = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in sorted(state.items())
        }

    def __getitem__(self, name):
        return self.__state[name].clone()

    def __iter__(self):
        return iter(self.__state)

    def __len__(self):
        return len(self.__state)


def hash_top_state(top_or_state):
    """Hash sorted tensor state including its schema and contiguous bytes."""
    state = (top_or_state.state_dict()
             if hasattr(top_or_state, 'state_dict') else top_or_state)
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError('top state must map string keys to tensors')
        fields = (
            name.encode('utf-8'),
            str(value.dtype).encode('ascii'),
            repr(tuple(value.shape)).encode('ascii'),
            value.detach().cpu().contiguous().reshape(-1)
            .view(torch.uint8).numpy().tobytes(),
        )
        for field in fields:
            digest.update(len(field).to_bytes(8, 'big'))
            digest.update(field)
    return digest.hexdigest()


def freeze_state(state):
    """Return a structurally frozen detached CPU clone of tensor state."""
    return _FrozenState(state)


def balanced_prototype_batch(prototypes, samples_per_class, seed, device):
    """Build a deterministic balanced batch without storing synthetic samples."""
    samples_per_class = int(samples_per_class)
    if not prototypes:
        raise ValueError('head consolidation requires at least one prototype')
    if samples_per_class <= 0:
        raise ValueError('samples_per_class must be positive')

    classes = sorted(int(class_id) for class_id in prototypes)
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    rows, labels = [], []
    reference_shape = None
    for class_id in classes:
        prototype = prototypes[class_id]
        mean = torch.as_tensor(prototype['mean']).detach().cpu()
        std = torch.as_tensor(prototype['std']).detach().cpu()
        if mean.ndim != 1 or std.shape != mean.shape:
            raise ValueError('prototype mean/std must be matching vectors')
        if reference_shape is not None and mean.shape != reference_shape:
            raise ValueError('all prototype vectors must share one shape')
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError('prototype mean/std must be finite')
        if bool((std < 0).any()):
            raise ValueError('prototype std must be non-negative')
        reference_shape = mean.shape

        # Include the exact mean, then sample the stored diagonal Gaussian. This
        # keeps every class represented by its stable center while preserving the
        # spread already used by the method's online prototype replay.
        rows.append(mean)
        labels.append(class_id)
        if samples_per_class > 1:
            noise = torch.randn(
                (samples_per_class - 1, mean.numel()),
                generator=generator,
                dtype=mean.dtype,
            )
            rows.extend(mean.unsqueeze(0) + noise * std.unsqueeze(0))
            labels.extend([class_id] * (samples_per_class - 1))

    return (
        torch.stack(rows).to(device),
        torch.tensor(labels, dtype=torch.long, device=device),
        classes,
    )


def balanced_embedding_replay_batch(replay_embeddings, samples_per_class, device):
    """Build a balanced batch from persistent training embeddings."""
    samples_per_class = int(samples_per_class)
    if not replay_embeddings:
        raise ValueError('head consolidation requires embedding replay')
    if samples_per_class <= 0:
        raise ValueError('samples_per_class must be positive')
    classes = sorted(int(class_id) for class_id in replay_embeddings)
    rows, labels = [], []
    reference_dim = None
    for class_id in classes:
        values = torch.as_tensor(replay_embeddings[class_id]).detach().cpu()
        if values.ndim != 2 or values.size(0) == 0:
            raise ValueError('embedding replay values must be non-empty matrices')
        if reference_dim is not None and values.size(1) != reference_dim:
            raise ValueError('all replay embeddings must share one feature size')
        if not torch.isfinite(values).all():
            raise ValueError('embedding replay must be finite')
        reference_dim = values.size(1)
        selected = values[:samples_per_class]
        rows.append(selected)
        labels.extend([class_id] * selected.size(0))
    return (
        torch.cat(rows).to(device),
        torch.tensor(labels, dtype=torch.long, device=device),
        classes,
    )


def _identity_raw_alpha(device, dtype):
    target = torch.tensor(1.0 - 1e-6, device=device, dtype=dtype)
    return torch.log(torch.expm1(target))


def _task_layout(task_classes, classes, device):
    if not task_classes:
        raise ValueError('task/class calibration requires task classes')
    task_ids = sorted(int(task_id) for task_id in task_classes)
    class_to_task = {}
    for position, task_id in enumerate(task_ids):
        values = [int(class_id) for class_id in task_classes[task_id]]
        if not values:
            raise ValueError('task/class calibration tasks must be non-empty')
        for class_id in values:
            if class_id in class_to_task:
                raise ValueError('each replay class must belong to exactly one task')
            class_to_task[class_id] = position
    if set(class_to_task) != set(classes):
        raise ValueError('task classes must exactly match replay classes')
    return torch.tensor(
        [class_to_task[class_id] for class_id in classes],
        dtype=torch.long,
        device=device,
    ), task_ids


def _calibrated_local_scores(logits, class_index, task_for_position,
                             alpha, beta, class_bias, task_weight):
    local = (
        logits.index_select(1, class_index) * alpha[task_for_position]
        + beta[task_for_position]
        + class_bias
    )
    task_ids = torch.unique(task_for_position, sorted=True)
    evidence = torch.stack([
        torch.logsumexp(local[:, task_for_position == task_id], dim=1)
        for task_id in task_ids
    ], dim=1)
    task_log_probability = F.log_softmax(evidence, dim=1)
    output = local.clone()
    for position, task_id in enumerate(task_ids):
        mask = task_for_position == task_id
        output[:, mask] = (
            F.log_softmax(local[:, mask], dim=1)
            + float(task_weight) * task_log_probability[:, position, None]
        )
    return output


def consolidate_task_class_bias(
        top_model, replay_embeddings, task_classes, class_regularization,
        task_regularization, task_weight, steps, lr, samples_per_class,
        device, persistent_raw_example_count=0,
        replay_source='balanced_current_encoder_raw_replay'):
    """Fit a small task/class logit calibrator without changing the classifier."""
    class_regularization = float(class_regularization)
    task_regularization = float(task_regularization)
    task_weight = float(task_weight)
    steps = int(steps)
    lr = float(lr)
    if class_regularization < 0 or task_regularization < 0:
        raise ValueError('task/class regularization must be non-negative')
    if task_weight <= 0 or steps <= 0 or lr <= 0:
        raise ValueError('task/class calibration hyperparameters are invalid')
    if not hasattr(top_model, 'set_logit_calibration'):
        raise ValueError('top model must support logit calibration')

    embeddings, labels, classes = balanced_embedding_replay_batch(
        replay_embeddings, samples_per_class, device
    )
    persistent_raw_example_count = int(persistent_raw_example_count)
    if persistent_raw_example_count < 0:
        raise ValueError('persistent raw example count must be non-negative')
    class_index = torch.tensor(classes, dtype=torch.long, device=device)
    positions = {class_id: position for position, class_id in enumerate(classes)}
    targets = torch.tensor(
        [positions[int(label)] for label in labels.detach().cpu().tolist()],
        dtype=torch.long,
        device=device,
    )
    task_for_position, task_ids = _task_layout(task_classes, classes, device)

    top_model.clear_logit_calibration()
    was_training = top_model.training
    top_model.eval()
    with torch.no_grad():
        logits = top_model(embeddings).detach()
    dtype = logits.dtype
    raw_alpha = torch.nn.Parameter(
        _identity_raw_alpha(device, dtype).repeat(len(task_ids))
    )
    beta_parameter = torch.nn.Parameter(
        torch.zeros(len(task_ids), device=device, dtype=dtype)
    )
    class_parameter = torch.nn.Parameter(
        torch.zeros(len(classes), device=device, dtype=dtype)
    )
    optimizer = torch.optim.Adam(
        [raw_alpha, beta_parameter, class_parameter], lr=lr
    )

    def parameters():
        alpha = F.softplus(raw_alpha) + 1e-6
        beta = beta_parameter - beta_parameter.mean()
        residual = class_parameter.clone()
        for task_position in range(len(task_ids)):
            mask = task_for_position == task_position
            residual[mask] = residual[mask] - residual[mask].mean()
        return alpha, beta, residual

    def objective():
        alpha, beta, residual = parameters()
        scores = _calibrated_local_scores(
            logits, class_index, task_for_position,
            alpha, beta, residual, task_weight,
        )
        ce = F.cross_entropy(scores, targets)
        task_penalty = (alpha - 1.0).square().mean() + beta.square().mean()
        loss = ce + class_regularization * residual.square().mean()
        return loss + task_regularization * task_penalty, ce

    with torch.no_grad():
        before_objective, before_ce = objective()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = objective()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        after_objective, after_ce = objective()
        alpha, beta, residual = parameters()
        alpha_per_class = alpha[task_for_position]
        bias_per_class = beta[task_for_position] + residual

    top_model.set_logit_calibration(
        classes, alpha_per_class, bias_per_class,
        task_for_position, task_weight,
    )
    top_model.train(was_training)
    return {
        'mode': 'task_class_bias',
        'classes': classes,
        'class_count': len(classes),
        'task_count': len(task_ids),
        'task_ids': task_ids,
        'samples_per_class': int(samples_per_class),
        'fit_sample_count': int(labels.numel()),
        'synthetic_sample_count': 0,
        'persistent_raw_example_count': persistent_raw_example_count,
        'persistent_embedding_count': 0,
        'learned_parameter_count': 2 * len(task_ids) + len(classes),
        'class_regularization': class_regularization,
        'task_regularization': task_regularization,
        'task_weight': task_weight,
        'steps': steps,
        'lr': lr,
        'objective_before': float(before_objective),
        'objective_after': float(after_objective),
        'cross_entropy_before': float(before_ce),
        'cross_entropy_after': float(after_ce),
        'alpha_min': float(alpha.min()),
        'alpha_max': float(alpha.max()),
        'task_bias_abs_max': float(beta.abs().max()),
        'class_bias_abs_max': float(residual.abs().max()),
        'source': str(replay_source),
        'classifier_parameters_changed': False,
        'validation_used': False,
        'test_used': False,
    }


def consolidate_classifier(top_model, prototypes, regularization, steps, lr,
                           samples_per_class, seed, device,
                           replay_embeddings=None,
                           replay_source='balanced_training_embedding_replay',
                           persistent_raw_example_count=0):
    """Optimize only classifier parameters against balanced prototype replay."""
    regularization = float(regularization)
    steps = int(steps)
    lr = float(lr)
    if regularization < 0 or steps <= 0 or lr <= 0:
        raise ValueError('head consolidation hyperparameters are invalid')
    if not hasattr(top_model, 'classifier'):
        raise ValueError('top model must expose a classifier module')

    if replay_embeddings is None:
        embeddings, labels, classes = balanced_prototype_batch(
            prototypes, samples_per_class, seed, device
        )
        source = 'global_gaussian_prototypes'
        synthetic_sample_count = int(labels.numel())
        persistent_embedding_count = 0
    else:
        embeddings, labels, classes = balanced_embedding_replay_batch(
            replay_embeddings, samples_per_class, device
        )
        source = str(replay_source)
        synthetic_sample_count = 0
        persistent_raw_example_count = int(persistent_raw_example_count)
        persistent_embedding_count = (
            0 if persistent_raw_example_count > 0 else int(labels.numel())
        )
    if persistent_raw_example_count < 0:
        raise ValueError('persistent raw example count must be non-negative')
    class_index = torch.tensor(classes, dtype=torch.long, device=device)
    positions = {class_id: position for position, class_id in enumerate(classes)}
    targets = torch.tensor(
        [positions[int(label)] for label in labels.detach().cpu().tolist()],
        dtype=torch.long,
        device=device,
    )
    classifier = top_model.classifier
    if max(classes) >= classifier.out_features:
        raise ValueError('prototype class is outside the classifier output range')

    parameter_states = {
        name: parameter.requires_grad
        for name, parameter in top_model.named_parameters()
    }
    classifier_ids = {id(parameter) for parameter in classifier.parameters()}
    for parameter in top_model.parameters():
        parameter.requires_grad = id(parameter) in classifier_ids
    anchors = {
        name: parameter.detach().clone()
        for name, parameter in classifier.named_parameters()
    }
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    was_training = top_model.training
    top_model.eval()

    def objective():
        scores = top_model(embeddings).index_select(1, class_index)
        ce = F.cross_entropy(scores, targets)
        penalty = (
            classifier.weight.index_select(0, class_index)
            - anchors['weight'].index_select(0, class_index)
        ).square().sum(dim=1).mean()
        if classifier.bias is not None:
            penalty = penalty + (
                classifier.bias.index_select(0, class_index)
                - anchors['bias'].index_select(0, class_index)
            ).square().mean()
        return ce + regularization * penalty, ce

    with torch.no_grad():
        before_objective, before_ce = objective()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = objective()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        after_objective, after_ce = objective()
        delta = (
            classifier.weight.index_select(0, class_index)
            - anchors['weight'].index_select(0, class_index)
        ).norm()

    for name, parameter in top_model.named_parameters():
        parameter.requires_grad = parameter_states[name]
    top_model.train(was_training)
    return {
        'classes': classes,
        'class_count': len(classes),
        'samples_per_class': int(samples_per_class),
        'fit_sample_count': int(labels.numel()),
        'synthetic_sample_count': synthetic_sample_count,
        'persistent_raw_example_count': persistent_raw_example_count,
        'persistent_embedding_count': persistent_embedding_count,
        'seed': int(seed),
        'regularization': regularization,
        'steps': steps,
        'lr': lr,
        'objective_before': float(before_objective),
        'objective_after': float(after_objective),
        'cross_entropy_before': float(before_ce),
        'cross_entropy_after': float(after_ce),
        'seen_weight_delta_l2': float(delta),
        'source': source,
        'validation_used': False,
        'test_used': False,
    }
