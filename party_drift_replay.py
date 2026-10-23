"""Training-only raw replay primitives for the four-party diagnostic."""

import numpy as np
import torch
import torch.nn.functional as F


class BalancedOldReplay:
    def __init__(self, data, seed, batch_size=64):
        if not data or batch_size < 1:
            raise ValueError('old replay requires samples and a positive batch size')
        self.data = data
        self.classes = sorted(int(class_id) for class_id in data)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.orders = {class_id: self.rng.permutation(len(data[class_id]))
                       for class_id in self.classes}
        self.positions = {class_id: 0 for class_id in self.classes}
        self.step = 0
        if any(not isinstance(data[c], torch.Tensor) or len(data[c]) == 0
               for c in self.classes):
            raise ValueError('old replay has an empty or invalid class')

    def sample(self):
        count = len(self.classes)
        quota, remainder = divmod(self.batch_size, count)
        slots = [quota] * count
        start = (self.step * remainder) % count
        for index in range(remainder):
            slots[(start + index) % count] += 1
        picked, labels, identities = [], [], []
        for class_id, amount in zip(self.classes, slots):
            for _ in range(amount):
                position = self.positions[class_id]
                if position == len(self.orders[class_id]):
                    self.orders[class_id] = self.rng.permutation(
                        len(self.data[class_id]))
                    position = 0
                row = int(self.orders[class_id][position])
                self.positions[class_id] = position + 1
                picked.append(self.data[class_id][row])
                labels.append(class_id)
                identities.append((class_id, row))
        self.step += 1
        return (torch.stack(picked), torch.tensor(labels, dtype=torch.long),
                identities, len(identities) - len(set(identities)))


def _old_class_log_odds(logits, labels, old_classes):
    classes = torch.as_tensor(old_classes, dtype=torch.long, device=logits.device)
    if len(classes) < 2 or len(set(old_classes)) != len(old_classes):
        raise ValueError('probe requires at least two distinct old classes')
    matches = labels[:, None] == classes
    if not matches.any(dim=1).all():
        raise ValueError('probe labels must be old classes')
    old_logits = logits.index_select(1, classes)
    true = old_logits.gather(1, matches.long().argmax(dim=1, keepdim=True)).squeeze(1)
    other = old_logits.masked_fill(matches, float('-inf')).logsumexp(dim=1)
    return true - other


def _aggregate(embeddings, aggregation):
    if aggregation == 'sum':
        return sum(embeddings)
    if aggregation == 'concat':
        return torch.cat(embeddings, dim=1)
    raise ValueError('unsupported replay aggregation')


def mobius_profile(benefits):
    """Return nonempty coalition contributions from ordered subset benefits."""
    if benefits.ndim != 2 or benefits.shape[1] < 2 or (
            benefits.shape[1] & (benefits.shape[1] - 1)):
        raise ValueError('benefits need a nonempty power-of-two subset axis')
    subsets = benefits.shape[1]
    coefficients = benefits.new_zeros((subsets - 1, subsets))
    for coalition in range(1, subsets):
        for subset in range(subsets):
            if subset & coalition == subset:
                sign = -1 if ((coalition.bit_count() - subset.bit_count()) % 2) else 1
                coefficients[coalition - 1, subset] = sign
    return benefits @ coefficients.T


def cmcdr_profile_loss(current_embeddings, old_embeddings, top, old_top,
                       labels, old_classes, *, aggregation='sum'):
    """Direct replay CMCDR profile matching, with a frozen teacher."""
    parties = len(current_embeddings)
    if parties < 1 or len(old_embeddings) != parties or labels.numel() == 0:
        raise ValueError('CMCDR inputs have incompatible party or batch sizes')
    current_benefits, old_benefits = [], []
    for subset in range(1 << parties):
        current_masked = [
            emb if subset & (1 << k) else torch.zeros_like(emb)
            for k, emb in enumerate(current_embeddings)
        ]
        current_benefits.append(_old_class_log_odds(
            top(_aggregate(current_masked, aggregation)), labels, old_classes))
        with torch.no_grad():
            old_masked = [
                emb.detach() if subset & (1 << k) else torch.zeros_like(emb)
                for k, emb in enumerate(old_embeddings)
            ]
            old_benefits.append(_old_class_log_odds(
                old_top(_aggregate(old_masked, aggregation)), labels,
                old_classes))
    current = mobius_profile(torch.stack(current_benefits, dim=1))
    old = mobius_profile(torch.stack(old_benefits, dim=1)).detach()
    raw = F.smooth_l1_loss(current, old, beta=1.0)
    current_norm = current / (current.abs().sum(dim=1, keepdim=True) + 1e-6)
    old_norm = old / (old.abs().sum(dim=1, keepdim=True) + 1e-6)
    return raw + F.smooth_l1_loss(current_norm, old_norm, beta=1.0)


def cmcdr_positions(*, seed, task, replay_steps, batch_size, old_classes,
                    probe_events, parties=4, probe_per_class=2):
    """Preselect examples whose full-subset evaluations fit the A1 probe cap."""
    if (min(seed, task) < 0 or replay_steps < 1 or batch_size < 1 or
            old_classes < 2 or probe_events < 1 or parties < 2 or
            probe_per_class < 1):
        raise ValueError('invalid CMCDR intervention budget')
    candidate_responses = 2 * (parties + 1) * probe_per_class * old_classes * probe_events
    per_example = 2 * (1 << parties)
    cap = min(replay_steps * batch_size, candidate_responses // per_example)
    rng = np.random.default_rng(np.random.SeedSequence([seed, task, 90211]))
    positions = rng.choice(replay_steps * batch_size, size=cap, replace=False)
    return {(int(position // batch_size), int(position % batch_size))
            for position in positions}


def contribution_attenuation(bottoms, top, old_bottoms, old_top,
                             parts, labels, old_classes, *, aggregation='sum'):
    """A1: mean positive loss of per-party old-class log-odds contribution."""
    if len(bottoms) != len(old_bottoms) or len(bottoms) != len(parts):
        raise ValueError('probe party layout differs')
    training_states = [module.training for module in (*bottoms, top)]
    for module in (*bottoms, top):
        module.eval()
    try:
        with torch.no_grad():
            current = [module(part) for module, part in zip(bottoms, parts)]
            teacher = [module(part) for module, part in zip(old_bottoms, parts)]
            full_current = _old_class_log_odds(
                top(_aggregate(current, aggregation)), labels, old_classes)
            full_teacher = _old_class_log_odds(
                old_top(_aggregate(teacher, aggregation)), labels, old_classes)
            decay = []
            for k in range(len(bottoms)):
                without_current = _old_class_log_odds(
                    top(_aggregate([emb if j != k else torch.zeros_like(emb)
                                    for j, emb in enumerate(current)], aggregation)),
                    labels, old_classes)
                without_teacher = _old_class_log_odds(
                    old_top(_aggregate([emb if j != k else torch.zeros_like(emb)
                                        for j, emb in enumerate(teacher)], aggregation)),
                    labels, old_classes)
                attenuation = (full_teacher - without_teacher) - (
                    full_current - without_current)
                per_class = [attenuation[labels == c].clamp_min(0).mean()
                             for c in old_classes]
                if any(not torch.isfinite(value) for value in per_class):
                    raise ValueError('probe class is missing or non-finite')
                decay.append(float(torch.stack(per_class).mean()))
            return decay
    finally:
        for module, was_training in zip((*bottoms, top), training_states):
            module.train(was_training)


def masked_replay_loss(bottoms, top, old_bottoms, old_top, parts,
                       labels, selected, old_classes, alpha, seed,
                       *, temperature=2.0, kd_weight=0.25,
                       aggregation='sum', cmcdr_indices=None,
                       cmcdr_weight=0.0):
    """Same party mask on student and frozen teacher; full-head CE, old-only KL."""
    if (not selected or len(set(selected)) != len(selected) or
            any(k < 0 or k >= len(bottoms) for k in selected) or
            len(bottoms) != len(old_bottoms) or len(bottoms) != len(parts) or
            alpha < 0 or temperature <= 0 or kd_weight < 0 or
            cmcdr_weight < 0):
        raise ValueError('invalid masked replay specification')
    if cmcdr_weight and (cmcdr_indices is None or
            len(selected) != len(bottoms) or
            len(set(cmcdr_indices)) != len(cmcdr_indices) or
            any(type(index) is not int or index < 0 or index >= len(labels)
                for index in cmcdr_indices)):
        raise ValueError('CMCDR requires full-party replay and valid rows')
    devices = [parts[0].device.index] if parts[0].is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(int(seed))
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(int(seed))
        student_selected = {k: bottoms[k](parts[k]) for k in selected}
        student_template = next(iter(student_selected.values()))
        student_embs = [student_selected.get(k, torch.zeros_like(student_template))
                        for k in range(len(bottoms))]
        student_logits = top(_aggregate(student_embs, aggregation))
        with torch.no_grad():
            teacher_selected = {k: old_bottoms[k](parts[k]) for k in selected}
            teacher_template = next(iter(teacher_selected.values()))
            teacher_embs = [teacher_selected.get(k, torch.zeros_like(teacher_template))
                            for k in range(len(bottoms))]
            teacher_logits = old_top(_aggregate(teacher_embs, aggregation))
        ce = F.cross_entropy(student_logits, labels)
        old = torch.as_tensor(old_classes, dtype=torch.long,
                              device=student_logits.device)
        kd = F.kl_div(
            F.log_softmax(student_logits.index_select(1, old) / temperature, dim=1),
            F.softmax(teacher_logits.index_select(1, old) / temperature, dim=1),
            reduction='batchmean',
        ) * temperature**2
        total = alpha * ce + kd_weight * kd
        components = {'ce': ce.detach(), 'logit_kd': kd.detach()}
        if cmcdr_weight:
            profile = total.new_zeros(())
            if cmcdr_indices:
                current = [emb[cmcdr_indices] for emb in student_embs]
                teacher = [emb[cmcdr_indices] for emb in teacher_embs]
                profile = cmcdr_profile_loss(
                    current, teacher, top, old_top, labels[cmcdr_indices],
                    old_classes, aggregation=aggregation)
                total = total + cmcdr_weight * profile
            components.update({'cmcdr': profile.detach(),
                               'cmcdr_examples': len(cmcdr_indices)})
        return total, components
