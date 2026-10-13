"""Adaptive layer freezing and similarity-aware retrieval for VFL."""
import random
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import split_features
from .er import ERCL


def choose_freeze_prefix(fisher, batch_ratio, forward_cost, backward_cost):
    """Paper BFC: saved-compute benefit less batch Fisher lost by freezing."""
    if (fisher.ndim != 1 or backward_cost.shape != fisher.shape
            or forward_cost <= 0 or batch_ratio < 0):
        raise ValueError('invalid aL-SAR freezing inputs')
    rates = [float(fisher[n:].sum()) /
             (float(forward_cost) + float(backward_cost[n:].sum()))
             for n in range(len(fisher) + 1)]
    best_rate = max(rates)
    benefits = [best_rate * float(backward_cost[:n].sum())
                - batch_ratio * float(fisher[:n].sum())
                for n in range(len(fisher) + 1)]
    return max(range(len(benefits)), key=lambda n: benefits[n])


def choose_freeze_prefixes(fisher, batch_ratio, forward_cost, backward_cost,
                           top_fisher, top_backward_cost):
    """Maximize paper Eq. 4-5 over four parallel party prefixes."""
    if (len(fisher) != len(backward_cost) or forward_cost <= 0
            or top_backward_cost < 0 or top_fisher < 0 or batch_ratio < 0
            or any(len(f) != len(b) for f, b in zip(fisher, backward_cost))):
        raise ValueError('invalid whole-model aL-SAR BFC inputs')
    choices = product(*(range(len(values) + 1) for values in fisher))
    best_rate = max(
        (top_fisher + sum(sum(values[n:]) for values, n in zip(fisher, p)))
        / (forward_cost + top_backward_cost
           + sum(sum(values[n:]) for values, n in zip(backward_cost, p)))
        for p in choices)
    return max(
        product(*(range(len(values) + 1) for values in fisher)),
        key=lambda p: best_rate * sum(
            sum(values[:n]) for values, n in zip(backward_cost, p))
        - batch_ratio * sum(sum(values[:n]) for values, n in zip(fisher, p)))


def sar_probabilities(labels, discounted_usage, similarity, temperature):
    """Paper Eq. 7-8 on aligned replay entries."""
    if (labels.ndim != 1 or discounted_usage.shape != labels.shape
            or similarity.ndim != 2 or similarity.size(0) != similarity.size(1)
            or temperature <= 0 or labels.numel() == 0
            or int(labels.max()) >= len(similarity)):
        raise ValueError('invalid aL-SAR retrieval state')
    class_usage = torch.zeros(len(similarity), dtype=discounted_usage.dtype,
                              device=discounted_usage.device)
    class_usage.scatter_add_(0, labels, discounted_usage)
    effective = discounted_usage + (similarity.T @ class_usage)[labels]
    return F.softmax(-effective / temperature, dim=0)


def classifier_sample_gradients(features, logits, labels):
    """Per-example gradients of an ordinary linear shared classifier."""
    errors = logits.softmax(1) - F.one_hot(labels, logits.size(1))
    return (errors[:, :, None] * features[:, None, :]).flatten(1)


def update_class_similarity(similarity, gradients, labels, rate):
    """EMA of sample-pair gradient cosines, excluding a sample with itself."""
    if len(gradients) != len(labels) or similarity.ndim != 2:
        raise ValueError('invalid class-similarity inputs')
    vectors = F.normalize(gradients, dim=1)
    cosines = vectors @ vectors.T
    updated = similarity.to(cosines.device).clone()
    groups = F.one_hot(labels, len(updated)).to(cosines.dtype)
    counts = groups.sum(0)
    pair_counts = counts[:, None] * counts[None, :]
    pair_counts.diagonal().sub_(counts)
    pair_sum = groups.T @ cosines @ groups
    self_sum = torch.zeros_like(counts).scatter_add_(
        0, labels, cosines.diagonal())
    pair_sum.diagonal().sub_(self_sum)
    observed = pair_counts > 0
    updated[observed] += rate * (
        pair_sum[observed] / pair_counts[observed] - updated[observed])
    return updated.to(similarity.device)


class ALSARCL(ERCL):
    def __init__(self, trainer, args):
        super().__init__(trainer, args)
        self.name = 'aL-SAR_VFL'
        self.temperature = float(getattr(args, 'alsar_temperature', 0.125))
        self.frequency_scale = 4.0
        self.fisher_rate = float(getattr(args, 'alsar_fisher_rate', 0.01))
        self.gradient_rate = float(getattr(args, 'alsar_gradient_rate', 0.01))
        if (self.temperature <= 0
                or not 0 < self.fisher_rate <= 1
                or not 0 < self.gradient_rate <= 1):
            raise ValueError('invalid aL-SAR hyperparameters')
        self.usage = {}
        self.step = 0
        self.class_similarity = torch.zeros(
            trainer.top_model.classifier.out_features,
            trainer.top_model.classifier.out_features)
        self.last_gradient_mean = 0.0
        self.replay_batch = None
        self.task_id = 0
        self._collect_training = False
        self._capture_labels = None
        self._forward_records = []
        self._setup_stages()

    def _setup_stages(self):
        self.stages = []
        for bottom in self.trainer.bottoms:
            if all(hasattr(bottom, f'layer{i}') for i in range(5)):
                parts = [getattr(bottom, f'layer{i}') for i in range(5)]
            elif hasattr(bottom, 'net'):
                parts = [module for module in bottom.net
                         if any(True for _ in module.parameters())]
            else:
                parts = [module for module in bottom.children()
                         if any(True for _ in module.parameters())]
            if not parts:
                parts = [bottom]
            self.stages.append(parts)
        self.fisher = [[0.0] * len(parts) for parts in self.stages]
        self.costs = [None] * len(self.stages)
        self.top_cost = None
        self.top_fisher = 0.0
        self.refresh_fisher = False
        self.frozen = []

    def before_task(self, task_id, new_classes, seen_classes):
        old_count = self.trainer.top_model.classifier.out_features
        super().before_task(task_id, new_classes, seen_classes)
        count = self.trainer.top_model.classifier.out_features
        if count != old_count:
            self.top_cost = None
        if count > len(self.class_similarity):
            expanded = torch.zeros(count, count)
            old = len(self.class_similarity)
            expanded[:old, :old] = self.class_similarity
            self.class_similarity = expanded
        self.frozen = []

    @torch.no_grad()
    def _preliminary_forward(self, examples):
        pieces = split_features(examples, self.args)
        module_modes = [module.training for module in
                        [*self.trainer.bottoms, self.trainer.top_model]]
        all_modules = [*self.trainer.bottoms, self.trainer.top_model]
        hooks = []
        cost = [[0.0] * len(parts) for parts in self.stages]
        top_cost = [0.0]

        def count_ops(_module, inputs, output, party, stage):
            source = inputs[0]
            if isinstance(_module, nn.Linear):
                operations = 2 * source.numel() * output.size(-1)
            elif isinstance(_module, nn.Conv2d):
                operations = (2 * output.numel() * source.size(1)
                              * _module.kernel_size[0] * _module.kernel_size[1])
            else:
                operations = output.numel()
            cost[party][stage] += operations

        def count_top(module, inputs, output):
            source = inputs[0]
            if isinstance(module, nn.Linear):
                top_cost[0] += 2 * source.numel() * output.size(-1)
            elif isinstance(module, nn.Conv2d):
                top_cost[0] += (2 * output.numel() * source.size(1)
                                * module.kernel_size[0] * module.kernel_size[1])
            else:
                top_cost[0] += output.numel()

        try:
            for module in all_modules:
                module.eval()
            if any(value is None for value in self.costs):
                for party, parts in enumerate(self.stages):
                    for stage, block in enumerate(parts):
                        for module in block.modules():
                            if isinstance(module, (nn.Linear, nn.Conv2d,
                                                   nn.BatchNorm2d)):
                                hooks.append(module.register_forward_hook(
                                    lambda m, i, o, p=party, s=stage:
                                    count_ops(m, i, o, p, s)))
            if self.top_cost is None:
                for module in self.trainer.top_model.modules():
                    if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
                        hooks.append(module.register_forward_hook(count_top))
            embeddings = [bottom(piece) for bottom, piece in
                          zip(self.trainer.bottoms, pieces)]
            features = self.trainer._aggregate(embeddings)
            logits = self.trainer.top_model(features)
            self.trainer.bytes_transmitted += sum(
                value.numel() * value.element_size() * 2 for value in embeddings)
            self.trainer.comm_rounds += 1
        finally:
            for hook in hooks:
                hook.remove()
            for module, training in zip(all_modules, module_modes):
                module.train(training)
        for party, measured in enumerate(cost):
            if self.costs[party] is None:
                self.costs[party] = torch.tensor(
                    [max(value / len(examples), 1.0) for value in measured],
                    dtype=torch.float64)
        if self.top_cost is None:
            self.top_cost = max(top_cost[0] / len(examples), 1.0)
        return features.detach(), logits.detach()

    def _before_batch(self, examples, labels):
        current_labels = labels
        self.replay_batch = self._select_replay() if self.task_id > 0 else None
        if self.replay_batch is not None:
            examples = torch.cat((examples, self.replay_batch[0]), dim=0)
            labels = torch.cat((labels, self.replay_batch[1]), dim=0)
        features, logits = self._preliminary_forward(examples)
        probabilities = logits.softmax(1)
        targets = F.one_hot(labels, logits.size(1)).float()
        output_gradient = (probabilities - targets) @ self.trainer.top_model.classifier.weight
        current_gradient = float(output_gradient.detach().square().sum(1).mean())
        ratio = current_gradient / max(self.last_gradient_mean, 1e-10)
        self.last_gradient_mean += self.fisher_rate * (
            current_gradient - self.last_gradient_mean)
        self.refresh_fisher = self.step % 4 == 0
        if self.refresh_fisher:
            prefixes = (0,) * len(self.stages)
        else:
            prefixes = choose_freeze_prefixes(
                self.fisher, ratio,
                self.top_cost + sum(float(cost.sum()) for cost in self.costs),
                [list((cost * 2).tolist()) for cost in self.costs],
                self.top_fisher, self.top_cost * 2)
        self.frozen = []
        for party, parts in enumerate(self.stages):
            prefix = prefixes[party]
            self.frozen.append(prefix)
            for block in parts[:prefix]:
                for parameter in block.parameters():
                    parameter.requires_grad_(False)
        self._forward_records = []
        self._capture_labels = current_labels.detach()
        self._collect_training = True
        self.step += 1

    def _record_gradients(self, _bottoms):
        if self.refresh_fisher:
            for party, parts in enumerate(self.stages):
                for index, block in enumerate(parts):
                    observed = sum(float(parameter.grad.detach().square().sum())
                                   for parameter in block.parameters()
                                   if parameter.grad is not None)
                    self.fisher[party][index] += self.fisher_rate * (
                        observed - self.fisher[party][index])

    def _record_top_gradients(self, models):
        if self.refresh_fisher:
            observed = sum(float(parameter.grad.detach().square().sum())
                           for model in models for parameter in model.parameters()
                           if parameter.grad is not None)
            self.top_fisher += self.fisher_rate * (observed - self.top_fisher)
        if self._forward_records:
            features = torch.cat([record[0] for record in self._forward_records])
            logits = torch.cat([record[1] for record in self._forward_records])
            labels = torch.cat([record[2] for record in self._forward_records])
            with torch.no_grad():
                gradients = classifier_sample_gradients(features, logits, labels)
                self.class_similarity = update_class_similarity(
                    self.class_similarity, gradients, labels,
                    self.gradient_rate)
        self._forward_records = []
        self._collect_training = False

    def _capture_top_forward(self, _model, inputs, output):
        if self._collect_training and self._capture_labels is not None:
            self._forward_records.append((inputs[0].detach(), output.detach(),
                                          self._capture_labels.detach()))

    def _after_batch(self):
        for parts, prefix in zip(self.stages, self.frozen):
            for block in parts[:prefix]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
        self._collect_training = False
        self._capture_labels = None

    def _similarity(self, device):
        return self.class_similarity.to(device)

    def _select_replay(self):
        indexed = [(class_id, index, example)
                   for class_id, examples in self.buffer.data.items()
                   for index, example in enumerate(examples)]
        if not indexed:
            return None
        labels = torch.tensor([item[0] for item in indexed],
                              dtype=torch.long, device=self.args.device)
        usage = torch.tensor([
            self.usage[item[0]][item[1]]
            for item in indexed], device=self.args.device)
        probabilities = sar_probabilities(
            labels, usage, self._similarity(self.args.device),
            self.temperature)
        count = min(self.buffer_batch_size, len(indexed))
        selected = torch.multinomial(probabilities, count,
                                     replacement=False).tolist()
        decay = 1 - count / (self.frequency_scale * len(indexed))
        chosen = set(selected)
        for position, (class_id, index, _) in enumerate(indexed):
            self.usage[class_id][index] = (self.usage[class_id][index] * decay
                                           + (1.0 if position in chosen else 0.0))
        replay_x = torch.stack([indexed[i][2] for i in selected]).to(
            self.args.device)
        replay_y = labels[selected]
        return replay_x, replay_y

    def _replay_loss(self, bottoms, top_model, batch_x, batch_y, loss_ce=None):
        if self.replay_batch is None:
            return loss_ce
        replay_x, replay_y = self.replay_batch
        pieces = split_features(replay_x, self.args)
        embeddings = [bottom(piece) for bottom, piece in zip(bottoms, pieces)]
        self.trainer.bytes_transmitted += sum(
            value.numel() * value.element_size() * 2 for value in embeddings)
        self.trainer.comm_rounds += 1
        self._capture_labels = replay_y
        replay_logits = top_model(self.trainer._aggregate(embeddings))
        return loss_ce + self.alpha * F.cross_entropy(replay_logits, replay_y)

    def train_task(self, train_loader, task_id):
        self.task_id = task_id
        capture = self.trainer.top_model.register_forward_hook(
            self._capture_top_forward)
        self.trainer.before_batch_fn = self._before_batch
        self.trainer.gradient_transform_top = self._record_top_gradients
        self.trainer.gradient_transform_bottom = self._record_gradients
        self.trainer.after_batch_fn = self._after_batch
        try:
            return super().train_task(train_loader, task_id)
        finally:
            self._after_batch()
            capture.remove()
            del self.trainer.before_batch_fn
            del self.trainer.gradient_transform_top
            del self.trainer.gradient_transform_bottom
            del self.trainer.after_batch_fn

    def after_task(self, train_loader, task_id):
        for batch_x, batch_y in train_loader:
            for example, label in zip(batch_x, batch_y):
                class_id = int(label)
                entries = self.buffer.data.setdefault(class_id, [])
                frequencies = self.usage.setdefault(class_id, [])
                seen = self.buffer.seen_count.get(class_id, 0)
                if len(entries) < self.per_class_size:
                    entries.append(example.detach().cpu().clone())
                    frequencies.append(0.0)
                else:
                    replacement = random.randint(0, seen)
                    if replacement < self.per_class_size:
                        entries[replacement] = example.detach().cpu().clone()
                        frequencies[replacement] = 0.0
                self.buffer.seen_count[class_id] = seen + 1

    def get_state(self):
        state = super().get_state()
        state.update(usage=self.usage, step=self.step, fisher=self.fisher,
                     top_fisher=self.top_fisher,
                     class_similarity=self.class_similarity,
                     last_gradient_mean=self.last_gradient_mean)
        return state

    def load_state(self, state):
        super().load_state(state)
        self.usage = state['usage']
        self.step = int(state['step'])
        self.class_similarity = state['class_similarity']
        self.last_gradient_mean = float(state['last_gradient_mean'])
        self.fisher = state['fisher']
        self.top_fisher = float(state['top_fisher'])
