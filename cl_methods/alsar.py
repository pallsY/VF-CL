"""Adaptive layer freezing and similarity-aware retrieval for VFL."""
import random

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
    effective = (discounted_usage
                 + (similarity.T @ class_usage)[labels]).clamp_min(0)
    return F.softmax(-effective / temperature, dim=0)


class ALSARCL(ERCL):
    def __init__(self, trainer, args):
        super().__init__(trainer, args)
        self.name = 'aL-SAR_VFL'
        self.temperature = float(getattr(args, 'alsar_temperature', 1.0))
        self.decay = float(getattr(args, 'alsar_decay', 0.99))
        self.fisher_rate = float(getattr(args, 'alsar_fisher_rate', 0.01))
        self.gradient_rate = float(getattr(args, 'alsar_gradient_rate', 0.01))
        if (self.temperature <= 0 or not 0 < self.decay < 1
                or not 0 < self.fisher_rate <= 1
                or not 0 < self.gradient_rate <= 1):
            raise ValueError('invalid aL-SAR hyperparameters')
        self.usage, self.last_used = {}, {}
        self.step = 0
        self.class_gradient = {}
        self.last_gradient_mean = 1.0
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
        self.frozen = []

    def before_task(self, task_id, new_classes, seen_classes):
        super().before_task(task_id, new_classes, seen_classes)
        self.frozen = []

    @torch.no_grad()
    def _preliminary_forward(self, examples):
        pieces = split_features(examples, self.args)
        module_modes = [module.training for module in
                        [*self.trainer.bottoms, self.trainer.top_model]]
        all_modules = [*self.trainer.bottoms, self.trainer.top_model]
        hooks = []
        cost = [[0.0] * len(parts) for parts in self.stages]

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
                    [max(value, 1.0) for value in measured], dtype=torch.float64)
        return features.detach(), logits.detach()

    def _before_batch(self, examples, labels):
        features, logits = self._preliminary_forward(examples)
        self.batch_features, self.batch_logits = features, logits
        self.batch_head_weight = (
            self.trainer.top_model.classifier.weight.detach().clone())
        probabilities = logits.softmax(1)
        targets = F.one_hot(labels, logits.size(1)).float()
        output_gradient = (probabilities - targets) @ self.trainer.top_model.classifier.weight
        current_gradient = float(output_gradient.square().sum(1).mean())
        ratio = current_gradient / max(self.last_gradient_mean, 1e-10)
        self.last_gradient_mean += self.fisher_rate * (
            current_gradient - self.last_gradient_mean)
        self.frozen = []
        for party, parts in enumerate(self.stages):
            backward = self.costs[party] * 2
            forward = float(self.costs[party].sum())
            prefix = choose_freeze_prefix(
                torch.tensor(self.fisher[party], dtype=torch.float64),
                ratio, forward, backward)
            self.frozen.append(prefix)
            for block in parts[:prefix]:
                for parameter in block.parameters():
                    parameter.requires_grad_(False)
        self.batch_labels = labels.detach()
        self.step += 1

    def _record_gradients(self, _bottoms):
        for party, parts in enumerate(self.stages):
            for index, block in enumerate(parts):
                if index < self.frozen[party]:
                    continue
                observed = sum(float(parameter.grad.detach().clamp(-1, 1).square().sum())
                               for parameter in block.parameters()
                               if parameter.grad is not None)
                self.fisher[party][index] += self.fisher_rate * (
                    observed - self.fisher[party][index])
        with torch.no_grad():
            probabilities = self.batch_logits.softmax(1)
            targets = F.one_hot(self.batch_labels,
                                self.batch_logits.size(1)).float()
            gradients = (probabilities - targets) @ self.batch_head_weight
            for class_id in self.batch_labels.unique().tolist():
                current = gradients[self.batch_labels == class_id].mean(0)
                previous = self.class_gradient.get(class_id)
                self.class_gradient[class_id] = (
                    current.cpu() if previous is None else
                    previous + self.gradient_rate * (current.cpu() - previous))

    def _after_batch(self):
        for parts, prefix in zip(self.stages, self.frozen):
            for block in parts[:prefix]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)

    def _similarity(self, device):
        classes = max(self.buffer.data.keys(), default=-1) + 1
        if classes == 0:
            return torch.empty(0, 0, device=device)
        vectors = torch.stack([
            self.class_gradient.get(c, torch.zeros(
                self.trainer.top_model.classifier.in_features))
            for c in range(classes)]).to(device)
        similarities = F.normalize(vectors, dim=1) @ F.normalize(vectors, dim=1).T
        similarities.fill_diagonal_(1.0)
        return similarities

    def _replay_loss(self, bottoms, top_model, batch_x, batch_y, loss_ce=None):
        indexed = [(class_id, index, example)
                   for class_id, examples in self.buffer.data.items()
                   for index, example in enumerate(examples)]
        if not indexed:
            return loss_ce
        labels = torch.tensor([item[0] for item in indexed],
                              dtype=torch.long, device=self.args.device)
        usage = torch.tensor([
            self.usage[item[0]][item[1]]
            * self.decay ** (self.step - self.last_used[item[0]][item[1]])
            for item in indexed], device=self.args.device)
        probabilities = sar_probabilities(
            labels, usage, self._similarity(self.args.device),
            self.temperature)
        count = min(self.buffer_batch_size, len(indexed))
        selected = torch.multinomial(probabilities, count,
                                     replacement=False).tolist()
        for position in selected:
            class_id, index, _ = indexed[position]
            self.usage[class_id][index] = float(usage[position]) + 1
            self.last_used[class_id][index] = self.step
        replay_x = torch.stack([indexed[i][2] for i in selected]).to(
            self.args.device)
        replay_y = labels[selected]
        pieces = split_features(replay_x, self.args)
        embeddings = [bottom(piece) for bottom, piece in zip(bottoms, pieces)]
        self.trainer.bytes_transmitted += sum(
            value.numel() * value.element_size() * 2 for value in embeddings)
        self.trainer.comm_rounds += 1
        replay_logits = top_model(self.trainer._aggregate(embeddings))
        return loss_ce + self.alpha * F.cross_entropy(replay_logits, replay_y)

    def train_task(self, train_loader, task_id):
        self.trainer.before_batch_fn = self._before_batch
        self.trainer.gradient_transform_bottom = self._record_gradients
        self.trainer.after_batch_fn = self._after_batch
        try:
            return super().train_task(train_loader, task_id)
        finally:
            self._after_batch()
            del self.trainer.before_batch_fn
            del self.trainer.gradient_transform_bottom
            del self.trainer.after_batch_fn

    def after_task(self, train_loader, task_id):
        for batch_x, batch_y in train_loader:
            for example, label in zip(batch_x, batch_y):
                class_id = int(label)
                entries = self.buffer.data.setdefault(class_id, [])
                frequencies = self.usage.setdefault(class_id, [])
                timestamps = self.last_used.setdefault(class_id, [])
                seen = self.buffer.seen_count.get(class_id, 0)
                if len(entries) < self.per_class_size:
                    entries.append(example.detach().cpu().clone())
                    frequencies.append(0.0)
                    timestamps.append(self.step)
                else:
                    replacement = random.randint(0, seen)
                    if replacement < self.per_class_size:
                        entries[replacement] = example.detach().cpu().clone()
                        frequencies[replacement] = 0.0
                        timestamps[replacement] = self.step
                self.buffer.seen_count[class_id] = seen + 1

    def get_state(self):
        state = super().get_state()
        state.update(usage=self.usage, last_used=self.last_used,
                     step=self.step, fisher=self.fisher,
                     class_gradient=self.class_gradient,
                     last_gradient_mean=self.last_gradient_mean)
        return state

    def load_state(self, state):
        super().load_state(state)
        self.usage = state['usage']
        self.last_used = state['last_used']
        self.step = int(state['step'])
        self.class_gradient = state['class_gradient']
        self.last_gradient_mean = float(state['last_gradient_mean'])
        self.fisher = state['fisher']
