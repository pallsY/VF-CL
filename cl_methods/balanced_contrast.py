"""Balanced Contrast CIL on joint VFL features (paper Eq. 11-13)."""
import random

import torch
import torch.nn.functional as F

from data_utils import split_features
from .er import ERCL


def batch_balance_weights(labels, task_number, gamma):
    if labels.ndim != 1 or not labels.numel() or task_number < 1 or gamma <= 0:
        raise ValueError('invalid balanced contrast labels or temperature')
    _, inverse, counts = labels.unique(return_inverse=True,
                                       return_counts=True)
    return counts[inverse].float().pow(-1.0 / (task_number * gamma))


def proxy_contrastive_loss(features, class_proxies, labels,
                           task_number, gamma=4.0, temperature=0.1):
    if (features.ndim != 2 or class_proxies.ndim != 2
            or len(features) != len(labels)
            or features.size(1) != class_proxies.size(1)
            or temperature <= 0):
        raise ValueError('invalid balanced contrast features or temperature')
    if (labels.min() < 0 or labels.max() >= len(class_proxies)):
        raise ValueError('balanced contrast label exceeds class proxies')
    # Each batch example contributes its class proxy to the denominator, exactly
    # as the paper's proxy-based positive/negative sample formulation does.
    scores = features @ class_proxies[labels].T / temperature
    log_probability = scores - scores.logsumexp(dim=1, keepdim=True)
    positives = labels[:, None].eq(labels[None, :])
    per_anchor = -(log_probability * positives).sum(1) / positives.sum(1)
    weights = batch_balance_weights(labels, task_number, gamma)
    return (weights * per_anchor).mean()


class BalancedContrastCL(ERCL):
    def __init__(self, trainer, args):
        super().__init__(trainer, args)
        self.name = 'BalancedContrast_VFL'
        self.task_number = 0
        self.gamma = float(getattr(args, 'balanced_gamma', 4.0))
        self.temperature = float(getattr(args, 'balanced_temperature', 0.1))
        if self.gamma <= 0 or self.temperature <= 0:
            raise ValueError('balanced contrast temperatures must be positive')

    def before_task(self, task_id, new_classes, seen_classes):
        super().before_task(task_id, new_classes, seen_classes)
        self.task_number = task_id + 1
        # The published similarity is a dot product with class proxy weights.
        # Its classifier has no independent class-bias term.
        bias = self.trainer.top_model.classifier.bias
        with torch.no_grad():
            bias.zero_()
        bias.requires_grad_(False)

    def _balanced_loss(self, bottoms, top_model, batch_x, batch_y, loss_ce=None):
        stored = [(example, label)
                  for label, examples in self.buffer.data.items()
                  for example in examples]
        if stored:
            picked = random.sample(stored, min(self.buffer_batch_size,
                                               len(stored)))
            replay_x = torch.stack([example for example, _ in picked]).to(
                self.args.device)
            replay_y = torch.tensor([label for _, label in picked],
                                    dtype=torch.long, device=self.args.device)
            examples = torch.cat((batch_x, replay_x), 0)
            labels = torch.cat((batch_y, replay_y), 0)
        else:
            examples, labels = batch_x, batch_y
        pieces = split_features(examples, self.args)
        embeddings = [bottom(piece) for bottom, piece in zip(bottoms, pieces)]
        self.trainer.bytes_transmitted += sum(
            value.numel() * value.element_size() * 2 for value in embeddings)
        self.trainer.comm_rounds += 1
        features = self.trainer._aggregate(embeddings)
        return proxy_contrastive_loss(
            features, top_model.classifier.weight, labels,
            self.task_number, self.gamma, self.temperature)

    def train_task(self, train_loader, task_id):
        return self.trainer.train_task(
            train_loader, self.args.epochs_per_task,
            extra_loss_fn=self._balanced_loss)

    def get_state(self):
        state = super().get_state()
        state['task_number'] = self.task_number
        return state

    def load_state(self, state):
        super().load_state(state)
        self.task_number = int(state['task_number'])
