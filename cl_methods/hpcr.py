"""HPCR proxy contrastive, proxy-logit, and structure distillation in VFL."""
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import split_features


def proxy_supcon_loss(features, proxies, labels, temperature=0.09,
                      min_samples=20):
    """Paper Eq. 18: selected proxies plus gated non-self sample pairs."""
    if (features.ndim != 2 or proxies.ndim != 2
            or len(features) != len(labels)
            or features.size(1) != proxies.size(1)
            or temperature <= 0 or min_samples < 1):
        raise ValueError('invalid HPCR proxy contrastive inputs')
    values = F.normalize(features, dim=1)
    selected = F.normalize(proxies[labels], dim=1)
    proxy_scores = values @ selected.T / temperature
    own_proxy = proxy_scores.diagonal()
    if len(labels) < min_samples:
        return (proxy_scores.logsumexp(1) - own_proxy).mean()
    sample_scores = values @ values.T / temperature
    nonself = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    denominator = torch.logsumexp(torch.cat((
        proxy_scores, sample_scores.masked_fill(~nonself, -torch.inf)), 1), 1)
    positives = labels[:, None].eq(labels[None, :]) & nonself
    numerator = torch.logaddexp(own_proxy[:, None], sample_scores)
    counts = positives.sum(1)
    per_anchor = denominator - (numerator * positives).sum(1) / counts.clamp_min(1)
    return torch.where(counts > 0, per_anchor, denominator - own_proxy).mean()


def similarity_distillation_loss(features, stored_features,
                                 temperature=0.09):
    """Paper Eq. 25: current non-self probabilities against stored log-probs."""
    if (features.shape != stored_features.shape or features.ndim != 2
            or temperature <= 0):
        raise ValueError('invalid HPCR structure distillation inputs')
    if len(features) < 2:
        return features.sum() * 0
    current = F.normalize(features, dim=1)
    stored = F.normalize(stored_features, dim=1)
    nonself = ~torch.eye(len(features), dtype=torch.bool,
                         device=features.device)
    current_logits = (current @ current.T / temperature)[nonself].reshape(
        len(features), -1)
    stored_logits = (stored @ stored.T / temperature)[nonself].reshape(
        len(features), -1)
    current_probability = F.softmax(current_logits, dim=1)
    return -(current_probability * F.log_softmax(stored_logits.detach(), dim=1)
             ).sum(1).mean()


def proxy_logit_distillation_loss(current_logits, stored_logits, batch_labels):
    """Paper Eq. 24, with positive squared error as required by its prose.

    The printed leading minus before the squared sum contradicts the stated
    minimization of Euclidean distance; positive error also matches author code.
    Repeated batch labels implement the paper's class multiplicities k_c.
    """
    columns = batch_labels[batch_labels < min(current_logits.size(1),
                                               stored_logits.size(1))]
    if not len(columns):
        return current_logits.sum() * 0
    current = current_logits[:, columns]
    stored = stored_logits[:, columns]
    valid = torch.isfinite(stored)
    if not valid.any():
        return current_logits.sum() * 0
    safe_stored = torch.where(valid, stored, torch.zeros_like(stored))
    errors = (current - safe_stored).square().masked_fill(~valid, 0)
    return errors.sum(1).mean()


class HPCRCL:
    def __init__(self, trainer, args):
        self.trainer, self.args = trainer, args
        self.name = 'HPCR_VFL'
        self.per_class_size = int(getattr(args, 'er_per_class', 20))
        self.replay_size = int(getattr(args, 'er_batch', 64))
        self.temperature = float(getattr(args, 'hpcr_temperature', 0.09))
        self.max_temp = float(getattr(args, 'hpcr_temp_max', 0.16))
        self.min_temp = float(getattr(args, 'hpcr_temp_min', 0.05))
        # Existing matrix/CLI values are tau/min_tau and tau/max_tau.
        # Decode that exact legacy pair so the effective schedule is Eq. 21.
        if self.max_temp == 1.8 and self.min_temp == 0.5625:
            self.max_temp = self.temperature / self.min_temp
            self.min_temp = self.temperature / 1.8
        self.min_samples = int(getattr(args, 'hpcr_min_samples', 20))
        self.pcd_alpha = float(getattr(args, 'hpcr_pcd_alpha', 0.1))
        self.scd_beta = float(getattr(args, 'hpcr_scd_beta', 0.5))
        if (self.per_class_size < 1 or self.replay_size < 1
                or min(self.temperature, self.max_temp, self.min_temp) <= 0
                or min(self.pcd_alpha, self.scd_beta) < 0
                or self.max_temp < self.min_temp or self.min_samples < 1):
            raise ValueError('invalid HPCR memory or loss settings')
        self.memory = {}
        self.seen_count = {}
        self.step = 0
        self._task_step = 0
        self._steps_per_epoch = None
        if hasattr(trainer, 'top_model'):
            self._use_cosine_head(trainer.top_model)

    def _use_cosine_head(self, top_model):
        # TopModel already implements normalized, bias-free cosine logits.
        # Configure this HPCR instance before checkpoints or optimizers are built.
        top_model.cosine = True
        value = 1.0 / self.temperature
        if hasattr(top_model, 'scale'):
            with torch.no_grad():
                top_model.scale.fill_(value)
            top_model.scale.requires_grad_(False)
        else:
            top_model.scale = nn.Parameter(
                top_model.classifier.weight.new_tensor(value),
                requires_grad=False)
        with torch.no_grad():
            top_model.classifier.bias.zero_()
        top_model.classifier.bias.requires_grad_(False)

    def before_task(self, task_id, new_classes, seen_classes):
        self.trainer.top_model.expand_classes(max(seen_classes) + 1,
                                              self.args.device)
        self._use_cosine_head(self.trainer.top_model)
        self._task_step = 0

    def _augment(self, examples):
        if examples.ndim == 4:
            flip = torch.rand(len(examples), device=examples.device) < 0.5
            augmented = examples.clone()
            augmented[flip] = augmented[flip].flip(-1)
            return augmented
        # Feature-space second view for ISOLET and UPMC's frozen descriptors.
        return examples + torch.randn_like(examples) * 0.01

    def _forward(self, bottoms, top_model, examples):
        pieces = split_features(examples, self.args)
        embeddings = [bottom(piece) for bottom, piece in zip(bottoms, pieces)]
        self.trainer.bytes_transmitted += sum(
            value.numel() * value.element_size() * 2 for value in embeddings)
        self.trainer.comm_rounds += 1
        features = self.trainer._aggregate(embeddings)
        return top_model(features), features

    def _sample(self):
        stored = [entry for entries in self.memory.values() for entry in entries]
        return random.sample(stored, min(self.replay_size, len(stored)))

    def _contrastive_scale(self):
        tau_step = ((self.max_temp - self.min_temp) * 0.5
                    * (1 + math.cos(2 * math.pi * self.step / 500))
                    + self.min_temp)
        return self.temperature / tau_step

    def _insert(self, examples, labels, logits, features):
        for example, label, logit, feature in zip(examples, labels, logits,
                                                   features):
            class_id = int(label)
            count = self.seen_count.get(class_id, 0)
            entries = self.memory.setdefault(class_id, [])
            packed_logits = torch.full((self.args.num_classes,), float('nan'))
            packed_logits[:len(logit)] = logit.detach().cpu()
            entry = (example.detach().cpu().clone(), class_id,
                     packed_logits, feature.detach().cpu().clone())
            if len(entries) < self.per_class_size:
                entries.append(entry)
            else:
                index = random.randint(0, count)
                if index < self.per_class_size:
                    entries[index] = entry
            self.seen_count[class_id] = count + 1

    def _hpcr_loss(self, bottoms, top_model, batch_x, batch_y, loss_ce=None):
        original_logits, original_features = self._forward(
            bottoms, top_model, batch_x)
        current_x = torch.cat((batch_x, self._augment(batch_x)))
        current_y = batch_y.repeat(2)
        _, current_features = self._forward(bottoms, top_model, current_x)
        selected = self._sample()
        if selected:
            old_x = torch.stack([entry[0] for entry in selected]).to(
                self.args.device)
            old_y = torch.tensor([entry[1] for entry in selected],
                                 dtype=torch.long, device=self.args.device)
            old_logits = torch.stack([entry[2] for entry in selected]).to(
                self.args.device)
            old_features = torch.stack([entry[3] for entry in selected]).to(
                self.args.device)
            memory_x = torch.cat((old_x, self._augment(old_x)))
            memory_y = old_y.repeat(2)
            memory_logits, memory_features = self._forward(
                bottoms, top_model, memory_x)
            all_features = torch.cat((memory_features, current_features))
            all_labels = torch.cat((memory_y, current_y))
            loss = self._contrastive_scale() * proxy_supcon_loss(
                all_features, top_model.classifier.weight, all_labels,
                self.temperature, self.min_samples)
            target_logits = old_logits.repeat(2, 1)
            loss = loss + self.pcd_alpha * proxy_logit_distillation_loss(
                memory_logits, target_logits, all_labels)
            loss = loss + self.scd_beta * similarity_distillation_loss(
                memory_features, old_features.repeat(2, 1),
                self.temperature)
        else:
            loss = self._contrastive_scale() * proxy_supcon_loss(
                current_features, top_model.classifier.weight, current_y,
                self.temperature, self.min_samples)
        if self._steps_per_epoch is None or self._task_step < self._steps_per_epoch:
            self._insert(batch_x, batch_y, original_logits, original_features)
        self._task_step += 1
        self.step += 1
        return loss

    def train_task(self, train_loader, task_id):
        self._task_step = 0
        self._steps_per_epoch = len(train_loader)
        try:
            return self.trainer.train_task(
                train_loader, self.args.epochs_per_task,
                extra_loss_fn=self._hpcr_loss)
        finally:
            self._steps_per_epoch = None

    def after_task(self, train_loader, task_id):
        pass

    def get_state(self):
        return {
            'per_class_size': self.per_class_size,
            'memory': self.memory,
            'seen_count': self.seen_count,
            'step': self.step,
        }

    def load_state(self, state):
        if state['per_class_size'] != self.per_class_size:
            raise ValueError('HPCR checkpoint memory budget differs')
        self.memory = state['memory']
        self.seen_count = state['seen_count']
        self.step = int(state['step'])
