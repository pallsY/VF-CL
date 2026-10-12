"""HPCR proxy contrastive, proxy-logit, and structure distillation in VFL."""
import math
import random

import torch
import torch.nn.functional as F

from data_utils import split_features


def proxy_supcon_loss(features, proxies, labels, temperature=0.09):
    """Author HPCR's SupConLoss in proxy mode, with class-weight proxies."""
    if (features.ndim != 2 or proxies.ndim != 2
            or len(features) != len(labels)
            or features.size(1) != proxies.size(1)
            or temperature <= 0):
        raise ValueError('invalid HPCR proxy contrastive inputs')
    values = F.normalize(features, dim=1)
    anchors = F.normalize(proxies[labels], dim=1)
    scores = values @ anchors.T / temperature
    log_probability = scores - scores.logsumexp(1, keepdim=True)
    positives = labels[:, None].eq(labels[None, :])
    return -((positives * log_probability).sum(1)
             / positives.sum(1)).mean()


def similarity_distillation_loss(features, stored_features,
                                 temperature=0.09):
    """Cross entropy of non-self similarity distributions (HPCR SCD)."""
    if (features.shape != stored_features.shape or features.ndim != 2
            or temperature <= 0):
        raise ValueError('invalid HPCR structure distillation inputs')
    if len(features) < 2:
        return features.sum() * 0
    current = F.normalize(features, dim=1)
    stored = F.normalize(stored_features, dim=1)
    current_logits = current @ current.T / temperature
    stored_logits = stored @ stored.T / temperature
    diagonal = torch.eye(len(features), dtype=torch.bool,
                         device=features.device)
    current_logits = current_logits.masked_fill(diagonal, -torch.inf)
    stored_logits = stored_logits.masked_fill(diagonal, -torch.inf)
    old_probability = F.softmax(stored_logits.detach(), dim=1)
    return -(old_probability * F.log_softmax(current_logits, dim=1)
             ).masked_fill(diagonal, 0).sum(1).mean()


class HPCRCL:
    def __init__(self, trainer, args):
        self.trainer, self.args = trainer, args
        self.name = 'HPCR_VFL'
        self.per_class_size = int(getattr(args, 'er_per_class', 20))
        self.replay_size = int(getattr(args, 'er_batch', 64))
        self.temperature = float(getattr(args, 'hpcr_temperature', 0.09))
        self.max_temp = float(getattr(args, 'hpcr_temp_max', 1.8))
        self.min_temp = float(getattr(args, 'hpcr_temp_min', 0.5625))
        self.pcd_alpha = float(getattr(args, 'hpcr_pcd_alpha', 0.1))
        self.scd_beta = float(getattr(args, 'hpcr_scd_beta', 0.5))
        if (self.per_class_size < 1 or self.replay_size < 1
                or min(self.temperature, self.max_temp, self.min_temp) <= 0
                or min(self.pcd_alpha, self.scd_beta) < 0):
            raise ValueError('invalid HPCR memory or loss settings')
        self.memory = {}
        self.seen_count = {}
        self.step = 0

    def before_task(self, task_id, new_classes, seen_classes):
        self.trainer.top_model.expand_classes(max(seen_classes) + 1,
                                              self.args.device)

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
            cyclic = ((self.max_temp - self.min_temp) * 0.5
                      * (1 - math.cos(2 * math.pi * self.step / 500))
                      + self.min_temp)
            loss = cyclic * proxy_supcon_loss(
                all_features, top_model.classifier.weight, all_labels,
                self.temperature)
            valid = torch.isfinite(old_logits)
            target_logits = old_logits.repeat(2, 1)
            valid = valid.repeat(2, 1)
            if valid.any():
                current_full = memory_logits.new_zeros(target_logits.shape)
                current_full[:, :memory_logits.size(1)] = memory_logits
                loss = loss + self.pcd_alpha * F.mse_loss(
                    current_full[valid], target_logits[valid])
            loss = loss + self.scd_beta * similarity_distillation_loss(
                memory_features, old_features.repeat(2, 1),
                self.temperature)
        else:
            loss = 0 * F.cross_entropy(original_logits, batch_y)
        self._insert(batch_x, batch_y, original_logits, original_features)
        self.step += 1
        return loss

    def train_task(self, train_loader, task_id):
        return self.trainer.train_task(
            train_loader, self.args.epochs_per_task,
            extra_loss_fn=self._hpcr_loss)

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
