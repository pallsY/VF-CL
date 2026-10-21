"""Deterministic old-class replay sampler for the isolated A/B diagnostic."""

import numpy as np
import torch


class BalancedOldReplay:
    def __init__(self, data, seed, batch_size=64):
        if not data or batch_size < 1:
            raise ValueError('old replay requires nonempty classes and batch')
        if any(not isinstance(raw, torch.Tensor) or len(raw) < 1
               for raw in data.values()):
            raise ValueError('old replay contains an empty class')
        self.data = data
        self.classes = sorted(int(class_id) for class_id in data)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.orders = {class_id: self.rng.permutation(len(data[class_id]))
                       for class_id in self.classes}
        self.positions = {class_id: 0 for class_id in self.classes}
        self.step = 0

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
                len(identities) - len(set(identities)))
