"""Experience Replay (ER) baseline for VFL-CL.

Reservoir sampling buffer storing (x, y) pairs from past tasks.
At each training batch on new task data, we also sample a buffer batch
and add its CE loss to the new-task CE loss.

This serves as a strong middle-ground baseline:
- Stronger than FineTune (which has zero protection).
- Weaker than Oracle (which jointly trains on all data every event).
- A CL method that fails to beat ER provides little practical value.

VFL note: the buffer stores the full image; at training time the trainer
splits it across parties via `split_features`. This is functionally equivalent
to each party caching its own private partition (the active party caches labels).
"""
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader


class ReservoirBuffer:
    """Per-class reservoir buffer: keep at most `per_class_size` samples per class."""
    def __init__(self, per_class_size=20):
        self.per_class_size = per_class_size
        self.data = {}        # class_id -> list of x tensors (CPU)
        self.seen_count = {}  # class_id -> number of samples ever seen for that class

    def add_batch(self, batch_x, batch_y):
        for i in range(batch_y.size(0)):
            c = int(batch_y[i].item())
            x = batch_x[i].detach().cpu().clone()
            self.seen_count[c] = self.seen_count.get(c, 0) + 1
            if c not in self.data:
                self.data[c] = []
            if len(self.data[c]) < self.per_class_size:
                self.data[c].append(x)
            else:
                # Reservoir replacement
                idx = random.randint(0, self.seen_count[c] - 1)
                if idx < self.per_class_size:
                    self.data[c][idx] = x

    def sample(self, n):
        """Sample ~n items CLASS-BALANCED: an equal per-class quota.

        The buffer is per-class with equal caps, so once filled it is already
        balanced over classes. Drawing an explicit per-class quota (rather than
        uniformly from the flattened pool) guarantees that *every* seen class is
        present in *every* replay batch with the same count -- uniform global
        sampling can, by chance, under-represent or omit a class in a given
        mini-batch, which leaves that old class momentarily unprotected and adds
        variance to the replay gradient. Returns (x_tensor, y_tensor) or None.
        """
        classes = [c for c, xs in self.data.items() if xs]
        if not classes:
            return None
        quota = max(1, n // len(classes))
        picked = []
        for c in classes:
            xs = self.data[c]
            k = min(quota, len(xs))
            picked.extend((x, c) for x in random.sample(xs, k))
        xs = torch.stack([p[0] for p in picked])
        ys = torch.tensor([p[1] for p in picked], dtype=torch.long)
        return xs, ys

    def total_size(self):
        return sum(len(v) for v in self.data.values())


class ERCL:
    """Experience Replay for VFL.
    Maintains a reservoir buffer. On task t>0, each training step also runs
    a forward+CE on a buffer mini-batch.
    """
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'ER_VFL'
        self.per_class_size = getattr(args, 'er_per_class', 300)
        self.buffer_batch_size = getattr(args, 'er_batch', args.batch_size)
        # Replay up-weight: new-task CE covers the 2 fresh classes densely while
        # replay is spread over all old classes, so per-class the new classes get
        # far more gradient. alpha>1 rebalances toward retention. alpha=1 keeps
        # the standard 1:1 ER objective.
        self.alpha = getattr(args, 'er_alpha', 1.0)
        self.buffer = ReservoirBuffer(self.per_class_size)

    def before_task(self, task_id, new_classes, seen_classes):
        req = max(seen_classes) + 1 if seen_classes else 0
        self.trainer.top_model.expand_classes(req, self.args.device)

    def _replay_loss(self, bottoms, top_model, batch_x, batch_y, loss_ce=None):
        """Sample a buffer batch and add its CE loss."""
        sample = self.buffer.sample(self.buffer_batch_size)
        if sample is None:
            return loss_ce if loss_ce is not None else torch.tensor(0.0, device=self.args.device)
        bx, by = sample
        bx = bx.to(self.args.device)
        by = by.to(self.args.device)
        from data_utils import split_features
        parts = split_features(bx, self.args)
        embs = [bottoms[i](parts[i]) for i in range(len(bottoms))]
        agg = self.trainer._aggregate(embs)
        replay_loss = nn.CrossEntropyLoss()(top_model(agg), by)
        if loss_ce is not None:
            return loss_ce + self.alpha * replay_loss
        return self.alpha * replay_loss

    def train_task(self, train_loader, task_id):
        extra = self._replay_loss if task_id > 0 and self.buffer.total_size() > 0 else None
        return self.trainer.train_task(train_loader, self.args.epochs_per_task, extra_loss_fn=extra)

    def after_task(self, train_loader, task_id):
        """Add current task's data to buffer."""
        for bx, by in train_loader:
            self.buffer.add_batch(bx, by)
        print(f"  ER buffer size after task {task_id}: {self.buffer.total_size()} samples, "
              f"{len(self.buffer.data)} classes")

    def get_state(self):
        return {
            'per_class_size': self.buffer.per_class_size,
            'data': {
                int(class_id): torch.stack(samples)
                for class_id, samples in self.buffer.data.items()
                if samples
            },
            'seen_count': dict(self.buffer.seen_count),
        }

    def load_state(self, s):
        if int(s.get('per_class_size', self.buffer.per_class_size)) != self.buffer.per_class_size:
            raise ValueError('ER checkpoint uses a different per-class buffer size')
        self.buffer.data = {
            int(class_id): [sample.clone() for sample in packed.unbind(0)]
            for class_id, packed in s.get('data', {}).items()
        }
        self.buffer.seen_count = {
            int(class_id): int(count)
            for class_id, count in s.get('seen_count', {}).items()
        }
