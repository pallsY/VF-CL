"""ER-ACE for VFL: asymmetric incoming CE plus raw-example replay CE."""
import random

import torch
import torch.nn.functional as F
from data_utils import split_features


class ReservoirBuffer:
    """Flat raw-example reservoir; Python RNG is restored by the runner."""

    def __init__(self, buffer_size, num_classes):
        if type(buffer_size) is not int or buffer_size <= 0:
            raise ValueError('ER-ACE buffer size must be a positive integer')
        if type(num_classes) is not int or num_classes <= 0:
            raise ValueError('ER-ACE class count must be a positive integer')
        self.buffer_size = buffer_size
        self.num_classes = num_classes
        self.num_seen = 0
        self.ex, self.lb = [], []

    def _validate_batch(self, examples, labels):
        if (not isinstance(labels, torch.Tensor) or labels.layout != torch.strided
                or labels.ndim != 1 or labels.dtype != torch.long):
            raise ValueError('ER-ACE labels must be a one-dimensional int64 tensor')
        if (not isinstance(examples, torch.Tensor)
                or examples.layout != torch.strided or examples.ndim < 2
                or not examples.is_floating_point()
                or any(dim == 0 for dim in examples.shape[1:])):
            raise ValueError('ER-ACE examples must be floating tensors with nonempty features')
        if examples.size(0) != labels.size(0):
            raise ValueError('ER-ACE examples and labels have different lengths')
        if not torch.isfinite(examples).all().item():
            raise ValueError('ER-ACE examples must be finite')
        if ((labels < 0) | (labels >= self.num_classes)).any().item():
            raise ValueError('ER-ACE labels are outside the class range')
        if self.ex and (examples.shape[1:] != self.ex[0].shape
                        or examples.dtype != self.ex[0].dtype):
            raise ValueError('ER-ACE example shape or dtype changed')

    def add_batch(self, examples, labels):
        self._validate_batch(examples, labels)
        for example, label in zip(examples, labels):
            index = (len(self.ex) if len(self.ex) < self.buffer_size
                     else random.randint(0, self.num_seen))
            if index < self.buffer_size:
                value = example.detach().cpu().clone()
                if index == len(self.ex):
                    self.ex.append(value)
                    self.lb.append(int(label))
                else:
                    self.ex[index], self.lb[index] = value, int(label)
            self.num_seen += 1

    def sample(self, count):
        if type(count) is not int or count < 0:
            raise ValueError('ER-ACE sample count must be a nonnegative integer')
        if not self.ex or count == 0:
            return None
        indices = random.sample(range(len(self.ex)), min(count, len(self.ex)))
        return (torch.stack([self.ex[i] for i in indices]),
                torch.tensor([self.lb[i] for i in indices], dtype=torch.long))

    def size(self):
        return len(self.ex)

    def purge_classes(self, forget_classes):
        """Atomically restart raw replay from retained examples after deletion."""
        if (type(self.num_seen) is not int or self.num_seen < 0
                or len(self.ex) != len(self.lb)
                or len(self.ex) != min(self.buffer_size, self.num_seen)):
            raise ValueError('ER-ACE reservoir fields/count are inconsistent')
        forgotten = {int(c) for c in forget_classes}
        keep = [i for i, label in enumerate(self.lb) if label not in forgotten]
        removed = len(self.lb) - len(keep)
        if removed:
            # No discarded-stream class counts exist. The retained reservoir is
            # the new effective stream; no-op purges keep the original history.
            self.ex, self.lb, self.num_seen = (
                [self.ex[i] for i in keep], [self.lb[i] for i in keep], len(keep))
        return removed

    def get_state(self):
        return {
            'buffer_size': self.buffer_size,
            'num_seen': self.num_seen,
            'examples': torch.stack(self.ex) if self.ex else None,
            'labels': torch.tensor(self.lb, dtype=torch.long),
        }

    def load_state(self, state):
        if not isinstance(state, dict) or set(state) != {
                'buffer_size', 'num_seen', 'examples', 'labels'}:
            raise ValueError('ER-ACE checkpoint requires exact reservoir fields')
        if (type(state['buffer_size']) is not int
                or state['buffer_size'] != self.buffer_size):
            raise ValueError('ER-ACE checkpoint uses a different buffer size')
        num_seen = state['num_seen']
        if type(num_seen) is not int or num_seen < 0:
            raise ValueError('ER-ACE stream count must be a nonnegative integer')
        examples, labels = state['examples'], state['labels']
        if examples is None:
            if (not isinstance(labels, torch.Tensor) or labels.layout != torch.strided
                    or labels.dtype != torch.long or labels.shape != (0,)):
                raise ValueError('ER-ACE empty buffer requires empty int64 labels')
            size = 0
        else:
            self._validate_batch(examples, labels)
            if examples.size(0) == 0:
                raise ValueError('ER-ACE empty checkpoint examples must be None')
            size = examples.size(0)
        if size != min(self.buffer_size, num_seen):
            raise ValueError('ER-ACE buffer size is inconsistent with stream count')
        new_ex = ([] if examples is None else
                  [value.detach().cpu().clone() for value in examples.unbind(0)])
        new_lb = labels.detach().cpu().tolist()
        # Assign only after the entire checkpoint has passed validation.
        self.ex, self.lb, self.num_seen = new_ex, new_lb, num_seen


class ERAccCL:
    """Current-task-only incoming CE plus full-classifier reservoir replay."""
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'ERACE_VFL'
        self.old_classes = []
        self.new_classes = []
        capacity = getattr(args, 'er_ace_buffer_size', 0)
        if type(capacity) is not int or capacity < 0:
            raise ValueError('ER-ACE buffer size must be a nonnegative integer')
        if type(args.num_classes) is not int or args.num_classes <= 0:
            raise ValueError('ER-ACE class count must be a positive integer')
        self.mb = getattr(args, 'er_ace_batch', 64)
        if type(self.mb) is not int or self.mb <= 0:
            raise ValueError('ER-ACE replay batch must be a positive integer')
        self.buffer_size = capacity if capacity else 20 * args.num_classes
        self.buffer = ReservoirBuffer(self.buffer_size, args.num_classes)

    def before_task(self, task_id, new_classes, seen_classes):
        for classes in (new_classes, seen_classes):
            if (not isinstance(classes, (list, tuple)) or not classes
                    or any(type(c) is not int or not 0 <= c < self.args.num_classes
                           for c in classes)
                    or len(set(classes)) != len(classes)):
                raise ValueError('ER-ACE classes must be unique in-range integers')
        if not set(new_classes).issubset(seen_classes):
            raise ValueError('ER-ACE current classes must be seen classes')
        req = max(seen_classes) + 1 if seen_classes else 0
        self.trainer.top_model.expand_classes(req, self.args.device)
        self.old_classes = [c for c in seen_classes if c not in new_classes]
        self.new_classes = list(new_classes)
        self.trainer.ce_classes = list(new_classes)

    def _forward(self, bottoms, top_model, batch_x):
        parts = split_features(batch_x.to(self.args.device), self.args)
        embs = [bottoms[i](parts[i]) for i in range(len(bottoms))]
        return top_model(self.trainer._aggregate(embs))

    def _er_ace_loss(self, bottoms, top_model, batch_x, batch_y, loss_ce=None):
        """Reuse first-forward incoming ACE and add replay before insertion."""
        self.buffer._validate_batch(batch_x, batch_y)
        if not self.new_classes:
            raise ValueError('ER-ACE current classes require before_task')
        device = self.args.device
        classes = torch.tensor(self.new_classes, device=device, dtype=torch.long)
        matches = batch_y.to(device)[:, None] == classes
        if batch_y.numel() == 0 or not matches.any(dim=1).all().item():
            raise ValueError('ER-ACE incoming labels must belong to current classes')
        if loss_ce is None:
            raise ValueError('ER-ACE requires the trainer incoming ACE loss')
        loss = loss_ce
        replay = self.buffer.sample(self.mb)
        if replay is not None:
            replay_x, replay_y = replay
            replay_logits = self._forward(bottoms, top_model, replay_x)
            loss = loss + F.cross_entropy(replay_logits, replay_y.to(device))
        # add_batch stores detached CPU clones; no extra forward or mode changes.
        self.buffer.add_batch(batch_x, batch_y)
        return loss

    def train_task(self, train_loader, task_id):
        return self.trainer.train_task(train_loader, self.args.epochs_per_task,
                                       extra_loss_fn=self._er_ace_loss)

    def after_task(self, train_loader, task_id):
        pass

    def get_state(self):
        return self.buffer.get_state()

    def load_state(self, s):
        self.buffer.load_state(s)
