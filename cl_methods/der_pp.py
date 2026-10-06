"""DER++ for VFL — ported faithfully from the official Mammoth implementation
(aimagelab/mammoth, models/derpp.py; Buzzega et al., NeurIPS 2020).

Mammoth observe() (per batch):
    loss  = CE(net(inputs), labels)                          # current task
    if not buffer.is_empty():
        bi, _,  blog = buffer.get_data(mb)                   # draw #1
        loss += alpha * MSE(net(bi), blog)                   # dark experience
        bi, blab, _  = buffer.get_data(mb)                   # draw #2 (independent)
        loss += beta  * CE(net(bi), blab)                    # replay
    buffer.add_data(examples=not_aug_inputs, labels=labels, logits=outputs.data)

Adaptation to this VFL / task-based harness:
  * The top head is now fixed full-width (models.build_models), so stored logits
    are always `num_classes` wide — no padding/masking needed (matches Mammoth's
    fixed output layer).
  * Mammoth updates the reservoir ONLINE (every step) via
        buffer.add_data(examples=not_aug_inputs, labels=labels, logits=outputs.data)
    so the *current* task is replayable during its own training. We do the same:
    each mini-batch is inserted into the reservoir from the per-batch loss hook
    with the current model's logits. (An earlier offline variant filled the buffer
    once in after_task with end-of-task logits — that left a "one-task lag" where a
    fresh task was not in the buffer until after it finished, so it underlearned and
    only recovered one task later. Online insertion removes that lag.)
    Two independent buffer draws are preserved.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from data_utils import split_features


class ReservoirBuffer:
    """Flat reservoir over the whole stream (Mammoth-style), storing
    (example, label, logits)."""
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_seen = 0
        self.ex, self.lb, self.lg = [], [], []

    def add(self, x, y, logit):
        x = x.detach().cpu().clone()
        logit = logit.detach().cpu().clone()
        if len(self.ex) < self.buffer_size:
            self.ex.append(x); self.lb.append(int(y)); self.lg.append(logit)
        else:
            idx = random.randint(0, self.num_seen)      # reservoir sampling
            if idx < self.buffer_size:
                self.ex[idx] = x; self.lb[idx] = int(y); self.lg[idx] = logit
        self.num_seen += 1

    def get_data(self, n):
        if not self.ex:
            return None
        idx = random.sample(range(len(self.ex)), min(n, len(self.ex)))
        xs = torch.stack([self.ex[i] for i in idx])
        ys = torch.tensor([self.lb[i] for i in idx], dtype=torch.long)
        lg = torch.stack([self.lg[i] for i in idx])
        return xs, ys, lg

    def size(self):
        return len(self.ex)


class DERppCL:
    """DER++ for VFL: reservoir buffer + logit-MSE (dark experience) + CE replay."""
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'DERpp_VFL'
        self.alpha = getattr(args, 'der_alpha', 0.5)   # MSE (dark experience)
        self.beta = getattr(args, 'der_beta', 0.5)     # CE replay
        # buffer/batch: 0 (or unset) -> auto defaults, else use the CLI override
        bs = getattr(args, 'der_buffer_size', 0)
        self.buffer_size = bs if bs and bs > 0 else 20 * args.num_classes
        mb = getattr(args, 'der_batch', 0)
        self.mb = mb if mb and mb > 0 else args.batch_size
        self.buffer = ReservoirBuffer(self.buffer_size)

    def before_task(self, task_id, new_classes, seen_classes):
        req = max(seen_classes) + 1 if seen_classes else 0
        self.trainer.top_model.expand_classes(req, self.args.device)  # no-op (fixed head)

    def _forward(self, bottoms, top_model, x):
        parts = split_features(x.to(self.args.device), self.args)
        embs = [bottoms[i](parts[i]) for i in range(len(bottoms))]
        return top_model(self.trainer._aggregate(embs))

    @torch.no_grad()
    def _insert_current_batch(self, bottoms, top_model, batch_x, batch_y):
        """Online reservoir insert of the current mini-batch with current-model
        logits — the per-step equivalent of Mammoth's
            buffer.add_data(examples=not_aug_inputs, labels=labels, logits=outputs.data)
        The logit forward runs in eval() so this extra pass does not perturb the
        BatchNorm running stats already updated by the main training forward; the
        models are restored to train() before returning (the hook is only ever
        called from train_epoch, where everything is in train mode)."""
        for b in bottoms:
            b.eval()
        top_model.eval()
        logits = self._forward(bottoms, top_model, batch_x)
        for b in bottoms:
            b.train()
        top_model.train()

        bx = batch_x.detach().cpu()
        by = batch_y.detach().cpu()
        lg = logits.detach().cpu()
        for i in range(by.size(0)):
            self.buffer.add(bx[i], by[i], lg[i])

    def _der_loss(self, bottoms, top_model, batch_x, batch_y, loss_ce=None):
        device = self.args.device
        total = loss_ce if loss_ce is not None else torch.tensor(0.0, device=device)

        # --- DER++ replay against the buffer, drawn BEFORE inserting the current
        #     batch (matches Mammoth's add-after-loss ordering, so a batch is not
        #     replayed in the same step it is first stored) ---
        if self.buffer.size() > 0:
            # draw #1 — dark-experience MSE against stored logits
            bx, _, blog = self.buffer.get_data(self.mb)
            out = self._forward(bottoms, top_model, bx)
            total = total + self.alpha * F.mse_loss(out, blog.to(device))

            # draw #2 — independent replay batch, CE against stored labels
            bx, blab, _ = self.buffer.get_data(self.mb)
            out = self._forward(bottoms, top_model, bx)
            total = total + self.beta * nn.CrossEntropyLoss()(out, blab.to(device))

        # --- ONLINE reservoir update: insert the CURRENT mini-batch so the new
        #     task becomes replayable during its OWN training (fixes one-task lag) ---
        self._insert_current_batch(bottoms, top_model, batch_x, batch_y)
        return total

    def train_task(self, train_loader, task_id):
        # Always hook the per-batch loss so the reservoir is updated ONLINE every
        # step — including task 0 (empty buffer) where the hook only inserts.
        return self.trainer.train_task(train_loader, self.args.epochs_per_task,
                                       extra_loss_fn=self._der_loss)

    def after_task(self, train_loader, task_id):
        """No-op: the reservoir is updated ONLINE during train_task (Mammoth-style).
        Kept for the harness lifecycle; reports the final buffer size."""
        print(f"  DER++ buffer size after task {task_id}: {self.buffer.size()}/{self.buffer_size}")

    def get_state(self):
        return {
            'buffer_size': self.buffer.buffer_size,
            'num_seen': self.buffer.num_seen,
            'examples': torch.stack(self.buffer.ex) if self.buffer.ex else None,
            'labels': torch.tensor(self.buffer.lb, dtype=torch.long),
            'logits': torch.stack(self.buffer.lg) if self.buffer.lg else None,
        }

    def load_state(self, s):
        if int(s.get('buffer_size', self.buffer.buffer_size)) != self.buffer.buffer_size:
            raise ValueError('DER++ checkpoint uses a different buffer size')
        examples = s.get('examples')
        logits = s.get('logits')
        labels = s.get('labels', torch.empty(0, dtype=torch.long))
        self.buffer.ex = [] if examples is None else [
            value.clone() for value in examples.unbind(0)
        ]
        self.buffer.lg = [] if logits is None else [
            value.clone() for value in logits.unbind(0)
        ]
        self.buffer.lb = [int(value) for value in labels.tolist()]
        self.buffer.num_seen = int(s.get('num_seen', len(self.buffer.ex)))
        if not (len(self.buffer.ex) == len(self.buffer.lb) == len(self.buffer.lg)):
            raise ValueError('DER++ checkpoint buffer fields have different lengths')
