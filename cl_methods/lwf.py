"""LwF (Learning without Forgetting) adapted to VFL.
Li & Hoiem (TPAMI 2018). Knowledge distillation from old model on new data.
Distillation only at server level (top model), no label needed at passive parties.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from data_utils import split_features


class LwFCL:
    """LwF for VFL: KD loss between old and new top model on current batch."""
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'LwF_VFL'
        self.old_bottoms = None
        self.old_top = None
        # Wire KD hyper-params to args (were hardcoded; args.lwf_alpha was ignored).
        self.temperature = getattr(args, 'lwf_temperature', 2.0)
        self.alpha = getattr(args, 'lwf_alpha', 0.5)  # legacy convex weight (unused when additive)
        # PyCIL/FACIL-style additive KD weight: loss = CE_new + lambda*KD_old. The old
        # convex (1-alpha)*CE + alpha*KD down-weighted CE and capped KD at parity; canonical
        # LwF keeps CE at full weight and sets KD >= CE (PyCIL lamda=3, FACIL/Avalanche=1).
        self.lwf_lambda = getattr(args, 'lwf_lambda', 1.0)
        # Canonical class-IL CE scope: CE only on the NEW-class logit slice (so old-class
        # columns are never pushed down by the CE on new-task batches). See before_task.
        self.ce_newonly = getattr(args, 'lwf_ce_newonly', True)
        # Feature-level distillation (summed-L2): anchor the current aggregated
        # embedding to the frozen teacher's embedding on current data. Logit-KD
        # alone pins only the *composed* function on current inputs; the bottoms
        # still rotate the feature space, so old-class inputs land in the new-class
        # region (argmax -> newest task) => class-IL old-task acc collapses to ~0.
        # Anchoring the embedding itself keeps old-class features in place. Summed
        # (not meaned) over the feature dims so a weight ~1 actually bites (matches
        # proto_evolve's proven fix). 0 = off (vanilla logit-KD-only LwF).
        self.feat_distill_weight = getattr(args, 'feat_distill_weight', 0.0)
        # Number of logit columns occupied by OLD classes (set in before_task).
        self.n_old_classes = 0

    def before_task(self, task_id, new_classes, seen_classes):
        req = max(seen_classes) + 1 if seen_classes else 0
        self.trainer.top_model.expand_classes(req, self.args.device)
        # Logit columns belonging to classes seen BEFORE this task. The head is a
        # fixed full-width Linear(dim, num_classes), so old_logits.size(1) is the
        # full width (== num_classes) and CANNOT be used to scope the KD. The
        # teacher has never seen `new_classes`, so its logits on those columns are
        # meaningless; distilling them fights the CE that is trying to raise the
        # new-class logits on new data. Restrict KD to these old columns only.
        old_classes = [c for c in seen_classes if c not in new_classes]
        self.n_old_classes = (max(old_classes) + 1) if old_classes else 0
        # Scope the trainer's CE to the NEW-class columns [n_old:req] with relabeled
        # targets (canonical PyCIL/FACIL class-IL LwF). Without this, the full-head CE
        # on a new-task batch pushes every OLD-class logit down each step -> old-task
        # acc collapses to exactly 0. Old columns are then shaped only by the KD term.
        if self.ce_newonly:
            self.trainer.ce_lo = self.n_old_classes
            self.trainer.ce_hi = req
        if task_id > 0:
            self.old_bottoms = [deepcopy(b).eval() for b in self.trainer.bottoms]
            self.old_top = deepcopy(self.trainer.top_model).eval()
            for ob in self.old_bottoms:
                for p in ob.parameters():
                    p.requires_grad = False
            for p in self.old_top.parameters():
                p.requires_grad = False

    def _lwf_loss(self, bottoms, top_model, batch_x, batch_y, loss_ce=None):
        """KD loss: new model should match old model's output on current data."""
        device = self.args.device
        if self.old_top is None:
            return loss_ce if loss_ce is not None else torch.tensor(0.0, device=device)

        parts = split_features(batch_x, self.args)

        # Old model forward (no grad)
        with torch.no_grad():
            old_embs = [self.old_bottoms[i](parts[i]) for i in range(len(self.old_bottoms))]
            old_agg = self.trainer._aggregate(old_embs)
            old_logits = self.old_top(old_agg)

        # Current model forward
        curr_embs = [bottoms[i](parts[i]) for i in range(len(bottoms))]
        curr_agg = self.trainer._aggregate(curr_embs)
        curr_logits = top_model(curr_agg)

        # KD loss on OLD classes only. Take the softmax over the old-class columns
        # of BOTH teacher and student so the distilled distribution never involves
        # the new-class logits (which the CE alone should drive on new data).
        T = self.temperature
        n_old = self.n_old_classes
        if n_old > 0 and curr_logits.size(1) >= n_old:
            old_probs = F.softmax(old_logits[:, :n_old] / T, dim=1)
            curr_log_probs = F.log_softmax(curr_logits[:, :n_old] / T, dim=1)
            kd_loss = F.kl_div(curr_log_probs, old_probs, reduction='batchmean') * (T * T)
        else:
            kd_loss = torch.tensor(0.0, device=device)

        # Feature-KD: summed-L2 anchor of the current aggregated embedding to the
        # frozen teacher's embedding on the SAME (current) inputs. old_agg is under
        # no_grad (teacher fixed); curr_agg carries grad to the bottoms, so this
        # term directly penalises feature-space drift. Exemplar-free.
        if self.feat_distill_weight > 0:
            feat_loss = ((curr_agg - old_agg) ** 2).sum(dim=1).mean()
        else:
            feat_loss = torch.tensor(0.0, device=device)

        # Additive PyCIL/FACIL-style combination: CE_new + lambda*KD_old (+ optional feat-KD).
        # loss_ce here is the NEW-class-slice CE (scoped via trainer.ce_lo/ce_hi in before_task),
        # so it no longer suppresses old-class columns; KD is their only shaping signal.
        if loss_ce is not None:
            return loss_ce + self.lwf_lambda * kd_loss + self.feat_distill_weight * feat_loss
        return self.lwf_lambda * kd_loss + self.feat_distill_weight * feat_loss

    def train_task(self, train_loader, task_id):
        extra = self._lwf_loss if task_id > 0 else None
        return self.trainer.train_task(train_loader, self.args.epochs_per_task, extra_loss_fn=extra)

    def after_task(self, train_loader, task_id):
        pass

    def get_state(self):
        return {}
    def load_state(self, s):
        pass
