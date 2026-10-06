"""LwF+FIM: dual protection for VFL CIL.

Combines two protection mechanisms motivated by the empirical
"top-only vs bottom-only" split observed across all baselines:

  - LwF logit KD: anchors top-model outputs on aggregated embeddings
    to the frozen old model. Protects against logit drift.
  - FIM-based hard freeze of important bottom params: keeps the
    embeddings stable so the anchored logit pattern remains
    semantically valid for old classes.

Plain LwF (logit KD only) reached AA=0.481 in our VFL-CIL setup. The
gap to V-LETO=0.484 / TARGET+FIM=0.546 wasn't the KD itself but the
bottom drift undoing the embedding semantics. LwF+FIM closes that
gap by borrowing V-LETO's FIM freeze without using prototype evolve.

FIM computation logic and mask accumulation reuse the V-LETO
implementation in cl_methods/proto_evolve.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from data_utils import split_features


class LwFFIMCL:
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'LwF_FIM_VFL'
        # LwF distillation
        self.old_bottoms = None
        self.old_top = None
        self.temperature = getattr(args, 'lwf_temperature', 2.0)
        self.alpha = getattr(args, 'lwf_alpha', 0.5)  # legacy convex weight (unused when additive)
        # --- Ported from the working lwf.py fix (the convex (1-alpha)CE+alpha*KD
        # left old-class columns collapsed; genuine lwf_fim AA_cil was ~0.50).
        # Canonical class-IL recipe: full-weight CE on the NEW-class slice only
        # (trainer.ce_lo/ce_hi, set in before_task) + additive lambda*KD on OLD
        # columns. The FIM freeze (below) then keeps the old-class embeddings
        # stable so the KD-anchored old logits stay semantically valid. ---
        self.lwf_lambda = getattr(args, 'lwf_lambda', 1.0)
        self.ce_newonly = getattr(args, 'lwf_ce_newonly', True)
        self.feat_distill_weight = getattr(args, 'feat_distill_weight', 0.0)
        # Number of logit columns occupied by OLD classes (set in before_task).
        self.n_old_classes = 0
        # FIM freeze (V-LETO style): quantile threshold over per-tensor mean
        # importances, controlled by args.fim_freeze_frac (default 0.25).
        self.fim_masks = [{} for _ in range(args.num_parties)]

    def before_task(self, task_id, new_classes, seen_classes):
        # Expand top
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

        # Scope the trainer's CE to the NEW-class columns [n_old:req] (canonical
        # class-IL LwF). Without this the full-head CE pushed every OLD-class logit
        # down each step -> old-task acc collapsed to 0 even WITH the FIM freeze.
        if self.ce_newonly:
            self.trainer.ce_lo = self.n_old_classes
            self.trainer.ce_hi = req

        # Snapshot old model (used for LwF KD on this task's training)
        if task_id > 0:
            self.old_bottoms = [deepcopy(b).eval() for b in self.trainer.bottoms]
            self.old_top = deepcopy(self.trainer.top_model).eval()
            for ob in self.old_bottoms:
                for p in ob.parameters():
                    p.requires_grad = False
            for p in self.old_top.parameters():
                p.requires_grad = False

        # Apply FIM freeze (mask accumulated from prior after_tasks)
        total_f, total_p = 0, 0
        for k in range(self.args.num_parties):
            for n, p in self.trainer.bottoms[k].named_parameters():
                total_p += 1
                if self.fim_masks[k].get(n, False):
                    p.requires_grad = False
                    total_f += 1
        if total_f > 0:
            print(f"  LwF+FIM freeze: {total_f}/{total_p} bottom params frozen")

    def _lwf_loss(self, bottoms, top_model, batch_x, batch_y, loss_ce=None):
        device = self.args.device
        if self.old_top is None:
            return loss_ce if loss_ce is not None else torch.tensor(0.0, device=device)

        parts = split_features(batch_x, self.args)
        with torch.no_grad():
            old_embs = [self.old_bottoms[i](parts[i]) for i in range(len(self.old_bottoms))]
            old_agg = self.trainer._aggregate(old_embs)
            old_logits = self.old_top(old_agg)
        curr_embs = [bottoms[i](parts[i]) for i in range(len(bottoms))]
        curr_agg = self.trainer._aggregate(curr_embs)
        curr_logits = top_model(curr_agg)

        # KD on OLD classes only. Take the softmax over the old-class columns of
        # BOTH teacher and student so the distilled distribution never involves
        # the new-class logits (which the CE alone should drive on new data).
        # Mirrors the fix in cl_methods/lwf.py and cl_methods/lwf_wa.py.
        T = self.temperature
        n_old = self.n_old_classes
        if n_old > 0 and curr_logits.size(1) >= n_old:
            old_probs = F.softmax(old_logits[:, :n_old] / T, dim=1)
            curr_log_probs = F.log_softmax(curr_logits[:, :n_old] / T, dim=1)
            kd_loss = F.kl_div(curr_log_probs, old_probs, reduction='batchmean') * (T * T)
        else:
            kd_loss = torch.tensor(0.0, device=device)

        # Optional summed-L2 feature-KD (off by default: weight 1 over-pins the
        # bottoms here and kills new-task plasticity; the FIM freeze already
        # provides the embedding-stability that feat-KD would).
        if self.feat_distill_weight > 0:
            feat_loss = ((curr_agg - old_agg) ** 2).sum(dim=1).mean()
        else:
            feat_loss = torch.tensor(0.0, device=device)

        # Additive PyCIL/FACIL combination: CE_new (scoped via trainer.ce_lo/ce_hi)
        # + lambda*KD_old (+ optional feat-KD). Old columns are shaped only by KD,
        # so CE on new batches no longer suppresses them.
        if loss_ce is not None:
            return loss_ce + self.lwf_lambda * kd_loss + self.feat_distill_weight * feat_loss
        return self.lwf_lambda * kd_loss + self.feat_distill_weight * feat_loss

    def train_task(self, train_loader, task_id):
        extra = self._lwf_loss if task_id > 0 else None
        history, elapsed = self.trainer.train_task(
            train_loader, self.args.epochs_per_task, extra_loss_fn=extra
        )
        # Unfreeze so FIM computation in after_task can see gradients
        for b in self.trainer.bottoms:
            for p in b.parameters():
                p.requires_grad = True
        return history, elapsed

    def after_task(self, train_loader, task_id):
        """Compute FIM and accumulate freeze masks for next task."""
        for b in self.trainer.bottoms:
            b.train()
        self.trainer.top_model.train()

        fim = [{n: torch.zeros_like(p) for n, p in self.trainer.bottoms[k].named_parameters()}
               for k in range(self.args.num_parties)]
        nb = 0
        for bx, by in train_loader:
            bx, by = bx.to(self.args.device), by.to(self.args.device)
            for b in self.trainer.bottoms:
                b.zero_grad()
            self.trainer.top_model.zero_grad()
            parts = split_features(bx, self.args)
            embs = [self.trainer.bottoms[i](parts[i]) for i in range(self.args.num_parties)]
            out = self.trainer.top_model(self.trainer._aggregate(embs))
            nn.CrossEntropyLoss()(out, by).backward()
            for k in range(self.args.num_parties):
                for n, p in self.trainer.bottoms[k].named_parameters():
                    if p.grad is not None:
                        fim[k][n] += p.grad.data ** 2
            nb += 1

        frac = getattr(self.args, 'fim_freeze_frac', 0.25)
        for k in range(self.args.num_parties):
            for n in fim[k]:
                fim[k][n] /= max(nb, 1)
            # Per-tensor importance = mean FIM of that parameter tensor (one
            # value per tensor), so the quantile is taken over tensors.
            imps = {n: fim[k][n].mean().item() for n in fim[k]}
            vals = torch.tensor(list(imps.values()))
            if vals.numel() == 0 or vals.sum() == 0:
                # Don't wipe accumulated mask just because this task gave no signal
                continue
            # Freeze the top `frac` most-important tensors (quantile threshold).
            # BUG FIX: the old threshold kappa = mean - (k0 + alpha*log(t+2))*std
            # with k0=15 went hugely negative, so EVERY non-negative FIM tensor
            # passed -> 100% of bottom params frozen -> backbone never adapts.
            # This mirrors the fix already in proto_evolve.py::_compute_fim.
            kappa = torch.quantile(vals, max(0.0, 1.0 - frac)).item()
            # Accumulate mask across tasks: once a tensor is marked important, keep it frozen
            nf = 0
            n_new_freeze = 0
            for n in fim[k]:
                is_important_now = imps[n] >= kappa
                was_important_before = self.fim_masks[k].get(n, False)
                new_state = was_important_before or is_important_now
                if new_state and not was_important_before:
                    n_new_freeze += 1
                self.fim_masks[k][n] = new_state
                if new_state:
                    nf += 1
            print(f"    Party {k}: {nf}/{len(fim[k])} frozen (+{n_new_freeze} new this task)")

        for b in self.trainer.bottoms:
            b.zero_grad()
            for p in b.parameters():
                p.requires_grad = True
        self.trainer.top_model.zero_grad()

    def get_state(self):
        return {'fim_masks': deepcopy(self.fim_masks)}

    def load_state(self, s):
        self.fim_masks = s.get('fim_masks', [{} for _ in range(self.args.num_parties)])
