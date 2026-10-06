"""LwF + Weight Aligning (WA, CVPR 2020).

Zhao et al. observed that after each task, the classifier rows for
newly learned classes have systematically larger L2 norm than rows
for old classes. At inference this produces a "new-class bias":
||W_new|| >> ||W_old|| -> softmax(W x) leans toward new classes
regardless of the input.

Weight Aligning fixes this post-hoc, with no extra training:
  ratio = mean(||W_old||) / mean(||W_new||)
  W_new <- ratio * W_new

This is a pure top-side trick — no exemplars, no extra forward, no
extra modules. It complements LwF's logit KD (which protects logit
*pattern* during training) by re-balancing the classifier *magnitude*
at the boundary of each task.

We attach WA as an after_task hook to the existing LwF baseline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from data_utils import split_features


class LwFWACL:
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'LwF_WA_VFL'
        # LwF distillation
        self.old_bottoms = None
        self.old_top = None
        self.temperature = getattr(args, 'lwf_temperature', 2.0)
        self.alpha = getattr(args, 'lwf_alpha', 0.5)  # legacy convex weight (unused when additive)
        # --- Ported from the working lwf.py fix (the convex (1-alpha)CE+alpha*KD
        # combination left old-class columns collapsed at exactly 0; AA_cil ~0.47).
        # Canonical class-IL recipe: full-weight CE on the NEW-class slice only
        # (trainer.ce_lo/ce_hi, set in before_task) + additive lambda*KD on the OLD
        # columns + optional summed-L2 feature-KD that pins the embedding space. ---
        self.lwf_lambda = getattr(args, 'lwf_lambda', 1.0)
        self.ce_newonly = getattr(args, 'lwf_ce_newonly', True)
        self.feat_distill_weight = getattr(args, 'feat_distill_weight', 0.0)
        # Number of classes seen before this task (i.e. "old" before WA)
        self._n_old_classes = 0

    def before_task(self, task_id, new_classes, seen_classes):
        req = max(seen_classes) + 1 if seen_classes else 0
        # Number of old classes = total seen minus this task's new ones
        self._n_old_classes = req - len(new_classes) if new_classes else 0

        self._n_new_classes = len(new_classes) if new_classes else 0
        self.trainer.top_model.expand_classes(req, self.args.device)
        # Scope the trainer's CE to the NEW-class columns [n_old:req] with relabeled
        # targets (canonical PyCIL/FACIL class-IL LwF). Without this, the full-head CE
        # pushed every OLD-class logit down each step -> old-task acc collapsed to 0.
        if self.ce_newonly:
            self.trainer.ce_lo = self._n_old_classes
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

        # KD over OLD-CLASS COLUMNS ONLY. The head is fixed full-width
        # (models.py builds num_classes from the start; expand_classes is a
        # no-op), so old_logits.size(1) == num_classes == 10. The previous
        # `n_old = old_logits.size(1)` therefore distilled the FULL logit
        # vector — including THIS task's new classes — toward a teacher that
        # never saw them. That spurious term fought CE on the new columns and
        # froze new-task plasticity (fresh diagonal acc ~0.02). Restrict KD to
        # the true old-class count tracked in before_task so it only protects
        # old classes and leaves the new columns free to learn.
        T = self.temperature
        n_old = self._n_old_classes
        if n_old > 0 and curr_logits.size(1) >= n_old:
            old_probs = F.softmax(old_logits[:, :n_old] / T, dim=1)
            curr_log_probs = F.log_softmax(curr_logits[:, :n_old] / T, dim=1)
            kd_loss = F.kl_div(curr_log_probs, old_probs, reduction='batchmean') * (T * T)
        else:
            kd_loss = torch.tensor(0.0, device=device)

        # Feature-KD: summed-L2 anchor of the current aggregated embedding to the
        # frozen teacher's embedding on the SAME inputs (mirrors lwf.py /
        # proto_evolve). Summed (not meaned) over feature dims so weight ~1 bites.
        if self.feat_distill_weight > 0:
            feat_loss = ((curr_agg - old_agg) ** 2).sum(dim=1).mean()
        else:
            feat_loss = torch.tensor(0.0, device=device)

        # Additive PyCIL/FACIL combination: CE_new (already scoped to the new-class
        # slice via trainer.ce_lo/ce_hi) + lambda*KD_old (+ optional feat-KD). Old
        # columns are shaped only by KD, so CE on new batches no longer suppresses
        # them. WA (after_task) then re-balances the head norms on a linear head.
        if loss_ce is not None:
            return loss_ce + self.lwf_lambda * kd_loss + self.feat_distill_weight * feat_loss
        return self.lwf_lambda * kd_loss + self.feat_distill_weight * feat_loss

    def train_task(self, train_loader, task_id):
        extra = self._lwf_loss if task_id > 0 else None
        return self.trainer.train_task(
            train_loader, self.args.epochs_per_task, extra_loss_fn=extra
        )

    def after_task(self, train_loader, task_id):
        """Apply Weight Aligning to the classifier rows.

        DEAD UNDER COSINE HEAD: WA rescales classifier weight-row L2 norms,
        but a cosine head (args.cosine_head, models.py) L2-normalises every
        weight row on every forward, so the rescale is cancelled and WA is a
        mathematical no-op. Skip it there to avoid a misleading "rescaled"
        log line; WA only does real work on a plain linear head.
        """
        if getattr(self.args, 'cosine_head', False):
            return
        if self._n_old_classes <= 0 or task_id == 0:
            return
        with torch.no_grad():
            W = self.trainer.top_model.classifier.weight  # (n_total, embed_dim)
            n_old = self._n_old_classes
            # On the fixed full-width head, rows beyond n_old+n_new are FUTURE
            # untrained classes; WA must use ONLY this task's new-class rows.
            n_new_end = n_old + self._n_new_classes
            old_norm_mean = W[:n_old].norm(p=2, dim=1).mean().item()
            new_norm_mean = W[n_old:n_new_end].norm(p=2, dim=1).mean().item()
            if new_norm_mean > 1e-12 and old_norm_mean > 1e-12:
                ratio = old_norm_mean / new_norm_mean
                W[n_old:n_new_end].mul_(ratio)
                print(f"    WA: rescaled new-class rows by {ratio:.4f} "
                      f"(old_norm_mean={old_norm_mean:.4f}, new_norm_mean={new_norm_mean:.4f})")

    def get_state(self):
        return {}

    def load_state(self, s):
        pass
