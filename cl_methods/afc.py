"""AFC (Adaptive Feature Consolidation) adapted to VFL.

Kang, Park, Han, CVPR 2022. Original AFC has three core ideas:
  1) Feature-level POD distillation across 5 layers
  2) Per-channel "importance" weights derived from squared gradient
     w.r.t. a channel-scale identity layer (empirical Fisher / Gauss-
     Newton diagonal at channel level)
  3) NCA cosine classifier + iCaRL exemplar replay

VFL-adapted minimum-viable port that preserves AFC's distinguishing
contribution (per-channel importance-weighted distillation):

  loss = CE_clf
       + distill_weight * sum_c importance_c * ||f_curr_c - f_old_c||_2

where f is the aggregated embedding (128-d) at the server, importance
is a 128-vector of empirical Fisher computed on the previous task's
training data via an extra backward pass that accumulates
(d loss / d f)^2 per channel.

Dropped from the full paper (clearly stated): the 5-layer per-party
POD (would require modifying SmallCNNBottom), NCA + cosine classifier
(orthogonal to the distillation contribution), and iCaRL exemplars
(incompatible with exemplar-free VFL).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from data_utils import split_features


class AFCCL:
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'AFC_VFL_min'
        self.distill_weight = getattr(args, 'afc_distill_weight', 2.0)
        self.task_size = getattr(args, 'classes_per_task', 2)

        # Snapshots from prior task
        self.old_bottoms = None
        self.old_top = None
        # Per-channel importance over aggregated embedding (1-D, embed_dim)
        self.importance = None  # torch tensor (embed_dim,)
        self._n_seen_classes = 0

    # -------------------------------------------------------------
    def _agg_forward(self, bottoms, batch_x):
        parts = split_features(batch_x, self.args)
        embs = [bottoms[i](parts[i]) for i in range(len(bottoms))]
        return self.trainer._aggregate(embs)

    def before_task(self, task_id, new_classes, seen_classes):
        req = max(seen_classes) + 1 if seen_classes else 0
        self.trainer.top_model.expand_classes(req, self.args.device)
        self._n_seen_classes = max(self._n_seen_classes, req)

        if task_id > 0:
            self.old_bottoms = [deepcopy(b).eval() for b in self.trainer.bottoms]
            for ob in self.old_bottoms:
                for p in ob.parameters():
                    p.requires_grad = False
            self.old_top = deepcopy(self.trainer.top_model).eval()
            for p in self.old_top.parameters():
                p.requires_grad = False

    def _feature_distil(self, new_emb, old_emb, importance):
        """Channel-importance-weighted MSE on aggregated embedding.

        AFC's POD applies pow(.,2) + L2-normalize over spatial + per-
        channel Frobenius. On a 1-D aggregated embedding there is no
        spatial dim, so the natural analogue is plain per-channel MSE
        weighted by importance:
            loss = mean_b sum_c importance_c * (f_c - f_old_c)^2
        Squared-energy distill (matching original POD literally) is
        numerically unstable here because the embedding magnitudes
        are O(1) so pow(.,2) then pow(diff,2) becomes 4th-order and
        the gradient blows up.
        """
        diff_sq = (new_emb - old_emb) ** 2  # (B, C)
        weighted = importance.view(1, -1) * diff_sq  # (B, C)
        return weighted.mean()

    def _afc_loss(self, bottoms, top_model, batch_x, batch_y, loss_ce=None):
        if self.old_bottoms is None or self.importance is None:
            # Task 0: no distillation, just CE
            return loss_ce if loss_ce is not None else torch.tensor(0.0, device=self.args.device)

        # Recompute current aggregated embedding so grad flows back to
        # bottoms via this graph (the trainer's loss_ce already went
        # through agg_detached for the top-model update).
        agg_new = self._agg_forward(bottoms, batch_x)
        with torch.no_grad():
            agg_old = self._agg_forward(self.old_bottoms, batch_x)

        importance = self.importance.to(self.args.device).detach()
        # AFC schedules the distill weight as factor * sqrt(n_seen / task_size)
        scale = self.distill_weight * float(np.sqrt(self._n_seen_classes / max(self.task_size, 1)))
        distill_loss = scale * self._feature_distil(agg_new, agg_old, importance)

        return (loss_ce + distill_loss) if loss_ce is not None else distill_loss

    def train_task(self, train_loader, task_id):
        return self.trainer.train_task(
            train_loader, self.args.epochs_per_task, extra_loss_fn=self._afc_loss
        )

    # -------------------------------------------------------------
    def _estimate_importance(self, train_loader):
        """Empirical Fisher on aggregated embedding's per-channel output.

        Run one epoch (no optimizer step) collecting (d CE / d agg)^2
        per channel. This is the AFC channel-importance hook applied at
        the aggregator output.
        """
        device = self.args.device
        embed_dim = self.trainer.top_model.classifier.in_features
        importance = torch.zeros(embed_dim, device=device)
        nb = 0

        for b in self.trainer.bottoms:
            b.eval()
        self.trainer.top_model.eval()

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            parts = split_features(bx, self.args)
            embs = [self.trainer.bottoms[i](parts[i]) for i in range(len(self.trainer.bottoms))]
            agg = self.trainer._aggregate(embs)
            agg = agg.detach().requires_grad_(True)  # capture grad here
            logits = self.trainer.top_model(agg)
            loss = F.cross_entropy(logits, by)
            grad = torch.autograd.grad(loss, agg, retain_graph=False)[0]
            # Per-channel grad^2, averaged over batch
            importance += (grad ** 2).mean(dim=0).detach()
            nb += 1

        importance = importance / max(nb, 1)
        # Normalize so the scale doesn't explode across tasks
        importance = importance / (importance.mean() + 1e-12)
        return importance.cpu()

    def after_task(self, train_loader, task_id):
        """Compute (or update) per-channel importance over aggregated embedding."""
        new_imp = self._estimate_importance(train_loader)
        if self.importance is None:
            self.importance = new_imp
        else:
            # Accumulate: once a channel was important, keep some weight.
            self.importance = self.importance + new_imp
        print(f"    AFC importance: mean={self.importance.mean():.4f}, "
              f"max={self.importance.max():.4f}, min={self.importance.min():.4f}")

    def get_state(self):
        return {
            'importance': self.importance.clone() if self.importance is not None else None,
            'n_seen': self._n_seen_classes,
        }

    def load_state(self, s):
        self.importance = s.get('importance', None)
        self._n_seen_classes = s.get('n_seen', 0)
