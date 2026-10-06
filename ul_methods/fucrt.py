"""FUCRT (Guo et al., ICCV 2025) adapted to VFL for class-level unlearning.

Class-Aware Representation Transformation: push the JOINT VFL embedding
of forget-class samples toward the mean of retain-class centroids,
while keeping retain-class CE for utility.

In VFL there is a single joint embedding space (sum/concat of bottom
outputs), so the per-client centroid aggregation in the HFL paper
collapses to a plain class-wise mean over the joint embeddings.

Loss
----
For each minibatch we draw `f` forget-class samples and `r` retain-class
samples (one batch from each loader):

    L = λ_t · MSE(joint_emb(f), target_h)            # transform
      + λ_l · KL( uniform_over_retain || softmax(top(f)) )  # logit alignment
      + λ_r · CE(top(r), y_r)                         # utility

`target_h` is the mean of available retain-class centroids — computed
once on the pretrained model before phase 2 (matches paper Algo. line 1).
"""
from __future__ import annotations
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import split_features


class FUCRTUL:
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'FUCRT'

    def _compute_centroids(self, loader, class_ids):
        """Class -> Tensor(embed_dim,)."""
        emb, lab = self.trainer.compute_embeddings(loader)
        out = {}
        for c in class_ids:
            mask = (lab == c)
            if mask.any():
                out[int(c)] = emb[mask].mean(dim=0).to(self.args.device)
        return out

    def unlearn(self, forget_classes, retain_train_loader, forget_train_loader, **kw):
        device = self.args.device
        eff = kw.get('effective_classes', [])
        forget_classes = list(forget_classes)
        retain_cls = [c for c in eff if c not in forget_classes]
        lambda_t = float(getattr(self.args, 'fucrt_lambda_t', 1.0))
        lambda_l = float(getattr(self.args, 'fucrt_lambda_l', 0.5))
        lambda_r = float(getattr(self.args, 'fucrt_lambda_r', 1.0))

        start = time.time()
        history = []

        # Phase 1: centroid discovery on retain classes
        retain_centroids = self._compute_centroids(retain_train_loader, retain_cls)
        if not retain_centroids:
            return {'history': history, 'time': time.time() - start,
                    'method': 'fucrt', 'skipped': 'no_retain_centroids'}
        target_h = torch.stack(list(retain_centroids.values())).mean(dim=0)  # (D,)

        # Build uniform-over-retain target distribution for the logit-alignment term
        n_logits = self.trainer.top_model.classifier.out_features
        uniform = torch.zeros(n_logits, device=device)
        for c in retain_cls:
            if 0 <= c < n_logits:
                uniform[c] = 1.0
        if uniform.sum().item() > 0:
            uniform = uniform / uniform.sum()
        else:
            uniform = None

        # Phase 2: local representation transformation
        for b in self.trainer.bottoms: b.train()
        self.trainer.top_model.train()
        opts_b, opt_t = self.trainer._create_optimizers(lr=self.args.ul_lr)
        any_bottom_trainable = any(p.requires_grad for b in self.trainer.bottoms for p in b.parameters())

        for ep in range(self.args.ul_epochs):
            ep_lt, ep_ll, ep_lr, nb = 0.0, 0.0, 0.0, 0
            # zip stops at shorter loader — fine, we sample from both each step
            for (f_bx, _f_by), (r_bx, r_by) in zip(forget_train_loader, retain_train_loader):
                # ---- forget term ----
                f_bx = f_bx.to(device)
                f_parts = split_features(f_bx, self.args)
                f_embs = [self.trainer.bottoms[i](f_parts[i]) for i in range(len(self.trainer.bottoms))]
                f_agg = self.trainer._aggregate(f_embs)
                f_logits = self.trainer.top_model(f_agg)

                loss_t = F.mse_loss(f_agg, target_h.unsqueeze(0).expand_as(f_agg))
                if uniform is not None:
                    log_p = F.log_softmax(f_logits, dim=-1)
                    loss_l = -(uniform.unsqueeze(0) * log_p).sum(dim=-1).mean()
                else:
                    loss_l = torch.tensor(0.0, device=device)

                # ---- retain term ----
                r_bx, r_by = r_bx.to(device), r_by.to(device)
                r_parts = split_features(r_bx, self.args)
                r_embs = [self.trainer.bottoms[i](r_parts[i]) for i in range(len(self.trainer.bottoms))]
                r_agg = self.trainer._aggregate(r_embs)
                loss_r = F.cross_entropy(self.trainer.top_model(r_agg), r_by)

                loss = lambda_t * loss_t + lambda_l * loss_l + lambda_r * loss_r

                for o in opts_b: o.zero_grad()
                opt_t.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for b in self.trainer.bottoms for p in b.parameters() if p.requires_grad] +
                    list(self.trainer.top_model.parameters()), 5.0)
                opt_t.step()
                if any_bottom_trainable:
                    for o in opts_b: o.step()

                ep_lt += float(loss_t.item()); ep_ll += float(loss_l.item())
                ep_lr += float(loss_r.item()); nb += 1
            history.append({'epoch': ep,
                            'loss_t': ep_lt / max(nb, 1),
                            'loss_l': ep_ll / max(nb, 1),
                            'loss_r': ep_lr / max(nb, 1)})

        return {'history': history, 'time': time.time() - start, 'method': 'fucrt',
                'config': {'lambda_t': lambda_t, 'lambda_l': lambda_l, 'lambda_r': lambda_r,
                           'n_retain_centroids': len(retain_centroids),
                           'target_norm': float(target_h.norm().item())}}
