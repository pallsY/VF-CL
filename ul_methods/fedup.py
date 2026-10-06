"""FedUP (Huang et al., TSC 2025) adapted to VFL for class-level unlearning.

The original FedUP is client-level (forget one client's contribution).
Adapted to class-level (consistent with FUCRT/MoDe/FedOSD adaptations
in this repo) by treating the forget-class prototype as the target to
push embeddings AWAY from, then running prototype-guided recovery on
retain data.

Two phases
----------
Phase 1 — gradient-ascent unlearning:
    L_U = − CE(top(agg), y_forget)  − μ · MSE(agg, forget_proto[y])

    Both terms are MAXIMISED (ascent): logits move away from true class
    AND embedding moves away from the forget centroid.  Norm-clipped to
    prevent gradient explosion.

Phase 2 — prototype-guided recovery on retain:
    L_R = CE(top(agg), y_retain)  +  λ · MSE(agg, retain_proto[y])

    Standard cross-entropy + a pull toward the per-class retain centroid
    so embedding geometry is preserved.  Retain prototypes are recomputed
    once per recovery epoch (matches Algo. 1's per-round prototype
    exchange in the original paper).
"""
from __future__ import annotations
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import split_features


class FedUPUL:
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'FedUP'

    def _compute_prototypes(self, loader, class_ids):
        emb, lab = self.trainer.compute_embeddings(loader)
        protos = {}
        for c in class_ids:
            mask = (lab == c)
            if mask.any():
                protos[int(c)] = emb[mask].mean(dim=0).to(self.args.device)
        return protos

    def _gather_target(self, labels, proto_dict, fallback):
        """Per-sample target tensor (B, D) — proto[label] if available, else fallback."""
        rows = [proto_dict[int(c.item())] if int(c.item()) in proto_dict else fallback
                for c in labels]
        return torch.stack(rows)

    def unlearn(self, forget_classes, retain_train_loader, forget_train_loader, **kw):
        device = self.args.device
        eff = kw.get('effective_classes', [])
        forget_classes = list(forget_classes)
        retain_cls = [c for c in eff if c not in forget_classes]
        mu = float(getattr(self.args, 'fedup_mu', 1.0))
        lam = float(getattr(self.args, 'fedup_lambda', 1.0))
        u_eps_default = max(1, self.args.ul_epochs // 2)
        u_raw = int(getattr(self.args, 'fedup_unlearn_epochs', u_eps_default))
        unlearn_epochs = u_raw if u_raw > 0 else u_eps_default
        r_raw = int(getattr(self.args, 'fedup_recovery_epochs',
                             self.args.ul_epochs - unlearn_epochs))
        recovery_epochs = r_raw if r_raw > 0 else max(0, self.args.ul_epochs - unlearn_epochs)

        start = time.time()
        history = []

        # ---- Phase 1: forget prototypes + GA on forget data ----
        forget_protos = self._compute_prototypes(forget_train_loader, forget_classes)
        if not forget_protos:
            return {'history': history, 'time': time.time() - start,
                    'method': 'fedup', 'skipped': 'no_forget_prototypes'}

        for b in self.trainer.bottoms: b.train()
        self.trainer.top_model.train()
        opts_b, opt_t = self.trainer._create_optimizers(lr=self.args.ul_lr)
        any_bottom_trainable = any(p.requires_grad for b in self.trainer.bottoms for p in b.parameters())
        zero_emb = torch.zeros_like(next(iter(forget_protos.values())))

        for ep in range(unlearn_epochs):
            ep_loss, nb = 0.0, 0
            for bx, by in forget_train_loader:
                bx, by = bx.to(device), by.to(device)
                parts = split_features(bx, self.args)
                embs = [self.trainer.bottoms[i](parts[i]) for i in range(len(self.trainer.bottoms))]
                agg = self.trainer._aggregate(embs)
                logits = self.trainer.top_model(agg)

                loss_ce = -F.cross_entropy(logits, by)
                tgt = self._gather_target(by, forget_protos, zero_emb)
                loss_proto = -F.mse_loss(agg, tgt)
                loss = loss_ce + mu * loss_proto

                for o in opts_b: o.zero_grad()
                opt_t.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for b in self.trainer.bottoms for p in b.parameters() if p.requires_grad] +
                    list(self.trainer.top_model.parameters()), 1.0)
                opt_t.step()
                if any_bottom_trainable:
                    for o in opts_b: o.step()
                ep_loss += float(loss.item()); nb += 1
            history.append({'phase': 'unlearn', 'epoch': ep, 'loss': ep_loss / max(nb, 1)})

        # ---- Phase 2: prototype-guided recovery on retain ----
        for ep in range(recovery_epochs):
            if not retain_cls:
                break
            retain_protos = self._compute_prototypes(retain_train_loader, retain_cls)
            ep_loss, nb = 0.0, 0
            for bx, by in retain_train_loader:
                bx, by = bx.to(device), by.to(device)
                parts = split_features(bx, self.args)
                embs = [self.trainer.bottoms[i](parts[i]) for i in range(len(self.trainer.bottoms))]
                agg = self.trainer._aggregate(embs)
                logits = self.trainer.top_model(agg)

                loss_ce = F.cross_entropy(logits, by)
                tgt = self._gather_target(by, retain_protos, agg.detach().mean(0))
                loss_proto = F.mse_loss(agg, tgt)
                loss = loss_ce + lam * loss_proto

                for o in opts_b: o.zero_grad()
                opt_t.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for b in self.trainer.bottoms for p in b.parameters() if p.requires_grad] +
                    list(self.trainer.top_model.parameters()), 5.0)
                opt_t.step()
                if any_bottom_trainable:
                    for o in opts_b: o.step()
                ep_loss += float(loss.item()); nb += 1
            history.append({'phase': 'recovery', 'epoch': ep, 'loss': ep_loss / max(nb, 1)})

        return {'history': history, 'time': time.time() - start, 'method': 'fedup',
                'config': {'mu': mu, 'lambda': lam,
                           'unlearn_epochs': unlearn_epochs,
                           'recovery_epochs': recovery_epochs}}
