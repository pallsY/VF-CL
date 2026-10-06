"""FedAU (Pan et al., 2022) adapted to VFL for class-level unlearning.

Original FedAU maintains an auxiliary classifier head ``classifier_ul``
trained in parallel during federated learning; at unlearn time the
server linearly modifies the main classifier::

    ul_class:   W ← W - W_ul         b ← b - b_ul
    ul_samples: W ← α·W + (1-α)·W_ul (α default 0.9)

The "unlearning during learning" requirement is incompatible with our
existing VFL CL pipeline (the CL methods are not aware of the aux
head).  We instead reproduce the linear-subtraction spirit POST-HOC:

  1. Build a fresh aux head ``W_ul`` initialised to zero (so a no-op
     subtraction is the trivial baseline).
  2. Freeze the bottom models; train ``W_ul`` for ``aux_epochs`` epochs
     on FORGET-class samples with CE loss against the joint embedding
     of the current trainer.  After training, ``W_ul`` is essentially
     "the linear layer that maps current features to forget-class
     predictions".
  3. Apply the FedAU linear unlearn op on the main top-model classifier.
  4. Short retain recovery using the existing trainer optimizers
     (CE on retain), so the rest of the model can absorb the shift.

This adaptation purposefully foregoes FedAU's "communication-free"
selling point (the original needs no further gradient steps at unlearn
time) — we pay one short aux-training pass at unlearn time.  The
linear-subtraction semantics that distinguish FedAU from GA/LUV are
preserved.
"""
from __future__ import annotations
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import split_features


class FedAUUL:
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'FedAU'

    def unlearn(self, forget_classes, retain_train_loader, forget_train_loader, **kw):
        device = self.args.device
        mode = str(getattr(self.args, 'fedau_mode', 'ul_class'))
        alpha = float(getattr(self.args, 'fedau_alpha', 0.9))
        aux_lr = float(getattr(self.args, 'fedau_aux_lr', 1e-2))
        aux_epochs = int(getattr(self.args, 'fedau_aux_epochs', 3))
        recovery_epochs = int(getattr(self.args, 'fedau_recovery_epochs', 2))

        start = time.time()
        history = []

        # ---- Build fresh aux head, init to zero ----
        in_dim = self.trainer.top_model.classifier.in_features
        out_dim = self.trainer.top_model.classifier.out_features
        aux_head = nn.Linear(in_dim, out_dim).to(device)
        with torch.no_grad():
            nn.init.zeros_(aux_head.weight)
            nn.init.zeros_(aux_head.bias)

        ce = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(aux_head.parameters(), lr=aux_lr,
                              momentum=self.args.momentum,
                              weight_decay=self.args.weight_decay)

        # ---- Phase 1: train aux head on forget data ----
        # Bottoms + main top frozen; only aux_head learns.
        for b in self.trainer.bottoms: b.eval()
        self.trainer.top_model.eval()
        aux_head.train()

        for ep in range(aux_epochs):
            ep_loss, nb = 0.0, 0
            for bx, by in forget_train_loader:
                bx, by = bx.to(device), by.to(device)
                with torch.no_grad():
                    parts = split_features(bx, self.args)
                    embs = [self.trainer.bottoms[i](parts[i]) for i in range(len(self.trainer.bottoms))]
                    agg = self.trainer._aggregate(embs)
                opt.zero_grad()
                loss = ce(aux_head(agg), by)
                loss.backward()
                opt.step()
                ep_loss += float(loss.item()); nb += 1
            history.append({'phase': 'aux_train', 'epoch': ep,
                            'loss': ep_loss / max(nb, 1),
                            'aux_norm': float(aux_head.weight.norm().item())})

        # ---- Phase 2: linear op on main classifier ----
        with torch.no_grad():
            W = self.trainer.top_model.classifier.weight.data
            b = self.trainer.top_model.classifier.bias.data
            W_ul = aux_head.weight.data
            b_ul = aux_head.bias.data
            pre_norm = float(W.norm().item())
            if mode == 'ul_class':
                self.trainer.top_model.classifier.weight.data = W - W_ul
                self.trainer.top_model.classifier.bias.data = b - b_ul
            elif mode == 'ul_samples':
                self.trainer.top_model.classifier.weight.data = alpha * W + (1.0 - alpha) * W_ul
                self.trainer.top_model.classifier.bias.data = alpha * b + (1.0 - alpha) * b_ul
            else:
                raise ValueError(f"unknown fedau_mode {mode!r}, use 'ul_class' or 'ul_samples'")
            post_norm = float(self.trainer.top_model.classifier.weight.data.norm().item())
        history.append({'phase': 'linear_op', 'mode': mode, 'alpha': alpha,
                        'pre_norm_W': pre_norm, 'post_norm_W': post_norm,
                        'delta_norm': post_norm - pre_norm,
                        'aux_norm': float(W_ul.norm().item())})

        # ---- Phase 3: short retain recovery ----
        if recovery_epochs > 0:
            for b_m in self.trainer.bottoms: b_m.train()
            self.trainer.top_model.train()
            opts_b, opt_t = self.trainer._create_optimizers(lr=self.args.ul_lr)
            any_bot = any(p.requires_grad for b_m in self.trainer.bottoms for p in b_m.parameters())
            for ep in range(recovery_epochs):
                ep_loss, nb = 0.0, 0
                for bx, by in retain_train_loader:
                    bx, by = bx.to(device), by.to(device)
                    parts = split_features(bx, self.args)
                    embs = [self.trainer.bottoms[i](parts[i]) for i in range(len(self.trainer.bottoms))]
                    agg = self.trainer._aggregate(embs)
                    loss = ce(self.trainer.top_model(agg), by)
                    for o in opts_b: o.zero_grad()
                    opt_t.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        [p for b_m in self.trainer.bottoms for p in b_m.parameters() if p.requires_grad] +
                        list(self.trainer.top_model.parameters()), 5.0)
                    opt_t.step()
                    if any_bot:
                        for o in opts_b: o.step()
                    ep_loss += float(loss.item()); nb += 1
                history.append({'phase': 'recovery', 'epoch': ep,
                                'loss': ep_loss / max(nb, 1)})

        return {'history': history, 'time': time.time() - start, 'method': 'fedau',
                'config': {'mode': mode, 'alpha': alpha,
                           'aux_lr': aux_lr, 'aux_epochs': aux_epochs,
                           'recovery_epochs': recovery_epochs}}
