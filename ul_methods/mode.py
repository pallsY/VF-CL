"""MoDe (Zhao et al., IoT 2024) adapted to VFL for class-level unlearning.

The HFL paper iterates between (1) FedAvg-training a degradation model
W_de on retain data, (2) momentum-degrading the global model
  W ← λ·W + (1-λ)·W_de  ,
and (3) one round of memory-guided FedAvg where forget-class samples
use M_de's hard pseudo-labels.

In our VFL setup there's a single node (bottoms[0..N-1] + top_model)
that already holds the joint model, so the per-client FedAvg collapses
to a single optimisation pass per round.  The shape of the algorithm
(warmup → mode_rounds × MoDe+MG → guidance_only_rounds × MG) is kept
exactly as in the paper.

FIM-freeze interaction
----------------------
Step 2 (`W ← λW + (1-λ)W_de`) interpolates EVERY parameter, including
ones that the FIM mask had previously frozen.  This deliberately
overrides FIM protection — that's the whole point of MoDe.  The user
can opt out via ``--mode_respect_fim true``, in which case frozen
params are restored after the interpolation step.
"""
from __future__ import annotations
import copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import split_features


def _reset_params(modules):
    """Random-init weights and reset BN running stats for each module."""
    for module in modules:
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None: nn.init.ones_(m.weight)
                if m.bias is not None:   nn.init.zeros_(m.bias)
                if getattr(m, 'running_mean', None) is not None: m.running_mean.zero_()
                if getattr(m, 'running_var',  None) is not None: m.running_var.fill_(1.0)


def _snapshot_frozen(modules):
    """Return list of (module_idx, param_name, tensor_clone) for every frozen param."""
    snap = []
    for i, m in enumerate(modules):
        for name, p in m.named_parameters():
            if not p.requires_grad:
                snap.append((i, name, p.data.clone()))
    return snap


def _restore_frozen(modules, snap):
    name_to_param = [dict(m.named_parameters()) for m in modules]
    with torch.no_grad():
        for i, name, t in snap:
            name_to_param[i][name].data.copy_(t)


def _interpolate(target_modules, source_modules, lam):
    """In-place: target ← lam * target + (1-lam) * source for all params+buffers."""
    with torch.no_grad():
        for tgt, src in zip(target_modules, source_modules):
            tgt_sd = tgt.state_dict()
            src_sd = src.state_dict()
            new_sd = {}
            for k in tgt_sd:
                t = tgt_sd[k].float()
                s = src_sd[k].float() if k in src_sd else t
                new_sd[k] = (lam * t + (1.0 - lam) * s).to(tgt_sd[k].dtype)
            tgt.load_state_dict(new_sd)


class MoDeUL:
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'MoDe'

    def _train_de_one_epoch(self, de_bottoms, de_top, retain_loader,
                             de_opts_b, de_opt_t, ce):
        device = self.args.device
        for b in de_bottoms: b.train()
        de_top.train()
        for bx, by in retain_loader:
            bx, by = bx.to(device), by.to(device)
            parts = split_features(bx, self.args)
            embs = [de_bottoms[i](parts[i]) for i in range(len(de_bottoms))]
            agg = sum(embs) if self.args.aggregation == 'sum' else torch.cat(embs, dim=1)
            loss = ce(de_top(agg), by)
            de_opt_t.zero_grad()
            for o in de_opts_b: o.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for b in de_bottoms for p in b.parameters()] + list(de_top.parameters()),
                5.0)
            de_opt_t.step()
            for o in de_opts_b: o.step()

    def _memory_guidance_epoch(self, de_bottoms, de_top, forget_loader,
                               retain_loader, opts_b, opt_t, ce):
        """One epoch of memory-guidance: forget batch uses M_de pseudo-labels;
        retain batch uses true labels."""
        device = self.args.device
        for b in self.trainer.bottoms: b.train()
        self.trainer.top_model.train()
        for de_b in de_bottoms: de_b.eval()
        de_top.eval()

        # Process both loaders interleaved (forget first, then retain) so that
        # the two losses balance through SGD momentum.
        total_loss, nb = 0.0, 0
        for loader, is_forget in [(forget_loader, True), (retain_loader, False)]:
            if loader is None: continue
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                parts = split_features(bx, self.args)
                embs = [self.trainer.bottoms[i](parts[i]) for i in range(len(self.trainer.bottoms))]
                agg = self.trainer._aggregate(embs)
                logits = self.trainer.top_model(agg)
                if is_forget:
                    with torch.no_grad():
                        de_embs = [de_bottoms[i](parts[i]) for i in range(len(de_bottoms))]
                        de_agg = sum(de_embs) if self.args.aggregation == 'sum' else torch.cat(de_embs, dim=1)
                        pseudo = de_top(de_agg).argmax(dim=-1)
                    loss = ce(logits, pseudo)
                else:
                    loss = ce(logits, by)
                for o in opts_b: o.zero_grad()
                opt_t.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for b in self.trainer.bottoms for p in b.parameters() if p.requires_grad] +
                    [p for p in self.trainer.top_model.parameters()],
                    5.0)
                opt_t.step()
                if any(p.requires_grad for b in self.trainer.bottoms for p in b.parameters()):
                    for o in opts_b: o.step()
                total_loss += loss.item(); nb += 1
        return total_loss / max(nb, 1)

    def unlearn(self, forget_classes, retain_train_loader, forget_train_loader, **kw):
        device = self.args.device
        lam = float(getattr(self.args, 'mode_lambda', 0.95))
        warmup_rounds = int(getattr(self.args, 'mode_warmup_rounds', 3))
        mode_rounds = int(getattr(self.args, 'mode_rounds', 5))
        guidance_only_rounds = int(getattr(self.args, 'mode_guidance_only_rounds', 3))
        de_lr = float(getattr(self.args, 'mode_de_lr', 1e-2))
        respect_fim = bool(getattr(self.args, 'mode_respect_fim', False))

        # Build degradation models: deep copies of the originals, randomised
        de_bottoms = [copy.deepcopy(b) for b in self.trainer.bottoms]
        de_top = copy.deepcopy(self.trainer.top_model)
        _reset_params(de_bottoms + [de_top])
        for m in de_bottoms + [de_top]: m.to(device)

        # Unfreeze all params of the degradation model — it must learn from scratch
        for m in de_bottoms + [de_top]:
            for p in m.parameters(): p.requires_grad_(True)

        de_opts_b = [torch.optim.SGD(b.parameters(), lr=de_lr,
                                      momentum=self.args.momentum,
                                      weight_decay=self.args.weight_decay)
                      for b in de_bottoms]
        de_opt_t = torch.optim.SGD(de_top.parameters(), lr=de_lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        ce = nn.CrossEntropyLoss()

        start = time.time()
        history = []

        # ---- Warmup degradation model on retain ----
        for wr in range(warmup_rounds):
            self._train_de_one_epoch(de_bottoms, de_top, retain_train_loader,
                                      de_opts_b, de_opt_t, ce)
            history.append({'phase': 'warmup', 'round': wr})

        # ---- Main loop ----
        total_rounds = mode_rounds + guidance_only_rounds
        for rnd in range(total_rounds):
            do_mode = rnd < mode_rounds
            if do_mode:
                # Continue training W_de on retain
                self._train_de_one_epoch(de_bottoms, de_top, retain_train_loader,
                                          de_opts_b, de_opt_t, ce)
                # Snapshot frozen if requested
                snap = _snapshot_frozen(list(self.trainer.bottoms) + [self.trainer.top_model]) \
                       if respect_fim else None
                # W ← λW + (1-λ)W_de  (bottoms + top)
                _interpolate(self.trainer.bottoms, de_bottoms, lam)
                _interpolate([self.trainer.top_model], [de_top], lam)
                if snap is not None:
                    _restore_frozen(list(self.trainer.bottoms) + [self.trainer.top_model], snap)

            # Memory guidance (every round, including guidance-only tail)
            opts_b, opt_t = self.trainer._create_optimizers(lr=self.args.ul_lr)
            avg_loss = self._memory_guidance_epoch(
                de_bottoms, de_top, forget_train_loader, retain_train_loader,
                opts_b, opt_t, ce)
            history.append({'phase': 'mode_mg' if do_mode else 'mg_only',
                            'round': rnd, 'loss': avg_loss})

        return {'history': history, 'time': time.time() - start, 'method': 'mode',
                'config': {'lambda': lam, 'warmup_rounds': warmup_rounds,
                           'mode_rounds': mode_rounds,
                           'guidance_only_rounds': guidance_only_rounds}}
