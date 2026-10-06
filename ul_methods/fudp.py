"""FUDP (Wang et al., WWW 2022) adapted to VFL for class-level unlearning.

Class-discriminative channel pruning, applied PER-PARTY to bottom-model
Conv2d layers:

  1. Forward-hook every Conv2d (or the BN immediately following it) of
     each ``trainer.bottoms[i]`` to record post-ReLU + global-avg-pool
     activation per channel.
  2. Build per-class activation matrix A* (|U| x C_out) by averaging
     activations of each seen class.
  3. Compute TF-IDF per channel per layer with the forget class as
     "document" (Eqs. 6-8 of the paper):

         TF[j]   = A*[u_forget, j] / sum_j A*[u_forget, j]
         IDF[j]  = log( (1+|U|) /
                        (1+|{u: A*[u,j] ≥ mean_j A*[u,:]}|) )
         TFIDF[j]= TF[j] * IDF[j]

  4. Top-R fraction of channels by TFIDF score are zeroed in the conv
     kernel (and BN gamma/beta if present), with a backward hook
     registered on each affected parameter so its gradient slice stays
     at zero through subsequent fine-tuning + later CL tasks.
  5. Short retain fine-tune to recover utility (`freeze_backbone=True`
     by default, mirroring the original safer protocol — set
     ``--fudp_freeze_backbone false`` to recover the paper's literal
     unfrozen variant).

We hook the BN output (not the raw conv output) when BN is present, so
the activation distribution we score against matches what the next
layer actually sees.  Without this fix, R=0.15 pruning leaves the
forget-class accuracy at ~30% — an empirically-validated detail from
the original code.
"""
from __future__ import annotations
import time
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import split_features


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _enumerate_conv_layers(module: nn.Module) -> List[Tuple[str, nn.Conv2d, Optional[nn.BatchNorm2d]]]:
    """Walk `module` recursively; return [(qualified_name, conv, bn_or_None), ...]
    in forward order.  ``bn`` is the first BN2d that follows ``conv`` inside
    the same parent before the next Conv2d."""
    out = []

    def _walk(m, prefix=""):
        children = list(m.named_children())
        i = 0
        while i < len(children):
            cname, cmod = children[i]
            full = f"{prefix}.{cname}" if prefix else cname
            if isinstance(cmod, nn.Conv2d):
                bn = None
                for j in range(i + 1, len(children)):
                    nxt = children[j][1]
                    if isinstance(nxt, nn.BatchNorm2d):
                        bn = nxt
                        break
                    if isinstance(nxt, nn.Conv2d):
                        break
                out.append((full, cmod, bn))
                _walk(cmod, full)
            else:
                _walk(cmod, full)
            i += 1

    _walk(module)
    return out


@torch.no_grad()
def _collect_class_channel_activation(party_idx: int, trainer, loader,
                                      n_classes: int, args) -> Dict[str, torch.Tensor]:
    """Forward all batches from `loader` through `trainer.bottoms[party_idx]`,
    accumulate per-class post-ReLU+AvgPool activation sums + counts.
    Return ``{layer_name: A*[c, j]}`` (|U|, C_out) tensors on CPU."""
    device = args.device
    bottom = trainer.bottoms[party_idx]
    layers = _enumerate_conv_layers(bottom)
    activations: Dict[str, torch.Tensor] = {}

    def _make_hook(name):
        def _hook(_m, _i, out):
            relu_out = F.relu(out)
            pooled = relu_out.mean(dim=(2, 3))
            activations[name] = pooled.detach()
        return _hook

    handles = []
    for name, conv, bn in layers:
        target = bn if bn is not None else conv
        handles.append(target.register_forward_hook(_make_hook(name)))

    sums = {name: torch.zeros(n_classes, conv.out_channels, device=device)
             for name, conv, _ in layers}
    counts = torch.zeros(n_classes, dtype=torch.long, device=device)

    bottom.eval()
    for bx, by in loader:
        bx = bx.to(device); by = by.to(device)
        parts = split_features(bx, args)
        activations.clear()
        _ = bottom(parts[party_idx])
        for name in sums:
            act = activations[name]
            present = torch.unique(by)
            for c in present.tolist():
                mask = (by == int(c))
                if mask.any():
                    sums[name][int(c)] += act[mask].sum(dim=0)
        for c in by.tolist():
            counts[int(c)] += 1

    for h in handles:
        h.remove()

    A_star = {}
    denom = counts.clamp(min=1).unsqueeze(1).float()
    for name in sums:
        A_star[name] = (sums[name] / denom).cpu()
    return A_star, counts.cpu()


def _tfidf_layer(A_layer: torch.Tensor, forget_class: int,
                 n_classes: int, eps: float = 1e-12) -> torch.Tensor:
    """Compute TFIDF score (C_out,) for one layer."""
    A = A_layer.clamp(min=0.0)
    row_u = A[forget_class]
    row_sum = row_u.sum().clamp(min=eps)
    tf = row_u / row_sum
    class_means = A.mean(dim=1, keepdim=True)
    above = (A >= class_means).float()
    df = above.sum(dim=0)
    idf = torch.log((1.0 + n_classes) / (1.0 + df))
    return tf * idf


def _build_prune_masks(A_star: Dict[str, torch.Tensor],
                       forget_classes: List[int],
                       n_classes: int,
                       R: float) -> Dict[str, torch.Tensor]:
    """Union of top-R channels (by TFIDF) across forget classes, per layer."""
    masks = {}
    for name, A in A_star.items():
        c_out = A.shape[1]
        n_keep_top = max(1, int(round(R * c_out)))
        union = torch.zeros(c_out, dtype=torch.bool)
        for f in forget_classes:
            scores = _tfidf_layer(A, f, n_classes)
            topk = torch.topk(scores, k=n_keep_top, largest=True).indices
            union[topk] = True
        masks[name] = union
    return masks


def _apply_prune_masks(bottom: nn.Module, masks: Dict[str, torch.Tensor], device):
    """Zero conv weights (+ BN γ/β) on masked channels and attach
    backward hooks so any subsequent training keeps them at zero.
    Returns ``(info_dict, handles_list)``.  Hooks are kept attached;
    the caller may remove them via ``[h.remove() for h in handles]``."""
    layers = _enumerate_conv_layers(bottom)
    info, handles = {}, []
    for name, conv, bn in layers:
        if name not in masks: continue
        mask = masks[name].to(device)
        if not mask.any():
            info[name] = {'frac_pruned': 0.0, 'n_channels': conv.out_channels, 'n_pruned': 0}
            continue
        with torch.no_grad():
            conv.weight.data[mask] = 0.0
            if conv.bias is not None:
                conv.bias.data[mask] = 0.0
            if bn is not None:
                if bn.weight is not None: bn.weight.data[mask] = 0.0
                if bn.bias   is not None: bn.bias.data[mask]   = 0.0

        def _mk_hook(m_local):
            def _hook(grad):
                grad = grad.clone()
                grad[m_local] = 0.0
                return grad
            return _hook

        handles.append(conv.weight.register_hook(_mk_hook(mask)))
        if conv.bias is not None:
            handles.append(conv.bias.register_hook(_mk_hook(mask)))
        if bn is not None:
            if bn.weight is not None:
                handles.append(bn.weight.register_hook(_mk_hook(mask)))
            if bn.bias is not None:
                handles.append(bn.bias.register_hook(_mk_hook(mask)))

        info[name] = {'frac_pruned': float(mask.float().mean().item()),
                      'n_channels': conv.out_channels,
                      'n_pruned': int(mask.sum().item())}
    return info, handles


# ----------------------------------------------------------------------
# UL method
# ----------------------------------------------------------------------

class FUDPUL:
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'FUDP'

    def unlearn(self, forget_classes, retain_train_loader, forget_train_loader, **kw):
        device = self.args.device
        eff = kw.get('effective_classes', []) or []
        # n_classes for IDF — use max seen class index + 1
        n_seen = max(max(eff) + 1 if eff else 0,
                     max(forget_classes) + 1 if forget_classes else 0,
                     self.trainer.top_model.classifier.out_features)
        R = float(getattr(self.args, 'fudp_R', 0.05))
        finetune_epochs = int(getattr(self.args, 'fudp_finetune_epochs', -1))
        if finetune_epochs < 0:                       # -1 sentinel -> use ul_epochs
            finetune_epochs = self.args.ul_epochs
        freeze_backbone = bool(getattr(self.args, 'fudp_freeze_backbone', True))

        start = time.time()
        history = []

        # ------------------------------------------------------------------
        # Phase 1+2: per-party TFIDF + prune mask + apply
        # ------------------------------------------------------------------
        total_pruned, total_channels = 0, 0
        per_party_info = {}
        all_handles = []
        for i in range(len(self.trainer.bottoms)):
            # Collect activation stats on BOTH retain and forget loaders so we
            # have per-class signal for every seen class (IDF needs all of them).
            A_retain, n_retain = _collect_class_channel_activation(
                i, self.trainer, retain_train_loader, n_seen, self.args)
            A_forget, n_forget = _collect_class_channel_activation(
                i, self.trainer, forget_train_loader, n_seen, self.args)

            # Merge: weighted mean per class
            A_merged = {}
            for name in A_retain:
                ar, af = A_retain[name].double(), A_forget[name].double()
                nr, nf = n_retain.double().unsqueeze(1), n_forget.double().unsqueeze(1)
                denom = (nr + nf).clamp(min=1.0)
                A_merged[name] = ((ar * nr + af * nf) / denom).float()

            masks = _build_prune_masks(A_merged, forget_classes, n_seen, R)
            info, handles = _apply_prune_masks(self.trainer.bottoms[i], masks, device)
            all_handles.extend(handles)
            per_party_info[f'party_{i}'] = info

            for _name, m in masks.items():
                total_pruned += int(m.sum().item())
                total_channels += int(m.numel())

        history.append({'phase': 'prune', 'R': R,
                        'total_pruned': total_pruned,
                        'total_channels': total_channels,
                        'frac_pruned': total_pruned / max(total_channels, 1)})

        # ------------------------------------------------------------------
        # Phase 3: retain fine-tune
        # ------------------------------------------------------------------
        # Save original requires_grad state so we can restore (so the next
        # CL task isn't accidentally frozen by us).
        orig_grad = []
        for b in self.trainer.bottoms:
            for p in b.parameters():
                orig_grad.append((p, p.requires_grad))
        if freeze_backbone:
            for b in self.trainer.bottoms:
                for p in b.parameters():
                    p.requires_grad_(False)

        opts_b, opt_t = self.trainer._create_optimizers(lr=self.args.ul_lr)
        ce = nn.CrossEntropyLoss()
        any_bot = any(p.requires_grad for b in self.trainer.bottoms for p in b.parameters())

        for ep in range(finetune_epochs):
            for b in self.trainer.bottoms: b.train()
            self.trainer.top_model.train()
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
                    [p for b in self.trainer.bottoms for p in b.parameters() if p.requires_grad] +
                    list(self.trainer.top_model.parameters()), 5.0)
                opt_t.step()
                if any_bot:
                    for o in opts_b: o.step()
                ep_loss += float(loss.item()); nb += 1
            history.append({'phase': 'finetune', 'epoch': ep, 'loss': ep_loss / max(nb, 1)})

        # Restore requires_grad so subsequent CL tasks behave normally.
        # The pruning grad-hooks remain attached (correct: pruned channels
        # must stay zero across the rest of the timeline).
        for p, was_grad in orig_grad:
            p.requires_grad_(was_grad)

        return {'history': history, 'time': time.time() - start, 'method': 'fudp',
                'config': {'R': R, 'finetune_epochs': finetune_epochs,
                           'freeze_backbone': freeze_backbone,
                           'per_party': per_party_info,
                           'total_pruned': total_pruned,
                           'total_channels': total_channels}}
