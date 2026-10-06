"""FedOSD (Pan et al., AAAI 2025) adapted to VFL for class-level unlearning.

Original FedOSD aggregates pseudo-gradients across multiple clients —
some "forget" (target client), some "retain" (the remaining
federation) — then picks a steepest direction d that is closest to
−g_u while orthogonal to the retain pseudo-gradients G.

In a single-node VFL setup we don't have a federation of clients, so
we substitute:
    g_u   ← pseudo-gradient from UCE local training on FORGET samples
    g_r   ← pseudo-gradient from CE  local training on RETAIN samples

`g_r` is treated as a single "remaining-client" direction.  The
orthogonalisation step removes the component of g_u parallel to g_r,
mirroring Eq. (6) of the paper for |G|=1.

Post-training stage projects the retain pseudo-gradient orthogonal to
g_a = ω_t − ω_0 whenever g_r · g_a > 0, preventing the model from
reverting toward its pre-unlearning state.

FIM freeze
----------
Only parameters with ``requires_grad=True`` participate in the
pseudo-gradient computation and the parameter-space update.  Frozen
parameters are left untouched.
"""
from __future__ import annotations
import copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import split_features


class UCELoss(nn.Module):
    """L_UCE = -Σ_c y_c · log(1 - p_c/2)   (Eq. 3 of FedOSD)"""
    def forward(self, logits, targets):
        p = F.softmax(logits, dim=-1)
        log_term = torch.log(1.0 - p / 2.0 + 1e-8)
        one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        return -(one_hot * log_term).sum(-1).mean()


def _trainable_param_specs(modules):
    """List of (module_idx, param_name, shape, numel) for every requires_grad param."""
    specs = []
    for i, m in enumerate(modules):
        for name, p in m.named_parameters():
            if p.requires_grad:
                specs.append((i, name, tuple(p.shape), p.numel()))
    return specs


def _flatten(modules, specs):
    """Concatenate values of trainable params into one 1-D vector."""
    name_dicts = [dict(m.named_parameters()) for m in modules]
    parts = [name_dicts[i][name].data.float().reshape(-1) for (i, name, _, _) in specs]
    return torch.cat(parts) if parts else torch.zeros(0, device=modules[0].parameters().__next__().device)


def _apply_delta(modules, delta, specs):
    """Add delta (1-D vector) to each trainable param in `modules`, in place."""
    name_dicts = [dict(m.named_parameters()) for m in modules]
    offset = 0
    with torch.no_grad():
        for (i, name, shape, n) in specs:
            chunk = delta[offset:offset+n].reshape(shape)
            param = name_dicts[i][name]
            param.add_(chunk.to(param.dtype))
            offset += n


class FedOSDUL:
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'FedOSD'

    def _local_train(self, loader, loss_fn, lr, epochs):
        """Deepcopy bottoms+top, train locally with the given loss. Return copies."""
        device = self.args.device
        bottoms = [copy.deepcopy(b).to(device) for b in self.trainer.bottoms]
        top = copy.deepcopy(self.trainer.top_model).to(device)
        for b in bottoms: b.train()
        top.train()

        params = [p for b in bottoms for p in b.parameters() if p.requires_grad] + \
                 [p for p in top.parameters() if p.requires_grad]
        if not params:
            return bottoms, top
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)

        for _ in range(epochs):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                parts = split_features(bx, self.args)
                embs = [bottoms[i](parts[i]) for i in range(len(bottoms))]
                agg = sum(embs) if self.args.aggregation == 'sum' else torch.cat(embs, dim=1)
                logits = top(agg)
                loss = loss_fn(logits, by)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
        return bottoms, top

    def unlearn(self, forget_classes, retain_train_loader, forget_train_loader, **kw):
        device = self.args.device
        T_u = int(getattr(self.args, 'fedosd_unlearn_rounds', 5))
        T_post = int(getattr(self.args, 'fedosd_post_rounds', 5))
        lr = float(getattr(self.args, 'fedosd_local_lr', self.args.ul_lr))
        local_epochs = int(getattr(self.args, 'fedosd_local_epochs', 1))
        global_lr = float(getattr(self.args, 'fedosd_global_lr', 0.5))
        max_step_norm = float(getattr(self.args, 'fedosd_max_step_norm', 5.0))

        modules = list(self.trainer.bottoms) + [self.trainer.top_model]
        specs = _trainable_param_specs(modules)
        if not specs:
            return {'history': [], 'time': 0.0, 'method': 'fedosd',
                    'skipped': 'no_trainable_params'}

        omega_0 = _flatten(modules, specs).to(device)
        uce = UCELoss()
        ce = nn.CrossEntropyLoss()

        start = time.time()
        history = []

        # ============ Unlearning stage ============
        for t in range(T_u):
            omega_t = _flatten(modules, specs).to(device)

            f_bs, f_t = self._local_train(forget_train_loader, uce, lr, local_epochs)
            omega_f = _flatten(list(f_bs) + [f_t], specs).to(device)
            g_u = (omega_t - omega_f) / lr
            del f_bs, f_t

            r_bs, r_t = self._local_train(retain_train_loader, ce, lr, local_epochs)
            omega_r = _flatten(list(r_bs) + [r_t], specs).to(device)
            g_r = (omega_t - omega_r) / lr
            del r_bs, r_t

            # d ∝ -(I - g_r g_r^T / ||g_r||^2) g_u
            gr_n2 = (g_r * g_r).sum()
            if gr_n2.item() > 1e-12:
                proj = (g_u * g_r).sum() / gr_n2
                ortho = g_u - proj * g_r
                d = -ortho
            else:
                d = -g_u

            gu_norm = g_u.norm().item()
            d_norm = d.norm().item()
            if d_norm > 1e-12:
                d = d * (gu_norm / d_norm)

            step = global_lr * d
            step_norm = step.norm().item()
            if step_norm > max_step_norm:
                step = step * (max_step_norm / step_norm)
                step_norm = max_step_norm

            _apply_delta(modules, step, specs)

            cos = float(F.cosine_similarity(g_u.unsqueeze(0), d.unsqueeze(0)).item())
            history.append({'phase': 'unlearn', 'round': t,
                            'gu_norm': gu_norm, 'gr_norm': float(g_r.norm().item()),
                            'step_norm': step_norm, 'cos_gu_d': cos})

        # ============ Post-training stage ============
        for t in range(T_post):
            omega_t = _flatten(modules, specs).to(device)
            g_a = omega_t - omega_0

            r_bs, r_t = self._local_train(retain_train_loader, ce, lr, local_epochs)
            omega_r = _flatten(list(r_bs) + [r_t], specs).to(device)
            g_r = (omega_t - omega_r) / lr
            del r_bs, r_t

            ga_n2 = (g_a * g_a).sum().item()
            dot = (g_r * g_a).sum().item()
            if dot > 0 and ga_n2 > 1e-12:
                g_proj = g_r - (dot / ga_n2) * g_a
                gr_norm = g_r.norm().item()
                gp_norm = g_proj.norm().item()
                if gp_norm > 1e-12:
                    g_proj = g_proj * (gr_norm / gp_norm)
                g_r = g_proj

            # gradient descent: ω ← ω - lr * g_r
            post_step = -lr * g_r
            psn = post_step.norm().item()
            psn_max = max_step_norm * 3.0
            if psn > psn_max:
                post_step = post_step * (psn_max / psn)
                psn = psn_max
            _apply_delta(modules, post_step, specs)

            new_omega = omega_t + post_step
            dist = (new_omega - omega_0).norm().item()
            history.append({'phase': 'post', 'round': t,
                            'gr_norm': float(g_r.norm().item()),
                            'post_step_norm': psn, 'dist_from_omega0': dist})

        return {'history': history, 'time': time.time() - start, 'method': 'fedosd',
                'config': {'T_u': T_u, 'T_post': T_post, 'lr': lr,
                           'global_lr': global_lr,
                           'max_step_norm': max_step_norm}}
