"""EWC (Elastic Weight Consolidation) adapted to VFL.
Kirkpatrick et al. (PNAS 2017). Soft regularization using Fisher Information.
Each party independently regularizes its bottom model.
"""
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from data_utils import split_features


def _normalize_fisher(new_fisher, n_samples):
    if int(n_samples) <= 0:
        raise ValueError('Fisher estimation requires at least one sample')
    normalized = []
    for party_id, parameters in enumerate(new_fisher):
        averaged = {name: value / int(n_samples)
                    for name, value in parameters.items()}
        if not averaged:
            raise ValueError(f'party {party_id} has no Fisher parameters')
        values = list(averaged.values())
        if any(not torch.isfinite(value).all() for value in values):
            raise FloatingPointError(f'party {party_id} Fisher must be finite')
        if any(value.lt(0).any() for value in values):
            raise FloatingPointError(f'party {party_id} Fisher must be non-negative')
        trace = sum(value.sum() for value in values)
        if not torch.isfinite(trace) or float(trace) <= 0:
            raise FloatingPointError(
                f'party {party_id} Fisher trace must be finite and positive'
            )
        normalized.append({name: (value / trace).to(dtype=torch.float32)
                           for name, value in averaged.items()})
    return normalized


class EWCCL:
    """EWC for VFL: soft FIM regularization on bottom models."""
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'EWC_VFL'
        # Penalty strength. Was hardcoded 1e6, which (combined with the
        # cross-task Fisher accumulation below) pinned the bottoms at the task-0
        # optimum so hard the model could not learn ANY new class. Now wired to
        # args and defaulted into the stability/plasticity range.
        self.ewc_lambda = getattr(args, 'ewc_lambda', 1000.0)
        # Online-EWC EMA factor for Fisher across tasks (bounded, vs. the old
        # unnormalized sum that grew stiffer every task).
        self.fisher_decay = getattr(args, 'ewc_fisher_decay', 0.9)
        # Cap on samples used for the diagonal-Fisher estimate (<=0 = all).
        self.fisher_samples = getattr(args, 'ewc_fisher_samples', 1024)
        self.fisher = [{} for _ in range(args.num_parties)]
        self.old_params = [{} for _ in range(args.num_parties)]
        # --- Ported from the working lwf.py fix. EWC's quadratic param penalty
        # alone does NOT stop class-IL collapse here: the trainer's FULL-head CE
        # pushed every old-class logit down on each new batch (old-task acc -> 0),
        # and under the cosine head the bottoms still rotate the feature space.
        # ce_newonly scopes CE to the new-class slice (old columns no longer
        # suppressed); feat-KD (summed-L2) anchors the embedding so old-class
        # inputs keep mapping to their old region. Both reuse the shared flags. ---
        self.ce_newonly = getattr(args, 'lwf_ce_newonly', True)
        self.feat_distill_weight = getattr(args, 'feat_distill_weight', 0.0)
        self.old_bottoms = None

    def before_task(self, task_id, new_classes, seen_classes):
        req = max(seen_classes) + 1 if seen_classes else 0
        self.trainer.top_model.expand_classes(req, self.args.device)
        # Canonical class-IL CE scope: CE only on the NEW-class columns [n_old:req].
        old_classes = [c for c in seen_classes if c not in new_classes]
        n_old = (max(old_classes) + 1) if old_classes else 0
        if self.ce_newonly:
            self.trainer.ce_lo = n_old
            self.trainer.ce_hi = req
        # Snapshot the frozen previous-task backbone as the feature-KD teacher.
        if task_id > 0 and self.feat_distill_weight > 0:
            self.old_bottoms = [deepcopy(b).eval() for b in self.trainer.bottoms]
            for ob in self.old_bottoms:
                for p in ob.parameters():
                    p.requires_grad = False

    def _ewc_loss(self, bottoms, top_model, batch_x, batch_y, loss_ce=None):
        """loss_ce(new-class slice) + (lambda/2) * sum F (theta-theta*)^2 + feat-KD."""
        device = self.args.device
        total = loss_ce if loss_ce is not None else torch.tensor(0.0, device=device)

        # (1) Standard EWC penalty (lambda/2) * sum F (theta-theta*)^2. The /2 is the
        # canonical single factor; plasticity is governed by ewc_lambda (--ewc_lambda).
        if self.fisher[0]:
            penalty = torch.tensor(0.0, device=device)
            for k in range(len(bottoms)):
                for name, param in bottoms[k].named_parameters():
                    if name in self.fisher[k]:
                        fisher_val = self.fisher[k][name].to(device)
                        old_val = self.old_params[k][name].to(device)
                        penalty = penalty + (fisher_val * (param - old_val) ** 2).sum()
            total = total + (self.ewc_lambda / 2.0) * penalty

        # (2) Feature-KD (summed-L2) against the frozen previous backbone on current
        # data: pins the embedding space so old-class inputs don't drift into the
        # new-class region (the EWC param penalty does not constrain feature
        # *direction* under the cosine head). Exemplar-free; weight ~1 calibrated.
        if self.feat_distill_weight > 0 and self.old_bottoms is not None:
            parts = split_features(batch_x, self.args)
            with torch.no_grad():
                old_agg = self.trainer._aggregate(
                    [self.old_bottoms[i](parts[i]) for i in range(len(bottoms))])
            curr_agg = self.trainer._aggregate(
                [bottoms[i](parts[i]) for i in range(len(bottoms))])
            total = total + self.feat_distill_weight * ((curr_agg - old_agg) ** 2).sum(dim=1).mean()
        if not torch.isfinite(total):
            raise FloatingPointError('non-finite EWC loss')
        return total

    def train_task(self, train_loader, task_id):
        extra = self._ewc_loss if task_id > 0 else None
        return self.trainer.train_task(train_loader, self.args.epochs_per_task, extra_loss_fn=extra)

    def after_task(self, train_loader, task_id):
        """Estimate the diagonal Fisher and snapshot the reference params.

        Two fixes vs. the original:
          - Fisher is the true PER-SAMPLE empirical Fisher F = (1/N) sum_i g_i^2
            (one backward per sample), not the squared *batch-mean* gradient
            averaged over batches. Eval mode is used so BatchNorm uses running
            stats (a batch-of-1 in train mode has zero variance and corrupts
            the estimate). Capped at self.fisher_samples for runtime.
          - Across tasks we keep a bounded online-EWC EMA
            F <- decay*F + (1-decay)*F_new, instead of the old unnormalized
            sum that made the penalty stiffer (and plasticity worse) every task.
        """
        was_training = [b.training for b in self.trainer.bottoms]
        top_training = self.trainer.top_model.training
        for b in self.trainer.bottoms:
            b.eval()
        self.trainer.top_model.eval()

        new_fisher = [
            {name: torch.zeros_like(param, dtype=torch.float64)
             for name, param in self.trainer.bottoms[party].named_parameters()}
            for party in range(self.args.num_parties)
        ]
        cap = self.fisher_samples if self.fisher_samples and self.fisher_samples > 0 else float('inf')
        n_samples = 0
        ce = nn.CrossEntropyLoss()
        done = False
        for bx, by in train_loader:
            if done:
                break
            bx, by = bx.to(self.args.device), by.to(self.args.device)
            for j in range(bx.size(0)):
                if n_samples >= cap:
                    done = True
                    break
                xj, yj = bx[j:j + 1], by[j:j + 1]
                for b in self.trainer.bottoms:
                    b.zero_grad()
                self.trainer.top_model.zero_grad()
                parts = split_features(xj, self.args)
                embs = [self.trainer.bottoms[i](parts[i]) for i in range(self.args.num_parties)]
                agg = self.trainer._aggregate(embs)
                loss = ce(self.trainer.top_model(agg), yj)
                loss.backward()
                gradients = []
                for k in range(self.args.num_parties):
                    for n, p in self.trainer.bottoms[k].named_parameters():
                        if p.grad is not None:
                            gradients.append((k, n, p.grad.detach()))
                finite = torch.stack([
                    torch.isfinite(gradient).all()
                    for _, _, gradient in gradients
                ])
                if not finite.all():
                    bad = int((~finite).nonzero()[0])
                    party, name, _ = gradients[bad]
                    raise FloatingPointError(
                        f'non-finite Fisher gradient at party {party} parameter {name}'
                    )
                for k, name, gradient in gradients:
                    new_fisher[k][name].add_(gradient.double().square())
                n_samples += 1

        # Per-sample normalization, then bounded online-EWC EMA across tasks.
        new_fisher = _normalize_fisher(new_fisher, n_samples)
        for k in range(self.args.num_parties):
            for name, value in new_fisher[k].items():
                if name in self.fisher[k]:
                    value = (self.fisher_decay * self.fisher[k][name]
                             + (1.0 - self.fisher_decay) * value)
                if not torch.isfinite(value).all() or value.lt(0).any():
                    raise FloatingPointError(
                        f'non-finite merged Fisher at party {k} parameter {name}'
                    )
                self.fisher[k][name] = value

        # Save current parameters as the new quadratic-penalty anchor.
        for k in range(self.args.num_parties):
            self.old_params[k] = {n: p.data.clone().cpu()
                                   for n, p in self.trainer.bottoms[k].named_parameters()}

        for b in self.trainer.bottoms:
            b.zero_grad()
        self.trainer.top_model.zero_grad()
        # Restore the modes the trainer expects.
        for b, was in zip(self.trainer.bottoms, was_training):
            b.train(was)
        self.trainer.top_model.train(top_training)

    def get_state(self):
        return {
            'fisher': deepcopy(self.fisher),
            'old_params': deepcopy(self.old_params),
        }

    def load_state(self, s):
        self.fisher = deepcopy(
            s.get('fisher', [{} for _ in range(self.args.num_parties)])
        )
        self.old_params = deepcopy(
            s.get('old_params', [{} for _ in range(self.args.num_parties)])
        )
        if len(self.fisher) != self.args.num_parties:
            raise ValueError('EWC checkpoint Fisher does not match num_parties')
        if len(self.old_params) != self.args.num_parties:
            raise ValueError('EWC checkpoint anchors do not match num_parties')
