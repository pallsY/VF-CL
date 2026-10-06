"""PRL (Prospective Representation Learning) adapted to VFL.

Shi et al., NeurIPS 2024. "Prospective Representation Learning for
Non-Exemplar Class-Incremental Learning". The original distills LwF-style
features, expands the classifier 4x via rotation prediction to reserve
prospective capacity, and replays mixed-up virtual prototypes.

Minimal port (no AE, no PES):
  loss = CE_clf  +  λ_fkd · ||f_curr - f_old||_2  +  λ_proto · CE_proto
where CE_clf is over the 4x-expanded label space, f_old comes from a
frozen snapshot of bottoms+top (server-side aggregated embedding), and
CE_proto is computed by mixing past-class prototypes with current
features.

VFL specifics:
- Rotation is applied to the FULL raw batch_x before split_features,
  so every party sees the same rotation applied to its own slice.
  This is leakage-free and preserves the 4x-class semantics.
- The frozen "old" feature uses the same aggregation as current.
- top_model gets wrapped (PRLTopWrapper): training -> 4*C logits,
  eval -> pool back to C logits via rotation-0 slice.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from data_utils import split_features


class PRLTopWrapper(nn.Module):
    """Wraps the existing TopModel so that evaluation collapses the 4x
    expanded logits back to num_classes by taking rotation-0 slots."""

    def __init__(self, base_top):
        super().__init__()
        self.base = base_top

    @property
    def classifier(self):
        return self.base.classifier

    def expand_classes(self, real_num, device):
        # The "real" class count grows by `real_num`; allocate 4x slots.
        self.base.expand_classes(real_num * 4, device)

    def forward(self, x):
        logits = self.base(x)  # (B, 4*C), block layout [rot0(C) | rot1(C) | rot2 | rot3]
        if self.training:
            return logits
        # Eval: rotation-0 block is the first C columns (= the real classes)
        return logits[:, :logits.size(1) // 4]


class PRLCL:
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'PRL_VFL_min'
        self.lambda_fkd = getattr(args, 'prl_lambda_fkd', 15.0)
        self.lambda_proto = getattr(args, 'prl_lambda_proto', 15.0)
        self.temp = getattr(args, 'prl_temp', 0.1)

        # Wrap top model once so eval pooling is in place.
        if not isinstance(self.trainer.top_model, PRLTopWrapper):
            self.trainer.top_model = PRLTopWrapper(self.trainer.top_model).to(args.device)

        # Frozen snapshots from the prior task
        self.old_bottoms = None
        self.old_top = None
        # Class prototypes (real_class_id -> embedding_dim numpy)
        self.protos = {}

    # ------------------------- helpers -------------------------
    def _expanded_labels(self, batch_y, rot_id):
        """Block layout: rotation rot_id of real label y -> column rot*C + y.
        rotation-0 of class y -> column y, which MATCHES the trainer's loss_ce
        (which supervises column y with the real label) -> no label-column conflict."""
        return rot_id * self.args.num_classes + batch_y

    def _build_rotated_batch(self, batch_x, batch_y):
        """Return (4N, ...) rotated images and their 4x-expanded labels."""
        rots = [torch.rot90(batch_x, k, dims=[2, 3]) for k in range(4)]
        rx = torch.cat(rots, dim=0)
        ry = torch.cat([self._expanded_labels(batch_y, k) for k in range(4)], dim=0)
        return rx, ry

    def _agg_forward(self, bottoms, batch_x):
        """Forward batch through given bottoms list + aggregate."""
        parts = split_features(batch_x, self.args)
        embs = [bottoms[i](parts[i]) for i in range(len(bottoms))]
        return self.trainer._aggregate(embs)

    # ------------------------- lifecycle -------------------------
    def before_task(self, task_id, new_classes, seen_classes):
        # Fixed full block-layout head: 4 * num_classes columns from the start,
        # so column indices (rot*C + y) are stable across tasks.
        self.trainer.top_model.expand_classes(self.args.num_classes, self.args.device)

        # Snapshot frozen old bottoms + top as the distillation reference.
        # Default: re-snapshot every task (chained one-step LwF). With
        # prl_fixed_anchor, snapshot ONCE (at task 1) and never refresh, so the
        # anchor always points at the task-0 backbone instead of chaining -
        # this stops the compounding drift that wipes early tasks by task 4.
        fixed_anchor = getattr(self.args, 'prl_fixed_anchor', False)
        if task_id > 0 and not (fixed_anchor and self.old_bottoms is not None):
            self.old_bottoms = [deepcopy(b).eval() for b in self.trainer.bottoms]
            for ob in self.old_bottoms:
                for p in ob.parameters():
                    p.requires_grad = False
            self.old_top = deepcopy(self.trainer.top_model).eval()
            for p in self.old_top.parameters():
                p.requires_grad = False

        # Frozen-backbone variant: after task `prl_freeze_after`, stop training
        # the bottoms entirely (weights AND BatchNorm running stats), turning
        # them into a fixed feature extractor. Stored prototypes then stay valid
        # (no feature drift), so old classes can't be expelled. -1 = never.
        freeze_after = getattr(self.args, 'prl_freeze_after', -1)
        if freeze_after >= 0 and task_id > freeze_after:
            for b in self.trainer.bottoms:
                b.eval()
                for p in b.parameters():
                    p.requires_grad = False
                for m in b.modules():
                    if isinstance(m, nn.modules.batchnorm._BatchNorm):
                        m.momentum = 0.0  # freeze running mean/var under train() mode

    def _prl_loss(self, bottoms, top_model, batch_x, batch_y, loss_ce=None):
        """Custom training loss replacing the trainer's plain CE.
        Note: when this is passed as extra_loss_fn, the trainer has
        already computed `loss_ce` using a single (un-rotated) forward
        on agg_detached. We instead compute everything from scratch
        on the rotated batch so 4x-class CE goes through bottoms too."""
        device = self.args.device

        # 1) Rotation augmentation
        rx, ry = self._build_rotated_batch(batch_x, batch_y)

        # 2) Forward through current bottoms + top (no detach so grads flow)
        agg = self._agg_forward(bottoms, rx)
        logits = top_model(agg)  # (4N, 4C)

        # 3) Classification CE on expanded labels (replaces the trainer's CE)
        loss_clf = F.cross_entropy(logits / self.temp, ry)

        # 4) Feature distillation against frozen old: mean over the batch of the
        # per-sample squared L2 feature distance. We sum over the feature dims
        # (NOT mean) so the anchor gradient does not get diluted by 1/embed_dim;
        # F.mse_loss (mean over batch AND all 512 dims) made the per-coordinate
        # gradient ~512x too weak, so it only bit at lambda_fkd~500-1000. With
        # this form lambda_fkd~1 anchors the bottom against the cosine-head CE
        # (which, training only on the new classes, otherwise pulls ALL features
        # toward the new-class directions -> old classes collapse to argmax 0).
        # The original `dist(agg, agg_old)/N` (a single global Frobenius scalar)
        # was weaker still.
        loss_fkd = torch.tensor(0.0, device=device)
        if self.old_bottoms is not None:
            with torch.no_grad():
                agg_old = self._agg_forward(self.old_bottoms, rx)
            loss_fkd = self.lambda_fkd * ((agg - agg_old) ** 2).sum(dim=1).mean()

        # 5) Virtual prototype CE (mixup variant from PRL paper)
        loss_proto = torch.tensor(0.0, device=device)
        if self.protos:
            n_proto = batch_y.size(0)
            old_classes = list(self.protos.keys())
            proto_embs, proto_lbls = [], []
            cur_feats = agg[:rx.size(0) // 4].detach()  # rot-0 portion as current features
            for i in range(n_proto):
                c = old_classes[np.random.randint(len(old_classes))]
                lam = np.random.beta(0.5, 0.5)
                base = torch.from_numpy(self.protos[c]).to(device)
                cur_i = cur_feats[i % cur_feats.size(0)]
                if np.random.random() >= 0.5:
                    mixed = (1 + lam) * base - lam * cur_i
                else:
                    mixed = (1 - lam) * base + lam * cur_i
                proto_embs.append(mixed)
                proto_lbls.append(c)  # rotation-0 slot = column c (block layout)
            pe = torch.stack(proto_embs)
            pl = torch.tensor(proto_lbls, dtype=torch.long, device=device)
            loss_proto = self.lambda_proto * F.cross_entropy(top_model(pe) / self.temp, pl)

        extra = loss_clf + loss_fkd + loss_proto
        # Add loss_ce (computed by trainer on agg_detached) so the split-learning
        # protocol sees agg_detached.grad and bottoms get stepped. With the block
        # layout, the trainer's loss_ce supervises column y == rotation-0 slot of
        # class y, so it AGREES with loss_clf (no label-column conflict).
        return (loss_ce + extra) if loss_ce is not None else extra

    def train_task(self, train_loader, task_id):
        return self.trainer.train_task(
            train_loader, self.args.epochs_per_task, extra_loss_fn=self._prl_loss
        )

    def after_task(self, train_loader, task_id):
        """Compute per-class prototypes on real (un-rotated) features, and -
        with --prl_sdc - first compensate OLD prototypes for backbone drift so a
        TRAINABLE backbone does not invalidate them (the reason heavy distillation
        only delays early-task collapse). Semantic Drift Compensation, Yu et al.,
        CVPR 2020: estimate each old prototype's drift from the drift of the
        current task's samples between the pre-task backbone (== the space the
        stored protos live in) and the post-task backbone, kernel-weighted by
        proximity in the old space. Requires the chained snapshot (NOT
        prl_fixed_anchor, whose old_bottoms is the stale task-0 backbone)."""
        device = self.args.device
        for b in self.trainer.bottoms:
            b.eval()
        use_sdc = (getattr(self.args, 'prl_sdc', False)
                   and self.old_bottoms is not None and len(self.protos) > 0)
        feats, labels, old_feats = [], [], []
        with torch.no_grad():
            for bx, by in train_loader:
                bx = bx.to(device)
                feats.append(self._agg_forward(self.trainer.bottoms, bx).cpu())
                labels.append(by)
                if use_sdc:
                    old_feats.append(self._agg_forward(self.old_bottoms, bx).cpu())
        F_all = torch.cat(feats)
        Y_all = torch.cat(labels).numpy()

        if use_sdc:
            O_all = torch.cat(old_feats)               # current samples in stored-proto space
            delta = F_all - O_all                      # per-sample drift to the new space
            for c in list(self.protos.keys()):
                pc = torch.from_numpy(self.protos[c]).float()
                d2 = ((O_all - pc) ** 2).sum(1)        # proximity in the OLD space
                sigma2 = d2.mean().clamp_min(1e-6)     # auto-bandwidth (per class)
                w = torch.exp(-d2 / (2 * sigma2))
                sw = w.sum()
                if sw > 1e-8:
                    self.protos[c] = (pc + (w.unsqueeze(1) * delta).sum(0) / sw).numpy()

        F_np = F_all.numpy()
        for c in np.unique(Y_all):
            self.protos[int(c)] = F_np[Y_all == c].mean(axis=0)
        print(f"    PRL prototypes: {len(self.protos)} classes (sdc={use_sdc})")

    def get_state(self):
        return {'protos': dict(self.protos)}

    def load_state(self, s):
        self.protos = s.get('protos', {})
