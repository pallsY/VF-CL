"""PASS+VFL: prototype augmentation + self-supervised rotation prediction.

Following Zhu et al. (CVPR 2021), augment training with rotation prediction
as an auxiliary self-supervised task to learn rotation-equivariant features.
The rotation head is attached to top_model so its parameters are managed by
the trainer's top-model optimizer.
"""
import torch, torch.nn as nn, numpy as np
from copy import deepcopy
from data_utils import split_features


class ProtoAugCL:
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'ProtoAug_VFL'
        self.protos = {}
        self.radius = 0.0
        self.ssl_weight = getattr(args, 'pass_ssl_weight', 1.0)
        self._attach_rotation_head()

    def _attach_rotation_head(self):
        """Attach a 4-way rotation classifier as a submodule of top_model.
        Idempotent: only attaches if not already present."""
        top = self.trainer.top_model
        if hasattr(top, 'rotation_head'):
            return
        embed_dim = top.classifier.in_features
        top.rotation_head = nn.Linear(embed_dim, 4).to(self.args.device)

    def before_task(self, task_id, new_classes, seen_classes):
        req = max(seen_classes) + 1 if seen_classes else 0
        self.trainer.top_model.expand_classes(req, self.args.device)
        self._attach_rotation_head()

    def _rotation_loss(self, bottoms, top_model, batch_x):
        """Forward 4 rotations through bottoms+aggregate, predict rotation id.
        Grad flows back into bottoms via agg (no detach), providing extra
        rotation-invariant supervision."""
        device = self.args.device
        # Stack rotations: [x, rot90, rot180, rot270] along batch dim
        rot_batch = torch.cat([torch.rot90(batch_x, k, dims=[2, 3]) for k in range(4)], dim=0)
        bs = batch_x.size(0)
        rot_labels = torch.arange(4, device=device).repeat_interleave(bs)

        parts = split_features(rot_batch, self.args)
        embs = [bottoms[i](parts[i]) for i in range(len(bottoms))]
        agg = self.trainer._aggregate(embs)
        rot_logits = top_model.rotation_head(agg)
        return nn.CrossEntropyLoss()(rot_logits, rot_labels)

    def _pass_ssl_loss(self, bottoms, top_model, batch_x, batch_y, loss_ce=None):
        device = self.args.device
        extra = torch.tensor(0.0, device=device)

        # Prototype replay (only if we have past prototypes)
        if self.protos:
            pc = list(self.protos.keys())
            bs = batch_y.size(0)
            pe, pl = [], []
            for _ in range(bs):
                c = pc[np.random.randint(len(pc))]
                pe.append(self.protos[c].to(device)
                          + torch.randn(self.protos[c].shape, device=device) * self.radius)
                pl.append(c)
            pe = torch.stack(pe).to(device)
            pl = torch.tensor(pl, dtype=torch.long, device=device)
            extra = extra + self.args.proto_aug_weight * nn.CrossEntropyLoss()(top_model(pe), pl)

        # Self-supervised rotation prediction (every task, not just task>0)
        if self.ssl_weight > 0:
            extra = extra + self.ssl_weight * self._rotation_loss(bottoms, top_model, batch_x)

        return (loss_ce + extra) if loss_ce is not None else extra

    def train_task(self, train_loader, task_id):
        # SSL is active from task 0; prototypes only kick in from task 1
        extra = self._pass_ssl_loss
        return self.trainer.train_task(train_loader, self.args.epochs_per_task, extra_loss_fn=extra)

    def after_task(self, train_loader, task_id):
        emb, labels = self.trainer.compute_embeddings(train_loader)
        radii = []
        for c in labels.unique().tolist():
            mask = labels == c
            mean = emb[mask].mean(0)
            self.protos[c] = mean
            radii.append(torch.norm(emb[mask] - mean, dim=1).mean().item())
        if radii:
            self.radius = np.mean(radii)

    def get_state(self):
        return {'protos': deepcopy(self.protos), 'radius': self.radius}

    def load_state(self, s):
        self.protos = s.get('protos', {})
        self.radius = s.get('radius', 0.)
