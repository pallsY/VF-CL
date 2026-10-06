"""FedSpace+VFL: prototype + representation loss."""
import torch, torch.nn as nn, numpy as np
from copy import deepcopy
from data_utils import split_features

class ProtoFedSpaceCL:
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'ProtoFedSpace_VFL'
        self.protos = {}
        self.radius = 0.0
        self.old_bottoms = None

    def before_task(self, task_id, new_classes, seen_classes):
        req = max(seen_classes)+1 if seen_classes else 0
        self.trainer.top_model.expand_classes(req, self.args.device)
        if task_id > 0:
            self.old_bottoms = [deepcopy(b).eval() for b in self.trainer.bottoms]
            for ob in self.old_bottoms:
                for p in ob.parameters(): p.requires_grad = False

    def _proto_repr_loss(self, bottoms, top_model, batch_x, batch_y, loss_ce=None):
        device = self.args.device
        loss = torch.tensor(0.0, device=device)
        if not self.protos: return loss_ce if loss_ce is not None else loss
        # Proto loss
        pc = list(self.protos.keys()); bs = batch_y.size(0)
        pe, pl = [], []
        for _ in range(bs):
            c = pc[np.random.randint(len(pc))]
            pe.append(self.protos[c].to(device) + torch.randn(self.protos[c].shape, device=device)*self.radius)
            pl.append(c)
        pe = torch.stack(pe).to(device)
        pl = torch.tensor(pl, dtype=torch.long, device=device)
        loss = self.args.proto_aug_weight * nn.CrossEntropyLoss()(top_model(pe), pl)
        # Repr loss
        if self.old_bottoms:
            parts = split_features(batch_x, self.args)
            repr_loss = 0.
            for i in range(len(bottoms)):
                curr = bottoms[i](parts[i])
                with torch.no_grad(): old = self.old_bottoms[i](parts[i])
                repr_loss += nn.MSELoss()(curr, old)
            loss += self.args.repr_loss_weight * repr_loss / len(bottoms)
        return (loss_ce + loss) if loss_ce is not None else loss

    def train_task(self, train_loader, task_id):
        extra = self._proto_repr_loss if task_id > 0 else None
        return self.trainer.train_task(train_loader, self.args.epochs_per_task, extra_loss_fn=extra)

    def after_task(self, train_loader, task_id):
        emb, labels = self.trainer.compute_embeddings(train_loader)
        radii = []
        for c in labels.unique().tolist():
            mask = labels == c
            mean = emb[mask].mean(0)
            self.protos[c] = mean
            radii.append(torch.norm(emb[mask]-mean, dim=1).mean().item())
        if radii: self.radius = np.mean(radii)

    def get_state(self):
        return {
            'protos': {c: proto.detach().cpu().clone()
                       for c, proto in self.protos.items()},
            'radius': self._normalized_radius(self.radius),
        }

    @staticmethod
    def _normalized_radius(radius):
        if (isinstance(radius, (bool, np.bool_))
                or not isinstance(radius, (int, float, np.integer, np.floating))):
            raise ValueError('ProtoFedSpace radius is invalid')
        try:
            normalized = float(radius)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError('ProtoFedSpace radius is invalid') from error
        if (not np.isfinite(normalized) or normalized < 0
                or (isinstance(radius, (int, np.integer))
                    and int(normalized) != int(radius))
                or (isinstance(radius, (float, np.floating))
                    and normalized != radius)):
            raise ValueError('ProtoFedSpace radius is invalid')
        return normalized

    def load_state(self, s):
        if not isinstance(s, dict):
            raise ValueError('ProtoFedSpace state must be a mapping')
        # A legacy empty payload is safe only on a pristine non-formal receiver.
        legacy_empty = not s or (set(s) == {'protos'} and s['protos'] == {})
        if legacy_empty:
            if (getattr(self.args, 'formal_deferred_evaluation', False)
                    or self.protos or self.radius != 0.0
                    or self.old_bottoms is not None):
                raise ValueError('ProtoFedSpace legacy continuation state is unsafe')
            s = {'protos': {}, 'radius': 0.0}
        if set(s) != {'protos', 'radius'}:
            raise ValueError('ProtoFedSpace continuation state keys are invalid')

        protos = s['protos']
        if not isinstance(protos, dict):
            raise ValueError('ProtoFedSpace prototypes must be a mapping')
        top = self.trainer.top_model
        classifier = getattr(top, 'classifier', top)
        if not isinstance(classifier, nn.Linear):
            raise ValueError('ProtoFedSpace top model is incompatible with prototypes')
        embed_dim = classifier.in_features
        if any(type(class_id) is not int or class_id < 0
               or class_id >= self.args.num_classes for class_id in protos):
            raise ValueError('ProtoFedSpace prototype class is invalid')
        if any(not isinstance(proto, torch.Tensor)
               or proto.layout != torch.strided
               or proto.ndim != 1
               or proto.numel() != embed_dim
               or not torch.is_floating_point(proto)
               or not torch.isfinite(proto).all().item()
               for proto in protos.values()):
            raise ValueError('ProtoFedSpace prototype tensor is invalid')
        if any(proto.dtype != classifier.weight.dtype for proto in protos.values()):
            raise ValueError('ProtoFedSpace prototype dtype is incompatible')
        radius = self._normalized_radius(s['radius'])

        # Commit only after the complete payload has validated.
        self.protos = {c: proto.detach().cpu().clone()
                       for c, proto in protos.items()}
        self.radius = radius
