"""TARGET+FIM adapted to VFL.
Based on: Zhang et al. (CVPR 2024) "TARGET: Federated Class-Continual Learning via Exemplar-Free Distillation"

VFL adaptation:
- Server trains conditional generators on aggregated embeddings
- FIM-based bottom freeze (required for VFL effectiveness)
- KD from old top model + synthetic replay from generators
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from data_utils import split_features


class ConditionalGenerator(nn.Module):
    """G(class_label, noise) -> embedding."""
    def __init__(self, num_classes, embed_dim=128, noise_dim=32, hidden=256):
        super().__init__()
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.class_emb = nn.Embedding(num_classes, 64)
        self.net = nn.Sequential(
            nn.Linear(64 + noise_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, local_labels, noise=None):
        bs = local_labels.size(0)
        if noise is None:
            noise = torch.randn(bs, self.noise_dim, device=local_labels.device)
        cls_emb = self.class_emb(local_labels)
        return self.net(torch.cat([cls_emb, noise], dim=1))


class TARGETCL:
    """TARGET+FIM for VFL."""

    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'TARGET_VFL'
        self.generators = {}
        self.task_classes = {}
        self.old_top = None
        # classes banned from synthetic replay after an unlearning event
        # (set by cl_methods.sanitize; ids stay in task_classes because they
        # index the generator's conditioning table)
        self.forgotten = set()
        self.alpha_kd = 0.5
        self.alpha_syn = 0.5
        self.temperature = 2.0
        self.generator_epochs = 50
        self.generator_lr = 1e-3
        self.replay_noise_std = 1.0
        # FIM (required for VFL CL to work)
        self.fim_masks = [{} for _ in range(args.num_parties)]
        self.k0 = 15
        self.fim_alpha = 3

    def before_task(self, task_id, new_classes, seen_classes):
        req = max(seen_classes) + 1 if seen_classes else 0
        self.trainer.top_model.expand_classes(req, self.args.device)
        if task_id > 0:
            # Snapshot old top model
            self.old_top = deepcopy(self.trainer.top_model).eval()
            for p in self.old_top.parameters():
                p.requires_grad = False
            # Apply FIM freeze
            total_f, total_p = 0, 0
            for k in range(self.args.num_parties):
                for n, p in self.trainer.bottoms[k].named_parameters():
                    total_p += 1
                    if self.fim_masks[k].get(n, False):
                        p.requires_grad = False
                        total_f += 1
            print(f"  TARGET+FIM freeze: {total_f}/{total_p} bottom params frozen")

    def _target_loss(self, bottoms, top_model, batch_x, batch_y, loss_ce=None):
        device = self.args.device
        loss = loss_ce if loss_ce is not None else torch.tensor(0.0, device=device)

        if not self.generators:
            return loss

        # KD loss
        if self.old_top is not None:
            parts = split_features(batch_x, self.args)
            curr_embs = [bottoms[i](parts[i]) for i in range(len(bottoms))]
            curr_agg = self.trainer._aggregate(curr_embs)
            with torch.no_grad():
                old_logits = self.old_top(curr_agg)
            curr_logits = top_model(curr_agg)
            T = self.temperature
            n_old = old_logits.size(1)
            if curr_logits.size(1) >= n_old:
                old_probs = F.softmax(old_logits / T, dim=1)
                curr_log_probs = F.log_softmax(curr_logits[:, :n_old] / T, dim=1)
                kd_loss = F.kl_div(curr_log_probs, old_probs, reduction='batchmean') * (T * T)
                loss = loss + self.alpha_kd * kd_loss

        # Synthetic replay
        bs = batch_y.size(0)
        all_old_classes = []
        for task_cls in self.task_classes.values():
            all_old_classes.extend(c for c in task_cls if c not in self.forgotten)

        if all_old_classes:
            sampled = np.random.choice(all_old_classes, size=bs, replace=True)
            task_groups = {}
            for gc in sampled:
                gc = int(gc)
                for tid, cls in self.task_classes.items():
                    if gc in cls:
                        task_groups.setdefault(tid, []).append(gc)
                        break

            synth_embs, synth_labels = [], []
            for tid, gc_list in task_groups.items():
                gen = self.generators[tid]
                gen.eval()
                cls = self.task_classes[tid]
                local = torch.tensor([cls.index(gc) for gc in gc_list],
                                     dtype=torch.long, device=device)
                with torch.no_grad():
                    noise = torch.randn(len(gc_list), gen.noise_dim,
                                        device=device) * self.replay_noise_std
                synth_embs.append(gen(local, noise))
                synth_labels.extend(gc_list)

            if synth_embs:
                se = torch.cat(synth_embs, dim=0)
                sl = torch.tensor(synth_labels, dtype=torch.long, device=device)
                syn_loss = F.cross_entropy(top_model(se), sl)
                loss = loss + self.alpha_syn * syn_loss

        return loss

    def train_task(self, train_loader, task_id):
        extra = self._target_loss if task_id > 0 else None
        history, elapsed = self.trainer.train_task(
            train_loader, self.args.epochs_per_task, extra_loss_fn=extra)
        # Unfreeze for FIM computation
        for b in self.trainer.bottoms:
            for p in b.parameters():
                p.requires_grad = True
        return history, elapsed

    def _compute_fim(self, loader, task_id):
        for b in self.trainer.bottoms:
            b.train()
        self.trainer.top_model.train()
        fim = [{n: torch.zeros_like(p) for n, p in self.trainer.bottoms[k].named_parameters()}
               for k in range(self.args.num_parties)]
        nb = 0
        for bx, by in loader:
            bx, by = bx.to(self.args.device), by.to(self.args.device)
            for b in self.trainer.bottoms:
                b.zero_grad()
            self.trainer.top_model.zero_grad()
            parts = split_features(bx, self.args)
            embs = [self.trainer.bottoms[i](parts[i])
                    for i in range(self.args.num_parties)]
            agg = self.trainer._aggregate(embs)
            nn.CrossEntropyLoss()(self.trainer.top_model(agg), by).backward()
            for k in range(self.args.num_parties):
                for n, p in self.trainer.bottoms[k].named_parameters():
                    if p.grad is not None:
                        fim[k][n] += p.grad.data ** 2
            nb += 1

        delta = self.k0 + self.fim_alpha * np.log(task_id + 2)
        for k in range(self.args.num_parties):
            for n in fim[k]:
                fim[k][n] /= max(nb, 1)
            vals = torch.cat([v.flatten() for v in fim[k].values()])
            if vals.sum() == 0:
                continue
            kappa = vals.mean().item() - delta * vals.std().item()
            n_new = 0
            for n in fim[k]:
                new_imp = bool(fim[k][n].mean().item() >= kappa)
                was = self.fim_masks[k].get(n, False)
                self.fim_masks[k][n] = was or new_imp
                if new_imp and not was:
                    n_new += 1
            total = sum(1 for v in self.fim_masks[k].values() if v)
            print(f"    Party {k}: {total}/{len(fim[k])} frozen (+{n_new} new)")

        for b in self.trainer.bottoms:
            b.zero_grad()
            for p in b.parameters():
                p.requires_grad = True
        self.trainer.top_model.zero_grad()

    def after_task(self, train_loader, task_id):
        # 1) FIM first (need fresh grads from training data)
        self._compute_fim(train_loader, task_id)

        # 2) Collect embeddings (bottoms now stable for this task)
        for b in self.trainer.bottoms:
            b.eval()
        all_embs, all_lbls = [], []
        with torch.no_grad():
            for bx, by in train_loader:
                bx = bx.to(self.args.device)
                parts = split_features(bx, self.args)
                embs = [self.trainer.bottoms[i](parts[i])
                        for i in range(len(self.trainer.bottoms))]
                agg = self.trainer._aggregate(embs)
                all_embs.append(agg.cpu())
                all_lbls.append(by)

        all_embs = torch.cat(all_embs)
        all_lbls = torch.cat(all_lbls)
        task_classes = sorted(set(all_lbls.tolist()))
        self.task_classes[task_id] = task_classes
        n_cls = len(task_classes)
        embed_dim = all_embs.size(1)
        cls_to_local = {c: i for i, c in enumerate(task_classes)}
        local_lbls = torch.tensor(
            [cls_to_local[l.item()] for l in all_lbls], dtype=torch.long)

        # 3) Train generator
        gen = ConditionalGenerator(num_classes=n_cls, embed_dim=embed_dim).to(self.args.device)
        opt = torch.optim.Adam(gen.parameters(), lr=self.generator_lr)
        n = all_embs.size(0)
        bs_g = min(64, n)
        print(f"  TARGET: train gen task {task_id}, {n} samples, {n_cls} classes")

        for ep in range(self.generator_epochs):
            perm = torch.randperm(n)
            tl, nb_g = 0., 0
            for i in range(0, n, bs_g):
                idx = perm[i:i + bs_g]
                real = all_embs[idx].to(self.args.device)
                lbls = local_lbls[idx].to(self.args.device)
                fake = gen(lbls)
                recon = F.mse_loss(fake, real)
                real_std = real.std(dim=0)
                fake_std = fake.std(dim=0)
                div = F.mse_loss(fake_std, real_std)
                loss = recon + 0.1 * div
                opt.zero_grad()
                loss.backward()
                opt.step()
                tl += loss.item()
                nb_g += 1
            if (ep + 1) % 20 == 0:
                print(f"    Gen T{task_id} ep{ep+1}: loss={tl/max(nb_g,1):.4f}")

        gen.eval()
        self.generators[task_id] = gen

    def get_state(self):
        return {
            'task_classes': deepcopy(self.task_classes),
            'forgotten': sorted(self.forgotten),
            'fim_masks': deepcopy(self.fim_masks),
            'generators': {
                tid: {
                    'num_classes': gen.num_classes,
                    'embed_dim': gen.embed_dim,
                    'noise_dim': gen.noise_dim,
                    'hidden': gen.hidden,
                    'state_dict': {name: value.detach().cpu().clone()
                                   for name, value in gen.state_dict().items()},
                } for tid, gen in self.generators.items()
            },
        }

    def load_state(self, s):
        if not isinstance(s, dict):
            raise ValueError('TARGET state must be a mapping')
        # Legacy state is safe only before any generator-dependent continuation.
        if (not getattr(self.args, 'formal_deferred_evaluation', False)
                and (not s or (set(s) == {'task_classes'}
                               and isinstance(s['task_classes'], dict)
                               and not s['task_classes']))):
            s = {'task_classes': {}, 'forgotten': [], 'generators': {},
                 'fim_masks': [{} for _ in range(self.args.num_parties)]}
        if set(s) != {'task_classes', 'forgotten', 'fim_masks', 'generators'}:
            raise ValueError('TARGET continuation state keys are invalid')

        task_classes, records = s['task_classes'], s['generators']
        if (not isinstance(task_classes, dict) or not isinstance(records, dict)
                or any(type(tid) is not int or tid < 0
                       for tid in list(task_classes) + list(records))
                or set(task_classes) != set(records)):
            raise ValueError('TARGET task/generator membership is invalid')
        all_classes = set()
        for classes in task_classes.values():
            if (not isinstance(classes, list) or not classes
                    or any(type(c) is not int or c < 0
                           or c >= self.args.num_classes for c in classes)
                    or len(set(classes)) != len(classes)
                    or all_classes.intersection(classes)):
                raise ValueError('TARGET task classes are invalid')
            all_classes.update(classes)
        forgotten = s['forgotten']
        if (not isinstance(forgotten, list)
                or any(type(c) is not int for c in forgotten)
                or len(set(forgotten)) != len(forgotten)
                or not set(forgotten).issubset(all_classes)):
            raise ValueError('TARGET forgotten classes are invalid')
        masks = s['fim_masks']
        if not isinstance(masks, list) or len(masks) != self.args.num_parties:
            raise ValueError('TARGET FIM mask party count is invalid')
        for k, mask in enumerate(masks):
            if (not isinstance(mask, dict)
                    or not set(mask).issubset(dict(self.trainer.bottoms[k].named_parameters()))
                    or any(type(value) is not bool for value in mask.values())):
                raise ValueError('TARGET FIM mask names/values are invalid')

        generators = {}
        metadata_keys = {'num_classes', 'embed_dim', 'noise_dim', 'hidden'}
        top = self.trainer.top_model
        classifier = getattr(top, 'classifier', top)
        for tid, record in records.items():
            if (not isinstance(record, dict)
                    or set(record) != metadata_keys | {'state_dict'}
                    or any(type(record[key]) is not int or record[key] <= 0
                           for key in metadata_keys)
                    or record['num_classes'] != len(task_classes[tid])):
                raise ValueError('TARGET generator metadata is invalid')
            if (not isinstance(classifier, nn.Linear)
                    or record['embed_dim'] != classifier.in_features):
                raise ValueError('TARGET generator embedding width is incompatible with top model')
            gen = ConditionalGenerator(**{key: record[key] for key in metadata_keys})
            weights = record['state_dict']
            expected = gen.state_dict()
            if (not isinstance(weights, dict) or set(weights) != set(expected)
                    or any(not isinstance(weights[name], torch.Tensor)
                           or weights[name].layout != torch.strided
                           or weights[name].shape != value.shape
                           or weights[name].dtype != value.dtype
                           or not torch.isfinite(weights[name]).all().item()
                           for name, value in expected.items())):
                raise ValueError('TARGET generator state dict is invalid')
            gen.load_state_dict(weights, strict=True)
            generators[tid] = gen.to(self.args.device).eval().requires_grad_(False)

        # Commit only after the entire payload has validated and reconstructed.
        self.task_classes = deepcopy(task_classes)
        self.forgotten = set(forgotten)
        self.fim_masks = deepcopy(masks)
        self.generators = generators
