"""V-LETO + Redundancy-Adaptive: same backbone as proto_evolve, but
(1) after each task, compute per-class single-party probe accuracy
    (= per-class redundancy_score = 1 - max_k probe_acc_k(c));
(2) for each newly-learned EASY class (high single-party-acc), boost the
    FIM-freeze fraction of its DOMINANT party by a small constant, so its
    specialization is preserved across subsequent tasks;
(3) stash redundancy_score and dominant_party on trainer.radapt_state so
    the matching UL method (radapt_router) can read them.

CL training itself is otherwise unchanged from proto_evolve — only FIM
modulation differs. This keeps CL safe while bridging CL→UL via shared
redundancy state.
"""
from __future__ import annotations
import torch, torch.nn as nn, numpy as np
from copy import deepcopy

from cl_methods.proto_evolve import ProtoEvolveCL
from data_utils import split_features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class ProtoEvolveRadaptCL(ProtoEvolveCL):
    """proto_evolve + redundancy-aware FIM modulation + per-party state for UL."""

    def __init__(self, trainer, args):
        super().__init__(trainer, args)
        self.name = 'ProtoEvolve_Radapt_VFL'
        # Per-class redundancy: max single-party probe acc (HIGH = easy = low redundancy)
        self.per_class_max_party_acc: dict = {}   # c -> max-party-acc (float in [0,1])
        self.dominant_party: dict = {}            # c -> int (argmax party for c)
        # FIM modulation knobs (overridable via args)
        self.radapt_easy_thresh = float(getattr(args, 'radapt_easy_thresh', 0.50))
        self.radapt_frac_boost_per_easy = float(getattr(args, 'radapt_frac_boost', 0.05))
        self.radapt_frac_boost_cap = float(getattr(args, 'radapt_frac_cap', 0.15))
        # Stash on trainer so UL can read without dependency injection
        if not hasattr(trainer, 'radapt_state'):
            trainer.radapt_state = {}
        trainer.radapt_state.update({
            'per_class_max_party_acc': self.per_class_max_party_acc,
            'dominant_party': self.dominant_party,
            'easy_thresh': self.radapt_easy_thresh,
        })

    # ------------------------------------------------------------------
    # Probe: per-class single-party probe acc on currently seen classes
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _extract_per_party_embeddings(self, loader):
        P = self.args.num_parties
        for b in self.trainer.bottoms: b.eval()
        embs = [[] for _ in range(P)]
        labels = []
        for bx, by in loader:
            bx = bx.to(self.args.device)
            parts = split_features(bx, self.args)
            for k in range(P):
                embs[k].append(self.trainer.bottoms[k](parts[k]).cpu().numpy())
            labels.append(by.numpy())
        return [np.concatenate(e) for e in embs], np.concatenate(labels)

    def _refresh_redundancy(self, dataset, seen_classes):
        """Train one LogReg per party on frozen embeddings, record per-class
        max-party-acc + dominant party. Cheap: <2 min on c100 even after task 4."""
        from torch.utils.data import DataLoader, Subset
        train_idx = dataset.get_class_indices(dataset.trainset, seen_classes)
        test_idx  = dataset.get_class_indices(dataset.testset,  seen_classes)
        bs = self.args.batch_size
        nw = self.args.num_workers
        tr_loader = DataLoader(Subset(dataset.trainset, train_idx), batch_size=bs,
                                shuffle=False, num_workers=nw)
        te_loader = DataLoader(Subset(dataset.testset,  test_idx),  batch_size=bs,
                                shuffle=False, num_workers=nw)
        Xtr, ytr = self._extract_per_party_embeddings(tr_loader)
        Xte, yte = self._extract_per_party_embeddings(te_loader)
        P = self.args.num_parties
        per_class_per_party_acc = {int(c): {} for c in seen_classes}
        for k in range(P):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000))
            clf.fit(Xtr[k], ytr)
            preds = clf.predict(Xte[k])
            for c in seen_classes:
                m = (yte == c)
                per_class_per_party_acc[int(c)][k] = (
                    float((preds[m] == yte[m]).mean()) if m.any() else 0.0)
        # Reduce: max acc and argmax party per class
        self.per_class_max_party_acc.clear()
        self.dominant_party.clear()
        for c in seen_classes:
            d = per_class_per_party_acc[int(c)]
            best_k = max(d, key=d.get)
            self.per_class_max_party_acc[int(c)] = float(d[best_k])
            self.dominant_party[int(c)] = int(best_k)
        # Re-publish onto trainer (dicts are same objects but be defensive)
        self.trainer.radapt_state['per_class_max_party_acc'] = dict(self.per_class_max_party_acc)
        self.trainer.radapt_state['dominant_party'] = dict(self.dominant_party)
        return per_class_per_party_acc

    # ------------------------------------------------------------------
    # FIM modulation: per-party frac boost for easy-class dominators
    # ------------------------------------------------------------------
    def _compute_fim(self, loader, task_id):
        """Same as parent but with per-party frac modulated by how many of THIS
        task's new classes are 'easy and dominated by this party'."""
        from cl_methods.proto_evolve import ProtoEvolveCL as _Base  # for clarity
        # We need access to the new classes learned in this task. The parent
        # has `self.global_protos` containing prev classes; new ones are those
        # in `loader` not already in global_protos. Cheap: scan loader once.
        seen_before = set(self.global_protos.keys()) | set(self.prev_protos.keys()) \
                      if hasattr(self, 'prev_protos') else set(self.global_protos.keys())
        new_classes = set()
        for _, by in loader:
            for c in by.unique().tolist():
                if c not in seen_before:
                    new_classes.add(int(c))
            if len(new_classes) >= 50:  # cap scan
                break

        # Determine per-party boost: count easy-class wins per party among NEW classes.
        # Note: redundancy for these NEW classes is unknown yet at FIM time
        # (probe runs after this method). Use OLD classes' dominant_party as proxy:
        # if a party historically dominates many easy classes, its specialization
        # bias is taken to extend to the new task as well.
        # If no history yet (task 0), use uniform behavior (parent).
        if not self.per_class_max_party_acc:
            return super()._compute_fim(loader, task_id)

        easy_dominations_per_party = [0 for _ in range(self.args.num_parties)]
        for c, acc in self.per_class_max_party_acc.items():
            if acc >= self.radapt_easy_thresh:
                k = self.dominant_party.get(int(c), -1)
                if 0 <= k < self.args.num_parties:
                    easy_dominations_per_party[k] += 1
        total_easy = max(1, sum(easy_dominations_per_party))
        base_frac = float(getattr(self.args, 'fim_freeze_frac', 0.25))

        # Inline-copy of parent _compute_fim with per-party frac modulation.
        # (Cleaner than monkey-patching torch.quantile inside parent.)
        for b in self.trainer.bottoms: b.train()
        self.trainer.top_model.train()
        fim = [{n: torch.zeros_like(p) for n, p in self.trainer.bottoms[k].named_parameters()}
               for k in range(self.args.num_parties)]
        nb = 0
        for bx, by in loader:
            bx, by = bx.to(self.args.device), by.to(self.args.device)
            for b in self.trainer.bottoms: b.zero_grad()
            self.trainer.top_model.zero_grad()
            parts = split_features(bx, self.args)
            embs = [self.trainer.bottoms[i](parts[i]) for i in range(self.args.num_parties)]
            out = self.trainer.top_model(self.trainer._aggregate(embs))
            nn.CrossEntropyLoss()(out, by).backward()
            for k in range(self.args.num_parties):
                for n, p in self.trainer.bottoms[k].named_parameters():
                    if p.grad is not None:
                        fim[k][n] += p.grad.data ** 2
            nb += 1

        for k in range(self.args.num_parties):
            for n in fim[k]:
                fim[k][n] /= max(nb, 1)
            imps = {n: fim[k][n].mean().item() for n in fim[k]}
            vals = torch.tensor(list(imps.values()))
            if vals.numel() == 0 or vals.sum() == 0:
                continue
            # Per-party frac: base + boost proportional to easy-class dominance
            dom_share = easy_dominations_per_party[k] / total_easy
            party_frac = min(
                base_frac + self.radapt_frac_boost_per_easy * easy_dominations_per_party[k],
                base_frac + self.radapt_frac_boost_cap,
            )
            kappa = torch.quantile(vals, max(0.0, 1.0 - party_frac)).item()
            nf, n_new = 0, 0
            for n in fim[k]:
                cur = imps[n] >= kappa
                prev = self.fim_masks[k].get(n, False)
                new_state = prev or cur
                if new_state and not prev:
                    n_new += 1
                self.fim_masks[k][n] = new_state
                if new_state: nf += 1
            print(f"    [radapt] Party {k}: frac={party_frac:.3f} (base={base_frac:.2f}, "
                  f"easy-dom-share={dom_share:.2f})  {nf}/{len(fim[k])} frozen "
                  f"(+{n_new} new this task)")

        for b in self.trainer.bottoms:
            b.zero_grad()
            for p in b.parameters(): p.requires_grad = True

    # ------------------------------------------------------------------
    # Hook the redundancy refresh AFTER everything else in after_task
    # ------------------------------------------------------------------
    def after_task(self, train_loader, task_id):
        super().after_task(train_loader, task_id)
        # Probe runs on ALL seen classes (cumulative). Dataset handle is on trainer.
        # We don't have direct dataset access here; the trainer passes it through.
        ds = getattr(self.trainer, 'dataset_ref', None)
        seen = sorted(self.global_protos.keys()) if self.global_protos else []
        if ds is None or not seen:
            print(f"  [radapt] skip probe (dataset={ds is not None}, seen={len(seen)})")
            return
        print(f"  [radapt] probing redundancy on {len(seen)} seen classes...")
        import time
        t0 = time.time()
        self._refresh_redundancy(ds, seen)
        n_easy = sum(1 for v in self.per_class_max_party_acc.values()
                      if v >= self.radapt_easy_thresh)
        print(f"  [radapt] probe done in {time.time()-t0:.1f}s. "
              f"{n_easy}/{len(seen)} classes flagged EASY (single-party-acc >= "
              f"{self.radapt_easy_thresh:.2f})")
