"""AdaGauss (NeurIPS 2024) adapted to VFL class-incremental learning.

Rypesc et al. "Divide and not forget: Ensemble of selectively trained
experts in Continual Learning" / "Adaptive Gaussian classifier for
exemplar-free CIL". The official implementation lives at
https://github.com/grypesc/AdaGauss (src/approach/ada_gauss.py).

Core ideas, ported to our split-learning VFL trainer:

  1. Per-class Gaussian (mu_c, Sigma_c) fit on the AGGREGATED 128-d
     embedding produced by summing party-bottom outputs. Shrinkage
     Sigma_c <- (1-alpha)*Sigma_c + alpha*I keeps the matrix
     invertible. Stored on CPU.

  2. Adapter MLP psi_t: R^128 -> R^128 (128 -> 256 -> 128 with ReLU)
     trained AFTER bottoms finish task t and BEFORE Gaussians are fit
     for task t. Target: psi(f_old) ~= f_new on the new task's images.
     The adapter then pushes all stored old Gaussians forward into the
     new feature space (sample-and-refit, K = adagauss_n_samples).

  3. Training-time loss for task > 0:
       L = L_CE + lambda_ac * L_AC + lambda_pkd * L_PKD
     - L_AC anti-collapse on the new aggregated batch covariance:
         L_AC = -mean(clamp(diag(chol(Sigma_batch + eps*I)), max=1))
       (official AdaGauss uses the same form on the backbone features.)
     - L_PKD projected distillation through the prior task's adapter:
         L_PKD = ||psi_prev(f_old) - f_new||^2
       At task 1 we have no prior adapter, so psi_prev = identity.

  4. Inference: Bayes / Mahalanobis classifier on the stored
     Gaussians. log p(x|c) = -0.5*(x-mu)^T Sigma^-1 (x-mu) - 0.5*log|Sigma|
     computed via Cholesky for numerical stability. We monkey-patch
     trainer.evaluate so the existing eval pipeline in main.py gets
     Bayes accuracy automatically. Falls back to the original linear
     softmax evaluate when no Gaussians are stored yet (task 0).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from data_utils import split_features


class _Adapter(nn.Module):
    """Two-layer MLP psi: R^d -> R^d (d -> 2d -> d with ReLU)."""

    def __init__(self, dim, hidden=None):
        super().__init__()
        h = hidden if hidden is not None else 2 * dim
        self.net = nn.Sequential(
            nn.Linear(dim, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, dim),
        )

    def forward(self, x):
        return self.net(x)


class AdaGaussCL:
    """AdaGauss adapted to VFL split-learning trainer."""

    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'AdaGauss_VFL'

        # Hyperparameters
        self.lambda_ac = getattr(args, 'adagauss_lambda_ac', 1.0)
        self.lambda_pkd = getattr(args, 'adagauss_lambda_pkd', 5.0)
        self.shrinkage = getattr(args, 'adagauss_shrinkage', 0.1)
        self.adapter_epochs = getattr(args, 'adagauss_adapter_epochs', 30)
        self.n_samples = getattr(args, 'adagauss_n_samples', 256)

        # Embedding dim from the top model's classifier
        self.dim = trainer.top_model.classifier.in_features

        # Gaussian memory: class id -> {'mean': (d,) cpu, 'cov': (d,d) cpu}
        self.gaussians = {}
        # Adapter trained at the end of each task (used as psi_prev next task)
        self.adapters = []
        # Frozen snapshot of bottoms at the START of the current task,
        # so KD and adapter training can compare old vs new features.
        self.old_bottoms = None

        # Monkey-patch evaluate so existing harness gets Bayes accuracy.
        self._orig_evaluate = trainer.evaluate
        trainer.evaluate = self._bayes_evaluate

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def before_task(self, task_id, new_classes, seen_classes):
        # Expand top head (kept around for the task-0 fallback evaluate)
        req = max(seen_classes) + 1 if seen_classes else 0
        self.trainer.top_model.expand_classes(req, self.args.device)

        # Snapshot old bottoms for KD reference and adapter targets
        if task_id > 0:
            self.old_bottoms = [deepcopy(b).eval() for b in self.trainer.bottoms]
            for ob in self.old_bottoms:
                for p in ob.parameters():
                    p.requires_grad = False
        else:
            self.old_bottoms = None

    def train_task(self, train_loader, task_id):
        extra = self._adagauss_loss if task_id > 0 else None
        history, elapsed = self.trainer.train_task(
            train_loader, self.args.epochs_per_task, extra_loss_fn=extra,
            grad_clip_norm=1.0,
        )
        return history, elapsed

    def after_task(self, train_loader, task_id):
        """Train new adapter, push old Gaussians forward, fit new Gaussians."""
        # 1. Train adapter psi_t: f_old -> f_new (skip for task 0; no old model)
        if task_id > 0 and self.old_bottoms is not None:
            adapter = self._train_adapter(train_loader)
            self.adapters.append(adapter)
            # 2. Push every stored old Gaussian forward through the new adapter
            self._adapt_old_gaussians(adapter)

        # 3. Fit Gaussians for the classes seen in this task
        self._fit_new_gaussians(train_loader)

        # Free the snapshot to keep memory bounded
        self.old_bottoms = None

    # ------------------------------------------------------------------
    # Training-time loss: L_CE + lambda_ac * L_AC + lambda_pkd * L_PKD
    # ------------------------------------------------------------------
    def _adagauss_loss(self, bottoms, top_model, batch_x, batch_y, loss_ce=None):
        device = self.args.device
        parts = split_features(batch_x, self.args)

        # Current aggregated embedding (with grad to bottoms)
        cur_embs = [bottoms[i](parts[i]) for i in range(len(bottoms))]
        cur_agg = self.trainer._aggregate(cur_embs)

        # Anti-collapse loss on the new batch covariance
        l_ac = self._loss_ac(cur_agg)

        # Projected feature distillation through the prior task's adapter
        l_pkd = torch.tensor(0.0, device=device)
        if self.old_bottoms is not None:
            with torch.no_grad():
                old_embs = [self.old_bottoms[i](parts[i]) for i in range(len(self.old_bottoms))]
                old_agg = self.trainer._aggregate(old_embs)
            if self.adapters:
                psi_prev = self.adapters[-1].to(device).eval()
                with torch.no_grad():
                    proj_old = psi_prev(old_agg)
            else:
                # First incremental task: no prior adapter, use identity
                proj_old = old_agg
            l_pkd = F.mse_loss(proj_old, cur_agg)

        extra = self.lambda_ac * l_ac + self.lambda_pkd * l_pkd
        if loss_ce is not None:
            return loss_ce + extra
        return extra

    def _loss_ac(self, features):
        """Anti-collapse loss on a batch of features.

        Mirrors AdaGauss `loss_ac`:
          loss = -mean(clamp(diag(chol(cov(features))), max=1))
        We add a small jitter (1e-4 * I) before Cholesky for safety; the
        official code relies on batch size > dim to keep cov PD, which is
        not always true in our VFL split-learning batches.
        """
        if features.size(0) < 2:
            return torch.tensor(0.0, device=features.device)
        d = features.size(1)
        # CUDA float32 Cholesky may report success yet return non-finite
        # factors for this low-rank covariance (batch size < embedding dim).
        # Factor in float64, as the Bayes evaluator already does.
        working = features.to(torch.float64)
        cov = torch.cov(working.t())
        eye = torch.eye(d, device=features.device, dtype=working.dtype)
        cov = cov + 1e-4 * eye
        try:
            L = torch.linalg.cholesky(cov)
        except RuntimeError:
            # Cov still not PD (rare); fall back to no AC term this batch.
            return torch.tensor(0.0, device=features.device)
        diag = torch.diagonal(L)
        return -torch.clamp(diag, max=1.0).mean().to(features.dtype)

    # ------------------------------------------------------------------
    # Adapter: psi(f_old) ~= f_new on the new task's images
    # ------------------------------------------------------------------
    def _train_adapter(self, train_loader):
        device = self.args.device
        adapter = _Adapter(self.dim).to(device)
        opt = torch.optim.Adam(adapter.parameters(), lr=1e-3)

        # Freeze both bottoms (current and old) for the adapter run
        for b in self.trainer.bottoms:
            b.eval()

        for ep in range(self.adapter_epochs):
            adapter.train()
            ep_loss, n = 0.0, 0
            for bx, _ in train_loader:
                bx = bx.to(device)
                parts = split_features(bx, self.args)
                with torch.no_grad():
                    old_embs = [self.old_bottoms[i](parts[i]) for i in range(len(self.old_bottoms))]
                    f_old = self.trainer._aggregate(old_embs)
                    new_embs = [self.trainer.bottoms[i](parts[i]) for i in range(len(self.trainer.bottoms))]
                    f_new = self.trainer._aggregate(new_embs)
                opt.zero_grad()
                loss = F.mse_loss(adapter(f_old), f_new)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        'non-finite AdaGauss adapter loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    adapter.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                ep_loss += loss.item() * bx.size(0)
                n += bx.size(0)
            if (ep + 1) % 10 == 0:
                print(f"    AdaGauss adapter epoch {ep + 1}/{self.adapter_epochs} mse={ep_loss / max(n, 1):.5f}")
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad = False
        return adapter

    @torch.no_grad()
    def _adapt_old_gaussians(self, adapter):
        """Push every stored old Gaussian forward through the new adapter.

        Sample K points from N(mu_c, Sigma_c), map through psi, refit
        Gaussian. This propagates the old class distributions into the
        new feature space (semantic-drift correction).
        """
        device = self.args.device
        adapter = adapter.to(device).eval()
        K = self.n_samples
        d = self.dim
        for c, g in list(self.gaussians.items()):
            dtype = next(adapter.parameters(), torch.empty(
                0, dtype=g['mean'].dtype)).dtype
            mu = g['mean'].to(device=device, dtype=torch.float64)
            cov = g['cov'].to(device=device, dtype=torch.float64)
            cov = (cov + cov.t()) / 2
            # Cholesky-based sampling: x = mu + L @ z, z ~ N(0, I)
            try:
                L = torch.linalg.cholesky(cov)
            except RuntimeError:
                # Repair legacy/floating-point drift with the official
                # scale-aware diagonal shrinkage before retrying.
                cov = self._shrink(cov)
                L = torch.linalg.cholesky(cov)
            z = torch.randn(K, d, device=device, dtype=torch.float64)
            samples = mu.unsqueeze(0) + z @ L.t()
            adapted = adapter(samples.to(dtype))
            new_mu = adapted.mean(0)
            new_cov = torch.cov(adapted.t())
            new_cov = self._shrink(new_cov)
            self.gaussians[c] = {'mean': new_mu.cpu(), 'cov': new_cov.cpu()}

    # ------------------------------------------------------------------
    # Gaussian fitting on aggregated embeddings
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _fit_new_gaussians(self, train_loader):
        device = self.args.device
        for b in self.trainer.bottoms:
            b.eval()
        class_feats = {}
        for bx, by in train_loader:
            bx = bx.to(device)
            parts = split_features(bx, self.args)
            embs = [self.trainer.bottoms[i](parts[i]) for i in range(len(self.trainer.bottoms))]
            agg = self.trainer._aggregate(embs)
            for i in range(len(by)):
                c = int(by[i].item())
                class_feats.setdefault(c, []).append(agg[i].cpu())

        d = self.dim
        for c, feats in class_feats.items():
            X = torch.stack(feats, dim=0)  # (N, d)
            if X.size(0) < 2:
                # Singleton class: tiny isotropic cov so log p still defined
                mu = X.mean(0)
                cov = torch.eye(d)
            else:
                mu = X.mean(0)
                cov = torch.cov(X.t())
                cov = self._shrink(cov.to(device)).cpu()
            self.gaussians[c] = {'mean': mu, 'cov': cov}
        print(f"    AdaGauss: stored Gaussians for {sorted(class_feats.keys())} (total={len(self.gaussians)})")

    def _shrink(self, cov):
        """Official AdaGauss scale-aware diagonal shrinkage."""
        original_dtype = cov.dtype
        cov = cov.to(torch.float64)
        cov = (cov + cov.t()) / 2
        diagonal_mean = torch.diagonal(cov).mean()
        if (not torch.isfinite(diagonal_mean).item()
                or diagonal_mean <= 0):
            raise FloatingPointError(
                'AdaGauss covariance diagonal mean must be positive finite')
        eye = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
        return (cov + self.shrinkage * diagonal_mean * eye).to(
            original_dtype)

    # ------------------------------------------------------------------
    # Bayes / Mahalanobis evaluate (monkey-patched onto trainer)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _bayes_evaluate(self, test_loader):
        # Fall back to original linear head if no Gaussians yet (task 0 eval)
        if not self.gaussians:
            return self._orig_evaluate(test_loader)

        device = self.args.device
        d = self.dim
        for b in self.trainer.bottoms:
            b.eval()
        self.trainer.top_model.eval()

        # Precompute per-class Cholesky factors and log|Sigma|
        class_ids = sorted(self.gaussians.keys())
        mus, Ls, logdets = [], [], []
        for c in class_ids:
            g = self.gaussians[c]
            # FP32 Cholesky can reject a positive, ill-conditioned covariance.
            mu = g['mean'].to(device=device, dtype=torch.float64)
            cov = g['cov'].to(device=device, dtype=torch.float64)
            try:
                L = torch.linalg.cholesky(cov)
            except RuntimeError:
                L = torch.linalg.cholesky(cov + 1e-4 * torch.eye(d, device=device, dtype=cov.dtype))
            logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
            mus.append(mu)
            Ls.append(L)
            logdets.append(logdet)
        mus = torch.stack(mus, dim=0)            # (C, d)
        Ls = torch.stack(Ls, dim=0)               # (C, d, d)
        logdets = torch.stack(logdets, dim=0)     # (C,)

        # id -> position in class_ids
        id_to_pos = {c: i for i, c in enumerate(class_ids)}

        correct, total = 0, 0
        all_probs, all_labels = [], []
        for bx, by in test_loader:
            bx = bx.to(device)
            by_dev = by.to(device)
            parts = split_features(bx, self.args)
            embs = [self.trainer.bottoms[i](parts[i]) for i in range(len(self.trainer.bottoms))]
            agg = self.trainer._aggregate(embs).to(torch.float64)  # (B, d)
            B = agg.size(0)
            C = mus.size(0)

            # For each class c: solve L_c @ y = (x - mu_c) -> y, mahalanobis = ||y||^2
            # Stack: diff has shape (C, B, d); Ls is (C, d, d)
            diff = agg.unsqueeze(0) - mus.unsqueeze(1)             # (C, B, d)
            rhs = diff.transpose(1, 2)                              # (C, d, B)
            y = torch.linalg.solve_triangular(Ls, rhs, upper=False)  # (C, d, B)
            mahal = (y ** 2).sum(dim=1)                              # (C, B)
            log_p = -0.5 * mahal - 0.5 * logdets.unsqueeze(1)        # (C, B)
            log_p = log_p.t()                                        # (B, C)

            # Argmax over class positions -> map back to true class ids
            pred_pos = log_p.argmax(dim=1)                           # (B,)
            preds = torch.tensor([class_ids[p.item()] for p in pred_pos],
                                  device=device, dtype=by_dev.dtype)
            correct += (preds == by_dev).sum().item()
            total += B

            # Probs over the known-class set (uniform prior -> softmax(log_p))
            probs_known = torch.softmax(log_p, dim=1).cpu()
            # Expand to full num_classes vector so downstream code (which may
            # index by raw class id) sees zeros for unknown classes.
            full = torch.zeros(B, self.args.num_classes)
            for j, c in enumerate(class_ids):
                if 0 <= c < self.args.num_classes:
                    full[:, c] = probs_known[:, j]
            all_probs.append(full)
            all_labels.append(by.cpu())

        acc = correct / max(total, 1)
        probs = torch.cat(all_probs) if all_probs else torch.tensor([])
        labels = torch.cat(all_labels) if all_labels else torch.tensor([])
        return acc, probs, labels

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def get_state(self):
        return {
            'gaussians': {c: {'mean': g['mean'].clone(), 'cov': g['cov'].clone()}
                          for c, g in self.gaussians.items()},
            'adapters': [deepcopy(a.state_dict()) for a in self.adapters],
        }

    def load_state(self, s):
        self.gaussians = {c: {'mean': g['mean'].clone(), 'cov': g['cov'].clone()}
                          for c, g in s.get('gaussians', {}).items()}
        sd_list = s.get('adapters', [])
        self.adapters = []
        for sd in sd_list:
            a = _Adapter(self.dim).to(self.args.device)
            a.load_state_dict(sd)
            a.eval()
            for p in a.parameters():
                p.requires_grad = False
            self.adapters.append(a)
