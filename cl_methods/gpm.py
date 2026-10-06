"""GPM (Gradient Projection Memory) adapted to VFL.

Saha et al., ICLR 2021. After each task, collect bottom-layer input
activations on a small representative batch, take SVD to extract the
subspace spanning past tasks' representations, and during subsequent
tasks project bottom-weight gradients into the orthogonal complement
of that subspace.

Why this fits VFL: each party owns its own bottom model and a private
slice of the input. SVD is computed locally per party, projection is
applied locally via parameter backward hooks. No labels and no raw
data leave the party. The top model is left unconstrained.

State per party (self.feature_lists[k]): dict mapping
  weight_param_name -> U tensor of shape (d_in, r)
where d_in is the layer's input dim (kH*kW*C_in for Conv2d, in_features
for Linear) and r is the number of singular vectors retained so that
cumulative energy meets self.threshold.

Projection during training (registered as a backward hook on each
weight parameter):
  M = U @ U.t()                                    # (d_in, d_in)
  flat = grad.view(C_out, -1)                      # (C_out, d_in)
  grad_proj = flat - flat @ M                      # remove old subspace
  return grad_proj.view_as(grad)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from data_utils import split_features


def _stable_svd(matrix):
    if matrix.ndim != 2:
        raise ValueError("SVD matrix must be two-dimensional")
    if not torch.isfinite(matrix).all():
        raise ValueError("SVD matrix must be finite")
    try:
        return torch.linalg.svd(matrix, full_matrices=False)
    except torch._C._LinAlgError as float_error:
        if matrix.dtype == torch.float64:
            raise FloatingPointError("float64 SVD failed to converge") from float_error
        try:
            factors = torch.linalg.svd(matrix.double(), full_matrices=False)
        except torch._C._LinAlgError as double_error:
            raise FloatingPointError("float64 SVD failed to converge") from double_error
        return tuple(value.to(matrix.dtype) for value in factors)


class GPMCL:
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'GPM_VFL'
        self.threshold = getattr(args, 'gpm_threshold', 0.95)
        self.n_samples = 125  # samples for SVD memory construction
        # Per party: weight-param-name -> U tensor (d_in, r), kept on CPU
        self.feature_lists = [{} for _ in range(args.num_parties)]
        # Head subspace: U tensor (embed_dim, r) protecting the classifier
        # against logit drift across CIL tasks. Without this, GPM (a task-IL
        # method) catastrophically forgets in class-IL because the server-side
        # head is outside the per-party bottom-projection.
        self.head_basis = None
        # Active hook handles, refreshed every task
        self._hooks = []

    # ---------- hook helpers ----------
    def _clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _make_proj_hook(self, mat):
        """Return a grad hook that projects out the column space of `mat`.
        `mat` has shape (d_in, d_in); grad has leading dim C_out and any
        trailing structure that flattens to d_in."""
        def hook(grad):
            orig_shape = grad.shape
            sz = orig_shape[0]
            flat = grad.reshape(sz, -1)
            if flat.size(1) != mat.size(0):
                return grad  # shape mismatch (shouldn't happen), skip
            proj = flat - flat @ mat
            return proj.reshape(orig_shape)
        return hook

    def before_task(self, task_id, new_classes, seen_classes):
        # Expand top model for new classes
        req = max(seen_classes) + 1 if seen_classes else 0
        self.trainer.top_model.expand_classes(req, self.args.device)

        # Refresh hooks
        self._clear_hooks()
        if task_id == 0:
            return  # nothing to project yet

        for k in range(self.args.num_parties):
            for n, p in self.trainer.bottoms[k].named_parameters():
                if 'weight' not in n or p.dim() < 2:
                    continue  # skip biases and 1-D (BN/scale) params
                if n not in self.feature_lists[k]:
                    continue
                U = self.feature_lists[k][n].to(self.args.device)
                M = U @ U.t()
                self._hooks.append(p.register_hook(self._make_proj_hook(M)))

        # Head protection: project classifier-weight grad out of the subspace
        # spanned by past tasks' aggregated embeddings, so old-class rows
        # don't drift on the embedding directions the model already learned.
        if self.head_basis is not None:
            U_h = self.head_basis.to(self.args.device)
            M_h = U_h @ U_h.t()
            head_w = self.trainer.top_model.classifier.weight
            self._hooks.append(head_w.register_hook(self._make_proj_hook(M_h)))
            head_rank = self.head_basis.size(1)
        else:
            head_rank = 0
        print(f"  GPM: installed {len(self._hooks)} hooks (task {task_id}, head_rank={head_rank})")

    def train_task(self, train_loader, task_id):
        # No extra loss term; projection is via hooks on the bottoms
        return self.trainer.train_task(
            train_loader, self.args.epochs_per_task, extra_loss_fn=None
        )

    # ---------- after-task: extract activations and update memory ----------
    def _capture_layer_inputs(self, bottom, batch_x):
        """Forward pass capturing each Conv2d/Linear's input activation.
        Returns dict keyed by module-name -> 2D numpy array (d_in, N_flat)."""
        captured = {}

        def make_fh(name):
            def fn(module, inp, out):
                x = inp[0].detach()
                if isinstance(module, nn.Conv2d):
                    kH, kW = module.kernel_size
                    sH, sW = module.stride
                    pH, pW = module.padding
                    unf = F.unfold(x, (kH, kW), stride=(sH, sW), padding=(pH, pW))
                    # (B, C*kH*kW, L) -> (C*kH*kW, B*L)
                    mat = unf.permute(1, 0, 2).reshape(unf.size(1), -1)
                else:  # Linear
                    mat = x.view(x.size(0), -1).t()  # (d_in, B)
                captured[name] = mat.cpu().numpy()
            return fn

        handles = []
        for n, m in bottom.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                handles.append(m.register_forward_hook(make_fh(n)))
        was_training = bottom.training
        bottom.eval()
        with torch.no_grad():
            bottom(batch_x)
        for h in handles:
            h.remove()
        if was_training:
            bottom.train()
        return captured

    def after_task(self, train_loader, task_id):
        """Update each party's feature_list via SVD on collected activations."""
        device = self.args.device
        # Use the first batch's first n_samples as representative inputs
        for bx, _ in train_loader:
            break
        bx = bx[: self.n_samples].to(device)
        parts = split_features(bx, self.args)

        for k in range(self.args.num_parties):
            activations = self._capture_layer_inputs(self.trainer.bottoms[k], parts[k])
            for mod_name, act_mat in activations.items():
                param_name = (mod_name + '.weight') if mod_name else 'weight'
                act = torch.from_numpy(act_mat).float()  # (d_in, N)

                if param_name not in self.feature_lists[k]:
                    # First time seeing this layer: SVD raw activation
                    U, S, _ = _stable_svd(act)
                    sval_sq = S ** 2
                    if sval_sq.sum() == 0:
                        continue
                    energy = sval_sq.cumsum(0) / sval_sq.sum()
                    r = int((energy < self.threshold).sum().item()) + 1
                    self.feature_lists[k][param_name] = U[:, :r].cpu()
                else:
                    # Subsequent: project out existing basis, SVD residual
                    U_old = self.feature_lists[k][param_name]
                    act_hat = act - U_old @ (U_old.t() @ act)
                    U, S, _ = _stable_svd(act_hat)
                    sval_sq = S ** 2
                    if sval_sq.sum() == 0:
                        continue
                    energy = sval_sq.cumsum(0) / sval_sq.sum()
                    r = int((energy < self.threshold).sum().item()) + 1
                    U_new = torch.cat([U_old, U[:, :r].cpu()], dim=1)
                    # Cap rank at d_in
                    if U_new.size(1) > U_new.size(0):
                        U_new = U_new[:, : U_new.size(0)]
                    self.feature_lists[k][param_name] = U_new

        # ------ Head subspace via aggregated embeddings ------
        emb, _ = self.trainer.compute_embeddings(train_loader)
        emb = emb[: self.n_samples].t().float()  # (embed_dim, N)
        if self.head_basis is None:
            U, S, _ = _stable_svd(emb)
            sval_sq = S ** 2
            if sval_sq.sum() > 0:
                energy = sval_sq.cumsum(0) / sval_sq.sum()
                r = int((energy < self.threshold).sum().item()) + 1
                self.head_basis = U[:, :r].cpu()
        else:
            U_old = self.head_basis
            emb_hat = emb - U_old @ (U_old.t() @ emb)
            U, S, _ = _stable_svd(emb_hat)
            sval_sq = S ** 2
            if sval_sq.sum() > 0:
                energy = sval_sq.cumsum(0) / sval_sq.sum()
                r = int((energy < self.threshold).sum().item()) + 1
                U_new = torch.cat([U_old, U[:, :r].cpu()], dim=1)
                if U_new.size(1) > U_new.size(0):
                    U_new = U_new[:, : U_new.size(0)]
                self.head_basis = U_new

        # Log per-party basis sizes and head rank
        for k in range(self.args.num_parties):
            sizes = {n: u.size(1) for n, u in self.feature_lists[k].items()}
            print(f"    GPM party {k} basis dims: {sizes}")
        if self.head_basis is not None:
            print(f"    GPM head basis dim: {self.head_basis.size(1)}/{self.head_basis.size(0)}")

    def get_state(self):
        return {
            'feature_lists': deepcopy(self.feature_lists),
            'head_basis': self.head_basis.clone() if self.head_basis is not None else None,
            'threshold': self.threshold,
        }

    def load_state(self, s):
        self.feature_lists = s.get(
            'feature_lists', [{} for _ in range(self.args.num_parties)]
        )
        self.head_basis = s.get('head_basis', None)
        self.threshold = s.get('threshold', self.threshold)
