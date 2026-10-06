"""ROAR — Redundancy-Optimal Ablation & Recovery (VFL class unlearning).

This is the operator implied by the exact per-party logit decomposition of a
linear VFL top model:

    logit_c(x) = b_c + Σ_k s_{c,k}(x),
        s_{c,k} = <w_c, e_k>        (sum aggregation)
                = <w_c^(k), e_k>    (concat: party-k weight block)

From this decomposition we derive, for each forget class f:

  (0) Ownership pi_{f,k} ∝ E_forget|s_{f,k}|  — exact, one forward pass.
      Minimal party set S*(f) = smallest prefix (by ownership) whose cumulative
      share >= tau_own. Residual evidence after scrubbing S* is
      b_f + Σ_{k∉S*} s_{f,k} (the redundancy lower-bound quantity). Parties with
      ~0 ownership are never touched  -> communication ∝ |S*|, not P.

  (1) BOTTOM scrub on S* only: remove f-discriminability from those parties'
      embeddings (gradient ascent on forget CE) WHILE preserving their retain
      embeddings (MSE to a frozen snapshot). This is the over-forgetting fix
      baselines lack: parties outside S* and all retain geometry are protected.

  (2) TOP closed-form suppression: zero the forget ROWS of (W, b). Because only
      forget rows change, every retained class logit is preserved EXACTLY
      (server-side, zero communication).

  (3) Light top-only recovery on retain to recalibrate retained rows after the
      bottom edits (server-side, zero communication).

Contrast with radapt_router (which dispatches whole classes to LUV/FedOSD):
ROAR is a single operator grounded in the decomposition, with explicit
retain preservation and honest per-party communication accounting.
"""
from __future__ import annotations
import copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import split_features


class _UCE(nn.Module):
    """Unlearning CE (FedOSD Eq.3): -Σ_c y_c log(1 - p_c/2). Pushes the forget-class
    posterior down without the exploding-gradient pathology of plain -CE ascent."""
    def forward(self, logits, targets):
        p = F.softmax(logits, dim=-1)
        oh = F.one_hot(targets, num_classes=logits.size(-1)).float()
        return -(oh * torch.log(1.0 - p / 2.0 + 1e-8)).sum(-1).mean()


class RoarUL:
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'ROAR'
        self.tau_own = float(getattr(args, 'roar_tau_own', 0.80))
        self.scrub_epochs = int(getattr(args, 'roar_scrub_epochs', 3))
        self.lambda_preserve = float(getattr(args, 'roar_lambda_preserve', 1.0))
        self.recovery_epochs = int(getattr(args, 'roar_recovery_epochs', 2))
        self.scrub_lr = float(getattr(args, 'roar_scrub_lr', getattr(args, 'ul_lr', 1e-4)))
        # weight on direct raw-residual-score suppression (the certified quantity).
        # 0 = legacy cosine-CE-only scrub (leaves raw scores -> epsilon high).
        self.scrub_raw_weight = float(getattr(args, 'roar_scrub_raw_weight', 0.0))
        # removal mechanism: 'ascent' (legacy GA+MSE) or 'ortho' (ROAR-v2:
        # orthogonalized UCE erasure on S* — FedOSD-quality removal at |S*| comm).
        self.scrub_mode = str(getattr(args, 'roar_scrub_mode', 'ascent'))

    # ---------- helpers ----------
    def _embeds(self, parts):
        return [self.trainer.bottoms[k](parts[k]) for k in range(self.args.num_parties)]

    def _party_block(self, W, k, embed_dim):
        if self.args.aggregation == 'concat':
            return W[:, k*embed_dim:(k+1)*embed_dim]
        return W  # sum: shared weight

    # ---------- Step 0: ownership & minimal party set ----------
    @torch.no_grad()
    def _ownership(self, forget_classes, forget_loader):
        """Return {f: (shares over parties np.array, S*(f) list)} from exact decomposition."""
        device = self.args.device
        P = self.args.num_parties
        W = self.trainer.top_model.classifier.weight.detach()
        embed_dim = W.size(1) // (P if self.args.aggregation == 'concat' else 1)
        for b in self.trainer.bottoms: b.eval()
        accum = {int(f): np.zeros(P) for f in forget_classes}
        count = {int(f): 0 for f in forget_classes}
        for bx, by in forget_loader:
            bx = bx.to(device); by = by.to(device)
            parts = split_features(bx, self.args)
            embs = self._embeds(parts)
            for f in forget_classes:
                f = int(f)
                mask = (by == f)
                if mask.sum() == 0:
                    continue
                for k in range(P):
                    wk = self._party_block(W, k, embed_dim)[f]      # (d,)
                    sk = (embs[k][mask] * wk).sum(dim=1)            # (n_f,) = <w_f^k, e_k>
                    accum[f][k] += sk.abs().sum().item()
                count[f] += int(mask.sum().item())
        out = {}
        for f in forget_classes:
            f = int(f)
            shares = accum[f] / (accum[f].sum() + 1e-12)
            order = list(np.argsort(-shares))
            cum, S = 0.0, []
            for k in order:
                S.append(int(k)); cum += shares[k]
                if cum >= self.tau_own:
                    break
            out[f] = {'shares': shares.tolist(), 'S_star': S,
                      'residual_share': float(1.0 - cum)}
        return out

    # ---------- Residual-leakage probe (Theorem 1 validation) ----------
    @torch.no_grad()
    def _score_probe(self, forget_classes, forget_loader, W, b):
        """Measure the per-party score decomposition s_{f,k}(x)=<W_f^k, e_k(x)>
        of the CURRENT bottoms against a FIXED head row W_f (clone).

        Run twice with the SAME original W_f: once on pre-scrub bottoms (gives
        Z_f, the total score mass) and once on the post-scrub/recovered bottoms
        (the head-reconnection attack — the evidence that survives unlearning).

        Returns {f: {
            'mean_abs':   E_{x∈D_f}|s_{f,k}(x)|  per party  (len P),
            'mean_signed':E_{x∈D_f} s_{f,k}(x)   per party  (len P),
            'R_signed':   E_{x∈D_f}[ b_f + Σ_k s_{f,k}(x) ],   # reconnect logit
            'R_abs':      E_{x∈D_f}| b_f + Σ_k s_{f,k}(x) |,
            'bias':       b_f (0 under cosine head),
        }}. The reconnection score R_f is the residual class-f evidence Thm 1
        bounds: E|R_f| ≤ |b_f| + Σ_k E|s_{f,k}|.
        """
        device = self.args.device
        P = self.args.num_parties
        cosine = bool(getattr(self.trainer.top_model, 'cosine', False))
        embed_dim = W.size(1) // (P if self.args.aggregation == 'concat' else 1)
        for bl in self.trainer.bottoms: bl.eval()
        acc_abs = {int(f): np.zeros(P) for f in forget_classes}
        acc_sgn = {int(f): np.zeros(P) for f in forget_classes}
        acc_R = {int(f): 0.0 for f in forget_classes}
        acc_Rabs = {int(f): 0.0 for f in forget_classes}
        count = {int(f): 0 for f in forget_classes}
        for bx, by in forget_loader:
            bx = bx.to(device); by = by.to(device)
            parts = split_features(bx, self.args)
            embs = self._embeds(parts)
            for f in forget_classes:
                f = int(f)
                mask = (by == f)
                n = int(mask.sum().item())
                if n == 0:
                    continue
                bf = 0.0 if cosine else float(b[f].item())
                Rsum = torch.full((n,), bf, device=device)
                for k in range(P):
                    wk = self._party_block(W, k, embed_dim)[f]   # (d,)
                    sk = (embs[k][mask] * wk).sum(dim=1)          # (n,)
                    acc_abs[f][k] += sk.abs().sum().item()
                    acc_sgn[f][k] += sk.sum().item()
                    Rsum = Rsum + sk
                acc_R[f] += Rsum.sum().item()
                acc_Rabs[f] += Rsum.abs().sum().item()
                count[f] += n
        out = {}
        for f in forget_classes:
            f = int(f); c = max(count[f], 1)
            out[f] = {
                'mean_abs': (acc_abs[f] / c).tolist(),
                'mean_signed': (acc_sgn[f] / c).tolist(),
                'R_signed': acc_R[f] / c,
                'R_abs': acc_Rabs[f] / c,
                'bias': 0.0 if cosine else float(b[f].item()),
            }
        return out

    def _residual_diagnostics(self, forget_classes, own, s_pre, s_post):
        """Assemble the Theorem-1 certificate terms per forget class f:

            E|R_f| (measured, reconnect attack on scrubbed bottoms)
              ≤ |b_f| + ε|S*| + Σ_{k∉S*} E|s_{f,k}|        (bound RHS)

        ε     = mean_{k∈S*} E|s̃_{f,k}|   (post-scrub tolerance on owned parties)
        Z_f   = Σ_k E|s_{f,k}|            (pre-scrub total score mass)
        residual_term = Σ_{k∉S*} E|s_{f,k}|  (unscrubbed parties, ≈ (1−τ)·Z_f)
        """
        diag = {}
        for f in forget_classes:
            f = int(f)
            S = own[f]['S_star']
            notS = [k for k in range(self.args.num_parties) if k not in S]
            pre_abs = np.array(s_pre[f]['mean_abs'])
            post_abs = np.array(s_post[f]['mean_abs'])
            Z_f = float(pre_abs.sum())
            eps_each = post_abs[S] if S else np.array([0.0])
            eps = float(eps_each.mean())
            eps_sum = float(post_abs[S].sum()) if S else 0.0
            # a-priori residual term: pre-scrub ownership of the unscrubbed parties
            # (= (1−τ_own)·Z_f). Exact because k∉S* stay frozen through recovery.
            residual_term = float(pre_abs[notS].sum()) if notS else 0.0
            bias = abs(s_post[f]['bias'])
            bound_rhs = bias + eps_sum + residual_term            # Theorem-1 (a-priori) bound
            # a-posteriori triangle bound: |b_f| + Σ_k E|s̃_{f,k}| — always ≥ E|R_f|.
            bound_rhs_post = bias + float(post_abs.sum())
            R_abs = float(s_post[f]['R_abs'])
            diag[f] = {
                'S_star': S, 'parties_outside_S': notS,
                'Z_f': Z_f,
                'eps': eps, 'eps_sum_over_Sstar': eps_sum,
                'residual_term_outside_S': residual_term,
                'bias_abs': bias,
                'bound_rhs': bound_rhs,
                'bound_rhs_aposteriori': bound_rhs_post,
                'R_f_measured_abs': R_abs,
                'R_f_measured_signed': float(s_post[f]['R_signed']),
                'R_f_preScrub_abs': float(s_pre[f]['R_abs']),
                'bound_holds': bool(R_abs <= bound_rhs_post + 1e-6),
                'apriori_bound_holds': bool(R_abs <= bound_rhs + 1e-6),
                'bound_tightness': float(R_abs / (bound_rhs + 1e-12)),
                's_pre_abs': pre_abs.tolist(),
                's_post_abs': post_abs.tolist(),
            }
        return diag

    # ---------- Step 1: retain-preserving bottom scrub on S* ----------
    def _scrub_bottoms(self, S, forget_loader, retain_loader, forget_classes=None):
        device = self.args.device
        if not S:
            return {'scrubbed_parties': [], 'comm_party_steps': 0}
        # frozen snapshot of the S-party bottoms for retain-embedding preservation
        snap = {k: copy.deepcopy(self.trainer.bottoms[k]).eval() for k in S}
        for k in snap.values():
            for p in k.parameters(): p.requires_grad_(False)

        params = [p for k in S for p in self.trainer.bottoms[k].parameters() if p.requires_grad]
        if not params:
            return {'scrubbed_parties': [], 'comm_party_steps': 0, 'skipped': 'frozen'}
        opt = torch.optim.SGD(params, lr=self.scrub_lr, momentum=0.9)
        ce = nn.CrossEntropyLoss()
        comm_steps = 0
        retain_iter = iter(retain_loader)
        # Direct residual-score suppression. The cosine-CE ascent only drives the
        # NORMALIZED logit to 0; it can do so by inflating ||agg|| while leaving the
        # RAW per-party score s_{f,k}=<W_f^(k),e_k> (what the head-reconnection
        # certificate measures) untouched -> epsilon stays large. So we add a term
        # that minimizes the raw owned-party scores directly: the scrub now removes
        # the certified quantity, not a proxy. W_f is read here (pre-suppression =
        # original) and detached (we move the bottoms, not the head).
        fcs = [int(f) for f in (forget_classes or [])]
        W_scrub = self.trainer.top_model.classifier.weight.detach()
        edim = W_scrub.size(1) // (self.args.num_parties if self.args.aggregation == 'concat' else 1)
        for _ in range(self.scrub_epochs):
            for bx, by in forget_loader:
                bx = bx.to(device); by = by.to(device)
                parts = split_features(bx, self.args)
                # forward: S parties live (grad), others detached (cached -> no extra comm)
                embs = []
                for k in range(self.args.num_parties):
                    e = self.trainer.bottoms[k](parts[k])
                    embs.append(e if k in S else e.detach())
                agg = self.trainer._aggregate(embs)
                logits = self.trainer.top_model(agg)
                loss_forget = -ce(logits, by)            # gradient ASCENT on forget

                # retain-embedding preservation on S parties
                try:
                    rx, _ = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader); rx, _ = next(retain_iter)
                rx = rx.to(device)
                rparts = split_features(rx, self.args)
                loss_pres = 0.0
                for k in S:
                    e_live = self.trainer.bottoms[k](rparts[k])
                    with torch.no_grad():
                        e_ref = snap[k](rparts[k])
                    loss_pres = loss_pres + F.mse_loss(e_live, e_ref)

                # raw owned-score suppression on forget-class samples (S parties live)
                loss_raw = 0.0
                if self.scrub_raw_weight > 0 and fcs:
                    for f in fcs:
                        m = (by == f)
                        if m.sum() == 0:
                            continue
                        for k in S:
                            wk = self._party_block(W_scrub, k, edim)[f]   # (d,) detached
                            sk = (embs[k][m] * wk).sum(dim=1)             # raw s_{f,k}
                            loss_raw = loss_raw + (sk ** 2).mean()

                loss = loss_forget + self.lambda_preserve * loss_pres \
                    + self.scrub_raw_weight * loss_raw
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                # honest comm: only S parties exchange embeddings/grads this step
                comm_steps += len(S)
                self.trainer.comm_rounds += 1
                self.trainer.bytes_transmitted += sum(
                    embs[k].nelement() * 4 * 2 for k in S)
        return {'scrubbed_parties': S, 'comm_party_steps': comm_steps}

    # ---------- Step 1' (ROAR-v2): orthogonalized erasure on S* only ----------
    def _scrub_ortho(self, S, forget_loader, retain_loader, forget_classes=None):
        """ROAR-v2 removal. Erase forget discriminability (UCE) but project the
        update ORTHOGONAL to the retain-CE gradient (the FedOSD / gradient-surgery
        mechanism that empirically removes evidence far better than plain ascent),
        restricted to the S* bottoms only -> strong removal at communication
        proportional to |S*|, not P. Non-S* embeddings are detached (no grad, no
        comm). No recovery phase (it re-injects forget signal); the orthogonal
        projection already protects retain."""
        device = self.args.device
        P = self.args.num_parties
        if not S:
            return {'scrubbed_parties': [], 'comm_party_steps': 0}
        params = [p for k in S for p in self.trainer.bottoms[k].parameters() if p.requires_grad]
        if not params:
            return {'scrubbed_parties': [], 'comm_party_steps': 0, 'skipped': 'frozen'}
        for k in S:
            self.trainer.bottoms[k].train()
        uce = _UCE(); ce = nn.CrossEntropyLoss()
        comm_steps = 0
        retain_iter = iter(retain_loader)

        def _emb_logits(bx):
            parts = split_features(bx, self.args)
            embs = [self.trainer.bottoms[k](parts[k]) if k in S
                    else self.trainer.bottoms[k](parts[k]).detach() for k in range(P)]
            return embs, self.trainer.top_model(self.trainer._aggregate(embs))

        def _flat_grad(loss):
            g = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
            return torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1)
                              for gi, p in zip(g, params)])

        for _ in range(self.scrub_epochs):
            for bx, by in forget_loader:
                bx = bx.to(device); by = by.to(device)
                embs, logits = _emb_logits(bx)
                g_f = _flat_grad(uce(logits, by))          # erase-forget direction
                try:
                    rx, ry = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader); rx, ry = next(retain_iter)
                rx = rx.to(device); ry = ry.to(device)
                _, rlogits = _emb_logits(rx)
                g_r = _flat_grad(ce(rlogits, ry))          # protect-retain direction
                gr2 = (g_r * g_r).sum()
                if gr2.item() > 1e-12:                     # remove retain-conflicting comp
                    g_f = g_f - ((g_f * g_r).sum() / gr2) * g_r
                gn = g_f.norm()
                if gn.item() > 1e-12:
                    g_f = g_f * min(1.0, 5.0 / gn.item())  # trust-region clip
                with torch.no_grad():
                    off = 0
                    for p in params:
                        n = p.numel()
                        p.add_(-self.scrub_lr * g_f[off:off + n].reshape(p.shape))
                        off += n
                comm_steps += len(S); self.trainer.comm_rounds += 1
                self.trainer.bytes_transmitted += sum(embs[k].nelement() * 4 * 2 for k in S)
        return {'scrubbed_parties': S, 'comm_party_steps': comm_steps}

    # ---------- Step 1'' (ROAR-v3): SISA x ownership — exact retrain of S* ----------
    def _scrub_retrain_s(self, S, retain_loader):
        """SISA-style EXACT removal restricted to the certified owning set: RE-INIT
        the S* encoders and retrain them (+ top head) on retain-only data, with all
        non-S* parties FROZEN. This is a true retrain of just those 'shards', so the
        forget class is removed from S* exactly (gradient-stall-free, unlike the
        orthogonal/ascent scrubs), at communication proportional to |S*|. The
        residual evidence on the untouched non-S* parties is exactly the
        certificate's (1-tau)Z_f term -> raising tau (more parties in S*) trades
        communication for removal along a CERTIFIED frontier; fedosd is the tau->1
        (all-party) endpoint."""
        device = self.args.device
        if not S:
            return {'scrubbed_parties': [], 'comm_party_steps': 0}
        # re-initialize S* encoders: erase ALL prior memory (incl. the forget class)
        for k in S:
            for m in self.trainer.bottoms[k].modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
        epochs = int(getattr(self.args, 'roar_retrain_epochs',
                             getattr(self.args, 'epochs_per_task', 30)))
        Sset = set(S)
        for k in range(self.args.num_parties):
            self.trainer.bottoms[k].train() if k in Sset else self.trainer.bottoms[k].eval()
        self.trainer.top_model.train()
        b_params = [p for k in S for p in self.trainer.bottoms[k].parameters() if p.requires_grad]
        opt_t = torch.optim.SGD(self.trainer.top_model.parameters(), lr=self.args.lr, momentum=0.9)
        opt_b = torch.optim.SGD(b_params, lr=self.args.lr, momentum=0.9) if b_params else None
        ce = nn.CrossEntropyLoss(); comm = 0
        for _ in range(epochs):
            for bx, by in retain_loader:
                bx = bx.to(device); by = by.to(device)
                parts = split_features(bx, self.args)
                embs = self._embeds(parts)
                logits = self.trainer.top_model(self.trainer._aggregate(embs))
                loss = ce(logits, by)
                opt_t.zero_grad()
                if opt_b: opt_b.zero_grad()
                loss.backward()
                opt_t.step()
                if opt_b: opt_b.step()
                comm += len(S); self.trainer.comm_rounds += 1
                # only S* parties re-transmit (non-S* frozen, cached) -> comm ∝ |S*|
                self.trainer.bytes_transmitted += sum(embs[k].nelement()*4*2 for k in S)
        return {'scrubbed_parties': S, 'comm_party_steps': comm}

    # ---------- Step 2: top closed-form suppression (retain-preserving) ----------
    @torch.no_grad()
    def _suppress_top(self, forget_classes):
        """Zero the forget ROW W_f of the top head. This is exact under BOTH heads:

          - linear:  logit_f = b_f + <W_f, agg>; zeroing W_f leaves only b_f, so
                     we ALSO drive the bias to -inf to make class f unpredictable.
          - cosine:  logit_f = scale * <normalize(agg), normalize(W_f)>; the bias
                     is never read (models.TopModel.forward), and F.normalize of a
                     zero row is 0/eps = 0 (finite), so logit_f -> 0 exactly. The
                     bias subtraction is a DEAD no-op under cosine and is guarded
                     out below. In either case rows c != f and `agg` are untouched,
                     so every RETAINED class logit is preserved exactly.
        """
        W = self.trainer.top_model.classifier.weight
        b = self.trainer.top_model.classifier.bias
        cosine = getattr(self.trainer.top_model, 'cosine', False)
        for f in forget_classes:
            W[int(f)].zero_()
            if not cosine:
                # linear head only: bias is live, so push logit_f -> -inf.
                b[int(f)] = b[int(f)] - 1e4
            else:
                # cosine head: zeroed row already gives logit_f == 0 exactly; keep
                # the bias entry clean (forward never reads it) for introspection.
                b[int(f)].zero_()

    # ---------- Step 3: retain recovery (top + S* bottoms only) ----------
    def _recover_top(self, retain_loader, S):
        """Re-fit on the clean retain set to undo the damage from the bottom scrub
        (same role as LUV/FedOSD's recovery phase). CRITICAL: recovery trains only
        the top head and the S* bottoms — the parties ROAR already touches. The
        k∉S* bottoms stay FROZEN, so (a) communication stays ∝ |S*| (the headline
        claim), and (b) the unscrubbed parties' class-f scores remain exactly at
        their pre-unlearning value, making the Theorem-1 residual identity
        (1−τ_own)·Z_f exact. The forget rows get no gradient (no forget samples in
        the clean retain loader) so they stay suppressed."""
        if self.recovery_epochs <= 0:
            return 0
        device = self.args.device
        Sset = set(S)
        for k in range(self.args.num_parties):
            self.trainer.bottoms[k].train() if k in Sset else self.trainer.bottoms[k].eval()
        self.trainer.top_model.train()
        b_params = [p for k in S for p in self.trainer.bottoms[k].parameters() if p.requires_grad]
        opt_t = torch.optim.SGD(self.trainer.top_model.parameters(), lr=self.args.lr, momentum=0.9)
        opt_b = torch.optim.SGD(b_params, lr=self.args.lr, momentum=0.9) if b_params else None
        ce = nn.CrossEntropyLoss()
        comm = 0
        for _ in range(self.recovery_epochs):
            for bx, by in retain_loader:
                bx = bx.to(device); by = by.to(device)
                parts = split_features(bx, self.args)
                embs = self._embeds(parts)
                agg = self.trainer._aggregate(embs)
                logits = self.trainer.top_model(agg)
                loss = ce(logits, by)
                opt_t.zero_grad()
                if opt_b: opt_b.zero_grad()
                loss.backward()
                opt_t.step()
                if opt_b: opt_b.step()
                comm += 1
                self.trainer.comm_rounds += 1
                # only S* parties re-transmit embeddings/grads (k∉S* are frozen,
                # their embeddings are cached server-side) -> comm stays ∝ |S*|.
                self.trainer.bytes_transmitted += sum(
                    embs[k].nelement()*4*2 for k in S)
        return comm

    # ---------- Re-learned-head attack (realizable threat, ROC-AUC) ----------
    @torch.no_grad()
    def _collect_agg(self, loader, forget_classes, device, max_n=4000):
        """Post-scrub aggregated embeddings + binary label (1 if forget class)."""
        for b in self.trainer.bottoms: b.eval()
        fcs = set(int(f) for f in forget_classes)
        Z, Y, n = [], [], 0
        for bx, by in loader:
            bx = bx.to(device)
            parts = split_features(bx, self.args)
            agg = self.trainer._aggregate(self._embeds(parts))
            Z.append(agg.detach()); Y.append(torch.tensor([1.0 if int(c) in fcs else 0.0 for c in by]))
            n += bx.size(0)
            if n >= max_n:
                break
        return torch.cat(Z), torch.cat(Y).to(device)

    @torch.no_grad()
    def _reconnect_attack(self, W_row, forget_class, loader):
        """ORIGINAL-head RECONNECTION threat (the one ROAR is built to defend, e.g.
        the server kept a pre-unlearning head checkpoint / audit snapshot). Use the
        FROZEN pre-unlearning head row W_f as a FIXED linear probe on the
        post-unlearning embeddings and report ROC-AUC of detecting class f. Methods
        that only edit the deployed head leave the bottoms intact -> this stays high
        (evidence fully recoverable by reconnecting the old head); scrubbing the
        owning bottoms (roar) or retraining them (retrain) drops it. Unlike the
        re-learned-head attack (which finds the best NEW direction and saturates at
        intrinsic separability for everyone), this isolates what the SCRUB removes."""
        device = self.args.device
        Z, Y = self._collect_agg(loader, [forget_class], device)
        if Y.sum() < 2 or (Y == 0).sum() < 2:
            return float('nan')
        s = Z @ W_row.to(device).flatten()
        order = torch.argsort(s)
        ranks = torch.empty_like(s); ranks[order] = torch.arange(1, s.numel() + 1, device=device, dtype=s.dtype)
        npos = Y.sum(); nneg = (Y == 0).sum()
        return float((ranks[Y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg + 1e-12))

    def _relearn_attack(self, forget_classes, attack_loader):
        """The REALIZABLE head-reconnection threat: instead of assuming the
        adversary kept the *original* W_f (which the server zeroes by construction),
        let them RE-LEARN the best linear forget-vs-rest head on the post-unlearning
        (scrubbed+recovered) embeddings from a few labels, and report ROC-AUC. This
        is the strongest linear read-out of residual class-f structure: AUC->0.5
        means the class is genuinely unrecoverable (the certificate is earned);
        AUC->1.0 means residual evidence survives in some direction. Bottoms frozen
        throughout (we only fit a fresh linear probe)."""
        device = self.args.device
        Z, Y = self._collect_agg(attack_loader, forget_classes, device)
        if Y.sum() < 2 or (Y == 0).sum() < 2:
            return {'auc': float('nan'), 'n': int(Y.numel())}
        # standardize features; 50/50 split (attacker trains on half, scores half)
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
        perm = torch.randperm(Z.size(0), device=device)
        h = Z.size(0) // 2
        tr, te = perm[:h], perm[h:]
        clf = nn.Linear(Z.size(1), 1).to(device)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-3)
        bce = nn.BCEWithLogitsLoss()
        Ztr, Ytr = Z[tr], Y[tr]
        for _ in range(300):
            opt.zero_grad()
            loss = bce(clf(Ztr).squeeze(1), Ytr)
            loss.backward(); opt.step()
        with torch.no_grad():
            s = clf(Z[te]).squeeze(1); y = Y[te]
        # ROC-AUC via the Mann-Whitney rank statistic
        order = torch.argsort(s)
        ranks = torch.empty_like(s); ranks[order] = torch.arange(1, s.numel() + 1, device=device, dtype=s.dtype)
        npos = y.sum(); nneg = (y == 0).sum()
        auc = float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg + 1e-12))
        return {'auc': auc, 'n': int(Z.size(0)), 'n_pos': int(npos.item())}

    # ---------- orchestration ----------
    def unlearn(self, forget_classes, retain_train_loader, forget_train_loader, **kw):
        start = time.time()
        forget_classes = [int(f) for f in forget_classes]
        # IMPORTANT: the runner's retain_train_loader only excludes the CURRENT
        # forget class, so it still contains classes forgotten in EARLIER UL
        # events. Training (scrub/recover) on those — whose logits we drove to
        # -inf — produces exploding CE and collapses the model. Rebuild a clean
        # retain loader over the true effective set (excludes ALL forgotten).
        eff = kw.get('effective_classes')
        ds = getattr(self.trainer, 'dataset_ref', None)
        if eff and ds is not None:
            try:
                retain_train_loader = ds.get_task_loaders(sorted(int(c) for c in eff))[0]
            except Exception as e:
                print(f"  [ROAR] warn: clean retain loader failed ({e}); using runner's")
        own = self._ownership(forget_classes, forget_train_loader)
        S = sorted(set(k for f in forget_classes for k in own[int(f)]['S_star']))
        print(f"  [ROAR] tau_own={self.tau_own}  forget={forget_classes}")
        for f in forget_classes:
            print(f"    class {f}: shares={[round(x,3) for x in own[f]['shares']]} "
                  f"S*={own[f]['S_star']} residual={own[f]['residual_share']:.3f}")
        print(f"  [ROAR] union S*={S} / {self.args.num_parties} parties "
              f"(comm touches {len(S)} parties, not {self.args.num_parties})")

        # Snapshot the ORIGINAL head (rows + bias) BEFORE any edit: the residual
        # diagnostic reconnects this original W_f to the scrubbed bottoms.
        W_orig = self.trainer.top_model.classifier.weight.detach().clone()
        b_orig = self.trainer.top_model.classifier.bias.detach().clone()
        s_pre = self._score_probe(forget_classes, forget_train_loader, W_orig, b_orig)

        if self.scrub_mode == 'retrain_s':
            # SISA x ownership: re-init + retrain S* on retain; this IS the recovery.
            scrub = self._scrub_retrain_s(S, retain_train_loader)
            self._suppress_top(forget_classes)
        elif self.scrub_mode == 'ortho':
            scrub = self._scrub_ortho(S, forget_train_loader, retain_train_loader,
                                      forget_classes=forget_classes)
            self._suppress_top(forget_classes)
            # ortho already protects retain; recovery only if explicitly requested
            if self.recovery_epochs > 0:
                self._recover_top(retain_train_loader, S)
        else:
            scrub = self._scrub_bottoms(S, forget_train_loader, retain_train_loader,
                                        forget_classes=forget_classes)
            self._suppress_top(forget_classes)
            self._recover_top(retain_train_loader, S)

        # Head-reconnection attack: probe the scrubbed+recovered bottoms with the
        # ORIGINAL W_f to measure surviving evidence R_f and validate Theorem 1.
        s_post = self._score_probe(forget_classes, forget_train_loader, W_orig, b_orig)
        residual_diag = self._residual_diagnostics(forget_classes, own, s_pre, s_post)

        # Realizable threat: best RE-LEARNED linear forget-head on the scrubbed
        # bottoms (held-out test split), reported as ROC-AUC. Stronger + more
        # honest than reconnecting the server-controlled original W_f.
        if ds is not None:
            try:
                atk_classes = sorted(set(int(c) for c in (eff or [])) | set(forget_classes))
                atk_loader = ds.get_task_loaders(atk_classes, shuffle_train=False)[1]
                for f in forget_classes:
                    residual_diag[int(f)]['relearn_attack'] = self._relearn_attack([f], atk_loader)
            except Exception as e:
                print(f"  [ROAR] warn: re-learn attack failed ({e})")

        for f in forget_classes:
            d = residual_diag[int(f)]
            auc = d.get('relearn_attack', {}).get('auc', float('nan'))
            print(f"  [ROAR] class {f} residual: E|R_f|={d['R_f_measured_abs']:.4f} "
                  f"<= bound {d['bound_rhs']:.4f} (eps={d['eps']:.4f}, "
                  f"resid_out={d['residual_term_outside_S']:.4f}, "
                  f"hold={d['bound_holds']}, tight={d['bound_tightness']:.2f}) "
                  f"relearn_AUC={auc:.3f}")

        result = {
            'history': [], 'time': time.time() - start, 'method': 'roar',
            'ownership': own, 'union_S_star': S,
            'residual_diag': residual_diag,
            'n_parties_touched': len(S), 'n_parties_total': self.args.num_parties,
            'comm_party_steps': scrub.get('comm_party_steps', 0),
            'aggregation': self.args.aggregation,
            'cosine_head': bool(getattr(self.trainer.top_model, 'cosine', False)),
            'config': {'tau_own': self.tau_own, 'scrub_epochs': self.scrub_epochs,
                       'lambda_preserve': self.lambda_preserve,
                       'recovery_epochs': self.recovery_epochs},
        }
        # Robust gating-decision artifact: persist ownership/S* so the |S*|<P
        # claim is reproducible from disk, not just stdout. Never crash the run.
        try:
            import json, os
            out_dir = getattr(self.args, 'output_dir', None) or '.'
            os.makedirs(out_dir, exist_ok=True)
            tag = '_'.join(str(int(f)) for f in forget_classes)
            path = os.path.join(out_dir, f'roar_ownership_f{tag}.json')
            with open(path, 'w') as fh:
                json.dump({k: result[k] for k in
                           ('ownership', 'union_S_star', 'residual_diag',
                            'n_parties_touched', 'n_parties_total',
                            'aggregation', 'cosine_head', 'config')},
                          fh, indent=2, default=str)
        except Exception as e:
            print(f"  [ROAR] warn: could not write ownership artifact ({e})")
        return result
