# OCAR paper-update implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement OCAR's old-class Fisher weighting and damped K-FAC update within a declared rank-limited approximation.

**Architecture:** Store each activation covariance as a rank-`ocar_max_rows` row factor `R` with `A=R.T@R`. Build unbiased sampled `R_new`, combine old/new covariance by square-root EMA weights, and truncate via top eigenspace of the small row Gram. Solve `(G⊗A+tau I) vec(grad)` exactly for this represented K-FAC block by eigendecomposing `G` and the small activation factor. Keep the existing true-Fisher pseudo-label sampling and checkpoint representation.

**Tech Stack:** Python 3.10, PyTorch, `unittest`.

## Global constraints

- OCAR paper Eq. 3 and Algorithm 1 are canonical: https://arxiv.org/html/2502.01866.
- Preserve four-party partition, data hashes, task order, frozen evaluation, seeds, and dataset-specific epochs 50/50/20.
- The rank cap is a disclosed approximation. The inverse test must be exact for its stored low-rank factors; do not mislabel separate factor damping as Eq. 3.
- The paper's `∇N+∇B` loss uses the two sample-distribution terms. Retain their separate CE means until a numerical equation-level reason proves a different normalization.

---

### Task 1: Damped block inverse

**Files:** Modify `test_ocar_vfl.py`, `cl_methods/ocar.py`.

**Interfaces:** `kfac_precondition(gradient, activation_rows, output_covariance, tau)` treats `A=activation_rows.T@activation_rows` and returns the matrix `X` solving `G@X@A + tau*X = gradient`. The factor is already normalized when stored.

- [ ] Replace the old test's factorwise-damped expected result with a direct dense Kronecker solve on a 2×3 positive-definite `G`, rank-2 `A`, and positive `tau`. Use an explicit matrix-operator solve or test `G@X@A+tau*X=gradient` to avoid vectorization-order confusion.
- [ ] Run `python -m unittest test_ocar_vfl -q` and confirm the old implementation fails on the new expected equation.
- [ ] Implement `eigh(G)` and compact `svd(R)` (or an equivalent small-Gram decomposition). Rotate the gradient into `G`'s eigenbasis. For each row eigenvalue `lambda`, apply `q/tau + (q@V)*(1/(tau+lambda*s²)-1/tau)@V.T`, then rotate back. Symmetrize PSD factors and clamp tiny negative eigenvalues caused by rounding.
- [ ] Run the focused test and a zero-activation covariance edge case.

### Task 2: Fisher class weight and covariance EMA

**Files:** Modify `test_ocar_vfl.py`, `cl_methods/ocar.py`.

**Interfaces:** `before_task` sets `lambda_buffer=old_class_count/new_class_count`; `_rows` samples aligned activation/output-gradient rows uniformly when capped; `R_new=Z/sqrt(number_of_kept_rows)`. On update, concatenate `sqrt(1-ema)*R_old` and `sqrt(ema)*R_new`, then retain the leading `max_rows` directions by singular values. For output covariance retain the existing matrix EMA. Classifier growth retains A and resets incompatible G.

- [ ] Add tests: old10/new10 yields lambda1 and buffer Fisher coefficient2; two uncapped factor updates reproduce `(1-ema)A_old+ema A_new`; rank cap preserves the top spectral directions and PSD covariance; classifier expansion retains A and clears G.
- [ ] Run new tests red.
- [ ] Implement the minimal updates, retaining the existing checkpoint `factors` schema `(R,G)`.
- [ ] Run `python -m unittest test_ocar_vfl -q` green, then `git diff --check` and commit.

### Task 3: Paper parameter step and four-party pilot

**Files:** `replay11_matrix.py`, `test_replay11_matrix.py`, `test_ocar_vfl.py`, and narrowly conditional trainer code only if tests prove it necessary.

- [ ] Make OCAR-only real parameter updates plain SGD with zero momentum/weight decay; pass `tau`, not `sqrt(tau)`, to the new damped solve. Test that top and all four bottoms take `−lr*preconditioned_gradient` on a tiny one-step model.
- [ ] Check OCAR-only command options and that ER/other method commands remain unchanged. Run method, matrix, and small four-party tests.
- [ ] After the full producer commit is frozen, run two-task vector then image deferred pilots, measuring time and peak GPU memory before full OCAR reruns. Keep old outputs separate.

## Acceptance

For represented factors, the operator residual `G@X@(R.T@R)+tau*X-gradient` is numerically zero; two-step covariance EMA and old-class Fisher weighting match the paper. The paper step has no extra optimizer momentum/AdamW, and the rank cap is reported.
