# LPR paper-update implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align LPR BatchNorm and sampled-reference gradient preconditioning with the paper's joint affine proximal update.

**Architecture:** Retain the existing Woodbury function. Record normalized BatchNorm inputs and the total-to-kept row mass for a bounded uniform sample. Apply the same affine inverse to each BatchNorm `(gamma,beta)` pair; fold the row mass into the inverse coefficient so bias and weight cross covariance are preserved.

**Tech Stack:** Python 3.10, PyTorch, `unittest`.

## Global constraints

- Paper formula is canonical: https://proceedings.mlr.press/v235/yoo24a.html. Reference subset and bounded activation rows are declared approximation/hyperparameters.
- Keep four-party data, frozen task order and hashes, shared per-dataset epochs (50 CIFAR100, 50 ISOLET, 20 UPMC), and deferred evaluation.
- Keep changes within LPR code/tests unless a proven integration need arises.

---

### Task 1: Numeric joint BatchNorm inverse

**Files:** Modify `test_lpr_vfl.py`, `cl_methods/lpr.py`.

**Interfaces:** `LPRCL.references[module]` carries `(normalized_rows, omega, row_mass)`. The BatchNorm branch in `_transform_modules` uses the 2×2 Gram of `[normalized_input, 1]` per channel, with coefficient `omega*row_mass/reference_count`, jointly transforming gamma and beta gradients. No separately scaled bias gradient remains.

- [ ] Add a test with a BatchNorm channel whose normalized reference mean is nonzero and whose gamma gradient is nonzero while beta gradient is zero; compare both adjusted gradients to a direct 2×2 `torch.linalg.solve`, requiring a nonzero adjusted beta gradient.
- [ ] Run `python -m unittest test_lpr_vfl -q` with `PYTHONUTF8=1` on the Windows worktree and confirm the new test fails on the old independent-diagonal BatchNorm branch.
- [ ] In `_record_inputs`, normalize BatchNorm input with its frozen running mean and variance plus epsilon before flattening to rows. In `_transform_modules`, compute vectorized `a=1+c*sum(z²)`, `b=c*sum(z)`, `d=1+c*nrows`, determinant `ad-b²`, then `(d*g_gamma-b*g_beta)/det` and `(-b*g_gamma+a*g_beta)/det`.
- [ ] Run `python -m unittest test_lpr_vfl -q` and confirm zero failures.

### Task 2: Preserve covariance mass under row cap

**Files:** Modify `test_lpr_vfl.py`, `cl_methods/lpr.py`.

**Interfaces:** For `N` total activation rows and `m=min(N,max_rows)` kept rows, record `row_mass=N/m`; choose rows uniformly without replacement on the current PyTorch RNG. For Linear/Conv2d pass `normalizer=reference_count/row_mass` into `precondition_affine_gradient`; for BatchNorm use the same mass in Task 1's coefficient. A full-row reference has mass 1.

- [ ] Add a constant-row Conv2d example with more patches than the cap; compare its sampled-row preconditioning with the full-row direct affine inverse. Constant rows make the expected equality exact for every uniform sample.
- [ ] Run the new test red; the old linspace truncation must produce too small a Gram.
- [ ] Change `_record_inputs` to store sampled rows and row mass; adapt `_transform_modules` as specified. Leave `lpr_max_rows` configurable and log its value through the existing command/provenance path.
- [ ] Run `python -m unittest test_lpr_vfl test_replay11_matrix -q` and confirm zero failures. Run `git diff --check` and commit.

### Task 3: Four-party integration and deferred pilot

**Files:** Add one compact four-party LPR test in `test_lpr_vfl.py`; production code only if it exposes a concrete defect.

- [ ] Test a one-step four-party top and bottom training pass with a prior-task buffer, checking refresh interval and that both top and bottom gradients are preconditioned before their optimizer steps.
- [ ] Run the focused test. After the full producer commit is frozen, run vector and image two-task deferred pilots with a success marker and source provenance; keep old LPR raw outputs separate.

## Acceptance

The 2×2 BatchNorm inverse matches a dense paper-formula calculation, and activation-row sampling preserves covariance mass. LPR-only changes must be verified against its paper equations and four-party training path before full reruns.
