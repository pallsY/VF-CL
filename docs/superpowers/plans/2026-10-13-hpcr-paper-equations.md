# HPCR Paper Equations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the task-based four-party HPCR baseline with paper Eq. 18 and 21–26 while preserving the approved dataset-specific training budgets.

**Architecture:** Keep HPCR math and the method-specific cosine classifier in `cl_methods/hpcr.py`. Override only this method's top-model forward during `before_task`, so shared training and evaluation both use cosine logits without changing `models.py` or `vfl_trainer.py`. Retain the bounded per-class replay store as a disclosed task-based VFL adaptation and insert each incoming example only on its first epoch pass.

**Tech Stack:** Python, PyTorch, `unittest`.

## Global Constraints

- Edit only `cl_methods/hpcr.py`, `test_hpcr_vfl.py`, and HPCR-only options in `replay11_matrix.py` if needed.
- Do not change shared trainer, runner, or model modules; do not stage, commit, or push.
- Keep CIFAR100 50, ISOLET 50, and UPMC 20 epochs per task and the existing per-class replay budget.
- Treat the paper equations as canonical. Eq. 24's printed minus sign conflicts with its minimizing-distance prose; use a positive squared-error penalty and document this in code.

---

### Task 1: Contrastive objective and temperature

**Files:** Modify `test_hpcr_vfl.py`, `cl_methods/hpcr.py`.

**Interfaces:** `proxy_supcon_loss(features, proxies, labels, temperature=0.09, min_samples=20)` computes Eq. 18; `HPCRCL._contrastive_scale()` returns `tau / tau(step)` from Eq. 21–22.

- [ ] Write tests comparing a tiny two-class batch to an independent scalar `logaddexp` calculation with sample gate off/on; test first-batch gradient without replay; test scale at steps 0, 125, and 250 for cycle 500, `tau=.09`, `tau_max=.16`, `tau_min=.05`.
- [ ] Run `python -m unittest test_hpcr_vfl.HPCRLossTest -v`; record expected failures in the new equation tests.
- [ ] Add the conditional non-self sample-similarity term to numerator/denominator and the exact reciprocal cosine schedule. Train on current augmented samples even with empty memory.
- [ ] Rerun the targeted tests to green.

### Task 2: PCD and SCD

**Files:** Modify `test_hpcr_vfl.py`, `cl_methods/hpcr.py`.

**Interfaces:** `proxy_logit_distillation_loss(current, stored, batch_labels)` uses the finite, batch-selected class columns with multiplicity; `similarity_distillation_loss` follows the printed Eq. 25 distribution orientation.

- [ ] Write tests where selected labels `[0,0,1]` give twice class 0's squared-error contribution, absent class 2 has zero gradient, and a three-sample SCD output equals a manual non-self `p_current * log(p_stored)` computation.
- [ ] Run the targeted tests and record the expected failures.
- [ ] Implement positive selected-column squared error (Eq. 24's minus is inconsistent with minimization) and printed SCD orientation; keep old logits/features detached.
- [ ] Rerun tests to green.

### Task 3: Cosine classifier and state integration

**Files:** Modify `test_hpcr_vfl.py`, `cl_methods/hpcr.py`.

**Interfaces:** `HPCRCL.before_task` installs method-only cosine, bias-free top-model forward used by both trainer and evaluation; `_forward` stores the same logits; `get_state/load_state` preserve method counters.

- [ ] Write a test proving training/stored/evaluation logits equal `normalize(features) @ normalize(weight).T / tau`, remain invariant to positive feature/proxy rescaling, and ignore classifier bias. Add a two-task four-party check for aligned split inputs, memory cap, restored state, and inference.
- [ ] Run targeted tests and record red output.
- [ ] Bind the HPCR-specific cosine forward in `before_task`, freeze/zero classifier bias, and insert examples only on the first epoch pass without wrapping the shared data loader.
- [ ] Rerun targeted tests and existing HPCR tests to green.

### Task 4: Final verification

**Files:** No additional files unless a test exposes a defect.

- [ ] Run `python -m unittest test_hpcr_vfl -v` and the smallest existing matrix/config smoke test covering HPCR.
- [ ] Inspect `git diff --check` and `git diff -- cl_methods/hpcr.py test_hpcr_vfl.py replay11_matrix.py`.
- [ ] Report red/green command outputs, remaining taskified VFL departures, and any inability to isolate the cosine head.
