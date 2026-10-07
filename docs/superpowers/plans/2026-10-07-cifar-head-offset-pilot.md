# CIFAR-100 Frozen-Encoder Head Offset Pilot Plan

> **For agentic workers:** Execute in order with TDD for reusable logic. The formal source result root is read-only.

**Goal:** Run one fresh seed-45 CIFAR-100 development stream and test one training-only scalar new-task head offset on its independently reserved validation split.

**Architecture:** A small launcher derives and records exact deviations from the audited seed-42 Adaptive config, then calls the existing runner with no final head consolidation. A separate offline analyzer loads the final checkpoint, selects 20 nonvalidation training images per class under a deterministic evaluation transform, fits one bounded scalar offset to frozen logits, and evaluates baseline versus offset on validation once.

**Tech Stack:** Python, PyTorch, SciPy, unittest, existing VF-CL runner.

## Global constraints

- Producer source commit `7bfe6b1d724fb1206bc0053a9008126bad86332d`; isolated worktree and fresh result root.
- Exact config overrides are those listed in `docs/superpowers/specs/2026-10-07-cifar-head-offset-pilot-design.md`; all others are unchanged.
- No test loader, published-root write, bottom update after training, validation-label fitting, or post-result adjustment of the scalar objective/bounds/thresholds.
- Required checks: source and output hashes, train/validation disjointness, 20 calibration rows per class, finite logits, unchanged Task-IL predictions, and new-root-only writes.

---

### Task 1: Freeze the pilot protocol and launcher

**Files:** Create `launch_head_offset_pilot.py`; test `test_head_offset_pilot.py`.

- [ ] Write a failing test for exact config override allowlist, CL-only timeline, no Adaptive head, new validation seed, and a new output root.
- [ ] Implement the minimal launcher using the saved config and existing `runner.run_experiment`; write the derived config and source hashes before training.
- [ ] Run focused tests and a read-only config/preflight check on 3080; commit the reviewed source and launch once.

### Task 2: Offline scalar correction

**Files:** Create `analyze_head_offset_pilot.py`; extend `test_head_offset_pilot.py`.

- [ ] Write failing tests for 20-per-class train-only selection and a controlled two-task logit example whose optimal scalar is positive, with frozen model state.
- [ ] Implement bounded scalar CE minimization on cached training-only logits, baseline/shifted validation metrics, and the fixed success rule; reject missing/nonfinite/overlapping evidence and a bound-hitting optimum.
- [ ] Run focused tests and a synthetic checkpoint/data smoke; commit.

### Task 3: Execute, audit, and report

**Files:** Create `docs/superpowers/reports/2026-10-07-cifar-head-offset-pilot.md`.

- [ ] Wait for seed-45 training completion, preserving logs and checkpoints; verify no test access and all old/new task events.
- [ ] Run the analyzer once on validation, save aggregate JSON to a separate new root, and verify provenance, no training-state change, and the prespecified pass/fail rule.
- [ ] Report numerical results, limitations, and the next decision. Preserve failed or inconclusive roots; do not launch seeds 46/47 or formal test automatically.
