# CIFAR-100 Final-Head Replay Intervention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit and compare three 20/class replay arms through the actual Adaptive final-head procedure on fresh seed 50, with an independent fixed BiC readout.

**Architecture:** Reuse the clean producer and seed-49 training-only launcher pattern, changing only seed 50 and the preregistered fresh BiC split. A read-only analyzer recovers/reconstructs A/B/C replay cohorts, reproduces the saved A head exactly, fits B/C with the same source candidate fitter and gate, then evaluates all frozen heads on one cached BiC cohort.

**Tech Stack:** Python 3.9, PyTorch/torchvision, NumPy, unittest, source VF-CL adaptive head modules, CUDA GPU 1.

## Global Constraints

- Producer checkout stays clean at `7bfe6b1d724fb1206bc0053a9008126bad86332d`.
- Fresh seed 50 uses `bic_split_seed=20261011`, 25/class BiC and unchanged lambda-validation split seed `20260729`.
- A/B/C each use exactly 20 raw training images/class and one identical frozen bottom encoder and pre-head.
- BiC holdout is excluded from all candidate/gate fitting and read once after heads are frozen; test loader is never accessed.
- A candidate hashes, gate and installed top must exactly reproduce the original final checkpoint.
- Store only aggregate/per-class counts, hashes and gate summaries; no raw images, IDs, embeddings, logits or predictions.

---

### Task 1: Lock and train fresh seed 50

**Files:** Create `launch_cifar_head_replay_intervention.py`; create `test_cifar_head_replay_launcher.py`.

**Interfaces:** `derive_config(source: dict, root: Path) -> tuple[dict, dict]` registers the exact seed/BiC/deferred/output overrides. The external launcher intercepts only deferred test evaluation, then verifies ten event checkpoints, final freeze, fresh holdout manifests and zero calibration/test-loader access.

- [ ] Write failing test for exact override set, seed 50, `bic_split_seed=20261011`, unchanged head schedule and 20/class budget, plus intentional stop.
- [ ] Run test red; implement launcher from the seed-49 guarded pattern without modifying producer code.
- [ ] Run test green, syntax check, server `--check`, and verify fresh BiC/lambda manifests have 25/class each with no overlap.
- [ ] Commit/push launcher and start a fresh GPU-1 training-only root.

### Task 2: Reconstruct replay cohorts and reproduce A

**Files:** Create `analyze_cifar_head_replay_intervention.py`; create `test_analyze_cifar_head_replay_intervention.py`.

**Interfaces:** `readout_metrics(log_probabilities: torch.Tensor, labels: torch.Tensor) -> dict` yields CIL/TIL/old/new/NLL and class counts. The analyzer CLI accepts run, pinned formal source record/config, and a new output root.

- [ ] Write a failing synthetic metric test covering a task-incremental correct prediction whose global CIL prediction is wrong, and a malformed label/log-probability input.
- [ ] Implement metrics, then use existing exact pixel matcher and event-checkpoint herding logic to rebuild 2,000 A/B/C training-image selections and verify the full eligible pool/holdouts.
- [ ] Load frozen final bottoms and pre-head; embed A saved raw replay and B/C deterministic raw replay. Verify A embeddings against the checkpoint's audit bundle, then refit A with `fit_adaptive_candidates`, canonical CPU gate solving and `install_and_reload_verify`; require exact candidate, gate and installed-head hashes.
- [ ] Fit B/C from the same pre-head/prototypes/task classes and saved lambda-validation embeddings/labels. Cache deterministic BiC holdout embeddings once, evaluate all three heads and save only counts, metrics and hashes atomically.
- [ ] Run local/server unit tests and syntax check, code review, `git diff --check`; commit/push analyzer.

### Task 3: Repeat, report and publish

**Files:** Create `docs/superpowers/reports/2026-10-08-cifar-head-replay-intervention.md`; add aggregate JSON under `docs/superpowers/reports/data/`.

- [ ] Verify seed-50 completion marker/checkpoint hashes and zero calibration/test access; run analyzer twice to separate new roots and require identical JSON SHA-256.
- [ ] Independently recalculate A/B/C CIL/TIL, old/new and NLL from saved aggregate counts/metrics; check evaluation cohort contains exactly 25/class and no training/validation overlap.
- [ ] Report A→B storage-view effect, B→C selection effect and C−A combined effect with single-seed/uncalibrated-head limits and no formal/test claim.
- [ ] Force-add report/data, verify tests and Git status, commit/push branch, open report for user.
