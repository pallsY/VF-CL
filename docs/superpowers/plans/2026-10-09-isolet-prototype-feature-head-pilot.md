# ISOLET Prototype-Feature Head Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test one fixed prototype-generated head augmentation against the exact current Full head on two fresh ISOLET development seeds.

**Architecture:** Extend the external training-only pilot launcher to accept seeds 49/50 and a pilot holdout seed while preserving old 47/48 defaults. Train only the existing herding stream; a separate read-only script reproduces the frozen Full head, fits the augmented Full head from saved mean/std and raw replay, and scores both on a disjoint holdout. No producer worktree edit or formal test access.

**Tech Stack:** Python 3.9, PyTorch, NumPy, existing VF-CL head/prototype routines, PowerShell/SSH.

## Global Constraints

- Producer commit `a575bbf446ae501cf8e580ba62c2cc5492a25f30`; fixed tuple `(0.05,0.10,0.02)`; 20 persistent raw samples/class, 13 tasks, same optimizer and training schedule.
- Seeds 49/50; original gate holdout 40/class; new independent training holdout 40/class with split seed `20261011`.
- Candidate uses exactly 20 temporary prototype-generated features/class in the final head fit; no persistent synthetic cache.
- Predeclared pass rule: both CIL deltas positive, mean at least +1.00 pp, newest-task delta at least −1.00 pp in each seed.

---

### Task 1: Fresh training-only streams

**Files:** Modify `launch_isolet_replay_selection_pilot.py`; update `test_isolet_replay_selection_pilot.py`.

**Interfaces:** Existing `--seed` additionally accepts 49/50; new `--holdout-seed` defaults to `20261010` for historical runs and accepts `20261011` for this pilot.

- [ ] Write a failing test for accepted seed 49/50 and deterministic, disjoint holdout seed `20261011`.
- [ ] Extend the launcher without changing 47/48 default behavior; test and compile it.
- [ ] Run `--check` for both new seeds, record config/data and gate/holdout identities, then train herding seeds 49/50 sequentially.
- [ ] Verify frozen checkpoints, 13 events, exactly 20 raw replay/class, zero test-loader iterations and matching fresh holdout manifests.

### Task 2: Paired Full-head capacity intervention

**Files:** Create `analyze_isolet_prototype_feature_head.py`; create `test_analyze_isolet_prototype_feature_head.py`.

**Interfaces:** The script consumes the two frozen training-only checkpoints and produces one aggregate JSON file with saved/refit Full20 and fixed raw20+synthetic20 scores.

- [ ] Test deterministic per-class feature generation from mean/std, 20/class count and absence of persistent synthetic output.
- [ ] Verify a 20/class refit exactly matches each saved Full-head state hash before fitting the candidate from the same pre-head state.
- [ ] Compare both heads on the new independent training holdout; save only aggregate counts/metrics, fit time and hashes.
- [ ] Independently recompute paired CIL/newest deltas, apply the locked pass rule and report memory/compute differences.

### Task 3: Record outcome

**Files:** Create `docs/superpowers/reports/2026-10-09-isolet-prototype-feature-head-pilot.md` and `docs/superpowers/reports/data/isolet-prototype-feature-head-pilot-20261009.json`.

- [ ] Explain development-only status, prior art, paired results and whether production integration is warranted.
- [ ] Run focused tests, JSON parse, artifact-hash checks and `git diff --check` before commit/push.
