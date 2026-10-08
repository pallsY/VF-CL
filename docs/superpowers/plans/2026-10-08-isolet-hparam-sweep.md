# ISOLET Adaptive Hyperparameter Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the approved one-factor ISOLET Adaptive sweep, record every candidate and select the best seed-45 tuple, then verify it against the formal tuple on fresh seed 46.

**Architecture:** Keep the exact formal producer source detached in a new 3080 worktree. A small external single-run launcher derives configs from the pinned formal ISOLET seed-42 config, checks data/manifest hashes, trains one candidate until final checkpoint freeze, stops before test access and scores the frozen adaptive top on its existing lambda-validation cache. The root agent runs seven seed-45 candidates sequentially and two paired seed-46 candidates, then writes a checked aggregate ledger/report.

**Tech Stack:** Python 3.9, PyTorch, existing VF-CL modules, unittest, Git, masked NVIDIA GPU 1.

## Global Constraints

- Exact producer commit `a575bbf446ae501cf8e580ba62c2cc5492a25f30`; clean source worktree.
- Exact ISOLET NPZ and metadata SHA-256 from the frozen formal record; only paths differ across hosts.
- The 13-task, 26-class, 4-party, 20/class-memory protocol and validation manifest remain unchanged.
- Candidate grid is fixed: `proto_lambda_a` 0.05/0.15/0.30, `distill_weight` 0.10/0.25/0.50, `feat_distill_weight` 0.02/0.05/0.10.
- Only the method's own runs are launched; baseline method outputs and formal roots remain read-only.
- No final test loader iteration or test-label use for scoring. The vector NPZ loader still materializes fixed test arrays at initialization. Validation scores use the same cohort as the Adaptive gate and are development evidence only.

---

### Task 1: Guarded single-run launcher

**Files:** Create `launch_isolet_hparam_trial.py`; create `test_isolet_hparam_trial.py`.

**Interfaces:** `derive_config(source, root, seed, proto_lambda_a, distill_weight, feat_distill_weight)` returns a config and explicit overrides, rejecting all changes outside the registered path/device/seed/deferred/three-knob set. `readout(log_probabilities, labels)` reports CIL, TIL, old/new CIL and NLL on all 40/class validation rows.

- [ ] Write failing tests for exact override set, default/changed knob propagation, the one-dimensional ranking tie-break, and CIL-vs-TIL metrics.
- [ ] Run tests red; implement the launcher using the existing seed-50 training-only stop pattern and source model `TopModel` for frozen-cache scoring.
- [ ] Run tests green, syntax check and server `--check`; verify source/data hashes, exact manifest, GPU/disk admission and zero test access.
- [ ] Commit/push launcher before starting candidate runs.

### Task 2: Development seed-45 sweep

**Files:** New result root on 3080; no source modifications.

- [ ] Run formal tuple `(0.15, 0.25, 0.05)` once as the within-host baseline.
- [ ] Run `proto_lambda_a=0.05` and `0.30`; choose best by final validation CIL, tie-breaking old-class CIL, Task-IL and distance from formal values.
- [ ] Hold chosen `proto_lambda_a`; run `distill_weight=0.10` and `0.50`, reusing `0.25` candidate; select stage best.
- [ ] Hold chosen first two values; run `feat_distill_weight=0.02` and `0.10`, reusing `0.05` candidate; select final tuple.
- [ ] Validate every completion marker/config/checkpoint/hash and zero test-loader access; preserve all candidate roots.

### Task 3: Paired seed-46 confirmation and publication

**Files:** Create `docs/superpowers/reports/2026-10-08-isolet-hparam-sweep.md`; add compact aggregate JSON under `docs/superpowers/reports/data/`.

- [ ] Run formal tuple and selected tuple on seed 46 in separate new roots, with identical host/data/split settings.
- [ ] Build a ledger of all seven seed-45 candidates and both seed-46 runs, including parameters, CIL/TIL/old/new/NLL/gate and provenance hashes. Recompute stage choices from the ledger.
- [ ] Report the selected tuple and whether it improves the paired confirmation; state the same-cohort gate/selection and cross-GPU limitations. Do not infer formal test improvement.
- [ ] Force-add report/data, verify tests and clean source worktree, commit/push branch and open report.
