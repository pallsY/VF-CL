# ISOLET Online Head-Memory Capacity Curve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether task-time 20/40/80 raw replay examples per class increase ISOLET final Full-head accuracy under otherwise identical training.

**Architecture:** Extend the external pilot launcher for seeds 51/52, holdout seed 20261012 and capacity 20/40/80. Keep the clean producer unchanged and stop before deferred test access. A read-only analyzer strictly refits the saved Full20 candidate and then fits Full heads using the available task-time replay at each capacity. Aggregate paired metrics and resource costs in a development report.

**Tech Stack:** Python 3.9, PyTorch, existing VF-CL herding and classifier-fit routines, PowerShell/SSH.

## Global Constraints

- Exact source commit `a575bbf446ae501cf8e580ba62c2cc5492a25f30`; selected loss tuple `(0.05,0.10,0.02)`; original task stream and evaluation definition.
- Seeds 51/52; existing gate split 40/class; fresh independent holdout 40/class with seed 20261012; no test-loader iteration.
- Capacity is real task-time persistent raw memory and must be measured, not simulated by revisiting old training data after the stream.
- The source's fixed Full20 result is a validation anchor; external refits at larger capacities are development-only until production audit changes are reviewed.

---

### Task 1: Development capacity capture

**Files:** Modify `launch_isolet_replay_selection_pilot.py`; update `test_isolet_replay_selection_pilot.py`.

- [ ] Write failing tests for seed51/52 + holdout seed20261012 and capacity20/40/80 config overrides; reject all other capacity/protocol changes.
- [ ] Allow the one development-only adaptive validation exception for capacity while validating every other adaptive option.
- [ ] Check `--check` on all six jobs, then run each seed/capacity in an isolated result root.
- [ ] Verify per-class raw replay count, frozen checkpoint and 13 event hashes, identical paired pre-head state, holdout/gate identity and zero test access. If the strict checkpoint audit rejects a capacity, stop and implement task-time shadow capture without disabling the audit.

### Task 2: Paired frozen-head refits and report

**Files:** Create `analyze_isolet_online_head_memory.py`, focused test, report and aggregate JSON under `docs/superpowers/reports`.

- [ ] Verify the 20/class refit reproduces the frozen Full candidate hash for each run.
- [ ] From each saved pre-head state and corresponding task-time raw replay, fit balanced Full heads on 20/40/80 current-encoder embeddings with the same Adam/500-step/LR/regularization settings.
- [ ] Evaluate only on the independent training holdout; record paired CIL/newest/task-ID metrics, old-to-old confusion, memory bytes/count, fit time, source/manifest/checkpoint hashes.
- [ ] Apply the locked resource-screen rule, distinguish development accuracy from formal test AA_final, run focused verification, review and push the report.
