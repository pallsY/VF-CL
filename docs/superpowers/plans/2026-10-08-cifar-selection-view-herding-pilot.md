# CIFAR-100 Task-Time Selection-View Herding Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare original-image sets selected at the same task boundary by stochastic versus deterministic herding on fresh seed 49, under a common final encoder and view.

**Architecture:** Keep producer source unchanged. Reuse the guarded seed-48 training-only launcher pattern for seed 49. A read-only analyzer reconstructs deterministic shadow selections from each event checkpoint, recovers stochastic replay IDs with the exact CIFAR matcher, and evaluates paired centroid errors under the frozen final bottoms.

**Tech Stack:** Python 3.9, PyTorch/torchvision, NumPy, unittest, existing VF-CL modules, CUDA GPU 1.

## Global Constraints

- Producer source stays clean at `7bfe6b1d724fb1206bc0053a9008126bad86332d`.
- Seed 49 is a fresh development seed, with no test-loader access or accuracy claim.
- Both task-time selectors use the same 450 eligible training images/class and select 20 original-image IDs/class.
- Shadow deterministic selections do not affect training, replay memory or head consolidation.
- Final S/D/O comparison uses the same final bottoms, deterministic image view and 450-image class target.
- Output only class-level errors, counts and hashes to new roots; two analysis outputs must be byte-identical.

---

### Task 1: Lock and launch seed 49

**Files:** Create `launch_cifar_selection_view_pilot.py`; create `test_cifar_selection_view_launcher.py`.

**Interfaces:** `derive_config(source: dict, root: Path) -> tuple[dict, dict]` allows only the five seed/output/deferred-evaluation overrides; `TrainingOnlyStop` intercepts the runner's deferred test call after `adaptive_final.pt` and `ADAPTIVE_STATE_FROZEN.json` are written.

- [ ] Write a failing test for exact override keys, seed 49, unchanged adaptive mode and 20/class memory, plus the intentional stop.
- [ ] Run the test red; implement the smallest guarded launcher using the seed-48 launcher as the reference.
- [ ] Run tests green, syntax check, and server `--check` against the pinned formal seed-42 record/config/data and exact clean producer commit.
- [ ] Commit/push launcher and spec, then start a fresh seed-49 training-only root on physical GPU 1.

### Task 2: Reconstruct and compare selections

**Files:** Create `analyze_cifar_selection_view_pilot.py`; create `test_analyze_cifar_selection_view_pilot.py`.

**Interfaces:** `summarize_classes(rows: list[dict]) -> dict` returns means, medians, paired S−D/D−O gaps and positive fractions for aggregate and four party spaces. The analyzer CLI accepts the seed-49 run, pinned formal source record/config, and a new output root.

- [ ] Write a failing synthetic test that includes a negative S−D difference and rejects incomplete class panels.
- [ ] Implement summary. Reuse `CifarReplayMatcher`, `herding_indices`, `embed_batches`, normalized centroid error and source-hash helpers already in the repository.
- [ ] For task t, verify `event_t_CIL.pt` schema, task/classes, final completion hash, then embed only classes 10t–10t+9 from the eligible deterministic train view with that checkpoint's bottoms and choose 20/class. Record task-boundary S/D errors. Restore no model state into training.
- [ ] Under final bottoms, embed all 45,000 eligible deterministic training rows once, compare S/D/O at 20/class in aggregate and party spaces, and save class rows/counts/hashes atomically. Reject distinct-original ambiguous replay, holdout overlap or any protected loader access.
- [ ] Run focused tests, Python syntax check, `git diff --check`, code review, and commit/push analyzer.

### Task 3: Verify, report and publish

**Files:** Create `docs/superpowers/reports/2026-10-08-cifar-selection-view-herding-pilot.md`; add aggregate JSON under `docs/superpowers/reports/data/`.

- [ ] Verify the training-only completion marker binds all ten event checkpoints and the final checkpoint, with zero test-loader accesses.
- [ ] Run the frozen analyzer twice to separate new roots; compare JSON SHA-256 and independently recalculate per-class summaries.
- [ ] Explain S/D/O, the final S−D sign and magnitude, task-boundary sanity check, party results, single-seed limit and why no CIL/TIL claim follows.
- [ ] Force-add the report and aggregate JSON, verify tests and Git status, commit/push the branch, and open the report for the user.
