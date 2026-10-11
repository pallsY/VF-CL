# Adaptive40 Fixed-Head Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce audited Full40 and Bias40 formal controls under the existing Adaptive40 three-dataset protocol.

**Architecture:** Reuse `ProtoEvolveCL`'s existing 40/class fixed-head fit paths. Restrict late formal head installation to Adaptive, bind nondefault fixed-head capacity to checkpoint protocol, then lock and run 18 new Full/Bias configurations. Preserve all 20/class and Adaptive40 artifacts.

**Tech Stack:** Python, PyTorch, unittest, existing VF-CL formal deferred runner, SSH-hosted RTX 3080/4090.

## Global Constraints

- Same selected loss tuple `proto_lambda_a=0.05`, `distill_weight=0.10`, `feat_distill_weight=0.02`.
- Full40: final-only head, `lr=0.01`, `steps=500`, regularization `0.01`, 40 raw examples/class.
- Bias40: final-only head, `lr=0.03`, `steps=600`, class/task regularization `0.01/0.01`, task weight `1.3`, 40 raw examples/class.
- Keep dataset files/hashes, validation split, task stream, seed 42/43/44, optimizer, epochs and formal `AA_final` formula fixed.
- Use the same host and physical GPU as the corresponding Adaptive40 seed. Preserve old 20/class checkpoint protocol bytes.
- No formal test access during code development or smoke. A published formal test may never be repeated.

---

### Task 1: Fix formal fixed-Bias installation and capacity identity

**Files:** `runner.py`, `adaptive_consolidation_audit.py`, `data_utils.py`, `test_fixed40_formal_controls.py`.

- [ ] Add a failing test showing a formal `task_class_bias` method fits exactly once in `after_task` and `_finalize_formal_internal_state` does not call it again. Use a tiny method spy and a dataset that raises on unexpected validation access.
- [ ] Add a failing test showing enabled fixed Full/Bias capacity 40 appears in `_checkpoint_protocol`, while capacity 20 keeps the old protocol without that field.
- [ ] Add a failing test showing fixed Full/Bias can reach the formal test authorization phase without a validation record when BiC is off, while Adaptive still requires its validation phase.
- [ ] Run `python -m unittest -q test_fixed40_formal_controls`; confirm both failures are from the missing behavior.
- [ ] Change `_is_formal_internal_method` to accept only `adaptive_dual_branch`. Change the matching internal-method checks in `adaptive_consolidation_audit.py` and the formal validation requirement in `data_utils.py` from “any enabled/non-Full head” to exactly Adaptive. Extend `_checkpoint_protocol` to record capacity 40 for enabled fixed Full/Bias modes; keep 20 omitted.
- [ ] Run the focused tests, existing Adaptive 20/40 tests and formal deferred audit tests on Linux; verify no v1 or Adaptive40 regression.

### Task 2: Exercise real 40/class fixed heads without formal test access

**Files:** one new training-only launcher outside the source worktree; save run identities and audit evidence in a new results root.

- [ ] Create a clean isolated source worktree on each host at the reviewed control commit and verify the dataset hashes and locked loss tuple.
- [ ] Run one ISOLET smoke per fixed mode, checking the formal test-phase transition and then stopping before authorization or iteration. Require 13 task checkpoints, exactly 40 persistent raw examples/class, exactly one final fit, matching fit sample count and no `results.json` or test-loader record.
- [ ] Compare fixed branch fit settings and the pre-fit model identity with Adaptive40 where the saved evidence permits. Fix a real failure before any formal launch; do not alter source/config after formal results begin.

### Task 3: Lock and launch 18 formal jobs

**Files:** `docs/superpowers/reports/data/adaptive40-fixed-head-locked-manifest-20261011.json`, 18 complete configs and launch scripts saved with the experiment.

- [ ] Derive each Full40/Bias40 config from the same-seed Adaptive40 config. Whitelist only head mode, branch-specific head LR/steps and fresh result names/paths; hash and commit all configs before testing.
- [ ] Keep ISOLET on 3080 GPU 1, CIFAR-100 on 3080 GPU 0 and UPMC clean image on 4090 GPU 0. Sequentially queue seeds 42/43/44 within each arm. Start a new CIFAR job only with sufficient free disk for one complete job plus a safety margin; pause the queue if the guard fails.
- [ ] For each completed run, require `exit.code=0`, `FORMAL_STATE_FROZEN.json`, `FORMAL_EVALUATION_PUBLISHED.json`, `results.json`, bound hashes, one fixed-head fit and 40 raw replay examples/class. Diagnose before any recovery; never repeat a published test.

### Task 4: Compare and report

**Files:** new report and machine-readable ledger under `docs/superpowers/reports`.

- [ ] Summarize per-seed and mean ± sample SD `AA_final`, BWT, replay memory and runtime for Adaptive40, Full40 and Bias40; state any source-commit and GPU differences.
- [ ] Distinguish memory-capacity benefit from the learned gate's independent contribution. The prior 20/class comparators were produced by older source commits; a same-commit 20/class rerun remains a separate control.
- [ ] Commit and push the final report to `codex/isolet-hparam-sweep`, then notify the user.
