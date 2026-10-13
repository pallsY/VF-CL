# Replay-11 deferred pilot implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a two-task, one-epoch pilot that exercises the same formal deferred evaluation transaction as full Replay-11 jobs.

**Architecture:** Add an explicit `pilot_deferred` flag to `replay11_matrix.py` command generation, job records and CLI. Keep the legacy nondeferred `smoke` flag unchanged. A pilot has two tasks/one epoch like smoke and the fixed `run/formal_final.pt` evidence path like a full job.

**Tech Stack:** Python 3.10, `unittest`, existing formal runner.

## Global constraints

- Preserve the 99 full job identities and dataset hashes.
- Pilot and legacy smoke flags are mutually exclusive.
- Pilot results live under a separate results root and cannot enter final summary.
- Formal source provenance, transaction and checkpoint validation are the same as full jobs.

---

### Task 1: Command and record contract

**Files:** `test_replay11_matrix.py`, `replay11_matrix.py`.

- [ ] Add a failing test for `command_for(job, ..., pilot_deferred=True)`: `num_tasks=2`, `epochs_per_task=1`, `formal_deferred_evaluation=1`, UPMC first two custom tasks and four parties; `smoke=True,pilot_deferred=True` raises. Keep existing smoke test asserting `formal_deferred_evaluation=0`.
- [ ] Run the new test red. Add the new keyword and use the existing two-task override for either pilot or smoke; keep `formal_deferred_evaluation=not smoke`.
- [ ] Run command tests green.

### Task 2: Verify deferred pilot output identity

**Files:** `test_replay11_matrix.py`, `replay11_matrix.py`.

- [ ] Add a failing temp-directory fixture for `verify_success_record` with `pilot_deferred=True`, `smoke=False`, `run/checkpoints/formal_final.pt`, exact hashes and a stubbed validator. Assert the validator is called with two tasks and `deferred=True`; ensure `summarize` refuses pilot records.
- [ ] Run red. Route pilot records through the fixed `run` directory, two-task validator and formal checkpoint. Add the flag to `run_one` plan/success, CLI parsing and call sites; use `--pilot-deferred` without changing legacy smoke.
- [ ] Run focused tests green and `git diff --check`; commit.

### Task 3: Linux transaction pilot

**Files:** No source changes unless the real pilot identifies a defect.

- [ ] Push a clean producer commit. Run one vector and one image two-task pilot on an idle Linux GPU with `--pilot-deferred` and separate pilot results root.
- [ ] Require `FORMAL_STATE_FROZEN`, `FORMAL_EVALUATION_PUBLISHED`, `formal_final.pt`, success SHA and exact metrics, then repeat for affected method×dataset cells before their full seed42 runs.

## Acceptance

The pilot command and saved record unambiguously say `pilot_deferred=true`; formal transaction publication and restored evaluation are verified by the same code path as full training, while old `--smoke` remains available for inexpensive nonformal checks.
