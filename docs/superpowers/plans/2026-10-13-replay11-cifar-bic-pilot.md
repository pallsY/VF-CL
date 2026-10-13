# CIFAR BiC preflight implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exercise the full ten-task CIFAR100 BiC and validation contract at one epoch per task before any corrected method runs fifty epochs per task.

**Architecture:** Add `pilot_bic` to Replay-11 command generation and job records. It preserves the full CIFAR task count, BiC holdout, validation holdout and formal deferred evaluation; it only changes epochs per task to one. It is allowed only for CIFAR100 and cannot be combined with `smoke` or `pilot_deferred`. Use a separate result root and keep pilot records out of final summaries.

**Tech Stack:** Python 3.10, `unittest`, existing formal runner.

## Global constraints

- Preserve authoritative CIFAR100 hashes, four-party partition, seeds and full-result command.
- The two-task deferred pilot remains BiC-free because the existing BiC protocol requires ten tasks.
- No pilot score enters the 99-item final table.

---

### Task 1: Command contract

**Files:** `test_replay11_matrix.py`, `replay11_matrix.py`.

- [ ] Add a failing test that `command_for(('cifar100','hpcr',42), ..., pilot_bic=True)` keeps ten tasks, BiC and validation enabled, formal deferred evaluation enabled, and sets `epochs_per_task=1`. Assert non-CIFAR and mutually combined pilot modes reject the flag.
- [ ] Run red; add the smallest conditional override for the epoch setting; run green.

### Task 2: Saved record and summary contract

**Files:** `test_replay11_matrix.py`, `replay11_matrix.py`.

- [ ] Add a failing record test proving a `pilot_bic=True` result validates `run/checkpoints/formal_final.pt` with ten tasks/deferred true, and cannot enter `summarize`.
- [ ] Run red; thread the flag through `run_one`, saved plan/success, `verify_success_record`, and CLI `--pilot-bic`; run green.
- [ ] Run all Replay-11 method/matrix tests, `git diff --check`, commit and push.

### Task 3: Server preflight

**Files:** No source change unless the formal ten-task pilot finds a reproducible defect.

- [ ] Run CIFAR100 HPCR seed42 `--pilot-bic` on an idle GPU under a new result root, check every formal transaction marker and success hash.
- [ ] Record wall time and GPU memory. If it fails, retain all output and diagnose before launching any full 50-epoch corrected CIFAR job.

## Acceptance

The BiC pilot exercises all ten tasks with the original fixed calibration and validation splits and the same frozen evaluation path as full jobs. Its one-epoch command and pilot identity are explicit in the saved record.
