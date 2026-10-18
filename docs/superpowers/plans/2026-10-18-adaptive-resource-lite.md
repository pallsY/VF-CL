# Adaptive Resource-Lite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Test a teacher-free, validation-free, 18-exemplar adaptive variant under the externally anchored resource ceilings without using formal test data for selection.

**Architecture:** Add an opt-in `resource_lite` contract to the isolated sample-only producer. Reuse its raw-example buffer and full-classifier consolidation. Adapt the existing training-only holdout runner so it stops before official test authorization and writes development-only readouts. Keep the later production/auditor/anchor separate from all frozen comparisons.

**Tech Stack:** Python 3.11, PyTorch 2.5.1+cu124, unittest, four-party VFL caches, audited NFS.

## Global constraints

- Do not edit source commits `467611a6`, `9973018`, `e2a559d`, or any frozen result root.
- Budget anchor `codex/resource-budget-anchors` at `38fa927eeb9efddd0ccdf99b8254e892b3460a4b` precedes optimization. The initial candidate has at most 18 raw examples per class, no prototypes, no full frozen teacher, and no extra formal vector validation access.
- Development random seeds are 54/55 on the fixed class/task order. Official test access is prohibited during development; 42/43/44 results cannot select candidates. Confirmation random seeds 45/46/47 and paired baseline reruns require a new frozen protocol.
- All dev outputs use new named roots. Preserve failed trials and never count low-budget smoke runs as formal results.

---

### Task 1: Guard the teacher-free method variant

**Files:** Modify `config.py`, `cl_methods/proto_evolve.py`; create `test_adaptive_resource_lite.py`.

- [ ] Write failing tests for `resource_lite=1`: require ProtoEvolve, 18 raw/class, `full_classifier` final head, sample-only cache, zero KD/PartyKD/SDC/feature-KD/prototype gradient, and reject nonempty prototype/teacher state on reload.
- [ ] Implement minimal opt-in guards and skip creating `_old_bottoms/_old_top` at task boundaries only in this mode. Preserve all existing modes byte-for-byte in behavior.
- [ ] Verify old sample-only/adaptive tests plus new tests on Windows and Linux; commit and push.

### Task 2: Training-only four-party development harness

**Files:** Create `run_resource_lite_pilot.py`, `test_resource_lite_pilot.py`.

- [ ] Port only the holdout/no-test controls from `run_replay_factorial_pilot.py` at commit `99e1da7`, with pinned four-party UPMC cache and seeds 54/55.
- [ ] Write tests that reject two-party UPMC, wrong holdout overlap, early test loader, any formal test marker, and config drift.
- [ ] Generate paired baseline sample-only and resource-lite development commands from one frozen four-party base protocol; require the official test to remain unconsumed.
- [ ] Run tests and a two-task one-epoch smoke for each vector dataset. Commit and push.

### Task 3: Measure candidate resources before accuracy selection

**Files:** Resource audit branch's existing ledger/probe tools and a new dev trial ledger.

- [ ] Run complete-task development trials plus paired iCaRL/DER++ development references on ISOLET and four-party UPMC for seeds 54/55 using only held-out training readouts. Preserve every trial; admit no more than four pre-recorded candidate configurations.
- [ ] Recount raw buffers, empty prototypes/teacher, model tensors, actual validation markers and task-boundary maxima. Probe model and final-head supported-op counts; include omitted-operator caveat.
- [ ] If a budget fails, adjust only the new branch and record the previous trial; never relax the budget. Select at most one candidate by mean paired holdout AA gap across the four vector cells, then lower compute/storage tie-breakers.
- [ ] Check CIFAR development seeds 54/55 against the paired 20-example sample-only reference; reject candidates more than 2 percentage points lower on mean holdout AA.

### Task 4: New confirmation study

**Files:** New production branch, independent audit branch, prelaunch anchor, and new NAS root.

- [ ] Freeze effective epochs, candidate source, data hashes, resource ceilings, validation rule, selected baseline variants, and seeds 45/46/47 before any official test access.
- [ ] Run complete-task low-budget preflights, source/flow/metric checks and seed-45 independent numeric replays, then full candidate and selected-baseline jobs with fail-stop queues.
- [ ] Require per-run resource gates and final independent audit before publishing AA/BWT/Task-IL, memory, access and compute tables.
- [ ] Anchor final ledger/CSV hashes outside the result root and report wins, losses and limitations without selecting from the confirmation test results.
