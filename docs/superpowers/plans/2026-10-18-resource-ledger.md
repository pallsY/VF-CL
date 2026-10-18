# VF-CL Resource Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Produce a read-only, independently checkable resource ledger for the audited 11 baselines, original adaptive, and sample-only adaptive.

**Architecture:** Use frozen checkpoint/config/plan files as authorities. A pure tensor counter classifies learning state, model parameters, and audit-only evidence. A separate CLI binds the counts to pinned results roots and final gates and exports per-cell and dataset summaries without changing historical trees.

**Tech Stack:** Python 3.11, PyTorch 2.5.1, unittest, JSON/CSV, audited NFS results.

## Global constraints

- Do not edit or rerun Replay11 producer `467611a603c9604ad57861920495a3ad05612037`, original adaptive producer `9973018295da270fd6098de07dcf36b9fe65ff61`, or sample-only producer `e2a559dc4945729848225cf21fdc95451762c231`.
- Do not change any existing final gate, ledger, snapshot, checkpoint, or result directory. Write outputs only under a separate resource-audit root or this new branch.
- Read only config/plan/checkpoints/logs and gates for resource accounting; do not use formal test accuracy to select development hyperparameters.
- A final checkpoint file size is not algorithmic memory; audit bundles, tracker/RNG, and source evidence are excluded from learning state.
- Wall time is reported separately and never labelled FLOPs. No strict same-resource claim until persistent and transient memory, compute, and validation access have evidence.

---

### Task 1: Pure payload counter and role classifier

**Files:**
- Create: `resource_accounting.py`
- Create: `test_resource_accounting.py`

**Interfaces:** `payload_bytes(value)` counts unique tensor storage payload; `account_method_state(method, state)` returns explicit raw-example count and bytes, logits, feature statistics, teachers, masks/adapters/other method state, and audit-only bytes. Unknown tensor-bearing fields fail closed.

- [ ] Write synthetic tests for aliasing, nested tensors, different dtypes, raw examples, logits, prototypes, teacher state, and audit-only bundles.
- [ ] Run `python -m unittest test_resource_accounting -v`; observe targeted failures.
- [ ] Implement the smallest explicit classifier covering all methods in the frozen matrix, using their actual `get_state` schemas.
- [ ] Run focused tests and `git diff --check`; commit.

### Task 2: Pinned checkpoint ledger

**Files:**
- Create: `resource_ledger.py`
- Create: `test_resource_ledger.py`

**Interfaces:** `read_cell(dataset, method, seed, authority)` verifies fixed root/commit/gate and returns category counts from final checkpoint plus task-boundary maxima. CLI emits deterministic JSON and CSV with all 3×13×3 cells, source hashes, and no test metrics.

- [ ] Write failing fixture tests for wrong root/commit, absent gate, changed checkpoint hash, wrong party count, duplicate/missing class, and forbidden metric ingestion.
- [ ] Implement read-only loader with restricted checkpoint decode and per-task state measurements.
- [ ] Run tests on Windows and brf3080 Linux. Process seed42 first; inspect any unclassified tensor fields before admitting them.
- [ ] Extend to 42/43/44 and independently verify all 117 cells and exported row counts; commit/push code and ledger hashes.

### Task 3: Compute and validation access inventory

**Files:**
- Create: `resource_compute_inventory.py`
- Create: `test_resource_compute_inventory.py`

**Interfaces:** Bind each frozen command to model/training epochs, batch size, replay/sample use, teacher forwards, final-head steps, and validation access records. Output measured counts and mark unmeasured FLOPs explicitly; no wall-time proxy.

- [ ] Write tests for command identity, 50/50/20 task budgets, final 500-step head, and access-marker checks.
- [ ] Implement source-derived operation counts and a small development-only profiler probe where static counts are insufficient.
- [ ] Run tests and a one-task development probe without touching frozen jobs; preserve runtime and profiler configuration.
- [ ] Commit a concise resource table with categories and limitations.

### Task 4: Freeze the resource comparison target

**Files:**
- Create: `docs/superpowers/specs/2026-10-18-resource-budget-v1.md`

**Interfaces:** Name the primary comparator subset and exact per-dataset ceilings for retained bytes, training-time model state, compute, and validation access. Specify the development seeds/validation-only selection rule and the new confirmation seeds before any optimization trials.

- [ ] Review the ledger for all 13 methods and three datasets; keep heterogeneous or zero-exemplar methods outside unsupported equality claims.
- [ ] Select a feasible pairwise cap that never weakens a baseline; if no feasible cap exists, document a resource-performance frontier instead.
- [ ] Commit and push the budget spec and an external pre-optimization anchor.
- [ ] Begin the separate optimization plan only after the cap is frozen.
