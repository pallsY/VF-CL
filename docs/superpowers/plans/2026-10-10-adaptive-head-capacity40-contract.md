# Adaptive Head Capacity 40 Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support a real version-2 Adaptive 40/class head while preserving the version-1 20/class method.

**Architecture:** A capacity-to-version helper controls candidate fitting and immutable evidence. The model, CL method, runner resume checks, source provenance and checkpoint auditor use this common version. New tests exercise both capacities and reject cross-version or replay-count corruption. Existing v1 path remains default and byte compatible.

**Tech Stack:** Python, PyTorch, existing VF-CL audit and unittest modules.

## Global Constraints

- v1 means 20/class and retains exact historical candidate-config dictionaries and method/top version 1.
- v2 means 40/class; both Full and Bias fit 40/class, candidate configs record capacity, strict audit checks raw/embedded counts.
- Do not change dataset split, tasks, selected loss tuple, gate algorithm or baseline implementations. No formal test use in development.

---

### Task 1: Versioned candidate contract

**Files:** Modify `adaptive_head_consolidation.py`; tests in `test_adaptive_head_consolidation.py` and `test_adaptive_dual_branch_validation.py`.

- [ ] First add failing tests for version mapping 20→1, 40→2, rejection of other capacities, unchanged v1 candidate config, exact v2 `samples_per_class=40` fields and 40/class fit audits.
- [ ] Add the smallest capacity/version/config helper and pass capacity into both candidate fit functions, defaulting to 20 for existing callers.
- [ ] Re-run focused tests, confirming v1 output/state evidence did not change.

### Task 2: Model installation and method state

**Files:** Modify `models.py`, `cl_methods/proto_evolve.py`, `config.py`, `runner.py`; focused model/proto/resume/config tests.

- [ ] First add failing tests that accept a v2 mixed top and reject mismatched v1/v2 state or invalid capacity, while v1 remains valid.
- [ ] Pass configured capacity/version through Full/Bias fit, installed top, result/history/bundle, saved method state and resume checks. Keep fixed endpoint v1 behavior.
- [ ] Update production config validation to accept exactly 20 or 40 in adaptive mode; reject other capacities. Run focused tests.

### Task 3: Strict provenance and audit

**Files:** Modify `adaptive_consolidation_audit.py`, relevant `three_dataset_formal_driver.py` validation and focused audit/driver tests.

- [ ] First add failing v2 provenance/checkpoint tests for source version, candidate configs, top version and exactly 40 raw+re-encoded examples/class; preserve v1 fixtures.
- [ ] Make run provenance and audit infer/verify version from configured capacity and require every v2 evidence field to agree.
- [ ] Run v1/v2 focused tests plus a fresh 40/class training-only ISOLET smoke; stop before test access and verify strict freeze audit.

### Task 4: Delivery

**Files:** Add concise implementation report under `docs/superpowers/reports`.

- [ ] Verify default20 behavior, all relevant tests, diff whitespace, checkpoint freeze and no test-loader access.
- [ ] Review, commit and push the versioned code; state that formal accuracy and matched-budget baselines remain to be run.
