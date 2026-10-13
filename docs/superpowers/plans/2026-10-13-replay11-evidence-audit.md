# Replay-11 Evidence Audit Implementation Plan

> **For agentic workers:** Use executing-plans inline; producer commit 99e81b28 and every experiment result remain read-only.

**Goal:** Build a portable integrity verifier for all 99 jobs and an independently tested frozen-state replay for the 33 preselected seed-42 cells.

**Architecture:** Add audit-only modules on branch `codex/replay11-audit-v2`. The verifier reads job directories and archived bytes without invoking the inode-bound producer transaction. The later strict replayer runs in a separate process against producer code 99e81b28 and emits reports outside experiment roots.

**Tech Stack:** Python standard library, existing PyTorch runner, `unittest`.

## Global Constraints

- Matrix: 3 datasets × 11 methods × seeds 42/43/44; strict replay selection is seed 42 for each method × dataset.
- Producer code, data, checkpoints, run directories, and success records cannot be edited by audit code.
- Missing or altered evidence fails closed. OOM/environment incompatibility is `inconclusive`, never a pass or a method-score failure.
- Byte-identical archives require a source-to-copy SHA-256 attestation. A copied file cannot claim its original inode.
- Metric equations are frozen by producer `replay11_matrix.metrics_from_history(..., deferred=True)`.

---

### Task 1: Portable single-job integrity check

**Files:** Create `replay11_evidence.py` and `test_replay11_evidence.py`.

**Interface:** `verify_local_job(job_dir: Path) -> dict` checks plan/success identity, config, authoritative dataset hashes recorded in the plan, four parties, complete task count, frozen transaction markers, final checkpoint/results SHA-256, and recomputed three metrics. `verify_archive_manifest(job_dir: Path, manifest: Path) -> dict` checks the byte tree and source-to-copy attestation.

- [ ] Write synthetic fixture tests for a valid job plus wrong seed, two-party config, changed result/checkpoint bytes, bad data hash, missing formal marker, wrong metric, and bad archive tree.
- [ ] Run `python -m unittest -v test_replay11_evidence`; see failures from the missing verifier.
- [ ] Implement only the checks needed for those tests, reusing `replay11_matrix.jobs`, `expected_data`, and `metrics_from_history`; do not call inode-bound `prepare_formal_deferred_evaluation` on a copy.
- [ ] Run the focused suite, `python -m unittest -q test_replay11_matrix`, and `git diff --check`; commit the audit-only files.
- [ ] Read-only validate one original MIR42 and one archived UPMC job; compare each with its success record.

### Task 2: Exact matrix inventory

**Files:** Modify `replay11_evidence.py` and `test_replay11_evidence.py`.

**Interface:** `merge_attestations(records: Iterable[dict]) -> dict` rejects missing/duplicate cells, mixed producer commits, smoke records, and unexpected methods; returns 99 job identities and the exact 33 seed-42 strict selections.

- [ ] Write failing tests for a complete catalog, one missing cell, a duplicate, mixed commits, and the 33-cell selection.
- [ ] Run red tests, implement the small merger, and run green tests plus `git diff --check`; commit.
- [ ] Collect attestations on each original host and archived root. Preserve source paths; do not claim 99 complete before all pass.

### Task 3: Independent numerical replay

**Files:** Create `replay11_strict_audit.py` and `test_replay11_strict_audit.py`.

**Interface:** A separate process validates source-to-copy byte evidence, restores frozen model/method state under producer code 99e81b28, checks validation/test isolation, re-evaluates the fixed data, and compares checkpoint/tracker/metrics using the existing exact value-comparison rule. All 11 methods need explicit state contracts; unknown methods fail.

- [ ] Write positive and mutation tests for method state, snapshot bytes, test access, and stored metrics; verify red failures.
- [ ] Implement only the frozen-state replay paths required by fixtures; verify green tests without changing producer code.
- [ ] Pilot real MIR42 on its original host and one archived UPMC plus ISOLET run in matching environments; record duration, peak memory, and exact failure reason.
- [ ] On pilot failure, preserve evidence and diagnose; do not relax metric equality or restart the held queue.
- [ ] After pilot success, replay each of the 33 preselected seed-42 cells exactly once and save immutable reports.

### Task 4: Method fidelity and training release

**Files:** Create `docs/replay11-method-fidelity.md`; update `REPLAY11_STATUS.md` only after evidence exists.

- [ ] Compare the seven new method modules with their primary papers and record each core algorithm step and VFL deviation.
- [ ] Run a two-task, one-epoch *deferred-evaluation* CIFAR pilot for aL-SAR, ER-ACE, and AdaGauss.
- [ ] Start their seed-42 full runs only after pilot and audit-path success; release seeds 43/44 only after the corresponding method × dataset gate passes.
- [ ] Generate the final table from 99 accepted basic attestations, and report 33 independent numerical replays separately.

## Completion Gate

Require 99 basic attestations, exactly 33 independent replay reports, seven fidelity reviews, and zero unresolved inconclusive cells before claiming the replay-11 audit complete. Preserve all failed reports and both v1/v2 protocol versions.
