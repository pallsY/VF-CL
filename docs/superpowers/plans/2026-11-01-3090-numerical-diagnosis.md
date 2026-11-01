# 3090 numerical diagnosis Implementation Plan

> **For agentic workers:** Use subagent-driven-development for isolated implementation and independent review. Steps use checkbox syntax.

**Goal:** Quantify the previously observed FIM-mean hash mismatch without modifying any formal queue or relaxing its historical rejection.

**Architecture:** Add a standalone wrapper around the unchanged GPU admission smoke to capture raw FIM tensors and actual reductions. Replay identical inputs on the reference GPU, reconstruct both original evidence hashes, and compare under the separately locked diagnostic rules.

**Tech Stack:** Existing Python, torch, numpy, stdlib/unittest; no new dependencies.

## Global Constraints

The spec is `docs/superpowers/specs/2026-11-01-3090-numerical-diagnosis.md`. Diagnostic atol=1e-12 and rtol=1e-6; selections exact. Never edit existing smoke/admission/training/queue files, formal manifests or outputs. Capture/replay only in fresh allowlisted admission directories. No real test loader or formal worker. Original rejection stays rejected.

### Task 1: Minimal diagnostic wrapper and meaningful checks

- [ ] Add `test_frozen_ablation_fim_diagnosis.py`, testing that a within-tolerance perturbation that flips selection still fails, nonfinite/missing/mismatched shape/dtype reject, hash/provenance mismatches reject, and immutable fresh output/no-test protections are retained. Run before implementation and observe failure.
- [ ] Add `frozen_ablation_fim_diagnosis.py`: capture via external wrapper around existing `smoke.value_hash`, using the tracked mean/quantile caller context; save tensor and actual output without changing computation. Run existing admission base. Verify new candidate matches old numerical evidence, exact full tensor hashes and expected8 quantile records. Replay mean on same raw inputs using reference GPU and recorded gates; reconstruct old mean/threshold hashes. Compare per-row values and exact selections/cumulative masks; write evidence and report with explicit two-task-only scope, never `admitted`.
- [ ] Run new tests plus existing GPU admission tests on Linux. Review wrapper identity, no-test path, hash binding, tolerance/selection logic and immutable old evidence. Commit and publish code/spec before numerical execution.

### Task 2: Execute and preserve bounded diagnosis

- [ ] Read only current scheduler/ledger/status; verify live identities. Seal spec/code SHA on NAS and GitHub before run.
- [ ] Capture3090 once under its original resource gate; preserve failures and do not automatically retry.
- [ ] Replay captured raw tensors on3080 GPU1 under gate; compare only after provenance reconstruction succeeds.
- [ ] Preserve raw evidence, report SHA and independent review. Report actual conclusion and limitations; prepare next protocol only if diagnosis passes. Do not activate hardware or handoff this turn.
