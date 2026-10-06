# Party Drift Telemetry Implementation Plan

> **For agentic workers:** Execute in order with test-driven development. Do not modify published runs.

**Goal:** Record per-class per-party previous-teacher-to-current feature drift from existing training replay without altering training.

**Architecture:** A pure function computes mean cosine drift from aligned old/current party embeddings. ProtoEvolve invokes it once after each new task has produced its bounded replay, before head consolidation and teacher replacement. A separate immutable JSON file stores only summary values and identities.

**Tech Stack:** Python, PyTorch, unittest.

## Global Constraints

- Default off. No change to frozen formal method or legacy checkpoint schema.
- CL-only training replay; reject scheduled unlearning, forgotten raw replay, and missing retained old classes before any measurement. No validation/test access during measurement.
- No extra persistent raw samples, embeddings, or full model snapshots.

---

### Task 1: Pure drift summary

**Files:** `party_drift_telemetry.py`, `test_party_drift_telemetry.py`.

- [ ] Write a failing test using two synthetic party models and two replay classes. Assert zero drift for unchanged models, positive drift for one rotated party, exact sample counts, and no parameter mutation.
- [ ] Run `python -m unittest -q test_party_drift_telemetry`; require failure before implementation.
- [ ] Implement a finite, deterministic per-class per-party mean cosine drift calculation over the same training replay tensor.
- [ ] Run the focused test and reject missing parties/classes, malformed replay, and non-finite features.

### Task 2: Optional ProtoEvolve integration

**Files:** `config.py`, `cl_methods/proto_evolve.py`, `test_party_drift_telemetry.py`.

- [ ] Add an opt-in `--party_drift_telemetry` flag with default zero and test its default/explicit parsing.
- [ ] Write a failing two-task method fixture that proves telemetry uses only old replay and the frozen previous-task bottoms, and leaves the training model state unchanged.
- [ ] Invoke the pure function before final head consolidation and teacher replacement. Write an atomic, idempotent task-boundary JSON summary under a new result root.
- [ ] Run focused ProtoEvolve, checkpoint/resume, and privacy tests on Linux; run a two-task smoke with telemetry enabled and compare model outputs to telemetry-disabled execution.

### Task 3: Scientific pilot

- [ ] Freeze source, configuration, data manifest, and success rule before reading development metrics.
- [ ] Run separate seed-42 diagnostic roots on CIFAR-100, ISOLET, and UPMC, subject to resource preflight. Do not overwrite current formal roots.
- [ ] Compare weighted and unweighted party drift with frozen training-validation cross-task error, accounting for task age. Report negative results and stop if no stable signal is present.
