# Factorized Adaptive Head V1 Implementation Plan

> **For agentic workers:** Execute the steps in order with test-driven development; keep existing formal methods and audits unchanged.

**Goal:** Add an explicit, stateless factorized inference readout and verify it from existing adaptive checkpoints.

**Architecture:** A pure probability composition function takes Mixed log probabilities, pre-head logits, and an aligned task map. TopModel exposes it as an explicit readout using already persisted state. A read-only checkpoint diagnostic exercises historical validation data before any test evaluation.

**Tech Stack:** Python, PyTorch, unittest.

## Global Constraints

- No changed training, gate fitting, old forward output, checkpoint schema, or audit records.
- No dataset-name branching or test-set model selection.
- Historical results remain read-only.

---

### Task 1: Pure factorization and model readout

**Files:** `factorized_head.py`, `models.py`, `test_factorized_head.py`.

- [ ] Write a test with two tasks and conflicting pre/Mixed predictions. Assert normalized probabilities, exact pre-head task-local argmax, and unchanged Mixed task masses.
- [ ] Run `python -m unittest -q test_factorized_head`; require the new behavior to fail before implementation.
- [ ] Implement `factorize_task_probabilities(log_p_mix, pre_logits, task_for_column)` with logsumexp and task-local log_softmax, rejecting missing or non-finite input.
- [ ] Expose `TopModel.factorized_log_probabilities(x)` using `_bias_logits`, existing adaptive `forward`, `_adaptive_class_order`, and `_logit_calibration_task`; do not change `forward`.
- [ ] Add a checkpoint round-trip test and rerun `python -m unittest -q test_factorized_head test_adaptive_top_model`.

### Task 2: Read-only validation and formal comparison preparation

**Files:** `factorized_validation.py`, `test_factorized_validation.py`.

- [ ] Write a test that a saved adaptive checkpoint gives identical repeated factorized output and is not modified.
- [ ] Run the failing test, implement the smallest read-only checkpoint entrypoint, and rerun it.
- [ ] On the isolated Linux worktree, run the focused and adaptive audit tests; run the entrypoint on CIFAR-100, ISOLET, and UPMC frozen training-validation checkpoints.
- [ ] Freeze a commit before accessing any new final test outputs. Record the historical checkpoint identities and evaluation protocol separately from old formal runs.
- [ ] Evaluate AA-final, BWT, and final Task-IL under the frozen three-dataset protocol only when complete compatible per-task evaluation evidence is available. Report any missing evidence rather than substitute validation values.
