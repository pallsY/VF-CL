# iCaRL Paper Equation VFL Adaptation Implementation Plan

> **For agentic workers:** Execute the tasks inline with red-green TDD. The user has already approved `docs/superpowers/specs/2026-10-13-replay11-paper-methods-v3-design.md` and explicitly forbids Git add, commit, and push.

**Goal:** Align iCaRL's exemplar budget and representation update with the paper while retaining the four-party, per-dataset 50/50/20-epoch VFL protocol.

**Architecture:** Keep iCaRL-specific algorithm and checkpoint logic in `cl_methods/icarl.py`. Test paper equations and state transitions in `test_icarl_vfl.py`. The joint current-data/exemplar loader is gated on formal audit compatibility; do not substitute a hidden replay schedule.

**Tech Stack:** Python, PyTorch, `unittest`, existing deterministic dataset loader and formal data-flow audit.

## Global Constraints

- Modify only `cl_methods/icarl.py`, `test_icarl_vfl.py`, and, if needed, iCaRL-only options in `replay11_matrix.py`. This plan file is explicitly requested.
- Do not modify the shared trainer, runner, or other methods. Do not add, commit, or push.
- `K = 20 * args.num_classes`; after a task, `m = K // number_of_seen_classes`. Preserve herding priority order and truncate old sets to their first `m` entries.
- Use all current-task examples and all old exemplars for the representation update; preserve binary old-class teacher targets, new-class labels, and NME inference.
- If the existing formal audit cannot verify a joint loader within the file boundary, report the smallest shared audit change and stop joint-loader implementation.

---

### Task 1: Fixed total exemplar budget and prefix reduction

**Files:** Modify `test_icarl_vfl.py` and `cl_methods/icarl.py`.

**Interfaces:** `ICaRLCL.after_task(train_loader, task_id)` updates `buffer.data`; `ICaRLCL.get_state()` and `load_state(state)` preserve budget identity. `herding_indices(features, count)` returns a priority-ordered unique index list.

- [ ] Add a failing three-task test with `num_classes=6`, `K=120`, one new class per task, and enough examples. Assert counts `120`, `60`, `40` per seen class, old lists are preserved prefixes, and total occupancy never exceeds `K`. Include a case where a class has fewer examples than `m`.
- [ ] Run `python -m unittest test_icarl_vfl.ICaRLHerdingTest.test_fixed_total_budget_reduces_old_prefixes -v`; confirm failure is the current fixed-per-class behavior.
- [ ] In `__init__`, compute and validate `K`. In `after_task`, compute `m`, truncate old `buffer.data` lists before selecting exactly `min(m, available examples)` new exemplars; retain the ordered herding algorithm. Include `K` in state and reject incompatible restored checkpoints.
- [ ] Run the focused test and then `python -m unittest test_icarl_vfl -v`; confirm all pass.

### Task 2: Preserve paper binary distillation and NME

**Files:** Modify `test_icarl_vfl.py`; change `cl_methods/icarl.py` only if a test exposes a real discrepancy.

**Interfaces:** `_icarl_loss(bottoms, top_model, batch_x, batch_y, loss_ce)` returns binary cross-entropy; `_class_means()` and `_evaluate_inference(...)` implement normalized nearest exemplar means.

- [ ] Add a failing-if-regressed numeric test for old-class `sigmoid(teacher_logits)` targets, new-class one-hot targets, and `binary_cross_entropy_with_logits`; add a small NME distance test that distinguishes the exemplar mean from top-head argmax.
- [ ] Run the focused tests. If they pass against current code, retain the implementation unchanged; if they fail, make only the corresponding minimal iCaRL correction and rerun.

### Task 3: Joint training and formal audit gate

**Files:** Inspect only `cl_methods/icarl.py`, `data_utils.py`, `vfl_trainer.py`, `runner.py`, and `three_dataset_formal_audit.py` first. Modify only the allowed iCaRL files if the audit contract can still be met.

**Interfaces:** `train_task(train_loader, task_id)` must send a loader containing each current example and each old exemplar once per epoch to `VFLTrainer.train_task`; the formal verifier must independently reconstruct its index order and batch hashes.

- [ ] Confirm whether the incoming deterministic loader's `audit_sampler`, `audit_key`, and `epoch_orders` can describe mixed current and old examples and whether `_strict_data_flow` reconstructs that same mixed loader.
- [ ] If yes, add an integration test with two tasks and four party ranges that records each joint sample identity exactly once per epoch, verifies split alignment and audit hashes, then implement the smallest iCaRL-local joint loader and run red-green.
- [ ] If no, do not write a replacement loader or mask replay in `_icarl_loss`. Report the exact minimum shared trainer/audit changes needed; mark the joint-training requirement blocked by the user's file boundary.

### Task 4: Verification and handoff

**Files:** None beyond the allowed code/test files.

- [ ] Run `python -m unittest test_icarl_vfl -v`, any newly added method-specific integration test, and `git diff --check`.
- [ ] Inspect `git diff --name-only`; verify no shared trainer/runner or other-method file changed.
- [ ] Report red/green evidence, any blocked joint-training requirement, retained BCE/NME behavior, and remaining protocol differences. Do not make a completion claim for blocked requirements.

## Task-based VFL resolution of the joint-loader gate

The strict training data-flow verifier reconstructs the original current-task loader and cannot verify a literal mixed loader without a shared audit schema change. The approved task-based VFL design therefore uses a stratified stochastic estimate of the paper's union BCE risk inside the existing audited current-task loader. If the current dataset contains `N` examples and memory contains `M` old exemplars, each step draws a uniform old minibatch and optimizes `(N * mean_current_BCE + M * mean_old_BCE)/(N+M)`. With fixed per-example logits this is an unbiased estimator of the BCE objective on `D_current ∪ exemplars`; it is not a complete joint shuffle per epoch, and BatchNorm composition can make its gradient differ from a literal mixed loader. Disclose both limits. The new test `test_stratified_replay_estimates_full_union_bce` checks the scalar and gradient against the weighted union for a model without BatchNorm. The old fixed 64:current-batch weighting fails that test.
