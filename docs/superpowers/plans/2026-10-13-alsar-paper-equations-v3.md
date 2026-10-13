# aL-SAR Paper Equations VFL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the task-based four-party aL-SAR adaptation with paper Equations 4–9 and its four-step Fisher refresh while preserving the approved dataset-specific epoch budgets.

**Architecture:** Keep the implementation in `cl_methods/alsar.py`. Select one aligned replay batch before the method's preliminary forward, use it again for the training loss, and retain the existing task-end reservoir contract. Compute a whole-model VFL BFC over four independent bottom prefixes and the always-trainable shared top. Collect per-example shared-top classifier gradients from the actual training forwards to update the paper's class-pair cosine EMA. Use eager frequency decay because the paper's rate changes with memory occupancy.

**Tech Stack:** Python 3, PyTorch, `unittest`, PowerShell on Windows.

## Global Constraints

- Approved design: `docs/superpowers/specs/2026-10-13-replay11-paper-methods-v3-design.md`.
- Edit only `cl_methods/alsar.py`, `test_alsar_vfl.py`, and the aL-SAR options in `replay11_matrix.py`; this plan is the only additional file.
- Do not edit the shared trainer, runner, other methods, or audit code. Stop and report any exact requirement that cannot be met within the allowed files.
- Preserve CIFAR100 50, ISOLET 50, and UPMC Food101 20 epochs per task. Preserve task-end replay insertion as a disclosed online-to-task adaptation.
- Paper equations take precedence over author code: `c_i + Σ_y S_{y,y_i}C_y`, no `|M|` division, no self correction, no negative-frequency clamp.
- Set `PYTHONUTF8=1` for test commands. Do not stage, commit, or push.

---

### Task 1: SAR equation and dynamic use frequency

**Files:** Modify `test_alsar_vfl.py`, `cl_methods/alsar.py`, and the aL-SAR dictionary in `replay11_matrix.py`.

**Interfaces:** `sar_probabilities(labels, usage, similarity, temperature)` retains its signature. `ALSARCL._select_replay()` returns the selected aligned `(x, y)` tensors or `None`; `ALSARCL._replay_loss(...)` consumes that selection. `usage[class_id][index]` stores the current discounted value directly; the obsolete timestamp exponent is removed from the new checkpoint state.

- [ ] **Step 1: Write failing numeric tests.** Assert Eq. 7–8 against a direct tensor expression with negative cross-class similarity, including an effective frequency below zero. Assert two retrieval steps with memory sizes 2 and 4 update every usage to `(1-count/(4|M|))*usage + selected_indicator`, where `count=min(B,|M|)`, and assert the command uses `alsar_temperature=0.125`. Keep the paper's frequency scale `k=4` inside this method because the shared argument parser is out of scope.
- [ ] **Step 2: Verify red.** Run `$env:PYTHONUTF8='1'; python -m unittest test_alsar_vfl -v`; expect the new equation and rate assertions to fail on the old clamp/fixed decay.
- [ ] **Step 3: Implement minimally.** Remove `.clamp_min(0)`; calculate `r=min(er_batch,|M|)/(4|M|)` after sampling; decay all stored usage and increment the selected entries; set temperature to 0.125 in the method and method-specific matrix options and `k=4` in the method. Do not divide Eq. 7 by memory size or remove the sample's own class contribution.
- [ ] **Step 4: Verify green.** Run the same focused unittest command; expect zero failures.

### Task 2: Whole-model VFL BFC and four-step Fisher refresh

**Files:** Modify `test_alsar_vfl.py` and `cl_methods/alsar.py`.

**Interfaces:** `choose_freeze_prefixes(fisher_by_party, batch_ratio, forward_cost, backward_cost_by_party, top_fisher, top_backward_cost)` returns one prefix per party. `ALSARCL` stores `top_fisher` and a measured top cost; its top gradient hook records Fisher before the shared top optimizer step. Every fourth batch starts unfrozen and refreshes all party stages and the top; other batches do not change the Fisher estimates.

- [ ] **Step 1: Write failing tests.** For tiny positive Fisher/cost arrays, compare the chosen prefixes to exhaustive evaluation of `max_m(I/C)_m * Σ BF_frozen - batch_ratio * Σ F_frozen`, including nonzero top Fisher and backward cost. On a four-batch toy training run, assert batch 1 is fully unfrozen and refreshes every Fisher block, batches 2–4 do not refresh, and batch 5 refreshes again.
- [ ] **Step 2: Verify red.** Run `$env:PYTHONUTF8='1'; python -m unittest test_alsar_vfl -v`; expect the new whole-model and cycle tests to fail.
- [ ] **Step 3: Implement minimally.** Measure shared-top forward cost alongside bottom stage costs, count top backward cost as twice the forward cost, maximize Eq. 4 over combinations of party prefixes, and apply Eq. 5 per combination. Set no freezing at `step % 4 == 0`; update top Fisher in `gradient_transform_top` and all bottom Fisher in `gradient_transform_bottom` only on those steps. Restore all frozen parameters after every batch.
- [ ] **Step 4: Verify green.** Run the focused unittest command; expect zero failures.

### Task 3: Per-example class similarity and method integration

**Files:** Modify `test_alsar_vfl.py` and `cl_methods/alsar.py`.

**Interfaces:** A method-local forward hook on the shared top records actual training features, logits, and labels. `class_similarity` is a symmetric matrix of EMA estimates for observed class pairs. `_similarity(device)` returns the matrix expanded to buffer class IDs; it does not force its diagonal to one.

- [ ] **Step 1: Write failing tests.** Construct per-example gradients with unequal magnitudes to distinguish mean cosine from cosine of class means. Assert each observed class pair, including distinct samples in the same class, follows the EMA of pairwise cosine; unobserved pairs remain unchanged. In a four-party two-task tiny integration, assert one selected replay batch feeds the same step's CE and preliminary BFC, all frozen parameters are restored, and a task-boundary checkpoint reload preserves usage, Fisher, similarity, and the next deterministic update under restored RNG.
- [ ] **Step 2: Verify red.** Run `$env:PYTHONUTF8='1'; python -m unittest test_alsar_vfl -v`; expect the per-example and integration assertions to fail.
- [ ] **Step 3: Implement minimally.** Collect actual top forward inputs and outputs without changing `VFLTrainer`; form each sample's linear-classifier gradient `vec((softmax(logits)-one_hot(label)) ⊗ feature)` and update the observed class-pair cosine EMA. Select replay before the preliminary BFC forward and consume exactly that selection in `_replay_loss`. Save/reload new method state at task boundaries. Keep the existing task-end buffer insertion and data-flow audit call order.
- [ ] **Step 4: Verify green.** Run the focused unittest and `python -m unittest test_replay11_matrix -v` with `PYTHONUTF8=1`; expect zero failures. Also run `git diff --check` and confirm only the allowed files plus this plan changed by this task.

## Self-review

- [ ] Verify Equations 4–9, refresh cadence, dynamic rate, temperature, replay alignment, checkpoint state, and the approved epoch budgets against the design.
- [ ] Confirm this plan contains no placeholder steps or unapproved shared-file edits.
- [ ] State residual departures: 50/50/20 task epochs, task-end reservoir insertion, current-plus-replay training rather than online memory-only training, and shared-top gradient subset for similarity.
