# MIR paper-update implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MIR's virtual and real classifier updates use the same full-head, plain-SGD objective while retaining the approved task-based VFL data schedule.

**Architecture:** Keep MI-1 scoring in `cl_methods/mir.py`. Set MIR-only command options in `replay11_matrix.py`, preserving other methods. Use a batch-size-weighted CE in the real update so the current/replay union is one SGD batch as in Algorithm 1.

**Tech Stack:** Python 3.10, PyTorch, `unittest`.

## Global constraints

- Paper Algorithm 1 is canonical: https://proceedings.neurips.cc/paper/2019/file/15825aee15eb335cc13f9b559f166ee8-Paper.pdf.
- Four-party data, data hashes, task order, seeds, and deferred evaluation are fixed.
- CIFAR100 and ISOLET keep 50 epochs/task; UPMC keeps 20 epochs/task.
- MIR buffer update remains at task end and must be disclosed as task-based adaptation.
- Existing commit `99e81b28` and its output are historical; corrected MIR jobs restart from initial weights.

---

### Task 1: MIR-only optimizer and CE contract

**Files:** Modify `test_replay11_matrix.py`, `replay11_matrix.py`.

**Interfaces:** `command_for((dataset, 'mir', seed), ...)` emits `--task_ce_mode full --optimizer sgd --momentum 0.0 --weight_decay 0.0`. Other methods keep their existing options. The virtual step in `MIRCL._interference_scores` already uses per-party learning-rate scaling and plain SGD.

- [ ] Add a test over all three datasets that parses each `command_for` result and checks MIR's four options and existing per-dataset epoch count (50/50/20); check ER's options remain as before.
- [ ] Run `python -m unittest test_replay11_matrix -q` and confirm the new test fails because UPMC/ISOLET MIR still use `current` CE and AdamW, and CIFAR100 still uses momentum/weight decay.
- [ ] Add only `task_ce_mode='full', optimizer='sgd', momentum=0.0, weight_decay=0.0` to `METHOD_OPTIONS['mir']`.
- [ ] Run `python -m unittest test_replay11_matrix -q` and confirm zero failures.
- [ ] Commit the test and command change.

### Task 2: Real update equals union-batch CE

**Files:** Modify `test_mir_vfl.py`, `cl_methods/mir.py`.

**Interfaces:** `MIRCL._replay_loss(bottoms, top_model, batch_x, batch_y, loss_ce)` returns `(n_current*loss_ce + n_replay*replay_ce)/(n_current+n_replay)` when replay exists and `loss_ce` when the buffer is empty. `_interference_scores` retains `CE(candidate, virtual)-CE(candidate, current)` and its plain SGD virtual step.

- [ ] Add a deterministic test with two current examples, one selected replay example, and a small linear head; stub only the candidate ranking and assert the returned scalar **and all model parameter gradients** equal `cross_entropy(concat(current_logits, replay_logits), concat(current_y, replay_y))`.
- [ ] Run `python -m unittest test_mir_vfl -q` and confirm failure against the old unweighted CE sum.
- [ ] Replace the last line of `_replay_loss` with the batch-size-weighted expression, using `batch_y.numel()` and `replay_y.numel()` rather than a fixed half weight.
- [ ] Run `python -m unittest test_mir_vfl -q`, then `python -m unittest test_replay11_matrix test_mir_vfl -q` and confirm zero failures.
- [ ] Commit the implementation and regression test.

### Task 3: Pilot and method evidence

**Files:** Update the method-fidelity note in the design spec if a deviation remains; do not alter old result directories.

**Interfaces:** The new Git commit identifies MIR corrected jobs. The formal job wrapper records that commit, command, data hash, snapshot, and deferred result.

- [ ] Run one four-party two-task MIR smoke on vector data with `--epochs_per_task 1` and verify success marker and buffer/sample alignment; run CIFAR smoke only after vector passes and compute is available.
- [ ] Review the diff of Task 1–2 to ensure no other baseline's training behavior changed; record MIR-only optimizer and task-end buffer as disclosed adaptation.
- [ ] Run focused tests and `git diff --check`; only then freeze the producer commit for full MIR reruns.

## Acceptance

MIR's virtual SGD step and actual SGD optimizer both use full-head incoming CE. Replay selection maximizes the before/after per-candidate CE increase. The real step uses CE over the union of fresh and selected replay examples. All accepted MIR scores must come from new runs; no old MIR score or checkpoint is relabeled.
