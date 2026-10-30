# Frozen Method Formal Module Ablation Implementation Plan

> **For agentic workers:** Execute this plan inline with checkpoints. Steps use checkbox syntax for tracking.

**Goal:** Run and audit 18 paired formal ablations of the already frozen sample-only method without changing its nine original results.

**Architecture:** Keep no-PartyKD training at producer commit `e2a559dc4945729848225cf21fdc95451762c231`; derive a separate minimal FIM mask application patch from it. A manifest generator checks all nine source configs and pins two arms × three datasets × three seeds. A fail-stop queue runs one new directory per attempt with physical GPU admission. An independent auditor checks all evidence before reading accuracy and applies the prelocked interpretation rule.

**Tech Stack:** Python, PyTorch, existing VF-CL runner, JSONL/SHA256, two RTX 3080 GPUs.

## Global Constraints

- Source code commit and formal result root are exactly those in `docs/superpowers/specs/2026-10-31-frozen-method-formal-module-ablation.md`.
- The nine original result directories and all old development results are read-only. New root is `/brf/users/yangxx/VFCL-results/frozen-method-module-ablation-v1-20261031`.
- Each arm uses seeds 42/43/44 on CIFAR-100, ISOLET and UPMC. No training budget, data split, class order, score function or calibration change.
- No automatic retry. Only a documented infrastructure failure before test access can receive one manually approved identical retry.
- The full 18-job audit precedes accuracy summarization. Main method remains frozen regardless of the ablation outcome.

---

### Task 1: FIM application switch

**Files:** `config.py`, `cl_methods/proto_evolve.py`, `test_frozen_fim_application.py`.

- [ ] Write focused failing tests for default-on equality, off-mode zero applied freezes after identical FIM/mask computation, and invalid switch values.
- [ ] Add a default-on integer argument `fim_apply_mask=1`; branch only where prior FIM masks set bottom `requires_grad=False`. Print actual frozen parameter count/fraction for off mode while preserving legacy default logging and method state.
- [ ] Run the new tests and existing ProtoEvolve/FIM tests, compile, check the complete diff against e2a559dc…, commit.
- [ ] On server, run matched two-task no-test smoke for all three datasets at one fixed seed under source and patched-default-on commits. Compare task 0/1 model, memory, FIM masks, losses, gradients and RNG bytewise; verify off mode computes the same task-0 FIM and freezes zero applied parameters. Record hashes and paths before 18 jobs.

### Task 2: Exact manifest and config differences

**Files:** `frozen_ablation_manifest.py`, `test_frozen_ablation_manifest.py`.

- [ ] Write tests for the exact 18-key matrix; no-PartyKD's three allowed semantic changes; FIM-off's one added switch; path-only differences; source config and data SHA checks; and duplicate/extra job rejection.
- [ ] Build the two arm configs from each original `full/<dataset>/adaptive/<seed>/run/config.json`; retain every source value except the enumerated arm changes and new output path/name. Bind each row to the producer commit (no-PartyKD) or audited patch commit (FIM-off), source config SHA, data SHA, result path and physical GPU UUID.
- [ ] Generate the new manifest once after preflight, hash exact bytes, independently review, and anchor on NAS plus GitHub before training.

### Task 3: Fail-stop execution

**Files:** `frozen_ablation_worker.py`, `frozen_ablation_queue.py`, `test_frozen_ablation_queue.py`.

- [ ] Test direct/duplicate invocation rejection, wrong code/config/GPU, absent manifest anchor, existing output root, failure classification and official-test access guard.
- [ ] Reuse the original formal runner and readout, not a development holdout worker. Use two physical GPU lanes with the locked 120-second memory checks, exclusive root lock, one-attempt per-job directories, durable ledger and log hashes. Stop on any failure.
- [ ] Verify a tiny synthetic no-test queue fixture, then deploy both code checkouts and the audited manifest. Start the real queue once and monitor quietly.

### Task 4: Independent audit and paper table

**Files:** `frozen_ablation_audit.py`, `test_frozen_ablation_audit.py`.

- [ ] Test that missing/duplicate jobs, nonzero return, wrong code/config/data/result hashes, test access count other than one, and wrong seed pairing block summary; test the two-of-three strict PartyKD rule.
- [ ] After all 18 jobs finish, independently verify the ledger, config diffs, checkpoints, formal result seals and original nine source anchors before reading accuracies.
- [ ] Produce per-seed Class-IL, paired differences, three-seed mean and sample SD for both arms. Apply the fixed wording rule; do not change the main method. Save output in the new root and copy a reviewable report locally.
