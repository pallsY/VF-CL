# All-Party FIM Final Attempt Implementation Plan

> **For agentic workers:** Implement in this isolated branch, test each stage before moving on. Do not touch the prior 183-job result root.

**Goal:** Run one prelocked 65-job development comparison, conditional 15-job PartyKD ablation, and conditional 9-job follow-up formal comparison with fail-stop audits.

**Architecture:** Preserve production commit `b2b9546f82765c25e18394874ad0f79e786ffb82` as the base. Add an opt-in FIM mask policy, then build a separate study manifest, worker, queue, and adjudicator that reuse the existing dataset/holdout/readout/audit helpers. Freeze each stage's code and manifest before jobs. Never mutate old results or open official test before both development gates pass.

**Tech Stack:** Python 3.12, PyTorch, existing VF-CL runner, JSONL + SHA256 audit, two RTX 3080 GPUs on `brf3080`.

## Global Constraints

- Three datasets; seeds 80–84; F0/F1/F-off; all-party alpha 0.25; 20 old exemplars/class; fixed 30/class development holdout; 50 epochs/task for CIFAR-100 and ISOLET, 20 for UPMC as in source configs.
- Stage A = 45 method + 20 baselines, stage B = 15 no-PartyKD after selection, stage C = 9 official runs only after both gates.
- Same data SHA, holdout SHA, validation SHA and native readout as prior 183-job manifest; no official test access in A or B.
- One attempt per job, append-only ledger, no automatic retry; all failures preserve logs and stop the queue.
- F0 mirrors old mask semantics; F1 replaces the mask after the same FIM pass; F-off computes FIM but applies none.
- Report paired seed effects and uncertainty. Stage A requires three mean wins, maximin tie-break F0/F1/F-off. Stage B requires PartyKD gain ≥0.5 pp on at least two datasets and ≥−0.5 pp on the third. Stage C requires all three formal mean wins.

## Tasks

### Task 1: Opt-in FIM mask policy

**Files:** `config.py`, `cl_methods/proto_evolve.py`, `test_replay_fim_policy.py`.

- [ ] Add tests that verify task-0 F0/F1 same mask, F-off no mask; task-1 F1 replaces while F0 accumulates; all three compute the same FIM scores from the same input; no effect on default legacy configuration.
- [ ] Confirm the tests fail before implementation.
- [ ] Add one argument with allowed values `cumulative`, `fresh`, `off`, default `cumulative`. Keep FIM pass unchanged; change only mask update and record per-task mask evidence.
- [ ] Run new and existing FIM/replay tests, compile, diff-check, commit.

### Task 2: Immutable stage-A and conditional stage-B manifests

**Files:** `replay_fim_manifest.py`, `test_replay_fim_manifest.py`.

- [ ] Test exact allowlists (65 stage-A; selected-policy 15 stage-B), source SHA and config SHA validation, no duplicate/extra job, fixed alpha/seeds, safe output root and disjoint old results.
- [ ] Confirm test failures before implementation.
- [ ] Generate config bytes from frozen source configurations using existing `party_drift_manifest.py` source anchors. Pin code commit, data, holdout, validation, and stage identity. Stage-B configs generated only after audited stage-A decision.
- [ ] Run tests, compile, diff-check, commit.

### Task 3: Fail-stop worker and GPU queue

**Files:** `run_replay_fim_development.py`, `replay_fim_queue.py`, `test_replay_fim_queue.py`.

- [ ] Test that a worker refuses unlisted config, changed source hashes, direct invocation, duplicate attempt, test access, incomplete selection ledger or invalid FIM mask; queue refuses existing result root, duplicate process, wrong GPU identity and any non-success without retry.
- [ ] Confirm failures before implementation.
- [ ] Reuse `PilotDataset`, checkpoint readout, `verify_selection_audit`, and sealed-job hashes; add FIM mask audit. Queue writes an exclusive lock, start/finish ledger, per-job log and success hash. Require GPU free-memory headroom for 120 seconds and recheck at launch; use physical UUID binding.
- [ ] Run focused tests and one small no-test smoke in a fresh temporary root; commit. Freeze/push code, then write and hash stage-A manifest separately.

### Task 4: Independent adjudication, PartyKD gate and formal launcher

**Files:** `replay_fim_adjudicate.py`, `run_replay_fim_formal.py`, `test_replay_fim_adjudicate.py`.

- [ ] Test exact count-based mean, best baseline per dataset, maximin tie break, PartyKD threshold, all-three formal gate, missing/duplicate gate rejection and no output on incomplete audit.
- [ ] Confirm failures before implementation.
- [ ] Build read-only stage-A/B decisions and conditional stage-C allowlist; stage-C uses selected fixed config and the original formal readout, never exposes official test until A/B passed.
- [ ] Run tests and independent dry-run audit, commit/push, freeze manifest and result roots.

### Task 5: Deployment and monitor

- [ ] Check input SHA anchors, GPU UUID, storage, resource thresholds and no active old queue. Verify isolated single-process vs party-sliced forward/backward numerical equivalence on a CPU fixture; label it an algebraic simulation check, not a distributed privacy guarantee.
- [ ] Start stage-A queue once, verify PID, lock, first start event and child GPU binding. Create a quiet heartbeat monitor for failure, milestones and completion only.
- [ ] After 65 audited successes, run stage-A adjudication once. If closed, stop. If passed, freeze stage-B 15-job manifest and run once; after audited completion apply PartyKD gate. If passed, freeze stage-C 9-job manifest and run once. Report all results without further tuning.
