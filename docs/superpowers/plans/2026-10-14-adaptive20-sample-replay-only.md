# Adaptive20 Raw-Sample-Only Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separately identified Adaptive20 variant whose only persistent replay data are 20 raw examples per class, then measure it on the existing training-only development holdouts.

**Architecture:** A default-off `sample_replay_only` protocol flag leaves all completed runs unchanged. In the new mode, task training has no prototype replay or feature distillation; the raw cache still feeds the existing final adaptive head. The pilot labels this variant separately and stops before formal test access.

**Tech Stack:** Existing Python/PyTorch VF-CL code, `unittest`, three existing datasets and holdout manifests, Windows linked worktree, existing SSH GPU hosts.

## Global Constraints

- Preserve existing Adaptive20 and all 24 completed P/R results. New results use new roots and source provenance.
- New mode: `cl_method=proto_evolve`, final `adaptive_dual_branch` head, raw capacity 20/class, `proto_replay_gradient_enabled=0`, `proto_sdc=false`, `feat_distill_weight=0`; no persistent prototype tensors or prototype/SCD training operations.
- Preserve logit/party KD, FIM, BiC, dependency tracking, task schedule, raw exemplar herding and final gate. The method is raw-sample-only **in replay memory**, not ER minibatch replay or an ablation of every regularizer.
- Keep seed54/55, original data identities and disjoint development holdout manifests. No formal test loader, publication, or `AA_final` claim.
- Assert raw cache 20/class and prototype bytes 0 in each frozen result. Archive all files to local D and check whole archive plus every file before remote cleanup. CIFAR output uses persistent storage, not the 3080 `/dev/shm` that was twice cleared.

---

### Task 1: Protocol identity and strict configuration

**Files:** Modify `config.py`, `runner.py`; create `test_sample_replay_only.py`.

**Interfaces:** `validate_sample_replay_only(args) -> None` checks new-mode constraints; `_checkpoint_protocol(args)` adds `sample_replay_only=True` only when enabled.

- [ ] Write a unit test that passes a `SimpleNamespace` with `sample_replay_only=1` and verifies the new checkpoint identity; verify the old/default protocol has no new key.
- [ ] Write a unit test that changes each required field (`head_consolidation_enabled`, mode, capacity, `proto_replay_gradient_enabled`, `proto_sdc`, `feat_distill_weight`, `party_drift_telemetry`) and expects `ValueError` from `validate_sample_replay_only`.
- [ ] Run `python -m unittest -q test_sample_replay_only`; confirm failure because the validation function is absent.
- [ ] Add `--sample_replay_only` default 0, choices 0/1; implement `validate_sample_replay_only` with the listed exact checks, call it after argument parsing, and bind the enabled flag in `_checkpoint_protocol`.
- [ ] Rerun the focused test and existing `test_replay_factorial_loss`, `test_runner_determinism`; commit the protocol change.

### Task 2: Keep raw replay while removing persistent prototype replay

**Files:** Modify `cl_methods/proto_evolve.py`; extend `test_sample_replay_only.py`.

**Interfaces:** `ProtoEvolveCL.sample_replay_only` gates task-end state and head class selection. `_compute_protos(..., replay_capacity=20)` may compute transient embeddings for existing herding, but new-mode `global_protos`, `prev_protos`, `_proto_anchor` remain empty; `_proto_loss` sees no prototypes and uses full current CE.

- [ ] Write a test with a synthetic two-class loader and stub trainer: in new mode, `after_task` stores 20 raw examples/class but `get_state()['global_protos']` and `prev_protos` are empty, and `_sdc_update` is never called. Verify the default mode still stores prototypes.
- [ ] Write a test that calls `_proto_loss` with empty prototypes and no old teacher and checks current CE gradient coefficient 1.0; test a saved new-mode state with nonempty prototypes is rejected on load.
- [ ] Run the new tests; confirm failure on old behavior.
- [ ] Add the default-off mode, select raw cache keys in `_consolidate_head`, pass `{}` prototypes to `fit_adaptive_candidates`, discard transient prototype stats at task end, and guard the no-prototype invariant on state load. Keep teacher/KD/FIM paths unchanged.
- [ ] Rerun `test_sample_replay_only`, `test_replay_factorial_loss`, `test_adaptive_dual_branch_validation` and commit.

### Task 3: Training-only pilot and audit identity

**Files:** Modify `run_replay_factorial_pilot.py`; extend `test_sample_replay_only.py`.

**Interfaces:** A config with `sample_replay_only=1` is accepted only in the P0R1/R1 shape and the readout records `variant='sample_replay_only'` without relabeling old P0R1 results.

- [ ] Write a test that the new config is accepted and has an independent output directory, while a new flag on R0 or P1 is rejected; test old four-arm readouts keep their previous identity.
- [ ] Run the test and confirm the new variant is not recognized.
- [ ] Add the new variant identity to pilot validation/readout. Preserve `PilotDataset`, holdout exclusion, `StopBeforeTest`, and existing artifact checks. Assert prototype bytes 0 and raw count 20/class before writing the readout.
- [ ] Run `python -m unittest -q test_sample_replay_only test_replay_factorial_pilot`; do one small ISOLET training-only smoke with deterministic settings and verify freeze/readout/no-test/prototype-zero/raw20. Commit the pilot change.

### Task 4: Locked six-run development matrix and report

**Files:** Create `docs/superpowers/reports/data/adaptive20-sample-replay-only-20261014-manifest.json`, six configs, machine ledger, and Chinese report.

**Interfaces:** Every run has source/config/data/holdout SHA-256, frozen checkpoint/readout, `variant='sample_replay_only'`, 0 prototype bytes, raw20/class, zero formal test accesses and a complete verified local archive.

- [ ] Derive ISOLET/UPMC/CIFAR seed54/55 configs from the completed Adaptive20 P0R1 configs; change only the new mode, `proto_sdc`, `feat_distill_weight`, output identity/paths, and any required host data path. Record exact config/data/source hashes before starting.
- [ ] Run ISOLET seed54 first and inspect the real checkpoint and audit; if it fails, diagnose before retrying in a fresh root.
- [ ] Run the remaining five training-only jobs sequentially subject to GPU/disk guards. After each success, check exit0, freeze/readout, holdout identity/count, raw20/class, prototype bytes0, absence of formal test files/loaders, then archive to local D and verify SHA-256 of the archive and every file before remote cleanup.
- [ ] Build machine ledger and Chinese report with per-seed and mean Class-IL/Task-IL, cache bytes and training time; compare original P0R1 only as a **different mechanism reference**, noting cross-host timing limits. Verify six unique keys, hashes, no official test use, `git diff --check`; commit and push `codex/raw-sample-replay`.
