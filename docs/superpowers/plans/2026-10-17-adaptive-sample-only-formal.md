# Adaptive sample-only formal ablation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task. Steps use checkbox syntax for tracking.

**Goal:** Produce an independently audited nine-cell, four-party sample-only ablation that differs from the frozen adaptive9 producer only in the approved feature-related mechanism bundle.

**Architecture:** Port the reviewed sample-only switch into an isolated copy of producer `9973018`, while leaving the original formal registry as the source of dataset and task options. Make an isolated auditor from `c044261` that pins the new producer and results root. Use the existing one-job plan/success and append-only ledger pattern.

**Tech Stack:** Python 3.11, PyTorch 2.5.1+cu124, unittest, Git worktrees, NAS NFS4, RTX 3080/5090.

## Global constraints

- Never edit the frozen Replay11 producer `467611a6`, Replay11 auditor `8029edeb`, adaptive producer `9973018`, adaptive auditor `c044261`, or their result roots.
- Formal cells are CIFAR-100, ISOLET, UPMC Food101 × seeds 42, 43, 44. All have four parties; UPMC uses the frozen image/image/text/text cache SHA-256 `dbdd3a2979203ba0ee804de7d3b128426f869b0a127e4b87ebbe023e5750023b`.
- CIFAR and ISOLET use 50 epochs per task; UPMC uses 20. Keep every other effective base option from `three_dataset_formal_registry.method_contract_for(FormalSpec(dataset, 'adaptive', seed))`, including `distill_weight=0.25` and the final 500-step head fit.
- New result root: `/brf/users/yangxx/VFCL-results/adaptive-sample-only-fourparty-v1-20261017`. Preflights use other named roots and never count toward the nine cells.
- The official test remains deferred until the frozen post-training state; no post-test tuning. The result is a follow-up bundled ablation, not a blind primary comparison or a same-resource claim.

---

### Task 1: Port and test the sample-only method behavior

**Files:**
- Modify: `config.py` and `cl_methods/proto_evolve.py`
- Create: `test_sample_replay_only_formal.py`

**Interfaces:** `validate_sample_replay_only(args)` rejects any sample-only configuration with prototype/SDC/feature-KD enabled; `ProtoEvolveCL` keeps 20 raw examples per seen class and empty `global_protos` and `prev_protos`; its final head uses re-encoded raw examples.

- [ ] Write tests asserting the validator requires `cl_method='proto_evolve'`, adaptive dual branch, final schedule, 20 raw examples per class, `sample_replay_only=1`, `proto_replay_gradient_enabled=0`, `proto_sdc=False`, and `feat_distill_weight=0` while allowing formal `distill_weight=0.25`.
- [ ] Run `python -m unittest test_sample_replay_only_formal -v` and confirm the tests fail before implementation.
- [ ] Port only the sample-only guards from reviewed development commit `99e1da7`: choose retained classes from `head_raw_replay`, pass an empty prototype map to the final head, never persist `global_protos` or `prev_protos`, and reject a loaded sample-only checkpoint with either cache populated. Add parser flags for the explicit mode and zero prototype gradient.
- [ ] Run the focused test and existing `test_proto_evolve_adaptive_head`, `test_proto_evolve_balanced_replay`, and `test_adaptive_protocol_config` suites; require zero failures. Commit only these method/config/test files.

### Task 2: Freeze the separate nine-cell producer command matrix

**Files:**
- Modify: `adaptive9_matrix.py`
- Test: `test_adaptive9_matrix.py` and `test_sample_replay_only_formal.py`

**Interfaces:** `command_for(dataset, seed, ...)` uses the original formal adaptive base options, then sets `sample_replay_only=1`, `proto_replay_gradient_enabled=0`, `proto_sdc=False`, and `feat_distill_weight=0`. `run_one` retains the atomic job-directory guard and records the new producer commit under the new results root.

- [ ] Add tests comparing every generated option with the frozen adaptive command: only the declared sample-only switches, output path, and source commit may differ. Assert all nine commands retain four parties, correct data hash/cache, epochs, seeds, validation, and deferred evaluation.
- [ ] Run `python -m unittest test_adaptive9_matrix test_sample_replay_only_formal -v` and observe the new checks fail before editing the matrix.
- [ ] Apply the four mode options in the command builder and require them in `validate_completed_run`, including empty final prototype caches and the 20-per-class raw buffer. Do not change the original registry's adaptive contract.
- [ ] Run focused tests and `git diff --check`; commit. Run a clean-checkout data `check` on brf3080 against `/brf/users/yangxx/VF-CL/data`.

### Task 3: Preflight all datasets and pin an independent auditor

**Files:**
- Create: isolated audit worktree from auditor `c044261`
- Modify there: `adaptive9_evidence.py`, `adaptive9_attest.py`, `replay11_numeric.py` only where the sample-only state/protocol needs it
- Test there: focused audit identity, restore, data-flow, and numeric tests
- Add a separate protocol anchor on GitHub before formal launch

**Interfaces:** auditor pins the new producer commit and sample-only result root, rejects any prototype state, verifies raw counts at task boundaries and final state, and preserves 9 latest basic plus 3 seed-42 numeric final-gate requirements.

- [ ] Deploy a clean producer checkout and execute full-task one-epoch preflights for ISOLET (13 tasks), UPMC (10), and CIFAR (10 with real BiC). Use new preflight roots. Confirm command, data, deferred access, checkpoint reload, and basic/numeric evidence on each.
- [ ] Write failing auditor tests for wrong producer commit/root, two-party UPMC cache, nonempty prototype cache, wrong raw count, sample-only flag off, and a reported metric that differs from CPU replay.
- [ ] Make the minimal auditor changes, run focused tests on Windows and brf3080 Linux, and independently review the diff. Commit and push producer and audit branches without changing frozen branches.
- [ ] Publish an independent prelaunch protocol anchor containing both exact commits, data hashes, nine commands, runtime identities, 9/3 gate, seed-42 prerequisite, and new NAS root. Copy the anchor and its hash reference to the new audit directory.

### Task 4: Run nine formal jobs, audit, and report

**Files:** New NAS root only; existing frozen roots remain read-only.

- [ ] Fresh-check Git commits, clean worktrees, true NFS inode behavior, data hashes, GPU availability, and absence of each target job directory before launch.
- [ ] Run the three seed-42 jobs and append basic plus independent CPU numeric verdicts. Require all three latest numeric passes before starting seeds 43/44.
- [ ] Run the remaining six jobs with disjoint GPU assignments. Audit each basic verdict, preserve failures, and stop dependent work on any failure.
- [ ] Run the new final gate only after 9 latest basic and 3 latest numeric passes. Independently verify its ledger SHA-256, chain head, source commits, and all nine result trees; anchor the final hashes to GitHub.
- [ ] Compute paired three-seed AA/BWT/Task-IL differences from this new version and the frozen adaptive9 records. Check storage and extra computation evidence separately; label the comparison as a follow-up ablation and state any resource limits explicitly.
