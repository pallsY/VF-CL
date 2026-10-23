# Party-drift development experiment implementation plan

> **For agentic workers:** Execute inline with test-driven-development and review after each task. Do not delegate or start accuracy jobs before the final manifest gate.

**Goal:** Produce and execute the immutable 183-job training-holdout comparison defined in `docs/superpowers/specs/2026-10-24-party-drift-development-design.md`.

**Architecture:** Extend the existing replay path with named arms while preserving the disabled path. One training-only driver derives configs from frozen sources, scores the locked holdout, and blocks official tests. A manifest launcher checks config and code hashes; a queue records attempts without replacing results.

**Tech stack:** Python, PyTorch, NumPy, existing `runner.py` and `run_resource_lite_pilot.py`, Git/SHA-256, brf3080 NAS.

## Global constraints

- Freeze old 42/43/44 code/results/audits; all new writes go to a new NAS root.
- Seeds 70/71/72, holdout seed 20261023, pilot seed 74; 20 original samples/class, batch 64, alpha grid 0.25/0.5/1.0.
- No official test loader in development; inspect no accuracy before all 183 rows complete.
- The launcher accepts only rows in the committed manifest and checks exact config SHA-256.
- Preserve every failed attempt and classify infrastructure vs training anomaly before retry.

---

### Task 1: Common replay-control semantics

**Files:** Modify `config.py`, `party_drift_local.py`, `cl_methods/proto_evolve.py`; add `test_party_drift_controls.py`.

**Interfaces:** `party_replay_arm in {off, all_party, uniform, static, drift, cmcdr, cmcdr_budget}`; `replay_alpha: float`; `static_weights(class_party_weights, old_classes, gamma=0.2) -> list[float]`.

- [ ] Write failing tests that the off arm matches frozen loss/gradients; all-party chooses four; uniform has equal weights; static reads only the stored task-boundary table; drift retains current behavior. Run `python -m unittest test_party_drift_controls -v` and observe the expected missing-option/behavior failures.
- [ ] Add the two config fields and the smallest arm dispatch. Keep existing disabled and default candidate behavior unchanged. Run the same test module and the existing `test_party_drift_replay.py`, `test_party_drift_local.py`.
- [ ] Re-run the independent ISOLET/UPMC/CIFAR disabled identity checks on fixed input; inspect exact tensor hashes before committing.

### Task 2: CMCDR profile loss and intervention cap

**Files:** Modify `party_drift_replay.py` and `cl_methods/proto_evolve.py`; add `test_party_drift_cmcdr.py`.

**Interfaces:** `cmcdr_profile_loss(current_embeddings, old_embeddings, top, old_top, labels, old_classes, aggregation) -> Tensor`; `cmcdr_positions(seed, task, replay_steps, batch_size, old_classes, probe_events) -> set[tuple[int,int]]`.

- [ ] Write failing tests for all 16 subset responses, the 15 Möbius coordinates, old-teacher stop-gradient, concat/sum masking, fixed cap formula, deterministic no-replacement positions, and no global RNG consumption. Run `python -m unittest test_party_drift_cmcdr -v` and observe intended failures.
- [ ] Implement the exact paper-based profile loss (Smooth-L1 raw plus normalized, beta 1, lambda 1) and the precomputed budget subset. Run new and existing replay tests. Compare a two-party hand-computed profile to the implementation.
- [ ] Commit only after profiling one tiny batch to confirm no avoidable repeated bottom encoding and after ensuring no persistent feature/prototype checkpoint state.

### Task 3: Training-only readout for all arms and selected baselines

**Files:** Add `run_party_drift_development.py`, `test_party_drift_development.py`.

**Interfaces:** `run_listed_job(config_path: Path, manifest_path: Path) -> None`; `score_development_checkpoint(args, dataset, checkpoint) -> dict`.

- [ ] Write failing tests for rejection of an unlisted config, test-loader access, mismatched holdout hash, wrong method/seed, and incorrect iCaRL linear-head scoring. Run the new test module and observe failures.
- [ ] Reuse `PilotDataset` and `validate_training_only_artifacts`; use iCaRL's own NME scorer and the method-aware baseline config. Report final holdout CIL and task breakdown in a result file **read only after queue completion**.
- [ ] Run synthetic two-task development-driver tests, then one seed-74 correctness pilot per new replay arm. Audit static source table, 16 CMCDR subsets, masks, counts, raw memory and test blocking. Preserve failed pilots.

### Task 4: Immutable manifest and whitelist launcher

**Files:** Add `party_drift_manifest.py`, `test_party_drift_manifest.py`; create generated `protocol_artifacts/development_manifest.jsonl` and `protocol_artifacts/development_manifest.sha256`; create configs under `protocol_artifacts/development_configs/`.

**Interfaces:** `build_manifest(source_roots: dict, commit: str, output: Path) -> list[dict]`; `load_approved_job(job_key: str, manifest: Path, digest: str) -> dict`.

- [ ] Write failing tests for duplicate keys, unexpected 184th job, changed config byte, changed manifest byte, wrong source/holdout hash, and path traversal. Run `python -m unittest test_party_drift_manifest -v` and observe failures.
- [ ] Generate exactly 162 replay, 9 no-replay and 12 strongest-baseline configs using frozen source configs, then SHA-256 every config and canonical manifest. Enforce seeds 70/71/72, the exact three alpha values, source-data identity and 30/class disjoint holdout in the loader.
- [ ] Independently count 183 unique `(dataset,method,alpha,seed)` keys, verify all config SHA-256 values and commit/code identity, commit/push manifest and digest, then compare GitHub remote HEAD and NAS anchor. Only then enable launches.

### Task 5: Queue, retry record and embargoed adjudication

**Files:** Add `party_drift_queue.py`, `party_drift_adjudicate.py`, `test_party_drift_queue.py`.

**Interfaces:** `run_queue(manifest: Path, root: Path) -> None`; `adjudicate(manifest: Path, root: Path) -> dict`.

- [ ] Write failing tests for duplicate launch rejection, append-only attempt history, retry of exact config after infrastructure interruption, rejection of changed seed/alpha, permanent recording of NaN, and refusal to adjudicate an incomplete manifest. Run `python -m unittest test_party_drift_queue -v` and observe failures.
- [ ] Implement one lock and one atomic plan file per job; schedule UPMC, ISOLET, CIFAR in that order and respect GPU occupancy. The health monitor may inspect exit status, finite-loss checks, data-flow audits and checkpoint existence, but must not parse or summarize holdout accuracy before the manifest closes.
- [ ] Run a synthetic queue on a temporary root, inspect all failure records, then start the approved NAS queue once. After all rows are accounted for, independently verify result and audit hashes, calculate alpha selections from unrounded three-seed means, fixed-alpha selector differences, and the four predeclared outcome cases. Publish all cells, including failures, without changing the manifest.

## Review gates

Before any GPU development job: tests pass, five new arms pass correctness pilots, frozen disabled identity still passes, all 183 configs and manifest hashes match the pushed commit, qualified GPUs are free. If any check fails, stop before accuracy training and retain evidence.
