# Balanced party-drift queue resume implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task. Steps use checkbox syntax for tracking.

**Goal:** Resume the existing 183-job development queue on both same-model RTX 3080 cards when GPU 1 has at least 5120 MiB free, without changing frozen training code, approved config bytes, results, or selection rules.

**Architecture:** A separate scheduler commit imports the frozen queue's verifier and launches only unstarted manifest jobs with the original producer executable. A guarded handoff freezes the original coordinator after its current child succeeds, captures append-only evidence, and resumes under the same OS lock. CUDA visibility maps a physical 3080 to the config's logical `cuda:0`; every mapping is recorded.

**Tech stack:** Python 3.12 standard library, PyTorch worker from producer `b2b9546f82765c25e18394874ad0f79e786ffb82`, Linux `flock`, `nvidia-smi`, NAS JSONL/SHA-256, unittest.

## Global constraints

- Manifest SHA-256: `532f551f74e8db2e3f6af088d0bbf3995d9426bec822b357dd7726d4d0734c20`; exactly 183 immutable rows.
- Do not read holdout scores until the complete queue gate; do not access official test inference.
- No duplicate launch, no retry of a training anomaly, no deletion or rewrite of old events, plans, logs, checkpoints, or failures.
- The active queue and its child keep running while code is prepared. The user approved GPU 1 co-scheduling with outside processes on 2026-10-27; require 5120 MiB free for 120 continuous seconds and disclose outside PIDs.
- The new scheduler is externally anchored before activation. Its producer path is a clean checkout at the original `b2b9546` commit.

---

### Task 1: Pure admission and history checks

**Files:** Create `party_drift_balanced_resume.py`; create `test_party_drift_balanced_resume.py`.

**Interfaces:** `pending_jobs(manifest_rows, events) -> list[dict]`; `validate_history(manifest_rows, events, root, manifest_sha256) -> dict`.

- [ ] Write a failing test with three listed jobs: one sealed success, one open start with sealed success, and one unstarted. Assert that the first two are accounted for exactly once and only the third is pending. Include a changed config hash and a training-anomaly history that must reject handoff.
- [ ] Run `python -m unittest test_party_drift_balanced_resume -v`; confirm the missing interface causes failure.
- [ ] Implement the smallest read-only history parser using `load_approved_job`, `verify_sealed_job`, and the frozen queue's event rules. Reject any unpaired start, changed SHA, missing source, duplicate key, or unresolved failure; do not read `HOLDOUT_READOUT.json` metrics.
- [ ] Rerun the test and the existing queue/adjudication tests; commit the admission gate.

### Task 2: Physical GPU dispatch and evidence

**Files:** Modify `party_drift_balanced_resume.py`; extend `test_party_drift_balanced_resume.py`.

**Interfaces:** `gpu_snapshot(index) -> dict`; `shared_headroom_snapshot(snapshot) -> bool`; `worker_environment(base_env, physical_gpu, seed, gpu_uuid) -> dict`; `run_balanced_queue(...) -> dict`.

- [ ] Write failing tests proving `CUDA_VISIBLE_DEVICES=1` leaves config `device='cuda:0'` unchanged, GPU UUID is recorded, a claimed job cannot be claimed again, and GPU 0 continues if GPU 1 is busy. Use a synthetic executor, not real training.
- [ ] Run the new tests and verify red for missing dispatch.
- [ ] Implement two workers that claim the next unstarted key only after physical GPU checks. GPU 1 requires 5120 MiB free for 120 continuous seconds and a fresh headroom check before launch; outside PIDs are recorded, not blocked. GPU 0 uses the original two-probe idle gate. Launch only `run_party_drift_development.py` from the clean `b2b9546` checkout. Append physical GPU index/UUID, observed co-tenant PIDs and free memory, and scheduler SHA to each start record before subprocess launch; use the frozen success verifier and append its gate SHA to the finish record.
- [ ] Write `QUEUE_COMPLETE.json` only after all 183 final events are accounted for and all success gates reverify; preserve training-anomaly and infrastructure fail-stop behavior. Run synthetic completion, failure, and concurrent-lock tests. Commit the dispatcher.

### Task 3: Non-destructive handoff and activation

**Files:** Extend `party_drift_balanced_resume.py`; extend `test_party_drift_balanced_resume.py`; add an operations appendix to `docs/superpowers/specs/2026-10-27-party-drift-balanced-resume-design.md`.

**Interfaces:** `prepare_handoff(root, producer, manifest, scheduler_sha) -> dict`; CLI `--prepare` and `--resume`.

- [ ] Write failing tests for an active original PID, active worker, changed ledger, absent success gate, GPU 1 with less than 5120 MiB free, and a second handoff attempt. All must reject before any new start.
- [ ] Implement preparation as a read-only evidence snapshot. Activation requires a human-visible `SIGSTOP` of only the original coordinator, waiting for its current child to finish, frozen gate verification, a final GPU 1 headroom check, then stopping the coordinator. If the child fails or GPU 1 loses headroom, `SIGCONT` the original coordinator and leave the balanced scheduler inactive.
- [ ] Under the original OS queue lock, reconstruct any sealed-but-unfinished child event with the frozen `recover_interrupted_attempts`, write the append-only handoff record and new resume lock, and start balanced dispatch once.
- [ ] Run the full relevant unit suite on Windows and the qualified 3080 Linux host. Use a seed-74 correctness-only two-task pilot on each physical 3080, with no official test or development score access, to verify CUDA visibility and worker parent authorization.
- [ ] Request independent code review. Fix all blocking findings. Commit and push the scheduler branch; externally anchor its commit, manifest hash, admission snapshot and GPU UUIDs before any cutover.
- [ ] Recheck GPU 1 headroom and the original queue. If the original queue is near completion or GPU 1 lacks headroom, leave it running. Otherwise execute the handoff once, verify no duplicate keys, intact ledger SHA chain, original producer commit in every new worker, and two live GPU jobs.

## Completion gate

The original adjudicator must accept `QUEUE_COMPLETE.json`, all 183 final cells, and the append-only ledger. If it does not, retain both queue histories and stop; do not edit frozen artifacts or relax a verifier.
