# GPU1 Quiescent Rebalance Implementation Plan

> Use subagent-driven-development with isolated implementation and independent review.

**Goal:** Safely use idle same-host GPU1 for unstarted formal ablations without changing frozen configs or interrupting training.

**Architecture:** Observe the current controller until a fully settled no-child idle point, fence and retire that controller, then use one new controller with explicit physical GPU aliases and an append-only execution epoch. New workers preserve the original publication path and all original audit criteria, with a separately verified physical execution mapping.

**Tech Stack:** Python stdlib, existing frozen modules, unittest; Linux /proc and signals. No dependencies.

**Constraints:** The exact spec is `../specs/2026-11-02-gpu1-quiescent-rebalance.md`. Do not signal live PIDs or launch formal workers during implementation. Do not edit live f473 code, frozen training roots, original configs, manifest, QUEUE_LOCK or existing ledger bytes. Never read real accuracies before all18 plus audit. All integration tests use synthetic files/processes in independent temporary directories.

### Task A: Quiescent handoff primitives

Create `frozen_ablation_quiescent.py` and `test_frozen_ablation_quiescent.py`.

- [x] TDD complete ledger parsing, exact identity matching, absent/unknown/partial states rejection and stable successful prefix snapshot.
- [x] Define `identity(pid)->dict` (pid,ppid,pgrp,session,state,start_ticks,boot_id,cmdline), `same_identity(expected,actual)->bool`, `ledger_state(root,rows)->dict` (raw_sha256,raw_bytes_length,success,unfinished,pending,failed; metadata only); pending requires no job root/plan. Verify successful finish JOB_SUCCESS/log raw hashes without parsing accuracies.
- [x] Define `try_quiescent_fence(root,rows,old_identity)->dict|None`: None means not idle, no signals; stable idle -> SIGSTOP all old threads, two snapshots, retire only bound old parent after no-child proof; return immutable transition receipt and ledger prefix. Before permanent retirement/epoch, races restore old parent only when no new writer/worker was created, then return None. Never adopt a live child.
- [x] Define deterministic `assign_pending(keys)->dict[str,int]` (CIFAR43->1, ISOLET->0; ISOLET-only alternating0/1). Test production-independent fake old scheduler/child normal completion, active no-signal, PID mismatch, half ledger, race, no duplicates and retirement.

### Task B: Continuation controller/worker and explicit audit extension

Create `frozen_ablation_gpu1_continue.py` and tests. Reuse Task A exact interfaces above.

- [x] TDD pending allowlist, READY/GO identity, original config bytes retained, visible-two-GPU alias mapping, every abort path, worker parent-death watchdog, global failure before next launch, full18 final accounting.
- [x] Import clean f473 legacy module from explicit pinned root, reuse load_jobs and all original validations. Keep training implementation/base/patch untouched. Armed state records initial candidate cap, previous identity, expected original source code, two fixed UUIDs and deterministic allocation.
- [x] Wait for Task A idle handoff, then one epoch exclusively append original ledger. Workers use explicit new authorization; never patch old getppid or lock. Launch env sets required determinism vars before interpreter and CVD exact UUID order; preserve source config/device bytes. Async READY/GO, exact original returncode/FIM-log/seal classification, parent-death watchdog, controller finally peer-stop, no retries. Metadata start/runtime/success binds epoch and actual execution GPU.
- [x] Add explicit optional extension mode to `frozen_ablation_audit.py`, with independently supplied epoch SHA and exact execution identity validation; default audit path unchanged. Validate complete original success prefix, bounded pending mapping, original configs/runtime and all original source/publication gates; new complete record advertises actual new orchestration commit. Preserve PartyKD statistics and labels.
- [x] Linux synthetic integration and original producer publication fixture checks. No production bypass flags or fixture training roots allowed in deployed CLI.

### Task C: Verify, seal, deploy

- [x] Independently review source and meaningful failure tests; run Linux checks. Check old default audit regression and frozen code/config diff unchanged.
- [x] Read-only refreshed queue/resources. Publish clean code and protocol on GitHub/NAS. Candidate cap may shrink only by keys already started under old scheduler; never expand beyond the seven original candidates or alter a started task.
- [x] Arm the new watcher under the user's same-host GPU1 request, with durable process identity and state log. It activates only at the tested quiescent point, otherwise old queue proceeds. Do not falsely report mapping active while merely armed. Observe initial continuation starts or report the precise still-pending transition.
- [x] Save handoff evidence and monitor PID/children/ledger/JOB_SUCCESS hashes/GPU identities/FAILURE/COMPLETE only. Report progress, effective assignments, checks, limits. Do not consume accuracies until complete/audited.

Execution status: fixed57af7ce passed79 Linux tests without skips, actual18-job readonly validation, GPU alias probe, NAS atomic-link probe and independent review. Watcher3268813 is armed, not yet an active mapping as of the latest snapshot; legacy3216187 still runs. Deployment and pending transition are recorded in reports/2026-11-02-gpu1-rebalance-armed.md. No job retry or accuracy reading.

Transition observed: epoch9d17fe3b327aba8806611e4eca23515d509685b0b1bd10dd2faf05b3ec09da7b created after11 legacy successes. Both GPU0 isolet/fim_off/42 andGPU1 cifar100/no_partykd/43 started undercontroller3268813. Independent remote CPU transition metadata audit passed. See active report; earlier armed-only note is historical.
