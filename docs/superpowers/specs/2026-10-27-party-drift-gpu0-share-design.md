# GPU 0 shared scheduling addendum

**Status:** Approved by the user on 2026-10-27, before reading any development accuracy. This amends only the physical scheduler for the remaining jobs of the existing 183-job matrix. Frozen producer `b2b9546f82765c25e18394874ad0f79e786ffb82`, manifest SHA-256 `532f551f74e8db2e3f6af088d0bbf3995d9426bec822b357dd7726d4d0734c20`, configs, seeds, results, and readout rules remain unchanged.

## Admission

- GPU 0 may share with the existing `hzy` process. Require `nvidia-smi memory.free >= 3072 MiB` for **120 continuous seconds** and recheck immediately before each new worker launch. Record the reported free memory and outside compute PIDs in its append-only start event. Do not use `memory.total - memory.used` as a substitute.
- GPU 1 retains its existing shared gate: at least 5120 MiB free for 120 continuous seconds and a fresh prelaunch check.
- Both cards remain the same qualified RTX 3080s on `brf3080`. The frozen worker still sees the selected physical GPU as logical `cuda:0` through its full CUDA GPU UUID. A job starts only if it is an unstarted row in the approved manifest.
- GPU 0 currently has about 3396 MiB free while `hzy` uses about 6458 MiB. A VF-CL worker typically uses about 2308–2418 MiB, leaving a narrow margin; the outside process may grow. CUDA OOM is an infrastructure failure: keep its evidence, stop the queue, and allow only documented exact-config recovery. Do not describe shared wall time as equal compute cost.

## Second handoff

The active scheduler remains unchanged while the new version is prepared. Stop only its coordinator at a task boundary with `SIGSTOP`; let its current child finish naturally and verify its sealed success gate. If the child fails or GPU 0 lacks the locked headroom, send `SIGCONT` and abandon the handoff. Do not kill a training child.

A read-only preparation pass rechecks all completed gates, the current manifest and producer, old scheduler PID, both GPU UUIDs, ledger SHA-256, and 120 continuous seconds of GPU 0 headroom. Write a new exclusive GPU 0 admission proof outside the result tree. Anchor the new scheduler commit, proof SHA-256, ledger SHA-256, manifest, producer, and GPU UUIDs to an independent GitHub branch before stopping the old coordinator. The earlier GPU 1 protocol and handoff anchors remain historical evidence.

Under the same OS queue lock, the new scheduler verifies the external anchor and GPU 0 proof, reconstructs a sealed but unfinished event if needed, writes a new append-only handoff/lock record, and claims only unstarted keys. No duplicate coordinator or worker may run. If activation fails before a new job starts, preserve evidence and use the existing documented recovery path; never silently relaunch.

## Checks

Tests must cover GPU 0 sharing with outside compute PIDs, 3072 MiB boundary, continuous 120-second gate, prelaunch rejection without plan/start, GPU 1's unchanged 5120 MiB gate, proof/anchor mismatch, and second handoff from a prior `QUEUE_LOCK_RESUME` record. Run the relevant suite both locally and in the frozen server Python environment. Do not read development accuracy until all 183 jobs complete and the final adjudicator passes.
