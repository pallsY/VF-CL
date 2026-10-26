# Party-drift development queue balance addendum

**Purpose.** Reduce the wall time of the already anchored 183-job development
matrix after one RTX 3080 lane has exhausted its assigned jobs. This addendum
changes physical scheduling only. It does not change the frozen producer
`b2b9546f82765c25e18394874ad0f79e786ffb82`, manifest SHA-256
`532f551f74e8db2e3f6af088d0bbf3995d9426bec822b357dd7726d4d0734c20`,
job config bytes, training seed, data, algorithm, readout, or outcome rule.
Existing results and failed records remain untouched.

## Activation gate

The original queue remains active until a second RTX 3080 on `brf3080` has
safe shared-memory headroom. The user explicitly approved concurrent use
with another account on 2026-10-27. Physical GPU 1 must have at least
**5120 MiB free for 120 continuous seconds** before handoff and again before
each new claim; outside compute PIDs and utilization are recorded but do not
block a job. This threshold was set before opening any development accuracy:
the current VF-CL worker uses about 2.4 GiB, leaving at least 2.6 GiB of
headroom for variation. An outside process may still grow or slow down; a
CUDA OOM is an infrastructure failure, remains in the ledger, and stops the
queue for documented exact-config recovery. Both
cards must be the same RTX 3080 model on the same host, driver, Python/Torch
environment, and NAS mount. The balanced scheduler and this addendum must be
committed and externally anchored before handoff. If these conditions never
hold, the original queue runs to completion without interruption.

## Handoff

1. Capture the original `ATTEMPTS.jsonl` SHA-256, last event, original queue
   PID, manifest SHA-256, producer commit, active worker PID, and both GPU UUIDs.
   Do not open development accuracy metrics.
2. Stop only the original queue coordinator with `SIGSTOP`. Its current
   training child continues. Wait for that child to finish and for its
   `JOB_SUCCESS.json` to pass the frozen verifier. Do not kill a training
   child or edit its files. If the child does not succeed or GPU 1 loses the
   5120 MiB headroom, send `SIGCONT` to the coordinator and abandon handoff so the
   original queue can classify the failure or continue normally.
3. Stop the coordinator after the child succeeds, release its OS queue lock,
   and start the new scheduler once. The new scheduler obtains the same OS
   exclusive lock, confirms the old PID and workers have exited, replays the
   append-only event history, and reconstructs a missing `finish` event from
   the sealed job gate. It never retries a training anomaly.
4. Record an append-only `BALANCED_HANDOFF.json` with original and new code
   commits, before/after ledger hashes, reason, process IDs, GPU UUIDs, and
   exact startup command. Preserve old lock, logs, checkpoint, and plans.

If any handoff check fails, the new scheduler stops before launching a job.
The original evidence remains available for diagnosis and a documented
recovery. No automatic retry or deletion occurs.

## Balanced scheduling

The new scheduler verifies all 183 canonical rows and config SHA-256 values,
the original producer commit, and every previous successful gate before
selecting remaining keys. It uses only a fixed job-key order and completion
state; it never reads a holdout score. It claims a job only after its target
physical GPU passes its admission gate. Either physical RTX 3080 may claim the next
unstarted job. A single OS queue lock and exclusive per-job plan prevent
duplicates. The worker executable always comes from the original clean
producer checkout.

For a config with logical `cuda:0`, the scheduler sets
`CUDA_VISIBLE_DEVICES` to the selected physical card's full `GPU-...` UUID,
verified against `nvidia-smi`. The original config remains byte-identical;
inside the worker the chosen physical GPU is logical `cuda:0`. Only unstarted jobs are eligible
for this remapping. The append-only `start` event records physical GPU index,
UUID, scheduler commit, producer commit, environment mapping, observed
outside compute PIDs, and free memory. If GPU 1 lacks the headroom before
claim, GPU 0 continues; no job is reserved for an unavailable GPU.

The scheduler uses the frozen worker's existing source, test-access, selection
ledger, checkpoint, data-flow, and readout gates. It writes the same
`QUEUE_COMPLETE.json` structure only after all 183 jobs have a final success
or documented training anomaly and every successful gate re-verifies. The
original adjudicator remains unchanged and sees only the completed ledger.

## Limits and reporting

This is a scheduling amendment, not a new method variant. Physical GPU UUID
and per-job wall time will be disclosed. Same-model GPUs can still produce
minor numeric differences; the existing three-seed variance and exact
checkpoint/readout audit remain the evidence boundary. If a training failure,
source mismatch, duplicate claim, or CUDA OOM appears, fail
stop and preserve all files. The run is not claimed as bitwise identical
across physical cards without a separate identity check.

## Operational handoff checklist

1. While the original queue runs, `--check` verifies all sealed gates and
   reports pending/active keys without using scores. Confirm GPU 1 has at
   least 5120 MiB free before attempting a handoff; disclose outside PIDs.
2. Send `SIGSTOP` only to the original queue coordinator. Its current worker
   keeps running. Wait until the worker exits and the frozen success gate
   verifies. If it fails, send `SIGCONT` to the coordinator and stop.
3. Run `--prepare-output` into a new, independent NAS audit directory. This
   requires the coordinator to be stopped and no worker to remain, verifies
   120 continuous seconds of GPU 1 memory headroom, and writes an exclusive
   `GPU1_ADMISSION_PROOF.json`. If it fails, send `SIGCONT` and stop.
4. External-anchor the new scheduler commit, the proof file SHA-256, original
   producer commit, original manifest SHA-256, source ledger SHA-256, and both
   GPU UUIDs. Recheck GPU 1. If free memory falls below 5120 MiB, send
   `SIGCONT` and stop. Outside compute processes are allowed and recorded.
   The proof expires 30 minutes after observation.
5. Only then stop the original coordinator (never its worker) and launch the
   new scheduler once with the exact anchor path and SHA-256. It obtains the
   same OS lock, reconstructs a sealed-but-unfinished event, writes a new
   append-only queue lock, and dispatches only unstarted jobs.
6. If the new scheduler fails before completion, keep every file. After
   verifying that it and all its workers have exited, the frozen original
   queue can be resumed with its documented `--recover-interruption-reason`
   path, preserving both histories. Never run both coordinators together.
