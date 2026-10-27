# GPU 0 shared scheduling implementation plan

**Goal:** Allow the current 183-job queue to use GPU 0 concurrently with `hzy` after 3072 MiB free for 120 continuous seconds, preserving all frozen training and result evidence.

**Architecture:** Change only the isolated scheduler's physical admission rule. At the current worker's successful task boundary, create a new externally anchored proof and transfer the original OS queue lock to one new coordinator.

**Constraints:** Preserve producer `b2b9546f82765c25e18394874ad0f79e786ffb82`, manifest SHA-256 `532f551f74e8db2e3f6af088d0bbf3995d9426bec822b357dd7726d4d0734c20`, all configs and existing outputs. GPU 1 keeps 5120 MiB/120 seconds. No test or development accuracy access before completion. All changes and failures remain append-only.

1. Add failing tests for GPU 0's 3072 MiB/120-second gate, outside PID allowance, fresh prelaunch check, and GPU 1's unchanged gate. Verify red, implement the minimum code, verify green.
2. Add failing tests for a GPU 0 admission proof and a second handoff from a resume lock. Verify red, implement the minimum code, verify green.
3. Run the local queue/adjudicator suite, `git diff --check`, and verify no diff from the frozen producer's method files. Request independent read-only review and fix blocking findings.
4. Commit/push the scheduler addendum, create a clean detached server checkout, run the same tests in the pinned server Python environment, and write/push a separate pre-handoff protocol anchor.
5. Observe the active child. Only after its sealed success: stop its coordinator, create a GPU 0 proof outside the result tree, externally anchor the proof and ledger, recheck headroom, then stop that coordinator and launch one new coordinator under the same OS lock. If any gate fails, resume the existing coordinator.
6. Verify the append-only handoff, unchanged prior successes, distinct pending keys, frozen worker path, physical GPU mapping, GPU 0 worker, and no failure gate. Update the existing heartbeat to monitor the new PID without repeating the handoff.
