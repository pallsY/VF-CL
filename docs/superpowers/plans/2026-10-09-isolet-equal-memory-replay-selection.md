# ISOLET Equal-Memory Replay Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether a fixed 10-herding/10-diversity replay selector improves ISOLET final head accuracy under the existing 20/class memory budget.

**Architecture:** A standalone launcher imports the exact clean ISOLET producer worktree, patches only its in-memory replay selector and pilot dataset loader, stops immediately before deferred test evaluation, and scores the frozen final checkpoint on a disjoint training-only holdout. Four result roots support two paired seeds; one aggregate ledger/report makes the decision.

**Tech Stack:** Python 3.9, PyTorch, NumPy, existing VF-CL modules; PowerShell/SSH for remote execution.

## Global Constraints

- Exact source commit `a575bbf446ae501cf8e580ba62c2cc5492a25f30`, clean producer worktree, verified ISOLET NPZ/metadata hashes.
- Keep `(proto_lambda_a, distill_weight, feat_distill_weight)=(0.05,0.10,0.02)`, 20 raw replay samples/class, 13 two-class tasks, four views, AdamW, 50 epochs/task, final-only adaptive head, original gate validation 40/class.
- Pilot holdout is 40/class, seed `20261010`, excluded from pilot training and disjoint from gate validation; no test loader iteration.
- Seeds 47/48, each with current herding and fixed 10+10 candidate. Select using holdout CIL only after all four runs finish.

---

### Task 1: Validate deterministic equal-memory selection

**Files:** Create `launch_isolet_replay_selection_pilot.py`; create `test_isolet_replay_selection_pilot.py`.

**Interfaces:** `hybrid_indices(embeddings: torch.Tensor, capacity: int) -> torch.LongTensor`; `PilotDataset(VFLDataset)` excludes a deterministic nested holdout only from `train` loaders.

- [ ] Write a failing synthetic test that asserts 20 distinct selected indices, exactly the first 10 original herding indices, deterministic farthest-first continuation, and no pilot-holdout index in a train loader.
- [ ] Run `python -m unittest -q test_isolet_replay_selection_pilot` and confirm the missing implementation causes failure.
- [ ] Implement the minimum selector and dataset wrapper; use existing `herding_indices`, `calibration_split.build_manifest`, and `VFLDataset._loader`.
- [ ] Rerun the focused test and `python -m py_compile launch_isolet_replay_selection_pilot.py`.

### Task 2: Launch four training-only pilot runs

**Files:** Modify `launch_isolet_replay_selection_pilot.py`; create separate remote result roots under `/home/c3080/YangXiaoXiang/VF-CL/results/isolet-equal-memory-replay-pilot-20261009-v1`.

**Interfaces:** CLI accepts `--source-config`, `--source-record`, `--root`, `--seed` (47/48), `--variant` (`herding`/`hybrid`), and `--check`; each run produces `PILOT_COMPLETE.json` and `HOLDOUT_READOUT.json` after final checkpoint freeze.

- [ ] Add tests for rejected config overrides, gate/holdout overlap, insufficient train capacity, and attempted test access.
- [ ] Implement source/data/config preflight, temporary selector and dataset hooks, `runner.run_experiment` with a stop before deferred test access, frozen checkpoint verification, and holdout scoring through saved bottoms/top.
- [ ] Execute `--check` for both seeds and variants; inspect the reported overrides and identical split manifests.
- [ ] Run the four jobs sequentially on the idle 3080 GPU; verify exit status, 13 event checkpoints, final adaptive checkpoint, 20 raw replay/class, no test loader access, and paired pre-head identities.

### Task 3: Audit and report

**Files:** Create `docs/superpowers/reports/2026-10-09-isolet-equal-memory-replay-pilot.md` and `docs/superpowers/reports/data/isolet-equal-memory-replay-pilot-20261009.json`.

**Interfaces:** Ledger records each run's source/config/checkpoint/holdout hashes, CIL/Task-ID/Task-IL/old/new metrics, memory count and paired differences.

- [ ] Independently recompute both seed deltas, mean gain and newest-task guard from the ledger.
- [ ] Apply the locked rule: both seeds improve CIL, mean gain at least +1.00 pp, newest-task drop no worse than −1.00 pp in either seed.
- [ ] State development-only limitations and next decision; run focused tests and `git diff --check` before commit/push.
