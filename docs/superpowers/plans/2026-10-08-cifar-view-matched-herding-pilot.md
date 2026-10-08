# CIFAR-100 View-Matched Herding Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the saved online replay's image-view effect and remaining optimistic offline gap on an independent Adaptive development seed.

**Architecture:** Keep producer training code unchanged. A pure CIFAR pixel matcher recovers the source ID of each saved augmented exemplar; a guarded launcher trains seed 48 from the audited config; a read-only analyzer embeds the three paired conditions under one final encoder and writes class-level aggregate JSON.

**Tech Stack:** Python, PyTorch/torchvision, NumPy, existing VF-CL modules, unittest, Git, CUDA GPU 1.

## Global Constraints

- Producer checkout must remain clean at `7bfe6b1d724fb1206bc0053a9008126bad86332d`.
- Seed 48 is an independent development run, not a formal-table result.
- Candidate images are CIFAR-100 **training** rows after excluding both holdouts: 450 per class.
- Memory is exactly 20 online exemplars per class; comparison B uses their exact recovered IDs.
- No test-loader access, head refit, raw-image/embedding export, or new selector.
- All outputs go to new roots; two analysis runs must produce byte-identical JSON.

---

### Task 1: Recover online replay source IDs

**Files:** Create `cifar_replay_id_match.py`; create `test_cifar_replay_id_match.py`.

**Interfaces:** `CifarReplayMatcher(cifar_data: np.ndarray, candidate_ids: list[int]).recover_id(augmented: torch.Tensor) -> int` restores uint8 pixels from CIFAR normalization, matches the central patch under 9×9 crops and both flip states, then checks the full augmented image. Byte-identical duplicate originals map to the smallest eligible ID and increment `identical_duplicate_matches`; zero matches or multiple distinct originals fail.

- [ ] Write failing tests for a known crop/flip, wrong-class candidates, byte-identical duplicate originals, and distinct originals with one identical crop. A synthetic uint8 source image passed through the same pad/crop/flip/normalize arithmetic should recover its index.
- [ ] Run `python -m unittest -q test_cifar_replay_id_match` and confirm the missing implementation failure.
- [ ] Implement only the matcher and integer-pixel conversion using existing NumPy/PyTorch dependencies. Whole-image byte comparison is the final acceptance check.
- [ ] Run the focused tests; then dry-run complete recovery on formal seed 42, requiring 2,000 eligible canonical IDs and no distinct-original ambiguity. Record the count, exact-duplicate matches and an ID-list SHA-256.
- [ ] Commit the matcher and tests.

### Task 2: Guarded seed-48 Adaptive launcher

**Files:** Create `launch_cifar_view_matched_pilot.py`; create `test_cifar_view_matched_launcher.py`.

**Interfaces:** `derive_config(source: dict, root: Path) -> tuple[dict, dict]` accepts only the audited seed-42 config, sets seed 48 and new output paths, and flips only `formal_deferred_evaluation` besides naming/path changes. The main launcher verifies formal source hashes, clean producer commit, GPU/environment and no test-loader access; it replaces only `runner.evaluate_deferred_cil_trajectory` with an intentional stop so the unmodified training path saves and freezes its final checkpoint without reading test data. It writes protocol and training-only completion manifests, with no `results.json` requirement.

- [ ] Write a failing test asserting the exact override key set and preserved Adaptive mode, 20/class, holdout parameters and task stream.
- [ ] Run the focused test red; implement the minimal launcher by reusing the guard/manifest pattern in `launch_head_offset_pilot.py`.
- [ ] Run the focused test green and the launcher's `--check` path on the server. Assert the intentional stop is observed only after `ADAPTIVE_STATE_FROZEN.json` and all ten event checkpoints exist.
- [ ] Commit launcher and tests before launching; copy it outside the clean producer checkout and start a fresh seed-48 output root.

### Task 3: Read-only matched-view analyzer

**Files:** Create `analyze_cifar_view_matched_pilot.py`; create `test_analyze_cifar_view_matched_pilot.py`.

**Interfaces:** Reuse `normalized_centroid_error`, `embed_batches`, `bottom_state_sha256`, and repository `herding_indices`. `summarize_classes(rows: list[dict]) -> dict` reports means, medians, paired A−B/B−C differences, and positive fractions in aggregate and party spaces.

- [ ] Write a failing synthetic paired-row test for the summary function, including a negative paired difference.
- [ ] Run the focused test red; implement summary and analyzer. Verify config/checkpoint/holdout/data/source hashes, exact 450 eligible training rows/class, exact ID recovery, frozen bottoms, no protected loader access and no output of raw examples.
- [ ] Run focused tests, syntax check and `git diff --check`; commit analyzer and tests.
- [ ] After training completes, run the analyzer twice in new roots under the same deterministic environment; verify identical JSON hashes and independently recalculate reported class statistics.

### Task 4: Report and publish

**Files:** Create `docs/superpowers/reports/2026-10-08-cifar-view-matched-herding-pilot.md`; add JSON under `docs/superpowers/reports/data/`.

- [ ] Report A, B and C errors, A−B and B−C, party results, exact-match count, source hashes, repeat hash, and interpretation limits.
- [ ] Verify report numbers against class-level JSON; force-add ignored report/data files and run `git diff --cached --check`.
- [ ] Commit and push `codex/cifar-view-matched-herding-pilot`; confirm both local and producer checkouts remain clean.
