# Replay Marginal Utility Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether pre-transition replay marginal utility improves a party-weighted logit-change score's association with later old-class validation loss on ISOLET and UPMC.

**Architecture:** A read-only offline script loads adjacent existing checkpoints, evaluates class-level replay and validation aggregates, compares prespecified weighting controls, and writes a JSON summary to a new root. Training code and old result roots are untouched.

**Tech Stack:** Python 3.9+, PyTorch, NumPy, SciPy, unittest.

## Global Constraints

- Use seed-42 v2 ISOLET and UPMC diagnostic checkpoints only; skip the final transition because the final full-classifier consolidation changes the readout.
- Pre-transition utility and party logit change use the checkpoint's existing training replay; validation labels only form the outcome.
- Compare utility, uniform, frozen-contribution, and deterministic cyclic-shuffle weights on exactly the same squared party logit changes.
- Do not persist individual samples, embeddings, or per-example logits; abort on missing, malformed, or non-finite evidence.
- Do not alter PartyKD, run CIFAR-100, or write into the source result roots.

---

### Task 1: Exact replay metric

**Files:** Create `party_utility_screen.py`; create `test_party_utility_screen.py`.

**Interfaces:** `replay_scores(old_trainer, new_trainer, replay: dict[int, Tensor], seen: list[int], frozen_weights: dict[int, list[float]], args) -> list[dict]` returns one aggregate row per class with `utility`, `uniform`, `frozen`, `shuffled` scores and utility fallback status. It consumes no validation data.

- [ ] **Step 1: Write a failing test.** Use two one-dimensional party embeddings and a two-class linear top head. Construct one replay class for which deleting party 0 raises CE while deleting party 1 does not; assert party 0 gets the larger nonnegative normalized utility weight. Assert identical old/new party logits yield all-zero scores and malformed replay/class coverage raises `ValueError`.
- [ ] **Step 2: Run `python -m unittest -q test_party_utility_screen` and confirm the test fails because the module or function is absent.**
- [ ] **Step 3: Implement exact class-party logits.** For concatenated embeddings, compute `F.linear(z_p, W[:, p*d:(p+1)*d], None)` and assert their sum plus bias matches `top(aggregate(z))` before final consolidation. Compute `u_p = mean(CE((full-party_p)[:, seen], c) - CE(full[:, seen], c))`; normalize `clamp_min(u, 0)` and fall back to `1/P` when its sum is zero. For the same replay, compute `d_p = mean((new_party_p[:, c]-old_party_p[:, c])**2)`. Return four weighted sums using utility, uniform, frozen, and `roll(utility, 1)` weights.
- [ ] **Step 4: Re-run the focused test, then commit the metric and test.**

### Task 2: Read-only checkpoint and validation screen

**Files:** Modify `party_utility_screen.py`; extend `test_party_utility_screen.py`.

**Interfaces:** `screen_run(run_dir: Path, output_dir: Path) -> dict` loads `config.json`, all required `event_i_CIL.pt` checkpoints, builds two model bundles through `build_models`, and evaluates validation loaders for the classes seen before each transition. `summarize_rows(rows) -> dict` reports dataset-specific centered Spearman correlations, boundary directional counts, fallback fraction, and score spreads.

- [ ] **Step 1: Write a failing tiny two-task fixture test.** Assert the routine reads no test loader, never writes under `run_dir`, restricts the primary CE at both times to the old seen classes, and rejects missing checkpoint/replay or non-finite values.
- [ ] **Step 2: Run the focused test and confirm the missing screen implementation fails.**
- [ ] **Step 3: Implement loading and evaluation.** Read checkpoints using the repository's safe loader, set `args.output_dir` to the new output root before building `VFLDataset`, call only `get_validation_loader`, and calculate mean old-only CE and ordinary CIL error for each class at both checkpoints. Use the same dataset object and validation manifest throughout. Exclude the final transition and write one aggregate JSON with source hashes and rows by atomic replace.
- [ ] **Step 4: Implement analysis.** For each boundary subtract its row mean from every score and primary CE delta, call `scipy.stats.spearmanr`, and compare utility against uniform/frozen with the exact design stop rule. Count positive within-boundary correlations only when at least three class rows are present.
- [ ] **Step 5: Run focused tests and Linux smoke on a minimal checkpoint pair, then commit.**

### Task 3: Existing-run screen and report

**Files:** Create `docs/superpowers/reports/2026-10-07-party-utility-screen.md`.

- [ ] **Step 1: Run the script separately on the frozen ISOLET and UPMC v2 run directories, writing to a new `party-utility-screen-20261007-v1` root.** Record the code commit and SHA-256 of source config, data, checkpoints, and output.
- [ ] **Step 2: Inspect coverage, finite values, validation manifest, per-boundary rows, controls, and the prespecified stop rule.** Do not select a new formula after seeing results.
- [ ] **Step 3: Write the report with exact results, limitations, and whether PartyKD/CIFAR remain deferred; commit and push the branch.**
