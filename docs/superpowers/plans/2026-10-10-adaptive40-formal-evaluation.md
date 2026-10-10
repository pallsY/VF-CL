# Adaptive40 Three-Dataset Formal Evaluation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Obtain one formal test-set `AA_final` for each ISOLET, UPMC Food-101 clean image, and CIFAR-100 seed 42/43/44 with the frozen Adaptive v2 40/class method.

**Architecture:** Reuse the previously published selected-hyperparameter Adaptive 20/class configs. Change only replay capacity and the new output paths/names. Run the existing `runner.run_experiment` deferred formal evaluation path from reviewed code commit `2d441589d96b55d8cb6b9511f4a227abf52bb8ec`, the commit used for the training-only v2 smoke. Bind each run to its locked config SHA-256, source commit and dataset SHA-256. Preserve all earlier results.

**Tech Stack:** Existing VF-CL Python/PyTorch runner and strict formal publication audit, Git, SHA-256, RTX 3080 and RTX 4090 hosts.

## Global Constraints

- Fixed tuple: `proto_lambda_a=0.05`, `distill_weight=0.10`, `feat_distill_weight=0.02`.
- Adaptive replay capacity: exactly 40 raw examples/class; method version 2.
- The only differences from each same-seed selected 20/class config are `head_consolidation_samples_per_class`, `results_dir`, `output_dir`, and `exp_name`.
- Preserve each seed, split, class/task order, optimizer, epochs, validation and deferred test evaluation.
- Do not inspect a test result to choose a different capacity or hyperparameter. Do not overwrite or rerun a published result.
- Locked configs and SHA-256 values are in `docs/superpowers/reports/data/adaptive40-formal-locked-manifest-20261010.json` and its adjacent config directory.

---

### Task 1: Lock source, data and configs

- [ ] Verify the nine configs differ from their published selected 20/class sources only by the four allowed fields, and each has the selected tuple, capacity 40 and deferred formal evaluation enabled.
- [ ] Verify ISOLET/CIFAR source checkout on 3080 and UPMC source checkout on 4090 are clean at commit `2d441589d96b55d8cb6b9511f4a227abf52bb8ec`.
- [ ] Rehash each dataset file against the locked manifest; verify available disk and GPU memory before launching.
- [ ] Commit and push this plan, the locked manifest, and all nine complete configs before any formal test starts.

### Task 2: Run once per seed

- [ ] Run ISOLET seeds 42, 43, 44 on 3080 GPU 1 in sequence, using the exact locked configs.
- [ ] Run CIFAR-100 seeds 42, 43, 44 on 3080 GPU 0 in sequence, using the exact locked configs.
- [ ] Run UPMC clean image seeds 42, 43, 44 on 4090 GPU 0 in sequence after the existing non-VF-CL GPU training frees enough resources. Do not interrupt that training.
- [ ] If a run fails, diagnose its log and checkpoint state before recovery. Do not rerun a run with published results.

### Task 3: Verify and report

- [ ] For each run, require `exit.code=0`, `results.json`, `FORMAL_EVALUATION_PUBLISHED.json`, final checkpoint and a v2 `ADAPTIVE_STATE_FROZEN.json`; verify bound hashes and no test access before state freeze.
- [ ] Verify the actual source commit, selected tuple, capacity/version and dataset identity for each run.
- [ ] Report per-seed `AA_final`, mean ± sample standard deviation, and paired differences against the same-seed selected 20/class runs on the same host. Include BWT and resource/memory costs so the larger replay budget is explicit.
- [ ] Save a machine-readable ledger and human-readable report on `codex/isolet-hparam-sweep`, then push GitHub.

Memory-matched fixed Full/Bias controls are a separate ablation phase. Their existing 20/class endpoints must not be relabeled as 40/class controls.
