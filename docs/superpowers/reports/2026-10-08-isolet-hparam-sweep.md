# ISOLET Adaptive one-factor hyperparameter sweep

Date: 2026-10-08 (Asia/Shanghai). Status: **development tuning result**, with one independent-seed paired check. No formal ISOLET test result is claimed.

## Fixed protocol and execution

The user requested tuning only their Adaptive method, one hyperparameter at a time; all other baselines stayed untouched. The exact formal ISOLET producer source commit was `a575bbf446ae501cf8e580ba62c2cc5492a25f30`, with formal seed-42 record SHA-256 `7dc5bf930600d097134948d6ee166fd55490327d16cde1fe2496ac374a73299b` and config SHA-256 `5702b8846ca8e9d279728a725550c1b053b8580fec832a9695b56d6ea7f2af34`. The ISOLET NPZ and metadata on the 3080 host match the formal payload SHA-256 values `d34312670de93198afcd2b126c95b79bae2b4cffeb30d480f097faf046b69514` and `79396dea1751b6094a5769f2dd789ad58b3ea9582a12c8c98d3ec07d8d1eb3cd`.

The original cloud 4090 host had only about 2 GiB free, so the sweep used a clean, detached worktree of that exact source commit on the 3080 host, masking physical GPU 1 as `cuda:0`. Only path, device placement, seed, non-formal deferred-evaluation flag and the three approved hyperparameters varied. The 13 two-class tasks, four ISOLET views, 26 classes, 40/class lambda-validation manifest, 20/class head replay, final-only adaptive head, 50 epochs/task, batch size 128, optimizer and all other losses were fixed. Each run used a new result root. The [launcher](../../../launch_isolet_hparam_trial.py) stopped after final checkpoint freeze and before deferred test evaluation. All nine completion records bind 13 CIL event checkpoints, the final checkpoint/config and the unchanged validation manifest; none iterated a test loader. The vector NPZ loader still materializes the fixed test arrays at initialization, but no test labels were used for scoring or tuning.

The score is final class-incremental accuracy on the **existing 40/class lambda-validation cohort** (1,040 samples), with old classes 0–23, newest task 24–25, Task-IL and NLL recorded. This cohort was also used by the adaptive gate, so the numbers are development-selection evidence, not untouched confirmation. The prespecified rank was highest CIL, then old-class CIL, then Task-IL, then proximity to the formal parameter tuple.

## Sequential seed-45 search

`a` = `proto_lambda_a`, `d` = `distill_weight`, `f` = `feat_distill_weight`. The baseline is `(0.15, 0.25, 0.05)`. Only one coordinate changed within a stage; a completed run was reused as the middle candidate of the next stage.

| Stage | `a` | `d` | `f` | CIL | TIL | Old CIL | Newest CIL | NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.15 | 0.25 | 0.05 | 88.75% | 99.13% | 88.65% | 90.00% | 0.3647 |
| 1 | **0.05** | 0.25 | 0.05 | **88.94%** | 99.04% | 89.27% | 85.00% | 0.3774 |
| 1 | 0.30 | 0.25 | 0.05 | 88.08% | 98.94% | 88.44% | 83.75% | 0.3835 |
| 2 | 0.05 | **0.10** | 0.05 | **89.71%** | 98.94% | 89.58% | 91.25% | 0.3642 |
| 2 | 0.05 | 0.50 | 0.05 | 88.27% | 98.75% | 87.92% | 92.50% | 0.3889 |
| 3 | 0.05 | 0.10 | **0.02** | **90.19%** | **99.33%** | **90.62%** | 85.00% | 0.3655 |
| 3 | 0.05 | 0.10 | 0.10 | 90.10% | 99.04% | 90.31% | 87.50% | **0.3431** |

Stage winners were `(0.05, 0.25, 0.05)`, then `(0.05, 0.10, 0.05)`, then the **selected tuple `(0.05, 0.10, 0.02)`**. The selected tuple beats the seed-45 baseline by **+1.44 CIL points** and **+1.98 old-class points**, but newest-task CIL falls **5.00 points**. It exceeds the stage-3 `f=0.10` runner-up by only **one correct validation sample** (90.19% versus 90.10%), so the last-coordinate choice is fragile. The [stage 1](data/isolet-hparam-stage1.json), [stage 2](data/isolet-hparam-stage2.json) and [stage 3](data/isolet-hparam-stage3.json) ledgers record candidate scores, readout hashes and the fixed ranking rule before seed-46 confirmation.

## Independent-seed paired check

Both seed-46 runs used the same 3080 host, formal source, ISOLET payload, task/split protocol and 20/class replay budget.

| Seed 46 | CIL | TIL | Old CIL | Newest CIL | NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Formal tuple `(0.15, 0.25, 0.05)` | 89.81% | 99.04% | 89.58% | 92.50% | 0.3640 |
| Selected tuple `(0.05, 0.10, 0.02)` | **91.15%** | 99.04% | **91.04%** | 92.50% | **0.3561** |
| Selected minus formal | **+1.35 pp** | 0.00 pp | **+1.46 pp** | 0.00 pp | −0.0078 |

The CIL gain repeats on seed 46 without newest-task or Task-IL loss. It does not erase seed 45's 5-point newest-task decline. This is a small two-seed development comparison on a validation cohort also used for gate selection. It is **not** evidence that formal test accuracy improved, nor that the tuned tuple is robust across more seeds/datasets. The existing 14-method formal baseline table was not changed or rerun. Because this sweep ran on a different physical GPU than that table, the within-host seed-46 pair is the relevant control; do not compare its raw numbers to the old 4090 formal test numbers as if they were matched runs.

## Reproduction and decision

All nine runs and their hashes are indexed in the [aggregate ledger](data/isolet-hparam-sweep-seed45-46.json), SHA-256 `29aca4cbf13e959e22b0a10e019a9aec37016217fd91d70b610d61f676067739`. Its stage-ledger SHA-256 values are `ca53cf487768803bfdd91c707f4c79164afaca56f386bcdbafb4c20d76b3c95f`, `21a7073c5df93e99998d5d1d1d5b508226db56439eb1e2c57ffb939afd936696` and `b6683db6bda9ad84ae5d4dfaf55f94e980332d7ba6595acf57edbd1663936249`. The stage winners and seed-46 deltas were independently recomputed from these files. The local and server-focused launcher tests each passed four tests. The producer worktree remained clean.

**Record `(0.05, 0.10, 0.02)` as the best tuple under the prespecified CIL criterion**, with a newest-task caution. Do not overwrite the formal method or promote this to a test-set claim. If the user wants a final accuracy comparison, freeze the tuple first and run a separate audited evaluation; if newest-task stability is important, repeat on more fresh seeds before adopting it.
