# ISOLET online head-memory capacity curve

Date: 2026-10-09. The user approved an ISOLET development screen of 20, 40 and 80 raw replay examples per class after two same-budget head candidates failed and a non-deployable 158/class head fit showed capacity headroom.

## Fixed scientific protocol

Use exact clean producer commit `a575bbf446ae501cf8e580ba62c2cc5492a25f30`, fixed loss tuple `(proto_lambda_a, distill_weight, feat_distill_weight)=(0.05,0.10,0.02)`, 13 two-class tasks, four views, 50 epochs/task, same task-time normalized-feature herding, and final-only adaptive head. Run fresh seeds 51 and 52 at per-class raw head-memory capacities 20, 40 and 80. The original 40/class gate validation split remains fixed. A new independent 40/class holdout from the training pool, split seed `20261012`, is excluded from all training and gate fitting. Formal data split, test set and baseline results remain untouched.

`config.validate_adaptive_head_consolidation` and `fit_adaptive_candidates` lock the production adaptive head to 20/class. The external launcher may override only this validation field for the **development memory screen**, while retaining all other validated adaptive options; the producer source worktree stays unchanged. Each training stream chooses and stores raw replay at task time under its stated capacity. The source's frozen adaptive Full/Bias candidates still use only the first 20/class; therefore a separate read-only head screen must fit Full heads at 20, 40 and 80/class from each stream's saved replay, starting from its saved pre-head state. The 20/class refit must reproduce the frozen Full state hash. Paired pre-head encoder/head identities must match across capacities within each seed, or the comparison is not head-only.

## Outcome and decision boundary

Report independent-holdout Class-IL, Task-ID, Task-IL, old/new Class-IL, old-to-old task errors, raw memory bytes/count and head-fit time for each capacity and seed. A capacity is promising only if both seeds improve CIL over 20/class, mean gain is at least +1.00 percentage point, and newest-task decline is no worse than 1.00 point per seed. This is a resource tradeoff screen, not a same-budget formal method comparison. If larger memory is promoted, integrate capacity into the adaptive fit and strict audit, then rerun relevant replay baselines at matched memory budget before a fair paper claim. No test-loader iteration during development selection.
