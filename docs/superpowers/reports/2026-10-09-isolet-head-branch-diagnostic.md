# ISOLET frozen-head branch diagnostic

Date: 2026-10-09 (Asia/Shanghai). Status: **read-only development diagnosis**, with no model-training or test-data access by the diagnostic.

## Question and source

The [fixed-parameter three-dataset test](2026-10-09-three-dataset-selected-hparam-aa-final.md) gave ISOLET mean final Class-IL accuracy 89.09% and Task-IL accuracy 98.97%. The question here is whether the current Adaptive Full/Bias mixture itself explains the large gap. The three selected ISOLET runs and their same-host old-parameter controls, seeds 42–44, were read from `/home/c3080/YangXiaoXiang/VF-CL/results/three-dataset-selected-hparam-20261009-v1` at exact producer commit `a575bbf446ae501cf8e580ba62c2cc5492a25f30`.

The [analysis script](../../../analyze_isolet_head_branches.py) opened only each frozen `formal_final.pt` checkpoint's saved **training-validation embeddings and labels**, its saved validation manifest, config, and formal publication marker for checkpoint identity. It did not open `results.json`, load test examples, or modify the model. Each validation cohort contains 40 examples per class (1,040 total). The same cohort was already used to select Adaptive's mixture gate and, in earlier development work, the three hyperparameters. These figures are descriptive and cannot validate a newly chosen method.

For each checkpoint the script strictly reloaded the installed top head, recomputed Full, Bias, and Mixed probabilities on the same frozen embeddings, and checked checkpoint/file hashes, the saved gate mixture, validation tensor hashes, and branch NLLs against frozen audit evidence. The focused synthetic [test](../../../test_analyze_isolet_head_branches.py) checks that task-ID and within-task errors are counted separately. All six runs passed the checks.

## Selected-parameter results on gate-used validation

| Head | Final Class-IL | Task-ID accuracy | Task-IL | Newest-task Class-IL | NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 89.58% | 90.32% | 98.97% | 89.17% | 0.3788 |
| Bias | 74.71% | 75.29% | 99.07% | 60.00% | 0.7629 |
| Mixed | **89.62%** | **90.35%** | 99.01% | 89.17% | **0.3720** |

The three frozen gates put weights **0.959, 0.991, and 0.981** on Full. Mixed exceeds Full by only **0.03 percentage points** in mean Class-IL accuracy; replacing Mixed with Full alone would not materially close the ISOLET gap. Bias alone is much worse, especially for the newest task. The embedded pre/post audit reports Task-ID accuracy rising from **58.43% before** final head consolidation to **90.35% after** it, so removing final head consolidation is not supported by this diagnostic.

Across three seeds there were 3,120 seed-example evaluations and **324** Mixed Class-IL errors. Of those, **301 (92.9%)** predicted a class from the wrong task: **258** confused one old task with another old task, **17** sent an old example to the newest task, and **26** sent a newest-task example to an old task. These are repeated predictions on the same validation cohort under three trained seeds, not 3,120 independent people or samples. A single old-versus-new score offset cannot directly correct the 258 old-to-old task errors.

The old-parameter same-host controls show the same qualitative pattern: Full 89.07%, Bias 77.08%, Mixed 89.04% Class-IL on this cohort. The selected training parameters improve the frozen Full branch, but the near-identical Full/Mixed result remains.

## Decision

Keep `(proto_lambda_a, distill_weight, feat_distill_weight) = (0.05, 0.10, 0.02)` and the existing final head for now. The next experiment should target **discrimination among all 13 tasks** under the existing 20-examples-per-class budget, rather than tune the Full/Bias gate or a single old/new offset. A narrow pilot can compare representative head-replay selection or task-level score calibration with the current head, on a fresh training-only holdout excluded from pilot training and gate fitting. Keep the formal dataset split, task order, and evaluation definition fixed. If a frozen-head candidate does not yield a repeatable development gain without a newest-task regression, investigate encoder/replay representation instead. The present formal test set must not select that candidate.

The aggregate [audit ledger](data/isolet-head-branch-diagnostic-20261009.json) contains per-seed branch metrics, task confusion counts, frozen audit summaries, gate values, and checkpoint/config/validation-manifest hashes. SHA-256: `6d8bba798381385a0a0044c54b45721a7c11de142145f14a57d43d2cf301b2f2`. The analysis-script SHA-256 is `02838832724a6d5cedb8b0bac0d1b3fa4228a8558ac26955a737682618db8d46`.
