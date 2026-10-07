# CIFAR-100 Frozen-Head Sample-Budget Screen

Date: 2026-10-07 (Asia/Shanghai). Status: **sample-sufficiency signal** in one preregistered development run; not a formal method claim.

## Locked protocol and evidence

The comparison was specified before training at commit `645bb89f97feac5386c09ffec61bde7c10814eb8`. A new seed-46 ten-task CIFAR stream used the exact audited producer source commit `7bfe6b1d724fb1206bc0053a9008126bad86332d`. Its new lambda-validation split seed `20261008` and disabled final head consolidation were among exactly eight recorded overrides from the frozen seed-42 Adaptive configuration. The launcher SHA-256 was `3be61cff8330c7a75970cf4567824a0ee1e580755293207713503ca38cfeb606`; the source config SHA-256 was `5ef9984e091b83d7f84ac373cd702811c0ed15595dc6f4075ca23cc515535955`.

Training root: `/home/c3080/YangXiaoXiang/VF-CL/results/cifar-head-budget-seed46-20261008-v1/seed_46_baseline`. Its completion marker binds config SHA-256 `46115d80edc303487ef144df3cf2a604e875032d0c80672fdddb0d03dd656c9e`, final checkpoint SHA-256 `38c90c564fcb895a4f216fac1be969a3f3fef5e792f1b684508f5b501536be39`, and results SHA-256 `3d82ac142e13d1c15174e9b9da3f3f0f00e045dcc73adfa3c3b5171017d120f2`. All 10 CIL events completed. The data-flow audit contains training, validation, and BiC-calibration access, with **zero test-loader access**.

The frozen analyzer at commit `8e9e095` had SHA-256 `d04ebdcc16b8ba5a424f195611487809db6dd9888df8503fe46bda3775a22492`. It rehashed the formal source config/record, exact eight config changes, training artifacts, five imported producer modules, helper and launcher scripts, all CIFAR payloads, and both saved holdout manifests. It used nested first-20, first-100, and first-400 **training-only** images per class, excluding both 2,500-example BiC and lambda-validation cohorts. Only the 100-way linear head was fitted, independently from the same initial classifier for each budget; the four bottom encoders and final training checkpoint were unchanged. No image, individual feature, or logit was saved in the aggregate output. The fitter audit's `persistent_raw_example_count` field denotes images read from the existing training corpus; this screen did not create a persistent replay store.

Output: `/home/c3080/YangXiaoXiang/VF-CL/results/cifar-head-budget-seed46-20261008-analysis-v1/head_budget.json`, SHA-256 `37717a2da9cc04ba90ebeeb0a84373e55c1bbb59e1ebcf023941ce187b95d573`. The same aggregate JSON is checked in at `docs/superpowers/reports/data/cifar-head-budget-seed46-20261008.json`. A separate repeat produced a byte-identical hash. The validation set has 25 examples per class (2,500 total). Its frozen features were extracted once; each of the three fixed head fits was scored once, with no refit after scoring.

## Results on the newly held-out training-validation split

All three head fits used the same existing Full-branch recipe: regularization `0.01`, Adam learning rate `0.01`, and 500 steps. No setting was selected after viewing validation performance.

| Readout | Fit images/class | CIL | Old classes | Newest task | Task-IL | Validation NLL | Fit CE after 500 steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original final head | — | 18.24% | 11.47% | 79.20% | 75.24% | 3.273 | — |
| Balanced Full head | 20 | 30.96% | 31.16% | 29.20% | 69.24% | 3.103 | 0.231 |
| Balanced Full head | 100 | 38.88% | 38.89% | 38.80% | 75.88% | 2.427 | 1.136 |
| Balanced Full head | 400 | **41.00%** | 41.16% | 39.60% | 77.80% | 2.261 | 1.564 |

The existing saved BiC readout from this same run is a contextual control: 34.80% overall, 35.47% old, and 28.80% newest-task accuracy on this validation split; it used its separate fixed calibration holdout. It was not a budget-selection target.

Increasing the fixed head-fit corpus from 20 to 400 per class raised CIL by **10.04 points**, old-class accuracy by 10.00 points, newest-task accuracy by 10.40 points, and Task-IL by 8.56 points. Validation NLL fell by 0.842. The 20-per-class training/validation CE gap was about 2.872, versus 0.697 at 400 per class. The gains from 100 to 400 per class were smaller (2.12 CIL points), suggesting diminishing returns within this locked recipe.

## Prespecified decision and limits

The preregistered sample-sufficiency label required the 400-per-class head to exceed the 20-per-class head by at least 5 CIL points without more than 2 points of loss in either old or newest-task accuracy. It passed both group guards, so the output label is `sample_sufficiency_signal`.

This is one seed and a development validation result. It shows that the fixed final features can support a better balanced head when many more **training** examples are available; it does not establish that a 400-per-class raw replay is a practical or novel method, nor does it isolate sample quantity from example diversity. Do not adopt the best budget as a paper method or access the final test set from this screen. The next confirmatory step, if pursued, is a memory-limited representative replay or feature-replay design with a fresh seed/holdout, compared against the existing BiC and relevant CL baselines while checking ISOLET/UPMC regressions.
