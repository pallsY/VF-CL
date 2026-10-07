# CIFAR-100 Equal-Memory Herding Screen

Date: 2026-10-07 (Asia/Shanghai). Status: **representative-selection signal** in one preregistered development run; not a new method claim.

## Locked source and data boundary

The equal-memory comparison was specified before training at commit `18d18fad6480f0854b42b9fbdc322c80544d2f4d`. A fresh seed-47 ten-task CIFAR stream used the exact audited producer source commit `7bfe6b1d724fb1206bc0053a9008126bad86332d`. Its new lambda-validation split seed `20261009` and disabled final head consolidation were among exactly eight recorded overrides from the frozen seed-42 Adaptive config (SHA-256 `5ef9984e091b83d7f84ac373cd702811c0ed15595dc6f4075ca23cc515535955`). The executed launcher SHA-256 was `c9f4f198c9a7ae6eda9c3c83b3c7477327dc1151aa0620af1c2faf2d97d861d7`.

Training root: `/home/c3080/YangXiaoXiang/VF-CL/results/cifar-herding-seed47-20261009-v1/seed_47_baseline`. Its completion marker binds config SHA-256 `171e9602e53ff3566a20a1b9e02d436134325c3b8e3af31e91a67bebfab28693`, final checkpoint SHA-256 `205edde799806f6cf9984aed1bd0bcf089db85b37a71efc7e765806bb477b376`, and results SHA-256 `382e8aecea5a4a25bf8b32a7eb279ebd81604adea6fd73060b208ab988ef9c17`. All 10 CIL events completed. The data-flow audit has training, validation and BiC-calibration access, with **zero test-loader access**.

The frozen analyzer at commit `c3c9734` had SHA-256 `4a626edc1dd12aa9cf8f4965adab7ae1e1608e5971223055755a0390447003bd`. It verified formal source config/record, exact overrides, training artifacts, producer modules including `cl_methods/proto_evolve.py`, data and scripts, and both saved holdout manifests. From 45,000 eligible **training** images (450/class, excluding both 2,500-example holdouts), it compared the first 20/class with the repository's unmodified normalized-feature `herding_indices` at 20/class. It fitted independent copies of the same original 100-way head with the same regularization `0.01`, Adam learning rate `0.01`, and 500 steps. The four bottom encoders and saved checkpoint were unchanged. Only aggregate metrics and selected-index hashes were saved.

Aggregate output: `/home/c3080/YangXiaoXiang/VF-CL/results/cifar-herding-seed47-20261009-analysis-v1/herding_screen.json`, SHA-256 `667c2627369f585210aaf4d08d56741ea90e0c7be525ecdd5ad105adea462725`. The same aggregate JSON is checked in at `docs/superpowers/reports/data/cifar-herding-seed47-20261009.json`; a separate repeat produced a byte-identical hash.

## New holdout result

| Readout | Raw images/class fitted | CIL | Old classes | Newest task | Task-IL | Validation NLL | Fit CE after 500 steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original final head | — | 17.00% | 9.87% | 81.20% | 75.48% | 3.347 | — |
| Fixed first-20 Full head | 20 | 29.08% | 28.44% | 34.80% | 68.44% | 3.183 | 0.213 |
| Herding-20 Full head | 20 | **34.00%** | **33.82%** | **35.60%** | **72.04%** | 2.965 | 0.331 |

The saved BiC readout from this same training run is context only: 32.56% overall, 32.67% old, and 31.60% newest-task accuracy on this validation split. It used a separate fixed calibration holdout and was not involved in the herding decision.

Herding-20 gained **4.92 CIL points**, 5.38 old-class points, 0.80 newest-task points and 3.60 Task-IL points over fixed first-20. Validation NLL fell by 0.218. The mean L2 error between each class's normalized full-training-feature center and its 20 selected normalized features fell from `0.06499` (first-20) to `0.03069` (herding-20). Only 96 of the 2,000 selected training examples overlapped between selectors. Herding's fit CE (`0.331`) was higher than first-20's (`0.213`), while validation NLL was lower, consistent with improved coverage rather than simply better fitting the selected examples.

## Decision and limitations

The prespecified label required at least +3 CIL points with no more than 2 points of loss in either old or newest-task accuracy. Herding-20 passed both group guards, so the aggregate result is `representative_selection_signal`.

This is **one seed**, and herding itself is established prior art (for example [iCaRL, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf)). More importantly, the diagnostic selected old-class examples **after** training using the final encoder and the complete historical training corpus. A deployed continual-learning method may not revisit all old raw images, so the result is an offline selection signal, not proof of an online memory-limited or privacy-preserving algorithm. It also does not prove that class-center approximation causes the accuracy gain. Do not enter this figure into formal tables or claim novelty from plain herding. A future VFL-specific proposal must select or update bounded examples when the class is available, account for party communication/privacy, and beat this existing herding control under fresh seeds and ISOLET/UPMC regression checks.
