# CIFAR-100 Equal-Memory Herding Screen

Date: 2026-10-07. Status: preregistered seed-47 development comparison, not a new method claim.

## Question and prior art

The independent seed-46 frozen-head screen found 20→100→400 training images per class improved CIFAR-100 CIL accuracy from 30.96% to 38.88% to 41.00%. The next question is narrower: with the **same 20 raw training examples per class**, does a representative selection rule recover part of that gap without sacrificing either old or newest-task accuracy? Herding of normalized feature vectors is established exemplar-selection prior art (for example [iCaRL, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf)), and `cl_methods.proto_evolve.herding_indices` already implements it. A positive result would motivate a later VFL-specific limited-memory method; herding itself is not claimed as novel.

## Fresh training run

Use the exact audited CIFAR producer source commit `7bfe6b1d724fb1206bc0053a9008126bad86332d` in the clean isolated 3080 worktree. Derive the frozen seed-42 Adaptive config with exactly the same eight deviation keys as the prior development runs, now using `seed=47`, `lambda_validation_split_seed=20261009`, `formal_deferred_evaluation=false`, `head_consolidation_enabled=0`, `head_consolidation_mode=full_classifier`, and a fresh root/name. Keep all other task, bottom training, optimizer, party KD, BiC, data, and checkpoint options fixed. Run the ten-task CL-only stream deterministically on physical GPU 1. The lambda-validation cohort is selected before training; no final-test loader may be used.

## One locked equal-memory comparison

After training, freeze all four bottom encoders. Rehash completed training evidence and the original CIFAR payloads; require both rebuilt BiC and lambda-validation manifests to match the saved manifests. From the true training partition, excluding both holdouts, extract the final aggregated feature for every eligible training image under deterministic evaluation transforms. For each class, use the first 20 eligible images as the control. For the candidate, run the repository's unmodified `herding_indices` on that class's final-encoder features and select exactly 20 distinct images. Reuse the same fixed feature matrix; only the **selected indices** differ. Record each selector's mean class-centroid approximation error on L2-normalized training features as a training-only mechanism check.

Starting from independent copies of the same original final classifier, fit one 100-way Full head per selection with the existing `consolidate_classifier` recipe: regularization `0.01`, Adam learning rate `0.01`, 500 steps, 20 selected training features per class. Evaluate the raw, first-20, and herding-20 heads once on the new 25-per-class lambda-validation split. Report overall/old/new CIL, Task-IL, NLL, and old→new/new→old rates. Include the ordinary saved BiC readout as context only. Save aggregates and selection-index hashes, not raw images, individual features, or logits. The bottom encoders and source checkpoint remain unchanged.

## Prespecified stop rule and limits

Label a **representative-selection signal** only if herding-20 exceeds first-20 by at least 3 CIL percentage points and neither old nor newest-task accuracy falls by more than 2 points. Otherwise stop this herding direction. No hyperparameter, sample count, selector, or success threshold may change after viewing validation outcomes. This one-seed screen cannot establish a publication result; it does not isolate herding from final-encoder access or communication cost. Do not launch extra seeds, fit another head, access final test data, or add its number to formal benchmark tables automatically.
