# CIFAR Frozen-Feature Full-Head Capacity Screen

Date: 2026-10-07. Exploratory, posthoc mechanism diagnostic; **not** a method-selection or formal performance claim.

## Question

The seed-45 no-consolidation CIFAR run has validation CIL accuracy 15.76% and Task-IL accuracy 74.20%. A training-only scalar old/new offset failed its prespecified newest-task guard; saved BiC reaches 33.28% CIL. Can a complete 100-way linear head, fitted on exactly the same frozen final encoder features, recover substantially more cross-task discrimination? The representation-quality linear-probe framing is established in class-incremental learning, but this run is limited to an explanatory upper-capacity check.

## Fixed diagnostic

Read the completed seed-45 pilot checkpoint from the exact formal-source commit `7bfe6b1d724fb1206bc0053a9008126bad86332d`. Rehash config/checkpoint and all CIFAR payloads; compare both rebuilt BiC and lambda-validation manifests with the saved versions. Use the same deterministic evaluation transform and the same first 20 eligible **training** examples per class selected for the scalar pilot, excluding both holdouts. Freeze and hash all four bottom encoders. Extract each sample's aggregated feature once.

Call the repository's existing `head_consolidation.consolidate_classifier` on the 2,000 balanced training embeddings with the previously frozen Full-branch settings: regularization `0.01`, Adam learning rate `0.01`, `500` steps, and `20` examples per class. These constants are not tuned against this validation split. Fit only the 100-way linear classifier, keeping bottom encoders and any calibration/gate disabled. Evaluate raw and fitted heads on the saved 2,500-example training-validation split. Report CIL, old/new accuracy, Task-IL, old→new/new→old error rates, and NLL. Save aggregates and the fitter's audit only, without individual images, features, or logits.

## Interpretation boundary

The same validation split has already been viewed in the scalar pilot and was used for ordinary per-task readouts. This screen can describe whether the fixed final representation is usable by a more flexible head, but cannot confirm an improvement or justify selecting this head for a paper. Compare with raw, scalar-offset, and saved BiC results as context only. Do not tune any hyperparameter, relaunch another variant, access final test data, or merge this exploratory result into formal tables. If a future method is proposed, test it under a newly preregistered and independent training/validation run.
