# CIFAR-100 view-matched online herding pilot

Date: 2026-10-08. The user approved the next controlled pilot after the online/offline representation audit. This is one independent development seed and a training-only mechanism analysis, not a formal result or a new selector.

## Question

The seed-42/43/44 audit found a mean final-encoder class-centroid error of 0.05606 for saved online augmented replay and 0.03068 for offline final-encoder herding. How much of that gap disappears when the **same online-selected original images** are evaluated with the deterministic transform used for the full candidate pool?

## Approaches considered

1. **Recover original CIFAR IDs from saved augmented replay (chosen).** Match a central pixel patch against eligible same-class training images under all 9×9 crop positions and horizontal flip states. Confirm each candidate by reproducing the entire augmentation byte for byte. Fail if an augmented replay image has zero or multiple source IDs. This leaves the producer source and replay behavior unchanged.
2. Modify the producer to emit sample IDs during selection. This is more direct, but changes the audited training path and needs a new method checkpoint format.
3. Rerun selection on deterministic views only. This changes which images herding chooses, so it does not isolate the view effect for the actually retained samples.

## Fixed experiment

First verify exact and unique ID recovery on the complete 2,000-image replay in one existing formal checkpoint, using only CIFAR training images and its BiC/lambda holdout manifests. If recovery fails, do not launch or interpret the new run. Then train seed 48 on the unmodified clean producer commit `7bfe6b1d724fb1206bc0053a9008126bad86332d`, deriving its config from the audited seed-42 Adaptive config. Change only seed, non-formal deferred-evaluation flag, output paths, and experiment name. Keep the audited model, ten-task stream, adaptive head consolidation, 20 images/class, BiC and lambda holdout settings, and deterministic environment. The Adaptive runner must record no test-loader access. This development run may use the existing prescribed validation split; no validation metric will choose or tune the mechanism analysis.

For each class at the final seed-48 checkpoint, evaluate under the same final bottom encoders: (A) the 20 stored randomly augmented online replay tensors; (B) the **same recovered 20 image IDs** under the deterministic training-view transform; and (C) 20 optimistic offline herding IDs selected from all 450 eligible deterministic training images/class with the repository's unchanged `herding_indices`. The target is the normalized-feature mean of the full 450-image eligible pool. Record aggregate and four party centroid errors, paired A−B view effect, paired B−C remaining representation gap, class counts/fractions, selection ID hashes and source hashes. No head fit, held-out accuracy, raw images, embeddings, or individual predictions are output.

## Interpretation and decision

A−B isolates the effect of the saved augmented versus deterministic view for the **same images** under the same final encoder. B−C remains an optimistic offline comparison; it can reflect task-time selection, encoder drift, different candidate availability, and offline access to the final encoder. It is not a causal estimate of any one factor. Report all signs and magnitudes without a significance threshold or a novel-method claim. If B−C remains material, a later party-aware online selector can be tested against existing online herding at equal memory and measured communication/privacy cost. If it nearly disappears, fix the replay-view protocol before designing a selector.

## Reproducibility gates

Save a locked config/protocol manifest before training. Require source commit and clean producer worktree, formal source config/data hashes, exact override set, new output root, physical GPU 1, deterministic environment, all ten CIL events and no test-loader access. The analyzer must verify training completion/checkpoint/config/manifests and recover exactly one eligible source ID per saved replay tensor. Run the analysis twice to separate fresh roots and compare JSON SHA-256. Commit script, tests, report, and class-level aggregate JSON to the GitHub branch.
