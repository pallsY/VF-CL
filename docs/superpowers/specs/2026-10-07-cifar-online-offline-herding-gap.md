# CIFAR Online/Offline Herding Representation Gap

Date: 2026-10-07. Read-only, training-only mechanism audit. No method fit or validation/test accuracy calculation.

## Question

The three audited CIFAR-100 Adaptive final checkpoints each retain exactly 20 raw herding examples for every class, selected at that class's task boundary. A separate seed-47 development run found that 20 examples selected **after** training with the final encoder beat a naive first-20 control. Before designing a new VFL selector, measure how representative the actually stored online examples remain under the final encoder relative to an optimistic final-encoder offline herding set on the **same model and class**.

## Fixed computation

Read only the audited Adaptive method-shard root `/home/c3080/YangXiaoXiang/VF-CL/results/formal-method-adaptive-20261005-v1`, seeds 42, 43, and 44. Require its method success marker, each completed record's checkpoint/config/dataset/validation-manifest hashes, the exact producer source commit `7bfe6b1d724fb1206bc0053a9008126bad86332d`, and all 100 saved classes with 20 online replay examples each. Rebuild and compare both BiC and lambda-validation manifests. Candidate pool: all 450 eligible **training** images per class after excluding both 25-per-class holdouts. Extract final-encoder aggregated and four party embeddings on this pool under the deterministic evaluation transform. Embed each saved online raw replay tensor with the same final encoder. Run the repository's unchanged `herding_indices` on the full eligible class pool to select 20 offline examples.

Normalize each embedding vector to unit length, as in `herding_indices`. For each class, compute the L2 distance between the all-eligible class mean and the mean of (a) online saved20 and (b) offline final-encoder20, first in the aggregated feature space and then separately for each party. Report per-class errors, three-seed means/medians, the fraction of classes whose online error exceeds offline error, and paired differences. Do not fit a head, evaluate validation/test labels, alter the stored replay, or save raw samples/embeddings/individual predictions. Save only class-level errors, counts, hashes, and aggregate summaries to a new root.

## Interpretation boundary

Offline herding is an optimistic reference because it can revisit the full historical training pool with a final encoder. A smaller offline error is expected by construction; the useful evidence is its **magnitude and consistency**, including across parties. The online stored tensors reflect random training augmentation, while the offline candidate pool uses deterministic evaluation transforms; this view mismatch prevents attributing the entire gap to selection timing or feature drift. The formal Adaptive gate already used its validation cohort, so this audit avoids validation accuracy entirely. No binary success threshold or new method claim is attached. If a material online/offline gap appears, a later VFL-specific online candidate must be compared with existing online herding at equal 20-per-class memory and measured communication/privacy cost under a new independent run.
