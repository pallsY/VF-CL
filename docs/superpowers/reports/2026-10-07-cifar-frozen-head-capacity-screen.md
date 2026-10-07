# CIFAR-100 Frozen-Feature Full-Head Capacity Screen

Date: 2026-10-07. **Exploratory posthoc diagnostic**, not a formal method result or a new head selection.

## Protocol and provenance

The screen was specified before execution at commit `adfc66c` and implemented initially at `c747ee2`. It reuses the completed seed-45 no-consolidation CIFAR model from the preceding head-offset pilot. The source trainer/checkpoint was not modified. The final script SHA-256 was `d87df947ce4b09a870c91b1630e147b1beeb51ca5ec9ab6758363fbcd5e2477f`. It ran from the exact clean formal producer checkout `7bfe6b1d724fb1206bc0053a9008126bad86332d`, verifying five imported source-module hashes against the formal record, the formal record hash against the pilot protocol, the frozen analysis-helper SHA-256 `346b3b95a5abe2f5d7c749bf2b4780b1fbbebd5a5ce7843ff43cab684b9682f9`, every CIFAR payload hash against the pilot protocol, the completed training marker and data-flow audit hash, and the absence of test-loader accesses.

Both rebuilt holdout manifests matched the saved BiC and lambda-validation manifests. Exactly 20 nonholdout training images per class (2,000 total) were selected under the deterministic evaluation transform; 2,500 held-out training-validation images formed the evaluation cohort. All four bottom encoders were frozen, and their state hash `1923f34fa81160ca4ed6e1f06820f6d5172e754fa942eb0275020` was unchanged after fitting. No individual image, feature, or logit was saved.

The only fit was the repository's existing `consolidate_classifier` on frozen aggregated training embeddings, with its already established Full-branch settings: regularization `0.01`, learning rate `0.01`, 500 Adam steps, and 20 examples per class. The fitter reported zero validation/test use. Its audit parameter `persistent_raw_example_count=2000` denotes the training images selected through the existing dataset API; this diagnostic did not create a new persistent raw-image store.

Aggregate result: `/home/c3080/YangXiaoXiang/VF-CL/results/cifar-frozen-head-capacity-seed45-20261007-v2/frozen_head_capacity.json`, SHA-256 `b9e206af82e110ebe94a8fbccfcd97596dfe91186bf29ab132935b756da9eee3`. The same aggregate JSON is checked in at `docs/superpowers/reports/data/cifar-frozen-head-capacity-seed45-20261007.json`. A separate repeat produced a byte-identical output hash; the v1 and v2 numerical metrics and fit audit are identical.

## Observed result

| Readout on the same seed-45 training-validation split | CIL | Old classes | Newest task | Task-IL | Validation NLL |
|---|---:|---:|---:|---:|---:|
| Original unconsolidated head | 15.76% | 8.76% | 78.80% | 74.20% | 3.3352 |
| Frozen-feature balanced Full head | 30.12% | 29.56% | 35.20% | 66.52% | 3.1141 |
| Earlier one-scalar offset | 26.44% | 25.20% | 37.60% | 74.20% | 2.8248 |
| Existing BiC readout from the same run | 33.28% | 33.20% | 34.00% | not recomputed here | not recomputed here |

The Full head raised overall CIL by 14.36 points over the original head, but lost 43.60 points on the newest task, reduced Task-IL by 7.68 points, and remained below existing BiC's 33.28% CIL. Its training-only replay cross-entropy fell from `2.93475` to `0.24381`, while validation NLL remained `3.11406`. This large train/validation gap is consistent with overfitting to the fixed 20-per-class subset; it does **not** establish that the frozen representation is intrinsically insufficient.

## Decision boundary

This is a posthoc mechanism screen on a validation split already inspected in the scalar pilot. It cannot be used to claim a new method improvement or tune more heads on this split. The fixed 20-per-class Full-head recipe does not solve the old/new tradeoff. Stop head-rule iteration here. A future confirmatory attempt needs a new training/holdout run and should separately test whether the limiting factor is scarce or unrepresentative training replay, cross-task feature overlap, or both. No additional seed, calibration variant, or final-test evaluation was launched from this result.
