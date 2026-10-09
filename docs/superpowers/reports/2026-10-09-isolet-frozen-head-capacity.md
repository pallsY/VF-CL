# ISOLET frozen-encoder head-capacity diagnostic

Date: 2026-10-09 (Asia/Shanghai). Status: **read-only, non-deployable development upper bound**. This experiment does not change the VF-CL method or produce formal test accuracy.

## Question and protocol

The [equal-memory 10+10 replay pilot](2026-10-09-isolet-equal-memory-replay-pilot.md) lost to the existing 20/class herding in both fresh seeds. Before changing the encoder training objective, this diagnostic asks whether the **same frozen final representation** can support a substantially better global classifier when the head is fitted with much more balanced training data.

The source is the completed herding runs for ISOLET seeds 47 and 48, under exact producer commit `a575bbf446ae501cf8e580ba62c2cc5492a25f30` and fixed Adaptive tuple `(0.05,0.10,0.02)`. Each pilot run excluded both the original 40/class gate-validation cohort and a separate 40/class training-only holdout from model training. The [analysis script](../../../analyze_isolet_frozen_head_capacity.py) loaded each frozen `adaptive_final.pt`, re-encoded only the remaining training examples and the independent holdout through the final encoder, and compared two Full-branch heads starting from the **same saved pre-consolidation classifier**:

- **20/class:** refit the exact 20 stored herding examples/class with the existing classifier routine. The resulting state SHA-256 matched the saved Full branch, confirming the fitting procedure was reproduced.
- **158/class offline upper bound:** use 158 balanced examples/class sampled deterministically from the post-task training pool (the minimum available per class after exclusions). Refit with the same Adam optimizer, 500 steps, learning rate 0.01 and regularization 0.01.

The offline 158/class fit revisits all historical training tasks after the sequence. It violates the intended bounded, online memory protocol and is **only a capacity diagnostic**. No test loader was iterated, and test labels were not used for fitting or scoring. The vector NPZ loader still materializes its fixed test arrays at initialization.

## Independent holdout result

| Seed | Saved/refit Full with 20/class | Offline Full with 158/class | Gain | Task-ID: 20→158 | Task-IL: 20→158 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 47 | 89.81% | 93.65% | +3.85 pp | 90.19% → 93.85% | 99.42% → 99.62% |
| 48 | 89.52% | 94.42% | +4.90 pp | 89.81% → 94.90% | 99.33% → 99.23% |
| **Mean** | **89.66%** | **94.04%** | **+4.38 pp** | | |

The gain appears mainly in cross-task discrimination: Task-ID accuracy rises by 3.65 and 5.10 points, while within-task accuracy stays near 99%. The same frozen encoder supports substantially higher holdout Class-IL accuracy with a much larger, balanced head-training pool. This contradicts the immediate hypothesis that an encoder loss change is necessary to pass 90% on this development cohort. It does **not** prove that sample quantity alone causes the gain; the available examples, their diversity and their offline access change together.

## Decision

Keep the current three hyperparameters and current 20/class herding as the deployable method. Do not add the failed 10+10 selector or the offline 158/class head to the formal method. The next controlled pilot should seek a better head fit **within the existing 20 raw examples/class memory**, for example by testing training-only prototype-generated feature augmentation with a fixed recipe against the exact current head on a fresh independent development cohort. Record any extra computation and derived-feature storage. Only if bounded head interventions fail should the encoder objective be changed. No formal test data should select that recipe.

The [aggregate data](data/isolet-frozen-head-capacity-20261009.json) contains checkpoint/manifest identities, per-class training counts, both head metrics and the offline fitted-head hashes. Data SHA-256: `549e36c60615834e1045aecd1b2a94d2a7d0dd315d054ea6503d18c446664bed`. Analysis-script SHA-256: `b7d9d609bbeb4c8a88679a28579b79be4ab306de8c126dbdf98d706d1c8c8b7d`.

Output root: `/home/c3080/YangXiaoXiang/VF-CL/results/isolet-frozen-head-capacity-20261009-v1`. The original pilot checkpoints were not modified.
