# ISOLET prototype-generated feature head pilot

Date: 2026-10-09 (Asia/Shanghai). Status: **development candidate failed the locked accuracy gate**. No training-method change or formal test-set result is claimed.

## Locked protocol

The [design](../specs/2026-10-09-isolet-prototype-feature-head-pilot-design.md) and [plan](../plans/2026-10-09-isolet-prototype-feature-head-pilot.md) were committed before outcomes. The exact clean producer commit was `a575bbf446ae501cf8e580ba62c2cc5492a25f30`. Training-only ISOLET seeds **49 and 50** kept the selected tuple `(proto_lambda_a, distill_weight, feat_distill_weight)=(0.05,0.10,0.02)`, current herding, 20 persistent raw examples/class, 13 two-class tasks, four views, 50 epochs/task, and the final-only adaptive head. The gate retained its original 40/class training-validation cohort. A separate 40/class holdout, split seed `20261011`, was excluded from both training and gate fitting. The extended external [launcher](../../../launch_isolet_replay_selection_pilot.py) preserved the historical 47/48 defaults; its exact revision hash is listed below. The former revision for the earlier replay pilot remains at Git commit `9dc0ab5`.

Both streams stopped after freezing the final adaptive checkpoint and before deferred test evaluation. For each frozen seed the [head analysis](../../../analyze_isolet_prototype_feature_head.py) started from the same pre-consolidation classifier and encoder, then fitted two **Full** heads with the same Adam optimizer, 500 steps, learning rate 0.01 and regularization 0.01:

- **Control:** the saved 20/class herding embeddings. The refit reproduced the original frozen Full-head state SHA-256 exactly.
- **Candidate:** those same 20 raw-replay embeddings plus 20 transient feature vectors/class generated deterministically from the method's already stored class mean/std: one exact mean and 19 diagonal-Gaussian draws. The generated vectors were used only during this head fit and were not saved as persistent memory.

This head-only test does not yet modify the production adaptive gate or checkpoint-audit protocol. It evaluates the frozen heads on the new training-only holdout. The vector NPZ loader materialized fixed test arrays at initialization, but no test loader was iterated and no test labels were used for fitting or scoring.

## Holdout results

| Seed | Full20 Class-IL | Raw20 + synthetic20 Class-IL | Change | Task-ID change | Task-IL: control → candidate | Newest-task change | Old→old task errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 49 | **88.27%** | 87.69% | **−0.58 pp** | −0.67 pp | 99.13% → 99.23% | 0.00 pp | 109 → 115 |
| 50 | **87.98%** | 87.69% | **−0.29 pp** | −0.29 pp | 99.04% → 99.04% | 0.00 pp | 109 → 110 |
| **Mean** | **88.13%** | **87.69%** | **−0.43 pp** | | | | |

The candidate lowered Class-IL on both fresh seeds; the newest task was unchanged and old-to-old task errors rose by 6 and 1. The locked rule required both CIL deltas to be positive, mean gain at least +1.00 pp, and no newest-task decline beyond 1.00 pp per seed. The newest-task guard passed, while both CIL requirements failed. Generation took about 0.007 seconds per seed and each classifier fit about 0.6 seconds on this host; these small timings describe only the isolated head step, not a full production runtime comparison. The candidate uses no extra persistent raw examples, but does create 520 temporary feature vectors and perform extra generation/fitting work.

## Decision and limits

**Keep the current 20/class herding and adaptive head; do not integrate this simple Gaussian augmentation.** The earlier non-deployable [158/class capacity screen](2026-10-09-isolet-frozen-head-capacity.md) showed that the encoder supports roughly 94% Class-IL on its own fresh development cohort, but this fixed 20+20 prototype recipe did not recover that headroom on the new cohort. The capacity screen's offline training-pool access is not a valid continual-learning method. This negative result does not rule out richer feature-distribution models or other bounded head objectives; further candidates need a new, predeclared training-only comparison. Existing formal test results and baseline tables were not changed.

Both training runs exited 0, saved all 13 event checkpoints and the final adaptive checkpoint, retained exactly 20 raw samples/class, and had zero test-loader records. The two new holdout/gate manifests were identical across seeds, and completion-marker hashes were checked against checkpoint, event, audit and manifest files. The [aggregate ledger](data/isolet-prototype-feature-head-pilot-20261009.json) stores per-seed metrics, hashes, raw/temporary counts, fit timings and the locked decision. Ledger SHA-256: `22fdc62c28ede3edebd737dd7a3fec8d936ff04f7d71bcb0e7a550bee0000e20`; launcher SHA-256: `6396ed5d4130772717fe03218bbd553630c16338f6b05052591112805feac996`; analysis-script SHA-256: `fdce8f270f25e4e4fea1927be550f42b59d45601a56c26a3cc813ead4614eb2d`.

Training root: `/home/c3080/YangXiaoXiang/VF-CL/results/isolet-prototype-feature-head-pilot-20261009-v1`. Head-readout root: `/home/c3080/YangXiaoXiang/VF-CL/results/isolet-prototype-feature-head-readout-20261009-v1`.
