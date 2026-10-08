# ISOLET Adaptive one-factor hyperparameter sweep

Date: 2026-10-08. The user approved sequential tuning of three effective Adaptive training hyperparameters while leaving ISOLET data splits, task order, evaluation formula, model/memory budget and all other baselines unchanged. This is a development sweep, not an amendment to the completed formal baseline matrix.

## Fixed source, data and host

Use the formal ISOLET Adaptive source commit `a575bbf446ae501cf8e580ba62c2cc5492a25f30` in a clean isolated checkout on the 3080 host. The cloud 4090 host's main disk has only about 2 GiB free, so no new training is launched there. The 3080 host has the ISOLET NPZ and metadata with the same SHA-256 as the formal run (`d34312670de93198afcd2b126c95b79bae2b4cffeb30d480f097faf046b69514` and `79396dea1751b6094a5769f2dd789ad58b3ea9582a12c8c98d3ec07d8d1eb3cd`). Pin the formal source record SHA-256 `7dc5bf930600d097134948d6ee166fd55490327d16cde1fe2496ac374a73299b` and config SHA-256 `5702b8846ca8e9d279728a725550c1b053b8580fec832a9695b56d6ea7f2af34`. The path/device/output changes required by the second host are recorded separately from the three tuned values.

Keep the 13 two-class tasks, four ISOLET views, 26 classes, 40/class lambda-validation manifest, 20/class head replay, adaptive dual-branch head, 50 epochs/task, batch size 128, AdamW and all other losses fixed. Use physical GPU 1 masked as `cuda:0`, deterministic environment, and new result roots. The external development launcher must stop after final checkpoint freeze and before deferred test evaluation. It must verify zero test-loader access and rebuild the original validation manifest exactly. Do not modify or append to the formal result root.

## Sequential sweep

Development seed 45 begins at the exact formal values:

| Stage | Parameter | Candidate values | Formal value |
| --- | --- | --- | --- |
| 1 | `proto_lambda_a` | `0.05`, `0.15`, `0.30` | `0.15` |
| 2 | `distill_weight` | `0.10`, `0.25`, `0.50` | `0.25` |
| 3 | `feat_distill_weight` | `0.02`, `0.05`, `0.10` | `0.05` |

Stage 1 changes only `proto_lambda_a`; hold its best value in stage 2. Hold stages 1–2 best values in stage 3. Reuse a completed run if its exact parameter tuple already exists. Seven distinct seed-45 runs are expected: one baseline plus two new alternatives per stage. Pick the highest final **class-incremental accuracy** on the same rebuilt 40/class lambda-validation cohort; tie-break by old-class CIL, then Task-IL, then smaller deviation from the formal values. Record all candidate CIL, Task-IL, old classes 0–23 and newest task 24–25, NLL, gate and configuration/checkpoint hashes. The lambda-validation cohort is also used by the Adaptive gate, so these scores are development-selection evidence, not an untouched test estimate.

After selecting the final tuple, run **both the formal baseline tuple and the selected tuple on a new seed 46** under identical host/data/split settings. This paired confirmation checks whether the seed-45 choice improves CIL without a material old/new or Task-IL regression. It does not replace the formal three-seed table. No final ISOLET test loader is iterated, and test labels are not used for scoring or tuning. The existing vector dataset loader still materializes the NPZ's fixed test arrays at initialization.

## Interpretation and limits

The primary output is the full candidate ledger and selected hyperparameter tuple, with development and paired seed-46 metrics. If the selected tuple fails to improve seed 46, report that instability rather than changing the grid or seed after inspection. The existing 14-method formal baselines stay frozen. Because the sweep runs on a different physical GPU, do not claim exact numeric comparability to the old 4090 formal test table; the within-host paired seed-46 baseline is the relevant control. If a later formal comparison is wanted, freeze this tuple first and run a separate audited evaluation protocol.
