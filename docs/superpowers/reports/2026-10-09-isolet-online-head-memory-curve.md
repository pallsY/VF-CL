# ISOLET online head-memory capacity curve

Date: 2026-10-09 (Asia/Shanghai). Status: **development capacity screen passed for both 40/class and 80/class**. These are independent training-holdout Full-head readouts, not formal test `AA_final` or a production Adaptive method.

## Locked protocol and measurement boundary

The [design](../specs/2026-10-09-isolet-online-head-memory-curve-design.md) and [plan](../plans/2026-10-09-isolet-online-head-memory-curve.md) were committed before outcomes. Six ISOLET runs used new seeds **51, 52** and task-time normalized-feature herding with persistent raw head-replay capacities **20, 40, 80/class**. The producer source stayed clean at commit `a575bbf446ae501cf8e580ba62c2cc5492a25f30`; the selected loss tuple `(proto_lambda_a, distill_weight, feat_distill_weight)=(0.05,0.10,0.02)`, four views, 13 two-class tasks, original class order, 50 epochs/task, optimizer and final-only head schedule were fixed. The gate retained its existing 40/class validation cohort. A separate 40/class training-only holdout with split seed `20261012` was excluded from both training and gate fitting.

The production adaptive config validator and `fit_adaptive_candidates` fix the head-fit capacity at 20/class. The external [development launcher](../../../launch_isolet_replay_selection_pilot.py) bypassed only that config validator field for the 40/80 **pilot**, while saving the true task-time raw replay capacity and marking the override. The source checkpoint's own Full/Bias branches still fit only the first 20/class. A separate read-only [analysis](../../../analyze_isolet_online_head_memory.py) therefore refitted a Full classifier at each actual stored capacity from the same frozen pre-consolidation head, Adam/500 steps/LR 0.01/regularization 0.01. Its 20/class refit reproduced each run's saved Full candidate hash.

Within each seed, the pre-final-head trainer-state hash, original first 20 raw replay examples/class, gate manifest and holdout manifest matched across all three capacities. The 40/class first-20 re-encoded embeddings matched exactly. For 80/class, encoding a larger class batch changed those same embeddings by at most `7.15e-7` in absolute value; first-20 holdout Class-IL/Task-ID stayed identical and NLL differed by less than `1e-6`. This small numerical difference, consistent with the changed re-encoding batch shape, is recorded in the ledger. The capacity comparisons use the external heads actually fitted with 20, 40 or 80 examples/class.

## Independent holdout results

| Seed | Capacity/class | Full-head Class-IL | Task-ID | Task-IL | Newest-task Class-IL | Old→old task errors | Persistent raw payload |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 51 | 20 | 90.87% | 91.15% | 99.13% | 92.50% | 81 | 1.22 MiB |
| 51 | 40 | 92.88% | 93.27% | 99.52% | 96.25% | 63 | 2.45 MiB |
| 51 | 80 | **94.52%** | 94.81% | 99.52% | 96.25% | **48** | 4.90 MiB |
| 52 | 20 | 90.48% | 91.25% | 98.94% | 93.75% | 83 | 1.22 MiB |
| 52 | 40 | 92.88% | 93.65% | 99.04% | 95.00% | 59 | 2.45 MiB |
| 52 | 80 | **92.98%** | 93.75% | 99.04% | **96.25%** | **58** | 4.90 MiB |

| Capacity/class | Mean Class-IL ± sample SD | Mean paired gain versus 20/class | Locked gate |
| --- | ---: | ---: | --- |
| 20 | 90.67% ± 0.27 pp | — | control |
| 40 | **92.88% ± 0.00 pp** | **+2.21 pp** | passed |
| 80 | **93.75% ± 1.09 pp** | **+3.08 pp** | passed |

Both larger capacities improved Class-IL in both seeds, exceeded the predeclared +1.00-point mean-gain floor, improved newest-task accuracy, and reduced old-to-old task errors. Doubling from 40 to 80/class adds another **0.87 point on average** but doubles raw payload again and varies more across these two seeds. The raw-payload figures count float32 replay rows only; model, prototypes, checkpoint and temporary training memory are separate resource items. More replay also adds task-time selection and final-head fitting work.

## Decision and formal follow-up

This is positive evidence that **online, task-time additional raw replay can close part of the ISOLET head gap**. For a resource-conscious method revision, **40/class is the preferred development choice**: it gains +2.21 points at 2× the old raw replay payload and is stable in these two seeds. If absolute ISOLET accuracy takes priority over memory, 80/class has the higher two-seed mean. Neither capacity should replace the existing formal result from this screen alone.

To promote 40/class or 80/class, update the production adaptive config contract, both Full/Bias candidate fits and strict checkpoint audit to consume the declared capacity; then rerun Adaptive on ISOLET, CIFAR-100 and UPMC under one fixed resource policy. For a same-budget paper comparison, rerun relevant replay baselines with matched persistent memory (or report the larger-budget result separately). Do not choose a formal method by looking at the already-seen test set.

All six streams exited 0, saved 13 event checkpoints and a frozen final adaptive checkpoint, retained exactly their declared raw replay/class, and had zero test-loader records. The NPZ loader materializes fixed test arrays during initialization but no test loader was iterated. The [aggregate ledger](data/isolet-online-head-memory-curve-20261009.json) records per-run config/checkpoint/manifest/head-state hashes, metrics, memory bytes, fit times, paired deltas and the numerical prefix bound. Ledger SHA-256: `ac73b592aa0dff1a14c7073244c2503a401ef7a807ce8df605fe8fe0b6d4cdae`; launcher SHA-256: `419dc05641a90bc4834ae9ca4f4bbb8079a8226df1226c7b72019a691cd3043f`; analysis SHA-256: `fedeac658e446c49bfa659d8167e5ab7b2d42fb82da6c74b6365a09496b99869`.

Training root: `/home/c3080/YangXiaoXiang/VF-CL/results/isolet-online-head-memory-curve-20261009-v1`. Analysis root: `/home/c3080/YangXiaoXiang/VF-CL/results/isolet-online-head-memory-readout-20261009-v1`.
