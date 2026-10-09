# ISOLET equal-memory head replay selection pilot

Date: 2026-10-09 (Asia/Shanghai). Status: **development pilot failed its prespecified gate**. The current herding selector remains the method; no formal test-set result was produced by this pilot.

## Locked comparison

The [design](../specs/2026-10-09-isolet-equal-memory-replay-selection-design.md) and [plan](../plans/2026-10-09-isolet-equal-memory-replay-selection.md) were committed before outcomes. The exact clean ISOLET producer commit was `a575bbf446ae501cf8e580ba62c2cc5492a25f30`; the fixed Adaptive tuple was `(proto_lambda_a, distill_weight, feat_distill_weight)=(0.05,0.10,0.02)`. Both variants used 13 two-class tasks, four views, 50 epochs/task, the same gate validation cohort (40/class), the same final-only adaptive head, and **20 raw training examples per class**. New training seeds 47 and 48 were paired. The class and task order, optimizer and all other losses stayed fixed.

Control: the existing normalized-feature herding takes 20/class. Candidate: take the first 10 from the same herding sequence, then 10 deterministic farthest-first samples in normalized feature space, within the currently available class at task time. The external [launcher](../../../launch_isolet_replay_selection_pilot.py) patched this selector in memory; the producer source worktree remained clean.

A separate training-only holdout was built with seed `20261010`, 40/class, disjoint from the 40/class gate cohort and excluded from every pilot training loader. Each class still had at least 158 training samples after both exclusions. Both variants used identical gate/holdout manifests within each seed. Pilot runs stopped after the final adaptive checkpoint and before deferred test evaluation. The vector NPZ loader materialized its fixed test arrays at initialization, but no test loader was iterated and no test labels were used for selection or scoring.

## Independent pilot-holdout results

| Seed | Selection | CIL | Task-ID | Task-IL | Newest-task CIL | Old→old task errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 47 | 20 herding | **89.62%** | 90.00% | 99.33% | **95.00%** | **96** |
| 47 | 10 herding + 10 diversity | 88.94% | 89.13% | 99.33% | 93.75% | 104 |
| 48 | 20 herding | **89.62%** | 89.90% | 99.33% | **93.75%** | **96** |
| 48 | 10 herding + 10 diversity | 87.40% | 87.60% | 99.52% | 91.25% | 117 |

The candidate's paired CIL changes were **−0.67** and **−2.21 percentage points** (mean **−1.44 points**). Newest-task CIL fell **1.25** and **2.50 points**. Old-to-old task errors increased by **8** and **21**. Within each seed the full trainer state immediately before final head consolidation had an identical SHA-256 across both variants, so the paired differences arose after changing the stored head replay and its final head fit. The independent holdout is the same across the four runs; these are two trained seeds, not four independent datasets.

The prespecified rule required positive CIL change in both seeds, mean gain at least +1.00 point, and newest-task decline no worse than 1.00 point per seed. The candidate fails every part that depends on CIL/newest-task performance. Its Task-IL remains high, while Task-ID falls, consistent with worse cross-task discrimination on this holdout.

## Decision and audit

**Keep the existing 20/class herding.** Do not add this 10+10 selector to the method or formal comparison table. A further accuracy attempt should investigate cross-task representation learning under the same memory and compute budget, using a fresh training-only development cohort and a prespecified comparison before any additional formal test evaluation. This pilot does not establish that all diversity-based replay schemes fail. The hybrid selector adds task-time selection computation; that cost was not profiled because the candidate failed the accuracy gate.

All four runs exited 0, saved 13 event checkpoints plus the final frozen adaptive state, stored exactly 20 raw examples for each of 26 classes, and had zero test-loader records. The checkpoint, event, data-flow, config, gate-manifest, holdout-manifest and readout hashes were verified against completion markers. The compact [aggregate ledger](data/isolet-equal-memory-replay-pilot-20261009.json) records each run, paired deltas, the locked decision, and the paired pre-head trainer hashes. Ledger SHA-256: `a2861e06ada3dca4dac73c91bc9c3cf9c21ffa5b7e187d04f0970b5804c64d6a`; launcher SHA-256: `e7da50e5fc24eda648635263fe71e441c27e9eebc89c84eb4e9de51b84eed71e`.

Result root: `/home/c3080/YangXiaoXiang/VF-CL/results/isolet-equal-memory-replay-pilot-20261009-v1`. One initial shell launch failed before Python started because it pointed to a nonexistent environment path; its log and exit code were retained. The corrected seed-47 control and all other runs completed without using that failed attempt's output.
