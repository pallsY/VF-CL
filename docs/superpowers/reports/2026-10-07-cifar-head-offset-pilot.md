# CIFAR-100 Frozen-Encoder Scalar Head Offset Pilot

Date: 2026-10-07. Status: preregistered one-seed development screen **failed**. No new formal test access or PartyKD change.

## Protocol and source

The original design was committed at `f8487fa00519b35f1a1b2fbf543188d4d7a34208`. A pre-evaluation amendment committed at `52cfaa3` clarified that the scalar-fit sample must exclude both the fixed BiC holdout and the lambda-validation holdout; the exact objective, `[-10, 10]` bound, 20-per-class count, and decision rule did not change. Its commit time was 13:01:51 +08:00, while training was still running; the training completion marker was written at 15:06:45 and the first analysis result at 15:08:10 +08:00. Thus the amendment preceded any offset fit or validation outcome inspection.

The new seed-45 CIFAR-100 run used the exact audited Adaptive producer source commit `7bfe6b1d724fb1206bc0053a9008126bad86332d` in `/home/c3080/YangXiaoXiang/VF-CL-worktrees/head-offset-pilot-20261007`. It derived the seed-42 Adaptive config (SHA-256 `5ef9984e091b83d7f84ac373cd702811c0ed15595dc6f4075ca23cc515535955`) with only the recorded changes: seed `45`, validation split seed `20261007`, normal validation-only evaluation rather than formal deferred evaluation, no final head consolidation, its disabled head mode label `full_classifier`, and a new root/name. The masked physical GPU was GPU 1. All 10 CL events completed. The source formal result root remained read-only.

Training root: `/home/c3080/YangXiaoXiang/VF-CL/results/cifar-head-offset-pilot-seed45-20261007-v1/seed_45_baseline`. Its `PILOT_TRAINING_COMPLETE.json` binds config SHA-256 `c85882da57cf1059a604cba4b5142c79b8acf2eab1198b8b716dc13230873581`, final checkpoint SHA-256 `be73d584b067487ef58adca3ac15ce6e4de0bf17905c35a36998f189df958462`, and results SHA-256 `dc0130b2d24cef022402b70f6e60514681aa1db2497bc8b17781ce4f08ac900f`. The executed launcher SHA-256 was `424e62b10e7b03b8eae4e805d30045b6ff7aeb203eb01e281c4cbd6c78ee8a40` (GitHub implementation commit `8c1b90b`; later revisions only strengthened preflight checks). The data-flow audit contains training, one validation-first-access, one BiC-calibration-first-access, and **zero test entries**. The unchanged runner's private root contains its ordinary per-example validation `final_probs.npz`; it is not published with this aggregate report.

The offline analyzer SHA-256 was `346b3b95a5abe2f5d7c749bf2b4780b1fbbebd5a5ce7843ff43cab684b9682f9`. It rehashed the original CIFAR payload files, compared both saved holdout manifests to rebuilt versions, selected exactly 20 nonholdout training images per class (2,000 total; selected-index SHA-256 `aa142091c03fc2775288ccda5958cfbb3f87727343805f57403a08a9547e5048`), froze every model parameter, and checked the trainer-state hash before and after inference. It optimized only one shared task-9 logit offset on the training-only subset, then evaluated it once on the 2,500-example lambda-validation split. Task-IL predictions were identical before/after.

Aggregate output: `/home/c3080/YangXiaoXiang/VF-CL/results/cifar-head-offset-pilot-seed45-20261007-analysis-v1/head_offset.json`, SHA-256 `b8a4eab2a7d6bcc7225c77ee442daad9a855a565263a969ed4aa2f3ef5c8e2a7`. The same aggregate JSON is checked in at `docs/superpowers/reports/data/cifar-head-offset-pilot-seed45-20261007.json`. A separate repeat evaluation produced the same SHA-256. No image, embedding, or per-example logit is in this JSON.

## Prespecified comparison

The bounded convex fit found an interior offset `delta = -2.86316` for all task-9 class logits. Its training-only calibration NLL was `2.42636`.

| Validation readout | CIL accuracy | Old-class accuracy | Newest-task accuracy | Old→new rate | New→old rate | Task-IL accuracy | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unmodified final head | 15.76% | 8.76% | 78.80% | 74.58% | 6.00% | 74.20% | 3.3352 |
| One scalar offset | 26.44% | 25.20% | 37.60% | 9.73% | 60.80% | 74.20% | 2.8248 |

The offset gained 10.68 points overall and 16.44 points on old classes, while losing 41.20 points on the newest task. The predeclared gate required at least +1 point overall, at least +15 points on the newest task, no more than -1 point on old classes, and unchanged Task-IL. It **failed** the newest-task guard. The ordinary saved BiC calibration from this same training run is a contextual control: 33.28% overall, 33.20% old, and 34.00% newest-task accuracy on the same validation split. It is not part of the scalar-offset gate and uses its separate fixed calibration holdout.

Old→new and new→old rates use all samples in their respective true-label groups as denominators (2,250 old and 250 newest-task validation examples).

## Decision and limits

Stop the one-scalar old/new offset direction. Do not retune `delta`, its bound, sample count, objective, or gate on this observed validation split; do not automatically launch seeds 46/47 or formal test evaluation. The raw no-consolidation head strongly favors new classes, whereas the earlier Adaptive Mixed head showed the opposite directional bias under different seeds/splits and a different final readout. That contrast is descriptive, not a paired causal head ablation. The one-scalar correction moves the tradeoff but does not protect both old and newest-task accuracy. A later method must be compared with the already available BiC control and tested under a fresh independent protocol.
