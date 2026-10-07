# CIFAR-100 Frozen-Encoder Head Offset Pilot

Date: 2026-10-07. Status: prespecified development experiment, not a formal result.

## Motivation and question

An audited three-seed CIFAR-100 Adaptive checkpoint diagnostic found a 38.48-point validation CIL–TIL gap and a 68.8% per-sample newest-task→old-task error rate in the Mixed branch. Merely switching saved Full/Bias/Mixed branches changes CIL accuracy by at most 3.52 points. This pilot asks whether one training-only, head-only new-task logit offset can reduce the directional error without sacrificing old-class accuracy. It does not modify party weights or bottom encoders.

## New training cohort

Start from the exact audited CIFAR Adaptive producer source commit `7bfe6b1d724fb1206bc0053a9008126bad86332d` in an isolated server worktree. Derive the seed-42 saved config at `/home/c3080/YangXiaoXiang/VF-CL/results/formal-method-adaptive-20261005-v1/runs/cifar100%3Aadaptive%3A42/config.json`. Change only: `seed=45`, `lambda_validation_split_seed=20261007`, `formal_deferred_evaluation=false`, `head_consolidation_enabled=0`, `head_consolidation_mode=full_classifier` (the disabled-mode label must no longer trigger the Adaptive gate contract), a fresh output root/name, and the physical GPU choice through `CUDA_VISIBLE_DEVICES=1`. Keep all other task, bottom-model, replay, PartyKD, optimizer, and data options. The final head is the naturally trained linear classifier, with no validation-fitted Adaptive gate. Require a CL-only timeline and preserve the frozen CIFAR data hashes. Use deterministic mode and the existing training-validation selection (`25` per class), never a final-test loader. The ordinary runner's test-oriented variable names still route to validation because `lambda_validation_enabled=1`.

## One correction, fitted before evaluation

After training, freeze both bottom encoders and the entire top classifier. Select exactly 20 training examples per class from the saved training partition, excluding **both** the fixed BiC calibration holdout and the lambda-validation holdout, with a fixed per-class sorted-index rule. Check both rebuilt holdout manifests against those saved during training. Apply the deterministic CIFAR evaluation transform to the selected images. Cache the frozen model's 100 logits on this training-only calibration set. Fit one scalar `delta` in `[-10, 10]` minimizing mean 100-way cross-entropy after adding `delta` to all task-9 class logits (classes 90–99). Use bounded scalar minimization with absolute tolerance `1e-6`; if the optimum is at either bound or the optimizer fails, stop. No other parameter, threshold, or class-specific offset is fitted.

Evaluate unmodified and shifted logits once on the newly reserved validation set. Report overall, old-class, and newest-task CIL accuracy, old→new and new→old rates, Task-IL accuracy, the fitted delta, and validation NLL. Validate that Task-IL predictions are identical before/after this common task-9 shift and that no BiC or lambda-validation example entered the calibration set. The **offline analyzer** saves aggregate results only; the unchanged training runner may retain its standard per-example `final_probs.npz` for the training-validation split inside the private pilot root. Preserve it for audit and do not publish it with the aggregate report.

## Pilot decision

The scalar-offset hypothesis passes this **one-seed screen** only if validation CIL accuracy improves by at least 1.0 percentage point, newest-task accuracy improves by at least 15 points, old-class accuracy falls by at most 1.0 point, and Task-IL accuracy is unchanged. Otherwise stop this offset direction; do not adjust the bound, objective, sample count, or threshold after seeing validation outcomes. Passing does not establish a paper claim: next repeat the same locked protocol on two fresh seeds, then check ISOLET and UPMC for regressions before any formal test evaluation. A failed or inconclusive run remains preserved in its separate result root.

The published formal roots, checkpoints, and success markers stay read-only. This pilot is not entered into formal benchmark tables.

## Pre-evaluation amendment — 2026-10-07 04:58 UTC

The original committed design `f8487fa` named the lambda-validation exclusion but omitted the separate fixed BiC calibration holdout. The audited source config has `bic_enabled=1`, and the actual training loader excludes **both** holdouts. While seed-45 training was still underway, before fitting an offset or viewing its validation outcome, the calibration selection was corrected to exclude both sets and compare both saved manifests. This preserves the original requirement to fit on the training partition. The sample count (20 per class), one-scalar objective, `[-10, 10]` bound, and pass rule did not change.

The unchanged runner also writes its usual per-example `final_probs.npz` from the validation loader into the private training root. The aggregate-only rule applies to the offline analyzer's published output. Preserve the runner artifact for audit; do not include it in a public aggregate report. The runner's already fitted BiC readout may be reported as context, but it is not part of the scalar-offset pass rule.
