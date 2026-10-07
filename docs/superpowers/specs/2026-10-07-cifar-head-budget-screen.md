# CIFAR-100 Frozen-Head Sample-Budget Screen

Date: 2026-10-07. Status: preregistered seed-46 development screen, not a formal result.

## Question

The preceding seed-45 fixed-encoder Full-head check fitted 20 training images per class and reduced training CE from 2.935 to 0.244, while validation NLL remained 3.114; newest-task and Task-IL accuracy declined. Does a larger, class-balanced train-only head-fit corpus improve held-out old and new class discrimination, or does performance saturate despite more examples? This diagnoses the fixed-feature head-fitting recipe; it does not prove whether representation or replay is causally limiting.

## Fresh training stream

Use the exact audited CIFAR producer source commit `7bfe6b1d724fb1206bc0053a9008126bad86332d` in its clean isolated 3080 worktree. Derive the saved seed-42 Adaptive config with the same eight deviation keys as the seed-45 pilot, but set `seed=46`, `lambda_validation_split_seed=20261008`, `formal_deferred_evaluation=false`, `head_consolidation_enabled=0`, `head_consolidation_mode=full_classifier`, and a fresh root/name. All task, bottom-training, party KD, BiC, optimizer, batch, and data options stay fixed. Run a CL-only ten-task deterministic stream on physical GPU 1. Preserve all formal and seed-45 roots. The held-out lambda-validation cohort is selected before training and used only for ordinary readouts and the later diagnosis; it is not used to fit any head.

## Locked head-fit comparison

At the completed final checkpoint, freeze all bottom models. Verify source/config/data/checkpoint/audit hashes, zero test-loader access, and exact saved BiC and lambda-validation manifests. From the training partition only, excluding **both** fixed holdouts, select the first 400 eligible images per class under deterministic CIFAR evaluation transform. Extract aggregated features once. For independent head fits, take nested first-20, first-100, and first-400 features per class, each time resetting a fresh copy of the same original classifier. Apply only the repository's existing `consolidate_classifier`: 100-way CE, regularization `0.01`, Adam learning rate `0.01`, 500 steps, equal class counts. No hyperparameter or fit budget may be added after inspecting validation results. Bottom encoders and the source checkpoint remain unchanged.

Evaluate the raw and three fitted heads once on the new 25-per-class lambda-validation split. Record old/new/overall CIL, Task-IL, NLL, old→new/new→old rates, fitter training CE before/after, and sample counts. The ordinary saved BiC readout is context, not a budget-selection target. Save aggregate metrics only; the unchanged runner's private root may contain its standard per-example validation `final_probs.npz`, which must not be published.

## Descriptive decision rule and limits

Before seeing the new validation outcomes, label **sample-sufficiency signal** only if the 400-per-class head exceeds the 20-per-class head by at least 5 CIL percentage points and neither old nor newest-task accuracy falls by more than 2 points. Label **early saturation** only if the CIL difference is below 2 points and 400-per-class validation NLL does not improve. Anything else is inconclusive. These labels are exploratory. If sample-sufficiency appears, the next independent experiment can test a memory-efficient representative replay scheme; if saturation appears, prioritize a representation-focused hypothesis. Do not choose the best budget as a method, access final test data, launch extra seeds, or enter these diagnostics into formal tables from this screen.
