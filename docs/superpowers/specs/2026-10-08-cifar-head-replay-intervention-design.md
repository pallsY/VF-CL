# CIFAR-100 final-head replay intervention

Date: 2026-10-08. The user approved an equal-memory actual replay trial after the seed-49 shadow-selection result. This is one fresh development seed for a controlled head-consolidation intervention, not a formal benchmark or a participant-aware method claim.

## Question and design choice

Seed 49 showed task-time deterministic-view herding selected images with lower final-encoder centroid error than current stochastic-view herding, but the deterministic set was never used by the classifier. In the audited Adaptive config, head consolidation is scheduled **only at the final task**; saved `head_raw_replay` is not used to train earlier bottom encoders. The final checkpoint retains the exact pre-consolidation head and frozen validation embeddings. Thus the smallest actual replay intervention is to fit the real Full/Bias head candidates from three 20/class replay conditions on one frozen final model.

Alternatives considered: (1) run three complete ten-task training streams, which duplicates unchanged bottom training and introduces between-run variation; (2) fit a simplified fresh classifier, which would not test the production Adaptive head; (3) **replay intervention at the production final-head boundary (chosen)**, using its original candidate fitter, gate solver and installed top architecture. The choice is valid only for this final-schedule configuration and does not measure effects that replay might have in a different training schedule.

## Fixed arms and data boundary

Train fresh seed 50 from the audited seed-42 CIFAR Adaptive config and exact clean producer commit `7bfe6b1d724fb1206bc0053a9008126bad86332d`. Change only seed, `bic_split_seed` to `20261011`, non-formal deferred-evaluation flag, output paths and experiment name. Keep model, ten-task stream, 50 epochs/task, 20/class memory, lambda-validation split seed and all head fit/gate hyperparameters unchanged. The fresh 25/class BiC holdout is excluded from training and lambda validation. The external launcher stops after final checkpoint freeze, before deferred evaluation, and must show **zero calibration and test-loader access**.

Use the final checkpoint's `adaptive_audit_bundle.pre_state` as the identical starting top for every arm, the same saved global prototypes, task classes, final four bottom encoders and 2,000 replay examples. Fit candidates with the repository's `fit_adaptive_candidates`; calculate the gate from the same saved lambda-validation embeddings/labels using the source's canonical CPU sequence; install with `install_and_reload_verify`.

- **A, original:** the existing stochastic-selected images in their saved stochastic crop/flip views. The refit A candidate hashes, gate and installed head must exactly reproduce the frozen seed-50 checkpoint, or the experiment stops.
- **B, storage-view control:** the exact same original-image content selected in A, each stored/evaluated as its deterministic CIFAR training view. Recover IDs from the saved tensors with exact crop/flip pixel matching; duplicate indices with identical original pixels use a fixed canonical eligible ID.
- **C, selection-view intervention:** the 20/class original IDs chosen by deterministic-view herding using each class's task-boundary encoder and eligible training pool, stored/evaluated as deterministic views. This is the shadow D selection from the previous study, now actually consumed by production head candidate fitting.

All arms use 20/class raw examples. A→B isolates the replay storage-view change for fixed selected image content; B→C isolates the selected image-content change under a common deterministic storage view. The bottom encoders and pre-head remain identical. The selected IDs are computed only from task-time eligible training images, with no future encoder or evaluation labels.

## Readout and interpretation

After all three heads and gates are frozen, evaluate them once on the **fresh BiC holdout** (25/class, 2,500 images) using its deterministic view. This cohort must not be used for candidate fitting, gate selection, model choice or hyperparameter changes. Report raw uncalibrated class-incremental accuracy, task-incremental accuracy using the true 10-class task, old-class (0–89) and newest-task (90–99) class-incremental accuracy, and 100-way NLL. Use identical cached bottom embeddings and ordered labels for all arms. Save aggregate and per-task/per-class correct counts, gate and candidate hashes, sample counts and provenance; no raw images, embeddings, logits or individual predictions. Do not call the final CIFAR test loader.

The primary descriptive contrast is C−B CIL with old/new safeguards; A→B shows the storage effect, and C−A the combined change. A positive result remains a one-seed development readout, not evidence for ISOLET/UPMC or paper novelty. Only after this gate should the selector be integrated into a general online method and repeated across seeds/datasets with communication/privacy accounting.

## Integrity gates

Pin the formal source record/config/data hashes and exact override set before training. Verify ten event checkpoint hashes, final checkpoint, both holdout manifests and no calibration/test access. Reconstruct A/B/C IDs from eligible train-only candidates; require exact A baseline reproduction. Run the frozen intervention/readout twice in separate new roots and compare JSON SHA-256. Commit code, tests, report and aggregate data to the GitHub branch.
