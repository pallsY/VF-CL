# VF-CL resource-matched comparison design

## Objective and evidence boundary

Determine which comparisons can support a claim that the adaptive method has no resource advantage, then optimize ISOLET and four-party UPMC within a frozen budget. Preserve the completed Replay11, original adaptive, and sample-only experiments and their audit chains. Existing 42/43/44 test results are context, not a tuning signal.

The resource study has three labels: same task/data/deferred-test schedule; same original-exemplar budget; and resource-matched. The last label requires evidence for storage, training-time model copies, compute, and actual validation access. No single universal memory equivalence is asserted across methods such as AdaGauss, iCaRL, and ER. In the frozen records, adaptive consumes a validation set on ISOLET and UPMC while ER-ACE and DER++ respectively record only final test access; CIFAR baseline calibration and adaptive validation occur at different phases. These are explicit comparison conditions, not presumed equal from matching config flags.

## Phase A: read-only resource ledger

Read only the already audited final checkpoints, task snapshots, configs, job plans, and logs. First establish all 13 methods (11 baselines plus original and sample-only adaptive) on seed 42 across CIFAR-100, ISOLET, and UPMC. Extend to seeds 43/44 to establish maxima and variation. Verify the frozen commits, data/cache identities, 4-party layout, and original final gates before recording values.

Classify tensors by algorithmic role rather than checkpoint file size: current model parameters; raw exemplars; stored logits or feature statistics; stored teacher weights; masks/adapters/other method state; and audit-only bundles. Count underlying tensor payload bytes and raw example counts. Avoid double-counting references. Keep training-time teacher copies separate from persistent post-task state when they are not serialized. Report both final and maximum task-boundary state. Checkpoint audit bundles and RNG/tracker data are evidence overhead, not learning memory.

Training compute cannot be inferred from wall time. Record the exact training and final-head step schedules from the frozen commands and source. Use a separate, explicitly labelled profiler or instrumented development run for forward/backward operation estimates; do not alter historical results or call wall-clock seconds FLOPs. Record all validation access and final-head optimization steps.

## Budget and comparisons

After the ledger is independently checked, freeze a comparator-specific budget before any model optimization. For each primary replay baseline, the adaptive candidate must use no more persistent learning bytes, training-time model memory, permitted validation data, and measured training compute. Preserve each published baseline as-is. Methods with a different memory type or no original exemplars remain in the broader protocol comparison and are not called strictly resource-matched without separate evidence.

If no feasible common cap exists, report a resource-performance frontier or a narrower pairwise comparison. Do not weaken a baseline to make the adaptive method look stronger. The exact comparator subset and ceilings are selected from the ledger and committed before development trials.

## Development and confirmation

Only after fixing the cap, use four-party development data and held-out validation for a bounded search aimed at ISOLET and UPMC while monitoring CIFAR non-regression. Record every attempted variant and selection rule. Do not use the completed formal test readouts of seeds 42/43/44 to choose hyperparameters. Freeze the selected source and protocol, run complete-task preflights and independent numeric checks, then conduct one new audited confirmation study. Prefer new, predeclared random seeds with the selected baselines rerun alongside the candidate. The class/task sequence stays fixed, and the underlying benchmark test images have been used in prior work; new random seeds are not an untouched test set.

Every report separates the audited original results, the sample-only bundled ablation, and any new resource-matched study. No accuracy advantage is attributed to a single mechanism when multiple mechanisms changed.
