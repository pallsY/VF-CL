# Adaptive40 fixed-head control design

Date: 2026-10-11. User-approved goal: test whether the learned Adaptive head gate adds value beyond retaining 40 raw examples/class. Compare Adaptive40 with fixed Full40 and fixed Bias40 under matched training data, loss tuple, replay capacity, seeds and formal metrics.

## Smallest compatible change

The existing `ProtoEvolveCL` fixed Full and Bias fit paths already accept `head_consolidation_samples_per_class=40`; no new classifier or optimizer is required. Full uses `head_consolidation_lr=0.01`, `head_consolidation_steps=500`; Bias uses `head_consolidation_lr=0.03`, `head_consolidation_steps=600`, class/task regularization `0.01/0.01`, task weight `1.3`. Those are the same branch-fit settings used by Adaptive40. Both keep the selected loss tuple `0.05/0.10/0.02`, task streams, splits and validation configuration.

In the formal deferred runner, `task_class_bias` currently consolidates at the last task and is then treated as an internal Adaptive head for a second final consolidation. Classify only `adaptive_dual_branch` as a late internal install. The fixed Bias head will fit once in `after_task`, as fixed Full already does. Align the formal audit's internal-method predicate with this distinction. Formal data access currently requires a validation phase for any enabled head; require that phase only for Adaptive. Fixed Full/Bias need no validation phase unless BiC calibration is separately enabled. Add `head_consolidation_samples_per_class: 40` to the checkpoint protocol for enabled fixed Full/Bias controls, while leaving every 20/class protocol unchanged. The locked config, fit audit, replay state and final checkpoint must all agree on 40/class.

## Scientific control

Create two complete fixed-head arms, Full40 and Bias40, from the three-dataset Adaptive40 locked configs. Only the head mode, branch-specific head optimizer settings, and new output names/paths may differ. Require a training-only smoke that stops before final test access, checks exactly one final fit and 40 raw/embedded examples per retained class. Lock all 18 configs before formal test use, then run seeds 42/43/44 on ISOLET, UPMC clean image and CIFAR-100 with the exact same host/physical-GPU assignment as Adaptive40; wait for those GPUs when occupied. Require `exit.code=0`, formal freeze/publication markers, bound result hashes and branch-fit audit.

The existing Adaptive40 runs used source commit `2d441589d96b55d8cb6b9511f4a227abf52bb8ec`; the control source commit will be newer to repair formal fixed-mode routing. Tests must show the Adaptive40 path is unchanged. A future same-commit Adaptive40 rerun would remove the residual source-version difference. Validation-set gate diagnostics already show ISOLET near Full, UPMC close to Full, and CIFAR with a material interior mixture; these are mechanism diagnostics, not test-set control results.

## Resource and reporting rule

Run one job per GPU at a time and check free space before every launch. Six CIFAR control outputs are expected to need about 26 GiB; the 3080 currently has about 34 GiB free. Stop a new job before Python/test access if space falls below the configured safety threshold. Never modify published Adaptive40 or selected-20 roots. Report per-seed `AA_final`, BWT, mean ± sample SD, paired differences, raw replay bytes and runtime. Do not retune the gate or loss tuple from formal results.
