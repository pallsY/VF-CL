# Adaptive resource-lite development design

## Purpose

Explore a separate, more constrained version of the user's adaptive method against the precommitted [resource budget](https://github.com/pallsY/VF-CL/blob/codex/resource-budget-anchors/RESOURCE_BUDGET_PROTOCOL_ANCHOR.json). This branch starts from audited sample-only producer `e2a559dc4945729848225cf21fdc95451762c231`; it cannot modify or relabel that experiment. The first candidate is an ablation, not a replacement claim about the original method.

## Opt-in method changes

Add `resource_lite=1` as an explicit protocol identity. Keep the four-party backbone, current-task training, FIM, and original-example head replay. Cap persistent raw memory at 18 examples per seen class, with empty `global_protos` and `prev_protos`. Set prototype gradient, SDC, feature KD, logit KD and PartyKD to zero. Do not create or restore a full frozen previous model teacher.

Use the existing training-replay `full_classifier` final consolidation path with one 500-step head fit. This path fits on the already stored raw examples and does not request the adaptive dual-branch validation gate; no formal ISOLET/UPMC validation access is allowed. CIFAR may use only the same predeclared BiC calibration set and seed as the strongest baseline, subject to an exact manifest and access-marker comparison. If the current formal runner unexpectedly requests validation, fail closed before any formal result.

Do not silently weaken baseline runs. A failed or less accurate candidate is retained in the development ledger. The label `resource_lite` clearly discloses that removing the teacher, PartyKD and second head changes the method relative to the user's original adaptive version.

## Development without official test access

Use the existing training-only holdout pattern from commit `99e1da7`'s `run_replay_factorial_pilot.py`: a fixed, class-balanced holdout drawn from training, excluded from task training and calibration/validation selections, and a hard stop at the first formal test authorization. The harness must reject test-loader creation and require the absence of test access markers, `results.json`, and formal publication markers. Development uses seeds 54 and 55, all four UPMC parties from the clean cache, and the same class/task schedules. Old two-party UPMC development records cannot be reused.

First run formula/config tests and a two-task low-budget smoke. Then run complete-task development candidates and paired iCaRL/DER++ development references on ISOLET/UPMC before the expensive CIFAR check. Keep all trial configs and validation-only readouts, including unsuccessful trials. Admit at most four pre-recorded candidate configurations. Among configurations passing every resource gate, select the highest mean paired holdout AA gap over ISOLET and UPMC × seeds 54/55; tie-break by lower measured compute, then lower storage. Require CIFAR development holdout AA on seeds 54/55 to be no more than 2 percentage points below a paired 20-example sample-only development reference. If none passes, report a negative development result and do not enter formal confirmation. No choice may be made from the completed 42/43/44 test readouts.

## Graduation gate

The candidate must satisfy the preanchored per-dataset retained-tensor and model-operator ceilings, no serialized or live full teacher, no extra formal vector validation access, and zero prototype state at every task boundary. Profile the development candidate and its final head before freezing epochs. After development selection, create new production and independent audit commits, a prelaunch protocol anchor, full-schedule preflights and a new 45/46/47 confirmation study rerunning the selected baselines. If the budgets or accuracy target cannot be met, report the negative result; do not retrofit the old formal artifacts.
