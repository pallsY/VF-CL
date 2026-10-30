# ISOLET F0 versus iCaRL error diagnostic

Status: frozen before any sample-level predictions are generated. This is a descriptive analysis of the already-seen stage-A development holdout. It does not reopen the stopped method search or authorize training or official-test access.

## Inputs and integrity

- Use only stage-A ISOLET F0 and iCaRL checkpoints for seeds 80–84 and the same locked 30/class development holdout. The stage-A manifest SHA-256 is `de1eabbecfcd1fcb7c1c38b5d61cc36329423931e06a781d9eae9f843ed39528`.
- Before inference, verify all ten sealed job gates, checkpoint/config/readout hashes, the common holdout identity, and the 65-job `QUEUE_COMPLETE.json`. Never edit stage-A artifacts.
- Run CPU inference from final checkpoints into the new directory `/brf/users/yangxx/VFCL-results/isolet-f0-icarl-error-diagnostic-v1-20261030`. F0 uses the final model head; iCaRL uses normalized exemplar means and negative Euclidean distance, exactly as stage A. No optimizer, training step, parameter update, official test loader, or formal result access.
- Each seed/method's inferred correct count must equal its sealed `HOLDOUT_READOUT.json` correct count. All ten checks and exact sample-ID/label alignment across methods and seeds must pass before any tables are published. The expected pooled totals are F0 3561/3900 and iCaRL 3567/3900.

## Definitions fixed before inference

- ISOLET has 13 tasks, two classes per task, with the frozen `custom_tasks` mapping. “New” is the final task's two classes; “old” is the other 24. All strata use the true class unless explicitly stated.
- Four exclusive error types, in this order: predicted class in the same task (`within_task`); old true class predicted as a final-task class (`old_to_new`); old true class predicted as a class in another old task (`old_to_other_old_task`); final-task true class predicted as an old class (`new_to_old`). Only incorrect predictions enter these types.
- E-set is fixed to B,C,D,E,G,P,T,V,Z, mapped to class IDs 1,2,3,4,6,15,19,21,25. The class-group stratum uses the **true** class: E-set versus non-E-set. Within-E-set confusions may be described, but do not create another decision group.
- For any stratum and seed, paired excess `D` is the number of holdout examples where iCaRL is correct and F0 wrong, minus the number where F0 is correct and iCaRL wrong. Both directional counts and each method's total errors are reported. Zero has no direction.
- Eight predeclared diagnostic groups: old, final task; the four error types; E-set, non-E-set. Per-task and per-class tables are descriptive and are not scanned for the gate.
- Mark a **descriptive F0-deficit pattern** only if the same predeclared group has `D > 0` in all five seeds. Also show groups with `D < 0` in all five seeds as F0 advantages, without treating them as deficit mechanisms. Report every seed's counts, pooled counts, and same-ID recurrence. No claim of statistical significance or causal attribution: seeds share one holdout and the eight groups overlap. The often-quoted 1/16 single-group probability and 0.5 expected groups assume independent seed signs and are not calibrated here.
- The pattern label is not permission to train another variant. The prelocked study remains closed regardless of this analysis.

## Outputs

Write five CSVs: task-level accuracy (all 13 tasks and seeds); error-type totals and paired excess; per-class accuracy and paired excess; per-sample cross-seed consistency keyed by source index; old/new prediction bias and true proportions. The bias table gives both methods' predicted old/new proportions; F0 alone also reports mean old/new raw logits (average over examples and classes). NME distance scores are not compared to head logits. Save a JSON provenance/diagnostic summary and per-sample predictions for audit. Do not include original features.
