# Party Drift Telemetry Design

Date: 2026-10-07

## Question

Before changing PartyKD, test whether old-class encoder drift differs by vertical party and whether the resulting class-party signal predicts later cross-task errors. Existing published task snapshots were pruned, so the source evidence cannot reconstruct genuine teacher-to-current feature drift. Seed-42 contribution-logit proxies were weak and inconsistent across CIFAR-100, ISOLET, and UPMC. This telemetry is diagnostic, not a new classifier or a successful method claim.

## Measurement

At the end of each task after task 0, use the frozen previous-task bottom models already held by ProtoEvolve and the current bottom models. For every retained old class, pass the same bounded training replay samples through each corresponding old/current party model. Record the mean `1 - cosine(old_embedding, current_embedding)` per party, replay count, and the already frozen class-party contribution weights. Read no validation or test examples in this step. Save no individual embeddings or raw samples beyond the replay already required by the frozen method.

The measurement is enabled only by an explicit recorded option, defaults off, and cannot change parameters, gradients, losses, the final head, replay selection, or the formal baseline. One JSON record per task boundary is written atomically and must be identical on deterministic replay/resume. An unexpected/missing teacher, class, party, replay tensor, or non-finite value is a hard diagnostic failure when enabled.

## Analysis boundary

A separate read-only report may join these training-only drift summaries to per-class cross-task error on the frozen training-validation split. Compare a contribution-weighted drift summary with an unweighted summary and a shuffled-party control. Correlation is descriptive; class age and task identity must be checked as confounders. Do not fit or select PartyKD weights using final test results. If drift is not consistently predictive, stop before adding a new training mode.

## Verification

Test exact party/class coverage, unchanged model state, zero drift for identical models, positive drift for a controlled rotation, finite output, bounded storage, idempotent records, and no effect when disabled. Run a two-task smoke before any full diagnostic run. New runs use separate roots and source commits; old formal runs and their pruned evidence remain read-only.
