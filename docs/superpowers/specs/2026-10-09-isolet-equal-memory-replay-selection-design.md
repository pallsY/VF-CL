# ISOLET equal-memory head replay selection pilot

Date: 2026-10-09. Approved direction: compare the current per-class 20-example normalized-feature herding with a fixed 10-herding/10-diversity selector. This is a development experiment, not a formal test-set run.

## Objective and fixed protocol

Use the exact ISOLET producer commit `a575bbf446ae501cf8e580ba62c2cc5492a25f30` on the 3080 host. Keep the selected training tuple `(proto_lambda_a, distill_weight, feat_distill_weight)=(0.05,0.10,0.02)`, 13 two-class tasks, four views, original task order, batch size 128, AdamW, 50 epochs/task, final-only adaptive dual-branch head, and 20 stored raw examples per class. Run new seeds 47 and 48 for both selectors, one result directory per run. No model/source-code changes in the producer worktree; an external pilot launcher supplies the dataset and selector changes and records its own SHA-256.

## Independent development readout

Keep the existing 40/class lambda-validation cohort for the adaptive gate. Build a deterministic, disjoint 40/class pilot holdout from the remaining ISOLET training indices using split seed `20261010`; exclude it from every pilot train loader. Both selectors use the exact same gate cohort and pilot holdout within a seed. Score the frozen final head on this holdout after stopping before deferred test evaluation. The formal dataset split remains unchanged for later production runs; these four pilot streams use a nested training-only holdout. Verify validation/holdout disjointness and zero test-loader iterations.

## Selector and decision

Control: existing `herding_indices(embeddings, 20)`. Candidate: keep the first 10 indices from that same herding function; select 10 further indices by deterministic farthest-first coverage in L2-normalized feature space, maximizing each remaining point's minimum cosine distance to the already selected set. This is task-time selection from the currently available class only. The same 20 raw examples per class are stored and re-encoded by the final head.

Primary: final class-incremental accuracy on the independent pilot holdout. Record task-ID, Task-IL, old/new Class-IL, old-to-old task errors, validation manifest and checkpoint hashes, gate, and memory count. The candidate passes only if CIL is higher for both seeds, the mean paired gain is at least 1.00 percentage point, and newest-task CIL is no more than 1.00 point below control in either seed. Otherwise keep current herding and investigate representation/inter-task learning. Do not use the known formal test accuracy for pilot selection.
