# CIFAR-100 task-time selection-view herding pilot

Date: 2026-10-08. The user approved the next controlled comparison after the seed-48 matched-view report. This is a training-only mechanism experiment on fresh development seed 49, not a new method or a formal benchmark result.

## Question

Seed 48 showed that matching the stored online exemplars to deterministic views removed about 53% of the aggregate online/offline centroid-error gap, while a mean 0.01316 gap remained. Does **task-time deterministic-view herding** select original images that remain more representative under the final encoder than the existing **task-time stochastic-view herding**, when both retained sets are later evaluated as deterministic views at the same 20/class budget?

## Approaches considered

1. **Task-boundary checkpoint reconstruction (chosen).** Train the unchanged Adaptive producer once. At each event checkpoint, use only the 450 eligible images/class of the current task and that checkpoint's bottom encoder to run unchanged `herding_indices` on deterministic views. Recover the existing stochastic selector's original image content from its saved replay tensors by exact pixel matching. Evaluate both sets under the same final encoder and deterministic view. This makes selection view the varied factor while keeping training fixed.
2. Instrument the producer to save two selector outputs during each task. This is more direct but changes the audited training path and checkpoint schema.
3. Train two separate runs with the selected memories fed back into the head. That tests downstream accuracy but changes model states and costs two full runs, obscuring the first mechanism question.

## Fixed protocol

Use fresh seed 49 from the audited seed-42 formal Adaptive config and clean producer commit `7bfe6b1d724fb1206bc0053a9008126bad86332d`. Change only seed, non-formal deferred-evaluation flag, output paths and experiment name. Keep the ten-task stream, adaptive head, 20 images/class, BiC/lambda holdouts, 50 epochs/task and deterministic environment. As in the seed-48 pilot, the external launcher must stop immediately after the final adaptive checkpoint is frozen, before deferred test evaluation, and verify all ten event checkpoints and zero test-loader accesses.

The analysis uses the same 450 eligible **training** images/class after excluding both holdouts. For each task 0–9, load its event checkpoint, verify task identity and source hashes, freeze its bottom models, embed the ten current classes using the deterministic training-view transform, and select 20 IDs/class with the repository's unmodified `herding_indices`. These are shadow deterministic task-time selections; no selected images are fed back into training. Recover the existing stochastic task-time replay's 20 original-image IDs/class from the final checkpoint using exact crop/flip matching. Byte-identical duplicate images may use a fixed canonical eligible ID; matches to distinct original content must fail.

At the final frozen encoder, compare three 20/class sets against each class's full 450-image deterministic normalized-feature mean: **S** = original IDs retained by existing stochastic-view online herding; **D** = IDs selected by deterministic-view herding at the same task boundary; **O** = optimistic final-encoder offline herding. All three are embedded with the same deterministic view. Report per-class aggregate and four-party centroid errors, paired S−D and D−O differences, fractions positive, memory counts, S/D overlap, selection-ID hashes and provenance. Also report the task-boundary S/D centroid errors for a selection-time sanity check. Do not fit a head or access held-out validation/test loaders in analysis.

## Interpretation

A positive final S−D means the deterministic-view task-time selector retained a more representative set for the final encoder on this seed, with the model fixed. It is not evidence of improved CIL/TIL accuracy: D was a shadow selector and never changed training or head consolidation. Task-boundary D is favored by its own deterministic centroid objective, so selection-time gains alone are expected. D−O remains an optimistic future-information gap, not a feature-drift estimate. Report the magnitude and class/party consistency without a binary novelty claim. A later full method run should only be considered if the final-encoder result supports it, and must measure accuracy, memory, communication/privacy cost and ISOLET/UPMC regressions.

## Reproducibility gates

Lock the config and source record hashes before seed-49 launch; require a clean producer checkout, physical GPU 1 and deterministic environment. Save a training-only completion manifest with final and ten task checkpoint hashes. Analysis must validate that manifest, both holdout manifests, full eligible pool, checkpoint task identities, exact replay content matches, frozen bottom states and no protected loader access. Run analysis twice to fresh roots and require byte-identical JSON. Publish only class-level metrics and hashes, never raw images, embeddings, IDs or individual predictions.
