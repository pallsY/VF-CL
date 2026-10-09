# ISOLET prototype-feature head pilot

Date: 2026-10-09. The user approved one fixed within-budget trial after the offline 158/class head-capacity diagnosis.

## Fixed question

Can the existing 20 raw replay examples per class plus 20 temporary feature samples per class, generated at final-head fit time from the already stored `global_protos` mean/std, improve ISOLET cross-task accuracy? The three selected loss parameters `(0.05,0.10,0.02)`, 13 two-class task sequence, four views, 50 epochs/task, final-only adaptive head, and raw replay budget 20/class remain fixed. This is a development head-only screen, not a formal method or test result.

## Independent training-only protocol

Run new training seeds 49 and 50 on exact clean producer commit `a575bbf446ae501cf8e580ba62c2cc5492a25f30`, with the current herding selector. Keep the 40/class lambda-validation cohort for the Adaptive gate. Exclude a new disjoint 40/class training holdout, seed `20261011`, from both training and gate fitting. Stop after the frozen final checkpoint before deferred test evaluation. Within each seed, start both heads from the identical saved pre-consolidation classifier and encoder:

- Control: refit the existing Full head on 20 stored raw replay embeddings/class, verifying exact state-hash identity with the frozen Full candidate.
- Candidate: deterministically generate 20 transient feature vectors/class from saved class mean/std (one exact mean plus 19 diagonal-Gaussian draws), concatenate with the same 20 raw replay embeddings/class, and fit with the same Adam, 500 steps, learning rate 0.01, regularization 0.01.

Do not cache synthetic vectors, revisit unavailable old raw examples, or update the encoder. Use a fixed, derived random seed for generation. Report Class-IL, Task-ID, Task-IL, old/new Class-IL, old-to-old task errors, fit time, raw/derived memory count, and artifact hashes. The candidate passes only if both seeds improve holdout CIL, mean gain is at least +1.00 pp, and newest-task decline is no worse than −1.00 pp in each seed. No formal test access during selection.

## Scope and limitations

This first screen compares the Full branch because the existing Bias branch was much worse and the gate weighted Full at 0.96–0.99. If it passes, integrating synthetic features into the production adaptive head and strict checkpoint audit is separate work; then regression checks on CIFAR-100 and UPMC are required. A positive result is a development signal, not a formal AA_final result. Existing feature-replay literature establishes prior art; this recipe is for accuracy screening, not a novelty claim.
