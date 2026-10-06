# Replay Marginal Utility Screen Design

Date: 2026-10-07. Status: exploratory mechanism screen, not a PartyKD change.

## Question

Does a party's contribution to old-class prediction, measured before a new task on existing training replay, help identify which party's changing logits matter for subsequent old-class validation loss? The previous contribution-weighted encoder cosine drift screen did not beat uniform or shuffled controls on ISOLET and UPMC.

## Available evidence and scope

Use the completed seed-42 ISOLET and UPMC v2 diagnostic checkpoints. Each task checkpoint contains the trainer state, frozen class-party weights, and up to 20 raw training-replay examples per seen class. The frozen validation split is available through `lambda_validation_enabled=1`; no final-test loader is used. Analyse adjacent checkpoints from task 0→1 through the penultimate transition, excluding the final transition because final head consolidation changes the readout. Do not retrain, update model parameters, change PartyKD, or write into source result roots.

## Compared approaches

1. **Recommended: replay marginal utility.** In the pre-transition model, decompose the linear concatenated head into party logit contributions. On class `c` replay, compute the mean cross-entropy increase when party `p`'s logit contribution is removed, keeping the classifier bias and the other parties. Restrict softmax to classes seen before the transition. Clip negative utilities to zero and normalize across parties; use uniform weights if all are zero. This is a fixed-model ablation, not a causal claim about retraining without that party.
2. **Existing contribution magnitude.** Reuse the frozen absolute-logit class-party weights already in the checkpoint. It is cheap and is the current PartyKD static candidate, but its sign and effect on prediction loss are not represented.
3. **Coalition/Shapley valuation.** Re-evaluate combinations of parties. This is more symmetric but multiplies evaluations and still depends on a chosen utility and coalition semantics. It is out of scope for a first screen.

## Score and outcome

For class `c` and party `p`, use the same pre-transition training replay to compute the mean squared change in that party's class-`c` logit contribution between adjacent checkpoints, `d[c,p]`. Compute three class scores from this same `d`: marginal-utility-weighted, uniform-weighted, and existing frozen-contribution-weighted; also include a deterministic cyclic shuffle of marginal-utility weights. The pre-transition utility weights use no validation labels. The primary outcome is the change in class-`c` validation cross-entropy from before to after the transition, with logits at both points restricted to the pre-transition seen classes. This isolates old-class discrimination from new-class competition. Report the ordinary class-incremental old-class error change as a secondary outcome only.

Require exact replay/class coverage, finite logits/losses/weights, identity of the frozen pre-transition head decomposition with its full linear logits, and a stable validation manifest. Save only aggregate class-by-boundary scores/outcomes and their counts, without raw examples, embeddings, or per-example predictions. A failed check aborts the diagnostic.

## Prespecified analysis and stop rule

Center scores and primary outcomes within each task boundary and compute pooled Spearman correlations separately by dataset. Also report the number of boundaries with a positive within-boundary rank association, the fraction of utility weights that fall back to uniform, and the score spread between weighting rules. The utility-weighted score passes this exploratory screen only if its centered correlation is positive and at least 0.10 higher than both uniform and frozen-contribution controls on **each** dataset, with more than half of eligible boundaries directionally positive. Otherwise stop before any PartyKD implementation or CIFAR-100 run. These descriptive comparisons do not establish statistical significance or a causal effect.

## Reproducibility

Record source checkpoint roots, Git commit, dataset and config hashes, exact formula, excluded final transition, summary counts, and output SHA-256 in a report. Keep the source diagnostic and formal result roots read-only. The existing telemetry branch remains available independently of this screen.
