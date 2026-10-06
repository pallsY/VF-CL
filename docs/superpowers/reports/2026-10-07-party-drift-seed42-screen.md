# Party Drift Telemetry: Seed-42 Development Screen

Date: 2026-10-07. Status: exploratory negative screen; no PartyKD training change.

## Question and measurement

The method records the same bounded old-class training replay through the previous-task frozen bottom and the current bottom at each task boundary. For each old class and vertical party it stores the mean cosine feature distance `1 - cos(z_teacher, z_current)`, replay count, and the existing frozen class-party contribution weight. It does not persist individual embeddings, new raw samples, or full model snapshots. The opt-in flag defaults off and is restricted to CL-only streams.

The existing published task-boundary models had been pruned to zero-byte tombstones. Earlier final-checkpoint contribution-logit proxies were weak and cannot establish true encoder drift. These new development runs measured genuine in-memory teacher-to-current drift without reading test examples.

## Execution and evidence

The Linux source commit was `e0af6a96f8553dfecdc6b1e41b07ff10ee067c4c` on the isolated 4090 checkout; the same feature is on GitHub branch `codex/party-drift-telemetry` at `2ac8d0f87dc2ca674064af3a91b31098dec9749e`.

A deterministic two-task vector smoke used identical data, seed, and training options with telemetry off/on. Both CIL checkpoint trainer-state hashes and all CL result rows matched exactly. The enabled run wrote one 808-byte task-boundary JSON record. Linux focused/resume suites passed 67 tests after the review fixes.

The two real development runs derived their task streams and training options from the published Adaptive seed-42 job specs. Recorded changes: CPU device, `formal_deferred_evaluation=0`, final `full_classifier` readout to keep test access disabled, `save_task_checkpoints=1`, new output root, and `party_drift_telemetry=1`. The bottom-model training losses and all other method hyperparameters were unchanged. The final head differs from Mixed, so the resulting error correlations are a development mechanism screen, not a paired evaluation of the frozen Adaptive method.

| Dataset | Source job spec SHA-256 | Dataset SHA-256 | Diagnostic root |
|---|---|---|---|
| ISOLET | `cd0fc475f68fe2770ac8ae415e7ce2326dffca7410223c844c00702c74647e7c` | `d34312670de93198afcd2b126c95b79bae2b4cffeb30d480f097faf046b69514` | `party-drift-isolet-dev-20261007-v1` |
| UPMC clean image | `b71b0d4bf308b1ba619c86f7cd70d9af1849811aab813c26fe4865fdd66885e1` | `dea0569f0d4195eaaecc9cab3a6cce533f07b8d88472e7dbca5113152a314339` | `party-drift-upmc-clean-dev-20261007-v1` |

ISOLET has 12 task-boundary records totaling 64,178 bytes; UPMC has 9 totaling 122,721 bytes. Every record has exact retained-old class coverage, finite drift for all parties, and `validation_used=false`, `test_used=false`. The run data-flow logs contain validation access and no test split access.

## Read-only analysis rule

For each old class, average each party's drift over all later task boundaries. Compute (1) the class's frozen contribution-weighted mean, (2) the uniform party mean, and (3) a deterministic one-position shuffle of party weights. The outcome is the final training-validation per-class cross-task error under the diagnostic run's full classifier. Report Spearman correlation across old classes and after centering both drift and error within the class's originating task to reduce task-age confounding. This is a descriptive screen, not a fitted predictor or causal estimate.

| Dataset | Old classes | Score | Raw Spearman rho | Within-task centered rho |
|---|---:|---|---:|---:|
| ISOLET | 24 | Contribution-weighted | -0.128 | +0.008 |
| ISOLET | 24 | Uniform | +0.107 | +0.103 |
| ISOLET | 24 | Shuffled weights | +0.221 | +0.172 |
| UPMC clean image | 91 | Contribution-weighted | -0.038 | +0.027 |
| UPMC clean image | 91 | Uniform | -0.002 | +0.078 |
| UPMC clean image | 91 | Shuffled weights | +0.139 | +0.173 |

The contribution-weighted statistic has no consistent positive association and does not outperform either control. UPMC has two parties, so normalized contribution-share change alone is mathematically degenerate; this screen instead uses actual per-party cosine feature drift. Correlation strength should not be interpreted as a significance test because classes within tasks are dependent and the sample is small.

## Decision

Stop before the approximately 2-hour CIFAR-100 full diagnostic run. Do not implement drift-weighted PartyKD from these data. The current hypothesis needs a different training-only signal or a narrower task-specific claim; choosing another weighting rule from the already exposed final test sets would not provide confirmatory evidence. The current Mixed formal method and all published result roots remain unchanged.
