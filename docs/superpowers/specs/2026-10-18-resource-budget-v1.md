# VF-CL resource-constrained adaptive development budget

## Scope

This is a new **development target**, not a revision of the audited Replay11, original adaptive, or sample-only formal results. For the primary pairwise resource claim, use the strongest original replay baseline by three-seed AA_final_CIL mean on each dataset: CIFAR-100 ER-ACE, ISOLET iCaRL, UPMC Food-101 DER++. Keep the other eight baselines in the broader protocol table. AdaGauss has no original exemplars and cannot share a literal exemplar-count budget.

## Frozen ceilings before accuracy trials

| Dataset | Comparator | Maximum measured persistent learning tensor bytes | Maximum measured model-training supported-op FLOP proxy | Runtime extra-data access |
| --- | --- | ---: | ---: | --- |
| CIFAR-100 | ER-ACE | 203,654,374 | 14,371,809,381,580,800 | At most its 2,500 calibration examples; a candidate with zero extra access is preferable |
| ISOLET | iCaRL | 3,715,420 | 1,208,549,376,000 | Zero additional validation examples |
| UPMC Food-101 | DER++ | 15,555,306 | 7,093,066,137,600 | Zero additional validation examples |

These are maxima across all three audited seeds for tensor payloads and a pinned seed-42 operator proxy for training steps. The operator proxy covers only operations supported by PyTorch `FlopCounterMode`; final-head fitting must be added, and validation encoding and uncounted operations remain separately disclosed. The ceiling is a necessary gate, not proof of total FLOP equivalence by itself. Check model tensor payload plus raw/logit/feature/teacher/auxiliary method state, excluding sealed audit-only evidence. Measure transient teacher copies and optimizer/CUDA memory separately; if those cannot be bounded below a comparator, make no strict full-memory claim.

The first development candidate will use at most **18 original examples per seen class** as persistent replay memory, zero stored prototypes, and no full frozen teacher copy during training. Its formal run must make zero extra validation/calibration data accesses outside the train-derived replay buffer. The final head may retain both branches and its fixed optimization schedule, but its measured training-plus-head proxy must remain below the comparator ceiling. The current task training epochs may not exceed the published 50/50/20 schedule; choose any reduction from profiling and development data before confirmation. These are upper bounds, not promises of accuracy.

## Development and selection

Develop only on four-party training/validation data with random seeds 54 and 55. The class/task sequence is fixed by the shared protocol. The old two-party UPMC development outputs are not reusable. Candidate selection uses validation AA/BWT and resource gates, with ISOLET and UPMC improvement as the main objective and CIFAR non-regression as a guard. Record every attempt, including failures, and never select a change from the already observed 42/43/44 formal test scores. Keep the number of candidate families small: teacher-free training and a predeclared training-only head gate first; head compression only if needed by the byte cap.

If a candidate survives development, freeze source, effective training budget, head rule, data hashes, resource limits, and audit checks before a new formal experiment. Prefer predeclared random seeds 45/46/47 and rerun the three selected baselines alongside the candidate on the same fixed class/task sequence. Preserve all results even if a baseline wins. Because earlier work has used the benchmark test images, describe this as a new-seed confirmation rather than a fully untouched external test set.

## Current evidence and limits

The [three-seed resource audit](../reports/2026-10-18-resource-audit-interim.md) shows the sample-only method has more persistent tensor bytes than ER-ACE on CIFAR-100, and uses an external validation set that iCaRL and DER++ did not use on the two vector datasets. The current candidate also has a live frozen teacher copy not present in its checkpoint state. The development target above is intentionally stricter; it cannot be called resource-matched until new artifacts independently verify every ceiling and access gate.
