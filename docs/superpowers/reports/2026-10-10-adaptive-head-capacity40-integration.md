# Adaptive head 40/class integration

Date: 2026-10-10. Scope: implement a versioned production Adaptive head with 40 persistent raw training examples per retained class. This is a code and training-only audit result, **not** a formal test accuracy result.

## Contract

| Capacity | Method version | Candidate configs | Checkpoint protocol |
| --- | ---: | --- | --- |
| 20/class (default) | 1 | Historical Full/Bias dictionaries, unchanged | Historical fields, unchanged |
| 40/class | 2 | Full and Bias each record `samples_per_class: 40` | Adds `head_consolidation_samples_per_class: 40` |

Both v2 candidates fit with 40/class. The installed mixed top, method state, result history, audit bundle and run provenance carry version 2. Resume and strict freeze audit derive the expected version from capacity. The v2 audit requires exactly 40 raw and current-encoder replay embeddings for each retained class and checks their manifests and hashes. The fixed Full/Bias endpoints remain on their existing version-1 contract. Production Adaptive config accepts only 20 or 40/class.

Code review found that the first implementation did not compare the saved method fields and audit bundle in every checkpoint path. A shared v2 consistency check now binds saved method/top versions, history, validation identity, class order, gate and audit bundle to the frozen result. Tampering tests failed before this fix and passed afterward.

The selected loss tuple (`proto_lambda_a=0.05`, `distill_weight=0.10`, `feat_distill_weight=0.02`), optimizer, gate rule, task order, validation split and baseline implementations were not changed.

## Verification

- Linux regression suite covering Adaptive 20/40 candidate fitting, mixed top, method state, checkpoint resume, strict freeze audit, deferred evaluation and formal driver resource accounting: **216 tests passed** on code commit `2d441589d96b55d8cb6b9511f4a227abf52bb8ec`.
- Local Windows focused candidate/model/method/config tests: **70 passed**. POSIX-only trusted-directory audit tests were run on Linux.
- A fresh ISOLET seed-53 training-only smoke on RTX 3080 ran all 13 tasks from the real ISOLET training data using the reviewed code commit `2d441589d96b55d8cb6b9511f4a227abf52bb8ec`. Its frozen final checkpoint passed the v2 strict audit with 40 retained raw examples/class. `SMOKE_COMPLETE.json` records checkpoint SHA-256 `ff23ea4e031b2e648a51f2a40332cfed4280432bedf4809309a767543a02da14`, freeze SHA-256 `282e76661c7ceec381cb793b351d5e111ddc1f095689694f62720dcdebf92b97`, and `test_loader_iterated: false`.
- Smoke output: `/home/c3080/YangXiaoXiang/VF-CL/results/adaptive40-contract-smoke-20261010-v3`; smoke log: `/home/c3080/YangXiaoXiang/VF-CL/results/adaptive40-smoke-v3-20261010.log`.

An attempted whole formal-driver regression suite in an isolated detached worktree reported environment-dependent errors: that suite expects a named source branch, its reviewed interpreter environment variable and earlier lineage commits. The directly relevant driver resource-state class passed separately. No formal publication was attempted from this smoke.

## Interpretation and next experiment

The code now supports a version-2 40/class Adaptive head and can freeze/audit a real ISOLET run before any test access. The earlier 20/40/80 developmental memory curve motivated 40/class, but it is not a three-dataset AA_final comparison. Formal ISOLET, UPMC Food-101 and CIFAR-100 test runs, and memory-matched baseline comparisons, remain separate work. Report additional persistent replay memory and head-fitting cost with those comparisons.
