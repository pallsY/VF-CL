# CIFAR-100 final-head replay intervention

Date: 2026-10-08 (Asia/Shanghai). Status: **one-seed development readout** on a fresh, fixed BiC-reserved holdout. This is not a formal CIFAR test result or evidence for a participant-aware method.

## Locked model and evaluation boundary

Fresh seed 50 used the clean audited Adaptive producer commit `7bfe6b1d724fb1206bc0053a9008126bad86332d`, formal seed-42 config SHA-256 `5ef9984e091b83d7f84ac373cd702811c0ed15595dc6f4075ca23cc515535955`, and source record SHA-256 `aad02da1ba8e87f29e79f5702e71fe8e99c8ef4cdf554976cc1b881af6dc0871`. The only configuration changes were seed, fresh `bic_split_seed=20261011`, non-formal deferred-evaluation flag, output paths and experiment name. The ten-task stream, model, final-only head schedule, 50 epochs/task, lambda-validation split seed and 20 replay images/class were held fixed. The [launcher](../../../launch_cifar_head_replay_intervention.py) stopped after final checkpoint freeze, before deferred evaluation. Its data-flow log has one lambda-validation access and **zero calibration or test-loader accesses**. The fresh BiC and lambda manifests each contain 25 images/class and do not overlap.

In this specific final-only schedule, the raw head replay is collected at task boundaries but is not used to train earlier bottom encoders. The final checkpoint retains the exact pre-consolidation head, final bottom models, prototypes, replay and lambda-validation embedding cache. The [intervention analyzer](../../../analyze_cifar_head_replay_intervention.py) fitted all arms through the repository's unchanged `fit_adaptive_candidates`, class-balanced gate solver on the same frozen lambda cache, and `install_and_reload_verify`:

- **A — original:** original stochastic-selected images in their saved random crop/flip views.
- **B — storage control:** the **same original-image content** as A, stored in deterministic views.
- **C — selection intervention:** 20/class images selected by deterministic-view herding with each task's own encoder and eligible training pool, stored in deterministic views.

All arms used the same final four bottom encoders, pre-head, 100 class prototypes and 2,000 replay examples. The analyzer recovered A's source images by exact pixel matching, reconstructed C from the ten task-boundary checkpoints, and excluded both holdouts from selection. Two A replay tensors mapped to eligible byte-identical duplicate originals; fixed canonical IDs represented the identical image content. **Before any BiC readout**, the refitted A pre-head, Full candidate, Bias candidate, gate and installed head each matched the frozen production evidence exactly. This verifies that the head-level intervention reproduces the actual final consolidation path for A.

Only after A/B/C heads and gates were frozen did the analyzer embed the **new BiC-reserved holdout** once (2,500 images, 25/class) and evaluate all three heads on the same cached embeddings. No BiC calibration was fitted. Reported CIL/TIL are raw adaptive-head readouts; TIL uses the true 10-class task at evaluation. Old classes are 0–89 (2,250 images), newest task is 90–99 (250 images). The final CIFAR test loader was never used.

## Readout

| Arm | CIL | TIL | Old-class CIL | Newest-task CIL | 100-way NLL | Gate `g` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A: original | 34.40% (860/2500) | 74.04% (1851/2500) | 34.71% (781/2250) | 31.60% (79/250) | 2.7039 | 0.31556 |
| B: same IDs, deterministic storage | 34.88% (872/2500) | 73.44% (1836/2500) | 34.84% (784/2250) | 35.20% (88/250) | 2.6288 | 0.30356 |
| C: deterministic selection and storage | **36.28% (907/2500)** | **74.72% (1868/2500)** | **36.44% (820/2250)** | 34.80% (87/250) | **2.6251** | 0.35270 |

| Contrast | Δ CIL | Δ TIL | Δ old CIL | Δ newest CIL | Δ NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| B−A: storage view | +0.48 pp | −0.60 pp | +0.13 pp | +3.60 pp | −0.0751 |
| C−B: selected content, same storage view | **+1.40 pp** | **+1.28 pp** | **+1.60 pp** | −0.40 pp | −0.0037 |
| C−A: combined change | +1.88 pp | +0.68 pp | +1.73 pp | +3.20 pp | −0.0788 |

The C−B CIL gain is **35 additional correct predictions** out of 2,500: 36 more old-class predictions and one fewer newest-task prediction. Across ten 250-image task groups, C improves CIL correct counts over B in six groups, ties in two and falls in two. The [aggregate JSON](data/cifar-head-replay-intervention-seed50-20261008.json) includes per-class and per-task correct/total counts and NLL sums; the table and contrasts were independently recalculated from those counts. It contains no raw images, selected IDs, embeddings, logits or individual predictions.

## Interpretation and decision

For this one fresh development seed, deterministic task-time selection improves the **actual final Adaptive head** over the same-image deterministic-storage control in CIL and TIL, while newest-task CIL falls by 0.40 points. The A→B comparison also shows that the storage view matters, especially for the newest task. The result supports a further equal-memory online-method trial, but the C−B gain is modest, the evaluation cohort has only 25 images/class, and this head-only intervention applies to the current final-only replay schedule. It does **not** establish formal test accuracy, statistical reliability across seeds, ISOLET/UPMC behavior, or participant-aware novelty.

The next gate should repeat this predeclared A/B/C protocol on additional fresh CIFAR seeds before promoting deterministic selection into the main method. If the old/new and TIL pattern persists, integrate the selector in training code and measure its real memory/communication/privacy cost, then run ISOLET and UPMC regression checks. Do not add participant-aware scoring on the basis of this single seed.

## Integrity

Training root: `/home/c3080/YangXiaoXiang/VF-CL/results/cifar-head-replay-intervention-seed50-20261008-v1/seed_50_training_only`. Final checkpoint SHA-256 `4fa81a2540619cded12cd8dd36ef21a1498049716ccc4b5627d7577f39807ee5`; training-only completion manifest SHA-256 `f23a4ff457ef2751816c634715dd7f440e6180f8b69e5a82e9050682601a9205`. The analyzer used the masked physical GPU 1 and the producer's deterministic settings; without those settings, re-embedded A features did not match the original audit, so the analyzer fails closed on device/environment mismatch.

The two independent analysis output roots (`cifar-head-replay-intervention-seed50-20261008-analysis-v1` and `-v1-repeat`) produced byte-identical JSON, SHA-256 `b7e1249ad43d6eb0c1f3276e18f75a05356732bdb8618a220db738b879efa922`. Executed analyzer SHA-256 `c722e979c88f64ce599b600c0cdc4489f53cfdbd1f8814252f2063a5d49d9ca1`. Four focused tests passed locally and on the server; the producer checkout remained clean.
