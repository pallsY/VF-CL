# CIFAR-100 view-matched online herding pilot

Date: 2026-10-08 (Asia/Shanghai). Status: **single-seed training-only mechanism result**. This is not a new selection method or a held-out accuracy result.

## Locked run and comparison

Seed 48 used the unmodified clean Adaptive producer at commit `7bfe6b1d724fb1206bc0053a9008126bad86332d`. The audited seed-42 config (SHA-256 `5ef9984e091b83d7f84ac373cd702811c0ed15595dc6f4075ca23cc515535955`) and formal record (SHA-256 `aad02da1ba8e87f29e79f5702e71fe8e99c8ef4cdf554976cc1b881af6dc0871`) were the locked source. Only seed, non-formal deferred-evaluation flag, output paths, and experiment name changed. The ten-task model, adaptive head, 20 replay images/class, BiC and lambda holdouts, and 50 epochs/task remained as configured; the prescribed validation split was reused. The external [launcher](../../../launch_cifar_view_matched_pilot.py) intercepted the runner immediately after it saved and audited `adaptive_final.pt`, before deferred test evaluation. All ten CIL event checkpoints exist; the data-flow log has **zero test-loader accesses**. No `results.json` or CIL/TIL accuracy was produced.

The [exact pixel matcher](../../../cifar_replay_id_match.py) inverted CIFAR normalization, searched the saved crop/flip against the 450 eligible same-class training images, and verified the whole 32×32 augmented image byte for byte. It recovered an original-image content match for every one of the 2,000 saved online replay tensors. One tensor matched two eligible indices containing **byte-identical original images**; the lower ID was used as a deterministic representative. No distinct-original ambiguity occurred. A prior formal seed-42 recovery preflight likewise matched all 2,000 replay tensors, with two byte-identical-index cases.

Under the **same final four bottom encoders**, each class's target was the normalized-feature mean of all 450 eligible deterministic training-view images. The [analyzer](../../../analyze_cifar_view_matched_pilot.py) measured L2 centroid error for:

- **A:** the 20 saved online replay tensors with their task-time random crop/flip;
- **B:** the same 20 recovered original images with the deterministic training-view transform;
- **C:** 20 optimistic offline images chosen from the full 450-image pool by the repository's unchanged `herding_indices` under the final encoder.

BiC and lambda holdouts were excluded from candidate selection and centroid targets. No head was fitted. The analyzer called no validation or test loader and saved no raw images, embeddings, or individual predictions.

## Result

Lower centroid error is better. Values are means over 100 paired classes.

| Feature space | A: augmented online | B: matched deterministic | C: offline deterministic | A−B | B−C | Classes A>B | Classes B>C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Aggregated | 0.05844 | 0.04349 | 0.03033 | 0.01495 | 0.01316 | 99/100 | 100/100 |
| Party 0 | 0.11000 | 0.06886 | 0.05425 | 0.04114 | 0.01461 | 100/100 | 91/100 |
| Party 1 | 0.09688 | 0.07832 | 0.06302 | 0.01855 | 0.01530 | 95/100 | 98/100 |
| Party 2 | 0.09587 | 0.07936 | 0.06263 | 0.01650 | 0.01674 | 92/100 | 96/100 |
| Party 3 | 0.10498 | 0.06926 | 0.05636 | 0.03572 | 0.01290 | 96/100 | 87/100 |

The aggregated A−C gap is `0.02812`; changing only the view for the exact online-selected image content accounts for `0.01495`, or **53.2% of this seed's measured gap**. A remaining B−C gap of `0.01316` persists under matched deterministic views and is positive in all 100 classes. Per-class errors, medians, paired differences and fractions are in the [aggregate JSON](data/cifar-view-matched-herding-seed48-20261008.json). The table and 53.2% calculation were independently recomputed from its 100 class rows.

## Interpretation and next decision

The stored random image view has a substantial effect on the previous online/offline centroid-error comparison. Matching the original-image content removes about half of the measured gap in this development seed, but does not eliminate it. The remaining B−C gap combines task-time selection under augmented views, encoder evolution, and the offline reference's access to the final encoder and full historical pool. It does **not** isolate feature drift or prove that a party-aware selector improves CIL accuracy. This is one development seed and should not be entered into formal benchmark tables.

The next causal comparison should keep the online 20/class budget and train a fresh development seed with **task-time deterministic-view herding versus the existing stochastic-view herding**, recording original IDs for both and evaluating both retained sets under the same deterministic view. That tests whether changing the selection view closes the remaining representation gap before adding party-aware scoring. Any proposed party-aware method then needs fresh-seed CIL/TIL evidence, an equal-memory online-herding control, communication/privacy cost, and ISOLET/UPMC regression checks.

## Integrity and reproduction

Training root: `/home/c3080/YangXiaoXiang/VF-CL/results/cifar-view-matched-herding-seed48-20261008-v1/seed_48_training_only`. Final checkpoint SHA-256 `cd23fa3b7b6e84d243bef0236a33ba31edfce878029bb748fd5d6e200cc62274`; completion manifest SHA-256 `d74b1eec2ecca4774eb3ce7eb98a90dcc103f39c8e26fd8cec82d716077046b8`. The recovered online ID-list hash is `a3786f80eb75635c5a7b6150d7d4e8ffdb85cf2aec9b77aaeb97551e5324e73a`; IDs themselves are not exported.

The analysis ran twice against the same frozen checkpoint in separate output roots (`cifar-view-matched-herding-seed48-20261008-analysis-v1` and `-v1-repeat`). Their JSON files are byte-identical, SHA-256 `fd781ea2031e431b4522d7d2815716c6f763381adea92a8e3508afd1e2e0705a`. Executed analyzer SHA-256 `c8bdd7ab5401c0d545620635541663a17762fc27ce2edd86b3cdff4f8e90a528`; matcher SHA-256 `ca28de7a92e384aa68c994f112a47dcf368091174e0ae1a337d5600ec15f40a3`. All eight focused unit tests passed locally and on the server. The producer checkout remained clean.
