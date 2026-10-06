# Factorized Adaptive Head V1: Paired Posthoc Evaluation

Date: 2026-10-06

## Method and evidence boundary

The frozen V1 readout keeps the existing Adaptive mixture's task probability and the original pre-consolidation classifier's class probability conditional on that task. It adds no training, fitted parameter, or checkpoint field. The paired evaluator reads each published Adaptive final checkpoint, computes Mixed and Factorized predictions on the same test batches, and validates the complete frozen-to-published transaction and a clean evaluator Git checkout, then rejects a run unless its Mixed per-task and summary metrics reproduce the published result. Output paths inside the published source tree are rejected. The source runs are unchanged.

This is a paired posthoc evaluation of previously published test sets, not a fresh training run or an independent held-out result. The training-validation pilot favored CIFAR-100 Task-IL, but no method or threshold was changed after final test evaluation began.

Evaluator source on GitHub: `ec27b023e1bf63497ffbf9a57e3a604894de0851`. Native evaluator commits: CIFAR-100 `89f84fb735793db1ffb022fb0a3a6905f5ca2a97`; ISOLET and UPMC `3e53560a7b94125f1e12cf645279115e41e86037`. These contain the same factorization and numeric final-task BWT rule on their respective source checkouts.

## Results

Each cell is `Mixed → Factorized`, in percent. BWT uses the original published diagonal and the paired final row.

| Dataset | Seed | AA-final | BWT | Final Task-IL |
|---|---:|---:|---:|---:|
| CIFAR-100 | 42 | 34.93 → 35.13 | -31.06 → -31.09 | 74.23 → 73.67 |
| CIFAR-100 | 43 | 35.02 → 34.81 | -31.29 → -31.59 | 74.37 → 74.05 |
| CIFAR-100 | 44 | 35.09 → 35.44 | -31.42 → -31.31 | 73.85 → 73.98 |
| CIFAR-100 | mean | 35.01 → 35.13 | -31.26 → -31.33 | 74.15 → 73.90 |
| ISOLET | 42 | 87.68 → 87.61 | 78.25 → 78.18 | 98.39 → 98.64 |
| ISOLET | 43 | 89.09 → 89.22 | 80.06 → 80.26 | 98.78 → 98.84 |
| ISOLET | 44 | 88.32 → 88.32 | 78.81 → 78.80 | 98.97 → 98.78 |
| ISOLET | mean | 88.36 → 88.38 | 79.04 → 79.08 | 98.71 → 98.75 |
| UPMC Food-101 clean image | 42 | 81.63 → 81.40 | 67.39 → 67.22 | 93.03 → 92.50 |
| UPMC Food-101 clean image | 43 | 81.63 → 81.26 | 67.60 → 67.35 | 93.07 → 92.61 |
| UPMC Food-101 clean image | 44 | 81.27 → 81.10 | 67.11 → 67.00 | 92.90 → 92.58 |
| UPMC Food-101 clean image | mean | 81.51 → 81.25 | 67.37 → 67.19 | 93.00 → 92.56 |

The mean paired changes in percentage points are CIFAR-100 `(+0.11, -0.07, -0.25)`, ISOLET `(+0.02, +0.04, +0.04)`, and UPMC `(-0.26, -0.18, -0.44)` for AA-final, BWT, and Task-IL respectively. UPMC declines on all three metrics at every seed. The CIFAR-100 Task-IL improvement observed on training-validation does not generalize to these final test readouts.

## Source and result roots

- CIFAR-100 source: `formal-method-adaptive-20261005-v1`; authoritative paired output: `factorized-posthoc-cifar-20261006-v3`.
- ISOLET source: `formal-isolet-local-4090-1-20260924-v4`; authoritative paired output: `factorized-posthoc-isolet-20261006-v3`.
- UPMC clean image source: `formal-upmc-image-clean-4090-1-20260925-v1`; authoritative paired output: `factorized-posthoc-upmc-clean-20261006-v2`.

Each output root contains exactly three per-seed `comparison.json` records with published source checkpoint/result hashes, test-source hash, evaluator commit, per-task counts, and both readout metrics. Earlier V1/V2 paired output roots are preserved and excluded from this summary. CIFAR V3, ISOLET V3, and UPMC V2 were regenerated under the strengthened evaluator; all nine metric triples exactly match the earlier successful paired calculations. ISOLET V1 stopped at a numeric task-order BWT check before writing a comparison record.

## Decision

Keep the existing Mixed head as the primary method. Factorized V1 is an informative ablation, not a replacement: it does not recover CIFAR-100 final Task-IL on average and consistently lowers UPMC performance. No further parameter or rule adjustment should be selected from these test results. A new head design needs an independent development/held-out sequence.
