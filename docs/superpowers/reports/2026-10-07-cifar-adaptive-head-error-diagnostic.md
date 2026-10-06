# CIFAR-100 Adaptive Head Error Diagnostic

Date: 2026-10-07. Read-only exploratory analysis; no model training or method change.

## Scope and provenance

The source is the audited three-seed Adaptive method shard at `/home/c3080/YangXiaoXiang/VF-CL/results/formal-method-adaptive-20261005-v1`, with `METHOD_SHARD_SUCCESS` and formal completed records for seeds 42, 43, and 44. All three records name producer commit `7bfe6b1d724fb1206bc0053a9008126bad86332d`. The analysis used each saved `formal_final.pt` model, the original 25-per-class CIFAR-100 **training-validation** manifest, and no final-test loader. Before evaluation it matched the source config, checkpoint, and validation-manifest file hashes to each completed record, rebuilt the validation manifest and compared the full contents, and checked the local model/data/branch source modules against the record's source hashes.

For each validation example, the script evaluated the saved Full and Bias branches via `TopModel.branch_log_probabilities` and the actual Mixed `TopModel.forward` output. It verified the Mixed output against the saved gate formula. Branch NLLs matched the frozen head-selection evidence within `4.44e-5` in every seed. Newest-task classes are task 9 (classes 90–99); old classes are tasks 0–8 (classes 0–89). Each seed evaluates 2,500 validation examples: 2,250 old and 250 newest-task examples. No individual example, embedding, or prediction was saved.

Script: `analyze_cifar_adaptive_head_error.py`, SHA-256 `253d34e89e728ef82bdb0e6f6352e4b398ce0aba37d824845299c3c32471fc17`. The script also enforces the formal record's hashes for its five imported model/data/branch modules and all three CIFAR source payload files. Output: `/home/c3080/YangXiaoXiang/VF-CL/results/cifar-adaptive-head-error-20261007-v2/head_error.json`, SHA-256 `3162e02f198780be3ebe9c6581217bc2b4c2026cc342d14baefc4b6051f64ab7`. The same aggregate output is checked in at `docs/superpowers/reports/data/cifar-adaptive-head-error-20261007.json`. A second run to a separate root produced a byte-identical output hash. The v2 branch metrics also match the preceding diagnostic exactly. The output contains per-seed aggregate metrics and a 10-by-10 task confusion matrix for each branch.

## Results

Three-seed means on the training-validation split:

| Final branch | CIL accuracy | Task-IL accuracy | Old-class accuracy | Newest-task accuracy | Old sample → new task | New sample → old tasks |
|---|---:|---:|---:|---:|---:|---:|
| Full | 31.77% | 70.63% | 32.00% | 29.73% | 6.98% | 68.13% |
| Bias | 34.41% | 74.01% | 35.32% | 26.27% | 6.53% | 71.87% |
| Mixed | 35.29% | 73.77% | 35.93% | 29.60% | 6.58% | 68.80% |

The percentages in the last two columns use **all samples in the corresponding true-label group** as denominators. Across three seeds, Mixed has 516 new→old and 444 old→new errors; these absolute counts are close because there are nine times as many old validation examples. The per-sample risk is strongly asymmetric: 68.8% of newest-task examples are assigned to old tasks, compared with 6.58% of old examples assigned to the newest task. Among Mixed's incorrectly classified newest-task examples, 97.7% are assigned to an old task. The rate asymmetry appears in every seed and all three branches. Mixed improves CIL accuracy by 3.52 percentage points over Full and 0.88 points over Bias on this split, yet retains a 38.48-point Task-IL versus CIL gap.

The formal test records independently report an Adaptive three-seed mean CIL accuracy of 35.01%, Task-IL accuracy of 74.15%, and BWT of -31.26 percentage points; all nine old tasks finish below their accuracy when first learned. Those published metrics establish that the CIFAR problem includes old-task degradation as well as cross-task class competition. They are not used to select a new calibration rule here.

## Interpretation and next gate

The immediate **per-sample** directional risk is newest-task samples being predicted as old classes. Switching among the already trained Full, Bias, and Mixed branches changes accuracy by a few points, but none approaches the Task-IL upper reference. This supports a targeted study of cross-task decision boundaries; it does not prove that the head alone can close the gap, because Task-IL supplies the true task identity and old-task BWT is negative.

The validation split was already used to choose Adaptive's mixture gate, so Mixed's advantage here is descriptive and may be optimistic. Do not tune an old/new offset or claim an improvement from this split. A subsequent head-only pilot should freeze the encoder, prespecify one correction and its old/new regression tolerance, and evaluate it on a newly reserved validation cohort from a fresh training run before any formal test access. If it fails to close a meaningful portion of the CIL–TIL gap, prioritize representation retention rather than another head rule.
