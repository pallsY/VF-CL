# Three-dataset Adaptive AA_final after the ISOLET hyperparameter selection

Date: 2026-10-09 (Asia/Shanghai). Status: completed fixed-parameter test readout for seeds 42, 43, and 44 on ISOLET, UPMC Food-101 clean image, and CIFAR-100. The selected Adaptive tuple was frozen from the earlier [ISOLET development sweep](2026-10-08-isolet-hparam-sweep.md) before these test runs:

`proto_lambda_a=0.05`, `distill_weight=0.10`, `feat_distill_weight=0.02`.

## Protocol and comparators

Each selected run was derived from the corresponding archived Adaptive seed-42/43/44 formal config. The task stream, data split, validation manifest, model, optimizer, epochs, replay budget, head procedure, deferred final-test evaluation, and seed remained fixed. Only the three named hyperparameters and output name/path changed. ISOLET ran on the 3080 host instead of its archived 4090 host, so its `data_path` and `vector_npz` also changed to byte-identical local copies. The SHA-256 data identities were checked before and after training. No baseline implementation was changed.

The runs used the existing `runner.run_experiment` formal deferred-evaluation path directly from the exact archived producer Git commits. They were kept in new result roots and were **not inserted into the old formal registry**. Every run exited with code 0, wrote `results.json`, and has `FORMAL_EVALUATION_PUBLISHED.json` binding the result-file SHA-256. The selected configs were compared with their source configs field by field, result provenance commits matched the archived records, and all three producer worktrees remained clean.

| Dataset | Exact producer commit | Execution host | Baseline comparator |
| --- | --- | --- | --- |
| ISOLET | `a575bbf446ae501cf8e580ba62c2cc5492a25f30` | 3080 | Old tuple rerun on the **same 3080 host**, seeds 42–44 |
| UPMC Food-101 clean image | `4de6fcd8874e5af9a813dc3e66a9d515bdb55b54` | 4090 | Archived formal old tuple on the same host, seeds 42–44 |
| CIFAR-100 | `7bfe6b1d724fb1206bc0053a9008126bad86332d` | 3080 | Archived formal old tuple on the same host, seeds 42–44 |

`AA_final` is the final-event class-incremental accuracy, averaged across the dataset's tasks. Standard deviations below are sample standard deviations across three seeds. The ISOLET archived 4090 baseline mean was 88.36%; it is **not** used for the paired comparison because the selected runs used a different physical GPU.

## Results

| Dataset | Seed | Old tuple AA_final | Selected tuple AA_final | Paired change |
| --- | ---: | ---: | ---: | ---: |
| ISOLET | 42 | 88.19% | 88.58% | +0.39 pp |
| ISOLET | 43 | 88.83% | 88.71% | −0.12 pp |
| ISOLET | 44 | 89.48% | 89.99% | +0.51 pp |
| UPMC clean image | 42 | 81.63% | 81.59% | −0.04 pp |
| UPMC clean image | 43 | 81.63% | 81.85% | +0.22 pp |
| UPMC clean image | 44 | 81.27% | 81.95% | +0.68 pp |
| CIFAR-100 | 42 | 34.93% | 36.35% | +1.42 pp |
| CIFAR-100 | 43 | 35.02% | 36.52% | +1.50 pp |
| CIFAR-100 | 44 | 35.09% | 36.32% | +1.23 pp |

| Dataset | Old tuple mean ± SD | Selected tuple mean ± SD | Mean change |
| --- | ---: | ---: | ---: |
| ISOLET | 88.83% ± 0.65 pp | **89.09% ± 0.78 pp** | **+0.26 pp** |
| UPMC clean image | 81.51% ± 0.21 pp | **81.80% ± 0.19 pp** | **+0.29 pp** |
| CIFAR-100 | 35.01% ± 0.08 pp | **36.40% ± 0.11 pp** | **+1.38 pp** |

The fixed tuple raises mean `AA_final` in all three datasets, but ISOLET seed 43 and UPMC seed 42 are lower than their controls. The ISOLET and UPMC mean gains are small. On CIFAR-100 all three seeds improve final accuracy, while mean BWT becomes more negative, from −31.26 to −35.46 percentage points. The selected tuple therefore does **not** improve every continual-learning metric. These are three-seed outcomes, not a significance test; the formal test results should not be used to retune the tuple.

## Audit trail

The [machine-readable ledger](data/three-dataset-selected-hparam-aa-final-20261009.json) contains each source record/config hash, selected config hash, formal publication-marker hash, bound `results.json` hash, exact `AA_final`, baseline value, allowed override list, and dataset means/standard deviations. Its SHA-256 is `ffeda7fa99e549d35ac1bd4fbd81296fd6a28e53a2cabf7216d2522949e45a3a`.

Result roots:

- 3080: `/home/c3080/YangXiaoXiang/VF-CL/results/three-dataset-selected-hparam-20261009-v1` (ISOLET selected and same-host baseline; CIFAR-100 selected).
- 4090: `/home/chase/Yangxx/VF-CL/results/three-dataset-selected-hparam-20261009-v1` (UPMC selected).

The first ISOLET-42 and UPMC-42 launch attempts stopped before Python started because the launcher pointed at a nonexistent environment path. Their `first_attempt.log` and `first_attempt_exit.code` remain in the run directories. The corrected runs completed; no result or test access came from those failed attempts.
