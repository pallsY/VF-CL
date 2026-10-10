# Adaptive v2 40/class: three-dataset formal AA_final

Date: 2026-10-11 (Asia/Shanghai). Nine formal test runs completed: ISOLET, UPMC Food-101 clean image and CIFAR-100, each at seeds 42/43/44. Adaptive v2 used 40 persistent raw training examples per retained class and the frozen loss tuple `proto_lambda_a=0.05`, `distill_weight=0.10`, `feat_distill_weight=0.02`.

## Locked comparison

All nine v2 runs used source commit `2d441589d96b55d8cb6b9511f4a227abf52bb8ec`. The [pre-run manifest](data/adaptive40-formal-locked-manifest-20261010.json) (SHA-256 `94e08dee0b119ae31ba4876e98cb8b5a15eba93bb0f5583cf71bf41746903dc6`) recorded every config and dataset hash before formal testing. Relative to its same-seed published selected-hyperparameter 20/class config, each v2 config changed only `head_consolidation_samples_per_class`, `results_dir`, `output_dir` and `exp_name`. Seeds, data split, task stream, model, optimizer, epochs, validation, loss tuple and deferred formal-test procedure remained fixed. The nine source/data identities were checked again after completion.

The 20/class comparators are the earlier [published selected-tuple runs](2026-10-09-three-dataset-selected-hparam-aa-final.md), not reruns from the new v2 source commit. Each dataset used the same host and GPU model as its comparator. Physical GPUs also matched for all three UPMC seeds, ISOLET seeds 42/44 and CIFAR seed 43; they differed for ISOLET seed 43 and CIFAR seeds 42/44. These hardware assignments and the different source commits limit strict attribution of the observed difference to replay capacity alone.

`AA_final` is final-event class-incremental accuracy averaged across tasks. Differences below are paired by dataset and seed, in percentage points.

| Dataset | Seed | 20/class AA_final | 40/class AA_final | Paired change |
| --- | ---: | ---: | ---: | ---: |
| ISOLET | 42 | 88.58% | 91.66% | +3.08 |
| ISOLET | 43 | 88.71% | 91.08% | +2.37 |
| ISOLET | 44 | 89.99% | 91.53% | +1.54 |
| UPMC clean image | 42 | 81.59% | 82.57% | +0.98 |
| UPMC clean image | 43 | 81.85% | 82.75% | +0.90 |
| UPMC clean image | 44 | 81.95% | 82.55% | +0.60 |
| CIFAR-100 | 42 | 36.35% | 37.83% | +1.48 |
| CIFAR-100 | 43 | 36.52% | 37.54% | +1.02 |
| CIFAR-100 | 44 | 36.32% | 37.76% | +1.44 |

Sample standard deviations below use the three seeds, with percentage-point units.

| Dataset | 20/class mean ± SD | 40/class mean ± SD | Mean paired change |
| --- | ---: | ---: | ---: |
| ISOLET | 89.09% ± 0.78 pp | **91.42% ± 0.30 pp** | **+2.33 pp** |
| UPMC clean image | 81.80% ± 0.19 pp | **82.62% ± 0.11 pp** | **+0.83 pp** |
| CIFAR-100 | 36.40% ± 0.11 pp | **37.71% ± 0.15 pp** | **+1.31 pp** |

All nine seed-paired AA_final differences are positive. With three seeds per dataset, these are descriptive results rather than a significance claim. No parameter or capacity was retuned after examining the formal results.

## Forgetting and resource cost

| Dataset | 20/class mean BWT | 40/class mean BWT | Persistent raw replay, 20 → 40/class | Observed v2 wall time, mean/run |
| --- | ---: | ---: | ---: | ---: |
| ISOLET | +79.22 pp | +81.56 pp | 1.22 → 2.45 MiB | 3 min 04 s |
| UPMC clean image | +65.93 pp | +66.78 pp | 11.84 → 23.67 MiB | 2 min 27 s |
| CIFAR-100 | −35.46 pp | −34.24 pp | 23.44 → 46.88 MiB | 3 h 34 min |

The raw replay footprint exactly doubled. The Full and Bias head fits still use 500 and 600 optimization steps, respectively, but operate on twice as many retained samples. Wall times are observations under different concurrent host workloads; they are not an isolated measure of head-fitting overhead.

## Audit trail and interpretation

Each of the nine runs has `exit.code=0`, a final checkpoint, `FORMAL_STATE_FROZEN.json`, `FORMAL_EVALUATION_PUBLISHED.json` and `results.json`. The publication record's checkpoint and result hashes matched the files. Restricted checkpoint reload verified Adaptive method/top/result version 2, the locked capacity 40, and exactly 40 raw and current-encoder replay embeddings for every class. Both producer worktrees remained clean at the locked commit; dataset hashes matched the pre-run manifest.

The [machine-readable ledger](data/adaptive40-three-dataset-aa-final-20261010.json) binds each per-seed result, source/config/checkpoint/freeze/publication hash, comparator hash, GPU assignment, memory count and observed runtime. Its SHA-256 is `af37fa5423f6e65d50baa2c80d793bc32d5c18c089f0666ef8eb11884e7a9fca`.

Result roots:

- 3080: `/home/c3080/YangXiaoXiang/VF-CL/results/adaptive40-three-dataset-formal-20261010-v1` (ISOLET and CIFAR-100).
- 4090: `/home/chase/Yangxx/VF-CL/results/adaptive40-three-dataset-formal-20261010-v1` (UPMC clean image).

This establishes a reproducible three-dataset improvement for the versioned 40/class setting over the previously published 20/class setting, at twice the raw replay memory. A same-commit 20/class rerun and 40/class fixed Full/Bias memory-matched controls remain necessary before attributing the gain specifically to the Adaptive gate rather than replay capacity or source-version differences.
