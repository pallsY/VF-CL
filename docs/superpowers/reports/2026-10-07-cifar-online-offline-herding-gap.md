# CIFAR-100 Adaptive: online versus offline herding representation audit

Date: 2026-10-07. This is a read-only, training-only mechanism audit, not an accuracy experiment or a proposed new method.

## Question and fixed comparison

For each formal Adaptive seed 42, 43, and 44, the final checkpoint contains 20 raw task-time herding exemplars per class. Under that **same final bottom encoder**, compare their normalized-feature centroid error with 20 examples selected by the repository's unchanged `herding_indices` from the full 450-image-per-class eligible training pool. Both errors use the mean of all 450 eligible deterministic training-view features as the class target. The offline set is an optimistic reference: it can revisit historical training data with the final encoder.

The four party errors use each party's bottom embedding, while exemplar selection uses the aggregated embedding in both cases. The analysis reads no held-out validation or test loader and fits no classifier. It checks formal record, config, checkpoint, dataset and validation-manifest hashes; rebuilds and matches the BiC and lambda-validation manifests; confirms that the complete train-minus-holdouts pool contains exactly 450 images per class; and verifies the imported producer modules against the formal source hashes. The source tree remains at clean commit `7bfe6b1d724fb1206bc0053a9008126bad86332d`.

## Results

Lower is better. Every class has 20 online, 450 candidate, and 20 offline examples.

| Seed | Online mean | Offline mean | Paired mean gap | Online worse |
| --- | ---: | ---: | ---: | ---: |
| 42 | 0.05391 | 0.03011 | 0.02380 | 100/100 |
| 43 | 0.05661 | 0.03090 | 0.02571 | 100/100 |
| 44 | 0.05766 | 0.03105 | 0.02661 | 100/100 |
| All classes | **0.05606** | **0.03068** | **0.02537** | **300/300** |

The same comparison in individual party embeddings:

| Party | Online mean | Offline mean | Paired mean gap | Online worse |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.10395 | 0.05464 | 0.04931 | 299/300 |
| 1 | 0.09181 | 0.06447 | 0.02734 | 299/300 |
| 2 | 0.09151 | 0.06521 | 0.02630 | 298/300 |
| 3 | 0.10079 | 0.05916 | 0.04164 | 297/300 |

Per-class values and per-seed medians, party medians, paired differences, and worse fractions are in the [aggregate JSON](data/2026-10-07-cifar-online-offline-herding-gap.json). This table was independently recomputed from its 300 class rows.

## Interpretation

The existing online memory is consistently less representative of the final encoder's training-class mean than the offline reference. This is evidence of a representation gap worth investigating. It is **not** evidence that a new selector improves class-incremental accuracy, nor does it isolate the cause of the gap. In particular, saved online tensors carry the random crop/flip training transform used when they were stored; the offline candidate pool uses the deterministic evaluation transform. Different views of the images can contribute to the measured gap. The offline selector also has access to the final encoder and all historical eligible training images, which an online method does not.

The next controlled pilot should preserve image IDs and compare the task-time online exemplars under a matched deterministic view, at the same 20-per-class budget, on an independent development seed. Only after separating this view effect from the remaining gap should a party-aware online selector be implemented and compared against the repository's existing online herding, including CIL/TIL accuracy, memory, communication, and privacy cost. No formal held-out result is claimed here.

## Reproduction and integrity

The audit script is [analyze_online_offline_herding_gap.py](../../../analyze_online_offline_herding_gap.py); unit tests are [test_online_offline_herding_gap.py](../../../test_online_offline_herding_gap.py). Run the script from the **clean formal producer checkout** with that checkout and the copied script on `PYTHONPATH`, as in the archived command below. The producer commit guard deliberately rejects a different working tree.

```sh
cd /home/c3080/YangXiaoXiang/VF-CL-worktrees/head-offset-pilot-20261007
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/tmp:$PWD /home/c3080/YangXiaoXiang/envs/vfcl/bin/python \
  /tmp/analyze_online_offline_herding_gap.py \
  --root /home/c3080/YangXiaoXiang/VF-CL/results/formal-method-adaptive-20261005-v1 \
  --output-dir /home/c3080/YangXiaoXiang/VF-CL/results/cifar-online-offline-herding-gap-20261007-v2 \
  --device cuda:0
```

Two independent output roots (`v2` and `v2-repeat`) produced byte-identical JSON, SHA-256 `85712c95e7528f3ccfea37c28a92c51598bab7e2785e73c5bc071812b102995c`. The executed script SHA-256 in that JSON is `c980e3347becc75984f91009aeccfe50fac8ffa87a022d43502690e194a6f5c3`. Two local and two server unit tests passed; the server producer checkout remained clean. The JSON contains class-level errors, counts and provenance hashes, with no raw images, embeddings, or individual predictions.
