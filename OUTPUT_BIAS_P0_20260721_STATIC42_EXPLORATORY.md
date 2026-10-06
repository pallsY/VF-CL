# P0 Output-Bias Diagnosis — Static Seed 42 (Exploratory)

## Status and scope

This is an exploratory, single-training-seed analysis. It is not the planned
multi-seed P0 gate and must not be used for a confirmatory claim, a formal
method comparison, or a deployable calibration result.

Evidence source:

~~~
/home/c3080/YangXiaoXiang/VF-CL/results/deterministic_formal_20260720_123945_3557884/ten_task/ten_task_a_20260720_140826
~~~

This Static seed 42 run is a complete-checkpoint deterministic repeat. The
formal Static/Uniform pilot runs only retain party-weight audit files and
therefore cannot be replayed for party logits.

## Reproducibility checks

- Stored AA_final: 0.1498
- Replayed AA_final: 0.1498
- Saved-probability AA_final: 0.1498
- Saved and replayed predictions: identical for all 10,000 test examples.
- Sum decomposition maximum absolute error: 5.344e-6.
- Allowed FP32 reassociation tolerance: 1e-5.

The analysis uses the exact sum-aggregation decomposition:

~~~
full_logits = sum_p F.linear(embedding_p, classifier_weight, None)
              + classifier_bias
~~~

The nonzero reconstruction error is FP32 reassociation only: direct
F.linear(sum_p embedding_p, weight, bias) matches the model output exactly.

## Findings

### Cross-task output collapse

Task 9 produces 75.32% of all final Class-IL predictions. Predicted task
fractions for Tasks 0 through 9 are:

~~~
0.0005, 0.0003, 0.0009, 0.0004, 0.0051,
0.0053, 0.0119, 0.0559, 0.1665, 0.7532
~~~

### Classifier parameters

Mean classifier row norms by task are:

~~~
0.8007, 0.8261, 0.8807, 0.8918, 0.9148,
0.8734, 0.8758, 0.8667, 0.8742, 0.8675
~~~

They are not monotonic in task recency; Task 4, rather than Task 9, has the
largest mean norm. In contrast, mean classifier bias is -0.4920 for Task 0
and ranges from +0.0243 to +0.0863 for Tasks 1–9. This makes classifier bias
a stronger candidate bias source than simple new-class row scale in this seed.

### Party contributions

Mean party contributions vary strongly by class task. For example, the
Task-9-minus-Task-0 marginal contribution is +0.2586, -1.2432, -1.6063, and
+0.7913 for Parties 0–3 respectively. This is evidence of task-specific party
heterogeneity, but it does not identify a stable party-recency correction from
one seed.

## Exploratory decision

NO_FORMAL_INFERENCE_CORRECTION_GATE

- Do not start inference-only Weight Aligning from this evidence: the required
  new-task row-norm pattern is not present.
- Do not start party-aware output alignment from this evidence alone: party
  heterogeneity requires deterministic multi-seed replication and separation
  from the classifier-bias effect.
- The next legal formal step remains retaining or reproducing full deterministic
  checkpoints for Static seed 43 and Uniform seeds 42/43, then rerunning the
  same read-only P0 protocol. Any later correction test must pre-register a
  Class-IL AA_final gain of at least +0.0100 and essentially unchanged
  Task-IL/BWT.
