# Factorized Adaptive Head V1 Design

Date: 2026-10-06

## Purpose

The frozen seed-42 training-validation diagnostics show that CIFAR-100 Adaptive improves cross-task routing over Bias while losing 0.24 percentage point of within-task accuracy. A read-only counterfactual retained Adaptive's task probability and used the pre-consolidation head within each task: CIFAR-100 within-task accuracy rose from 73.60% to 74.64%, while Class-IL changed from 34.80% to 34.76%. ISOLET and UPMC Class-IL changed by +0.10 and -0.09 point. These are development observations, not test evidence.

## Frozen method

Use the existing two branches, validation-fitted scalar gate, task mapping, and pre-consolidation classifier. For task t and class y in t, define

`log p_F(y|x) = log p_M(t|x) + log p_pre(y|t,x)`.

Here `p_M` is the existing Adaptive probability mixture, `p_M(t|x)` is its sum over classes in t, and `p_pre(y|t,x)` is the pre-consolidation classifier's softmax restricted to task t. The rule has no fitted parameters, dataset identity checks, or test-derived inputs. It preserves the pre-head ranking within every task exactly.

## Runtime and compatibility

Add an explicit inference readout on TopModel; its existing `forward`, training, gate fitting, checkpoint schema, and stored audit diagnostics remain unchanged. The readout derives everything from existing adaptive state, original classifier weights, and the saved class-to-task calibration map. A reloaded checkpoint must reproduce its output bit-for-bit on deterministic CPU fixtures. Reject missing or inconsistent task maps, incomplete adaptive state, non-finite logits, or output that is not normalized. Legacy and fixed-endpoint checkpoints retain their behavior.

## Evaluation sequence

1. Test the mathematical rule, malformed-state rejection, cosine and linear heads, and checkpoint round trip.
2. Reproduce the three frozen seed-42 training-validation counterfactuals from historical checkpoints. Read no test examples in this stage.
3. Freeze a code commit and exact evaluation protocol before final test access. Compare the new readout with Adaptive, Full, and Bias on CIFAR-100, ISOLET, and UPMC Food-101 under the same task streams and metrics: AA-final, BWT, and final Task-IL. Keep validation and test evidence separate.
4. Report all datasets and seeds, including negative results. Do not select a rule from test metrics. Do not relabel or overwrite existing formal runs. Any independent generalization claim requires an untouched held-out dataset after the method is frozen.
