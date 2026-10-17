# Four-party adaptive sample-only formal ablation

## Question and comparison

Measure the effect of removing persistent prototype statistics, generated
prototype replay, SDC, and feature distillation from the frozen adaptive method.
This is a bundled, secondary ablation. It is not pure ER and does not establish
equal total memory or compute by itself. The existing Replay11 and adaptive9
production commits, results, and audit ledgers remain untouched.

## Producer protocol

Start from frozen adaptive producer `9973018295da270fd6098de07dcf36b9fe65ff61`
and port only the reviewed sample-only behavior from development source
`99e1da7e05e7eeb8b57587b6d3a75c6eaa490066`. Keep the formal dataset
registry, training budgets, task order, model, optimizer, validation split,
deferred evaluation, and effective non-feature hyperparameters. In particular,
retain `distill_weight=0.25`, PartyKD, FIM, BiC, the class-balanced adaptive
dual-branch head, its final-only 500-step fit, and 20 raw training examples per
class. Set `sample_replay_only=1`, `proto_sdc=false`, and
`feat_distill_weight=0`; no old-class prototype cache or generated prototype
loss may contribute to training. The raw buffer is re-encoded for the final
head fit. The variant is not minibatch ER.

Run CIFAR-100 and ISOLET for 50 epochs per task, UPMC Food101 for 20, with
seeds 42, 43, and 44. All datasets have four parties. UPMC uses the frozen
clean four-party cache with image/image/text/text feature owners and SHA-256
`dbdd3a2979203ba0ee804de7d3b128426f869b0a127e4b87ebbe023e5750023b`.
The approved data hashes and task lists otherwise match adaptive9. Use a new
results root `/brf/users/yangxx/VFCL-results/adaptive-sample-only-fourparty-v1-20261017`
with `full` and `audit` directories. This root cannot admit old two-party UPMC
or development holdout runs.

## Evidence and gate

Use isolated producer and auditor commits. Freeze and externally anchor both
commits, data hashes, the nine-cell command matrix, and the fixed audit root
before formal production. Every job writes one immutable plan and success
record, full task snapshots, delayed evaluation transactions, and source/data
identities. The independent auditor uses a clean checkout and checks the
sample-only flag, empty `global_protos` and `prev_protos`, exactly 20 raw
examples per seen class, source/command/data hashes, training data flow,
checkpoint restore, and metric arithmetic. Seed 42 also receives independent
CPU numerical replay on all three datasets. Its three basic and numerical
verdicts must pass before seeds 43 and 44 start. Failures append a ledger
event, stop dependent launches, and preserve original files. Finalize only
after 9 latest basic and 3 latest numerical passes; anchor the final ledger
SHA-256 and chain head outside the result root.

## Validation and interpretation

First run focused formula/config/state tests and full-task, one-epoch
preflights for ISOLET, UPMC, and CIFAR-100. CIFAR preflight retains all ten
tasks and the real BiC path. Preflights are never formal results. Freeze code
and audit after preflight, then launch formal jobs without test-guided tuning.
The official test set was already viewed during earlier experiments, so this
variant is reported as a follow-up ablation rather than independent blind
confirmation. Compare paired seeds to the frozen adaptive9 results and show
the original Replay11 baselines only as context. Report raw memory, auxiliary
state, and extra computation separately; do not claim equal resources without
measured evidence.
