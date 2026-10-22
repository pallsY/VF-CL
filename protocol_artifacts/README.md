# Partial-party replay: local preparation (2026-10-23)

This is a **pre-training simulated message ledger**, not network traffic or an
accuracy result. The frozen sample-only runs and their audit trees were read
only. The reference is **all-party replay**, not the frozen no-replay method.

The denominator includes current-task training, FIM, herding, final-head raw
sample encoding, final validation, CIFAR BiC calibration at each stage, and one
development tuning-holdout evaluation. Test inference and independent audits
are excluded. Current-task old-teacher embedding uploads are included. Each
selected party receives a 16-byte notification (`task`, `epoch`, `step`,
`party`, four big-endian unsigned 32-bit integers) and 8-byte sample IDs.

The first ledger mistakenly counted only one of the two probe uploads.
`message_ledger_before_training.json` corrects this: each probe uploads both
current and frozen-old embeddings. With a 64-sample replay batch, two of four
parties on ordinary steps and one all-party step every four steps, probing
every epoch fails the **development ISOLET total-message 10% gate** (9.689%).
The preregistered sole fallback, probing every **two** epochs on all datasets,
passes the modelled 25% replay and 10% total gates in both development and
final configurations:

| Dataset | Development replay / total | Final replay / total |
| --- | ---: | ---: |
| ISOLET | 32.780% / 11.251% | 33.750% / 12.566% |
| Four-party UPMC | 35.876% / 10.967% | 35.953% / 11.053% |
| CIFAR-100 | 36.625% / 17.492% | 36.688% / 17.565% |

These are **conditional** counts. The ISOLET 30-per-class tuning holdout has
been generated from a local NPZ whose SHA-256 matches the frozen formal data
anchor. Its selected-index digest is
`668cf1ce874c55a587647940d58181d7a42dc0ae0aa75c1dd4625004ef923089`.
The local UPMC NPZ has a different SHA-256 than the frozen four-party source,
so the tool refused to generate a UPMC holdout. The CIFAR training source is
not local. Their development ledger rows therefore remain projections until
the exact anchored sources are available and both holdout manifests are made.
No training or accuracy evaluation has started.

Before the two-task correctness preflight: generate and verify the UPMC and
CIFAR holdouts, independently rerun both message ledgers from their actual
manifests, and check every dataset against both gates. If either gate still
fails with the two-epoch probe schedule, stop this candidate under the locked
protocol.
