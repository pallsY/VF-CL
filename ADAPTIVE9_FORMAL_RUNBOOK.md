# Adaptive9 four-party formal comparison

The user's adaptive method is a separate proposed-method series. It is not
AdaGauss and is not one of the 99 Replay11 baseline cells. The baseline code,
results, and audit ledger remain frozen independently.

## Frozen scientific protocol

- Cells: CIFAR100, ISOLET, UPMC Food101 × seeds 42, 43, 44 = 9.
- Parties: four for every dataset. UPMC uses two image feature owners and two
  text feature owners from `upmc_food101_image_clean_4party_v1`.
- Class order, train/validation/test split, model, optimizer, replay cap,
  deterministic sampler, and deferred Class-IL evaluation come from the
  reviewed `adaptive9_matrix.command_for` and the Replay11 dataset protocol.
- Epochs per task: CIFAR100 50, ISOLET 50, UPMC 20.
- Proposed method: `proto_evolve` with `adaptive_dual_branch` head
  consolidation, class-balanced gate, and 20 raw head exemplars per class.
- Producer commit:
  `9973018295da270fd6098de07dcf36b9fe65ff61`.
- Audit commit: pin the reviewed HEAD of `codex/adaptive9-audit-v1` in the
  external GitHub anchor before any formal job; pass that exact SHA to every
  audit command. Changing either frozen commit requires a new versioned run.

## Fixed locations and eligible hosts

- Data: `/brf/users/yangxx/VF-CL/data`.
- Results: `/brf/users/yangxx/VFCL-results/adaptive9-fourparty-v1-20261015/full`.
- Append-only ledger:
  `/brf/users/yangxx/VFCL-results/adaptive9-fourparty-v1-20261015/audit/attestations.jsonl`.
- Final gate: sibling `FINAL_GATE.json` in the audit directory.
- The auditor runs on brf3080 with a clean producer checkout at
  `/tmp/vfcl-adaptive9-v3` and a separate checkout of the anchored audit
  commit. Production may run on brf3080 or c5090 only.
  Both hosts must report the exact runtime versions pinned in
  `adaptive9_evidence.RUNTIME`; c5090 must have an idle GPU before launch.
- c40901 may run local preflights. Its present SSHFS mount fails immutable
  inode verification, so it cannot produce formal cells.

## Launch and audit order

1. Confirm the external anchor, clean producer and auditor checkouts, exact
   source commit, three dataset hashes, true NFS inode behavior, runtime
   versions, GPU ownership, and available result storage.
2. Launch each `(dataset, adaptive, seed)` at most once with
   `adaptive9_matrix.py run-one`. Its fixed job directory must not already
   exist. Keep task/seed assignments disjoint across hosts.
3. Complete and audit seed 42 for all three datasets first. On brf3080 run
   `adaptive9_attest.py audit-one --dataset DATASET --seed 42
   --audit-commit AUDIT_SHA`. This appends basic and independent numeric
   verdicts to one hash-chained ledger. A failure stops dependent launches.
4. Only after all three seed-42 numeric verdicts pass, launch seeds 43 and 44.
   Audit each completed job with the same command and frozen audit SHA.
5. If an audit attempt fails, retain its ledger event and source tree. Diagnose
   the cause; a new attempt appends fresh basic and, for seed 42, numeric
   verdicts. Never edit or delete old ledger rows.
6. Once 9 basic and 3 numeric latest verdicts pass, run
   `adaptive9_attest.py finalize --audit-commit AUDIT_SHA`. This rechecks all
   original trees and publishes the immutable final gate. Anchor the ledger
   SHA-256 and chain head on GitHub, then assemble comparison tables.

Preflights under `adaptive9-*-preflight-*` use one epoch per task and are never
formal results. UPMC and ISOLET full-task preflights passed their respective
evidence and numeric checks. CIFAR requires its full ten-task BiC path even
for a low-budget preflight; shortening to two tasks fails the reviewed
configuration guard before training.
