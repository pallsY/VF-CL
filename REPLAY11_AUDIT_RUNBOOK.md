# Replay-11 v3 formal audit runbook

The final matrix uses only producer commit
`467611a603c9604ad57861920495a3ad05612037` and original jobs under
`/brf/users/yangxx/VFCL-results/replay11-paper-v3-20261013/full-3080`.
The authoritative data root is `/brf/users/yangxx/VF-CL/data`. The older
`99e81b2` jobs and their copies remain historical diagnostics and cannot
enter the final 99-cell table.

Freeze the audit checkout at a clean Git commit before running a job. On the
Linux host that can read the NAS and producer checkout, run:

```bash
/home/c3080/miniconda3/envs/hy3d/bin/python replay11_attest.py \
  isolet/mir/42 --producer-root /tmp/vfcl-replay11-paper-v3-full
```

The entrypoint always checks the 99-cell job registration, fixed producer
commit, original result and data roots, full command/data provenance,
training batches reconstructed from authoritative data, frozen task classes,
final checkpoint state and strict method restore, formal transaction and
metric arithmetic. Seed 42 then receives independent CPU numerical replay
from all frozen snapshots. Seeds 43 and 44 require the latest verdict for
their method and dataset's seed 42 to be a numerical pass from this same
audit commit. `replay11_numeric.py --worker` is an internal diagnostic and
does not create a formal verdict.

Each stage appends a success or failure to
`/brf/users/yangxx/VFCL-results/replay11-paper-v3-20261013/audit-v3/attestations.jsonl`.
Records include UTC time, host, auditor and producer commits, original path,
five key file hashes, whole-tree hashes before and after a stage, its result
or error, and a hash chain. A preflight failure is also recorded when the
auditor is dirty or a key file is unreadable. Do not rewrite this ledger or
erase a failure; reruns append new verdicts. The chain detects accidental
edits but is not an independent tamper-proof signature. Publish its head
hash to an external GitHub anchor branch after each formal audit batch and
record those anchor commits in the final report.

The 33 numerical jobs were fixed before results were viewed: seed 42 for
each of the 11 methods on each of the three datasets. A numerical result
without its paired basic pass, or a producer run without `JOB_SUCCESS.json`,
is not admitted. The final aggregate requires 99 basic passes and 33
numerical passes, all from a single audit commit and the fixed producer
commit. Pilot results are excluded.
