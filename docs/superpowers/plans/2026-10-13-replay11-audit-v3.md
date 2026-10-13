# Replay-11 v3 Basic Audit Implementation Plan

**Goal:** Verify actual producer commits and certify eligible old results against corrected code.

**Architecture:** Extend the read-only v2 verifier. Load committed command generators, compare normalized formal commands, and compare relevant Git source blobs. Preserve transaction and archive checks.

**Tech Stack:** Python standard library, unittest, Git CLI. Use `PYTHONUTF8=1` on Windows.

## Global Constraints

- Old producer: `99e81b28e0544e182f777a34d7e7a786a1c3b920`.
- Corrected producer: `467611a603c9604ad57861920495a3ad05612037`.
- Formal matrix has 99 cells; `pilot_deferred` and `pilot_bic` never count.
- Source and NAS results remain read-only; seed-42 numerical replay is separate.

### Task 1: Producer identity

**Files:** `test_replay11_evidence.py`, `replay11_evidence.py`.

- [ ] Test corrected job acceptance, old changed-method rejection, pilot rejection, and mismatched JOB_PLAN/JOB_SUCCESS commit rejection; observe red.
- [ ] Derive producer commit from matching records and check it throughout result provenance; observe green.

### Task 2: Source and command equivalence

**Files:** `test_replay11_evidence.py`, `replay11_evidence.py`.

- [ ] Test five unchanged methods across datasets, changed command and changed source rejection; observe red.
- [ ] Generate committed formal commands, normalize only machine paths, and compare the target method plus shared execution source blobs; return explicit certificate; observe green.

### Task 3: Catalog and archive

**Files:** `test_replay11_evidence.py`, `replay11_evidence.py`.

- [ ] Test mixed producer catalog, forged certificate, pilot catalog and corrected archive record; observe red.
- [ ] Accept only known producer identities and certified old rows, retain real producer commit in archive and matrix outputs; observe green.

### Task 4: Verify and review

- [ ] Run `python -m unittest -q test_replay11_evidence test_replay11_matrix`, `git diff --check`, and inspect changed paths.
- [ ] Commit audit-only changes locally, report limits and numerical replay handoff; do not push.

### Review corrections after first audit commit

- [ ] Bind COMPLETE evaluation payload to results and restrict the corrected producer entrypoint to the observed frozen executable path.
- [ ] Include Balanced Contrast's ER parent source in its reuse certificate.
- [ ] Read and validate training data-flow records; hash authoritative data files when `data_root` is supplied. Without that root, mark the row `EVIDENCE_INCOMPLETE`.
- [ ] Make the public 99-cell merger reopen each job directory and actual data root; reject copied jobs presented as originals.
- [ ] Distinguish destination archive integrity from source-to-copy equivalence. If the original tree cannot be read, return `SOURCE_EQUIVALENCE_UNPROVEN` and keep it out of the formal merger.
- [ ] Trial one corrected ISOLET/MIR original and one historical D: archive read-only, then record what each trial establishes.

The later 33-cell numerical stage must use the frozen `REPLAY11_AUDIT_V3_INTEGER_DERIVATION_20261013.md`: obtain authoritative task totals, uniquely invert each four-decimal stored accuracy to an integer correct count, and compare an independent replay's integers exactly. The original results do not explicitly store those integers.

### Final corrected-producer admission

- [ ] Enumerate all 99 cells only beneath `/brf/users/yangxx/VFCL-results/replay11-paper-v3-20261013/full-3080`, with producer `467611a603c9604ad57861920495a3ad05612037` and data root `/brf/users/yangxx/VF-CL/data` pinned outside each job record. Old jobs are historical diagnostics only.
- [ ] Reject a copied job even when its internal results path is edited; require both command `results_dir` and `data_path` to equal externally pinned locators.
- [ ] Independently rebuild all task/epoch train loaders from authoritative bytes and producer deterministic sampler/worker settings; compare each batch index, shape, dtype, and post-augmentation SHA. Any mismatch fails closed.
- [ ] Restricted-load the final checkpoint, check four-party model/tracker state and exact method state keys against corrected producer definitions, then run the fixed seed-42 numerical module separately.
- [ ] Trial the corrected ISOLET/MIR/42 original and record wall time, CPU, memory, and evidence digests before extending to the remaining 98 cells.
