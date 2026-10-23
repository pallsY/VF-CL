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

The original accounting omitted the **frozen-old encoder upload** in each
probe. This was a message-counting error, discovered before any accuracy
reading or new training. After counting both current and frozen-old uploads,
probing every epoch gives **9.689% total-message savings on development
ISOLET**, below the locked 10% gate. This accounting correction, not a training
result, triggered the **sole preregistered fallback**: probe every **two**
epochs on **all three datasets**. `message_ledger_before_training.json`
preserves both calculations. With a 64-sample replay batch, two of four parties
on ordinary steps and one all-party step every four steps, the two-epoch
schedule passes the modelled 25% replay and 10% total gates in both development
and final configurations:

| Dataset | Development replay / total | Final replay / total |
| --- | ---: | ---: |
| ISOLET | 32.780% / 11.251% | 33.750% / 12.566% |
| Four-party UPMC | 35.876% / 10.967% | 35.953% / 11.053% |
| CIFAR-100 | 36.625% / 17.492% | 36.688% / 17.565% |

**Fallback exhausted.** The two-epoch schedule is now part of candidate A.
After the exact-source holdout manifests are generated, if any dataset in
either configuration fails either gate, stop candidate A under this protocol.
Do not change the number of selected parties `m`, full-party cadence `r`,
probe samples per class `p`, or probe frequency. Accuracy cannot be used to
revisit this decision.

## Selection controls at the locked cadence

| Arm | Weight source | Weight-refresh boundary | Probe communication |
| --- | --- | --- | --- |
| Uniform random | Equal party weights | Every two epochs (same values) | None |
| Existing `static` importance | Frozen class-party logit-contribution statistics, aggregated over old classes | Re-read and aggregate every two epochs | None; no new contribution measurement |
| Candidate A | Old-class contribution attenuation, A1 | Recompute from the probe every two epochs | Current and old embedding uploads |

All three use the same weighted-without-replacement sampler, independent
random stream, two-party ordinary steps, and all-party step every fourth replay
step. `static` is intentionally **not re-estimated on fresh data**: the frozen
implementation's importance statistics are fixed at their task boundary.
Re-reading them at each shared boundary aligns the schedule without silently
turning `static` into a new dynamic method. Uniform random requires no probe.
For `static`, let `a(k,c)` be the stored normalized absolute logit contribution
for old class `c` and party `k`. Use `s_k = mean_c a(k,c)`, then the same
`gamma = 0.2` smoothing and sampler as candidate A. This aggregation is fixed
before accuracy results; no class or party is selected from development scores.

The frozen configurations do **not** share one aggregation rule: ISOLET and
four-party UPMC use `concat` (four 128-dimensional party slots); CIFAR-100 uses
`sum` (four 512-dimensional embeddings). The earlier design discussion that
assumed `sum` for every dataset was incorrect. For a missing party the replay
implementation zeroes that party's own slot before `concat`, or adds zero in
`sum`. The same mask is applied to current and frozen-old embeddings. This
correction followed the first correctness pilot's shape failure, before any
accuracy readout. The message-byte ledger is unchanged because it counted
per-party embedding widths and selections, independent of aggregation.

These are **pre-run simulated message counts**, now reconciled with all three
exact-source 30-per-class tuning holdout manifests in
`message_ledger_verified.json`. They are not measured network bytes or
accuracy results. The ISOLET holdout was generated from a local NPZ whose
SHA-256 matches the frozen formal data anchor. Its selected-index digest is
`668cf1ce874c55a587647940d58181d7a42dc0ae0aa75c1dd4625004ef923089`.
The local UPMC NPZ is the earlier **two-party** clean cache (SHA-256
`dea0569f0d4195eaaecc9cab3a6cce533f07b8d88472e7dbca5113152a314339`,
image columns 0:512 and text columns 512:1536). The formal source is the
derived **four-party** cache at
`/brf/users/yangxx/VF-CL/data/upmc_food101_image_clean_4party_v1/upmc_food101_vfl.npz`
(SHA-256
`dbdd3a2979203ba0ee804de7d3b128426f869b0a127e4b87ebbe023e5750023b`).
The frozen `prepare_upmc_four_party.py` splits those columns into 0:256,
256:512, 512:1024, and 1024:1536 while preserving rows, labels, source IDs,
and train/test indices. The SHA mismatch is therefore an expected **different
data artifact**, not evidence that the local file is the formal four-party
cache. The tool correctly refused to make its holdout. After brf3080 became
reachable, the formal UPMC file and metadata hashes were checked against the
anchor, as was the CIFAR-100 train-file hash. The original validation and BiC
manifests were reconstructed from those exact sources before creating new
tuning manifests. Independently, all new manifests were checked for 30 unique
indices per class and zero overlap with existing validation and BiC indices.
UPMC selected-index digest:
`d19918d00acf93cd65f3095ea3a61914da54144e192ee0fd411d76bfaf3e595f`;
CIFAR selected-index digest:
`55b8130fee6f828d0e6be8fbc39e8db70f05aebb55130073da47873c77a9e5ea`.
All three are mirrored in the new NAS evidence root
`/brf/users/yangxx/VFCL-results/party-drift-scheduler-protocol-v1-20261023/holdout`.
No training or accuracy evaluation has started.

Both message gates now pass in the development and final configurations for
all three datasets. The correctness-only preflight is recorded below. If any
later exact message count fails either threshold with the
two-epoch probe schedule, stop this candidate under the locked protocol.

## Correctness-only preflight record

The disabled-switch identity probe compared independent clean checkouts of
frozen `e2a559d` and the new branch on fixed ISOLET and four-party UPMC input.
Task-0/task-1 model tensors, bounded raw memory, FIM masks, task-1 loss, and
gradients matched bit-for-bit. The first probe driver failed before training
because its manually constructed loader lacked a recording data-flow sampler;
the driver alone was corrected, and both identity probes then passed.

Earlier pilot failures remain intact on NAS: v1 required two deterministic
DataLoader workers; v2 used a NAS `TMPDIR` too long for an AF_UNIX socket; v3
exposed the mistaken sum-only aggregation assumption; CIFAR v5 exposed the
frozen BiC requirement for all ten tasks. These failures led to correctness
fixes, **not** accuracy selection. No official test metric was read.

| Pilot | Locked scope | Probe events / replay steps | Independent selection and message audit | Gate file SHA-256 |
| --- | --- | ---: | --- | --- |
| ISOLET v4 | 2 tasks, 1 epoch each, CPU | 1 / 3 | Passed; test access blocked | `b99ce1b7a3c30c8fc68b5f316bdb55c0f0e9d08556751dc01d7143ec55e09eee` |
| Four-party UPMC v4 | 2 tasks, 1 epoch each, CPU | 1 / 41 | Passed; 10 all-party steps; test access blocked | `2c4495f4f14dfa9f6b9381def9ddba977033984acd18848e8edad16835ea3858` |
| CIFAR-100 v6 | Full 10-task BiC path, 1 epoch each, GPU1 | 9 / 594 | Passed; 144 all-party steps; test access blocked | `5bd1a8a56d52344c1cc7a7c2050b2b6ab41921c08ee23ce74f52ebd88ee58b91` |

The exact gate JSON files are in `pilot_gates/`; full logs, audit rows, and
training-only checkpoints remain under the corresponding independent NAS pilot
roots. The vector pilots ran at code commit `f6500e6`; CIFAR ran at `f2291ee`.
Between those commits only the pilot driver and its test changed; the replay
method files did not. This closes the **correctness preflight** gate only.
Candidate A remains unproven on novelty and accuracy; no 70/71/72 development
run or 42/43/44 formal run has been started.
