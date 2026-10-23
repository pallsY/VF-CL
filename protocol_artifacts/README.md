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
all three datasets. The next gate is the two-task **correctness-only**
preflight. If any later exact message count fails either threshold with the
two-epoch probe schedule, stop this candidate under the locked protocol.
