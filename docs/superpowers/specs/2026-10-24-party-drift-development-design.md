# Party-drift scheduler development experiment

Status: locked design before development accuracy access. Frozen Replay11 and
sample-only 42/43/44 source, results, and audits remain read-only. The existing
candidate implementation is at `f2291ee`; correctness pilots and disabled-path
identity are evidence, not development accuracy jobs.

## Cohorts and runs

Use the exact-source 30-per-class training holdout manifests already anchored
in `protocol_artifacts/`. Training seeds are 70, 71, and 72. The fixed
correctness-only seed is 74. The source data, task order, model, training
epochs, original 20 raw exemplars per class, validation/BiC access, and final
head procedure follow each method's frozen formal configuration. All
development evaluation uses the training holdout. The official test loader is
blocked in code and no official test metric is generated.

There are six replay arms: all-party replay; uniformly random 2-of-4-party
selection; selection from frozen `static` class-party importance; candidate A
selection from old-class contribution attenuation; direct four-party CMCDR
transfer; and an intervention-budget-matched CMCDR variant. All replay arms
share replay batch size 64, full-head replay CE, old-class logit KD (`T=2`,
weight `0.25`), and alpha grid `{0.25, 0.5, 1.0}`. The two partial-party
controls and candidate share the same 2-of-4 weighted-without-replacement
sampler, dedicated random-stream derivation, a full-party fourth replay step,
and one weight-refresh boundary every two epochs. Uniform weights are always
equal. `static` weights are derived solely from the existing frozen task-boundary
class-party contribution table, averaged across old classes, smoothed with
`gamma=0.2`; there is no fresh `static` probe. Candidate A uses the already
correctness-tested A1 attenuation probe with `p=2` old raw examples/class.

Direct CMCDR computes all 16 party-subset responses, including the empty
subset, on each replay batch. It transforms old-class target-vs-rest benefit
responses into the 15 nonempty Möbius contribution coordinates and adds the
paper's absolute plus normalized Smooth-L1 profile loss (`beta=1`,
`epsilon=1e-6`, transition `delta=1`) between current and stop-gradient frozen
models. Its own regularizer weight is fixed at the paper's main value `1.0`;
alpha scales only replay CE. The intervention-budget-matched variant uses the
same 16-subset loss but evaluates it on a deterministic subset of replay
examples. Its task-level cap is computed **before training** from candidate
A's scheduled count of full/leave-one-out top-model responses. With `C` old
classes and `P` candidate probe events in a task, candidate A uses
`2*(M+1)*p*C*P` sample responses. Each CMCDR example uses `2*2^M` responses;
the cap is `floor(2*(M+1)*p*C*P / (2*2^M))`, bounded by the available replay
positions. Before the task, an independent seeded RNG selects exactly that
many `(step, within-batch position)` pairs without replacement from the fixed
replay schedule. Selection is independent of labels and accuracy; the
underlying replay batches remain class-balanced. This matches a specific
intervention-count proxy, not total FLOPs. Actual model FLOP proxies,
wall time, stored bytes, and simulated messages are reported separately. The
cap calculation and selected replay IDs are audited; no development score may
change it. Both CMCDR variants have all parties on replay steps, as in a direct
vertical transfer.

The no-replay sample-only method is run on the same nine dataset-seed cells.
Selected strongest baselines are ER-ACE on CIFAR-100, iCaRL on ISOLET, and
DER++ plus iCaRL on four-party UPMC, all on the same holdout and 70/71/72
seeds. Baseline configurations derive from their own frozen formal protocols;
iCaRL is scored with its NME inference path. The matrix is 162 replay jobs,
9 no-replay jobs, and 12 selected-baseline jobs: **183 unique jobs**.

## Manifest and execution boundary

Generate one immutable UTF-8 JSON config per job and one canonical manifest
with exactly one row per unique `(dataset, method, alpha, seed)`. Each row
contains the config relative path and SHA-256, source-data/config SHA-256,
holdout manifest SHA-256, and approved production commit. The canonical
manifest's SHA-256 is committed and copied to the independent NAS audit root
before any accuracy job. A launcher receives a job key and the manifest path,
verifies the manifest digest, config bytes/hash, commit, data identity,
training-only holdout, and unique output path, and refuses all unlisted work.
The queue writes an exclusive plan and append-only start event before spawning
the worker. The worker verifies its parent queue PID, active queue lock, plan,
and start event before training; an independent CLI invocation cannot bypass
the retry policy. The approved config bytes remain intact beside a runtime
config containing the actual attempt output path. No implicit scan for extra configs,
overwrite, or duplicate launch is allowed.

For each new replay control, run one correctness-only seed-74 pilot before the
183-job queue: two tasks for ISOLET and four-party UPMC, and the full ten-task
CIFAR stream at one epoch per task. The frozen CIFAR data loader requires the
BiC calibration split to coexist with validation, and its formal BiC cache
requires all ten tasks. The abandoned two-task/BiC-off CIFAR driver stopped
before training; its log and config remain in the failed v2 pilot root.
Verify exact selection/weight source, 16-subset CMCDR
interventions where applicable, teacher/student masking, replay and probe
counts, full-head CE and old-class KD, raw sample budget, RNG isolation, and
test blocking. The candidate's full ten-task BiC path and disabled-switch
identity also have separate evidence.
Failures preserve files and logs. No failed pilot is converted into an
accuracy job.

The development queue starts with UPMC, then ISOLET, then CIFAR-100. It may
schedule independent listed jobs across idle qualified GPUs but never compete
with another user's process. It records health, runtime, checkpoint, and audit
status while withholding accuracy summaries until all listed jobs finish or a
terminal failure requires a decision. If a host outage, interruption, or GPU
memory failure occurs, the exact same config and seed may be retried once the
resource condition is corrected; each attempt, cause, and output path remains
in an append-only record. If the queue dies before writing a failure gate,
recovery requires an explicit reason and checks the original queue and workers
are no longer running. It reconstructs unfinished start events from preserved
logs and success gates, marks training anomalies without retry, and authorizes
only infrastructure attempts with a per-job append-only resume record.
A NaN, nonfinite/divergent loss, or other
training-origin anomaly (nonfinite epoch loss or absolute mean epoch loss
above `1e6`) is the configuration's recorded outcome: no retry,
changed alpha, batch, or replacement seed. Failed or incomplete jobs cannot
be silently excluded from the result matrix.

Before a job is sealed, the selection ledger must replay exactly: refresh and
probe epochs, per-task step counts, party subsets, sampled old exemplar IDs,
CMCDR interventions, and simulated message bytes. Its SHA-256 and the readout
SHA-256 are recorded in `JOB_SUCCESS.json`; the success gate's SHA-256 is
recorded in the append-only queue event and checked again before adjudication.
No-replay arms are required to have no selection ledger.

## Frozen reading and decision rules

Once all jobs and audit checks finish, for each replay method and dataset
select alpha by the highest **unrounded** three-seed mean final holdout
Class-IL accuracy; exact ties choose smaller alpha. Publish every arm/alpha/
seed result. Candidate-vs-uniform and candidate-vs-static differences at each
identical alpha isolate selection; independently tuned best-of-grid scores
compare equal search opportunities. Paired strongest-baseline comparisons use
the listed methods and seeds, with iCaRL NME.
For CIFAR, both adaptive and ER-ACE use the frozen final joint-alpha/beta BiC
fit rule: adaptive fits on its reserved validation cohort and ER-ACE on its
reserved calibration cohort, then both are scored on the separate development
tuning holdout. Raw and calibrated Class-IL are both archived; calibrated is
the predeclared comparison metric. The official test split is never opened.

The candidate progresses to a new, separately frozen 42/43/44 confirmation
only if its selected-alpha three-seed mean beats every listed strongest
baseline on **each** dataset and, at that same candidate-selected alpha for
the dataset, its three-seed paired mean difference is positive against both
uniform and static selection. A tie is not a win. An alpha with a training
anomaly in any seed has no three-seed mean and is ineligible for selection;
its failed cells remain in the published grid. If only baseline superiority
holds, replay improved accuracy but
the selection mechanism is unsupported. If only selector superiority holds,
the mechanism has development evidence but fails the three-dataset-first
goal. If neither holds, close the direction. For either mixed outcome, stop
under this holdout; a revised mechanism would require a new protocol,
development seeds, and holdout. No result from 42/43/44 is used to select
components or alpha.
