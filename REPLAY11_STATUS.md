# Four-party replay baseline status

The 11 selected methods are Balanced Contrast CIL, LPR, HPCR, OCAR,
aL-SAR, MIR, iCaRL, ER, DER++, ER-ACE, and AdaGauss. The first seven have
method-specific VFL implementations under `cl_methods/`; the last four
reuse existing implementations. Each party receives an aligned feature view.

The UPMC clean cache is versioned separately. Image embeddings are split
into two 256-dimensional parties; text embeddings into two 512-dimensional
parties. `prepare_upmc_four_party.py` derives the new cache from the
selected clean source and preserves row order, labels, sample IDs, and
train/test indices. The original two-party cache remains available.

`replay11_matrix.py` defines 3 datasets × 11 methods × 3 seeds. Its current
`run-one` and `run-shard` commands are **pilot** runners: they evaluate the
test set after each task (`formal_deferred_evaluation=0`). The resulting
`REPLAY11_PILOT_TABLE.csv` must not be presented as a formally audited paper
table. The existing deferred evaluator and audit must be extended to all
seven new method checkpoint states and the four-party UPMC cache before a
formal 99-cell comparison is run. Full pilot jobs can still check runtime,
memory, and model behavior.

The VFL adaptation uses the shared model and optimizer protocol. MIR's
virtual SGD step follows the original retrieval algorithm, while the real
training optimizer remains the shared AdamW configuration. Method-specific
losses, exemplar selection, projection/preconditioning, or regularization
remain in each implementation; image-specific augmentation in HPCR is not
applied to the vector datasets.

Verification on the development server included method unit tests, real
UPMC/ISOLET two-task runs for the new methods, CIFAR ResNet path smokes, and
four-party cache equality checks. Those runs are feasibility checks, not
three-seed accuracy results. See the branch test outputs and job records for
precise evidence; do not infer a completed formal comparison from this file.
