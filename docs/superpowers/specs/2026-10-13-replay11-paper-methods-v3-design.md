# Replay-11 paper-method VFL adaptation design

Date: 2026-10-13. Approved design choices: paper equations are canonical; author code is supporting evidence; preserve the existing task-based VFL training budget and disclose departures from the online stream protocols.

## Objective and fixed experiment contract

Produce an auditable, four-party, task-based VFL comparison of eleven replay baselines on CIFAR100, ISOLET, and UPMC Food101. The matrix is 3 datasets × 11 methods × seeds 42, 43, 44 = 99 jobs. Preserve the already constructed UPMC image-two-party plus text-two-party split and its data hashes. The existing *within-dataset* budgets are CIFAR100 50, ISOLET 50, and UPMC 20 epochs per task; the earlier statement that all three use 50 was incorrect. Keep the class order, train/validation/test partition, deferred final evaluation, and metrics protocol fixed.

The seven paper methods are Balanced Contrast CIL, LPR, HPCR, OCAR, aL-SAR, MIR, and iCaRL. ER, DER++, ER-ACE, and AdaGauss remain the other four baselines. The target is a **paper-equation-aligned task-based VFL adaptation**, not numerical reproduction of the original online papers. The report must list each online-to-task adaptation explicitly, including replay insertion timing and sampling distribution. Never call a method paper-faithful when a core equation, selection rule, or required state transition is absent.

## Algorithm boundaries

| Method | Required alignment | Deliberate task-based VFL departure |
|---|---|---|
| Balanced Contrast CIL | Keep the balanced contrast proxy loss and its class-frequency weights; verify numeric loss, gradient, and task index. | Four-party joint embedding; common task-end replay buffer and per-dataset epochs rather than the paper's one-pass stream. |
| LPR | Use its Woodbury affine gradient preconditioner. Jointly precondition BatchNorm scale and bias with the normalized-input 2×2 covariance. Preserve activation covariance mass when restricting reference samples/rows. | Reference subset and refresh interval are declared hyperparameters under a task-based replay loop. |
| HPCR | Implement the paper's conditional sample-similarity terms in proxy contrast, cosine bias-free head, cosine temperature schedule, class-selected weighted PCD, and printed SCD orientation. Document the apparent sign inconsistency of the paper's PCD equation against the prose/author code; minimize positive squared error. | Multi-party embeddings, shared task budget, and specified bounded memory. |
| OCAR | Correct old/new class weighting, A/G exponential covariance estimates, and the damped inverse of the K-FAC block (within any declared low-rank approximation). Apply the preconditioned parameter update without an extra momentum/AdamW transformation. | Multi-party layers, bounded factor rank, and task-based replay. Factor-rank approximation is disclosed and checked against small dense equations. |
| aL-SAR | Implement paper Eq. 4–9: whole-model compute/Fisher BFC, every-fourth-step full Fisher refresh, per-example gradient-cosine similarity, dynamic frequency discount, and SAR temperature. Retain the paper's additive frequency formula; remove negative-similarity clipping. | Four-party parallel cost model and task-based arrival/replay schedule. The absence of exact online memory-only data flow must remain disclosed if retained. |
| MIR | Keep MI-1 before/after candidate loss ranking and make the virtual update use the same current-batch objective and effective optimizer update as the real step. | Candidate buffer is assembled under the task-based replay contract. |
| iCaRL | Use a dataset-fixed total exemplar budget `K=20 × final number of classes`, `m=floor(K/seen classes)`, ordered herding and old-class prefix reduction; optimize the BCE risk of the union of current-task examples and all old exemplars using their dataset-size weights; keep old-class sigmoid distillation and nearest-exemplar-mean inference. | Four-party feature extraction and common per-dataset epoch budget. Old exemplars are drawn as a stratified random minibatch, so this is an unbiased per-example estimator of the union risk rather than one complete joint shuffle per epoch. BatchNorm batch composition can still differ from a literal joint loader and must be reported. The fixed `K` equals the maximum capacity implied by the shared 20-per-class cap; occupancy over early tasks may differ by method. |

The existing source commit `99e81b28e0544e182f777a34d7e7a786a1c3b920` and its raw results are immutable historical evidence. Corrected jobs start from initial weights on a new producer commit; no corrected run resumes an old method checkpoint.

## Implementation and evidence flow

1. Keep each method's algorithm in its existing `cl_methods/<method>.py` file and use narrow, method-conditional trainer or model changes only where necessary. Avoid a new generic continual-learning framework.
2. Before each change, write a test that numerically compares a small case to the cited paper equation or detects a state-transition difference. Run it red, implement the smallest correction, and run it green. Add one four-party two-task integration test per corrected method for replay alignment, checkpoint restore, and inference.
3. Freeze the new producer commit and command matrix after tests and pilot deferred evaluations. Run complete jobs for changed methods × datasets × three seeds. An old unchanged-method result can enter the final table only after a signed equivalence record proves identical data, command, relevant source behavior, evaluation code, and metrics; otherwise rerun it.
4. Produce per-job provenance rather than silently mixing Git commits. Check all 99 final table entries for source/data/config/hash/metric integrity, and independently replay the predetermined seed-42 entry of every method × dataset cell (33 entries). A failed pilot or strict replay halts that method/data cohort until diagnosed; the numerical pass threshold is fixed before the first replay.
5. Publish mean and standard deviation only from accepted final jobs. Keep old scores labelled as preliminary task-based variants and show the online-to-task adaptations and any approximation alongside the new baseline table.

## Source references

- Balanced Contrast CIL: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.70060
- LPR: https://proceedings.mlr.press/v235/yoo24a.html
- HPCR: https://arxiv.org/pdf/2309.15038
- OCAR: https://arxiv.org/html/2502.01866
- aL-SAR: https://proceedings.iclr.cc/paper_files/paper/2025/file/760564ebba4797d0dcf1678e96e8cbcb-Paper-Conference.pdf
- MIR: https://proceedings.neurips.cc/paper/2019/file/15825aee15eb335cc13f9b559f166ee8-Paper.pdf
- iCaRL: https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf
