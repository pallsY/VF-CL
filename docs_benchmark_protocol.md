# VFL-CLU Benchmark Protocol (v1, 2026-07-02)

Benchmark of **Continual Learning × Unlearning in Vertical Federated Learning**.
Main table = full CL×UL grid; curation into paper tables happens post-hoc.

## 1. Method grid

**CL (12 rows).** finetune (lower bound), ewc, lwf, lwf_wa, lwf_fim, afc, gpm,
adagauss^g, target^g, proto_evolve, proto_fedspace, ours (proto+OC,
`--own_concentrate_weight`). ^g = generative/prototype replay (allowed).
**Excluded:** er / der_pp / er_ace — raw-exemplar replay stores the very samples
unlearning must delete (protocol forbids raw old-class sample storage);
proto_aug / prl — rot90 self-supervision undefined on 1-D feature vectors;
proto_evolve_radapt — internal variant.

**UL (9 cols).** retrain (oracle), gradient_ascent, luv, mode, fucrt, fedup,
fedosd, fedau, ours (roar `--roar_scrub_mode retrain_s` localized retrain).
**Excluded:** fudp — prunes Conv2d only, no-op on MLP bottoms; radapt_router —
internal variant.

Grid = 12 × 9 = 108 combos.

## 2. Datasets (3, fixed)

| | parties (P) | classes | why it is in |
|---|---|---|---|
| CIFAR-100 (column split) | 4 (equal image strips) | 100 | long queue (10 tasks), P-scaling axis, homogeneous-split regime, CNN bottoms, comparable to V-LETO-line VFL-CL work |
| Covertype (cross-org semantic split) | 6 (terrain/hydro/infra/illum/wild/soil) | 7 | real cross-org tabular, heterogeneous, non-saturated (joint ≈ 0.74) |
| NUS-WIDE (multi-modal) | 6 (CH 64 / CORR 144 / EDH 73 / WT 128 / CM55 225 / Tags1k 1000) | 10 | canonical VFL benchmark (image+text), multi-modal heterogeneity; single-label subset (see prep_nuswide.py, min class = 3,900) |

mfeat / HAR / synthvfl are appendix or controlled-axis material only
(synth ρ-sweep = heterogeneity axis; synth P-sweep supplements CIFAR P axis).

## 3. Task queues (two orders per dataset)

Order 1 forgets OLD classes; Order 2 forgets RECENTLY-learned classes.
CLI: `--custom_tasks A|B|... --unlearn_after_tasks i,j --unlearn_classes "x;y"`.

**CIFAR-100** (10 tasks × 10 classes, `custom_tasks "0..9|10..19|...|90..99"`):
- Order 1: UL after T2 forget {0,1}; after T5 forget {20,25}; after T8 forget {50,55}
- Order 2: UL after T2 forget {28,29}; after T5 forget {58,59}; after T8 forget {88,89}

**Covertype** (`custom_tasks "0,1,2|3,4|5,6"`):
- Order 1: UL after T1 forget {1}; after T2 forget {6}   (1=diffuse-owned, 6=concentrated)
- Order 2: UL after T0 forget {0}; after T2 forget {4}

**NUS-WIDE** (`custom_tasks "0,1,2,3|4,5,6|7,8,9"`):
- Order 1: UL after T1 forget {0}; after T2 forget {5}
- Order 2: UL after T0 forget {2}; after T2 forget {8}

## 4. Metrics (per run; the Table-1 columns)

- **RA↑** retained-class accuracy at stream end (`final_ul_eval.retain_acc`)
- **FM** forgetting/BWT on retained classes (tracker CL metrics)
- **UA↓** forget-class accuracy **at stream end** (`final_ul_eval.forget_acc`,
  per-class in `forget_acc_per_class_final`) — measured after ALL subsequent
  learning, so it includes relapse; per-event values remain in `ul_eval`
- **MIA↓** re-learned linear head ROC-AUC at stream end
  (`final_ul_eval.relearn_auc_final`), reported as excess over the retrain
  oracle's AUC (intrinsic-separability floor), not vs 0.5
- **#P↓** parties whose encoder was modified per UL event (`parties_touched`)
- **RTE** wall-clock per event (tracker timing)

## 5. State-sanitization protocol (cl_methods/sanitize.py)

At every UL event, after the UL operator runs: purge class-keyed caches
(prototypes/gaussians), re-snapshot frozen teachers + EWC anchors from the
post-unlearning model, ban forget classes from generative replay, drop
forget-class buffer rows. Rationale: otherwise the next CL event distills or
replays the forgotten class back in (relapse), and teachers/anchors actively
pull weights toward the pre-unlearning model. `--sanitize_cl_state 0` is the
ablation (expected finding: prototype/distillation CL relapses hardest).

## 6. Run matrix

Main grid: 108 combos × 3 datasets × 2 orders × 3 seeds (42/43/44).
Tabular runs are minutes; CIFAR-100 dominates compute — dispatch Order 1 first.
Sensitivity axes reuse the curated row subset: P ∈ {2,4,8,16} (CIFAR + synth),
heterogeneity (real random-vs-semantic split + synth ρ, x-axis = measured
ownership-entropy ratio via probe_ownership.py).
