# CIFAR-100 task-time selection-view herding pilot

Date: 2026-10-08 (Asia/Shanghai). Status: **single-seed representation signal** in a training-only development run. This is not a deployed method, held-out accuracy result, or paper novelty claim.

## Locked experiment

Fresh seed 49 used the clean Adaptive producer commit `7bfe6b1d724fb1206bc0053a9008126bad86332d`, the audited seed-42 config (SHA-256 `5ef9984e091b83d7f84ac373cd702811c0ed15595dc6f4075ca23cc515535955`) and formal record (SHA-256 `aad02da1ba8e87f29e79f5702e71fe8e99c8ef4cdf554976cc1b881af6dc0871`). Only the seed, non-formal deferred-evaluation flag, output paths and experiment name changed. The model, ten-task stream, 50 epochs/task, 20 replay images/class, and BiC/lambda holdouts remained fixed; the prescribed validation split was reused. The external [launcher](../../../launch_cifar_selection_view_pilot.py) stopped after the frozen final checkpoint and before deferred test evaluation. All ten CIL event checkpoints exist, and the data-flow log has **zero test-loader accesses**. No CIL/TIL accuracy was produced.

The [analyzer](../../../analyze_cifar_selection_view_pilot.py) reconstructed two task-time selections from the same 450 eligible training images/class, excluding both holdouts:

- **S:** original-image content selected by the producer's existing herding over one random crop/flip view per image, recovered from saved replay tensors by [exact pixel matching](../../../cifar_replay_id_match.py);
- **D:** a shadow set selected by the repository's unchanged `herding_indices` over deterministic views, using the bottom encoder saved at that class's task boundary;
- **O:** an optimistic reference selected by the same `herding_indices` over deterministic views with the **final** encoder and complete historical training pool.

S and D both contain 20 original-image selections/class. D was reconstructed from event checkpoints after training and **never entered the training memory or head fit**. At the final checkpoint, all S/D/O images were embedded with the same deterministic training-view transform and the same four bottom encoders. Their L2 errors use the normalized-feature mean of all 450 eligible images/class as the target. The analyzer fitted no head and called no validation or test loader. One saved S tensor matched byte-identical original images under multiple eligible indices; a fixed canonical ID represented that content. No match to distinct original content was ambiguous.

## Results

Lower centroid error is better. Values are means over 100 paired classes.

| Feature space at final encoder | S: stochastic selection | D: deterministic selection | O: offline final selection | S−D | D−O | Classes S>D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Aggregated | 0.04544 | 0.03316 | 0.03142 | **0.01228** | 0.00174 | **100/100** |
| Party 0 | 0.08103 | 0.06270 | 0.06107 | 0.01834 | 0.00162 | 91/100 |
| Party 1 | 0.08124 | 0.06721 | 0.06513 | 0.01403 | 0.00208 | 96/100 |
| Party 2 | 0.07887 | 0.06660 | 0.06503 | 0.01226 | 0.00158 | 90/100 |
| Party 3 | 0.07007 | 0.05650 | 0.05691 | 0.01357 | −0.00041 | 87/100 |

The final aggregate S−O gap is `0.01402`; D reduces it by `0.01228`, or **87.6% in this seed**. D is close to O on average, but O is greedy and not guaranteed to win every class: D has higher aggregate error than O in 71/100 classes. At the task boundary, before encoder evolution, S averaged `0.04848` and D `0.03363`, with S>D in 100/100 classes. That selection-time advantage is expected because D directly minimizes the deterministic centroid objective. The important observation is that the advantage remains under the final encoder. S and D retained sets overlap by only **149 of 2,000 image-content slots** (1.49/class on average), so this is a substantive change in retained content.

The [aggregate JSON](data/cifar-selection-view-herding-seed49-20261008.json) contains class-level aggregate and party errors, task-boundary errors, overlap counts, and provenance hashes. These summary values, signs, task-group means and overlap counts were independently recomputed from its 100 class rows. Every one of the ten task groups has a positive final aggregate S−D mean.

## Interpretation and next step

With model training fixed and both sets evaluated through the same final view, deterministic task-time herding selected more representative image content than the current stochastic-view herding on this development seed. This supports a **selection-view mechanism hypothesis**. It does not establish improved CIL/TIL accuracy, because D was only a shadow selection and did not influence training or final head consolidation. It also does not establish a participant-aware contribution mechanism or performance on ISOLET/UPMC.

The next experiment should integrate deterministic task-time selection into an actual equal-memory online replay method and compare it against the current online herding on fresh seeds. The stored replay-image view must be controlled or explicitly factored in, so a change in selected IDs is distinguishable from a change in what the head later sees. Only after CIL/TIL gains and old/new-class guards are measured should participant-aware scoring be layered on; report memory, communication and privacy cost plus ISOLET/UPMC regression checks.

## Reproduction and integrity

Training root: `/home/c3080/YangXiaoXiang/VF-CL/results/cifar-selection-view-herding-seed49-20261008-v1/seed_49_training_only`. Frozen final checkpoint SHA-256 `82318034eddeb994dc34e78f9acd93abda74949bccebd39c74336de3238b5c96`; training-only completion manifest SHA-256 `f80bf02d3e947ec5908aab2cf3d87a8f969a06ae38fd9c0cd0bfec008f51acc5`. Stochastic and deterministic retained-ID list hashes are `44785ba82307082b82a5e73eaad91eef378ed8ac411c1541bfb2d551ba6116d5` and `573838b4d84d54609c6c06abe473d45cc5b812495d40d20d0b2e717dad064bc4`; the IDs are not exported.

The analyzer ran twice against the same frozen checkpoints in separate output roots (`cifar-selection-view-herding-seed49-20261008-analysis-v1` and `-v1-repeat`). Both JSON files are byte-identical, SHA-256 `c88c63542650f984917600daf4717b8860f379d16c881fa3c9a500c2e393e20f`. Executed analyzer SHA-256 `14ec4af1a53fed22171bc2333413a5075626c4c06e41db4d940337ee02805485`. Four focused launcher/analyzer tests passed locally and on the server; a read-only seed-48 event-checkpoint embedding preflight also passed. The producer checkout remained clean.
