# Candidate A: prior-work check and development-grid decision

Status: **preliminary novelty screen, not a novelty claim or an accuracy result**.
The frozen 42/43/44 outputs remain untouched. No 70/71/72 accuracy run has
started. Candidate A uses old-class contribution *attenuation* only to schedule
which parties communicate on old-sample replay steps; it does not introduce a
contribution-matching regularizer. The single-process implementation counts
simulated messages. It does not establish actual network savings or privacy.

| Work | Setting and mechanism verified in primary source | Difference to candidate A | Boundary |
| --- | --- | --- | --- |
| [V-LETO](https://arxiv.org/html/2502.09152v2) | Vertical class/feature continual learning; evolving class prototypes and Fisher-based local parameter protection. | No old-class attenuation-based selection of parties specifically for replay communication. | Candidate inherits some V-LETO-style machinery; cannot claim Fisher protection or vertical CL itself as new. |
| [CMCDR](https://arxiv.org/html/2607.27260v2) | Multimodal continual learning; compares current and frozen old-sample modality contribution profiles and regularizes drift, including interactions. | Candidate borrows the contribution-drift idea, uses a cheaper full/leave-one-out attenuation probe, and changes *communication participation* rather than adding its drift loss. | The contribution estimator and old-sample drift concept are prior art. The narrower scheduling use must outperform matched random/static selection. CMCDR's main regularizer uses weight 1.0 in its paper. |
| [VF-PS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0e1a2388cd2f78069f4d048d935cb218-Abstract-Conference.html) | Selects important VFL participants using mutual-information estimation and group testing. | Candidate selects per old-replay step from changes in old-class contribution across tasks. | Party selection itself is prior art. |
| [VFLMG author code](https://github.com/Hjh233/VFLMG/tree/fc63084cc9cb20ff6232478714bdaa06af45bdae) | `Greedy.py` computes a joint-mutual-information participant ranking before `MultipleClient.py` trains with a fixed participant list throughout its epochs. | Candidate selects a new subset on each old-replay step using weights refreshed every two epochs; all parties still train on current-task batches. | Publisher method PDF was inaccessible, but this execution order is directly visible in the authors' code at `fc63084` and corroborated by VFL-RPS's method comparison. |
| [VFL-RPS](https://arxiv.org/html/2502.14375) | Relevant-party selection is explicitly a pre-training step; it includes VFLMG as a participant-selection baseline. | Candidate does not exclude parties from the full training run. | Pre-training selection is prior art; its reported accuracy/efficiency cannot be taken as a dynamic-replay baseline. |
| [Hounding Data Diversity](https://jszy.whu.edu.cn/_resources/group1/M00/00/FC/wKgCBWi0OBSAPEDLAAv1jdsMSd8223.pdf) | Valuation and selection identify a fixed number of participants to conduct VFL training. | Candidate changes replay-step participation, not consortium membership. | Its valuation/selection concepts are prior art. |
| [FedBCD](https://arxiv.org/html/1912.11187), [CELU-VFL](https://arxiv.org/html/2207.14628), [VFL-Cafe](https://www.mdpi.com/1099-4300/27/1/66) | Save VFL communication through multiple local updates and/or reuse of cached intermediate results; VFL-Cafe also selects features. | Candidate selects the senders of *old-replay* embeddings using a forgetting-related signal, without adding a persistent per-sample embedding cache. | These are meaningful communication baselines. Their cost/accuracy trade-off has not been measured against candidate A; local updates need not necessarily violate the raw-sample budget. |
| [LASER-VFL](https://arxiv.org/abs/2410.22564) | Trains with missing feature blocks and task-sampled party subsets for robustness. | Candidate chooses replay participants using old-class attenuation, while current-task steps retain all parties. | Masking/subset training is prior art; only the continual-replay scheduling signal and scope may be new. |
| [Good et al.](https://aclanthology.org/2023.emnlp-industry.32/) | Coordinated replay *sample* selection in continual federated learning with sample-partitioned clients. | Candidate schedules *feature-holding parties* in vertical replay, not samples in horizontal replay. | Federated coordination of replay is prior art. |

The primary sources and VFLMG authors' code inspected in this pass did not
show the exact combination of old-class attenuation, dynamic party
participation, and replay-only message scheduling. This is **not proof of
global novelty**. A reviewer may see a straightforward combination of CMCDR's
signal and existing party selection. The defensible claim requires clear
mechanistic differences and evidence that candidate A beats uniform random
and existing static-importance selection at the same replay budget. The
publisher text of VFLMG remains inaccessible; retain that source-access
limitation rather than claiming complete paper-level verification.

For VF-PS, VFL-RPS, Hounding Data Diversity, and the VFLMG author-code
workflow, selection establishes a participant set for subsequent training.
Candidate A keeps all four parties on every current-task step and on periodic
full-party replay steps. Only ordinary old-replay steps use a newly sampled
subset; their sampling weights are refreshed every two epochs from old-class
attenuation. This timing/scope distinction is concrete, but by itself does not
establish that the combination is non-obvious or empirically superior.

## Development alpha grid: choose option 1

Use the same predeclared `alpha in {0.25, 0.5, 1.0}` grid for **every** replay
arm: all-party replay, uniform random selection, static-importance selection,
candidate A, direct CMCDR transfer, and its compute-matched variant. Replay
batch size is fixed at 64. Keep CMCDR's own regularizer weight at its published
main-experiment value 1.0; `alpha` controls only the common replay loss.
The frozen no-replay method and the strongest paired dataset baselines have no
replay-alpha grid. Use only development seeds 70/71/72 and the locked training
holdout; official test access remains blocked.

For each replay arm **and each dataset**, choose its best alpha by the highest
mean final Class-IL accuracy over development seeds 70/71/72 on the locked
tuning holdout. Compare unrounded means; an exact tie takes the smaller alpha.
This rule is fixed before any development accuracy is read. Report every grid
cell, including negative cells. Also compare candidate A with uniform and
static selection at each *fixed alpha*, because only those matched-alpha
contrasts isolate the selection rule. Independently tuned best-of-grid scores
answer a different question: how strong each arm is when given equal tuning
opportunity. Dataset-specific selected alpha values must be disclosed.

With six replay arms, three alpha values, three datasets, and three seeds,
this is 162 replay jobs. If the frozen no-replay arm is paired on all nine
dataset-seed cells, and the four preselected strongest-baseline configurations
are run on their three seeds, the total is 183 jobs. This is an upper-level
development budget estimate, not permission to launch them: the novelty gate,
full job protocol and resource schedule must be reviewed first. In particular,
the candidate's three-dataset-first accuracy goal cannot be replaced by a
message-saving result.
