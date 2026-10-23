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
| [VFLMG author code](https://github.com/Hjh233/VFLMG) | Adaptive VFL participant selection using a greedy joint-mutual-information calculation. | Candidate's signal is longitudinal old-class attenuation and scope is incremental replay messages. | Publisher method PDF was inaccessible in this screen; the difference needs final paper-level confirmation. |
| [FedBCD](https://arxiv.org/html/1912.11187), [CELU-VFL](https://arxiv.org/html/2207.14628), [VFL-Cafe](https://www.mdpi.com/1099-4300/27/1/66) | Save VFL communication through multiple local updates and/or reuse of cached intermediate results; VFL-Cafe also selects features. | Candidate selects the senders of *old-replay* embeddings using a forgetting-related signal, without adding a persistent per-sample embedding cache. | These are meaningful communication baselines. Their cost/accuracy trade-off has not been measured against candidate A; local updates need not necessarily violate the raw-sample budget. |
| [LASER-VFL](https://arxiv.org/abs/2410.22564) | Trains with missing feature blocks and task-sampled party subsets for robustness. | Candidate chooses replay participants using old-class attenuation, while current-task steps retain all parties. | Masking/subset training is prior art; only the continual-replay scheduling signal and scope may be new. |
| [Good et al.](https://aclanthology.org/2023.emnlp-industry.32/) | Coordinated replay *sample* selection in continual federated learning with sample-partitioned clients. | Candidate schedules *feature-holding parties* in vertical replay, not samples in horizontal replay. | Federated coordination of replay is prior art. |

The primary sources inspected in this pass did not show the exact combination
of old-class attenuation, dynamic party participation, and replay-only message
scheduling. This is **not proof of global novelty**. Before calling it an
innovation, check VFLMG's complete method and other recent vertical continual
learning and party-selection papers; then require candidate A to beat uniform
random and existing static-importance selection at the same replay budget.

## Development alpha grid: choose option 1

Use the same predeclared `alpha in {0.25, 0.5, 1.0}` grid for **every** replay
arm: all-party replay, uniform random selection, static-importance selection,
candidate A, direct CMCDR transfer, and its compute-matched variant. Replay
batch size is fixed at 64. Keep CMCDR's own regularizer weight at its published
main-experiment value 1.0; `alpha` controls only the common replay loss.
The frozen no-replay method and the strongest paired dataset baselines have no
replay-alpha grid. Use only development seeds 70/71/72 and the locked training
holdout; official test access remains blocked.

For each arm, choose one alpha **shared across the three datasets** using its
three-seed holdout means: maximize the smallest paired difference to the
preselected strongest dataset baseline; break ties by larger mean of the three
paired differences and then smaller alpha. Report every grid cell, including
negative cells. Also compare candidate A with uniform and static selection at
each *fixed alpha*, because only those matched-alpha contrasts isolate the
selection rule. Independently tuned best-of-grid scores answer a different
question: how strong each arm is when given equal tuning opportunity.

With six replay arms, three alpha values, three datasets, and three seeds,
this is 162 replay jobs. If the frozen no-replay arm is paired on all nine
dataset-seed cells, and the four preselected strongest-baseline configurations
are run on their three seeds, the total is 183 jobs. This is an upper-level
development budget estimate, not permission to launch them: the novelty gate,
full job protocol and resource schedule must be reviewed first. In particular,
the candidate's three-dataset-first accuracy goal cannot be replaced by a
message-saving result.
