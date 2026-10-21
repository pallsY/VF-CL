# 仅样本版训练期旧样本回放 A/B 开发协议（2026-10-21）

**状态：待提交；提交后即锁定。** 门槛与两组定义在任何开发训练前固定，锁定后不得修改。

**目的。** 只检验一个原因：旧样本在增量训练期间参与分类损失，能否提高冻结仅样本方法在 ISOLET 上的训练留出 Class-IL。这是归因实验，不是创新点验证；普通回放已有 ER/iCaRL 先例，B 胜过 A 不构成新机制。本协议不跑 iCaRL，结论只能写成"回放是否帮助了我们的方法"，不能写成"已追上 iCaRL"。

**不参与本次 A/B：** 正式 42/43/44 结果；CIFAR 三臂消融；仅旧类 KD、真零冻结 FIM 等开发结果与代码；分方本地约束设计。

## 代码基线

- 分支 `codex/sampleonly-train-replay-ab-v1`，直接基于冻结提交 `e2a559d`。不合入 `5b08c9d`（仅旧类 KD）、`7a6974f`（`fim_freeze_frac=0` 真不冻结参数；`f2dd8c1` 只是记录此事的文档提交）及其后的任何方法改动。
- 方法代码（`cl_methods/`、`vfl_trainer.py`、`models.py`、`config.py`、`data_utils.py`、`runner.py`）相对 `e2a559d` 只允许一类改动：由新开关 `train_replay_enabled`（默认 0）守护的回放分支及其计数器。`train_replay_enabled=0` 时任何代码路径、随机数消耗与 `e2a559d` 相同。
- 开发基础设施（训练留出、测试阻断、留出评分）从后续提交原样引入为独立文件，主要是 `run_resource_lite_pilot.py` 中的 `PilotDataset`、`StopBeforeTest`、`validate_training_only_artifacts`、`evaluate_holdout`。若引入时发现它依赖后续提交的方法改动，停止并单独报告，不顺带合入方法改动。

## 两组定义

**A＝冻结方法。** `e2a559d` 的仅样本 20 + 均匀 PartyKD + 最终双分支头，`train_replay_enabled=0`。全宽 KD（`old_logits[:, :n]`）；`fim_freeze_frac=0` 按冻结代码的实际语义，每方仍冻结 FIM 最大张量并逐任务累积。

**B＝A + 训练期旧样本 CE 回放。** 唯一差别是 `train_replay_enabled=1`。任务 t≥1 的每个当前任务批次，额外抽恰好一个大小为 64 的旧样本批次：

```
L_B = CE_new(当前批次) + 1.0 · CE_replay(旧样本批次) + 原有全宽 KD + 原有 PartyKD
```

- `CE_new` 沿用冻结配置；`CE_replay` 对顶层模型的完整输出头（从一开始固定为 `num_classes` 宽，`models.py:658`）求交叉熵，不加已见类掩码，与仓库 ER 实现（`cl_methods/er.py:106`）一致。已见类掩码是另一种回放损失，本协议不用。α=1 固定。
- KD、PartyKD、`_track_dependencies` 只看当前任务批次；旧样本不进入这三项。
- 旧样本批次走与 ER 相同的联合前向（拆分到四方底层模型、求和聚合、顶层模型）。回放前向时底层与顶层固定为 train 模式。ISOLET、UPMC 的 `MLPBottom` 没有 BN 和 dropout，此项目前不改变计算，仍逐作业记录模式，供以后扩展到含 BN 的主干时核查。
- FIM 冻结张量 `requires_grad=False`，回放梯度不会更新它们；任务结束后的 FIM 统计仍只用当前任务训练数据。
- 仅样本模式下 `global_protos` 为空，`(1-λ_A)` 缩放不生效，两组 `CE_new` 系数都是 1。

**共同不变项。** 四方数据与主干、任务顺序、batch、每任务 epoch（ISOLET 50、UPMC 20）、优化器、每类 20 个原始样本预算与 herding 规则、最终双分支头、验证访问。回放只读取已有的 `head_raw_replay`，不新增存储。任务 0 两组模型相同、保存样本相同；任务 1 起 B 的模型不同，herding 选出的样本 ID 可随之变化，这是回放的直接后果，不视为混杂。

**轮转抽样规则。** 设当前任务开始时旧类为 K 个，按类 ID 升序。

1. 每类配额 `64 // K`；余数 `r = 64 mod K` 个名额按类轮转：第 s 步从游标 `(s·r) mod K` 起，依次给 r 个类各加 1。每步恰好 64 个，长期按类均衡。
2. 类内对已保存样本无放回循环：每类一个随机排列，取完重新打乱。`64/K > 20` 时批内会重复，允许，计数记录。
3. 随机性只来自独立生成器 `derive_seed(seed, 'train_replay', task_id)`，不消耗方法、全局或数据加载器 RNG；A、B 的当前任务批次顺序与增强相同。

## 关口 1：锁定协议

本文件在本地提交后即锁定，提交哈希记为协议锚点。通过条件：两组定义、抽样规则、门槛、种子与结果根均已写明，且提交成功。进入关口 4 前，该提交必须已推送或另行独立锚定（哈希与时间有运行前的外部记录）；每个开发作业的配置与读数都写入协议提交哈希。

## 关口 2：A 的身份核对（CPU）

用两个独立代码副本、两个独立进程：参照为停在 `e2a559d` 且无本地改动的工作树 `adaptive-sample-only-formal-v1`（只读）；对照为新分支，`train_replay_enabled=0`。比较脚本在两个进程中分别运行，并记录被加载模块的 `__file__` 路径，证明没有比较同一份已加载代码。两进程使用相同的确定性设置（`determinism.py`）、torch 线程数与环境。

使用 ISOLET 与 UPMC 的 `reference20` 派生配置、同一种子、同一数据，至少运行任务 0 和第一个增量任务（任务 1），每任务 1 轮。逐位比较：

1. 每步总损失及各分项（CE、全宽 KD、PartyKD）；
2. 每步反向后全部底层与顶层参数梯度，包括因 FIM 冻结而为空的梯度；
3. 任务 0 结束后计算、任务 1 开始时应用的 FIM 冻结掩码，以及任务 1 结束后的掩码；
4. 每个任务结束后的模型 state_dict 与 `head_raw_replay`。

另在新分支上比较 A 与 B（`train_replay_enabled=1`）任务 0 结束时的检查点张量与保存样本，必须逐位相等。方法代码相对 `e2a559d` 的 diff 只能出现开关守护的新增块。

**任何一项不一致即停止修复，不进入关口 3。**

## 关口 3：预检（种子 63，完整任务序列，每任务 1 轮，CPU）

预检结果不计入准确率判断。

1. **抽样**：回放步数 = Σ_{t≥1} 批次数 × epoch；读取量 = 64 × 回放步数；每步各类名额符合轮转规则；记录批内重复数。
2. **留出隔离**：以索引为准，所有训练加载器（含 herding 所用加载器）的索引与留出索引交集为空。另将保存的回放张量与留出样本逐行比对作为辅助检查；若出现相同行，必须核实它是源数据中不同索引的重复样本，否则判为失败。
3. **测试访问硬阻断**：入口复用 `PilotDataset`。请求正式测试时 `authorize_formal_access(split='test')` 抛 `StopBeforeTest` 立即停止，`get_test_loader` 断言失败，不创建测试加载器；未收到 `StopBeforeTest` 视为失败。结束后 `validate_training_only_artifacts` 通过：存在 `FORMAL_STATE_FROZEN.json`、`FORMAL_EVALUATION_PENDING.json`、`checkpoints/formal_final.pt`；不存在 `FORMAL_EVALUATION_CONSUMING/PUBLISHED/COMPLETE.json` 与 `results.json`；`data_flow_audit.jsonl` 中无 `split=test` 或 `('test'` 前缀加载器。
4. **资源计数**：额外前向/反向次数、墙钟时间、进程峰值内存（CPU 预检不记录显存）；四方通信按每回放步每方上传 64×d_k 嵌入、接收同尺寸梯度，记录公式值与实测字节，A 为 0。

**任一项失败即停止，保留原件，修复后重做关口 2 与关口 3。**

## 关口 4：配对开发实验

- 前提：关口 1 的协议提交已推送或独立锚定。
- ISOLET、四方 UPMC × 开发种子 60/61/62 × A/B，共 12 项全预算 GPU 作业；每项记录峰值显存（`torch.cuda.max_memory_allocated`）与额外计算。
- 新结果根 `sampleonly-train-replay-ab-dev-v1-20261021`，不复用任何旧作业目录。
- 训练留出种子 `pilot_holdout_seed=20261021`，与 54/55 留出、CIFAR 三臂的 20261020 均不同；同一种子下 A、B 的留出与验证清单哈希必须相同。
- 配置从已批准的 `reference20` 配置派生，只改种子、留出种子、作业身份与 `train_replay_enabled`；来源哈希写入作业目录。
- GPU 被他人占用时等待，不抢占。作业失败保留原件并停止，不静默重跑。
- 只读训练留出；报告逐种子最终 Class-IL 的 B−A 与资源差值（存储、读取、计算、四方通信）。

## 判定（已锁定）

- **ISOLET**：60/61/62 三个种子 B−A 均 > 0，且均值 ≥ 0.5 个百分点。
- **UPMC**：每个种子 B−A ≥ −0.5 个百分点。

两条同时满足，记为"训练期回放对现有方法有帮助"，之后才讨论分方本地约束能否以更少回放通信保持效果。任一条不满足，记录负结果并停止这套回放方案；不在同一留出上调 α、回放批次大小或抽样规则，不追加种子。
