# Adaptive20 双因子回放消融执行计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 在三数据集的独立训练留出集上，配对测量原型回放损失梯度 P 与原始样本支持的最终分类头 R 的条件贡献，正式测试集零访问。

**Architecture:** 在 `ProtoEvolveCL` 增加默认开启、只控制原型项梯度的显式标志；R 使用已有 `head_consolidation_enabled` 最终整合开关。通用训练专用启动器从已锁定的 Adaptive20 配置派生四组，排除独立留出集、冻结最终状态、一次性计算留出集指标并发布审计文件。先验证 ISOLET，再顺序扩展 UPMC 与 CIFAR-100。

**Tech Stack:** Python 标准库、PyTorch、现有 `VFLDataset`/`runner`/`calibration_split`，Windows 开发工作树，两台已有 SSH GPU 主机。

## 全局约束

- 主预算 20 个旧原始样本/类；固定 `proto_lambda_a=0.05`、`distill_weight=0.10`、`feat_distill_weight=0.02`、任务顺序、训练轮数和相同种子。
- 四组和含义以[设计](../specs/2026-10-12-adaptive20-two-factor-replay-ablation-design.md)为准；P0 是去掉原型项梯度，R0 是去掉原始样本及最终头整合，不称“纯样本/纯特征”。
- 开发种子为 54/55；每个数据集独立留出集与训练、门控验证、BiC 校准不重叠；零正式测试 loader 访问。
- 使用 formal-deferred 训练路径并在最终测试授权前停止，保留其强制生成的每任务检查点和冻结标记；开发冻结标记不是正式发布。任何失败用新结果根安全恢复，不覆盖已完成读数或正式发布结果。
- CIFAR 先核对磁盘空间；空间不足时不启动。GPU 等待不得抢占外部任务。
- 每次局部行为修改先写失败测试，再实现最小改动；现有默认 Adaptive20/40 输出不得变化。

---

### Task 1: 原型损失梯度开关与协议绑定

**Files:**
- Modify: `cl_methods/proto_evolve.py`
- Modify: `config.py`
- Modify: `runner.py`
- Create: `test_replay_factorial_loss.py`

**Interfaces:**
- Config: `proto_replay_gradient_enabled` 为 0/1，默认 1。
- `ProtoEvolveCL._proto_loss` 在 P0 仍执行相同原型抽样，普通 CE 系数保持 `1−proto_lambda_a`，原型 CE 的梯度贡献为零，KD/其他项不变。
- `runner._checkpoint_protocol` 仅为关闭状态增加显式身份键，旧默认协议不变。

- [ ] 写测试：相同初始模型与随机数状态下 P1/P0 原型抽样相同；P0 梯度等于 `(1−0.05)×CE`，P1 多出原型梯度；P0 仍有 0.95 CE 系数。
- [ ] 运行 `python -m unittest -q test_replay_factorial_loss`，确认因标志未实现而失败。
- [ ] 实现默认开启开关与协议身份键，不改默认路径。
- [ ] 重跑该测试与 `python -m unittest -q test_fixed40_formal_controls test_adaptive_protocol_config`，确认通过。

### Task 2: 四组配置、独立留出集与训练专用读数

**Files:**
- Create: `run_replay_factorial_pilot.py`
- Create: `test_replay_factorial_pilot.py`

**Interfaces:**
- `derive_config(source, root, dataset, seed, arm)` 只允许种子、两个因子及路径/训练专用标志改变；P0 用显式梯度标志，R0 同时设置 `head_consolidation_enabled=0`、`head_consolidation_mode=full_classifier`。
- `PilotDataset` 用现有 `build_manifest` 创建每类平衡、与门控/校准集合不交叠的独立留出集，并从所有训练 loader 中剔除。
- `run_pilot` 在最终检查点冻结后拦截正式测试，验证数据流无 test 记录，加载冻结模型并输出 holdout Class-IL、Task-IL、按任务混淆、原始回放和原型字节数、时间及哈希。
- 两个 R 组在同一 P 条件下，最终头整合前的训练状态/底部网络哈希必须相等；分析器不接受缺少配对证据的结果。

- [ ] 写单元与小型合成端到端测试：四组仅有允许的配置差异，holdout 排除有效，R0 不存 raw、R1 保留20/class，拦截发生在正式测试前。
- [ ] 运行 `python -m unittest -q test_replay_factorial_pilot`，确认预期失败。
- [ ] 复用现有 `calibration_split`、`VFLDataset` 与 `score_branch` 实现最小启动器；无新依赖。
- [ ] 运行 focused tests 与现有 220 项 Python 测试；确认默认方法结果/协议不变。

### Task 3: 冻结运行矩阵并做训练专用试跑

**Files:**
- Create: `docs/superpowers/reports/data/adaptive20-replay-factorial-configs-20261012/`
- Create: `docs/superpowers/reports/data/adaptive20-replay-factorial-manifest-20261012.json`

**Interfaces:**
- 每组记录源码提交、配置 SHA-256、数据与 holdout 清单哈希、seed、物理 GPU、允许变更字段。
- 3080 顺序运行 ISOLET，4090 等待 GPU 空闲后运行 UPMC 与 CIFAR；启动前检查最低磁盘/GPU 空闲量。

- [ ] 先执行 ISOLET seed54 四组训练专用 smoke，核对冻结、无 test、原型开关、raw20/0 与 R 配对状态。
- [ ] 通过后锁定全部三数据集 seed54/55 配置；若样本不足或清单重叠则在启动前失败。
- [ ] 顺序运行剩余 20 次训练专用实验；每个结果有退出码、冻结检查点、holdout 读数及哈希。CIFAR 每组可能占约 4 GiB，先确认完整矩阵所需空间；空间不足时只运行已容纳的成对组并报告剩余工作，不删除正式结果。健康后台运行时由安静跟进监控。
- [ ] 对失败运行先诊断日志与冻结状态，不重复完成的读数；磁盘不足时暂停新任务而不删除正式结果。

### Task 4: 汇总与研究决策

**Files:**
- Create: `docs/superpowers/reports/data/adaptive20-replay-factorial-pilot-20261012.json`
- Create: `docs/superpowers/reports/2026-10-12-adaptive20-replay-factorial-pilot.md`

- [ ] 验证 24 次读数与冻结/配置/数据哈希、零正式测试访问、P 条件内 R1/R0 配对状态；输出逐种子和两种子均值、四项条件差、交互项及实际内存。
- [ ] 明确报告该试验是开发集证据，不是正式 `AA_final` 或论文显著性结论；若某配对状态不相等则标无效并诊断。
- [ ] 运行报告和台账一致性校验、`git diff --check`，提交推送 `codex/replay-component-ablation`。
- [ ] 若用户需要正式论文消融，依据冻结定义另写正式协议并在新结果根执行，不复用开发读数冒充正式测试。
