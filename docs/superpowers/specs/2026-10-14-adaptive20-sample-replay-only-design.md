# Adaptive20 仅原始样本回放版本设计

日期：2026-10-14。目标是在已完成的 Adaptive20 P/R 开发消融之外，增加一个**回放记忆仅为原始样本**的 VF-CL 变体，用相同开发留出协议读取结果。原 24 组结果和正式测试结果保持只读。

## 名称和边界

“仅样本回放”描述回放数据的类型：每类最多 20 个原始样本。各方特征提取器仍需对自己的纵向分片编码，最终头仍用原始样本经当前提取器重新编码后的表示做整合；这些即时表示不作为旧特征缓存。保留 logit/party 蒸馏、FIM、BiC、依赖统计及原 Adaptive20 最终头门控，它们是明确列出的非回放机制。因此不能声称方法的全部抗遗忘收益只来自样本回放，也不能把它叫标准 ER。

## 训练行为

新配置身份 `sample_replay_only=1`，默认 0，不改变原方法。该身份要求 `cl_method=proto_evolve`、`head_consolidation_enabled=1`、`head_consolidation_mode=adaptive_dual_branch`、`head_consolidation_samples_per_class=20`、`proto_replay_gradient_enabled=0`、`proto_sdc=false`、`feat_distill_weight=0`，并禁止原型相关遥测。训练任务仍只读当前任务训练集；旧原始样本不混入每个批次，而是在最终头整合阶段使用。关闭原型缓存后，任务交叉熵使用完整系数 1.0；旧模型的 logit/party 蒸馏及 FIM 路径保持原设定。

每个任务仍根据当前编码表示按既有 herding 规则选取并持久保留最多 20/class **原始样本**。允许瞬时计算当前表示用于选择，但检查点中的 `global_protos`、`prev_protos` 和 SDC 锚均为空，训练中不采样旧原型、不施加原型损失或特征蒸馏。最终双分支头从原始样本缓存确定保留类别，将其重新编码为临时输入；Full 与 Bias 候选均只用该批原始样本支持的数据拟合，验证门控沿用相同独立验证集。现有 `fit_adaptive_candidates` 在提供 raw replay embeddings 时不使用原型值；传入空原型映射以显式固定来源。

## 审计与实验

正式 checkpoint 协议记录 `sample_replay_only=1`，防止与原 P0R1 状态混用；运行读数单列变体身份，不复用或重命名原 P0R1。用原训练专用开发留出方法，锁定 ISOLET、UPMC、CIFAR-100 的 seed54/55、任务顺序、每类 20 样本、数据划分及开发留出清单；每组在正式测试授权前停止，核对冻结状态、零正式测试访问、原型字节数 0、原始样本 20/class、配置/数据/源码/结果 SHA-256。报告 Class-IL、Task-IL、缓存内存与训练时间，逐种子并列原 P0R1 作为机制不同的参考，不将其当成同一结果，也不称正式 `AA_final`。

先用单元和小型训练专用试跑验证无原型状态、无原型/特征回放梯度、头只取 raw 缓存、默认旧配置行为不变。正式开发矩阵再按主机空间和 GPU 可用性顺序运行；每组完整备份并逐文件校验后才清理远端大文件。CIFAR 使用持久磁盘，避开此前反复被清空的 3080 `/dev/shm`。
