# 四方均衡原始样本选择实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在冻结开发主干上验证四方均衡 herding 是否改善仅样本版最终头。

**Architecture:** 复用原 `herding_indices`，只替换其输入特征；从每个任务边界重建当前任务候选集，先验证原样本集合可复现，再比较两组固定 20/类样本的最终头。

**Tech Stack:** Python、PyTorch、现有 VF-CL checkpoint、开发训练集留出。

## Global Constraints

- 只用 ISOLET/UPMC 54/55 训练和留出；正式 42/43/44、三套正式结果只读。
- 历史任务 `t` 的候选只能来自任务 `t` 的训练加载器，不可在最终时访问过去完整训练集来重新选择。
- 样本数恒为 20/类；额外特征只作临时计算，不写入学习记忆。
- 原 herding 样本及最终头若无法复现，停止；四项开发 CIL 不全提升也停止。

---

### Task 1: 选择特征与历史样本复现

**Files:** Create `party_balanced_herding_dev.py`; test `test_party_balanced_herding_dev.py`。

- [ ] 先写失败测试：每方单独归一化后尺度相等；输出行数与输入一致；无效方数/行数被拒绝。
- [ ] 运行 `python -m unittest -v test_party_balanced_herding_dev`，确认缺少实现的失败。
- [ ] 实现最小 `party_balanced_features(list[Tensor]) -> Tensor`，复用原 `herding_indices`。
- [ ] 运行测试；从保存的 ISOLET54 历史检查点与该任务训练加载器重建原样本，检查它与冻结 `head_raw_replay` 的集合重合度和每类数量。

### Task 2: 配对冻结头对照

**Files:** Create `run_party_balanced_herding_dev.py`; modify `docs/superpowers/reports/2026-10-19-sampleonly-diagnostic.md`。

- [ ] 在同一历史任务顺序生成原融合 herding 与四方均衡 herding 的原始样本；原集合必须逐类复现冻结作业。
- [ ] 复用 `fit_adaptive_candidates`、原验证嵌入与 `solve_global_mixture_weight`；原选择先重建原门控及最终头读数，再评估新选择。两组都只用原训练期验证样本拟合门控。
- [ ] 运行 ISOLET/UPMC 54/55；输出新根 JSON、检查点/代码/清单哈希和逐项 CIL；严格按门槛继续或停止。
- [ ] 测试、复核、提交并推送开发分支；不修改旧正式结果。
