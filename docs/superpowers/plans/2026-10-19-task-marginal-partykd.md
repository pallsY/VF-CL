# 任务汇总四方 PartyKD 开发实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 验证训练期任务汇总方级权重是否比均匀 PartyKD 更准确。

**Architecture:** 扩展已验证的训练集留出驱动，只增加 `task_marginal` 模式；其余数据/训练/头协议从四份哈希锁定的 `reference20` 配置派生。新结果根与先前关闭 PartyKD 消融分开。

**Tech Stack:** Python、PyTorch、现有 VF-CL PartyKD 模式与开发留出驱动。

## Global Constraints

- 四份 `reference20` 源配置及其 SHA-256 沿用 `2026-10-19-sampleonly-component-ablation.md`，不得改动。
- 只改 `party_kd_mode`、`expected_party_kd_variant` 和作业身份；验证/测试数据访问、样本记忆和训练预算不变。
- 先两个数据集完整任务顺序、每任务一轮 CPU 预检；四个完整作业均更好才继续 CIFAR 开发，不访问旧正式测试。

---

### Task 1: 新模式合同

**Files:** Modify `run_sampleonly_partykd_ablation_dev.py`, `test_sampleonly_partykd_ablation_dev.py`。

- [ ] 先写失败测试，证明新模式保留 PartyKD 和依赖跟踪，仅将方级权重模式设为 `task_marginal`。
- [ ] 跑该测试并确认缺少模式参数而失败。
- [ ] 实现新派生分支，运行测试/语法检查，提交并推送代码。

### Task 2: 开发运行与停门

**Files:** Modify `docs/superpowers/reports/2026-10-19-sampleonly-diagnostic.md`。

- [ ] 旧结果根不存在检查、两数据集完整任务顺序 CPU 一轮预检、确定性环境与 GPU 空闲检查。缩短任务数的首轮预检失败目录须保留。
- [ ] ISOLET/UPMC 54/55 四项完整训练；逐项核对训练冻结、清单哈希、无测试访问和 CIL。
- [ ] 严格按预锚定“4/4 高于均匀”门槛决定继续或停止，把负结果和源码/结果哈希写入报告并推送。
