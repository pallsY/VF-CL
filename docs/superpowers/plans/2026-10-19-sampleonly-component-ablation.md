# 仅样本版 PartyKD 贡献消融实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 成对比较仅样本版启用/关闭 PartyKD 的开发留出结果，判断它能否支撑核心贡献。

**Architecture:** 新驱动从四份冻结 `reference20` 配置派生唯一变化为 PartyKD `base` 变体和输出路径的配置，复用原开发 `PilotDataset` 的测试访问拦截、完整训练与开发留出评估。独立队列分配四项到空闲 GPU 并 fail-stop。

**Tech Stack:** Python、PyTorch、现有 VF-CL 训练与数据代码。

## Global Constraints

- 四份参考配置 SHA-256 固定为 ISOLET54 `5775c71254a8f0f0eb41b420670fcbec5201acd071d059c0058355964f740cfe`、ISOLET55 `0282d27f205b258ba2ed53a51dc0b16702799544dcda34d4547433da4ac8fd58`、UPMC54 `1eeb07c79238cdcfa16a437cee14f1299254e238e2323a2655692ccd4dadb202`、UPMC55 `37562007542b87d45f075eeedbf3482462710afd8870a688e9ef89212c6a09a6`。
- 只用训练集留出，不创建测试加载器；原正式结果和旧开发作业只读。
- 新根缺失/作业已存在时拒绝重复启动；失败保留日志与检查点。

---

### Task 1: 派生配置和训练驱动

**Files:** Create `run_sampleonly_partykd_ablation_dev.py`; test `test_sampleonly_partykd_ablation_dev.py`。

- [ ] 先写失败测试：四份来源之外拒绝；派生配置只改 PartyKD base 三字段、variant/输出身份；输入哈希不符拒绝。
- [ ] 跑 `python -m unittest -v test_sampleonly_partykd_ablation_dev`，确认缺少入口而失败。
- [ ] 实现派生、原 `PilotDataset` 的训练期测试拦截和 `evaluate_holdout`；训练前核对源数据、四方和预算。
- [ ] 跑测试并提交。

### Task 2: 预检与四项完整开发训练

**Files:** Create `run_sampleonly_partykd_ablation_queue.py`; modify `docs/superpowers/reports/2026-10-19-sampleonly-diagnostic.md`。

- [ ] ISOLET/UPMC 两任务一轮 CPU 预检，仅检查数据流、检查点与禁止测试访问，不计入正式四项。
- [ ] 确认 GPU 空闲、训练提交干净、全新结果根、四份源配置与留出清单哈希一致后，启动一次带锁的 fail-stop 队列。
- [ ] 四项完整预算均生成 `FORMAL_EVALUATION_PENDING`、独立 `HOLDOUT_READOUT` 且没有测试访问；逐项核对同数据集/种子留出清单哈希与参考一致。
- [ ] 写成对比较和 PartyKD 贡献结论；未完成/失败则明确保留中间状态，不宣称结果。
