# 仅样本版 FIM 冻结贡献消融实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用成对开发作业判断 FIM 参数冻结是否改善仅样本版。

**Architecture:** 在新开发提交里使 `fim_freeze_frac=0` 确实不冻结参数，同时保留原 FIM 统计；从四份批准的 `reference20` 配置派生零冻结作业，并在独立服务器 checkout/结果根运行。

**Tech Stack:** Python、PyTorch、现有 VF-CL 训练和开发留出代码。

## Global Constraints

- 原 Replay11、原 adaptive、仅样本版 42/43/44 正式源码与结果完全冻结。
- 源配置 SHA-256 沿用 `2026-10-19-sampleonly-component-ablation.md` 的四份固定值。
- 除 `fim_freeze_frac=0` 和输出身份外，不能改变模型、记忆、PartyKD、epoch、验证或任务顺序。
- GPU 由其他用户占用时等待；失败原件保留，不重复启动。

---

### Task 1: 零冻结语义

**Files:** Modify `cl_methods/proto_evolve.py`; create `test_fim_zero_freeze.py`。

- [ ] 写失败测试：两方小模型执行 `_compute_fim` 时，比例 0 不能标记任何底层参数。
- [ ] 运行测试，确认原实现仍标记至少一个参数。
- [ ] 最小修正零值分支并拒绝超出 [0,1] 的比例；重新运行测试及相关现有测试。
- [ ] 提交并推送新开发提交，不更改旧生产/审计分支。

### Task 2: 四作业派生驱动与预检

**Files:** Modify `run_sampleonly_partykd_ablation_dev.py`, `test_sampleonly_partykd_ablation_dev.py`。

- [ ] 写失败测试，要求 `fim_off` 仅改变冻结比例和新作业身份；`party_kd_mode` 与 `expected_party_kd_variant` 保持 `uniform`。
- [ ] 实现并运行测试；在 brf3080 用新干净 checkout 做 ISOLET/UPMC 完整任务一轮 CPU 预检。
- [ ] 预检逐任务确认冻结掩码全空、无测试加载器、同一留出清单；不合格即停止。

### Task 3: 全预算开发归因

**Files:** Modify `docs/superpowers/reports/2026-10-19-sampleonly-diagnostic.md`。

- [ ] GPU 空闲时按 54/55 启动四项独立作业；只读监控，不抢占，不重启已有作业。
- [ ] 核对四份读数、源配置差异、留出清单哈希、训练冻结标记与测试访问记录。
- [ ] 写成对 CIL/Task-IL、旧新任务结果和归因边界；测试、提交并推送报告。
