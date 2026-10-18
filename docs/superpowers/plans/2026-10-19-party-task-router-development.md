# 四方任务证据路由开发实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不重训主干和不增加数据访问的条件下，测试四方任务证据能否稳定改善仅样本版跨任务识别。

**Architecture:** 一个纯张量模块拟合四个方级校正系数；一个只读运行入口复用已有开发留出与冻结检查点。验证集只拟合系数，留出集只评价；输出新 JSON 和完整来源哈希。

**Tech Stack:** Python 3、PyTorch、现有 VF-CL checkpoint/数据加载代码。

## Global Constraints

- 不读取正式测试、不修改三个冻结正式实验和既有 54/55 开发产物。
- 只使用最终头已访问的训练验证嵌入、相同四方模型、同一留出清单；固定 L2=0.01 和最多 80 次优化。
- 路由输入只有四方任务证据和当前头任务概率；禁止真实任务 ID 进入推理。
- 通过门槛是 ISOLET/UPMC 的四项开发 CIL 都优于原头且不低于通用任务偏置。未过门槛不启动 CIFAR 或新正式训练。

---

### Task 1: 纯四方任务校正

**Files:**
- Create: `party_task_router_dev.py`
- Test: `test_party_task_router_dev.py`

**Interfaces:**
- Consumes: `base_log_probs: Tensor[N,C]`、`party_logits: list[Tensor[N,C]]`、`task_classes: dict[int,list[int]]`、验证类别标签。
- Produces: `fit_router(...) -> dict`，含四个系数与四个标准差；`route(...) -> Tensor[N,C]`，保留任务内类别分布。

- [ ] 写玩具两任务、两方的失败测试：路由不得改变任务内两个类别的相对概率；交换留出标签不得改变拟合参数；输入维度错误时拒绝。
- [ ] 运行 `python -m unittest -v test_party_task_router_dev`，确认因模块缺失失败。
- [ ] 最小实现：任务 log-mass、方级 log-mean-exp 及中心化标准差；LBFGS 最多 80 次最小化验证任务交叉熵 + L2；用训练验证统计对留出路由。
- [ ] 重跑该测试并确认通过，提交纯逻辑。

### Task 2: 冻结开发作业评价

**Files:**
- Create: `run_party_task_router_dev.py`
- Modify: `docs/superpowers/reports/2026-10-19-sampleonly-diagnostic.md`

**Interfaces:**
- Consumes: `diagnose_sampleonly_head.validate_run_inputs`、`run_resource_lite_pilot.PilotDataset` 和四份 `reference20` 检查点。
- Produces: 每个开发作业一份新 JSON，含现有头和路由后 CIL/任务识别、验证目标、模型系数、来源哈希。

- [ ] 运行入口只读重建验证/留出证据；对全分类器分方 logits 核验四方相加等于融合 logits，对现有头的 CIL 核验等于冻结读数。
- [ ] 在 ISOLET/UPMC 的 54/55 四份作业各运行一次，并独立核对输出 SHA-256、无测试访问、共同规则与指标。
- [ ] 按预先门槛决定继续或停止：若不全过，将负结果和所需下一步写入报告；若全过，再以相同开发规则检查 CIFAR-100，不修改正式结果。
- [ ] 运行相关单元测试、检查差异、提交并推送开发分支。
