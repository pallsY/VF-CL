# 仅样本版开发留出诊断实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用现有 ISOLET、四方 UPMC 的种子 54/55 开发作业，量化最终头安装前后的留出表现，为下一项单因素创新消融选定位置。

**Architecture:** 一个只读诊断入口复用 `run_resource_lite_pilot.py` 的 `PilotDataset`、`score_holdout_checkpoint` 与哈希函数。它在临时目录重建留出清单，从已冻结的两份检查点读取预测，输出新的诊断 JSON；原作业目录不写入任何文件。

**Tech Stack:** Python 3、PyTorch、标准库、现有 VF-CL 数据集和检查点加载代码。

## Global Constraints

- 正式测试种子 42/43/44、旧审计和所有旧训练目录只读。
- 诊断仅用 `resource-lite-dev-full-v2-20261018` 的 `reference20` 54/55，结果标记为训练集留出。
- 运行前后都核验配置、最终检查点和留出清单哈希；禁止创建测试加载器。
- 输出另置 `sampleonly-diagnostic-dev-v1-20261019`；不把开发指标填入正式比较表。

---

### Task 1: 只读前后分类头诊断

**Files:**
- Create: `diagnose_sampleonly_head.py`
- Test: `test_diagnose_sampleonly_head.py`

**Interfaces:**
- Consumes: `run_resource_lite_pilot.PilotDataset`, `score_holdout_checkpoint`, `sha256`；每个作业的 `config.json`、`HOLDOUT_READOUT.json`、`pilot_holdout/manifest.json` 和检查点。
- Produces: `diagnose_run(run: Path) -> dict`，含 dataset/seed、原始哈希、`pre_head`、`post_head` 和两个指标的差值。

- [ ] **Step 1: 编写失败测试。** 构造临时作业，模拟评分函数；断言修改清单或最终检查点时 `diagnose_run` 抛出异常。再断言作业若含 `results.json` 也拒绝处理。
- [ ] **Step 2: 运行测试并确认失败。** `python -m unittest -v test_diagnose_sampleonly_head`；预期因诊断入口不存在而失败。
- [ ] **Step 3: 最小实现。** 读取现有读数与配置，按 `sha256(run / 'config.json')`、`sha256(final_path)`、`sha256(run / 'pilot_holdout/manifest.json')` 对比三个声明哈希；创建临时 `PilotDataset` 后再次核对重建清单。按 `event_{num_tasks-1}_CIL.pt` 和 `formal_final.pt` 调用现有 `score_holdout_checkpoint`。检查后者的 CIL 与既有读数一致，再只输出开发留出摘要。
- [ ] **Step 4: 运行测试。** `python -m unittest -v test_diagnose_sampleonly_head`；预期全部通过。
- [ ] **Step 5: 提交。** `git add diagnose_sampleonly_head.py test_diagnose_sampleonly_head.py && git commit -m "feat: compare sample-only dev head checkpoints"`。

### Task 2: 四个开发作业复算与判读

**Files:**
- Create: `docs/superpowers/reports/2026-10-19-sampleonly-diagnostic.md`

**Interfaces:**
- Consumes: Task 1 的诊断 JSON，以及四个开发作业既有 `HOLDOUT_READOUT.json`。
- Produces: 一张 ISOLET/UPMC × 54/55 的前后对照表，记录输出 JSON SHA-256、错误类别和下一步只改分类头或只改 PartyKD 的决定。

- [ ] **Step 1: 环境和作业门槛。** 复核 `/tmp/vfcl-resource-lite-run-v2` 干净、作业四份 `reference20` 均有 `FORMAL_EVALUATION_PENDING.json`，没有测试访问与 `results.json`。
- [ ] **Step 2: 独立运行。** 在与开发训练相容的 Python 环境中以 `OMP_NUM_THREADS=2` 执行 `diagnose_sampleonly_head.py`，将 JSON 写入全新的 `sampleonly-diagnostic-dev-v1-20261019`；每份作业只复算一次。
- [ ] **Step 3: 交叉核验。** 核对每份诊断的 `post_head.cil` 等于原 `HOLDOUT_READOUT.metrics.cil`、重建清单哈希一致、开发测试访问记录不存在，并记录输出 SHA-256。
- [ ] **Step 4: 报告决定。** 按设计文件的判读规则写入表格和下一项单因素消融方向；不宣称新方法已经产生。
- [ ] **Step 5: 提交。** 强制添加被 `docs/` 忽略的报告，提交并推送新的开发分支。
