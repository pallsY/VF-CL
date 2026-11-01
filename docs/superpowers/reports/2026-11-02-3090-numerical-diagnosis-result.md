# 3090两任务FIM数值诊断结果（不代表正式准入）

## 结论

已完成用户单独确认后的914818f短预检和3080归约重放。诊断状态为 `diagnosis_supported`：原来5/8均值向量哈希不同，可以在同一份完整FIM张量输入下由跨主机GPU均值归约重现；差异未改变本次两任务的阈值、逐张量冻结选择或累积掩码。

此前严格逐位准入的 `rejected` 不变。本报告是独立、预先固定容差的数值诊断，不追溯改写准入失败，不宣称全任务、其他种子或消融已验证，也不授权3090加入正式队列。3090与3080的GPU和驱动同时不同，本诊断不能分离硬件和驱动各自的影响。

## 预先固定的范围与判据

范围是原base CIFAR seed42、每类前64个训练样本、每任务1epoch、保留BiC、仅task0/1。只做训练预检和FIM归约，真实测试授权与loader被拦截；重放不读取数据集、不训练。

在读取差异大小之前封存的判据为 `abs(candidate-reference) <= 1e-12 + 1e-6*abs(reference)`，所有冻结选择及累积mask完全相同。该工程容差不是统计等价性结论。协议SHA256 `de9b951795689416219ec2fb457a88a39a151fbb9eb90bb3e3d675cbbe0e78ae`。

## 核验结果

| 比较项 | 结果 |
| --- | --- |
| 完整FIM输入 | 480个张量，逐位一致且对应旧双方证据 |
| 原生GPU均值 | 6/480个标量不同 |
| 最大绝对差 | 9.094947017729282 × 10^-13 |
| 最大相对差 | 1.242515089103242 × 10^-7 |
| 超出先验容差 | 0 |
| quantile阈值 | 8/8完全相同 |
| 逐张量选择变化 | 0 |
| 累积mask变化 | 0；重建2个任务边界原始mask哈希 |
| 旧reference的均值向量/阈值哈希 | 8/8重建成功 |
| 新3090预检与旧candidate | states、losses、gradient_steps、fim_scores、fim_tensor_values、sample_orders、applied_freeze全部逐位一致 |
| 旧双方其余数值证据 | 模型、记忆、梯度、损失、RNG、样本顺序、完整FIM和mask均逐位一致 |
| 真实测试访问 | 无 |

3090驱动535.230.02，3080驱动580.178.04，均与各自旧记录相同。参考重放PyTorch峰值reserved为23,068,672字节，低于预锁定2048MiB allocator预算。两端仍执行原UUID、连续120秒显存门槛和启动前复核。

## 执行与封存

执行代码 `914818f5cac638cdb5b9cb1d13764b3bd906c7a6`。修复后的固定环境shell启动器在Python启动前设置CUBLAS_WORKSPACE_CONFIG、OMP_NUM_THREADS、MKL_NUM_THREADS及PYTHONHASHSEED；两端26项CPU测试和独立代码审查已通过。旧7e9014d启动失败及无训练证据均保留，详情见同目录 `2026-11-01-3090-numerical-startup-failure.md`。

NAS根 `/brf/users/yangxx/VFCL-admission/3090-20261101/numerical-diagnosis-v1`：

| 证据 | SHA256 |
| --- | --- |
| RETRY_EXECUTION_ANCHOR.json | 06d32560befea0de56ebfa81f369bd605cddd6e7d70d19421ed3a35962a80de8 |
| capture-914818f/SEALED.json | 0616fa456b24afe7cb5035edfd2658226a788ef477f269b28d8eb15b56703ab1 |
| capture-914818f/RAW_FIM.pt | 56c09de91cb910c356e1f2571754ed8f4072234dfe41859d4c25a71ad6de7325 |
| reference-replay-914818f/SEALED.json | 4ba276d2098e65f60070be5f298ec05141dbe94102f932cac8a7c1687b7889fa |
| reference-replay-914818f/DIAGNOSIS.json | 0052976fdbe020d45da97f91c395170be5580ba09383a7cbbca20897dbf4171e |
| NUMERICAL_RESULT_SUMMARY_914818f.json | 5403ee0188141f06f590833416160499bd77a21df80c037a4b606a29f4f5a647 |
| 旧严格准入EARLY_BASE_REJECTION.json（同级7c2cc36根） | 4cf4084d512783d9a82482c1b6c005487b605f1106b63263a7fb697bbfabb1a8 |

原始capture、参考归约、GPU资源记录、无测试凭据、训练状态哈希、失败记录及执行日志均保留。没有读取18项正式队列的准确率。

## 接入边界与下一步

本次结果解释了旧短预检的哈希差异；尚未验证全任务跨卡轨迹或其他profile，不能据此宣称3090已通过旧严格门槛。任何后续跨硬件正式消融须另行锁定同卡配对：拟转交的CIFAR seed43两个消融如果仍未开始，需同时安排3090上的原方法seed43对照，正式主结果表保持冻结，补充消融按同硬件对照计算差值并公开说明。

调度层仍无可直接启用的热接入路径。原调度器不支持动态drain/skip，不能另起第二个scheduler争抢同一清单。具体交接须满足原单写账本、未开始任务唯一分配、已运行进程不重训、精确退出状态和全局失败停机，并先实现、测试及独立审查；技术约束见 `2026-11-01-3090-handoff-feasibility.md`。数值诊断支持不替代这些条件或最终激活确认。

## 独立审计与正式队列只读快照

独立审计者 `/root/fim_numeric_review` 使用914818f代码完成CPU审计，干净退出码0，CUDA未初始化。核对两端seal完整清单（capture15个文件、replay3个文件）、代码哈希、原严格拒绝哈希、新旧candidate全部7类数值证据、旧双方非score证据、480个原始输入的值/形状/dtype/stride，以及8个score和2个mask历史哈希；独立复算488行比较，最大差异、6个不同均值、0超界/选择/mask变化均与封存报告一致。未训练、重放、写入原证据、读取数据集或读取正式准确率。该审计不独立认证实时队列状态，后者由主执行者另行检查。

主执行者末次快照UTC2026-11-01 16:13:15（上海2026-11-02 00:13:15）：正式调度PID3003183存活，原workers3003425/3003428存活；账本仍只有2个start、没有finish，根目录只有QUEUE_LOCK，无QUEUE_FAILURE/COMPLETE。manifest SHA256仍为 `bb10fa6c62719b68124e7b625a204defc6b3e396ac285c068849fff8866e8fca`，账本SHA256仍为 `51078862e8dd4a3de881aa932eb0afeee1094a03e0855c9e9f643774cf7c45e2`，均与执行前一致。两项诊断进程已经退出。没有交接、正式3090 worker或重复启动。

后续硬件配对草案见 `../specs/2026-11-02-3090-hardware-paired-ablation-draft.md`，仅供审查，尚不授权开跑。
