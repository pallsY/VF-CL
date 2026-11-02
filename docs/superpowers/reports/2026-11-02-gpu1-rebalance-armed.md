# GPU1同机重分配：监视器已启用，等待空闲点

## 实际状态与后续行为

用户已请求把尚未开始的任务分给空闲GPU1。已部署固定版本 `57af7ce701f1b5823347f00027bae620c77fc8ce` 的空闲点切换监视器，实际Python PID3268813；独立session的conda启动wrapper PID3268792。监视器已创建ARMED记录，尚未创建EXECUTION_EPOCH，GPU1尚未启动新正式任务。

上海2026-11-02 11:00左右只读检查：原控制器PID3003183及GPU0的 `cifar100/no_partykd/42` worker PID3216187仍在正常运行；10/18成功、1运行、7未开始、无失败。原manifest、配置、历史ledger和已完成结果没有改写；未读取或汇总正式准确率。3090未参与。

监视器等待所有已启动任务成功结算、旧控制器没有任何训练子进程、稳定快照通过之后才退役旧父调度器；不暂停/迁移/重训活worker。若届时CIFAR no-PartyKD43仍未开始，GPU1负责它，GPU0并行处理ISOLET6项；若43已被原调度器启动，则只对剩余ISOLET按固定顺序交替分给两卡。候选只能缩小，不能重复启动；不足两项并行工作则保持原队列。

每张卡仍串行一项。GPU0 free>=3072MiB、GPU1>=5120MiB连续120秒并在启动前复核，允许外部共享。GPU配置字节不变，通过显式两卡CUDA_VISIBLE_DEVICES顺序把逻辑cuda:0映射到实际物理卡；JOB_RUNTIME与扩展记录实际UUID、控制器和worker身份。

## 验证

- 最终版本79项Linux合成测试全部通过、无跳过，包含无活child才退役、PID复用防护、旧前缀不变、半行/未知目录拒绝、单attempt、READY/GO原子发布、父死亡watchdog、活/死worker及loader清理、同伴失败停止后续启动、实际producer publication/seal和默认/扩展审计。
- 实际旧清单18项的训练代码、原配置、数据SHA、runtime、preflight、原重启锚点及外部旧PID身份全部只读核验通过。
- 实际PyTorch两卡反向UUID顺序核验通过；两卡均10GB RTX3080、compute8.6、driver580.178.04。兼容了目标Python缺少os.pidfd_open而libc支持的情况，以及torch UUID字符串没有GPU-前缀的格式。
- 在同一NAS的独立准入目录验证了fsync临时文件→独占hardlink最终文件→拒绝重复发布，正式READY/GO使用该原子路径。
- 独立审查者 `/root/gpu1_rebalance_review` 对最终版本审查通过，无阻断项。

准备时发现并修正了旧helper模块缓存指向新代码，以及旧RESTART_ANCHOR未列入新检查的问题；均为只读预检拒绝，没有消耗正式attempt。原f473验证器使用真实原模块位置和代码，不伪造Git提交或放松旧证据检查。

## 锚点与监控

准入根 `/brf/users/yangxx/VFCL-admission/gpu1-rebalance-20261102`：

| 证据 | SHA256 |
| --- | --- |
| DEPLOYMENT_ANCHOR_57af7ce.json | c9080f241296554499c0e76e1e6d89a4188ea94ad419ecb123eb46818637d117 |
| OLD_CONTROLLER_IDENTITY.json | e4140b5dcaadf9055a0cf8a413c850670e7793478d268159dfebeef1b71d897a |
| READONLY_PREFLIGHT_57af7ce.json | 27454d75cc1d0bf278e1ec3d502cdd71380be47cf76099239d7b89537c9760fa |

正式根 `/brf/users/yangxx/VFCL-results/frozen-method-module-ablation-v2r2-20261101` 下 `gpu1-rebalance-v1/ARMED.json` SHA256为 `8caab88c2370482b593a9c42fb686af3b3f16599616a095b219c3c317494334b`。GitHub分支 `codex/gpu1-quiescent-rebalance-v1` 与标签 `gpu1-quiescent-rebalance-ready-20261102` 已发布。原manifest SHA仍为 `bb10fa6c62719b68124e7b625a204defc6b3e396ac285c068849fff8866e8fca`，原重启锚点仍为 `61d6e229783f12a3d0d02af81b158efea459830d6fe83667df5aab1037d2f00b`。

监视器输出在准入根 `WATCHER_57af7ce.log`。不得再次启动监视器或改写ARMED。出现EXECUTION_EPOCH后，原QUEUE_LOCK中的PID只是历史身份；有效控制器以固定来源的epoch.controller_identity为准。监控只读PID/子进程、ledger、JOB_SUCCESS哈希、GPU身份及FAILURE/COMPLETE；旧PID退役本身不是训练失败。

## 完成后的审计

全18项成功且QUEUE_COMPLETE出现后，使用固定新执行代码的 `frozen_ablation_audit.py --manifest 原manifest --manifest-sha256 原SHA --execution-epoch gpu1-rebalance-v1/EXECUTION_EPOCH.json --execution-epoch-sha256 独立封存的实际epochSHA` 做独立metadata审计。其保留所有原来源、配置、publication及seal条件，并验证新执行映射和完整原成功前缀；新完成记录标记真实新orchestration_commit与原f473来源，不伪装旧producer。

metadata审计通过且独立封存之后才允许 `--summarize`，原PartyKD贡献和FIM解释规则不变，不以消融选择或修改主方法。若没有发生切换而原队列自行完成，则使用原f473审计路径；不得把ARMED误作已发生的执行扩展。
