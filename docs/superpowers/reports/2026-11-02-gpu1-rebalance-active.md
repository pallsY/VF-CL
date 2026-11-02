# GPU1同机重分配已生效：双卡并行

上海2026-11-02 11:10起，接管已完成，两个新worker已通过READY/GO、原配置验证与物理GPU映射检查并开始运行。不是仅ARM状态。

| 执行卡 | 作业 | worker PID |
| --- | --- | --- |
| GPU1 | cifar100/no_partykd/43 | 3273632 |
| GPU0 | isolet/fim_off/42 | 3273624 |

GPU0后续串行处理其余ISOLET5项，GPU1负责剩余较长的CIFAR任务。独立核验时为11/18成功、2运行、5未开始，无失败。原GPU0 CIFAR no-PartyKD42成功结束后，旧调度器才在完全结算、无child且两次停止快照稳定时退役；没有中断、收养、重训活worker，也没有重跑已完成结果。

执行固定代码57af7ce701f1b5823347f00027bae620c77fc8ce，新唯一控制器PID3268813；原控制器PID3003183已按计划退役，原QUEUE_LOCK中的PID保持历史记录，不作为活控制器。原manifest SHA仍为bb10fa6c62719b68124e7b625a204defc6b3e396ac285c068849fff8866e8fca。训练base/patch、原配置、数据、种子和预算均不变；只用两卡UUID次序记录逻辑cuda:0到实际物理卡的授权映射。

## 证据与独立核验

原成功前缀22条start/finish事件、11项success完整保留；未开始的7项只由新controller领取。实际EXECUTION_EPOCH SHA256：`9d17fe3b327aba8806611e4eca23515d509685b0b1bd10dd2faf05b3ec09da7b`，路径为正式结果根 `gpu1-rebalance-v1/EXECUTION_EPOCH.json`。

主执行者逐项验证两项活worker的JOB_RUNTIME、READY/GO、PID/parent、epoch、实际GPU UUID顺序以及config.json/APPROVED_CONFIG字节SHA均与原清单一致。准入根 `/brf/users/yangxx/VFCL-admission/gpu1-rebalance-20261102/LIVE_TWO_GPU_METADATA_57af7ce.json` SHA256：`0048ebf8c15d2131db82d40b9524d9a2323551b13f1789c8a3b77ecd2399b217`。

独立审查者 `/root/gpu1_rebalance_review` 实际执行只读CPU核验，退出码0：调用verify_execution_extension及verify_execution_source，核对固定源代码、manifest、ARMED、epoch、11项成功前缀、7项限定pending、两次全部T且无child的停止快照、旧PID退出、新PID身份、两项worker的唯一attempt/plan/READY==GO/runtime及物理映射和原配置字节。结论transition_metadata_verified；没有调用完整audit_metadata，没有读取准确率、结果、checkpoint或训练log，没有查询GPU、写文件或发送信号。

79项Linux合成/真实producer fixture检查与部署前只读核验、代码审查已通过，详情见 `2026-11-02-gpu1-rebalance-armed.md`。以上交接核验不替代全18项完成后的独立最终审计。

## 后续监控与审计

- 活控制器读取epoch.controller_identity，worker实际GPU读取start.gpu_index/gpu_uuid及JOB_RUNTIME；原manifest physical_gpu是原分配，不可据此误报新的实际卡。
- GPU0>=3072MiB、GPU1>=5120MiB连续120秒及启动复核保持；外部共享允许，失败停本研究全部车道、保留证据，不自动重跑。
- 全18项成功且新metadata审计通过前，不读取或汇总正式准确率。最终用57af7ce的audit传明确execution-epoch和上述独立封存SHA；metadata审计封存后才允许summarize，原PartyKD/FIM解释规则和主方法不改。
- 3090没有接入，本次同机GPU1加速与之前3090跨机候选协议相互独立。
